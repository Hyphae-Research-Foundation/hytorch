"""Law-0 parallel block + HyphaeWrite (STE). Spec v2.2 §03–§06.

- ParallelBlock: LN reads h; attn and mlp read the normalized h and sum into
  their OWN buffer (delta_hat). No op inside the block receives a mutable h.
- HyphaeWrite: the ONLY writer of h. Forward = pack → allocate → apply via
  the pinned policy (the .so on CPU, the gate-authorized kernels on device —
  bit-identical by the M3 gate). Backward = the STE symbolic mirror
  h' = h + mag·n̂(C[f]) over the COMMITTED set, in CLOSED FORM (no nested
  autograd): for n̂ = c/D with D = ‖c‖+ε,
      ∂L/∂c = u/D − c·(cᵀu)/(‖c‖·D²),  u = mag·g_leaf   (per commit, index_add)
      ∂L/∂mag = ⟨g_leaf, n̂⟩            (STE through the bf16 of the binding)
      ∂L/∂leaf = ∂L/∂mag · n̂           (score path)

Zero-sync device path (Lámina C, "cero sync por capa"): verdicts stay on
device; the journal D2H is an async copy to pinned memory materialized AFTER
backward (d2h.overlap = backward).

Aux balance (gradient-contract fix): the earlier aux was computed from
detached counts — a CONSTANT in the graph, zero gradient, decorative. The
Switch-Transformer form is used instead: aux = N_f · Σ_f f_f · P_f with
f_f = hard committed fraction (data) and P_f = |score| mass share recomputed
DIFFERENTIABLY from delta_hat against detached n̂. Minimizing it pushes score
mass away from over-used features. Journalized as part of the gradient
contract, not the POLICY record (§3.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os

import numpy as np
import torch
import torch.nn as nn

from .applyref import (
    ApplyRef,
    VERDICT_COMMIT,
    bf16_bits_to_fp32,
    fp32_to_bf16_bits,
)

EPS = 2.0 ** -14


@dataclass
class CatalogPolicy:
    s_slots: int
    d_slot: int
    n_features: int
    k: int
    mag_max: float
    policy_id: int = 1
    selection: int = 0  # 0=global_topk (phase 1), 1=slot_topk (phase 2)
    # Eval-time POLICY intervention (phase-2 causal ablation): features in
    # this list are forced to ABORT(reason=policy) after allocate. The
    # non-fact is journalized like any other. Mutable on purpose: it is an
    # intervention, and interventions are policy acts, not weights.
    deny_features: list = field(default_factory=list)
    # Proposal norm clip (PHASE5-CHANNEL-COLLAPSE, review 2026-09-02): each
    # slot's proposal is projected to ||delta_slot|| <= proposal_clip BEFORE
    # the pack (direction preserved; detached-norm scale, gradient flows via
    # the direction). None/0 = off. Declared in the manifest under
    # catalog.proposal_clip and echoed in POLICY; the bridge no longer
    # hardcodes mag_max/2. Applied INSIDE HyphaeWrite so every caller
    # (run.py, hallu.py, nanochat, trainium, two-phase) shares one contract.
    proposal_clip: float | None = None


class JournalEntry:
    """One layer's verdicts. Device path: raw bytes live on GPU; a pinned
    async copy is materialized to numpy only when .recs() is first called
    (after backward). CPU path: numpy immediately; fields lifted to torch."""

    def __init__(self, *, recs: np.ndarray | None = None,
                 raw_pinned: torch.Tensor | None = None,
                 fields: dict | None = None):
        self._recs = recs
        self._raw_pinned = raw_pinned
        self.fields = fields  # dict of device tensors: pos/slot/feature/mag_bits/verdict/commit_mask

    def recs(self) -> np.ndarray:
        if self._recs is None:
            from .devext import VERDICT_DTYPE
            # Caller must have synchronized (we do it once per step).
            buf = self._raw_pinned.numpy().tobytes()
            self._recs = np.frombuffer(buf, dtype=VERDICT_DTYPE).copy()
        return self._recs


def _fields_from_recs(recs: np.ndarray, device) -> dict:
    commit = recs["verdict"] == VERDICT_COMMIT
    to = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dt)
    return {
        "pos": to(recs["pos"].astype(np.int64), torch.int64),
        "slot": to(recs["slot"].astype(np.int64), torch.int64),
        "feature": to(recs["feature"].astype(np.int64), torch.int64),
        "mag_bits": to(recs["mag_bf16"].astype(np.int32), torch.int32),
        "commit_mask": to(commit, torch.bool),
    }


def clip_proposal(delta_hat: torch.Tensor, pol: CatalogPolicy) -> torch.Tensor:
    """Per-slot norm clip of the proposal (see CatalogPolicy.proposal_clip).
    Under autograd this is a plain differentiable op on the graph tensor;
    inside HyphaeWrite.forward it acts on the already-detached input, which
    is exactly the value the policy judges — the backward mirror sees the
    clipped facts, consistent with what was committed."""
    clip = pol.proposal_clip
    if not clip or clip <= 0:
        return delta_hat
    if delta_hat.device.type == "xla" or os.environ.get("HYTORCH_CLIP_FLAT", "") == "1":
        return _clip_proposal_flat(delta_hat, pol)
    B, T, D = delta_hat.shape
    d = delta_hat.view(B, T, pol.s_slots, pol.d_slot)
    norms = d.detach().float().norm(dim=-1, keepdim=True)
    scale = (clip / norms.clamp_min(1e-12)).clamp(max=1.0).to(delta_hat.dtype)
    return (d * scale).view(B, T, D)


_SLOT_ONEHOT: dict = {}


def _slot_onehot(pol: CatalogPolicy, device) -> torch.Tensor:
    """[D, S] fp32 indicator E[j, s] = 1 iff leaf j belongs to slot s."""
    key = (pol.s_slots, pol.d_slot, str(device))
    e = _SLOT_ONEHOT.get(key)
    if e is None:
        slot_of_leaf = torch.arange(pol.s_slots * pol.d_slot) // pol.d_slot
        e = torch.nn.functional.one_hot(slot_of_leaf, pol.s_slots).to(torch.float32).to(device)
        _SLOT_ONEHOT[key] = e
    return e


def _clip_proposal_flat(delta_hat: torch.Tensor, pol: CatalogPolicy) -> torch.Tensor:
    """clip_proposal without any d_slot-shaped tensor on the device graph.
    On Trainium the reshape [nt, D] -> [nt, S, d_slot] under the layout the
    neighbouring matmuls impose is lowered to one transpose per leaf row;
    with ~180 such tensors per step the fwd+bwd NEFF exceeds the compiler's
    instruction limit (NCC_EXTP003, 6M > 300k at d10). Per-slot sums of
    squares and the expansion of the per-slot scale back to leaves are
    one-hot matmuls on [nt, D] instead. Where scale == 1 (the unclipped
    slots) the result is bit-identical to clip_proposal; on clipped slots
    the norm carries the matmul's rounding (a hint-level transform, and the
    policy judges the materialized clipped values either way)."""
    clip = pol.proposal_clip
    B, T, D = delta_hat.shape
    e = _slot_onehot(pol, delta_hat.device)
    x = delta_hat.detach().to(torch.float32).reshape(B * T, D)
    sumsq = (x * x) @ e                                              # [nt, S]
    scale_s = (clip / sumsq.sqrt().clamp_min(1e-12)).clamp(max=1.0)  # [nt, S]
    scale = (scale_s @ e.t()).reshape(B, T, D).to(delta_hat.dtype)   # exact expansion
    return delta_hat * scale


class HyphaeWrite(torch.autograd.Function):
    """h' = apply(h, allocate(pack(delta_hat, C))) — pinned-policy forward,
    closed-form STE mirror backward. All-device on CUDA; zero per-layer syncs.
    """

    @staticmethod
    def forward(ctx, h, delta_hat, codebook, backend, pol: CatalogPolicy, journal: list):
        B, T, D = h.shape
        nt = B * T
        # NOTE: the proposal clip is applied by the caller (catalog_write /
        # HyphaeWrite.apply wrapper below) INSIDE autograd, never here. Applying
        # it on the detached input silently drops the clip's Jacobian from the
        # STE mirror: a proposal 400x over the clip then receives a 400x
        # amplified gradient, compounding per layer (measured: 1e35 -> inf by
        # layer 13, the phase-5 SFT NaN). See catalog_write().

        # Two-phase backend (SPEC_AMEND-004, policy 7 — Trainium/XLA path):
        # device proposes candidates, host reference disposes exact verdicts,
        # device applies under the bit contract. Same journal, same STE
        # mirror backward (below) — the backward only needs the facts.
        if type(backend).__name__ == "TwoPhaseCatalog":
            cands = backend.propose(delta_hat, codebook)
            recs = backend.dispose(delta_hat, codebook, cands)
            if pol.deny_features:
                deny = np.isin(recs["feature"].astype(int), pol.deny_features)
                hit = deny & (recs["verdict"] == VERDICT_COMMIT)
                recs["verdict"][hit] = 2   # ABORT
                recs["reason"][hit] = 3    # policy
            h_out = backend.apply_commits(h, codebook, recs).to(h.dtype)
            # On XLA keep the fact fields on the HOST: 5 device uploads per
            # write unit are 5 graph inputs whose values change every step —
            # harmless for CUDA, but each is a lazy-graph edge on Trainium.
            # The journal consumer (seam shim) reads .recs() (numpy) anyway.
            fields = _fields_from_recs(recs, "cpu" if h.device.type == "xla" else h.device)
            entry = JournalEntry(recs=recs, fields=fields)
            journal.append(entry)
            # XLA (Trainium): the STE mirror's boolean-mask gathers and the
            # uint16<<16 bit-reinterpret are not lowerable (SIGSEGV in the
            # Neuron PJRT backward, measured). The facts are host-resident
            # numpy anyway: keep the COMMIT facts on ctx as numpy and run the
            # mirror on host in backward (backward_host). Dense grads only
            # cross the device boundary.
            ctx.host_mirror = (h.device.type == "xla"
                               or os.environ.get("HYTORCH_HOST_MIRROR", "") == "1")
            if ctx.host_mirror:
                cm_np = recs["verdict"] == VERDICT_COMMIT
                ctx.commits_np = recs[cm_np]
                ctx.save_for_backward(codebook)
            else:
                ctx.save_for_backward(
                    codebook, fields["pos"], fields["slot"], fields["feature"],
                    fields["mag_bits"], fields["commit_mask"])
            ctx.pol = pol
            ctx.h_shape = h.shape
            return h_out

        if isinstance(backend, ApplyRef):
            dh_bits = fp32_to_bf16_bits(
                delta_hat.detach().to(torch.float32).cpu().numpy()
            ).reshape(nt, pol.s_slots, pol.d_slot)
            cb_bits = fp32_to_bf16_bits(codebook.detach().to(torch.float32).cpu().numpy())
            recs = backend.pack_allocate(dh_bits, cb_bits, pol.k, pol.mag_max, pol.selection)
            if pol.deny_features:
                deny = np.isin(recs["feature"].astype(int), pol.deny_features)
                hit = deny & (recs["verdict"] == VERDICT_COMMIT)
                recs["verdict"][hit] = 2   # ABORT
                recs["reason"][hit] = 3    # policy

            h_bits = fp32_to_bf16_bits(h.detach().to(torch.float32).cpu().numpy()).reshape(
                nt, pol.s_slots, pol.d_slot
            )
            h_bits = np.ascontiguousarray(h_bits)
            commits = recs[recs["verdict"] == VERDICT_COMMIT]
            backend.apply(h_bits, cb_bits, commits)
            h_out = torch.from_numpy(
                bf16_bits_to_fp32(h_bits.reshape(-1)).reshape(h.shape).copy()
            ).to(h.dtype).to(h.device)

            fields = _fields_from_recs(recs, h.device)
            entry = JournalEntry(recs=recs, fields=fields)
        else:
            # Device path: bits never leave the GPU except the packed verdict
            # stream, and that leaves as ONE async copy per layer.
            from .devext import raw_fields

            dh_bits = delta_hat.detach().to(torch.bfloat16).reshape(
                nt, pol.s_slots, pol.d_slot
            ).contiguous().view(torch.uint16)
            cb_bits = codebook.detach().to(torch.bfloat16).contiguous().view(torch.uint16)
            raw = backend.pack_allocate_t(dh_bits, cb_bits, pol.k, pol.mag_max, pol.selection)
            f = raw_fields(raw)
            if pol.deny_features:
                deny_t = torch.as_tensor(pol.deny_features, device=raw.device,
                                         dtype=torch.int64)
                hit = torch.isin(f["feature"], deny_t) & (f["verdict"] == 0)
                # Patch the raw wire words so the journal carries the ABORT:
                # w2 verdict byte -> 2; w3 reason byte -> 3.
                w = f["w"]
                w[:, 2] = torch.where(hit, (w[:, 2] & 0x00FFFFFF) | (2 << 24), w[:, 2])
                w[:, 3] = torch.where(hit, (w[:, 3] & ~0xFF) | 3, w[:, 3])
                f["verdict"] = torch.where(hit, torch.full_like(f["verdict"], 2), f["verdict"])

            h_bits = (
                h.detach().to(torch.bfloat16).reshape(nt, pol.s_slots, pol.d_slot)
                .contiguous().view(torch.uint16)
            )
            cm = backend.apply_commits_raw(h_bits, raw, f)
            h_out = h_bits.view(torch.bfloat16).reshape(h.shape).to(h.dtype)

            pinned = torch.empty_like(raw, device="cpu", pin_memory=True)
            pinned.copy_(raw, non_blocking=True)  # overlaps with backward
            fields = {
                "pos": f["pos"], "slot": f["slot"], "feature": f["feature"],
                "mag_bits": f["mag_bits"], "commit_mask": cm,
            }
            entry = JournalEntry(raw_pinned=pinned, fields=fields)

        journal.append(entry)

        ctx.save_for_backward(
            codebook,
            fields["pos"], fields["slot"], fields["feature"],
            fields["mag_bits"], fields["commit_mask"],
        )
        ctx.pol = pol
        ctx.h_shape = h.shape
        return h_out

    @staticmethod
    def backward(ctx, grad_out):
        if getattr(ctx, "host_mirror", False):
            return HyphaeWrite._backward_host(ctx, grad_out)
        codebook, pos, slot, feature, mag_bits, cm = ctx.saved_tensors
        pol: CatalogPolicy = ctx.pol
        B, T, D = ctx.h_shape
        nt = B * T
        dev = grad_out.device

        # Mirror: h' = h + Σ_committed mag_i · n̂(C[f_i]) at (pos_i, slot_i).
        grad_h = grad_out
        grad_delta = torch.zeros(nt, pol.s_slots, pol.d_slot, device=dev,
                                 dtype=grad_out.dtype)
        grad_codebook = torch.zeros_like(codebook)

        p, s, fidx = pos[cm], slot[cm], feature[cm]
        if p.numel() > 0:
            mag = (mag_bits[cm].to(torch.int32) << 16).contiguous().view(torch.float32)
            mag = mag.to(grad_out.dtype)

            # Mirror math in fp32 regardless of the autograd dtype (bf16
            # trainers like nanochat hit dtype mismatches otherwise); cast
            # back at the boundaries.
            go = grad_out.reshape(nt, pol.s_slots, pol.d_slot)
            g_leaf = go[p, s].to(torch.float32)                  # [n, d]
            mag = mag.to(torch.float32)

            c = codebook.detach()[fidx].to(torch.float32)        # [n, d]
            norm = c.norm(dim=1, keepdim=True)                   # ‖c‖
            denom = norm + EPS                                   # D
            nhat = c / denom

            # ∂L/∂mag = ⟨g_leaf, n̂⟩ ; ∂L/∂leaf = grad_mag · n̂  (STE)
            grad_mag = (g_leaf * nhat).sum(dim=1)
            grad_delta.index_put_(
                (p, s),
                (grad_mag.unsqueeze(1) * nhat).to(grad_delta.dtype),
                accumulate=True)

            # ∂L/∂c closed form: u/D − c·(cᵀu)/(‖c‖·D²), u = mag·g_leaf.
            u = mag.unsqueeze(1) * g_leaf
            cu = (c * u).sum(dim=1, keepdim=True)
            gc = u / denom - c * (cu / (norm.clamp_min(EPS) * denom * denom))
            grad_codebook.index_add_(0, fidx, gc.to(grad_codebook.dtype))

        return grad_h, grad_delta.reshape(B, T, D), grad_codebook, None, None, None


def hyphae_write(h, delta_hat, codebook, backend, pol: CatalogPolicy, journal: list):
    """The ONLY entry point callers should use: clips the proposal in-graph
    (so the clip's Jacobian reaches the proposing block through autograd),
    then runs the pinned forward/STE-mirror Function on the clipped value."""
    return HyphaeWrite.apply(h, clip_proposal(delta_hat, pol), codebook, backend, pol, journal)


def _hyphae_backward_host(ctx, grad_out):
    """Same STE mirror math as HyphaeWrite.backward, executed on the host
    from the numpy COMMIT facts (XLA path). Returns device tensors."""
    (codebook,) = ctx.saved_tensors
    pol: CatalogPolicy = ctx.pol
    B, T, D = ctx.h_shape
    nt = B * T
    dev = grad_out.device
    c_np = ctx.commits_np
    go = grad_out.detach().to(torch.float32).cpu().reshape(nt, pol.s_slots, pol.d_slot)
    grad_delta = torch.zeros(nt, pol.s_slots, pol.d_slot, dtype=torch.float32)
    cb_cpu = codebook.detach().to(torch.float32).cpu()
    grad_codebook = torch.zeros_like(cb_cpu)
    if len(c_np) > 0:
        p = torch.from_numpy(c_np["pos"].astype(np.int64))
        s = torch.from_numpy(c_np["slot"].astype(np.int64))
        fidx = torch.from_numpy(c_np["feature"].astype(np.int64))
        mag = torch.from_numpy((c_np["mag_bf16"].astype(np.uint32) << 16).view(np.float32).copy())
        g_leaf = go[p, s]                                     # [n, d]
        c = cb_cpu[fidx]
        norm = c.norm(dim=1, keepdim=True)
        denom = norm + EPS
        nhat = c / denom
        grad_mag = (g_leaf * nhat).sum(dim=1)
        grad_delta.index_put_((p, s), grad_mag.unsqueeze(1) * nhat, accumulate=True)
        u = mag.unsqueeze(1) * g_leaf
        cu = (c * u).sum(dim=1, keepdim=True)
        gc = u / denom - c * (cu / (norm.clamp_min(EPS) * denom * denom))
        grad_codebook.index_add_(0, fidx, gc)
    if HOST_CODEBOOK_GRAD is not None:
        # Trainium: a [N_f, d_slot] gradient must never enter the device
        # graph (neuronx-cc linearizes it for the all-reduce bucket with one
        # transpose per row: 6M instructions at d10, NCC_EXTP003). The host
        # keeps the sum; the bridge all-reduces it over gloo and uploads it
        # once per step, after the bucket (trn_bridge.sync_codebook_grad).
        HOST_CODEBOOK_GRAD.add_(grad_codebook)
        gcb = None
    else:
        gcb = grad_codebook.to(dev).to(codebook.dtype)
    return (grad_out,
            grad_delta.reshape(B, T, D).to(dev).to(grad_out.dtype),
            gcb,
            None, None, None)


HyphaeWrite._backward_host = staticmethod(_hyphae_backward_host)

# Host-side codebook gradient accumulator (CPU fp32 [N_f, d_slot]) or None.
# Installed by the XLA bridge via install_host_codebook_grad(codebook).
HOST_CODEBOOK_GRAD: torch.Tensor | None = None


def install_host_codebook_grad(codebook: torch.Tensor) -> torch.Tensor:
    global HOST_CODEBOOK_GRAD
    HOST_CODEBOOK_GRAD = torch.zeros(codebook.shape, dtype=torch.float32)
    return HOST_CODEBOOK_GRAD


def take_host_codebook_grad() -> torch.Tensor:
    """Return the accumulated host gradient and reset the accumulator."""
    g = HOST_CODEBOOK_GRAD.clone()
    HOST_CODEBOOK_GRAD.zero_()
    return g


def switch_aux(delta_hat: torch.Tensor, entry: JournalEntry,
               codebook: torch.Tensor, pol: CatalogPolicy) -> torch.Tensor:
    """Differentiable load-balance pressure (Switch-style f·P).

    f_f  = committed count share per feature (hard, data).
    P_f  = |score| mass share, scores recomputed from the GRAPH delta_hat
           against detached n̂ — gradient flows to the block.
    aux  = N_f · Σ_f f_f · P_f   (≈1 uniform; grows with concentration).
    """
    f = entry.fields
    cm = f["commit_mask"]
    p, s, fidx = f["pos"][cm], f["slot"][cm], f["feature"][cm]
    n = p.numel()
    if n == 0:
        return delta_hat.sum() * 0.0
    B, T, D = delta_hat.shape
    leaves = delta_hat.reshape(B * T, pol.s_slots, pol.d_slot)[p, s]   # graph
    with torch.no_grad():
        c = codebook.detach()[fidx]
        nhat = (c / (c.norm(dim=1, keepdim=True) + EPS)).to(leaves.dtype)
    scores = (leaves * nhat).sum(dim=1).abs()                          # [n], graph
    mass = torch.zeros(pol.n_features, device=delta_hat.device,
                       dtype=scores.dtype).index_add_(0, fidx, scores)
    counts = torch.zeros(pol.n_features, device=delta_hat.device,
                         dtype=scores.dtype).index_add_(
        0, fidx, torch.ones_like(scores))
    f_share = counts / counts.sum().clamp_min(1.0)                     # data
    p_share = mass / mass.sum().clamp_min(1e-20)                       # graph
    return pol.n_features * (f_share.detach() * p_share).sum()


class ParallelBlock(nn.Module):
    """GPT-J/PaLM-style parallel block, law-0: proposes delta_hat, never
    writes h. The residual write happens OUTSIDE, through HyphaeWrite.

    SPEC_AMEND-003: NO ReZero gate. The original law 1 ("write path born at
    zero") conflated two different needs: (a) an ANCHOR for verification —
    which H(h₀) already provides, and (b) an initialization trick — which
    the ledger never needed and which DISTORTED phase-1 results: the scalar
    gate compressed all magnitudes against mag_max (the P1 freeze was a
    gate×mag_max interaction), forced the degenerate all-zero step-0
    selection, and diverged from the real GPT-J/PaLM baseline (which has no
    gate). Standard init instead: output projections scaled by 1/√(2L)
    (GPT-2 style) so the residual grows controlled from step 0. Hyphae
    journals the negative space (ABORT/OVERFLOW) regardless — the model does
    not need to be born mute for the ledger to see it.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int = 12):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        # GPT-2-style scaled init on the residual-writing projections.
        scale = (2 * n_layers) ** -0.5
        with torch.no_grad():
            self.attn.out_proj.weight.mul_(scale)
            self.mlp[2].weight.mul_(scale)

    def forward(self, h: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = self.ln(h)                      # reads h — never writes it
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        m = self.mlp(x)
        return a + m                        # delta_hat in its OWN buffer


class CatalogedTransformer(nn.Module):
    """Phase-1 model: embed → L × (ParallelBlock → write) → head.

    backend=None ⇒ DENSE BASELINE (spec C2): same parallel-block topology,
    same scaled init (SPEC_AMEND-003: no ReZero), dense `h = h + delta_hat`,
    no catalog.

    forward returns (logits, aux): aux is the differentiable Switch-style
    balance term (0 for the baseline).
    """

    def __init__(self, vocab: int, d_model: int, n_heads: int, n_layers: int,
                 pol: CatalogPolicy, backend):
        super().__init__()
        assert d_model == pol.s_slots * pol.d_slot
        self.embed = nn.Embedding(vocab, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)
        self.blocks = nn.ModuleList(
            [ParallelBlock(d_model, n_heads, n_layers) for _ in range(n_layers)]
        )
        self.codebook = nn.Parameter(torch.randn(pol.n_features, pol.d_slot) * 0.02)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.pol = pol
        self.backend = backend
        self.ln_f = nn.LayerNorm(d_model)

    @property
    def is_baseline(self) -> bool:
        return self.backend is None

    def forward(self, tokens: torch.Tensor, journal: list, alias_check: bool = False,
                capture: dict | None = None):
        """capture (audited steps): dict receives 'h0' and 'h_final' detached
        tensors — the device's own claim of the residual before/after the
        writes, for the T1 spill. bf16→fp32→bf16 round-trips are exact for
        values that came from bf16 bits, so hashing the downcast is honest."""
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        h = self.embed(tokens) + self.pos_embed(pos)[None, :, :]
        if capture is not None:
            capture["h0"] = h.detach().clone()
        mask = torch.triu(torch.full((T, T), float("-inf"), device=h.device), diagonal=1)
        aux = h.new_zeros(())
        for blk in self.blocks:
            snap = h.detach().clone() if alias_check else None
            delta_hat = blk(h, attn_mask=mask)
            if alias_check:
                # C7a: byte comparison, never id().
                assert torch.equal(h.detach(), snap), "law-0 violated: block wrote h"
            if self.backend is None:
                h = h + delta_hat  # dense baseline: the += the spec names
            else:
                h = hyphae_write(
                    h, delta_hat, self.codebook, self.backend, self.pol, journal
                )
                aux = aux + switch_aux(delta_hat, journal[-1], self.codebook, self.pol)
        if capture is not None:
            capture["h_final"] = h.detach().clone()
        return self.head(self.ln_f(h)), aux


def renorm_or_reset(codebook: nn.Parameter, min_norm: float, rng: np.random.Generator):
    """C9/D2: measure PRE-renorm norms after opt.step(); dead rows reset
    (CODEBOOK_RESET fact is emitted by the caller); live rows renormalize to 1.
    Returns the list of dead feature indices (the fact payload).
    """
    with torch.no_grad():
        norms = codebook.norm(dim=1)
        dead = (norms < min_norm).nonzero(as_tuple=True)[0]
        for f in dead.tolist():
            v = torch.from_numpy(rng.standard_normal(codebook.shape[1])).to(
                dtype=codebook.dtype, device=codebook.device
            )
            codebook[f] = v / v.norm()
        live = norms >= min_norm
        codebook[live] = codebook[live] / norms[live].unsqueeze(1)
    return dead.tolist()
