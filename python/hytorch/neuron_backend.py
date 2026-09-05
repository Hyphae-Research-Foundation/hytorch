"""Two-phase catalog backend for accelerators without custom kernels
(SPEC_AMEND-004, policy 7 "two_phase_topk"). Built for AWS Trainium2 via
torch-xla; runs identically on CPU/CUDA (pure torch ops, no pybind, no
custom kernels — XLA compiles the proposer into the training graph).

Phase A — PROPOSE (device, fast, NOT verified):
    scores = leaf_slots @ nhat^T  in bf16 (TensorE food), top-M per token.
    The proposal is a hint. Any numeric sloppiness here can only change
    WHICH facts get written, never their truth.

Phase B — DISPOSE (host, exact, verified):
    apply_ref.pack_allocate_candidates recomputes candidate scores with the
    pinned math and runs the exact policy-6 verdict machinery. Facts out.

Phase C — APPLY (device, mirrored exactly):
    h' = h + mag * nhat[f] at (pos, slot) for COMMITs. On-device apply uses
    the same bf16 quantization contract as the reference (promote bits,
    fp32 math per component, RNE downcast); the CPU replay gate remains the
    arbiter (apply_ref.apply on the same facts must reproduce h' bits).

The device NEVER owns the codebook math silently: nhat is computed from the
bf16 codebook bits the ledger hashes, and T1 spills recompute any cited mag
from scratch. What stays declared-not-verified is proposal completeness
(see SPEC_AMEND-004: miss-rate measured, thresholded in the manifest).
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .applyref import ApplyRef, VERDICT_COMMIT
from .model import CatalogPolicy, JournalEntry


def bf16_bits_from_tensor(x: torch.Tensor) -> torch.Tensor:
    """fp32/bf16 tensor -> uint16 bf16 bits (RNE), pure torch (XLA-safe)."""
    xb = x.to(torch.bfloat16)
    return xb.view(torch.uint16) if hasattr(xb, "view") else xb


def _nhat_fp32(codebook: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalized codebook rows from the QUANTIZED (bf16) weights — the same
    bits the ledger hashes. fp32 math; sub-ulp accumulation differences vs
    the sequential reference are proposal-side only (phase B recomputes)."""
    cq = codebook.detach().to(torch.bfloat16).to(torch.float32)
    denom = cq.square().sum(dim=1, keepdim=True).sqrt() + eps
    return cq / denom


class TwoPhaseCatalog:
    """Drop-in backend for HyphaeWrite-style forwards under policy 7.

    forward(h, delta_hat) -> (h_out, recs) where recs is the structured
    verdict array (numpy, canonical order) ready for the journal/wire.
    """

    EPS = float(np.frombuffer(np.uint32(0x38800000).tobytes(), dtype=np.float32)[0])

    # neuronx-cc 2.26 dies with an internal assertion (NCC_INAS001 / ISGV902,
    # SimplifyTongaTensor.buildAccessRanges) lowering AwsNeuronTopK over a
    # row of 32768 scores when k >= 16; rows <= 16384 or k <= 8 compile.
    # Only a 2-D, last-dim, sorted=True topk is pattern-matched to that
    # custom call at all — anything else lowers to `sort`, which trn2 rejects.
    TOPK_ROW_LIMIT = 16384

    def __init__(self, pol: CatalogPolicy, ref: ApplyRef,
                 m_candidates: int | None = None, topk_mode: str | None = None):
        assert pol.selection == 0, "two-phase base is global_topk"
        self.pol = pol
        self.ref = ref
        self.m = m_candidates or int(os.environ.get("HYTORCH_M_CANDIDATES", "32"))
        assert self.m >= pol.k, "M must be >= k"
        mode = topk_mode or os.environ.get("HYTORCH_PROPOSE_TOPK", "auto")
        assert mode in ("auto", "global", "two_stage"), mode
        self.topk_mode = mode

    # ---- Phase A: device proposal (pure torch; XLA compiles this) ----
    @torch.no_grad()
    def propose(self, delta_hat: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """[B,T,D] -> [nt, M] candidate feature ids (int32 on device)."""
        pol = self.pol
        B, T, D = delta_hat.shape
        nt = B * T
        leaves = delta_hat.detach().reshape(nt, pol.s_slots, pol.d_slot)
        nhat = _nhat_fp32(codebook, self.EPS)                    # [NF, d]
        nf = nhat.shape[0]
        # scores[t, f] = <leaf[t, f % S], nhat[f]> — group features by home
        # slot so each group is ONE dense matmul (TensorE-shaped).
        # nhat grouped: [S, NF/S, d]; leaves: [nt, S, d] -> scores [nt, S, NF/S]
        per_slot = nf // pol.s_slots
        nh_g = nhat.reshape(per_slot, pol.s_slots, pol.d_slot).permute(1, 0, 2)
        lv = leaves.to(torch.bfloat16).to(torch.float32)         # quantized leaves
        sc = torch.einsum("nsd,spd->nsp", lv, nh_g)              # [nt, S, per_slot]
        m = min(self.m, nf)
        mode = self.topk_mode
        if mode == "auto":
            mode = "two_stage" if nf > self.TOPK_ROW_LIMIT else "global"
        if mode == "two_stage":
            # Exact top-M in two 2-D topks: an element of the global top-M
            # has < M scores above it, so it is in its slot's top-M. Also
            # avoids transposing the [nt, S, per_slot] scores.
            # TIES decide the init: nanochat zero-inits the projections, so
            # at step 0 every score is exactly 0 and the candidate set IS the
            # tie-break. The exact policy yields features 0..k-1 = one feature
            # in each of slots 0..k-1; those must be PRESENT or the STE mirror
            # only ever trains the slots that did win and the channel
            # collapses (trn2 d10 take 5: commit 12.5% = 1/k from step 0,
            # all candidates from slot 0, guard kill at step 149). Neither CPU
            # torch.topk nor a device TopK promises lowest-index-first on
            # ties, so the tie-break is made explicit: a composite key that
            # subtracts p*2^-40 (stage 1) and s*2^-50 (stage 2) — below the
            # fp32 ulp of any non-tiny score, so only exact ties are ordered,
            # by feature ascending exactly like the reference.
            m1 = min(m, per_slot)
            dev = sc.device
            k1 = sc - torch.arange(per_slot, device=dev, dtype=torch.float32) * 2.0 ** -40
            v1, p1 = torch.topk(k1.reshape(nt * pol.s_slots, per_slot), m1,
                                dim=1, largest=True, sorted=True)   # [nt*S, m1]
            v1 = v1.reshape(nt, pol.s_slots, m1).transpose(1, 2)    # [nt, m1, S]
            p1 = p1.reshape(nt, pol.s_slots, m1).transpose(1, 2)
            k2 = v1 - torch.arange(pol.s_slots, device=dev, dtype=torch.float32) * 2.0 ** -50
            _, j = torch.topk(k2.reshape(nt, m1 * pol.s_slots), m,
                              dim=1, largest=True, sorted=True)     # [nt, M]
            s = j % pol.s_slots
            p = torch.gather(p1.reshape(nt, m1 * pol.s_slots), 1, j)
            return (p * pol.s_slots + s).to(torch.int32)          # [nt, M]
        # flatten back to feature ids: f = p * S + s  (home σ(f) = f mod S)
        scores = sc.permute(0, 2, 1).reshape(nt, per_slot * pol.s_slots)
        # top-M by score value; ties broken by feature asc via composite key
        # (approximate is fine — phase B re-ties exactly; we only need the
        # true winners to be PRESENT, and value-ties keep both present)
        _, idx = torch.topk(scores, m, dim=1, largest=True, sorted=True)
        return idx.to(torch.int32)                                # [nt, M]

    # ---- Phase B: exact verdicts on the host reference ----
    def dispose(self, delta_hat: torch.Tensor, codebook: torch.Tensor,
                cand_ids: torch.Tensor) -> np.ndarray:
        from .applyref import fp32_to_bf16_bits
        pol = self.pol
        B, T, D = delta_hat.shape
        nt = B * T
        dh_bits = fp32_to_bf16_bits(
            delta_hat.detach().to(torch.float32).cpu().numpy()
        ).reshape(nt, pol.s_slots, pol.d_slot)
        cb_bits = fp32_to_bf16_bits(codebook.detach().to(torch.float32).cpu().numpy())
        cands = cand_ids.cpu().numpy().astype(np.uint16)
        return self.ref.pack_allocate_candidates(
            dh_bits, cb_bits, cands, pol.k, pol.mag_max)

    # ---- Phase C: apply. HOST-EXACT by default (measured on Trainium:
    # the Neuron compiler does not honor the bf16 contract — 1-4 ulp drift on
    # every touched leaf, fp32 FMA/rounding differences; CPU torch is
    # bit-exact). The apply touches <= S*nt leaves of d_slot each (160
    # scalars per token at d20): moving it to the host reference costs a
    # tiny D2H/H2D of the touched leaves and buys bit-exact residuals on
    # ANY silicon — the same 'reference disposes' principle as phase B.
    # HYTORCH_DEVICE_APPLY=1 opts back into the on-device path where the
    # device is known to pass gate D2 (CUDA/HIP kernels do; XLA does not).
    @torch.no_grad()
    def apply_commits(self, h: torch.Tensor, codebook: torch.Tensor,
                      recs: np.ndarray) -> torch.Tensor:
        pol = self.pol
        B, T, D = h.shape
        nt = B * T
        commits = recs[recs["verdict"] == VERDICT_COMMIT]
        if len(commits) == 0:
            return h
        if os.environ.get("HYTORCH_DEVICE_APPLY", "") == "1":
            return self._apply_commits_device(h, codebook, recs)
        from .applyref import fp32_to_bf16_bits, bf16_bits_to_fp32
        # XLA graph stability: every device op here must have a FIXED shape
        # across calls or the lazy compiler re-traces (328 NEFFs in the d2
        # smoke). Under global_topk there is at most ONE commit per (pos,
        # slot), so a dense update mask is exact and fixed-shape: host
        # computes the new leaf values for touched slots, device does one
        # where(). The only D2H is the whole residual once (fixed shape).
        # The mask is expanded to leaf granularity ON THE HOST and the where
        # runs on h's own [B,T,D] shape: no device tensor may carry d_slot as
        # a dimension. At d10 the reshape [nt,640]->[nt,64,10] under the
        # layout the surrounding matmuls impose costs neuronx-cc one
        # transpose_1x10 per leaf row (~6M instructions for a step, NCC_EXTP003).
        h_cpu = h.detach().to(torch.float32).cpu().reshape(nt, pol.s_slots, pol.d_slot)
        h_bits = np.ascontiguousarray(fp32_to_bf16_bits(h_cpu.numpy()))
        cb_bits = fp32_to_bf16_bits(codebook.detach().to(torch.float32).cpu().numpy())
        self.ref.apply(h_bits, cb_bits, commits)      # in place, pinned reference
        new_cpu = torch.from_numpy(bf16_bits_to_fp32(h_bits.reshape(-1)).reshape(B, T, D).copy())
        touched = np.zeros((nt, pol.s_slots), dtype=bool)
        touched[commits["pos"].astype(np.int64), commits["slot"].astype(np.int64)] = True
        mask_cpu = torch.from_numpy(np.repeat(touched, pol.d_slot, axis=1).reshape(B, T, D))
        new_dev = new_cpu.to(h.device).to(h.dtype)
        return torch.where(mask_cpu.to(h.device), new_dev, h.detach())

    @torch.no_grad()
    def _apply_commits_device(self, h: torch.Tensor, codebook: torch.Tensor,
                              recs: np.ndarray) -> torch.Tensor:
        pol = self.pol
        B, T, D = h.shape
        nt = B * T
        commits = recs[recs["verdict"] == VERDICT_COMMIT]
        dev = h.device
        p = torch.from_numpy(commits["pos"].astype(np.int64)).to(dev)
        s = torch.from_numpy(commits["slot"].astype(np.int64)).to(dev)
        f = torch.from_numpy(commits["feature"].astype(np.int64)).to(dev)
        mag_np = (commits["mag_bf16"].astype(np.uint32) << 16).view(np.float32)
        mag = torch.from_numpy(mag_np.copy()).to(dev)
        nhat = _nhat_fp32(codebook, self.EPS)
        upd = mag.unsqueeze(1) * nhat[f]
        hq = h.detach().reshape(nt, pol.s_slots, pol.d_slot)
        leaf = hq[p, s].to(torch.bfloat16).to(torch.float32)
        new_leaf = (leaf + upd).to(torch.bfloat16)
        out = hq.clone()
        out[p, s] = new_leaf.to(h.dtype)
        return out.reshape(B, T, D)

    # ---- full pipeline (journal entry compatible with the bridge) ----
    def forward(self, h: torch.Tensor, delta_hat: torch.Tensor,
                codebook: torch.Tensor, journal: list) -> torch.Tensor:
        cands = self.propose(delta_hat, codebook)
        recs = self.dispose(delta_hat, codebook, cands)
        h_out = self.apply_commits(h, codebook, recs)
        fields = _fields_from_recs_torch(recs, h.device)
        journal.append(JournalEntry(recs=recs, fields=fields))
        return h_out


def _fields_from_recs_torch(recs: np.ndarray, device) -> dict:
    cm = torch.from_numpy((recs["verdict"] == VERDICT_COMMIT).copy()).to(device)
    return {
        "pos": torch.from_numpy(recs["pos"].astype(np.int64)).to(device),
        "slot": torch.from_numpy(recs["slot"].astype(np.int64)).to(device),
        "feature": torch.from_numpy(recs["feature"].astype(np.int64)).to(device),
        "mag_bits": torch.from_numpy(recs["mag_bf16"].astype(np.uint16)).to(device),
        "commit_mask": cm,
    }


def measure_miss_rate(ref: ApplyRef, pol: CatalogPolicy, backend: TwoPhaseCatalog,
                      delta_hat: torch.Tensor, codebook: torch.Tensor) -> dict:
    """Proposal completeness audit (SPEC_AMEND-004): exact policy-6 top-k vs
    two-phase verdicts on the same inputs. A 'miss' = a token whose exact
    top-k cites any feature absent from the proposal's verdict set."""
    from .applyref import fp32_to_bf16_bits
    B, T, D = delta_hat.shape
    nt = B * T
    dh_bits = fp32_to_bf16_bits(
        delta_hat.detach().to(torch.float32).cpu().numpy()
    ).reshape(nt, pol.s_slots, pol.d_slot)
    cb_bits = fp32_to_bf16_bits(codebook.detach().to(torch.float32).cpu().numpy())
    exact = ref.pack_allocate(dh_bits, cb_bits, pol.k, pol.mag_max, 0)
    cands = backend.propose(delta_hat, codebook)
    prop = backend.dispose(delta_hat, codebook, cands)
    miss_tokens = 0
    for pos in range(nt):
        fe = set(exact[exact["pos"] == pos]["feature"].tolist())
        fp = set(prop[prop["pos"] == pos]["feature"].tolist())
        if fe - fp:
            miss_tokens += 1
    return {"n_tokens": nt, "miss_tokens": miss_tokens,
            "miss_rate": miss_tokens / max(1, nt)}
