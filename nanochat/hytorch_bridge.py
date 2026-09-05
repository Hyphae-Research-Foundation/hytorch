"""hytorch bridge for nanochat — law-0 catalog writes with DDP-aware seam.

Design (nanochat/INTEGRATION.md):
- catalog_write(x, delta, layer_idx, unit) is the ONLY writer of the
  residual when HYTORCH_CATALOG=1. Vanilla path (env unset) is `x + delta`,
  bit-identical to upstream nanochat — the baseline arm is this same tree.
- Two write units per layer (attn=0, mlp=1); frame layer id = 2*L + unit
  (declared in the manifest; wire format unchanged).
- Multi-rank (torchrun): every rank journals its own verdicts with
  microbatch_id=rank; ONLY rank 0 runs the seam client. step() gating is
  collective: rank0 waits the receipt and broadcasts the head; all ranks
  proceed or all die (§5.4 has no partial survivors).

The heavy lifting (pack/allocate/apply kernels, STE backward) is imported
from hytorch's python package, which ships alongside on the droplet.
"""

from __future__ import annotations

import os
import sys

import torch

_STATE: dict = {"on": False}


def resolve_proposal_clip() -> float | None:
    """Manifest is the authority (catalog.proposal_clip, falling back to
    policy.proposal_clip); HYTORCH_PROPOSAL_CLIP env overrides for
    experiments; absent everywhere = no clip (phase-1/2 semantics)."""
    env = os.environ.get("HYTORCH_PROPOSAL_CLIP")
    if env is not None:
        v = float(env)
        return v if v > 0 else None
    man = os.environ.get("HYTORCH_MANIFEST")
    if man and os.path.exists(man):
        import json
        m = json.load(open(man))
        for sect in ("catalog", "policy"):
            v = (m.get(sect) or {}).get("proposal_clip")
            if v is not None:
                return float(v) if float(v) > 0 else None
    return None


def _init_once():
    if "init" in _STATE:
        return _STATE
    _STATE["init"] = True
    _STATE["on"] = os.environ.get("HYTORCH_CATALOG", "") == "1"
    if not _STATE["on"]:
        return _STATE

    hytorch_root = os.environ["HYTORCH_ROOT"]  # /opt/hytorch on droplets
    sys.path.insert(0, os.path.join(hytorch_root, "python"))
    from hytorch.applyref import ApplyRef
    from hytorch.model import CatalogPolicy, HyphaeWrite, hyphae_write

    d_slot = int(os.environ.get("HYTORCH_D_SLOT", "0"))
    assert d_slot > 0, "set HYTORCH_D_SLOT = n_embd // 64"
    pol = CatalogPolicy(
        s_slots=64,
        d_slot=d_slot,
        n_features=int(os.environ.get("HYTORCH_NF", "32768")),
        k=int(os.environ.get("HYTORCH_K", "8")),
        mag_max=float(os.environ.get("HYTORCH_MAG_MAX", "64.0")),
        policy_id=int(os.environ.get("HYTORCH_POLICY_ID", "6")),
        selection=0,
        proposal_clip=resolve_proposal_clip(),
    )
    _STATE["pol"] = pol
    _STATE["HyphaeWrite"] = HyphaeWrite
    _STATE["write"] = hyphae_write

    if torch.cuda.is_available():
        from hytorch.devext import DeviceCatalog
        _STATE["backend"] = DeviceCatalog()
    else:
        _STATE["backend"] = ApplyRef.load(
            os.path.join(hytorch_root, "target", "release", "libapply_ref.so"))

    # Shared learned codebook (one per process; DDP averages its grads like
    # any parameter). Registered lazily into the model by steal_codebook().
    _STATE["codebook"] = None
    _STATE["journal"] = []          # consumed by the trainer loop each step
    return _STATE


def attach_codebook(model: torch.nn.Module, n_embd: int, device, dtype,
                    init_from: torch.Tensor | None = None):
    """Create + register the codebook parameter on the model (called once
    from the trainer after model init; participates in DDP + optimizer).

    init_from: restore the codebook from a checkpoint (resume / SFT / eval).
    Checkpoint loaders pop 'hytorch_codebook' from the state dict (the bare
    GPT has no such param at load time) and pass it back here."""
    st = _init_once()
    if not st["on"]:
        return None
    pol = st["pol"]
    assert n_embd == 64 * pol.d_slot, f"n_embd {n_embd} != 64*d_slot"
    if init_from is not None:
        assert tuple(init_from.shape) == (pol.n_features, pol.d_slot), \
            f"checkpoint codebook {tuple(init_from.shape)} != policy {(pol.n_features, pol.d_slot)}"
        data = init_from.detach().to(device=device, dtype=torch.float32)
        cb = torch.nn.Parameter(data.clone())
    else:
        cb = torch.nn.Parameter(
            torch.randn(pol.n_features, pol.d_slot, device=device, dtype=torch.float32) * 0.02)
    model.register_parameter("hytorch_codebook", cb)
    st["codebook"] = cb
    return cb


def pop_codebook(state_dict: dict):
    """Remove hytorch_codebook from a checkpoint state dict (if present) and
    stash it. The bare GPT can then load with strict=True; the trainer/eval
    script re-attaches via attach_codebook(init_from=stashed_codebook())."""
    t = None
    for key in [k for k in state_dict if k.endswith("hytorch_codebook")]:
        t = state_dict.pop(key)
    if t is not None:
        _STATE["stashed_codebook"] = t
    return t


def stashed_codebook():
    return _STATE.get("stashed_codebook")


def restore_codebook_for_inference(device):
    """Eval/chat entry points (no optimizer): make the stashed checkpoint
    codebook the active one for catalog_write, WITHOUT registering a model
    parameter (setup_optimizer's param-coverage assert stays untouched).
    A cataloged model without its codebook cannot run — fail loudly."""
    st = _init_once()
    if not st["on"]:
        return None
    t = _STATE.get("stashed_codebook")
    assert t is not None, (
        "HYTORCH_CATALOG=1 but the checkpoint carried no hytorch_codebook — "
        "a cataloged model cannot run without its codebook")
    st["codebook"] = t.detach().to(device=device, dtype=torch.float32)
    return st["codebook"]


_EVAL_SCRATCH: list = []   # journal sink for eval/inference forwards


def catalog_write(x: torch.Tensor, delta: torch.Tensor, layer_idx: int, unit: int):
    st = _init_once()
    if not st["on"]:
        return x + delta            # vanilla nanochat, bit-identical
    HyphaeWrite = st["HyphaeWrite"]
    # Proposal clip is applied inside HyphaeWrite from pol.proposal_clip
    # (manifest-declared; see resolve_proposal_clip). No hardcode here.
    # Journal routing (phase 5): training forwards (grad enabled) journal to
    # the seam; eval/sampling forwards (no_grad/inference_mode) write through
    # the catalog but are NOT training facts — they go to a scratch list that
    # is dropped, so eval_every/core_metric no longer poison or OOM the
    # training journal. Journaled INFERENCE (runtime.JournaledGenerator)
    # opts back in with HYTORCH_JOURNAL_INFERENCE=1.
    if torch.is_grad_enabled() or os.environ.get("HYTORCH_JOURNAL_INFERENCE", "") == "1":
        j = st["journal"]
    else:
        j = _EVAL_SCRATCH
        j.clear()
    out = st["write"](x, delta, st["codebook"], st["backend"], st["pol"], j)
    j[-1].meta = (layer_idx, unit)   # frame id = 2*L + unit
    return out


def drain_journal():
    """Trainer calls this once per step: returns [(frame_layer_id, entry)]
    and clears. Rank-local; the seam shim ships them."""
    st = _init_once()
    entries = [(2 * e.meta[0] + e.meta[1], e) for e in st["journal"]]
    st["journal"].clear()
    return entries
