"""hytorch bridge for the Trainium train.py — law-0 catalog writes, policy 7
(SPEC_AMEND-004 two-phase: device proposes candidates, host reference
disposes exact verdicts). Mirrors nanochat/hytorch_bridge.py; differences:

- backend is ALWAYS TwoPhaseCatalog (no custom kernels on Neuron);
- the model is bf16-weights: the codebook parameter stays fp32 (optimizer
  precision) and quantizes at the seam like everywhere else;
- journal routing identical (train journals, eval scratch).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

_STATE: dict = {"on": False}


def _resolve_clip():
    env = os.environ.get("HYTORCH_PROPOSAL_CLIP")
    if env is not None:
        return float(env) if float(env) > 0 else None
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

    hytorch_root = os.environ["HYTORCH_ROOT"]
    sys.path.insert(0, os.path.join(hytorch_root, "python"))
    from hytorch.applyref import ApplyRef
    from hytorch.model import CatalogPolicy, HyphaeWrite, hyphae_write
    from hytorch.neuron_backend import TwoPhaseCatalog

    d_slot = int(os.environ.get("HYTORCH_D_SLOT", "0"))
    assert d_slot > 0, "set HYTORCH_D_SLOT = n_embd // 64"
    pol = CatalogPolicy(
        s_slots=64,
        d_slot=d_slot,
        n_features=int(os.environ.get("HYTORCH_NF", "32768")),
        k=int(os.environ.get("HYTORCH_K", "8")),
        mag_max=float(os.environ.get("HYTORCH_MAG_MAX", "64.0")),
        policy_id=int(os.environ.get("HYTORCH_POLICY_ID", "7")),
        selection=0,
        proposal_clip=_resolve_clip(),
    )
    ref = ApplyRef.load(
        os.path.join(hytorch_root, "target", "release", "libapply_ref.so"))
    _STATE["pol"] = pol
    _STATE["HyphaeWrite"] = HyphaeWrite
    _STATE["write"] = hyphae_write
    _STATE["backend"] = TwoPhaseCatalog(pol, ref)
    _STATE["codebook"] = None
    _STATE["journal"] = []
    return _STATE


def attach_codebook(model: torch.nn.Module, n_embd: int, device, dtype,
                    init_from: torch.Tensor | None = None):
    st = _init_once()
    if not st["on"]:
        return None
    pol = st["pol"]
    assert n_embd == 64 * pol.d_slot, f"n_embd {n_embd} != 64*d_slot"
    if init_from is not None:
        cb = torch.nn.Parameter(
            init_from.detach().to(device=device, dtype=torch.float32).clone())
    else:
        cb = torch.nn.Parameter(
            torch.randn(pol.n_features, pol.d_slot, device=device,
                        dtype=torch.float32) * 0.02)
    model.register_parameter("hytorch_codebook", cb)
    st["codebook"] = cb
    if device.type == "xla" or os.environ.get("HYTORCH_HOST_CB_GRAD", "") == "1":
        # The codebook gradient lives on the HOST between backward and the
        # optimizer (see sync_codebook_grad): a [N_f, d_slot] tensor in the
        # fwd+bwd+all-reduce NEFF is linearized row by row by neuronx-cc
        # (6M instructions at d10, NCC_EXTP003).
        from hytorch.model import install_host_codebook_grad
        install_host_codebook_grad(cb)
        st["host_cb_grad"] = True
    return cb


_GLOO: dict = {}


def sync_codebook_grad() -> None:
    """Call right after GradBucket.all_reduce() and before optimizer.step():
    sum the host-accumulated codebook gradient over ranks (gloo, 1.3 MB at
    d10) and upload it once as codebook.grad. Bit-for-bit this is the same
    gradient the device path would have produced, minus the device adds."""
    st = _init_once()
    if not st.get("host_cb_grad"):
        return
    from hytorch.model import take_host_codebook_grad
    g = take_host_codebook_grad()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        if "pg" not in _GLOO:
            _GLOO["pg"] = dist.new_group(backend="gloo")
        dist.all_reduce(g, op=dist.ReduceOp.SUM, group=_GLOO["pg"])
        g.mul_(1.0 / dist.get_world_size())
    cb = st["codebook"]
    cb.grad = g.to(device=cb.device, dtype=cb.dtype)


_EVAL_SCRATCH: list = []


def catalog_write(x: torch.Tensor, delta: torch.Tensor, layer_idx: int, unit: int):
    st = _init_once()
    if not st["on"]:
        return x + delta            # vanilla, bit-identical
    HyphaeWrite = st["HyphaeWrite"]
    pol = st["pol"]
    # clip applied inside HyphaeWrite from pol.proposal_clip (manifest)
    if torch.is_grad_enabled() or os.environ.get("HYTORCH_JOURNAL_INFERENCE", "") == "1":
        j = st["journal"]
    else:
        j = _EVAL_SCRATCH
        j.clear()
    out = st["write"](x, delta, st["codebook"], st["backend"], pol, j)
    j[-1].meta = (layer_idx, unit)
    return out


def drain_journal():
    st = _init_once()
    entries = [(2 * e.meta[0] + e.meta[1], e) for e in st["journal"]]
    st["journal"].clear()
    return entries
