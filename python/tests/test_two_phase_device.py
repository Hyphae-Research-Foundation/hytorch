"""Two-phase gate ON THE ACCELERATOR (Trainium via torch_xla, or CUDA).

The host gate (test_two_phase.py) proves the policy; this proves the DEVICE
proposal path: propose() compiled/executed on the accelerator, dispose() on
the host reference, apply_commits() on the accelerator, then:
  D1  facts are exact (T1 recompute: every COMMIT mag == bf16(dot(leaf,nhat_f)))
  D2  device apply == apply_ref replay, bit for bit
  D3  miss-rate of the DEVICE proposal vs exact policy-6 (the number the
      manifest thresholds at 0.5%); reported per battery
  D4  wall time of propose() per token-layer (the step-time input)

Usage:  PJRT_DEVICE=NEURON python python/tests/test_two_phase_device.py
"""
from __future__ import annotations

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch

from hytorch.applyref import ApplyRef, fp32_to_bf16_bits, bf16_bits_to_fp32, VERDICT_COMMIT
from hytorch.model import CatalogPolicy
from hytorch.neuron_backend import TwoPhaseCatalog, _nhat_fp32

SO = os.path.join(os.path.dirname(__file__), "..", "..", "target", "release", "libapply_ref.so")


def get_device():
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device(), "xla"
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"


def sync(dev_kind):
    if dev_kind == "xla":
        import torch_xla.core.xla_model as xm
        xm.mark_step(); xm.wait_device_ops()
    elif dev_kind == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    dev, kind = get_device()
    print(f"device: {dev} ({kind})")
    ref = ApplyRef.load(SO)
    rng = np.random.default_rng(7)
    # d20-shaped policy (the real one), smaller token count
    pol = CatalogPolicy(s_slots=64, d_slot=20, n_features=32768, k=8, mag_max=64.0,
                        policy_id=7, selection=0, proposal_clip=32.0)
    backend = TwoPhaseCatalog(pol, ref, m_candidates=32)
    cb_cpu = torch.from_numpy(rng.standard_normal((pol.n_features, pol.d_slot)).astype(np.float32))
    cb_cpu = cb_cpu / cb_cpu.norm(dim=1, keepdim=True)
    cb = cb_cpu.to(dev)

    batteries = {
        "gaussian": lambda: rng.standard_normal((2, 64, 1280)).astype(np.float32),
        "scaled_x20": lambda: (rng.standard_normal((2, 64, 1280)) * 20).astype(np.float32),
        "tie_storm": lambda: np.ones((2, 64, 1280), np.float32) * 0.5,
        "near_zero": lambda: (rng.standard_normal((2, 64, 1280)) * 1e-6).astype(np.float32),
    }
    worst_miss = 0.0
    for name, mk in batteries.items():
        dh_cpu = torch.from_numpy(mk())
        # bridge-style clip (what the trainer would do) so magnitudes are in-policy
        from hytorch.model import clip_proposal
        dh_cpu = clip_proposal(dh_cpu, pol)
        dh = dh_cpu.to(dev)
        h_cpu = torch.from_numpy(rng.standard_normal((2, 64, 1280)).astype(np.float32))
        h = h_cpu.to(dev)

        # --- device propose (timed) ---
        t0 = time.time()
        cands = backend.propose(dh, cb)
        sync(kind)
        t_prop = time.time() - t0
        cands_cpu = cands.cpu()

        # --- host dispose (exact) ---
        recs = backend.dispose(dh_cpu, cb_cpu, cands_cpu)
        commits = recs[recs["verdict"] == VERDICT_COMMIT]

        # D1: T1 recompute of every commit mag from leaf + nhat (host, pinned)
        nt = 2 * 64
        dh_bits = fp32_to_bf16_bits(dh_cpu.numpy()).reshape(nt, pol.s_slots, pol.d_slot)
        leaves = bf16_bits_to_fp32(dh_bits.reshape(-1)).reshape(nt, pol.s_slots, pol.d_slot)
        cb_bits = fp32_to_bf16_bits(cb_cpu.numpy())
        nhat_ref = _nhat_fp32(torch.from_numpy(bf16_bits_to_fp32(cb_bits.reshape(-1)).reshape(cb_bits.shape)), backend.EPS).numpy()
        bad = 0
        for c in commits[: min(2000, len(commits))]:
            lf = leaves[c["pos"], c["slot"]]
            acc = np.float32(0.0)
            row = nhat_ref[c["feature"]]
            for j in range(pol.d_slot):
                acc = np.float32(acc + np.float32(lf[j] * row[j]))
            if fp32_to_bf16_bits(np.array([acc], np.float32))[0] != c["mag_bf16"]:
                bad += 1
        assert bad == 0, f"D1 FAIL [{name}]: {bad} commits with mag != exact recompute"

        # D2: device apply vs reference replay
        h_dev = backend.apply_commits(h, cb, recs)
        sync(kind)
        h_bits = np.ascontiguousarray(fp32_to_bf16_bits(h_cpu.numpy()).reshape(nt, pol.s_slots, pol.d_slot))
        ref.apply(h_bits, cb_bits, commits)
        dev_bits = fp32_to_bf16_bits(h_dev.cpu().to(torch.float32).numpy()).reshape(-1)
        assert np.array_equal(dev_bits, h_bits.reshape(-1)), f"D2 FAIL [{name}]: device apply != reference replay"

        # D3: miss-rate vs exact policy-6
        exact = ref.pack_allocate(dh_bits, cb_bits, pol.k, pol.mag_max, 0)
        miss = 0
        for pos in range(nt):
            fe = set(exact[exact["pos"] == pos]["feature"].tolist())
            fp = set(recs[recs["pos"] == pos]["feature"].tolist())
            if fe - fp:
                miss += 1
        mr = miss / nt
        worst_miss = max(worst_miss, mr)
        cr = float((recs["verdict"] == VERDICT_COMMIT).mean())
        print(f"[{name:10s}] D1 exact-mags OK | D2 apply==replay OK | D3 miss-rate {mr:.3%} | commit {cr:.1%} | propose {t_prop*1000:.1f} ms / {nt} tok")

    print(f"WORST MISS-RATE: {worst_miss:.3%} (manifest threshold 0.5%)")
    assert worst_miss < 0.005, "D3 FAIL: miss-rate over threshold"
    print(f"TWO-PHASE DEVICE GATE ({kind}): ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
