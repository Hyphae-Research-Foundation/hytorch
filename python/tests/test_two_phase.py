"""Two-phase backend gate (SPEC_AMEND-004) — runs on CPU, no accelerator.

Gates, in order of severity:
  G1  BRIDGE: pack_allocate_candidates with M=N_f (all features, natural
      order) == policy-6 pack_allocate, byte-for-byte on (pos, feature,
      mag, slot, verdict, reason) — the formal equivalence.
  G2  APPLY REPLAY: TwoPhaseCatalog.apply_commits output bits == the pinned
      reference apply_ref.apply on the SAME facts (the residual replay
      contract that makes facts auditable).
  G3  PIPELINE: full forward on adversarial deltas — verdict partition
      #C+#O+#A == k per token, canonical order, journal fields consistent.
  G4  MISS-RATE: proposal completeness on realistic magnitudes (gaussian
      deltas, trained-ish codebook) with the manifest M — reported, and
      thresholded at the prereg value (<0.5% tokens with any miss).

Usage:  python -m tests.test_two_phase   (from python/, with ../target built)
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from hytorch.applyref import ApplyRef, fp32_to_bf16_bits, bf16_bits_to_fp32, VERDICT_COMMIT
from hytorch.model import CatalogPolicy
from hytorch.neuron_backend import TwoPhaseCatalog, measure_miss_rate

SO = os.path.join(os.path.dirname(__file__), "..", "..", "target", "release",
                  "libapply_ref.so")


def adversarial_batteries(pol: CatalogPolicy, rng: np.random.Generator):
    """The hytorch adversarial battery, sized for the pipeline."""
    S, D = pol.s_slots, pol.d_slot
    nt = 64
    shape = (nt, S * D)

    def t(x):
        return torch.from_numpy(x.astype(np.float32)).reshape(1, nt, S * D)

    yield "gaussian", t(rng.standard_normal(shape))
    yield "zeros", t(np.zeros(shape))
    yield "neg_zeros", t(np.full(shape, -0.0))
    yield "tiny_denormal-ish", t(rng.standard_normal(shape) * 1e-38)
    yield "large", t(rng.standard_normal(shape) * 1e4)
    yield "over_cap", t(rng.standard_normal(shape) * 200.0)
    x = rng.standard_normal(shape); x[::3, ::5] = np.nan
    yield "nan_holes", t(x)
    x = rng.standard_normal(shape); x[::7, ::3] = np.inf; x[1::7, ::4] = -np.inf
    yield "inf_holes", t(x)
    x = np.zeros(shape); x[:, : D] = 1.0
    yield "slot0_only", t(x)
    yield "tie_everything", t(np.ones(shape) * 0.5)


def main() -> int:
    ref = ApplyRef.load(SO)
    rng = np.random.default_rng(20260901)
    pol = CatalogPolicy(s_slots=8, d_slot=6, n_features=256, k=8,
                        mag_max=8.0, policy_id=7, selection=0)
    cb = torch.from_numpy(rng.standard_normal((pol.n_features, pol.d_slot)).astype(np.float32)) * 0.3

    # ---- G1: bridge (M = N_f, natural order) ----
    nt_total = 0
    for name, dh in adversarial_batteries(pol, rng):
        B, T, _ = dh.shape
        nt = B * T
        dh_bits = fp32_to_bf16_bits(dh.numpy()).reshape(nt, pol.s_slots, pol.d_slot)
        cb_bits = fp32_to_bf16_bits(cb.numpy())
        exact = ref.pack_allocate(dh_bits, cb_bits, pol.k, pol.mag_max, 0)
        cand_all = np.tile(np.arange(pol.n_features, dtype=np.uint16), (nt, 1))
        bridged = ref.pack_allocate_candidates(dh_bits, cb_bits, cand_all,
                                               pol.k, pol.mag_max)
        for fld in ("pos", "feature", "mag_bf16", "slot", "verdict", "reason"):
            assert np.array_equal(exact[fld], bridged[fld]), \
                f"G1 FAIL [{name}] field {fld}"
        nt_total += nt
    print(f"G1 BRIDGE: policy7(M=N_f) == policy6 byte-exact on {nt_total} tokens, 10 batteries")

    # ---- G2: apply replay ----
    backend = TwoPhaseCatalog(pol, ref, m_candidates=32)
    for name, dh in adversarial_batteries(pol, rng):
        B, T, _ = dh.shape
        nt = B * T
        h = torch.from_numpy(rng.standard_normal((B, T, pol.s_slots * pol.d_slot)).astype(np.float32))
        cands = backend.propose(dh, cb)
        recs = backend.dispose(dh, cb, cands)
        h_dev = backend.apply_commits(h, cb, recs)
        # pinned replay: same facts through apply_ref.apply on h bits
        h_bits = fp32_to_bf16_bits(h.numpy()).reshape(nt, pol.s_slots, pol.d_slot)
        h_bits = np.ascontiguousarray(h_bits)
        cb_bits = fp32_to_bf16_bits(cb.numpy())
        commits = recs[recs["verdict"] == VERDICT_COMMIT]
        ref.apply(h_bits, cb_bits, commits)
        h_ref = bf16_bits_to_fp32(h_bits.reshape(-1)).reshape(B, T, -1)
        h_dev_bits = fp32_to_bf16_bits(h_dev.to(torch.float32).numpy())
        h_ref_bits = fp32_to_bf16_bits(h_ref)
        # NOTE: device h stays fp32 upstream; the contract is on bf16 bits.
        assert np.array_equal(h_dev_bits.reshape(-1), h_ref_bits.reshape(-1)), \
            f"G2 FAIL [{name}]: device apply != reference replay"
    print("G2 APPLY REPLAY: device apply == apply_ref bits on all batteries")

    # ---- G3: pipeline invariants ----
    for name, dh in adversarial_batteries(pol, rng):
        B, T, _ = dh.shape
        nt = B * T
        h = torch.zeros(B, T, pol.s_slots * pol.d_slot)
        journal: list = []
        _ = backend.forward(h, dh, cb, journal)
        recs = journal[-1].recs()
        for pos in range(nt):
            tok = recs[recs["pos"] == pos]
            assert len(tok) == pol.k, f"G3 FAIL [{name}] pos {pos}: {len(tok)} != k"
            slots = tok["slot"]
            assert all(slots[i] <= slots[i + 1] for i in range(len(slots) - 1)), \
                f"G3 FAIL [{name}]: slot order"
    print("G3 PIPELINE: partition #C+#O+#A==k, canonical order, all batteries")

    # ---- G4: miss-rate (realistic regime, clipped like the bridge does) ----
    dh = torch.from_numpy(rng.standard_normal((2, 128, pol.s_slots * pol.d_slot)).astype(np.float32))
    d_slots = dh.view(2, 128, pol.s_slots, pol.d_slot)
    norms = d_slots.norm(dim=-1, keepdim=True)
    clip = pol.mag_max * 0.5
    d_slots = d_slots * (clip / norms.clamp_min(1e-12)).clamp(max=1.0)
    dh = d_slots.reshape(2, 128, -1)
    stats = measure_miss_rate(ref, pol, backend, dh, cb)
    print(f"G4 MISS-RATE: {stats}")
    assert stats["miss_rate"] < 0.005, f"G4 FAIL: miss rate {stats['miss_rate']} >= 0.5%"

    # ---- G5: two-stage top-M == global top-M (trn2 shapes: NF=32768, S=64) ----
    # The two-stage path exists because neuronx-cc cannot lower a k>=16 topk
    # over a 32768-wide row (ISGV902); it must propose the SAME candidate set.
    pol_big = CatalogPolicy(s_slots=64, d_slot=10, n_features=32768, k=8,
                            mag_max=8.0, policy_id=7, selection=0)
    cb_big = torch.from_numpy(rng.standard_normal((pol_big.n_features, pol_big.d_slot)).astype(np.float32))
    dh_big = torch.from_numpy(rng.standard_normal((1, 256, pol_big.s_slots * pol_big.d_slot)).astype(np.float32))
    g = TwoPhaseCatalog(pol_big, ref, m_candidates=32, topk_mode="global").propose(dh_big, cb_big)
    t = TwoPhaseCatalog(pol_big, ref, m_candidates=32, topk_mode="two_stage").propose(dh_big, cb_big)
    auto = TwoPhaseCatalog(pol_big, ref, m_candidates=32)
    assert auto.topk_mode == "auto" and auto.propose(dh_big, cb_big).shape == (256, 32)
    assert g.shape == t.shape == (256, 32)
    g_sets = np.sort(g.numpy(), axis=1); t_sets = np.sort(t.numpy(), axis=1)
    assert np.array_equal(g_sets, t_sets), "G5 FAIL: two-stage candidate set != global topk"
    assert np.array_equal(g.numpy(), t.numpy()), "G5 FAIL: two-stage order != global order"
    print("G5 TWO-STAGE: candidate ids and order == single topk on 256 tokens x 32768 features")
    # Ties are the init condition (zero-init projections => every score 0):
    # the exact winners (features 0..k-1, one per slot) must be PRESENT in the
    # two-stage candidates, else the channel collapses into one slot (trn2 d10
    # take 5). Miss-rate under exact ties must be 0 on the trn2 shapes.
    two = TwoPhaseCatalog(pol_big, ref, m_candidates=32, topk_mode="two_stage")
    for name, dh_t in (("zeros", torch.zeros(1, 64, 640)),
                       ("const", torch.full((1, 64, 640), 0.25)),
                       ("one_slot_live", torch.cat([torch.randn(1, 64, 10), torch.zeros(1, 64, 630)], -1))):
        st = measure_miss_rate(ref, pol_big, two, dh_t, cb_big)
        c = two.propose(dh_t, cb_big)
        n_slots = len(set((c[0] % 64).tolist()))
        assert st["miss_rate"] == 0.0, f"G5 FAIL [{name}]: two-stage miss rate {st['miss_rate']} under ties"
        if name == "zeros":
            assert set(c[0].tolist()) == set(range(32)), f"G5 FAIL [zeros]: tie reservoir != features 0..31: {c[0].tolist()}"
        print(f"G5 TIES [{name}]: miss_rate 0, candidates span {n_slots} slots")

    # ---- G6: flat clip (XLA path) == view clip ----
    from hytorch.model import clip_proposal, _clip_proposal_flat, CatalogPolicy as _CP
    pol_clip = _CP(s_slots=64, d_slot=10, n_features=32768, k=8, mag_max=64.0,
                   policy_id=7, selection=0, proposal_clip=32.0)
    dh_c = torch.from_numpy(rng.standard_normal((2, 64, 640)).astype(np.float32))
    dh_c[0, :16] *= 40.0                       # some slots far over the clip
    a = dh_c.clone().requires_grad_(True); b = dh_c.clone().requires_grad_(True)
    ya = clip_proposal(a, pol_clip); yb = _clip_proposal_flat(b, pol_clip)
    norms = dh_c.view(2, 64, 64, 10).norm(dim=-1)
    unclipped = (norms <= 32.0).unsqueeze(-1).expand(2, 64, 64, 10).reshape(2, 64, 640)
    assert torch.equal(ya[unclipped], yb[unclipped]), "G6 FAIL: flat clip differs on unclipped slots"
    assert torch.allclose(ya, yb, rtol=1e-5, atol=1e-5), "G6 FAIL: flat clip off on clipped slots"
    g = torch.randn_like(dh_c); ya.backward(g); yb.backward(g)
    assert torch.allclose(a.grad, b.grad, rtol=1e-5, atol=1e-5), "G6 FAIL: flat clip gradient"
    n_clipped = int((~unclipped).sum()) // 10
    print(f"G6 FLAT CLIP: == view clip (bit-exact on unclipped, 1e-5 on {n_clipped} clipped slots), grads match")

    # ---- G7: host-accumulated codebook grad == autograd codebook grad ----
    import hytorch.model as hm
    from hytorch.model import hyphae_write
    os.environ["HYTORCH_HOST_MIRROR"] = "1"       # CPU stands in for XLA
    try:
        cb7 = (torch.from_numpy(rng.standard_normal((pol.n_features, pol.d_slot)).astype(np.float32)) * 0.3)
        h7 = torch.from_numpy(rng.standard_normal((2, 32, pol.s_slots * pol.d_slot)).astype(np.float32))
        d7 = torch.from_numpy(rng.standard_normal((2, 32, pol.s_slots * pol.d_slot)).astype(np.float32))
        gout = torch.randn_like(h7)
        def run(with_host_acc):
            hm.HOST_CODEBOOK_GRAD = None
            cb = cb7.clone().requires_grad_(True)
            acc = hm.install_host_codebook_grad(cb) if with_host_acc else None
            journal7: list = []
            out = hyphae_write(h7.clone().requires_grad_(True), d7.clone().requires_grad_(True),
                               cb, backend, pol, journal7)
            (out * gout).sum().backward()
            g = hm.take_host_codebook_grad() if with_host_acc else None
            hm.HOST_CODEBOOK_GRAD = None
            return cb.grad, g, journal7[-1].recs()
        g_dev, _, recs_a = run(False)
        g_none, g_host, recs_b = run(True)
        assert np.array_equal(recs_a, recs_b), "G7 FAIL: facts differ between runs"
        assert g_none is None, "G7 FAIL: autograd received a codebook grad with the host accumulator on"
        assert g_dev is not None and torch.equal(g_dev, g_host), "G7 FAIL: host-accumulated grad != autograd grad"
        print(f"G7 HOST CB GRAD: accumulator == autograd codebook grad bit-for-bit (norm {g_host.norm():.4f})")
    finally:
        os.environ.pop("HYTORCH_HOST_MIRROR", None)
        hm.HOST_CODEBOOK_GRAD = None

    print("TWO-PHASE GATE: ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_two_phase():
    """pytest entry (review 2026-09-02): the script's main() is the test."""
    assert main() in (0, None)
