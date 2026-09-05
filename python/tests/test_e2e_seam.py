"""E2E v2: trainer ↔ seam ↔ embedded Hyphae, protocol v2.

1. RUN_START handshake (and its refusal path for placeholder digests).
2. 5 steps: frames → barrier (Phase A receipt == local T2) → step chain
   with real c_prev/c_next (Phase B chainack).
3. Codebook chain break: c_prev(t) != c_next(t-1) → NO chainack → timeout.
4. Kill-test: seam dead → barrier fail-stop.
5. EXPORT fact.

Run: .venv/bin/python python/tests/test_e2e_seam.py
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hytorch.applyref import ApplyRef, VERDICT_COMMIT, fp32_to_bf16_bits  # noqa: E402
from hytorch.seam import StepChain  # noqa: E402
from hytorch.seamclient import BarrierTimeout, SeamClient  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SO = os.path.join(ROOT, "target", "release", "libapply_ref.so")
SEAM = os.path.join(ROOT, "target", "release", "hytorch-seam")

S, D, NF, K, NT, L = 8, 12, 64, 4, 16, 3


def build_facts(so_hash):
    return {
        "build.apply_ref_hash": so_hash,
        "build.harness_commit": "test-commit",
        "build.torch_wheel": "torch-test",
        "build.backend_wheel": "cpu",
    }


def main() -> int:
    work = tempfile.mkdtemp(prefix="hytorch-e2e-")
    data = os.path.join(work, "hyphae")
    spool = os.path.join(work, "spool")
    os.makedirs(spool, exist_ok=True)

    ref = ApplyRef.load(SO)
    rng = np.random.default_rng(7)
    client = SeamClient(spool, policy_id=1)

    proc = subprocess.Popen([SEAM, data, "e2e-01", spool],
                            stderr=subprocess.PIPE, text=True)
    try:
        # ---- RUN_START refusal: placeholder digest gets NO ack ----
        bad = build_facts("FILLED_BY_CAPTURE_BUILD")
        try:
            client.run_start("mh", bad, {"region": "local"}, {"policy_id": 1},
                             budget_ms=1500)
            print("FAIL: placeholder RUN_START acked", file=sys.stderr)
            return 1
        except BarrierTimeout:
            print("PART 0 OK: placeholder RUN_START refused (no ack)")
        os.remove(os.path.join(spool, "run-start.json"))

        # ---- honest RUN_START ----
        client.run_start("mh", build_facts(ref.sha256),
                         {"region": "local", "device_slug": "test"},
                         {"policy_id": 1, "k": K, "s_slots": S,
                          "n_features": NF, "mag_max": 8.0})
        print("PART 1 OK: RUN_START + POLICY committed")

        cb_bits = fp32_to_bf16_bits(rng.standard_normal((NF, D)) * 0.5)
        h_bits = np.ascontiguousarray(
            fp32_to_bf16_bits(rng.standard_normal((NT, S, D)) * 0.1))
        h_bits[0, 0, 0] = 0x8000  # -0.0 landmine, as always

        c_chain = hashlib.sha256(b"c0").digest()
        for step in range(5):
            chain = StepChain(step_id=step, mb=0, policy_id=1, elide_zeros=True)
            for layer in range(L):
                dh = fp32_to_bf16_bits(rng.standard_normal((NT, S, D)) * 0.05)
                recs = ref.pack_allocate(dh, cb_bits, K, 8.0)
                client.spool_layer(step, layer, 0, recs, device=0)
                chain.add_layer(layer, recs, device=0)
                commits = recs[recs["verdict"] == VERDICT_COMMIT]
                ref.apply(h_bits, cb_bits, commits)

            head = client.barrier(step, budget_ms=5000)
            assert head == chain.head.hex(), f"step {step}: receipt != local T2"

            c_next = hashlib.sha256(f"c{step+1}".encode()).digest()
            client.step_chain(step, 3e-3, 1.0, c_chain, c_next,
                              resets=[7] if step == 2 else [],
                              seal_theta=hashlib.sha256(b"th").digest() if step == 3 else None,
                              budget_ms=5000)
            c_chain = c_next
            print(f"step {step}: receipt + chainack OK ({head[:16]}…)")

        print("PART 2 OK: 5 steps, receipts match independent T2, chain continuous")

        # ---- codebook chain break: wrong c_prev → NO chainack ----
        step = 5
        dh = fp32_to_bf16_bits(rng.standard_normal((NT, S, D)) * 0.05)
        recs = ref.pack_allocate(dh, cb_bits, K, 8.0)
        client.spool_layer(step, 0, 0, recs)
        client.barrier(step, budget_ms=5000)
        try:
            client.step_chain(step, 1e-3, 0.5,
                              hashlib.sha256(b"WRONG").digest(),
                              hashlib.sha256(b"c6").digest(),
                              resets=[], seal_theta=None, budget_ms=1500)
            print("FAIL: broken codebook chain acked", file=sys.stderr)
            return 1
        except BarrierTimeout:
            print("PART 3 OK: codebook chain break → no chainack (fail-stop)")

        # ---- EXPORT ----
        client.export(4, "th" * 32, "ch" * 32, "hh" * 32)
        print("PART 4 OK: EXPORT committed")
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    # ---- kill-test: seam is DEAD; the barrier must fail-stop ----
    step = 99
    dh = fp32_to_bf16_bits(rng.standard_normal((NT, S, D)) * 0.05)
    recs = ref.pack_allocate(dh, cb_bits, K, 8.0)
    client.spool_layer(step, 0, 0, recs)
    try:
        client.barrier(step, budget_ms=1500)
        print("FAIL: barrier returned without a live seam", file=sys.stderr)
        return 1
    except BarrierTimeout as e:
        print(f"PART 5 OK: fail-stop fired: {e}")

    assert os.path.isdir(data) and len(os.listdir(data)) > 0
    print("PART 6 OK: Hyphae data dir persisted:", data)

    shutil.rmtree(work, ignore_errors=True)
    print("E2E v2 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_e2e_seam():
    """pytest entry (review 2026-09-02): the script's main() is the test."""
    assert main() in (0, None)
