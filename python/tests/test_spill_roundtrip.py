"""Cross-language integration test: Python spill ↔ Rust verifier.

Simulates a 3-layer forward using ONLY the .so (pack_allocate + apply),
builds the T2 chain with zero elision, writes a spill, and runs
hytorch-verify on it. Green = the whole seam layout is consistent.

Run: .venv/bin/python python/tests/test_spill_roundtrip.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hytorch.applyref import ApplyRef, VERDICT_COMMIT, fp32_to_bf16_bits  # noqa: E402
from hytorch.seam import StepChain  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SO = os.path.join(ROOT, "target", "release", "libapply_ref.so")
VERIFY = os.path.join(ROOT, "target", "release", "hytorch-verify")

S, D, NF, K, NT = 8, 12, 64, 4, 16
MAG_MAX = 8.0


def main() -> int:
    ref = ApplyRef.load(SO)
    rng = np.random.default_rng(1337)

    h_bits = fp32_to_bf16_bits(rng.standard_normal((NT, S, D)) * 0.1)
    # Salt with -0.0 leaves: the D1 landmine must survive the full circuit.
    h_bits[0, 0, 0] = 0x8000
    h_bits[3, 2, 5] = 0x8000
    h0 = h_bits.copy()

    cb_bits = fp32_to_bf16_bits(rng.standard_normal((NF, D)) * 0.5)

    chain = StepChain(step_id=7, mb=0, policy_id=1, elide_zeros=True)
    h_cur = np.ascontiguousarray(h_bits)
    for layer in range(3):
        # delta_hat: pseudo-proposal (in real training this is the block).
        dh = fp32_to_bf16_bits(rng.standard_normal((NT, S, D)) * 0.05)
        recs = ref.pack_allocate(dh, cb_bits, K, MAG_MAX)
        chain.add_layer(layer, recs, device=0)
        commits = recs[recs["verdict"] == VERDICT_COMMIT]
        ref.apply(h_cur, cb_bits, commits)

    spill = chain.spill_bytes(h0.reshape(-1), cb_bits, S, D, h_cur.reshape(-1))
    with tempfile.NamedTemporaryFile(suffix=".spill", delete=False) as f:
        f.write(spill)
        path = f.name

    out = subprocess.run([VERIFY, path, "--json"], capture_output=True, text=True)
    print("verifier:", out.stdout.strip() or out.stderr.strip())
    if out.returncode != 0:
        print("FAIL: honest spill rejected", file=sys.stderr)
        return 1

    # Tamper 1 bit inside a persisted binding region → must be caught.
    tampered = bytearray(spill)
    tampered[-3] ^= 0x04  # inside the last layer's wire
    with tempfile.NamedTemporaryFile(suffix=".spill", delete=False) as f:
        f.write(bytes(tampered))
        tpath = f.name
    out2 = subprocess.run([VERIFY, tpath, "--json"], capture_output=True, text=True)
    print("tampered wire:", out2.stdout.strip() or out2.stderr.strip())
    if out2.returncode == 0:
        print("FAIL: tampered spill accepted", file=sys.stderr)
        return 1

    # Tamper the CODEBOOK region (spill v2 gap-closer): chain stays intact,
    # residual hash must catch it.
    tampered2 = bytearray(spill)
    cb_off = 112 + h0.size * 2  # header v2 + h0
    tampered2[cb_off + 7] ^= 0x40
    with tempfile.NamedTemporaryFile(suffix=".spill", delete=False) as f:
        f.write(bytes(tampered2))
        t2path = f.name
    out3 = subprocess.run([VERIFY, t2path, "--json"], capture_output=True, text=True)
    print("tampered codebook:", out3.stdout.strip() or out3.stderr.strip())
    if out3.returncode == 0 or "residual_mismatch" not in out3.stdout:
        print("FAIL: codebook tamper not caught as residual_mismatch", file=sys.stderr)
        return 1

    print("OK: honest accepted, wire tamper rejected, codebook tamper rejected, -0.0 survived")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_spill_roundtrip():
    """pytest entry (review 2026-09-02): the script's main() is the test."""
    assert main() in (0, None)
