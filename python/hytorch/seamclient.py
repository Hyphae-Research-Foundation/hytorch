"""Trainer-side seam client, protocol v2: RUN_START handshake, per-step
barrier (Phase A), step chain with real c_prev/c_next (Phase B), EXPORT.

The barrier (spec §5.4): if an ack does not appear within budget_ms, the run
dies — there is NO degradation to running without the ledger.

Wire encode is vectorized: one numpy scatter per layer, not a Python loop
per record (at 24k facts/step the loop was the bottleneck).
"""

from __future__ import annotations

import json
import os
import struct
import time

import numpy as np

FRAME_MAGIC = b"HYTFRAME"
FRAME_VERSION = 1

WIRE_DTYPE = np.dtype([
    ("feature", "<u2"), ("slot", "u1"), ("device", "u1"), ("mag_bf16", "<u2"),
    ("layer", "<u2"), ("pos", "<u4"), ("cand", "<u2"), ("verdict", "u1"),
    ("reason", "u1"),
])
assert WIRE_DTYPE.itemsize == 16


class BarrierTimeout(RuntimeError):
    pass


def encode_wire(recs: np.ndarray, layer: int, device: int,
                elide_zeros: bool) -> tuple[bytes, int, int]:
    """Vectorized BindingMin encode. Returns (wire, n_persisted, n_elided)."""
    if elide_zeros:
        zero_commit = (recs["verdict"] == 0) & ((recs["mag_bf16"] & 0x7FFF) == 0)
        keep = recs[~zero_commit]
        n_elided = int(zero_commit.sum())
    else:
        keep = recs
        n_elided = 0
    out = np.zeros(len(keep), dtype=WIRE_DTYPE)
    out["feature"] = keep["feature"]
    out["slot"] = keep["slot"]
    out["device"] = device
    out["mag_bf16"] = keep["mag_bf16"]
    out["layer"] = layer
    out["pos"] = keep["pos"]
    out["cand"] = keep["cand"]
    out["verdict"] = keep["verdict"]
    out["reason"] = keep["reason"]
    return out.tobytes(), len(keep), n_elided


class SeamClient:
    def __init__(self, spool_dir: str, policy_id: int, elide_zeros: bool = True):
        self.spool = spool_dir
        self.policy_id = policy_id
        self.elide = elide_zeros
        os.makedirs(spool_dir, exist_ok=True)

    # ---- atomic writers ----
    def _write(self, name: str, data: bytes) -> None:
        path = os.path.join(self.spool, name)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)

    def _wait(self, name: str, budget_ms: int, consume: bool = True) -> str:
        path = os.path.join(self.spool, name)
        deadline = time.monotonic() + budget_ms / 1000.0
        while time.monotonic() < deadline:
            if os.path.exists(path):
                with open(path) as f:
                    content = f.read().strip()
                if consume:
                    os.remove(path)
                return content
            time.sleep(0.001)
        raise BarrierTimeout(
            f"no {name} within {budget_ms} ms — KILL THE RUN "
            f"(spec §5.4: no ledger-less degradation)"
        )

    # ---- protocol v2 ----
    def run_start(self, manifest_sha256: str, build: dict, infra: dict,
                  policy: dict, budget_ms: int = 30000) -> None:
        payload = {
            "manifest_sha256": manifest_sha256,
            "build.apply_ref_hash": build["build.apply_ref_hash"],
            "build.harness_commit": build["build.harness_commit"],
            "build.torch_wheel": build["build.torch_wheel"],
            "build.backend_wheel": build["build.backend_wheel"],
            **{k: str(v) for k, v in infra.items()},
            **policy,
        }
        self._write("run-start.json", json.dumps(payload).encode())
        self._wait("run-start.ack", budget_ms, consume=False)

    def spool_layer(self, step_id: int, layer: int, mb: int, recs: np.ndarray,
                    device: int = 0, suffix: str = "") -> tuple[int, int]:
        """suffix (e.g. '-r00-m01') disambiguates multi-rank / grad-accum
        frames; the seam sorts filenames, so the T2 order is (layer, rank,
        micro) — deterministic. Header fields are unchanged."""
        wire, n_pers, n_elided = encode_wire(recs, layer, device, self.elide)
        header = FRAME_MAGIC + struct.pack(
            "<IQHIIIQ", FRAME_VERSION, step_id, layer, mb, n_pers, n_elided,
            self.policy_id,
        )
        self._write(f"step-{step_id}-layer-{layer:04d}{suffix}.frame", header + wire)
        return n_pers, n_elided

    def barrier(self, step_id: int, budget_ms: int) -> str:
        """Phase A: all frames spooled → wait for the receipt head.
        opt.step() without this return value is ILLEGAL."""
        self._write(f"step-{step_id}.barrier", b"ok\n")
        return self._wait(f"step-{step_id}.receipt", budget_ms)

    def step_chain(self, step_id: int, lr: float, grad_norm: float,
                   c_prev: bytes, c_next: bytes, resets: list[int],
                   seal_theta: bytes | None, budget_ms: int,
                   bypass: dict[str, list[float]] | None = None) -> None:
        """Phase B: after opt.step()+renorm, commit the STEP chain.

        bypass: Law-0 bypass declaration — residual mutations outside the
        catalog write (e.g. nanochat per-layer resid/x0 lambdas). Recorded as
        a BYPASS fact next to the STEP; the ledger never pretends they
        do not exist (review 2026-09-02)."""
        lines = [
            f"{lr} {grad_norm} {self.policy_id}",
            c_prev.hex(),
            c_next.hex(),
            ",".join(str(r) for r in resets),
            seal_theta.hex() if seal_theta else "-",
            ";".join(f"{k}={','.join(repr(float(v)) for v in vals)}"
                     for k, vals in (bypass or {}).items()) or "-",
        ]
        self._write(f"step-{step_id}.facts", ("\n".join(lines) + "\n").encode())
        self._wait(f"step-{step_id}.chainack", budget_ms)

    def export(self, step_id: int, theta_sha256: str, c_sha256: str,
               final_head: str, budget_ms: int = 30000) -> None:
        self._write("export.json", json.dumps({
            "step_id": step_id, "theta_sha256": theta_sha256,
            "c_sha256": c_sha256, "final_head": final_head,
        }).encode())
        self._wait("export.ack", budget_ms, consume=False)
