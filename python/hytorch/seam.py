"""Spill writer + T2 chain — Python mirror of verifier/src/spill.rs (v1).

The trainer dumps audited microbatches here; hytorch-verify replays them
same-binary. Byte layout must match crates/verifier/src/spill.rs exactly.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

GENESIS_HEAD = b"\x00" * 32
BINDING_MIN_SIZE = 16


def encode_min(rec) -> bytes:
    """rec: one row of the structured array from ApplyRef.pack_allocate,
    plus layer/device context via .item() access."""
    return struct.pack(
        "<HBBHHIHBB",
        int(rec["feature"]),
        int(rec["slot"]),
        int(rec["device"]) if "device" in rec.dtype.names else 0,
        int(rec["mag_bf16"]),
        int(rec["layer"]) if "layer" in rec.dtype.names else 0,
        int(rec["pos"]),
        int(rec["cand"]),
        int(rec["verdict"]),
        int(rec["reason"]),
    )


def layer_meta_bytes(step_id: int, layer: int, mb: int, n_persisted: int,
                     n_elided: int, policy_id: int) -> bytes:
    return struct.pack("<QHIIIQ", step_id, layer, mb, n_persisted, n_elided, policy_id)


def layer_head(prev_head: bytes, wire: bytes, meta: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(prev_head)
    h.update(wire)
    h.update(meta)
    return h.digest()


def is_zero_mag(mag_bf16: int) -> bool:
    return (mag_bf16 & 0x7FFF) == 0


class StepChain:
    """Builds the per-layer T2 chain for one microbatch, with zero elision
    BEFORE the hash (D1), and produces the spill payload for audited mbs."""

    def __init__(self, step_id: int, mb: int, policy_id: int, elide_zeros: bool = True):
        self.step_id = step_id
        self.mb = mb
        self.policy_id = policy_id
        self.elide = elide_zeros
        self.head = GENESIS_HEAD
        self.layers: list[tuple[bytes, int, int]] = []  # (wire, n_pers, n_elided)

    def add_layer(self, layer: int, recs: np.ndarray, device: int = 0) -> bytes:
        from .seamclient import encode_wire  # vectorized, single source of truth

        wire, n_pers, n_elided = encode_wire(recs, layer, device, self.elide)
        meta = layer_meta_bytes(self.step_id, layer, self.mb, n_pers, n_elided, self.policy_id)
        self.head = layer_head(self.head, wire, meta)
        self.layers.append((wire, n_pers, n_elided))
        return self.head

    def spill_bytes(self, h0_bits: np.ndarray, codebook_bits: np.ndarray,
                    s_slots: int, d_slot: int, h_final_bits: np.ndarray) -> bytes:
        """Spill v2: carries H(h_final) so T1 closes apply↔journal↔codebook.

        h_final_bits: the residual bits the device ACTUALLY produced after
        this microbatch's last apply (same shape as h0).
        """
        nt = h0_bits.size // (s_slots * d_slot)
        nf = codebook_bits.shape[0]
        h_final_sha = hashlib.sha256(
            h_final_bits.astype("<u2").tobytes()
        ).digest()
        out = bytearray()
        out += b"HYTSPILL"
        out += struct.pack("<I", 2)
        out += struct.pack("<QI", self.step_id, self.mb)
        out += struct.pack("<IB3xIHH", nt, s_slots, d_slot, len(self.layers), nf)
        out += self.head
        out += h_final_sha
        out += struct.pack("<Q", self.policy_id)
        out += h0_bits.astype("<u2").tobytes()
        out += codebook_bits.astype("<u2").tobytes()
        for wire, n_pers, n_elided in self.layers:
            out += struct.pack("<II", n_pers, n_elided)
            out += wire
        return bytes(out)
