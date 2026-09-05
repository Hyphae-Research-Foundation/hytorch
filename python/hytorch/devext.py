"""Device extension loader + GPU-native verdict handling.

Loads the gate-authorized kernels as a torch extension (JIT, nvcc/hipcc).
The -fmad=false / -ffp-contract=off flags are NON-NEGOTIABLE (spec §04).

Zero-sync layer loop (the Lámina C contract, "cero sync por capa"):
- pack_allocate_t returns the raw verdict tensor ON DEVICE.
- Field extraction (pos/feature/mag/slot/verdict) is int32 word arithmetic
  ON DEVICE — no numpy roundtrip.
- The DevBinding commit buffer for apply is built ON DEVICE.
- The journal D2H is an async copy to pinned memory, materialized to numpy
  AFTER backward (d2h.overlap = backward).

VerdictRec wire layout (16 B, little-endian) as int32 words:
  w0 = pos
  w1 = feature | mag_bf16 << 16
  w2 = cand | slot << 16 | verdict << 24
  w3 = reason | pad
DevBinding (12 B) as int32 words: b0 = pos, b1 = w1, b2 = slot.
"""

from __future__ import annotations

import os

import numpy as np
import torch

_EXT = None


def load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "..", "..", "kernels", "cuda", "torch_ext.cu")
    if getattr(torch.version, "hip", None):
        flags = [
            "-O2",
            "-ffp-contract=off",
            "-fhip-fp32-correctly-rounded-divide-sqrt",
        ]
    else:
        flags = ["-O2", "-fmad=false"]
    _EXT = load(
        name="hytorch_dev",
        sources=[src],
        extra_cuda_cflags=flags,
        extra_include_paths=[os.path.dirname(src)],
        verbose=False,
    )
    return _EXT


VERDICT_DTYPE = np.dtype(
    [
        ("pos", "<u4"), ("feature", "<u2"), ("mag_bf16", "<u2"), ("cand", "<u2"),
        ("slot", "u1"), ("verdict", "u1"), ("reason", "u1"), ("_pad", "V3"),
    ]
)


def raw_fields(raw: torch.Tensor) -> dict:
    """Extract VerdictRec fields from the raw [N,16] uint8 tensor ON DEVICE.

    int32 word arithmetic; bitwise ops are two's-complement so the sign bit
    from mag ≥ 0x8000 in w1 is harmless under the masks.
    """
    w = raw.view(torch.int32)  # [N, 4]
    pos = w[:, 0].to(torch.int64)
    feature = (w[:, 1] & 0xFFFF).to(torch.int64)
    mag_bits = (w[:, 1] >> 16) & 0xFFFF
    cand = (w[:, 2] & 0xFFFF).to(torch.int64)
    slot = ((w[:, 2] >> 16) & 0xFF).to(torch.int64)
    verdict = (w[:, 2] >> 24) & 0xFF
    return {
        "pos": pos, "feature": feature, "mag_bits": mag_bits,
        "cand": cand, "slot": slot, "verdict": verdict, "w": w,
    }


def mag_bits_to_f32(mag_bits: torch.Tensor) -> torch.Tensor:
    """bf16 bits (as int32 low half) → fp32 value, exact by construction."""
    return (mag_bits.to(torch.int32) << 16).contiguous().view(torch.float32)


class DeviceCatalog:
    """Device-side pack/allocate/apply using the gate-authorized kernels.

    The verdict stream is BIT-COMPATIBLE with the reference (that is exactly
    what the M3 differential gate proves, per silicon).
    """

    def __init__(self):
        self.ext = load_ext()
        self._nhat = None

    def pack_allocate_t(self, delta_bits: torch.Tensor, codebook_bits: torch.Tensor,
                        k: int, mag_max: float, selection: int = 0) -> torch.Tensor:
        """→ raw verdicts [NT*k, 16] uint8, ON DEVICE. No syncs."""
        nhat = self.ext.normalize_rows(codebook_bits)
        raw = self.ext.pack_allocate(delta_bits, nhat, k, mag_max, selection)
        self._nhat = nhat
        return raw

    def apply_commits_raw(self, h_bits: torch.Tensor, raw: torch.Tensor,
                          fields: dict) -> torch.Tensor:
        """Build DevBinding buffer ON DEVICE from COMMIT rows and apply.
        Returns the commit mask (device). No syncs."""
        cm = fields["verdict"] == 0
        w = fields["w"]
        slot32 = fields["slot"].to(torch.int32)
        db = torch.stack([w[:, 0], w[:, 1], slot32], dim=1)[cm].contiguous()
        commits_u8 = db.view(torch.uint8).reshape(-1, 12)
        self.ext.apply_committed(h_bits, self._nhat, commits_u8)
        return cm
