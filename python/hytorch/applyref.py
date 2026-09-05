"""ctypes bindings to libapply_ref.so — the pinned reference.

The .so is the single authority (build.apply_ref_hash). Python never
reimplements the policy; it CALLS it. Spec v2.2 §04, §03.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from dataclasses import dataclass

import numpy as np


class FfiBinding(ctypes.Structure):
    _fields_ = [
        ("pos", ctypes.c_uint32),
        ("feature", ctypes.c_uint16),
        ("mag_bf16", ctypes.c_uint16),
        ("slot", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 3),
    ]


class VerdictRec(ctypes.Structure):
    _fields_ = [
        ("pos", ctypes.c_uint32),
        ("feature", ctypes.c_uint16),
        ("mag_bf16", ctypes.c_uint16),
        ("cand", ctypes.c_uint16),
        ("slot", ctypes.c_uint8),
        ("verdict", ctypes.c_uint8),
        ("reason", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 3),
    ]


VERDICT_COMMIT = 0
VERDICT_OVERFLOW = 1
VERDICT_ABORT = 2


@dataclass
class ApplyRef:
    path: str
    sha256: str
    lib: ctypes.CDLL

    @classmethod
    def load(cls, path: str) -> "ApplyRef":
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        lib = ctypes.CDLL(os.path.abspath(path))
        lib.apply_ref_abi_version.restype = ctypes.c_uint32
        assert lib.apply_ref_abi_version() == 1, "ABI mismatch"

        lib.apply_ref_apply.restype = ctypes.c_int32
        lib.apply_ref_apply.argtypes = [
            ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint32, ctypes.c_uint8,
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16,
            ctypes.POINTER(FfiBinding), ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.apply_ref_pack_allocate_v2.restype = ctypes.c_int32
        lib.apply_ref_pack_allocate_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint32, ctypes.c_uint8,
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16,
            ctypes.c_uint16, ctypes.c_float, ctypes.c_uint8,
            ctypes.POINTER(VerdictRec), ctypes.POINTER(ctypes.c_size_t),
        ]
        # v3: two-phase candidates (SPEC_AMEND-004, policy 7). Older .so
        # builds lack the symbol; probe so phase<7 users keep working.
        try:
            lib.apply_ref_pack_allocate_candidates.restype = ctypes.c_int32
            lib.apply_ref_pack_allocate_candidates.argtypes = [
                ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint32, ctypes.c_uint8,
                ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint16,
                ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint32,
                ctypes.c_uint16, ctypes.c_float,
                ctypes.POINTER(VerdictRec), ctypes.POINTER(ctypes.c_size_t),
            ]
        except AttributeError:
            pass
        return cls(path=path, sha256=digest, lib=lib)

    def pack_allocate(
        self,
        delta_hat_bits: np.ndarray,  # uint16 [n_tokens, S, d_slot]
        codebook_bits: np.ndarray,   # uint16 [n_features, d_slot]
        k: int,
        mag_max: float,
        selection: int = 0,          # 0=global_topk (phase 1), 1=slot_topk
    ) -> np.ndarray:
        nt, s, d = delta_hat_bits.shape
        nf = codebook_bits.shape[0]
        dh = np.ascontiguousarray(delta_hat_bits, dtype=np.uint16)
        cb = np.ascontiguousarray(codebook_bits, dtype=np.uint16)
        out = (VerdictRec * (nt * k))()
        n_out = ctypes.c_size_t(0)
        rc = self.lib.apply_ref_pack_allocate_v2(
            dh.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_uint32(nt), ctypes.c_uint8(s), ctypes.c_uint32(d),
            cb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_uint16(nf), ctypes.c_uint16(k), ctypes.c_float(mag_max),
            ctypes.c_uint8(selection),
            out, ctypes.byref(n_out),
        )
        if rc != 0:
            raise RuntimeError(f"apply_ref_pack_allocate_v2 rc={rc}")
        n = n_out.value
        recs = np.zeros(n, dtype=[
            ("pos", "<u4"), ("feature", "<u2"), ("mag_bf16", "<u2"),
            ("cand", "<u2"), ("slot", "u1"), ("verdict", "u1"), ("reason", "u1"),
        ])
        for i in range(n):
            r = out[i]
            recs[i] = (r.pos, r.feature, r.mag_bf16, r.cand, r.slot, r.verdict, r.reason)
        return recs

    def pack_allocate_candidates(
        self,
        delta_hat_bits: np.ndarray,  # uint16 [n_tokens, S, d_slot]
        codebook_bits: np.ndarray,   # uint16 [n_features, d_slot]
        candidates: np.ndarray,      # uint16 [n_tokens, M] device-proposed ids
        k: int,
        mag_max: float,
    ) -> np.ndarray:
        """Policy 7 (SPEC_AMEND-004): exact selection over proposed candidates."""
        nt, s, d = delta_hat_bits.shape
        nf = codebook_bits.shape[0]
        m = candidates.shape[1]
        assert candidates.shape[0] == nt
        dh = np.ascontiguousarray(delta_hat_bits, dtype=np.uint16)
        cb = np.ascontiguousarray(codebook_bits, dtype=np.uint16)
        cd = np.ascontiguousarray(candidates, dtype=np.uint16)
        out = (VerdictRec * (nt * k))()
        n_out = ctypes.c_size_t(0)
        rc = self.lib.apply_ref_pack_allocate_candidates(
            dh.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_uint32(nt), ctypes.c_uint8(s), ctypes.c_uint32(d),
            cb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_uint16(nf),
            cd.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_uint32(m),
            ctypes.c_uint16(k), ctypes.c_float(mag_max),
            out, ctypes.byref(n_out),
        )
        if rc != 0:
            raise RuntimeError(f"apply_ref_pack_allocate_candidates rc={rc}")
        n = n_out.value
        buf = np.frombuffer(out, dtype=np.uint8, count=n * 16).reshape(n, 16)
        recs = np.zeros(n, dtype=[
            ("pos", "<u4"), ("feature", "<u2"), ("mag_bf16", "<u2"),
            ("cand", "<u2"), ("slot", "u1"), ("verdict", "u1"), ("reason", "u1"),
        ])
        recs["pos"] = buf[:, 0:4].copy().view("<u4").ravel()
        recs["feature"] = buf[:, 4:6].copy().view("<u2").ravel()
        recs["mag_bf16"] = buf[:, 6:8].copy().view("<u2").ravel()
        recs["cand"] = buf[:, 8:10].copy().view("<u2").ravel()
        recs["slot"] = buf[:, 10]
        recs["verdict"] = buf[:, 11]
        recs["reason"] = buf[:, 12]
        return recs

    def apply(
        self,
        h_bits: np.ndarray,          # uint16 [n_tokens, S, d_slot] — mutated!
        codebook_bits: np.ndarray,   # uint16 [n_features, d_slot]
        commits: np.ndarray,         # structured array from pack_allocate, COMMITs only
    ) -> int:
        nt, s, d = h_bits.shape
        nf = codebook_bits.shape[0]
        assert h_bits.flags["C_CONTIGUOUS"]
        cb = np.ascontiguousarray(codebook_bits, dtype=np.uint16)
        n = len(commits)
        arr = (FfiBinding * n)()
        for i, c in enumerate(commits):
            arr[i].pos = int(c["pos"])
            arr[i].feature = int(c["feature"])
            arr[i].mag_bf16 = int(c["mag_bf16"])
            arr[i].slot = int(c["slot"])
        written = ctypes.c_uint64(0)
        rc = self.lib.apply_ref_apply(
            h_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_uint32(nt), ctypes.c_uint8(s), ctypes.c_uint32(d),
            cb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_uint16(nf), arr, ctypes.c_size_t(n), ctypes.byref(written),
        )
        if rc != 0:
            raise RuntimeError(f"apply_ref_apply rc={rc}")
        return written.value


def fp32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """RNE downcast matching the reference (vectorized numpy mirror).

    Only used to PREPARE inputs (delta_hat, codebook) for the .so; every
    authoritative operation happens inside the .so itself.
    """
    b = x.astype(np.float32).view(np.uint32)
    nan = (b & 0x7F800000 == 0x7F800000) & (b & 0x007FFFFF != 0)
    lsb = (b >> 16) & 1
    rounded = b + 0x7FFF + lsb
    out = (rounded >> 16).astype(np.uint16)
    out[nan] = (((b[nan] >> 16) & 0x8000) | 0x7FC0).astype(np.uint16)
    return out


def bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)
