//! apply-ref: the bit-exact software semiring of spec v2.2 §04.
//!
//! This crate is the single authority on what `apply()` computes. The trainer's
//! device kernels must match it bit for bit or the backend does not enter the
//! harness. The verifier links this exact object (same `build.apply_ref_hash`).
//!
//! Contract (spec §04):
//!   n̂[f]  = rnd_fp32( C_fp32[f] / (‖C_fp32[f]‖₂ + ε) )
//!   leaf' = rnd_bf16( leaf_fp32 + rnd_fp32(mag_bf16) * n̂[f] )
//!   si mag_bf16 == ±0:  leaf' = leaf   (bit-identical copy; the sum is NOT executed)
//!
//! - bf16 -> fp32 promotion by mantissa extension (exact: `(bits as u32) << 16`).
//! - fp32 ops are IEEE-754, round-to-nearest-even, **no FMA** (each op rounded).
//! - fp32 -> bf16 downcast: RNE on bit 16, NaN payload policy defined below.
//! - Denormals are NOT flushed (DTZ off).
//! - Norm reduction: sequential by increasing index, fp32, no FMA.
//! - Application order: slots increasing within a (b,t); tokens row-major.
//!
//! No dependencies. No allocation in the hot path beyond caller buffers.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod bits;
pub mod apply;
pub mod pack;
pub mod ffi;

pub use apply::{apply_bindings, normalize_row, ApplyError, Binding, EPS_BITS, MAG_ZERO_MASK};
pub use bits::{bf16_to_fp32, fp32_to_bf16_rne};
pub use pack::{pack_allocate, PackParams, VerdictRec, SELECT_GLOBAL_TOPK, SELECT_SLOT_TOPK};
