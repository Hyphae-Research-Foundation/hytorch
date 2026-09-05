//! C ABI for the reference. The trainer's differential harness and the Python
//! side load this exact `.so`; its SHA-256 is `build.apply_ref_hash`.
//!
//! All pointers are caller-owned. Returns 0 on success, negative error codes
//! mirroring `ApplyError` otherwise; `written_out` receives the number of
//! non-noop leaf writes.

use crate::apply::{apply_bindings, Binding};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfiBinding {
    pub pos: u32,
    pub feature: u16,
    pub mag_bf16: u16,
    pub slot: u8,
    pub _pad: [u8; 3],
}

pub const APPLY_REF_ABI_VERSION: u32 = 1;

#[no_mangle]
pub extern "C" fn apply_ref_abi_version() -> u32 {
    APPLY_REF_ABI_VERSION
}

/// # Safety
/// - `h` points to `n_tokens * s_slots * d_slot` u16 elements, mutable.
/// - `codebook` points to `n_features * d_slot` u16 elements.
/// - `bindings` points to `n_bindings` FfiBinding, in spec §04 total order.
#[no_mangle]
pub unsafe extern "C" fn apply_ref_apply(
    h: *mut u16,
    n_tokens: u32,
    s_slots: u8,
    d_slot: u32,
    codebook: *const u16,
    n_features: u16,
    bindings: *const FfiBinding,
    n_bindings: usize,
    written_out: *mut u64,
) -> i32 {
    if h.is_null() || codebook.is_null() || (n_bindings > 0 && bindings.is_null()) {
        return -100;
    }
    let h_len = n_tokens as usize * s_slots as usize * d_slot as usize;
    let c_len = n_features as usize * d_slot as usize;
    let h_slice = unsafe { core::slice::from_raw_parts_mut(h, h_len) };
    let c_slice = unsafe { core::slice::from_raw_parts(codebook, c_len) };
    let b_slice = unsafe { core::slice::from_raw_parts(bindings, n_bindings) };

    let owned: Vec<Binding> = b_slice
        .iter()
        .map(|f| Binding { pos: f.pos, slot: f.slot, feature: f.feature, mag_bf16: f.mag_bf16 })
        .collect();

    match apply_bindings(h_slice, n_tokens, s_slots, d_slot, c_slice, n_features, &owned) {
        Ok(w) => {
            if !written_out.is_null() {
                unsafe { *written_out = w };
            }
            0
        }
        Err(e) => match e {
            crate::apply::ApplyError::ShapeMismatch => -1,
            crate::apply::ApplyError::PosOutOfRange { .. } => -2,
            crate::apply::ApplyError::SlotOutOfRange { .. } => -3,
            crate::apply::ApplyError::FeatureOutOfRange { .. } => -4,
            crate::apply::ApplyError::OrderViolation { .. } => -5,
        },
    }
}

/// Scalar helpers exported for the device differential harness (single-value
/// probes of the rounding policy).
#[no_mangle]
pub extern "C" fn apply_ref_bf16_to_fp32(bits: u16) -> f32 {
    crate::bits::bf16_to_fp32(bits)
}

#[no_mangle]
pub extern "C" fn apply_ref_fp32_to_bf16_rne(x: f32) -> u16 {
    crate::bits::fp32_to_bf16_rne(x)
}

/// pack + allocate over a microbatch. Caller provides the output buffer of
/// exactly `n_tokens * k` VerdictRec (spec §07: the partition is exact).
/// Returns 0 on success and writes `n_out`; negative on argument errors.
///
/// # Safety
/// - `delta_hat`: `n_tokens * s_slots * d_slot` u16.
/// - `codebook`: `n_features * d_slot` u16.
/// - `out`: `n_tokens * k` VerdictRec, caller-owned.
#[no_mangle]
pub unsafe extern "C" fn apply_ref_pack_allocate(
    delta_hat: *const u16,
    n_tokens: u32,
    s_slots: u8,
    d_slot: u32,
    codebook: *const u16,
    n_features: u16,
    k: u16,
    mag_max: f32,
    out: *mut crate::pack::VerdictRec,
    n_out: *mut usize,
) -> i32 {
    if delta_hat.is_null() || codebook.is_null() || out.is_null() {
        return -100;
    }
    if k == 0 || k > 8 || n_features < k {
        return -101;
    }
    let dh_len = n_tokens as usize * s_slots as usize * d_slot as usize;
    let c_len = n_features as usize * d_slot as usize;
    let dh = unsafe { core::slice::from_raw_parts(delta_hat, dh_len) };
    let cb = unsafe { core::slice::from_raw_parts(codebook, c_len) };
    let recs = crate::pack::pack_allocate(
        dh,
        n_tokens,
        cb,
        crate::pack::PackParams {
            s_slots, d_slot, n_features, k, mag_max,
            selection: crate::pack::SELECT_GLOBAL_TOPK,
        },
    );
    let cap = n_tokens as usize * k as usize;
    if recs.len() > cap {
        return -102;
    }
    let out_slice = unsafe { core::slice::from_raw_parts_mut(out, cap) };
    out_slice[..recs.len()].copy_from_slice(&recs);
    if !n_out.is_null() {
        unsafe { *n_out = recs.len() };
    }
    0
}

/// pack + allocate v3 (SPEC_AMEND-004, policy 7 two_phase_topk): exact
/// policy-6 selection machinery over a DEVICE-PROPOSED candidate set.
/// `candidates` is `[n_tokens][m]` u16 feature ids (>= n_features = padding,
/// duplicates deduped keeping the first proposal rank). Scores of candidates
/// are recomputed with the pinned math; verdicts/mags/order are exact.
///
/// # Safety
/// Same contracts as v2, plus `candidates` points to `n_tokens * m` u16.
#[no_mangle]
pub unsafe extern "C" fn apply_ref_pack_allocate_candidates(
    delta_hat: *const u16,
    n_tokens: u32,
    s_slots: u8,
    d_slot: u32,
    codebook: *const u16,
    n_features: u16,
    candidates: *const u16,
    m: u32,
    k: u16,
    mag_max: f32,
    out: *mut crate::pack::VerdictRec,
    n_out: *mut usize,
) -> i32 {
    if delta_hat.is_null() || codebook.is_null() || candidates.is_null() || out.is_null() {
        return -100;
    }
    if k == 0 || k > 8 || n_features < k || m == 0 {
        return -101;
    }
    let dh_len = n_tokens as usize * s_slots as usize * d_slot as usize;
    let c_len = n_features as usize * d_slot as usize;
    let cd_len = n_tokens as usize * m as usize;
    let dh = unsafe { core::slice::from_raw_parts(delta_hat, dh_len) };
    let cb = unsafe { core::slice::from_raw_parts(codebook, c_len) };
    let cd = unsafe { core::slice::from_raw_parts(candidates, cd_len) };
    let recs = crate::pack::pack_allocate_candidates(
        dh,
        n_tokens,
        cb,
        cd,
        m,
        crate::pack::PackParams {
            s_slots, d_slot, n_features, k, mag_max,
            selection: crate::pack::SELECT_GLOBAL_TOPK,
        },
    );
    let cap = n_tokens as usize * k as usize;
    if recs.len() > cap {
        return -102;
    }
    let out_slice = unsafe { core::slice::from_raw_parts_mut(out, cap) };
    out_slice[..recs.len()].copy_from_slice(&recs);
    if !n_out.is_null() {
        unsafe { *n_out = recs.len() };
    }
    0
}

/// pack + allocate v2: adds the selection policy byte (SELECT_GLOBAL_TOPK=0,
/// SELECT_SLOT_TOPK=1). v1 stays global-only for ABI stability.
///
/// # Safety
/// Same contracts as `apply_ref_pack_allocate`.
#[no_mangle]
pub unsafe extern "C" fn apply_ref_pack_allocate_v2(
    delta_hat: *const u16,
    n_tokens: u32,
    s_slots: u8,
    d_slot: u32,
    codebook: *const u16,
    n_features: u16,
    k: u16,
    mag_max: f32,
    selection: u8,
    out: *mut crate::pack::VerdictRec,
    n_out: *mut usize,
) -> i32 {
    if delta_hat.is_null() || codebook.is_null() || out.is_null() {
        return -100;
    }
    if k == 0 || k > 8 || n_features < k {
        return -101;
    }
    if selection > crate::pack::SELECT_SLOT_TOPK {
        return -103;
    }
    if selection == crate::pack::SELECT_SLOT_TOPK && k > s_slots as u16 {
        return -104;
    }
    let dh_len = n_tokens as usize * s_slots as usize * d_slot as usize;
    let c_len = n_features as usize * d_slot as usize;
    let dh = unsafe { core::slice::from_raw_parts(delta_hat, dh_len) };
    let cb = unsafe { core::slice::from_raw_parts(codebook, c_len) };
    let recs = crate::pack::pack_allocate(
        dh,
        n_tokens,
        cb,
        crate::pack::PackParams { s_slots, d_slot, n_features, k, mag_max, selection },
    );
    let cap = n_tokens as usize * k as usize;
    if recs.len() > cap {
        return -102;
    }
    let out_slice = unsafe { core::slice::from_raw_parts_mut(out, cap) };
    out_slice[..recs.len()].copy_from_slice(&recs);
    if !n_out.is_null() {
        unsafe { *n_out = recs.len() };
    }
    0
}
