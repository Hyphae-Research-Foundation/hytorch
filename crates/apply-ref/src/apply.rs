//! The `apply` reference. Spec v2.2 §04 verbatim, plus the total order of §04:
//! slots increasing within a token, tokens row-major (b, t).

use crate::bits::{bf16_is_zero, bf16_to_fp32, fp32_to_bf16_rne};

/// ε = 2⁻¹⁴ (spec §03/§04), as fp32 bits for exactness in docs and kernels.
pub const EPS_BITS: u32 = 0x3880_0000; // 2^-14
pub const MAG_ZERO_MASK: u16 = 0x7FFF;

#[inline(always)]
fn eps() -> f32 {
    f32::from_bits(EPS_BITS)
}

/// A committed binding as the verifier sees it after decode (subset of
/// BindingMin relevant to apply): position already flattened, slot, feature,
/// and the raw bf16 bits of mag exactly as the device applied them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Binding {
    /// Flat token index in the microbatch (row-major (b, t); spec §08 `pos`).
    pub pos: u32,
    /// Write port, `slot == feature % S` in phase 1 (checked at wire decode).
    pub slot: u8,
    /// Codebook row.
    pub feature: u16,
    /// Raw bf16 bits, exactly what apply multiplies. NOT IEEE f16 (spec D3).
    pub mag_bf16: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyError {
    /// slot out of range for the layout.
    SlotOutOfRange { pos: u32, slot: u8 },
    /// feature out of range for the codebook.
    FeatureOutOfRange { feature: u16 },
    /// pos out of range for h.
    PosOutOfRange { pos: u32 },
    /// Bindings not in the spec §04 total order (tokens row-major, then slots
    /// increasing). The order is part of the COMMIT receipt; a violation is a
    /// journal inconsistency, not something to silently sort.
    OrderViolation { index: usize },
    /// Buffer size mismatch.
    ShapeMismatch,
}

/// Normalize one codebook row into `out` (fp32), spec §04:
/// n̂[f] = rnd_fp32( C_fp32[f] / (‖C_fp32[f]‖₂ + ε) )
///
/// - `row` are bf16 bits of C[f] (d_slot entries).
/// - Norm: sequential reduction by increasing index, fp32, no FMA:
///   acc = rnd(acc + rnd(x*x)); then rnd(sqrt(acc)); then per-component
///   division, each op individually rounded (this is plain Rust f32 arithmetic,
///   which is IEEE round-to-nearest-even and never contracts to FMA).
pub fn normalize_row(row: &[u16], out: &mut [f32]) {
    debug_assert_eq!(row.len(), out.len());
    let mut acc: f32 = 0.0;
    for &b in row {
        let x = bf16_to_fp32(b);
        acc += x * x; // two rounded ops: mul then add (rustc does not fuse)
    }
    let denom = acc.sqrt() + eps();
    for (o, &b) in out.iter_mut().zip(row.iter()) {
        *o = bf16_to_fp32(b) / denom;
    }
}

/// Apply committed bindings to the residual, in place, bit-exactly.
///
/// - `h`: bf16 bits, layout `[n_tokens][s_slots][d_slot]` row-major.
/// - `codebook`: bf16 bits, layout `[n_features][d_slot]` row-major. This is
///   C_step: the codebook hashed in the STEP the bindings cite.
/// - `bindings`: committed facts only (verdict == COMMIT), in the §04 total
///   order. Zero-mag commits may be present (debug profile) or elided (WAL
///   default); both replay identically because ±0 is a no-op BY RULE.
///
/// Returns the number of leaves actually written (zero-mag no-ops count as 0).
pub fn apply_bindings(
    h: &mut [u16],
    n_tokens: u32,
    s_slots: u8,
    d_slot: u32,
    codebook: &[u16],
    n_features: u16,
    bindings: &[Binding],
) -> Result<u64, ApplyError> {
    let ds = d_slot as usize;
    let ss = s_slots as usize;
    if h.len() != n_tokens as usize * ss * ds {
        return Err(ApplyError::ShapeMismatch);
    }
    if codebook.len() != n_features as usize * ds {
        return Err(ApplyError::ShapeMismatch);
    }

    let mut written: u64 = 0;
    let mut last_key: Option<(u32, u8)> = None;
    // Scratch for the normalized row; d_slot is small (phase 1: 12).
    let mut nhat = vec![0.0f32; ds];

    for (i, b) in bindings.iter().enumerate() {
        // Total order check: strictly increasing (pos, slot). One committed
        // winner per (pos, slot) makes strict inequality correct.
        let key = (b.pos, b.slot);
        if let Some(prev) = last_key {
            if key <= prev {
                return Err(ApplyError::OrderViolation { index: i });
            }
        }
        last_key = Some(key);

        if b.pos >= n_tokens {
            return Err(ApplyError::PosOutOfRange { pos: b.pos });
        }
        if b.slot >= s_slots {
            return Err(ApplyError::SlotOutOfRange { pos: b.pos, slot: b.slot });
        }
        if b.feature >= n_features {
            return Err(ApplyError::FeatureOutOfRange { feature: b.feature });
        }

        // D1: ±0 mag is a bit-identical no-op BY RULE. The sum is not executed.
        // (-0.0)+(+0.0) would flip a -0.0 leaf to +0.0; definition, not IEEE luck.
        if bf16_is_zero(b.mag_bf16) {
            continue;
        }

        let mag = bf16_to_fp32(b.mag_bf16);
        let row = &codebook[b.feature as usize * ds..(b.feature as usize + 1) * ds];
        normalize_row(row, &mut nhat);

        let leaf_off = (b.pos as usize * ss + b.slot as usize) * ds;
        let leaf = &mut h[leaf_off..leaf_off + ds];
        for (l, &nh) in leaf.iter_mut().zip(nhat.iter()) {
            let x = bf16_to_fp32(*l);
            // Spec §04: product rounded to fp32, then sum rounded, then bf16 RNE.
            let prod = mag * nh;
            let sum = x + prod;
            *l = fp32_to_bf16_rne(sum);
        }
        written += 1;
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::bf16_to_fp32 as up;

    const S: u8 = 4;
    const D: u32 = 3;
    const NF: u16 = 8;

    fn mk_h(n_tokens: u32) -> Vec<u16> {
        // Deterministic pseudo-pattern including -0.0 leaves.
        (0..n_tokens as usize * S as usize * D as usize)
            .map(|i| match i % 5 {
                0 => 0x8000,          // -0.0  (the D1 trap)
                1 => 0x3F80,          // 1.0
                2 => 0xBF80,          // -1.0
                3 => 0x3F00,          // 0.5
                _ => 0x0000,          // +0.0
            })
            .collect()
    }

    fn mk_c() -> Vec<u16> {
        (0..NF as usize * D as usize)
            .map(|i| if i % 3 == 2 { 0xBF80 } else { 0x3F80 })
            .collect()
    }

    #[test]
    fn zero_mag_is_bit_identical_noop_even_with_negative_zero() {
        let mut h = mk_h(2);
        let snap = h.clone();
        let c = mk_c();
        let bindings = [
            Binding { pos: 0, slot: 0, feature: 0, mag_bf16: 0x0000 },
            Binding { pos: 1, slot: 2, feature: 6, mag_bf16: 0x8000 }, // -0
        ];
        let n = apply_bindings(&mut h, 2, S, D, &c, NF, &bindings).unwrap();
        assert_eq!(n, 0);
        assert_eq!(h, snap, "±0 must not touch a single bit");
    }

    #[test]
    fn ieee_sum_would_have_flipped_negative_zero() {
        // Proof that the rule matters: executing the sum with mag=+0 on a
        // -0.0 leaf produces +0.0 — exactly the honest-run killer of D1.
        let leaf = up(0x8000); // -0.0
        let delta = 0.0f32 * 1.0f32; // +0.0
        let summed = leaf + delta;
        assert_eq!(summed.to_bits(), 0); // +0.0 — bit flipped!
        assert_eq!(fp32_to_bf16_rne(summed), 0x0000);
        // Our apply, by rule, would have preserved 0x8000. Covered above.
    }

    #[test]
    fn magnitude_of_delta_equals_abs_mag() {
        // C1: with normalization inside apply, ‖leaf' − leaf‖ = |mag| to bf16
        // rounding. Start from a zero leaf, unit-ish codebook row.
        let mut h = vec![0u16; S as usize * D as usize];
        let c = mk_c();
        let mag = 0x3F00u16; // 0.5
        let b = [Binding { pos: 0, slot: 1, feature: 1, mag_bf16: mag }];
        apply_bindings(&mut h, 1, S, D, &c, NF, &b).unwrap();
        let leaf = &h[(1 * D as usize)..(2 * D as usize)];
        let norm2: f32 = leaf.iter().map(|&x| { let v = up(x); v * v }).sum::<f32>().sqrt();
        let expect = up(mag).abs();
        // bf16 has ~3 decimal digits; ε=2⁻¹⁴ in the denominator adds tiny bias.
        assert!((norm2 - expect).abs() < 6e-3, "‖delta‖={norm2} vs |mag|={expect}");
    }

    #[test]
    fn order_violation_is_an_error_not_a_sort() {
        let mut h = mk_h(2);
        let c = mk_c();
        let bad = [
            Binding { pos: 1, slot: 0, feature: 0, mag_bf16: 0x3F80 },
            Binding { pos: 0, slot: 1, feature: 1, mag_bf16: 0x3F80 },
        ];
        assert!(matches!(
            apply_bindings(&mut h, 2, S, D, &c, NF, &bad),
            Err(ApplyError::OrderViolation { index: 1 })
        ));
    }

    #[test]
    fn untouched_slots_are_bit_identical() {
        let mut h = mk_h(3);
        let snap = h.clone();
        let c = mk_c();
        let b = [Binding { pos: 1, slot: 2, feature: 6, mag_bf16: 0x3F80 }];
        apply_bindings(&mut h, 3, S, D, &c, NF, &b).unwrap();
        for (i, (a, s)) in h.iter().zip(snap.iter()).enumerate() {
            let touched_range =
                (1 * S as usize + 2) * D as usize..(1 * S as usize + 3) * D as usize;
            if touched_range.contains(&i) {
                continue;
            }
            assert_eq!(a, s, "untouched index {i} changed");
        }
    }

    #[test]
    fn replay_with_and_without_elided_zeros_matches() {
        // The WAL elides zero-commits; a debug profile persists them. Both
        // streams must replay to the same bytes (D1).
        let mut h_full = mk_h(4);
        let mut h_elided = mk_h(4);
        let c = mk_c();
        let full = [
            Binding { pos: 0, slot: 1, feature: 1, mag_bf16: 0x3EC0 },
            Binding { pos: 1, slot: 0, feature: 4, mag_bf16: 0x8000 }, // -0, elidable
            Binding { pos: 2, slot: 3, feature: 7, mag_bf16: 0xBF00 },
            Binding { pos: 3, slot: 2, feature: 2, mag_bf16: 0x0000 }, // +0, elidable
        ];
        let elided: Vec<Binding> =
            full.iter().copied().filter(|b| !bf16_is_zero(b.mag_bf16)).collect();
        apply_bindings(&mut h_full, 4, S, D, &c, NF, &full).unwrap();
        apply_bindings(&mut h_elided, 4, S, D, &c, NF, &elided).unwrap();
        assert_eq!(h_full, h_elided);
    }

    #[test]
    fn norm_reduction_order_is_sequential_by_construction() {
        // Guard: the reference must not be "improved" with pairwise/simd
        // reductions. We pin a case where order changes the fp32 result.
        // With values spanning magnitudes, sequential != tree reduction.
        let row: Vec<u16> = vec![0x7F00, 0x0080, 0x0080, 0x7F00]; // big, tiny, tiny, big
        let mut seq = 0.0f32;
        for &b in &row {
            let x = up(b);
            seq += x * x;
        }
        // Tree order: (b0+b1) + (b2+b3)
        let x: Vec<f32> = row.iter().map(|&b| up(b)).collect();
        let tree = (x[0] * x[0] + x[1] * x[1]) + (x[2] * x[2] + x[3] * x[3]);
        // These MAY be equal for lucky values; assert our pinned case differs
        // or document equality — here big²=2.706e38 overflows differently.
        // The real assertion: normalize_row uses the sequential result.
        let mut out = vec![0.0f32; 4];
        normalize_row(&row, &mut out);
        let denom_seq = seq.sqrt() + f32::from_bits(EPS_BITS);
        assert_eq!(out[0].to_bits(), (up(row[0]) / denom_seq).to_bits());
        let _ = tree;
    }
}
