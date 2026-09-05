//! bf16 <-> fp32 bit manipulation. Spec v2.2 §04.
//!
//! bf16 here means bfloat16: 1 sign, 8 exponent, 7 mantissa bits — the top 16
//! bits of an IEEE-754 binary32. It is NOT IEEE binary16 (f16); see spec D3.

/// Promote bf16 bits to fp32 by mantissa extension. Exact by construction:
/// bf16 is the top half of binary32, so `bits << 16` reproduces the value
/// (including ±0, subnormals, ±inf and NaN payloads) with zero rounding.
#[inline(always)]
pub const fn bf16_to_fp32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Downcast fp32 to bf16 bits with round-to-nearest-even on bit 16.
///
/// Policy (part of the reference; device kernels must match):
/// - NaN: preserve sign + quiet the result. Any NaN maps to a canonical
///   quiet NaN with the top mantissa bit set: `sign | 0x7FC0`. This avoids
///   the classic "NaN rounds to inf" bug of naive RNE and gives one canonical
///   NaN encoding so byte comparison is total.
/// - Overflow to ±inf happens naturally via RNE carry (IEEE behavior).
/// - Denormals pass through untouched (DTZ off).
#[inline(always)]
pub const fn fp32_to_bf16_rne(x: f32) -> u16 {
    let b = x.to_bits();
    // NaN check: exponent all ones and mantissa non-zero.
    if (b & 0x7F80_0000) == 0x7F80_0000 && (b & 0x007F_FFFF) != 0 {
        return (((b >> 16) as u16) & 0x8000) | 0x7FC0;
    }
    // RNE: add rounding bias; ties (lower half == 0x8000) round to even LSB.
    let lsb = (b >> 16) & 1;
    let rounded = b.wrapping_add(0x7FFF).wrapping_add(lsb);
    (rounded >> 16) as u16
}

/// True iff the bf16 bit pattern is +0 or -0 (spec §04 / D1: the no-op case).
#[inline(always)]
pub const fn bf16_is_zero(bits: u16) -> bool {
    bits & 0x7FFF == 0
}

/// True iff the bf16 bit pattern is NaN or ±inf (abort rule `nonfinite`).
#[inline(always)]
pub const fn bf16_is_nonfinite(bits: u16) -> bool {
    bits & 0x7F80 == 0x7F80
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_is_exact_for_all_bf16() {
        // Exhaustive: every one of the 65536 bf16 patterns round-trips
        // through fp32 promotion + RNE downcast unchanged (identity),
        // except non-canonical NaNs which map to the canonical quiet NaN.
        for bits in 0u16..=u16::MAX {
            let f = bf16_to_fp32(bits);
            let back = fp32_to_bf16_rne(f);
            if bf16_is_nonfinite(bits) && (bits & 0x007F) != 0 {
                // NaN: canonicalized, sign preserved.
                assert_eq!(back, (bits & 0x8000) | 0x7FC0, "bits={bits:#06x}");
            } else {
                assert_eq!(back, bits, "bits={bits:#06x}");
            }
        }
    }

    #[test]
    fn rne_ties_round_to_even_on_bit16() {
        // Construct fp32 values exactly halfway between two bf16 neighbors:
        // lower half == 0x8000. Even LSB stays, odd LSB rounds up.
        // 1.0 = 0x3F80_0000 (bf16 0x3F80, LSB 0). Halfway to next: 0x3F80_8000.
        assert_eq!(fp32_to_bf16_rne(f32::from_bits(0x3F80_8000)), 0x3F80);
        // 0x3F81 has LSB 1; halfway 0x3F81_8000 must round UP to 0x3F82.
        assert_eq!(fp32_to_bf16_rne(f32::from_bits(0x3F81_8000)), 0x3F82);
        // Just above/below the tie go to nearest.
        assert_eq!(fp32_to_bf16_rne(f32::from_bits(0x3F80_8001)), 0x3F81);
        assert_eq!(fp32_to_bf16_rne(f32::from_bits(0x3F80_7FFF)), 0x3F80);
    }

    #[test]
    fn signed_zero_preserved() {
        assert_eq!(fp32_to_bf16_rne(0.0f32), 0x0000);
        assert_eq!(fp32_to_bf16_rne(-0.0f32), 0x8000);
        assert!(bf16_is_zero(0x0000));
        assert!(bf16_is_zero(0x8000));
        assert!(!bf16_is_zero(0x0001));
    }

    #[test]
    fn overflow_rounds_to_inf() {
        // Max finite bf16 is 0x7F7F. Values above the RNE midpoint carry to inf.
        let just_over = f32::from_bits(0x7F7F_8001);
        assert_eq!(fp32_to_bf16_rne(just_over), 0x7F80); // +inf
    }

    #[test]
    fn denormals_pass_through() {
        // Smallest positive bf16 subnormal: 0x0001 -> promoted, back unchanged.
        let sub = bf16_to_fp32(0x0001);
        assert!(sub > 0.0 && sub < f32::MIN_POSITIVE);
        assert_eq!(fp32_to_bf16_rne(sub), 0x0001);
    }
}
