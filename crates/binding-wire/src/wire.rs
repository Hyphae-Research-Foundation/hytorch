//! Wire structs. Spec v2.2 §08 (Lámina F), D3, D4, D5c.
//!
//! ```text
//! struct BindingMin {              // packed, little-endian, 16 bytes
//!   u16  feature;   // 2  N_f ≤ 32768
//!   u8   slot;      // 1  S ≤ 64; invariant slot == feature % S (phase 1)
//!   u8   device;    // 1  cuda=0, rocm=1
//!   u16  mag_bf16;  // 2  raw bf16 bits, exactly what apply multiplied (D3)
//!   u16  layer;     // 2
//!   u32  pos;       // 4  flat index in the microbatch
//!   u16  cand;      // 2  0..k-1, stable tiebreak
//!   u8   verdict;   // 1  COMMIT=0 OVERFLOW=1 ABORT=2      (D4)
//!   u8   reason;    // 1  none=0 nonfinite=1 mag_overflow=2 policy=3 (D4)
//! };
//! ```
//! step_id / policy_id / run_id / microbatch_id live in the STEP / layer head,
//! once per layer, not per fact.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Verdict {
    Commit = 0,
    Overflow = 1,
    Abort = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Reason {
    None = 0,
    Nonfinite = 1,
    MagOverflow = 2,
    Policy = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindingMin {
    pub feature: u16,
    pub slot: u8,
    pub device: u8,
    pub mag_bf16: u16,
    pub layer: u16,
    pub pos: u32,
    pub cand: u16,
    pub verdict: Verdict,
    pub reason: Reason,
}

pub const BINDING_MIN_SIZE: usize = 16;
pub const BINDING_T15_SIZE: usize = 80;

/// t1_5 profile: min + prev head + H(pre_leaf). Spec §08.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindingT15 {
    pub base: BindingMin,
    pub prev: [u8; 32],
    pub pre_leaf_h: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireError {
    ShortBuffer,
    BadVerdict(u8),
    BadReason(u8),
    BadDevice(u8),
    /// Phase-1 ingest invariant: slot == feature % S (spec D5c, declared
    /// tautology, verified per record).
    SlotHomeViolation { feature: u16, slot: u8 },
    /// Verdict/reason combinations that cannot exist (e.g. COMMIT+nonfinite).
    VerdictReasonMismatch { verdict: u8, reason: u8 },
}

pub fn encode_min(b: &BindingMin, out: &mut [u8]) -> Result<(), WireError> {
    if out.len() < BINDING_MIN_SIZE {
        return Err(WireError::ShortBuffer);
    }
    out[0..2].copy_from_slice(&b.feature.to_le_bytes());
    out[2] = b.slot;
    out[3] = b.device;
    out[4..6].copy_from_slice(&b.mag_bf16.to_le_bytes());
    out[6..8].copy_from_slice(&b.layer.to_le_bytes());
    out[8..12].copy_from_slice(&b.pos.to_le_bytes());
    out[12..14].copy_from_slice(&b.cand.to_le_bytes());
    out[14] = b.verdict as u8;
    out[15] = b.reason as u8;
    Ok(())
}

/// Decode + validate. `s_slots` enforces the phase-1 home invariant.
pub fn decode_min(buf: &[u8], s_slots: u8) -> Result<BindingMin, WireError> {
    if buf.len() < BINDING_MIN_SIZE {
        return Err(WireError::ShortBuffer);
    }
    let feature = u16::from_le_bytes([buf[0], buf[1]]);
    let slot = buf[2];
    let device = buf[3];
    let mag_bf16 = u16::from_le_bytes([buf[4], buf[5]]);
    let layer = u16::from_le_bytes([buf[6], buf[7]]);
    let pos = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]);
    let cand = u16::from_le_bytes([buf[12], buf[13]]);
    let verdict = match buf[14] {
        0 => Verdict::Commit,
        1 => Verdict::Overflow,
        2 => Verdict::Abort,
        v => return Err(WireError::BadVerdict(v)),
    };
    let reason = match buf[15] {
        0 => Reason::None,
        1 => Reason::Nonfinite,
        2 => Reason::MagOverflow,
        3 => Reason::Policy,
        r => return Err(WireError::BadReason(r)),
    };
    if device > 1 {
        return Err(WireError::BadDevice(device));
    }
    // Phase-1 ingest invariant (D5c): slot == feature mod S.
    if s_slots > 0 && slot != (feature % s_slots as u16) as u8 {
        return Err(WireError::SlotHomeViolation { feature, slot });
    }
    // D4 sanity: COMMIT carries reason none; ABORT carries a real reason;
    // OVERFLOW carries none (losing is its own fact).
    let ok = match verdict {
        Verdict::Commit | Verdict::Overflow => reason == Reason::None,
        Verdict::Abort => reason != Reason::None,
    };
    if !ok {
        return Err(WireError::VerdictReasonMismatch { verdict: buf[14], reason: buf[15] });
    }
    Ok(BindingMin { feature, slot, device, mag_bf16, layer, pos, cand, verdict, reason })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(i: u32) -> BindingMin {
        // Deterministic generator with valid home invariant for S=64.
        let feature = (i % 32768) as u16;
        BindingMin {
            feature,
            slot: (feature % 64) as u8,
            device: (i % 2) as u8,
            mag_bf16: (i.wrapping_mul(2654435761) >> 16) as u16,
            layer: (i % 12) as u16,
            pos: i % 2048,
            cand: (i % 8) as u16,
            verdict: match i % 3 {
                0 => Verdict::Commit,
                1 => Verdict::Overflow,
                _ => Verdict::Abort,
            },
            reason: match i % 3 {
                0 | 1 => Reason::None,
                _ => match i % 2 {
                    0 => Reason::Nonfinite,
                    _ => Reason::MagOverflow,
                },
            },
        }
    }

    #[test]
    fn size_is_exactly_16() {
        assert_eq!(BINDING_MIN_SIZE, 16);
        let mut buf = [0u8; 16];
        encode_min(&sample(1), &mut buf).unwrap();
    }

    #[test]
    fn roundtrip_1e6() {
        let mut buf = [0u8; 16];
        for i in 0..1_000_000u32 {
            let b = sample(i);
            encode_min(&b, &mut buf).unwrap();
            let d = decode_min(&buf, 64).unwrap();
            assert_eq!(b, d, "i={i}");
        }
    }

    #[test]
    fn home_invariant_enforced() {
        let mut b = sample(7);
        b.slot = b.slot.wrapping_add(1) % 64;
        let mut buf = [0u8; 16];
        encode_min(&b, &mut buf).unwrap();
        assert!(matches!(decode_min(&buf, 64), Err(WireError::SlotHomeViolation { .. })));
    }

    #[test]
    fn commit_with_reason_rejected() {
        let mut b = sample(0);
        assert_eq!(b.verdict, Verdict::Commit);
        b.reason = Reason::Nonfinite;
        let mut buf = [0u8; 16];
        encode_min(&b, &mut buf).unwrap();
        assert!(matches!(decode_min(&buf, 64), Err(WireError::VerdictReasonMismatch { .. })));
    }

    #[test]
    fn abort_without_reason_rejected() {
        let mut b = sample(2);
        assert_eq!(b.verdict, Verdict::Abort);
        b.reason = Reason::None;
        let mut buf = [0u8; 16];
        encode_min(&b, &mut buf).unwrap();
        assert!(matches!(decode_min(&buf, 64), Err(WireError::VerdictReasonMismatch { .. })));
    }

    #[test]
    fn t15_size_is_80() {
        assert_eq!(core::mem::size_of::<[u8; 32]>() * 2 + BINDING_MIN_SIZE, BINDING_T15_SIZE);
    }
}
