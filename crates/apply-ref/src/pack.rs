//! pack + allocate reference. Spec v2.2 §03 (fase 1, cerrada).
//!
//! This module is part of the pinned reference object (`build.apply_ref_hash`
//! covers it): the policy that turns `delta_hat` into verdicts is not a kernel
//! detail, it is the POLICY object executed bit-identically on host and device.
//!
//! Phase-1 parameters (from the POLICY record / manifest):
//!   home σ(f) = f mod S · score = ⟨leaf_s, Ĉ[f]⟩ · reduction seq fp32 no-FMA
//!   top-k global per token over the N_f legal pairs
//!   tiebreak: score desc → feature asc (slot is determined by feature)
//!   collision: winner by |mag| (abs bf16 bits, integer compare) desc,
//!              tie feature asc; losers are OVERFLOW
//!   abort rules on the bf16 bits of mag: nonfinite → ABORT, |mag| > mag_max
//!              → ABORT. mag == ±0 does NOT abort (C8).
//!
//! Determinism notes (both sides implement EXACTLY this):
//! - Score ordering uses the IEEE-754 total order on fp32 bits (monotone key
//!   trick). NaN scores sort ABOVE +inf: a NaN delta_hat gets selected and
//!   then ABORTed as nonfinite — insanity surfaces as journalized facts.
//! - Collision |mag| uses integer compare of (bits & 0x7FFF): monotone for
//!   finite bf16, NaN above inf. No float-compare pitfalls.
//! - Canonical output order per token: slot asc; within slot the winner
//!   (COMMIT/ABORT) first, then OVERFLOW losers feature asc. Extracting the
//!   COMMITs in this order yields the §04 total application order.

use crate::bits::{bf16_is_nonfinite, bf16_to_fp32, fp32_to_bf16_rne};
use crate::apply::normalize_row;

/// Verdict codes, wire-compatible with binding-wire (D4).
pub const VERDICT_COMMIT: u8 = 0;
pub const VERDICT_OVERFLOW: u8 = 1;
pub const VERDICT_ABORT: u8 = 2;

pub const REASON_NONE: u8 = 0;
pub const REASON_NONFINITE: u8 = 1;
pub const REASON_MAG_OVERFLOW: u8 = 2;

/// One pack/allocate verdict. `layer` and `device` are added by the caller
/// when building the wire BindingMin (they are frame context, not policy).
///
/// Exactly 16 bytes, NO hidden padding: the differential harness compares
/// these structs with memcmp, so every byte must be defined. (The original
/// `_pad: u8` left 2 bytes of compiler padding that compared by luck.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct VerdictRec {
    pub pos: u32,      // 0..4
    pub feature: u16,  // 4..6
    pub mag_bf16: u16, // 6..8
    pub cand: u16,     // 8..10
    pub slot: u8,      // 10
    pub verdict: u8,   // 11
    pub reason: u8,    // 12
    pub _pad: [u8; 3], // 13..16 — explicit, always zero
}

const _: () = assert!(core::mem::size_of::<VerdictRec>() == 16);
const _: () = assert!(core::mem::align_of::<VerdictRec>() == 4);

/// Candidate selection policy (spec §03; slot_topk is the phase-2 POLICY
/// lever named against slot concentration — "top-k por slot = fase posterior,
/// cambio de política, hecho nuevo").
pub const SELECT_GLOBAL_TOPK: u8 = 0;
pub const SELECT_SLOT_TOPK: u8 = 1;

#[derive(Debug, Clone, Copy)]
pub struct PackParams {
    pub s_slots: u8,
    pub d_slot: u32,
    pub n_features: u16,
    pub k: u16,
    /// POLICY mag_max, fp32. Judged on the *promoted bf16* |mag|.
    pub mag_max: f32,
    /// SELECT_GLOBAL_TOPK: top-k over all N_f legal pairs, collisions become
    ///   OVERFLOW facts (phase-1 policy).
    /// SELECT_SLOT_TOPK: winner per slot first (intra-slot losers are
    ///   SILENCE — never candidates, spec law 3), then top-k among the S
    ///   slot winners ranked by score. #C+#A = k, OVERFLOW = 0 structurally:
    ///   this policy admits no collision, so none can be journalized.
    pub selection: u8,
}

/// Monotone key for IEEE-754 total order on f32 bits (ascending).
#[inline(always)]
fn total_order_key(bits: u32) -> u32 {
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

/// pack + allocate for one microbatch of `delta_hat` (bf16 bits, layout
/// `[n_tokens][s_slots][d_slot]`). Returns verdicts in canonical order.
/// Exactly k verdicts per token (spec §07: #C + #O + #A = k).
pub fn pack_allocate(
    delta_hat: &[u16],
    n_tokens: u32,
    codebook: &[u16],
    p: PackParams,
) -> Vec<VerdictRec> {
    let ds = p.d_slot as usize;
    let ss = p.s_slots as usize;
    let nf = p.n_features as usize;
    let k = p.k as usize;
    assert!(k >= 1 && k <= 8, "phase 1: k in 1..=8");
    assert!(nf >= k, "manifest prohibits N_f < k");
    assert_eq!(delta_hat.len(), n_tokens as usize * ss * ds);
    assert_eq!(codebook.len(), nf * ds);

    // Ĉ: normalize every row once (sequential reduction inside normalize_row).
    let mut nhat = vec![0.0f32; nf * ds];
    for f in 0..nf {
        normalize_row(&codebook[f * ds..(f + 1) * ds], &mut nhat[f * ds..(f + 1) * ds]);
    }

    if p.selection == SELECT_SLOT_TOPK {
        assert!(k <= ss, "slot_topk: k slots max, one winner each");
    } else {
        assert_eq!(p.selection, SELECT_GLOBAL_TOPK, "unknown selection policy");
    }

    let mut out = Vec::with_capacity(n_tokens as usize * k);
    let mut leaf = vec![0.0f32; ss * ds];
    // (total_order_key, feature) of the selected top-k, selection rank = index.
    let mut selected: Vec<(u32, u16, u32)> = Vec::with_capacity(k); // (key, feature, score_bits)
    // slot_topk scratch: per-slot best (key, feature, score_bits).
    let mut slot_best: Vec<(u32, u16, u32)> = vec![(0, 0, 0); ss];

    for pos in 0..n_tokens {
        // Promote the token's leaves once.
        let base = pos as usize * ss * ds;
        for (i, l) in leaf.iter_mut().enumerate() {
            *l = bf16_to_fp32(delta_hat[base + i]);
        }

        selected.clear();
        if p.selection == SELECT_SLOT_TOPK {
            // Pass 1: winner per slot (score desc, feature asc). Ascending f
            // scan with strict > keeps the earlier feature on ties. Intra-slot
            // losers are SILENCE: they were never candidates (law 3).
            for sb in slot_best.iter_mut() {
                *sb = (0, 0, 0); // key 0 = most-negative under total order; f scan overwrites
            }
            let mut slot_init = vec![false; ss];
            for f in 0..nf {
                let s = f % ss;
                let row = &nhat[f * ds..(f + 1) * ds];
                let lf = &leaf[s * ds..(s + 1) * ds];
                let mut acc = 0.0f32;
                for j in 0..ds {
                    acc += lf[j] * row[j];
                }
                let score_bits =
                    if acc.is_nan() { 0x7FC0_0000u32 } else { acc.to_bits() };
                let key = total_order_key(score_bits);
                if !slot_init[s] || key > slot_best[s].0 {
                    slot_best[s] = (key, f as u16, score_bits);
                    slot_init[s] = true;
                }
            }
            // Pass 2: top-k among the S slot winners (key desc, feature asc —
            // ascending slot scan + strict > keeps earlier feature/slot).
            for s in 0..ss {
                if !slot_init[s] {
                    continue;
                }
                let (key, f, sb) = slot_best[s];
                if selected.len() < k {
                    let ins = selected
                        .iter()
                        .position(|&(ek, _, _)| key > ek)
                        .unwrap_or(selected.len());
                    selected.insert(ins, (key, f, sb));
                } else if key > selected[k - 1].0 {
                    selected.pop();
                    let ins = selected
                        .iter()
                        .position(|&(ek, _, _)| key > ek)
                        .unwrap_or(selected.len());
                    selected.insert(ins, (key, f, sb));
                }
            }
        } else {
            // Global top-k over all N_f legal pairs (score desc, feature asc).
            for f in 0..nf {
                let s = f % ss;
                let row = &nhat[f * ds..(f + 1) * ds];
                let lf = &leaf[s * ds..(s + 1) * ds];
                let mut acc = 0.0f32;
                for j in 0..ds {
                    acc += lf[j] * row[j]; // two rounded ops, sequential, no FMA
                }
                // SPEC_AMEND-002: canonicalize NaN scores to +qNaN (0x7FC00000)
                // BEFORE ordering and mag derivation. NaN payloads/signs differ
                // between host (x86 emits -qNaN for inf*0) and device (+qNaN);
                // without this, the same delta_hat selects different features.
                // Found by the M3 differential gate on H100 (BACKEND_REJECTED).
                let score_bits =
                    if acc.is_nan() { 0x7FC0_0000u32 } else { acc.to_bits() };
                let key = total_order_key(score_bits);
                // Insert if better than current worst. Ascending f scan + strict
                // comparison ⇒ equal scores prefer the earlier (lower) feature.
                if selected.len() < k {
                    let ins = selected
                        .iter()
                        .position(|&(ek, _, _)| key > ek)
                        .unwrap_or(selected.len());
                    selected.insert(ins, (key, f as u16, score_bits));
                } else if key > selected[k - 1].0 {
                    selected.pop();
                    let ins = selected
                        .iter()
                        .position(|&(ek, _, _)| key > ek)
                        .unwrap_or(selected.len());
                    selected.insert(ins, (key, f as u16, score_bits));
                }
            }
        }

        // Winner per slot by |mag| bf16 abs-bits desc, tie feature asc.
        // mag is the score downcast to bf16 — the bits apply will multiply.
        // cand = selection rank.
        #[derive(Clone, Copy)]
        struct Cand {
            feature: u16,
            slot: u8,
            mag: u16,
            cand: u16,
        }
        let cands: Vec<Cand> = selected
            .iter()
            .enumerate()
            .map(|(rank, &(_, f, score_bits))| Cand {
                feature: f,
                slot: (f as usize % ss) as u8,
                mag: fp32_to_bf16_rne(f32::from_bits(score_bits)),
                cand: rank as u16,
            })
            .collect();

        // Canonical emission: slot asc; winner first, then losers feature asc.
        let mut slots_present: Vec<u8> = cands.iter().map(|c| c.slot).collect();
        slots_present.sort_unstable();
        slots_present.dedup();
        for &s in &slots_present {
            let mut group: Vec<Cand> =
                cands.iter().copied().filter(|c| c.slot == s).collect();
            // Winner: abs-bits desc, tie feature asc.
            group.sort_by(|a, b| {
                let ka = a.mag & 0x7FFF;
                let kb = b.mag & 0x7FFF;
                kb.cmp(&ka).then(a.feature.cmp(&b.feature))
            });
            let winner = group[0];
            let (verdict, reason) = if bf16_is_nonfinite(winner.mag) {
                (VERDICT_ABORT, REASON_NONFINITE)
            } else if bf16_to_fp32(winner.mag).abs() > p.mag_max {
                (VERDICT_ABORT, REASON_MAG_OVERFLOW)
            } else {
                (VERDICT_COMMIT, REASON_NONE) // ±0 commits (C8)
            };
            out.push(VerdictRec {
                pos,
                feature: winner.feature,
                mag_bf16: winner.mag,
                cand: winner.cand,
                slot: s,
                verdict,
                reason,
                _pad: [0; 3],
            });
            // Losers: OVERFLOW, feature asc.
            let mut losers: Vec<Cand> = group[1..].to_vec();
            losers.sort_by_key(|c| c.feature);
            for l in losers {
                out.push(VerdictRec {
                    pos,
                    feature: l.feature,
                    mag_bf16: l.mag,
                    cand: l.cand,
                    slot: s,
                    verdict: VERDICT_OVERFLOW,
                    reason: REASON_NONE,
                    _pad: [0; 3],
                });
            }
        }
    }
    out
}

/// pack + allocate over a device-proposed candidate set (SPEC_AMEND-004,
/// policy 7 "two_phase_topk"). The device (TensorE/XLA/anything fast)
/// proposes up to M candidate features per token; THIS function recomputes
/// their scores with the pinned math (sequential fp32 no-FMA over the
/// normalized codebook row, NaN canonicalized per SPEC_AMEND-002) and runs
/// the EXACT policy-6 machinery over that set: global top-k by
/// (total-order key desc, feature asc), collision winner per slot by
/// |mag| bits desc / feature asc, abort rules, canonical emission order.
///
/// Facts remain bit-exactly verifiable and replayable: mag IS the exact
/// score of the cited feature (T1 recompute unchanged), apply unchanged,
/// T2 unchanged. What is NOT claimed: that no feature outside the proposed
/// set had a higher score — that property is DECLARED as device-proposed
/// in the POLICY record and measured as a miss-rate, never assumed.
///
/// `candidates`: `[n_tokens][m]` feature ids; entries >= n_features are
/// padding (ignored). Duplicates allowed (deduped; first occurrence keeps
/// the proposal rank). The equivalence bridge: candidates = 0..N_f for
/// every token makes this IDENTICAL to policy-6 global_topk (gated test).
///
/// `cand` in the emitted VerdictRec = rank of the feature in the PROPOSAL
/// (0..m-1): every fact carries its provenance in the device's ranking.
pub fn pack_allocate_candidates(
    delta_hat: &[u16],
    n_tokens: u32,
    codebook: &[u16],
    candidates: &[u16],
    m: u32,
    p: PackParams,
) -> Vec<VerdictRec> {
    let ds = p.d_slot as usize;
    let ss = p.s_slots as usize;
    let nf = p.n_features as usize;
    let k = p.k as usize;
    let mu = m as usize;
    assert!(k >= 1 && k <= 8, "k in 1..=8");
    assert!(nf >= k, "manifest prohibits N_f < k");
    assert_eq!(delta_hat.len(), n_tokens as usize * ss * ds);
    assert_eq!(codebook.len(), nf * ds);
    assert_eq!(candidates.len(), n_tokens as usize * mu);
    assert_eq!(
        p.selection, SELECT_GLOBAL_TOPK,
        "two-phase base policy is global_topk (policy 7 = 6 over candidates)"
    );

    // Ĉ rows for the features that can appear (normalize lazily, memoized:
    // candidate sets are tiny and repeat features across tokens).
    let mut nhat = vec![0.0f32; nf * ds];
    let mut have = vec![false; nf];

    let mut out = Vec::with_capacity(n_tokens as usize * k);
    let mut leaf = vec![0.0f32; ss * ds];
    let mut selected: Vec<(u32, u16, u32)> = Vec::with_capacity(k);
    let mut seen = vec![u32::MAX; nf]; // token tag for dedup

    for pos in 0..n_tokens {
        let base = pos as usize * ss * ds;
        for (i, l) in leaf.iter_mut().enumerate() {
            *l = bf16_to_fp32(delta_hat[base + i]);
        }

        selected.clear();
        let cbase = pos as usize * mu;
        for rank in 0..mu {
            let f = candidates[cbase + rank] as usize;
            if f >= nf || seen[f] == pos {
                continue; // padding or duplicate proposal
            }
            seen[f] = pos;
            if !have[f] {
                normalize_row(&codebook[f * ds..(f + 1) * ds], &mut nhat[f * ds..(f + 1) * ds]);
                have[f] = true;
            }
            let s = f % ss;
            let row = &nhat[f * ds..(f + 1) * ds];
            let lf = &leaf[s * ds..(s + 1) * ds];
            let mut acc = 0.0f32;
            for j in 0..ds {
                acc += lf[j] * row[j]; // sequential, two rounded ops, no FMA
            }
            let score_bits = if acc.is_nan() { 0x7FC0_0000u32 } else { acc.to_bits() };
            let key = total_order_key(score_bits);
            // (key desc, feature asc): strict > + ascending-rank scan is NOT
            // feature-ascending here (proposal order is arbitrary), so ties
            // must compare features explicitly.
            let better = |ek: u32, ef: u16| key > ek || (key == ek && (f as u16) < ef);
            if selected.len() < k {
                let ins = selected
                    .iter()
                    .position(|&(ek, ef, _)| better(ek, ef))
                    .unwrap_or(selected.len());
                selected.insert(ins, (key, f as u16, score_bits));
            } else {
                let (wk, wf, _) = selected[k - 1];
                if better(wk, wf) {
                    selected.pop();
                    let ins = selected
                        .iter()
                        .position(|&(ek, ef, _)| better(ek, ef))
                        .unwrap_or(selected.len());
                    selected.insert(ins, (key, f as u16, score_bits));
                }
            }
        }

        // From here: EXACTLY the policy-6 emission machinery, except `cand`
        // records the PROPOSAL rank (provenance) instead of selection rank.
        #[derive(Clone, Copy)]
        struct Cand {
            feature: u16,
            slot: u8,
            mag: u16,
            cand: u16,
        }
        let cands: Vec<Cand> = selected
            .iter()
            .map(|&(_, f, score_bits)| {
                // proposal rank = first occurrence in the candidate list
                let rank = (0..mu)
                    .position(|r| candidates[cbase + r] == f)
                    .unwrap_or(0) as u16;
                Cand {
                    feature: f,
                    slot: (f as usize % ss) as u8,
                    mag: fp32_to_bf16_rne(f32::from_bits(score_bits)),
                    cand: rank,
                }
            })
            .collect();

        let mut slots_present: Vec<u8> = cands.iter().map(|c| c.slot).collect();
        slots_present.sort_unstable();
        slots_present.dedup();
        for &s in &slots_present {
            let mut group: Vec<Cand> =
                cands.iter().copied().filter(|c| c.slot == s).collect();
            group.sort_by(|a, b| {
                let ka = a.mag & 0x7FFF;
                let kb = b.mag & 0x7FFF;
                kb.cmp(&ka).then(a.feature.cmp(&b.feature))
            });
            let winner = group[0];
            let (verdict, reason) = if bf16_is_nonfinite(winner.mag) {
                (VERDICT_ABORT, REASON_NONFINITE)
            } else if bf16_to_fp32(winner.mag).abs() > p.mag_max {
                (VERDICT_ABORT, REASON_MAG_OVERFLOW)
            } else {
                (VERDICT_COMMIT, REASON_NONE)
            };
            out.push(VerdictRec {
                pos, feature: winner.feature, mag_bf16: winner.mag,
                cand: winner.cand, slot: s, verdict, reason, _pad: [0; 3],
            });
            let mut losers: Vec<Cand> = group[1..].to_vec();
            losers.sort_by_key(|c| c.feature);
            for l in losers {
                out.push(VerdictRec {
                    pos, feature: l.feature, mag_bf16: l.mag, cand: l.cand,
                    slot: s, verdict: VERDICT_OVERFLOW, reason: REASON_NONE,
                    _pad: [0; 3],
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: u8 = 4;
    const D: u32 = 3;
    const NF: u16 = 16;

    fn params(k: u16) -> PackParams {
        PackParams {
            s_slots: S, d_slot: D, n_features: NF, k, mag_max: 8.0,
            selection: SELECT_GLOBAL_TOPK,
        }
    }

    fn params_slot(k: u16) -> PackParams {
        PackParams {
            s_slots: S, d_slot: D, n_features: NF, k, mag_max: 8.0,
            selection: SELECT_SLOT_TOPK,
        }
    }

    fn codebook_identityish() -> Vec<u16> {
        // Rows with distinct directions; all finite.
        (0..NF as usize * D as usize)
            .map(|i| match i % 3 {
                0 => 0x3F80, // 1.0
                1 => 0x3F00, // 0.5
                _ => 0xBE80, // -0.25
            })
            .collect()
    }

    #[test]
    fn step0_dynamics_declared_d5a() {
        // delta_hat = 0 ⇒ all scores 0 ⇒ tiebreak selects features 0..k-1 for
        // EVERY token; slots 0..k-1; all COMMIT with mag ±0 (C8).
        let k = 4u16;
        let nt = 3u32;
        let dh = vec![0u16; nt as usize * S as usize * D as usize];
        let recs = pack_allocate(&dh, nt, &codebook_identityish(), params(k));
        assert_eq!(recs.len(), nt as usize * k as usize);
        for pos in 0..nt {
            let tok: Vec<_> = recs.iter().filter(|r| r.pos == pos).collect();
            let feats: Vec<u16> = tok.iter().map(|r| r.feature).collect();
            assert_eq!(feats, vec![0, 1, 2, 3], "same features for every token");
            assert!(tok.iter().all(|r| r.verdict == VERDICT_COMMIT));
            assert!(tok.iter().all(|r| r.mag_bf16 & 0x7FFF == 0), "mag ±0 commits");
        }
    }

    #[test]
    fn partition_is_exactly_k() {
        let k = 8u16;
        let nt = 5u32;
        let dh: Vec<u16> = (0..nt as usize * S as usize * D as usize)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 16) as u16 & 0x7FFF | 0x3000)
            .collect();
        let recs = pack_allocate(&dh, nt, &codebook_identityish(), params(k));
        for pos in 0..nt {
            let n = recs.iter().filter(|r| r.pos == pos).count();
            assert_eq!(n, k as usize, "#C+#O+#A = k (spec §07)");
        }
    }

    #[test]
    fn collision_emits_overflow_with_winner_by_abs_mag() {
        // S=2: features 0,2,4,… share slot 0; 1,3,5,… share slot 1. k=4 over
        // NF=8 forces ≥2 candidates in some slot.
        let p = PackParams { s_slots: 2, d_slot: D, n_features: 8, k: 4, mag_max: 8.0, selection: SELECT_GLOBAL_TOPK };
        let nt = 1u32;
        let dh: Vec<u16> = vec![0x3F80, 0x3F00, 0xBE80, 0x3EC0, 0x3F40, 0xBF00];
        let cb: Vec<u16> = (0..8 * D as usize)
            .map(|i| if i % 2 == 0 { 0x3F80 } else { 0xBF00 })
            .collect();
        let recs = pack_allocate(&dh, nt, &cb, p);
        assert_eq!(recs.len(), 4);
        let overflows: Vec<_> =
            recs.iter().filter(|r| r.verdict == VERDICT_OVERFLOW).collect();
        assert!(!overflows.is_empty(), "collision must journalize OVERFLOW");
        // Per slot: winner |mag| >= every loser |mag| (abs-bits order).
        for s in 0..2u8 {
            let group: Vec<_> = recs.iter().filter(|r| r.slot == s).collect();
            if group.len() > 1 {
                let w = group.iter().find(|r| r.verdict != VERDICT_OVERFLOW).unwrap();
                for l in group.iter().filter(|r| r.verdict == VERDICT_OVERFLOW) {
                    assert!(
                        (w.mag_bf16 & 0x7FFF) >= (l.mag_bf16 & 0x7FFF),
                        "winner by abs-bits"
                    );
                }
            }
        }
    }

    #[test]
    fn nan_delta_hat_becomes_journalized_abort() {
        let mut dh = vec![0u16; S as usize * D as usize];
        dh[0] = 0x7FC0; // NaN leaf component in slot 0
        let recs = pack_allocate(&dh, 1, &codebook_identityish(), params(2));
        // NaN score sorts above +inf ⇒ selected ⇒ mag NaN ⇒ ABORT nonfinite.
        let aborts: Vec<_> = recs.iter().filter(|r| r.verdict == VERDICT_ABORT).collect();
        assert!(!aborts.is_empty());
        assert!(aborts.iter().all(|r| r.reason == REASON_NONFINITE));
    }

    #[test]
    fn mag_overflow_aborts() {
        // Big leaf ⇒ |score| > mag_max ⇒ ABORT mag_overflow.
        let p = PackParams { s_slots: S, d_slot: D, n_features: NF, k: 1, mag_max: 0.5, selection: SELECT_GLOBAL_TOPK };
        let dh: Vec<u16> = vec![0x4120; S as usize * D as usize]; // 10.0 everywhere
        let recs = pack_allocate(&dh, 1, &codebook_identityish(), p);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].verdict, VERDICT_ABORT);
        assert_eq!(recs[0].reason, REASON_MAG_OVERFLOW);
    }

    #[test]
    fn deterministic_bytes() {
        let dh: Vec<u16> = (0..2 * S as usize * D as usize)
            .map(|i| ((i * 7919) % 65536) as u16)
            .collect();
        let a = pack_allocate(&dh, 2, &codebook_identityish(), params(4));
        let b = pack_allocate(&dh, 2, &codebook_identityish(), params(4));
        assert_eq!(a, b);
    }

    #[test]
    fn commits_extracted_in_canonical_order_satisfy_apply_order() {
        let dh: Vec<u16> = (0..4 * S as usize * D as usize)
            .map(|i| (((i * 31) % 128) as u16) | 0x3C00)
            .collect();
        let recs = pack_allocate(&dh, 4, &codebook_identityish(), params(4));
        let commits: Vec<_> = recs.iter().filter(|r| r.verdict == VERDICT_COMMIT).collect();
        let mut last: Option<(u32, u8)> = None;
        for c in commits {
            let key = (c.pos, c.slot);
            if let Some(prev) = last {
                assert!(key > prev, "strictly increasing (pos, slot)");
            }
            last = Some(key);
        }
    }

    // ---- slot_topk (phase-2 POLICY) ----

    #[test]
    fn slot_topk_no_overflow_distinct_slots() {
        // Structural: one winner per slot, k distinct slots, zero OVERFLOW.
        let dh: Vec<u16> = (0..6 * S as usize * D as usize)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 17) as u16 & 0x3FFF | 0x3800)
            .collect();
        let recs = pack_allocate(&dh, 6, &codebook_identityish(), params_slot(3));
        for pos in 0..6u32 {
            let tok: Vec<_> = recs.iter().filter(|r| r.pos == pos).collect();
            assert_eq!(tok.len(), 3, "#C+#A = k, OVERFLOW structurally 0");
            assert!(tok.iter().all(|r| r.verdict != VERDICT_OVERFLOW));
            let mut slots: Vec<u8> = tok.iter().map(|r| r.slot).collect();
            slots.dedup();
            assert_eq!(slots.len(), 3, "k distinct slots");
            // Winner cited per slot must have the slot-max |score|: verify by
            // recomputation over all features homed to that slot.
        }
    }

    #[test]
    fn slot_topk_winner_is_slot_argmax() {
        let dh: Vec<u16> = (0..2 * S as usize * D as usize)
            .map(|i| (((i * 131) % 16384) as u16) | 0x3400)
            .collect();
        let recs = pack_allocate(&dh, 2, &codebook_identityish(), params_slot(4));
        // Reference scores via normalize_row + sequential dot (same as impl).
        let ds = D as usize;
        let nf = NF as usize;
        let cb = codebook_identityish();
        let mut nhat = vec![0.0f32; nf * ds];
        for f in 0..nf {
            normalize_row(&cb[f * ds..(f + 1) * ds], &mut nhat[f * ds..(f + 1) * ds]);
        }
        for r in recs.iter().filter(|r| r.pos == 0) {
            let s = r.slot as usize;
            let leaf: Vec<f32> = (0..ds)
                .map(|j| bf16_to_fp32(dh[s * ds + j]))
                .collect();
            let mut best_key = 0u32;
            let mut best_f = 0u16;
            let mut init = false;
            for f in (0..nf).filter(|f| f % S as usize == s) {
                let mut acc = 0.0f32;
                for j in 0..ds {
                    acc += leaf[j] * nhat[f * ds + j];
                }
                let sb = if acc.is_nan() { 0x7FC0_0000 } else { acc.to_bits() };
                let key = total_order_key(sb);
                if !init || key > best_key {
                    best_key = key;
                    best_f = f as u16;
                    init = true;
                }
            }
            assert_eq!(r.feature, best_f, "slot {s}: cited winner is the argmax");
        }
    }

    #[test]
    fn slot_topk_step0_dynamics() {
        // All-zero delta_hat: every slot winner is feature s (lowest homed),
        // slot ranking ties → slots 0..k-1 win (ascending scan keeps earlier).
        let k = 3u16;
        let dh = vec![0u16; 2 * S as usize * D as usize];
        let recs = pack_allocate(&dh, 2, &codebook_identityish(), params_slot(k));
        for pos in 0..2u32 {
            let tok: Vec<_> = recs.iter().filter(|r| r.pos == pos).collect();
            let slots: Vec<u8> = tok.iter().map(|r| r.slot).collect();
            assert_eq!(slots, vec![0, 1, 2]);
            for r in &tok {
                assert_eq!(r.feature, r.slot as u16, "lowest feature homed to slot");
                assert_eq!(r.verdict, VERDICT_COMMIT);
            }
        }
    }

    // ---- two-phase candidates (SPEC_AMEND-004, policy 7) ----

    fn all_features_cands(nt: u32) -> Vec<u16> {
        (0..nt).flat_map(|_| 0..NF).collect()
    }

    #[test]
    fn two_phase_full_candidates_equals_policy6() {
        // The formal bridge: candidates = ALL features (natural order) must
        // reproduce policy-6 global_topk BIT-identically, byte for byte
        // (except `cand`, which is proposal rank == selection scan order —
        // compare all other fields).
        let nt = 7u32;
        let dh: Vec<u16> = (0..nt as usize * S as usize * D as usize)
            .map(|i| ((i as u32).wrapping_mul(2654435761) >> 16) as u16)
            .collect();
        let cb = codebook_identityish();
        for k in [1u16, 3, 8] {
            let a = pack_allocate(&dh, nt, &cb, params(k));
            let b = pack_allocate_candidates(
                &dh, nt, &cb, &all_features_cands(nt), NF as u32, params(k));
            assert_eq!(a.len(), b.len(), "k={k}");
            for (x, y) in a.iter().zip(b.iter()) {
                assert_eq!(
                    (x.pos, x.feature, x.mag_bf16, x.slot, x.verdict, x.reason),
                    (y.pos, y.feature, y.mag_bf16, y.slot, y.verdict, y.reason),
                    "k={k}"
                );
            }
        }
    }

    #[test]
    fn two_phase_restricted_candidates_select_within_set() {
        // Candidates limited to features {1, 5, 9, 13}: every verdict must
        // cite one of them, k verdicts per token, exact scores (spot-check
        // via mag == bf16(dot(leaf_s, nhat_f))).
        let nt = 3u32;
        let dh: Vec<u16> = (0..nt as usize * S as usize * D as usize)
            .map(|i| (((i * 131) % 16000) as u16) | 0x3400)
            .collect();
        let cb = codebook_identityish();
        let cands: Vec<u16> = (0..nt).flat_map(|_| [1u16, 5, 9, 13]).collect();
        let recs = pack_allocate_candidates(&dh, nt, &cb, &cands, 4, params(4));
        assert_eq!(recs.len(), nt as usize * 4);
        for r in &recs {
            assert!([1, 5, 9, 13].contains(&r.feature));
            // proposal rank provenance
            assert_eq!([1u16, 5, 9, 13][r.cand as usize], r.feature);
        }
    }

    #[test]
    fn two_phase_padding_and_duplicates_ignored() {
        // Padding (>= NF) and duplicate proposals must not distort verdicts:
        // effective set {2} with k=1 → one verdict per token citing feature 2.
        let nt = 2u32;
        let dh: Vec<u16> = (0..nt as usize * S as usize * D as usize)
            .map(|i| ((i * 7919) % 32768) as u16)
            .collect();
        let cb = codebook_identityish();
        let cands: Vec<u16> = (0..nt).flat_map(|_| [2u16, 2, 0xFFFF, 2]).collect();
        let recs = pack_allocate_candidates(&dh, nt, &cb, &cands, 4, params(1));
        assert_eq!(recs.len(), nt as usize);
        for r in &recs {
            assert_eq!(r.feature, 2);
            assert_eq!(r.cand, 0, "first occurrence keeps the proposal rank");
        }
    }

    #[test]
    fn two_phase_tie_prefers_lower_feature_regardless_of_proposal_order() {
        // Zero delta_hat → all scores exactly 0.0 → ties everywhere. With
        // proposal order [9, 1, 5] the selection must still be feature-asc:
        // k=2 selects features 1 and 5, NOT 9.
        let dh = vec![0u16; S as usize * D as usize];
        let cb = codebook_identityish();
        let recs = pack_allocate_candidates(&dh, 1, &cb, &[9, 1, 5], 3, params(2));
        let mut feats: Vec<u16> = recs.iter().map(|r| r.feature).collect();
        feats.sort_unstable();
        assert_eq!(feats, vec![1, 5]);
    }

    #[test]
    fn slot_topk_deterministic_and_abort_rules_apply() {
        let dh: Vec<u16> = (0..3 * S as usize * D as usize)
            .map(|i| ((i * 97) % 65536) as u16)
            .collect();
        let a = pack_allocate(&dh, 3, &codebook_identityish(), params_slot(4));
        let b = pack_allocate(&dh, 3, &codebook_identityish(), params_slot(4));
        assert_eq!(a, b);
        // mag_max tiny ⇒ winners abort, still k verdicts per token.
        let mut p = params_slot(2);
        p.mag_max = 1e-6;
        let dh_big: Vec<u16> = vec![0x4120; 1 * S as usize * D as usize]; // 10.0
        let recs = pack_allocate(&dh_big, 1, &codebook_identityish(), p);
        assert_eq!(recs.len(), 2);
        assert!(recs.iter().all(|r| r.verdict == VERDICT_ABORT));
    }
}
