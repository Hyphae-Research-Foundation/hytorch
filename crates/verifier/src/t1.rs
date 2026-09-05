//! T1: same-binary replay of an audited microbatch. Spec v2.2 §09.
//!
//! h' = apply_ref(h₀, committed bindings in total order, C_step)
//! then per layer: head_{ℓ+1} = H(head_ℓ ‖ wire_ℓ ‖ meta_ℓ)  (T2 recompute)
//! Verdict: recomputed final head == expected head from receipts, AND the
//! final residual hash matches if the spill carries one (phase 1: head only).
//!
//! Any bit of difference kills the run (spec §6.2 of addendum v2.1).

use apply_ref::{apply_bindings, Binding};
use binding_wire::{layer_head, Verdict, GENESIS_HEAD};
use sha2::{Digest, Sha256};

use crate::spill::Spill;

#[derive(Debug, PartialEq, Eq)]
pub enum T1Outcome {
    /// Everything replayed; heads match.
    Consistent { n_applied: u64, final_head: [u8; 32] },
    /// Head chain diverged at this layer (first bad link).
    HeadMismatch { first_bad_layer: u16, got: [u8; 32], want_final: [u8; 32] },
    /// The T2 chain is intact but the REPLAYED RESIDUAL differs: the journal
    /// is internally consistent yet apply/journal/codebook do not close
    /// (e.g. tampered codebook, wrong C_step). Spill v2 catches this.
    ResidualMismatch { got: [u8; 32], want: [u8; 32] },
    /// apply_ref rejected the journal (order violation, range error…).
    ApplyError { layer: u16, detail: String },
}

fn sha256_bits(bits: &[u16]) -> [u8; 32] {
    let mut h = Sha256::new();
    for &v in bits {
        h.update(v.to_le_bytes());
    }
    h.finalize().into()
}

pub fn t1_replay(spill: &Spill) -> T1Outcome {
    let mut h = spill.h0.clone();
    let mut head = GENESIS_HEAD;
    let mut n_applied: u64 = 0;

    for (meta, bindings, wire) in &spill.layers {
        // T2 link over persisted bytes (already elided).
        head = layer_head(&head, wire, meta);

        // Replay COMMITs only, in wire order (which must be the §04 total order).
        let commits: Vec<Binding> = bindings
            .iter()
            .filter(|b| b.verdict == Verdict::Commit)
            .map(|b| Binding {
                pos: b.pos,
                slot: b.slot,
                feature: b.feature,
                mag_bf16: b.mag_bf16,
            })
            .collect();
        match apply_bindings(
            &mut h,
            spill.n_tokens,
            spill.s_slots,
            spill.d_slot,
            &spill.codebook,
            spill.n_features,
            &commits,
        ) {
            Ok(w) => n_applied += w,
            Err(e) => {
                return T1Outcome::ApplyError { layer: meta.layer, detail: format!("{e:?}") }
            }
        }
    }

    if head != spill.expected_final_head {
        // Localize: re-walk to find first divergence is impossible without the
        // expected per-layer heads; phase 1 receipts commit the final head, so
        // we report the final mismatch. T1.5 (H(pre_leaf)) narrows further.
        return T1Outcome::HeadMismatch {
            first_bad_layer: spill.n_layers.saturating_sub(1),
            got: head,
            want_final: spill.expected_final_head,
        };
    }

    // v2: the replayed residual must hash to what the device produced.
    // This closes apply ↔ journal ↔ codebook (tampered C_step, wrong step's
    // codebook, or an apply that diverged silently all land here).
    let got_h = sha256_bits(&h);
    if got_h != spill.expected_h_final_sha256 {
        return T1Outcome::ResidualMismatch {
            got: got_h,
            want: spill.expected_h_final_sha256,
        };
    }
    T1Outcome::Consistent { n_applied, final_head: head }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spill::{read_spill, write_spill, Spill};
    use binding_wire::{encode_min, BindingMin, LayerMeta, Reason, BINDING_MIN_SIZE};

    const S: u8 = 4;
    const D: u32 = 3;
    const NF: u16 = 16;
    const NT: u32 = 8;

    fn commit(feature: u16, pos: u32, mag: u16, layer: u16) -> BindingMin {
        BindingMin {
            feature,
            slot: (feature % S as u16) as u8,
            device: 0,
            mag_bf16: mag,
            layer,
            pos,
            cand: 0,
            verdict: Verdict::Commit,
            reason: Reason::None,
        }
    }

    /// Build an honest spill by simulating what the device would do with
    /// apply_ref itself (same-binary in the truest sense).
    fn honest_spill() -> Spill {
        let h0: Vec<u16> = (0..NT as usize * S as usize * D as usize)
            .map(|i| match i % 4 {
                0 => 0x8000, // -0.0: the D1 landmine, on purpose
                1 => 0x3F80,
                2 => 0xBE80,
                _ => 0x0000,
            })
            .collect();
        let codebook: Vec<u16> = (0..NF as usize * D as usize)
            .map(|i| if i % 2 == 0 { 0x3F80 } else { 0x3F00 })
            .collect();

        let per_layer: Vec<Vec<BindingMin>> = (0..3u16)
            .map(|l| {
                vec![
                    commit(4 + l, 0, 0x3EC0, l),
                    commit(9, 2, 0xBF00, l),
                    commit(14, 5, 0x3F20, l),
                ]
            })
            .collect();

        // Simulate: replay with apply_ref to get the true h_final and heads.
        let mut h = h0.clone();
        let mut head = GENESIS_HEAD;
        let mut layers = Vec::new();
        for (l, bs) in per_layer.iter().enumerate() {
            let mut wire = vec![0u8; bs.len() * BINDING_MIN_SIZE];
            for (i, b) in bs.iter().enumerate() {
                encode_min(b, &mut wire[i * BINDING_MIN_SIZE..]).unwrap();
            }
            let meta = LayerMeta {
                step_id: 1,
                layer: l as u16,
                microbatch_id: 0,
                n_persisted: bs.len() as u32,
                n_elided: 5,
                policy_id: 1,
            };
            head = layer_head(&head, &wire, &meta);
            let commits: Vec<apply_ref::Binding> = bs
                .iter()
                .map(|b| apply_ref::Binding {
                    pos: b.pos,
                    slot: b.slot,
                    feature: b.feature,
                    mag_bf16: b.mag_bf16,
                })
                .collect();
            apply_bindings(&mut h, NT, S, D, &codebook, NF, &commits).unwrap();
            layers.push((meta, bs.clone(), wire));
        }

        Spill {
            step_id: 1,
            microbatch_id: 0,
            n_tokens: NT,
            s_slots: S,
            d_slot: D,
            n_layers: 3,
            n_features: NF,
            expected_final_head: head,
            expected_h_final_sha256: sha256_bits(&h),
            policy_id: 1,
            h0,
            codebook,
            layers,
        }
    }

    #[test]
    fn honest_run_is_consistent() {
        let s = honest_spill();
        match t1_replay(&s) {
            T1Outcome::Consistent { n_applied, .. } => assert_eq!(n_applied, 9),
            other => panic!("expected consistent, got {other:?}"),
        }
    }

    #[test]
    fn spill_roundtrip_preserves_verdict() {
        let s = honest_spill();
        let bytes = write_spill(&s);
        let s2 = read_spill(&bytes).unwrap();
        assert_eq!(t1_replay(&s2), t1_replay(&s));
    }

    #[test]
    fn one_bit_flip_in_wal_is_caught() {
        let s = honest_spill();
        let mut bytes = write_spill(&s);
        // Flip one bit inside the layer-1 wire region (after header+h0+codebook).
        let header = 112usize; // spill v2 header size
        let h0_len = s.h0.len() * 2;
        let c_len = s.codebook.len() * 2;
        let layer0 = header + h0_len + c_len + 8; // skip counts of layer 0
        let target = layer0 + 5; // inside first binding of layer 0
        bytes[target] ^= 0x10;
        match read_spill(&bytes) {
            // Either the wire decode rejects it (invariant violation)...
            Err(_) => {}
            // ...or T1 catches the head mismatch. Both are captures.
            Ok(s2) => match t1_replay(&s2) {
                T1Outcome::Consistent { .. } => panic!("tampered WAL replayed clean"),
                _ => {}
            },
        }
    }

    #[test]
    fn adulterated_codebook_is_caught_by_residual_hash() {
        // Spill v2 closed the v1 gap: the T2 head covers only the wire, so a
        // tampered codebook used to replay to a DIFFERENT residual silently.
        // Now expected_h_final_sha256 catches it as ResidualMismatch.
        let mut tampered = honest_spill();
        // Feature 9 is committed in every layer; its row is [9*D .. 10*D).
        tampered.codebook[9 * D as usize] ^= 0x0040;
        match t1_replay(&tampered) {
            T1Outcome::ResidualMismatch { .. } => {}
            other => panic!("tampered codebook must be ResidualMismatch, got {other:?}"),
        }
        // And the honest one still passes, of course.
        assert!(matches!(t1_replay(&honest_spill()), T1Outcome::Consistent { .. }));
    }

}
