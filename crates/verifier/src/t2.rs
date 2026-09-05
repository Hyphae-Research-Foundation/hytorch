//! T2 walk: recompute the chain over persisted layers and localize the first
//! divergent link against a list of expected heads (when receipts carry them)
//! or against a final head. Spec v2.2 §09.

use binding_wire::{layer_head, LayerMeta, GENESIS_HEAD};

#[derive(Debug, PartialEq, Eq)]
pub enum T2Outcome {
    Intact { final_head: [u8; 32] },
    /// First link whose recomputed head differs from the expected one.
    Broken { first_bad_layer: u16, got: [u8; 32], want: [u8; 32] },
}

/// Walk with per-layer expected heads (full receipt mode).
pub fn t2_walk(
    layers: &[(LayerMeta, Vec<u8>)],
    expected_heads: &[[u8; 32]],
) -> T2Outcome {
    let mut head = GENESIS_HEAD;
    for (i, (meta, wire)) in layers.iter().enumerate() {
        head = layer_head(&head, wire, meta);
        if let Some(want) = expected_heads.get(i) {
            if &head != want {
                return T2Outcome::Broken { first_bad_layer: meta.layer, got: head, want: *want };
            }
        }
    }
    T2Outcome::Intact { final_head: head }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(layer: u16) -> LayerMeta {
        LayerMeta {
            step_id: 9,
            layer,
            microbatch_id: 0,
            n_persisted: 1,
            n_elided: 0,
            policy_id: 1,
        }
    }

    #[test]
    fn walk_localizes_first_bad_link() {
        let layers: Vec<(LayerMeta, Vec<u8>)> =
            (0..5).map(|l| (meta(l), vec![l as u8; 16])).collect();
        let mut head = GENESIS_HEAD;
        let expected: Vec<[u8; 32]> = layers
            .iter()
            .map(|(m, w)| {
                head = layer_head(&head, w, m);
                head
            })
            .collect();

        // Untampered: intact.
        assert!(matches!(t2_walk(&layers, &expected), T2Outcome::Intact { .. }));

        // Tamper layer 3's wire: first bad link is layer 3.
        let mut bad = layers.clone();
        bad[3].1[0] ^= 0x80;
        match t2_walk(&bad, &expected) {
            T2Outcome::Broken { first_bad_layer, .. } => assert_eq!(first_bad_layer, 3),
            _ => panic!("tamper not detected"),
        }

        // Tamper layer 3's meta (n_elided lies): also layer 3.
        let mut bad_meta = layers.clone();
        bad_meta[3].0.n_elided = 999;
        match t2_walk(&bad_meta, &expected) {
            T2Outcome::Broken { first_bad_layer, .. } => assert_eq!(first_bad_layer, 3),
            _ => panic!("meta tamper not detected"),
        }
    }
}
