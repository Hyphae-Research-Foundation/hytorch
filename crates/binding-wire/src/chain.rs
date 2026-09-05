//! T2: the per-layer hash chain over PERSISTED facts. Spec v2.2 §09.
//!
//! head_{ℓ+1} = SHA-256( head_ℓ ‖ bindings_ℓ (wire bytes, in order) ‖ meta_ℓ )
//!
//! meta_ℓ includes layer, n_persistidos, n_elididos and policy_id so that a
//! step-0 layer with every zero elided is not an amorphous link (spec §09).
//! Elision happens BEFORE this hash (D1): the chain covers what the WAL holds.

use sha2::{Digest, Sha256};

pub const GENESIS_HEAD: [u8; 32] = [0u8; 32];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerMeta {
    pub step_id: u64,
    pub layer: u16,
    pub microbatch_id: u32,
    pub n_persisted: u32,
    pub n_elided: u32,
    pub policy_id: u64,
}

impl LayerMeta {
    pub fn to_bytes(&self) -> [u8; 30] {
        let mut out = [0u8; 30];
        out[0..8].copy_from_slice(&self.step_id.to_le_bytes());
        out[8..10].copy_from_slice(&self.layer.to_le_bytes());
        out[10..14].copy_from_slice(&self.microbatch_id.to_le_bytes());
        out[14..18].copy_from_slice(&self.n_persisted.to_le_bytes());
        out[18..22].copy_from_slice(&self.n_elided.to_le_bytes());
        out[22..30].copy_from_slice(&self.policy_id.to_le_bytes());
        out
    }
}

/// Compute head_{ℓ+1} from head_ℓ, the persisted wire bytes of the layer
/// (concatenated BindingMin/T15 records, already elided), and the meta.
pub fn layer_head(prev_head: &[u8; 32], wire_bytes: &[u8], meta: &LayerMeta) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(prev_head);
    h.update(wire_bytes);
    h.update(meta.to_bytes());
    h.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{encode_min, BindingMin, Reason, Verdict, BINDING_MIN_SIZE};

    fn meta(layer: u16, n_p: u32, n_e: u32) -> LayerMeta {
        LayerMeta {
            step_id: 42,
            layer,
            microbatch_id: 7,
            n_persisted: n_p,
            n_elided: n_e,
            policy_id: 1,
        }
    }

    fn wire(bindings: &[BindingMin]) -> Vec<u8> {
        let mut out = vec![0u8; bindings.len() * BINDING_MIN_SIZE];
        for (i, b) in bindings.iter().enumerate() {
            encode_min(b, &mut out[i * BINDING_MIN_SIZE..]).unwrap();
        }
        out
    }

    fn b(feature: u16, pos: u32, mag: u16) -> BindingMin {
        BindingMin {
            feature,
            slot: (feature % 64) as u8,
            device: 0,
            mag_bf16: mag,
            layer: 3,
            pos,
            cand: 0,
            verdict: Verdict::Commit,
            reason: Reason::None,
        }
    }

    #[test]
    fn single_bit_flip_changes_head() {
        let bytes = wire(&[b(100, 0, 0x3F80), b(200, 1, 0xBF00)]);
        let m = meta(3, 2, 0);
        let h0 = layer_head(&GENESIS_HEAD, &bytes, &m);
        for bit in 0..bytes.len() * 8 {
            let mut tampered = bytes.clone();
            tampered[bit / 8] ^= 1 << (bit % 8);
            let ht = layer_head(&GENESIS_HEAD, &tampered, &m);
            assert_ne!(h0, ht, "bit {bit} flip not detected");
        }
    }

    #[test]
    fn empty_layer_is_not_amorphous() {
        // Two empty layers with different elision counts have different heads:
        // meta carries n_elided (spec §09).
        let m1 = meta(0, 0, 8192);
        let m2 = meta(0, 0, 8191);
        assert_ne!(layer_head(&GENESIS_HEAD, &[], &m1), layer_head(&GENESIS_HEAD, &[], &m2));
    }

    #[test]
    fn chain_localizes_tamper_to_layer() {
        let layers: Vec<Vec<u8>> =
            (0..4).map(|l| wire(&[b(100 + l as u16, 0, 0x3F80)])).collect();
        let metas: Vec<LayerMeta> = (0..4).map(|l| meta(l as u16, 1, 0)).collect();

        let mut heads = vec![GENESIS_HEAD];
        for (w, m) in layers.iter().zip(metas.iter()) {
            let prev = *heads.last().unwrap();
            heads.push(layer_head(&prev, w, m));
        }

        // Tamper layer 2, re-walk, find first divergence.
        let mut tampered = layers.clone();
        tampered[2][0] ^= 0x01;
        let mut t_heads = vec![GENESIS_HEAD];
        for (w, m) in tampered.iter().zip(metas.iter()) {
            let prev = *t_heads.last().unwrap();
            t_heads.push(layer_head(&prev, w, m));
        }
        let first_bad = heads.iter().zip(t_heads.iter()).position(|(a, b)| a != b).unwrap();
        assert_eq!(first_bad, 3, "divergence starts at head after layer 2");
    }
}
