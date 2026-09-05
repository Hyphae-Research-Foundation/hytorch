//! Spill format: what the trainer dumps for an audited microbatch (spec §09).
//!
//! One file per audited microbatch, little-endian, versioned:
//!
//! ```text
//! [0..8)     magic  "HYTSPILL"
//! [8..12)    version u32 = 2
//! [12..20)   step_id u64
//! [20..24)   microbatch_id u32
//! [24..28)   n_tokens u32
//! [28..29)   s_slots u8
//! [29..32)   pad [3]
//! [32..36)   d_slot u32
//! [36..38)   n_layers u16
//! [38..40)   n_features u16 (codebook rows)
//! [40..72)   expected_final_head [32]        (last layer head from receipts)
//! [72..104)  expected_h_final_sha256 [32]    (v2: H of the residual bits the
//!                                             device ACTUALLY produced)
//! [104..112) policy_id u64
//! then:
//!   h0:        n_tokens * s_slots * d_slot * 2 bytes  (bf16 bits, raw)
//!   codebook:  n_features * d_slot * 2 bytes          (bf16 bits, C_step = c_next)
//!   per layer, n_layers times:
//!     meta: microbatch-local LayerMeta fields we don't already know:
//!       n_persisted u32, n_elided u32
//!     wire: n_persisted * 16 bytes (BindingMin, elided already)
//! ```
//!
//! The layer order in the file IS the forward order. Heads are recomputed by
//! the verifier from GENESIS_HEAD; the trainer only commits the final head.
//!
//! Why v2 exists: the T2 head covers the WIRE, not the codebook contents.
//! A tampered codebook in the spill replayed to a different residual and
//! nothing caught it (documented gap in the v1 tests). With
//! `expected_h_final_sha256`, T1 now closes apply ↔ journal ↔ codebook:
//! any inconsistency between the three changes the replayed residual hash.

use binding_wire::{decode_min, BindingMin, LayerMeta, BINDING_MIN_SIZE};

pub const SPILL_MAGIC: &[u8; 8] = b"HYTSPILL";
pub const SPILL_VERSION: u32 = 2;

#[derive(Debug)]
pub struct Spill {
    pub step_id: u64,
    pub microbatch_id: u32,
    pub n_tokens: u32,
    pub s_slots: u8,
    pub d_slot: u32,
    pub n_layers: u16,
    pub n_features: u16,
    pub expected_final_head: [u8; 32],
    pub expected_h_final_sha256: [u8; 32],
    pub policy_id: u64,
    pub h0: Vec<u16>,
    pub codebook: Vec<u16>,
    /// Per layer: (meta, decoded bindings, raw wire bytes as persisted).
    pub layers: Vec<(LayerMeta, Vec<BindingMin>, Vec<u8>)>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum SpillError {
    BadMagic,
    BadVersion(u32),
    Truncated(&'static str),
    Wire(&'static str),
}

fn take<'a>(buf: &mut &'a [u8], n: usize, what: &'static str) -> Result<&'a [u8], SpillError> {
    if buf.len() < n {
        return Err(SpillError::Truncated(what));
    }
    let (head, rest) = buf.split_at(n);
    *buf = rest;
    Ok(head)
}

fn u16s_of(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

pub fn read_spill(mut buf: &[u8]) -> Result<Spill, SpillError> {
    let magic = take(&mut buf, 8, "magic")?;
    if magic != SPILL_MAGIC {
        return Err(SpillError::BadMagic);
    }
    let ver = u32::from_le_bytes(take(&mut buf, 4, "version")?.try_into().unwrap());
    if ver != SPILL_VERSION {
        return Err(SpillError::BadVersion(ver));
    }
    let step_id = u64::from_le_bytes(take(&mut buf, 8, "step_id")?.try_into().unwrap());
    let microbatch_id = u32::from_le_bytes(take(&mut buf, 4, "mb")?.try_into().unwrap());
    let n_tokens = u32::from_le_bytes(take(&mut buf, 4, "n_tokens")?.try_into().unwrap());
    let s_slots = take(&mut buf, 1, "s_slots")?[0];
    let _pad = take(&mut buf, 3, "pad")?;
    let d_slot = u32::from_le_bytes(take(&mut buf, 4, "d_slot")?.try_into().unwrap());
    let n_layers = u16::from_le_bytes(take(&mut buf, 2, "n_layers")?.try_into().unwrap());
    let n_features = u16::from_le_bytes(take(&mut buf, 2, "n_features")?.try_into().unwrap());
    let expected_final_head: [u8; 32] =
        take(&mut buf, 32, "final_head")?.try_into().unwrap();
    let expected_h_final_sha256: [u8; 32] =
        take(&mut buf, 32, "h_final_sha")?.try_into().unwrap();
    let policy_id = u64::from_le_bytes(take(&mut buf, 8, "policy_id")?.try_into().unwrap());

    let h_len = n_tokens as usize * s_slots as usize * d_slot as usize * 2;
    let h0 = u16s_of(take(&mut buf, h_len, "h0")?);
    let c_len = n_features as usize * d_slot as usize * 2;
    let codebook = u16s_of(take(&mut buf, c_len, "codebook")?);

    let mut layers = Vec::with_capacity(n_layers as usize);
    for layer in 0..n_layers {
        let n_persisted =
            u32::from_le_bytes(take(&mut buf, 4, "n_persisted")?.try_into().unwrap());
        let n_elided = u32::from_le_bytes(take(&mut buf, 4, "n_elided")?.try_into().unwrap());
        let wire = take(&mut buf, n_persisted as usize * BINDING_MIN_SIZE, "wire")?.to_vec();
        let mut decoded = Vec::with_capacity(n_persisted as usize);
        for i in 0..n_persisted as usize {
            let b = decode_min(&wire[i * BINDING_MIN_SIZE..], s_slots)
                .map_err(|_| SpillError::Wire("decode_min"))?;
            decoded.push(b);
        }
        let meta = LayerMeta {
            step_id,
            layer,
            microbatch_id,
            n_persisted,
            n_elided,
            policy_id,
        };
        layers.push((meta, decoded, wire));
    }
    Ok(Spill {
        step_id,
        microbatch_id,
        n_tokens,
        s_slots,
        d_slot,
        n_layers,
        n_features,
        expected_final_head,
        expected_h_final_sha256,
        policy_id,
        h0,
        codebook,
        layers,
    })
}

/// Writer used by tests and (later) the Python harness via file convention.
pub fn write_spill(s: &Spill) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(SPILL_MAGIC);
    out.extend_from_slice(&SPILL_VERSION.to_le_bytes());
    out.extend_from_slice(&s.step_id.to_le_bytes());
    out.extend_from_slice(&s.microbatch_id.to_le_bytes());
    out.extend_from_slice(&s.n_tokens.to_le_bytes());
    out.push(s.s_slots);
    out.extend_from_slice(&[0u8; 3]);
    out.extend_from_slice(&s.d_slot.to_le_bytes());
    out.extend_from_slice(&s.n_layers.to_le_bytes());
    out.extend_from_slice(&s.n_features.to_le_bytes());
    out.extend_from_slice(&s.expected_final_head);
    out.extend_from_slice(&s.expected_h_final_sha256);
    out.extend_from_slice(&s.policy_id.to_le_bytes());
    for &v in &s.h0 {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &s.codebook {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for (meta, _, wire) in &s.layers {
        out.extend_from_slice(&meta.n_persisted.to_le_bytes());
        out.extend_from_slice(&meta.n_elided.to_le_bytes());
        out.extend_from_slice(wire);
    }
    out
}
