//! Frame format: what the trainer spools per layer. Little-endian, versioned.
//!
//! ```text
//! [0..8)   magic "HYTFRAME"
//! [8..12)  version u32 = 1
//! [12..20) step_id u64
//! [20..22) layer u16
//! [22..26) microbatch_id u32
//! [26..30) n_persisted u32
//! [30..34) n_elided u32
//! [34..42) policy_id u64
//! [42..)   n_persisted * 16 bytes of BindingMin wire (already elided)
//! ```
//! One file per (step, layer) in the spool directory; the seam consumes them
//! in (step, layer) order. In the ring-buffer future the framing is identical.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    pub step_id: u64,
    pub layer: u16,
    pub microbatch_id: u32,
    pub n_persisted: u32,
    pub n_elided: u32,
    pub policy_id: u64,
}

pub const FRAME_MAGIC: &[u8; 8] = b"HYTFRAME";
pub const FRAME_VERSION: u32 = 1;
pub const FRAME_HEADER_SIZE: usize = 42;

#[derive(Debug)]
pub enum FrameError {
    BadMagic,
    BadVersion(u32),
    Truncated,
}

pub fn encode_frame(h: &FrameHeader, wire: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(FRAME_HEADER_SIZE + wire.len());
    out.extend_from_slice(FRAME_MAGIC);
    out.extend_from_slice(&FRAME_VERSION.to_le_bytes());
    out.extend_from_slice(&h.step_id.to_le_bytes());
    out.extend_from_slice(&h.layer.to_le_bytes());
    out.extend_from_slice(&h.microbatch_id.to_le_bytes());
    out.extend_from_slice(&h.n_persisted.to_le_bytes());
    out.extend_from_slice(&h.n_elided.to_le_bytes());
    out.extend_from_slice(&h.policy_id.to_le_bytes());
    out.extend_from_slice(wire);
    out
}

pub fn decode_frame(buf: &[u8]) -> Result<(FrameHeader, &[u8]), FrameError> {
    if buf.len() < FRAME_HEADER_SIZE {
        return Err(FrameError::Truncated);
    }
    if &buf[0..8] != FRAME_MAGIC {
        return Err(FrameError::BadMagic);
    }
    let ver = u32::from_le_bytes(buf[8..12].try_into().unwrap());
    if ver != FRAME_VERSION {
        return Err(FrameError::BadVersion(ver));
    }
    let h = FrameHeader {
        step_id: u64::from_le_bytes(buf[12..20].try_into().unwrap()),
        layer: u16::from_le_bytes(buf[20..22].try_into().unwrap()),
        microbatch_id: u32::from_le_bytes(buf[22..26].try_into().unwrap()),
        n_persisted: u32::from_le_bytes(buf[26..30].try_into().unwrap()),
        n_elided: u32::from_le_bytes(buf[30..34].try_into().unwrap()),
        policy_id: u64::from_le_bytes(buf[34..42].try_into().unwrap()),
    };
    let wire_len = h.n_persisted as usize * 16;
    if buf.len() < FRAME_HEADER_SIZE + wire_len {
        return Err(FrameError::Truncated);
    }
    Ok((h, &buf[FRAME_HEADER_SIZE..FRAME_HEADER_SIZE + wire_len]))
}

/// All frames of one step, in layer order, ready for the T2 walk + persist.
#[derive(Debug, Default)]
pub struct StepFrames {
    pub frames: Vec<(FrameHeader, Vec<u8>)>,
}

impl StepFrames {
    pub fn push(&mut self, h: FrameHeader, wire: Vec<u8>) {
        self.frames.push((h, wire));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let h = FrameHeader {
            step_id: 42,
            layer: 3,
            microbatch_id: 1,
            n_persisted: 2,
            n_elided: 6,
            policy_id: 1,
        };
        let wire = vec![0xAB; 32];
        let enc = encode_frame(&h, &wire);
        let (h2, w2) = decode_frame(&enc).unwrap();
        assert_eq!(h, h2);
        assert_eq!(&wire[..], w2);
    }

    #[test]
    fn truncation_rejected() {
        let h = FrameHeader {
            step_id: 1,
            layer: 0,
            microbatch_id: 0,
            n_persisted: 4,
            n_elided: 0,
            policy_id: 1,
        };
        let enc = encode_frame(&h, &vec![0u8; 64]);
        assert!(decode_frame(&enc[..enc.len() - 1]).is_err());
    }
}
