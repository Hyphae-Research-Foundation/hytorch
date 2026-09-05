//! verifier: T1 replay (same-binary) and T2 walk. Spec v2.2 §09.
//!
//! T1 proves consistency apply ↔ journal ↔ codebook-of-STEP on the audited
//! 1/N sample. T2 seals the persisted WAL, always. Neither is omniscience:
//! the invariant on the complement is the binary cited in RUN_START.

pub mod spill;
pub mod t1;
pub mod t2;

pub use spill::{read_spill, Spill, SpillError, SPILL_MAGIC, SPILL_VERSION};
pub use t1::{t1_replay, T1Outcome};
pub use t2::{t2_walk, T2Outcome};
