//! seam: authority ≠ durability, made code. Spec v2.2 §06–07.
//!
//! The trainer writes layer frames (BindingMin wire, already elided) to a
//! spool; the seam computes the T2 chain, persists facts in embedded Hyphae
//! (one atomic batch per step = the step transaction), and emits a receipt.
//! `opt.step()` on the trainer side is illegal until the receipt exists.
//!
//! Fact records (all under one atomic `put_records` batch per step):
//!   run/<run_id>/step/<step>/layer/<l>          → wire bytes + meta + head
//!   run/<run_id>/step/<step>/STEP               → c_prev/c_next, lr, grad_norm
//!   run/<run_id>/step/<step>/RECEIPT            → final head, counts, ts
//! Plus, once per run:
//!   run/<run_id>/RUN_START                      → manifest hash + build.* four
//!   run/<run_id>/POLICY/<policy_id>             → the allocate policy object

pub mod frames;
pub mod store;

pub use frames::{FrameHeader, StepFrames, FRAME_MAGIC};
pub use store::{SeamStore, StepFacts, StepReceipt};
