//! binding-wire: the packed fact formats and the T2 chain. Spec v2.2 §08–09.
//!
//! BindingMin is 16 bytes little-endian, no id string: identity is position
//! in the T2 chain. `mag` is raw bf16 bits (NOT IEEE f16 — spec D3). OVERFLOW
//! and ABORT have wire representation (spec D4): the non-fact is data.

pub mod wire;
pub mod chain;

pub use chain::{layer_head, LayerMeta, GENESIS_HEAD};
pub use wire::{
    decode_min, encode_min, BindingMin, BindingT15, Reason, Verdict, WireError, BINDING_MIN_SIZE,
    BINDING_T15_SIZE,
};
