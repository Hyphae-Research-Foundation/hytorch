# crates/ — the Rust core

Four crates, one workspace (`cargo build --workspace --release`). All Apache-2.0.

| crate | binary / artifact | role |
|---|---|---|
| `apply-ref` | `libapply_ref.so` (cdylib) | **The pinned policy.** pack → allocate → apply as a bit-exact bf16/fp32 software semiring: promote bits, sequential fp32 reductions without FMA, RNE downcast, canonical NaN. Linked identically by the trainer (ctypes) and the verifier; its SHA-256 is a fact of every run. Also exposes `pack_allocate_candidates` for policy 7 (two-phase). |
| `binding-wire` | library | Fact formats: `BindingMin` (16 B) and `BindingT15` (80 B, audited spills), verdict/reason codes, and the SHA-256 T2 chain over frames. |
| `seam` | `hytorch-seam`, `hytorch-ledger-query` | **Authority ≠ durability.** Rank-0 process that consumes per-layer wire frames from a spool, walks T2, and commits `{layer records, RECEIPT, STEP chain, POLICY, BYPASS, CODEBOOK_RESET}` into an embedded Hyphae store (`hyphae-engine` 2.2). Emits the receipt that gates `optimizer.step()`. `hytorch-ledger-query` dumps wire and records for analysis. |
| `verifier` | `hytorch-verify` | CPU-only. T1: replay audited microbatch spills through the same `libapply_ref.so`, compare residual bits. T2: walk the chain. |

Spec references in the module headers (`//! ... Spec v2.2 §NN`) point at `docs/spec/`.

Release profile is deliberately boring (`lto = false`, `codegen-units = 1`,
`panic = "abort"`): the reference object must be reproducible, because its hash is cited.
