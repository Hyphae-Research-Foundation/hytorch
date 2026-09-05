# The Missing Medium: A Transactional Residual Stream for Transformer Training

**Working title.** Paper skeleton — every claim below cites a file in `results/`
or a record in a Hyphae ledger. Numbers marked ⏳ await the 20k citable run.

## Thesis (one paragraph)

Modern training treats the residual stream as an anonymous accumulator: any
kernel may execute `h += d` and no record survives. We present the first
training loop in which the residual stream is a **typed, transactional
channel**: every write is a 16-byte fact (binding) with slot, feature,
magnitude and verdict; every contention (OVERFLOW) and rejection (ABORT) is
journalized; every optimizer step is gated by a durable commit receipt; and a
CPU-only verifier replays audited microbatches bit-for-bit against a pinned
software reference — on both NVIDIA and AMD silicon, from the same reference
binary. The model may remain opaque; its *state transitions* cannot.

## Contributions

**C1 — System (the strong one).** A training harness where authority ≠
durability is realized as code:
- Bit-exact software semiring (`apply_ref`): bf16/fp32, RNE, no FMA,
  signed-zero no-op by rule. Its `.so` hash is a fact every run cites.
- Per-silicon differential gate: 200k adversarial cases, verdict-stream and
  residual equality bit-for-bit. **BIT_IDENTICAL on NVIDIA H100 (sm_90),
  RTX 4000 Ada, H200, and AMD MI355X (gfx950)** — one reference, two vendors,
  no `if nvidia / if amd` in the seam.
  [results/gates/*, results/train/*/summary.json]
- Zero per-layer syncs: verdicts stay on device; journal D2H is async into
  pinned memory materialized after backward; ONE sync per step. Ledger
  overhead **7.5% of wall** (< 10% preregistered budget). ⏳ final number.
- Fail-stop semantics proven by test: no receipt → no `opt.step()`; codebook
  mutated outside the ledger → no chainack → run dies. No ledger-less
  degradation path exists. [python/tests/test_e2e_seam.py]

**C2 — Verification.** T1 (same-binary replay of audited microbatches: h₀ +
journal + C_step → residual hash) and T2 (per-layer hash chain over persisted
facts, elision-aware). Fault injection over REAL training spills: **32/32
faults detected (100%), each with the correct outcome class** (wire tamper →
head_mismatch; codebook tamper → residual_mismatch; reorder → apply_error).
Honest runs: zero false positives, including `-0.0` leaves.
[results/inject/inject-summary.json]

**C3 — Empirics (the honest one).** The catalog is not free; the ledger made
its costs *visible and diagnosable*:
- **P1 catalog freeze, found BY the journal**: at mag_max=8.0 the write path
  froze (step 299: COMMIT=2, OVERFLOW=118 986, ABORT=77 620 — exact
  k-partition) while loss kept dropping via embed+head. Invisible without
  the ledger. [results/train/.../phase1-k8-nf32768-bee02f19d243/NOTES.md]
- **P2 policy revision as a journalized fact** (mag_max 64): freeze resolved,
  gap +2.9% @ 300 steps. [.../phase1-k8-nf32768-p2-5ec0785c430c/NOTES.md]
- **Aux-loss finding**: the Switch-style differentiable balance term HURTS at
  this scale (sweep on fixed silicon: coeff 0 → +5.0% gap; 0.002 → +16.6%;
  0.02 → +23.3%). Collapse remains guarded by preregistered entropy floor.
  [results/sweep/.../p3aux-15153883c56f/]
- **Preregistered threshold**: PPL gap ≤ 10% at matched FLOPs vs a dense
  parallel-block twin (same topology, per spec C2), 20k steps, wikitext-103,
  sha256-pinned data. ⏳ RESULT PENDING — pass or fail, it is publishable:
  the negative with manifest is a result by contract.

**C4 — Method.** The spec's correction registry (E1–E13, A1–A6, C1–C15,
D1–D5, SPEC_AMEND-001/002) as a first-class artifact: "a spec that hides its
history cannot ask a model to leave one." Includes a cross-vendor determinism
bug (NaN canonicalization) found by the gate, not by review.
[docs/, docs/spec/SPEC_AMEND-002-nan-canonical.md]

## Evidence table (auto-collected)

| kind | silicon | verdict | detail | file |
|---|---|---|---|---|
| gate | AMD MI355X (gfx950) | BIT_IDENTICAL | 200k | results/gates/AMD-Instinct-MI355X/9819b6d795ca/ |
| gate | NVIDIA H100 (sm_90) | BIT_IDENTICAL | 200k | results/gates/NVIDIA-H100-80GB-HBM3/3820e6370fa3/ |
| rehearsal | MI355X | gate + 9/9 T1 | spot $4.50/h | results/gpu-rehearsal/AMD-Instinct-MI355X/249d1d140698/ |
| rehearsal | H100 | gate + 9/9 T1 | — | results/gpu-rehearsal/NVIDIA-H100-80GB-HBM3/744fa86b3803/ |
| inject | — | 100% (32/32) | 6 fault families | results/inject/ |
| train P1 | H100 | freeze found | 300 steps | results/train/.../phase1-k8-nf32768-bee02f19d243/ |
| train P2 | H100 | gap +2.9% | 300 steps | results/train/.../phase1-k8-nf32768-p2-5ec0785c430c/ |
| sweep P3 | RTX 4000 Ada | aux hurts | 3×300 steps | results/sweep/.../p3aux-15153883c56f/ |
| **citable** | **H200** | ⏳ running | 20k steps, both arms | results/train/... (pending) |
| citable AMD | MI355X | planned | replication | — |

## Venue reasoning

- If C1/C2 dominate the narrative → **MLSys** (systems for ML: novel training
  infrastructure, cross-vendor bit-exactness, verifiable state transitions).
- If C3 surprises (gap well under threshold at 20k) → ICLR/NeurIPS workshop
  first (Reliable ML / ML Safety), main track later with scale.
- The honest default: MLSys paper with C3 as evaluation. The negative result
  does not weaken C1/C2 — the artifact's job is to make exactly that cost
  measurable and citable.

## What is NOT claimed (lista negra of the paper)

- No cross-vendor reproducibility of delta_hat genesis (declared per spec §04).
- No interpretability claims: bindings are facts, not meanings.
- No adversarial-trainer security (threat model C6: implementation faults).
- No scale beyond the toy (12L/768d/124M params) in phase 1.
