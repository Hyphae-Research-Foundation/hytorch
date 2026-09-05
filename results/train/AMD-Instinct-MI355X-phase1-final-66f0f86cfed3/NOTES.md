# CITABLE REPLICATION — phase1-final (P3), AMD MI355X (spot), 20 000 steps (2026-08-30)

Same immutable manifest (`phase1-final.json`), same sha256-pinned data bins,
same `libapply_ref.so` reference, same seed — different vendor. Spot at
$4.50/h, zero preemptions absorbed.

## Verdict against the preregistered threshold

| | catalog | baseline (dense twin) |
|---|---|---|
| val PPL | **185.15** | **158.21** |
| wall | 10 661.1 s (2.96 h) | 509.7 s |
| T1 | **291/291 consistent** | — |
| ledger | 354.1 s (**3.3% wall**) | — |
| last-step verdicts | C=29 899 / O=166 574 / A=135 | — |
| seals / resets | 19 / 0 | — |

**Gap = +17.03% > 10% → FAIL on AMD too.** The negative replicates
cross-vendor: the branch stays dead, now with two-silicon evidence.

## What replicates and what does not (spec §04 honesty)

**Replicates (the claims):**
- The threshold verdict: FAIL on both vendors (+11.17% H200, +17.03% MI355X).
- The audit trail: gate BIT_IDENTICAL on gfx950, 291/291 T1 green, 20k/20k
  receipt-gated steps, continuous codebook chain, 19 seals, EXPORT.
- The catalog's qualitative behavior: write path alive to the end, verdict
  mix in the same regime (C≈15%, O≈85%, A≈0.07%), zero collapse.

**Does not replicate (declared, not a bug):** absolute PPL numbers.
delta_hat genesis (matmuls, attention) is not comparable across vendors —
the spec never promised it; bindings declare their device. The dense
baselines themselves differ (161.69 vs 158.21), so the gap difference
(11.17 vs 17.03) mixes vendor-genesis effects in BOTH arms. What is
bit-identical across vendors is the APPLICATION of facts (the gate) and the
verification (same-binary replay).

## Interesting datapoint for phase 2

MI355X baseline wall (509.7 s) is ~2× faster than H200's (1 012.5 s) at this
tiny scale, while the catalog arm is SLOWER (10 661 vs 8 587 s): the pack
top-k sweep dominates and its per-token-thread kernel maps worse to CDNA
occupancy here. Ledger overhead is LOWER on AMD (3.3% vs 9.5%) because the
compute denominator is larger. Both point the phase-2 engineering at pack,
not at the ledger.
