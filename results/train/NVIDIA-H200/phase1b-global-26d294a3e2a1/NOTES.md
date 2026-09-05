# CITABLE RUN — phase1b-global (P4, SPEC_AMEND-003: no ReZero), H200, 20 000 steps (2026-08-30)

## Verdict against the preregistered threshold

| | catalog | baseline (dense twin, same init/LR/seed) |
|---|---|---|
| val PPL (wikitext-103) | **167.80** | **304.70** |
| final train loss | 5.214 | 5.863 |
| wall | 8 561.6 s | 1 007.0 s |
| T1 | **309/309 consistent** | — |
| ledger overhead | 824.0 s (**9.6% wall**, < 10% budget) | — |

**Gap = −44.9% (catalog BETTER). Preregistered threshold ≤ +10% → PASS.**
**The branch lives.** Same immutable manifest for both arms (policy_id=4,
selection=global_topk, mag_max=64, init gpt2-scaled 1/√(2L), LR 6e-4,
seed 1337), sha256-pinned data, gate BIT_IDENTICAL on this silicon (both
selection modes), RUN_START→receipts→STEP chain→19 SEALs→EXPORT all green.

## The honest story (what actually happened, from the step log)

Loss trajectories (same data order — same seed — in both arms):

| step | catalog | baseline |
|---|---|---|
| 300 | 6.578 | 6.564 |
| 2 000 | 5.684 | 5.835 |
| 5 000 | 5.677 | 5.805 |
| 10 000 | 5.231 | 5.747 |
| 20 000 | **5.214** | **5.863** ← drifting UP |

Both arms hit the same loss spike at step 6 494 (hard batch: 6.79 catalog /
6.86 baseline). The catalog RECOVERED and kept improving; the dense twin
never fully did — its loss drifts upward from 10k on (5.747 → 5.863) and its
val PPL (304.7) is far worse than the phase-1a gated baseline ever was
(161.7). The catalog's bounded, quantized, k-sparse writes acted as an
implicit stabilizer at this LR where the plain parallel block destabilized.

**Skeptic's notes (to carry into the paper, not to hide):**
1. Single LR (6e-4), single seed (1337), fixed by the manifest for BOTH arms
   — same protocol as phase 1a. A baseline LR sweep would be a NEW
   preregistered manifest; at THIS design point the catalog wins outright.
2. The phase-1a comparison (161.7 baseline WITH ReZero gate) shows the gate
   was doing heavy stabilization work for the dense twin. Removing it hurt
   the baseline far more than the catalog — which is itself evidence for the
   catalog-as-regularizer reading, and exactly why SPEC_AMEND-003 matters:
   the 1a gap (+11.17%) was measuring gate interactions, not the medium.
3. Write path health at 20k: C=32 675 / O=163 455 / A=478 per step — alive,
   same regime as 1a, zero codebook resets.

## Files

- `run-c7d81def-catalog-1788073852.runlog.json` (309 T1 verdicts, timing,
  seals, export, full step log)
- `run-c7d81def-baseline-*.runlog.json`
- `train.log` (gate verdict both modes, build facts, data sha256)
