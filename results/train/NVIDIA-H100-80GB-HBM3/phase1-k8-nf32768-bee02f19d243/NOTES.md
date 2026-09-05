# Validation run notes — H100, phase1-k8-nf32768, 300 steps (2026-08-29)

**Purpose:** validate the full citable circuit on real data (wikitext-103,
gpt2 BPE, vocab 50257) on H100. NOT the 20k-step citable run.

## Circuit verdicts (all green)

- Differential gate on this silicon: BIT_IDENTICAL (200k cases).
- RUN_START + POLICY committed in Hyphae; receipts gated every opt.step();
  STEP chain continuous (real c_prev/c_next post-renorm); EXPORT committed.
- T1 audits: 5/5 consistent (steps 147, 198, 210, 261, 289).
- Loss sanity: first loss 11.01 ≈ ln(50257)=10.82 ✓.

## The finding the journal surfaced (this is the artifact working)

Late-training verdict distribution (step 299, N=196 608 = 12×2048×8 exact):

| verdict | count | share |
|---|---|---|
| COMMIT | 2 | ~0.001% |
| OVERFLOW | 118 986 | 60.5% |
| ABORT (mag_overflow) | 77 620 | 39.5% |

The write path effectively froze: as the ReZero gate and block outputs grow,
raw scores `⟨leaf, Ĉ⟩` exceed `mag_max=8.0` (ABORT) and global top-k
concentrates candidates into few slots (OVERFLOW). Loss kept dropping
(11.0 → 6.8) because embed+head keep learning — but the residual channel
carried almost nothing by step 300.

**This is journalized, consultable, and citable — not a harness bug.** The
partition #C+#O+#A = k held exactly; T1 replays green; the no-facts are data.

## Implications for the citable 20k run (policy work, prerregistered)

1. `mag_max` and/or score scaling need a POLICY revision (a new POLICY record
   — a fact, not a hotfix, per spec §02). Candidates: normalize leaf before
   scoring (score = cosine, |mag| ≤ 1), or mag_max sweep.
2. The aux-balance coefficient interacts: with everything aborted, the
   committed-feature distribution is degenerate and aux pressure is noise.
3. Wall overhead: catalog arm 134 s vs baseline 16.5 s at 300 steps (~8×) —
   dominated by the known per-layer device→CPU pack roundtrip debt, not by
   Hyphae (receipts are per-step and cheap). The fused-ring seam remains the
   fix; measure again after.

## Gap snapshot (NOT the preregistered threshold — 300 steps only)

val PPL: catalog 875.3 vs baseline 725.7 (gap ≈ +20.6% at 300 steps, both
arms same seed, same eval windows, FLOPs/token 275.4M vs 265.9M).
