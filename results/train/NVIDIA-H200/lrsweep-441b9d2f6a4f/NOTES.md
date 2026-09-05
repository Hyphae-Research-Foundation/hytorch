# LR SWEEP — phase1b at 3 more LRs, both arms, H200, 20k each (2026-08-30)

Preregistered before running: "best-vs-best is the defensible comparison;
we report whatever happens." Here is whatever happened.

## Full grid (val PPL, wikitext-103, 20k steps, seed 1337)

| LR | catalog | baseline | winner |
|---|---|---|---|
| 1e-4 | 187.84 | **58.65** | baseline |
| 3e-4 | 144.13 | **49.33** | baseline |
| 6e-4 (headline) | **167.80** | 304.70 | catalog |
| 1.2e-3 | **153.44** | 303.93 | catalog |

**Best-vs-best: baseline 49.33 (lr=3e-4) vs catalog 144.13 (lr=3e-4).
The dense twin, properly tuned, is ~2.9× better in PPL. The −44.9% headline
was an artifact of comparing both arms at ONE LR (6e-4) where the dense twin
destabilizes.** 939/939 T1 audits green across the sweep.

## The honest reading (both directions)

1. **The catalog costs capacity at this scale.** At the dense twin's best
   LR, the k=8-sparse quantized channel cannot match dense residual writes.
   The preregistered phase-1b threshold (gap ≤ +10%) FAILS under best-vs-best
   framing: 144.13/49.33 = +192%. The single-LR PASS was real but fragile —
   exactly why this sweep was preregistered before celebrating.
2. **The catalog is far more LR-robust and seed-robust.** Baseline collapses
   above 3e-4 (49→304 PPL, 6×); catalog degrades gently across the entire
   sweep (144–188, 1.3×) and had 7× less seed variance. The channel acts as
   a stabilizer — a real property, not spin — but stability at 3× the
   perplexity is not a win at this scale; it is a trade.
3. Every claim above is receipt-backed: 939 audited microbatches, continuous
   codebook chains, gate BIT_IDENTICAL per silicon.

## Implication for the paper and for phase 3

- The paper's §7 must lead with the best-vs-best table, not the single-LR
  headline. The honest contributions stand unchanged: C1/C2 (the system,
  the audits, cross-vendor bit-exactness) never depended on winning PPL;
  C3's centerpiece is now "the ledger measured the true cost of the typed
  channel, including catching our own too-good-to-be-true headline."
- Phase 3 (nanochat) remains the right next step BUT with corrected
  expectations: the d20 question is not "does the catalog win?" — it is
  "what does the typed channel cost at speaking-model scale, measured
  end-to-end with receipts, and what does the journal reveal about a real
  model's internals?" Both answers are publishable; neither is spin.
