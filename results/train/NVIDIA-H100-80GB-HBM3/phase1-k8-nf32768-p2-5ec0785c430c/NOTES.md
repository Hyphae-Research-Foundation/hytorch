# P2 validation — H100, 300 steps (2026-08-29)

**Change vs P1** (journalized as POLICY revision, `manifests/phase1-k8-nf32768-p2.json`):
`mag_max` 8.0 → 64.0, `aux_balance_coeff` 0.01 → 0.02, `policy_id` 2.

## Result: the catalog freeze is gone

| | P1 (step 299) | P2 (step 299) |
|---|---|---|
| COMMIT | 2 (~0%) | 31 753 (16.1%) |
| OVERFLOW | 118 986 (60.5%) | 164 804 (83.8%) |
| ABORT | 77 620 (39.5%) | 51 (0.03%) |
| val PPL gap vs baseline | +20.6% | **+2.9%** (747.3 vs 726.2) |

- T1: 4/4 consistent. All receipts green, chain continuous, EXPORT committed.
- The write path stays alive: COMMITs settle at ~16% of the k-budget (one
  winner per contended slot), ABORT is now the rare guard it was meant to be.
- OVERFLOW at ~84% says global top-k concentrates candidates on few slots —
  the aux pressure (aux ≈ 1114 vs uniform 4096… lower is more uniform here,
  metric = N_f·Σp²) improved from P1 but slot concentration remains the
  dominant journalized phenomenon. Next policy lever if needed: per-slot
  top-k or home remapping — BOTH are new POLICY records by spec §02.

## Verdict for M8

At 300 steps the preregistered 10% PPL-gap threshold would PASS (+2.9%).
This does NOT preregister the 20k run — it validates that the harness,
policy P2 and both arms are ready for it. Cost of this validation: ~$1.
