# CITABLE REPLICATION — phase1b-global (P4, no ReZero), AMD MI355X (spot), 20 000 steps (2026-08-30)

Same immutable manifest (`phase1b-global.json`), same sha256-pinned data,
same `libapply_ref.so`, same seed — different vendor. Spot at $4.50/h,
zero preemptions.

## Verdict against the preregistered threshold

| | catalog | baseline (dense twin) |
|---|---|---|
| val PPL | **169.68** | **298.78** |
| final train loss | 5.239 | 5.786 |
| wall | 10 743.9 s | 502.0 s |
| T1 | **317/317 consistent** | — |
| gate (gfx950, both selection modes) | BIT_IDENTICAL (200k) | — |

**Gap = −43.2% (catalog better) → PASS on AMD too. The PASS replicates
cross-vendor.**

## Two-vendor picture, phase 1b (no ReZero)

| | H200 catalog | H200 twin | MI355X catalog | MI355X twin |
|---|---|---|---|---|
| val PPL | 167.80 | 304.70 | 169.68 | 298.78 |
| gap | **−44.9%** | — | **−43.2%** | — |
| T1 | 309/309 | — | 317/317 | — |

The same qualitative story on both chips: the dense twin destabilizes at
this LR (its loss drifts up after ~10k) while the catalog's bounded,
quantized, k-sparse writes keep improving. Absolute PPLs differ across
vendors (genesis is vendor-local, §04 — never promised); the verdict and
the audit replicate.

Phase-1 arc, complete and journalized: 1a with gate → preregistered FAIL
both vendors (+11.17% / +17.03%) → SPEC_AMEND-003 removes the gate (user
directive) → 1b PASS both vendors (−44.9% / −43.2%), same signed threshold,
same protocol, 626 green T1 audits in 1b alone.
