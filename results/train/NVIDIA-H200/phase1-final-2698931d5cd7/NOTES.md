# CITABLE RUN — phase1-final (P3), H200, 20 000 steps (2026-08-29/30)

## Verdict against the preregistered threshold

| | catalog | baseline (dense parallel twin) |
|---|---|---|
| val PPL (wikitext-103, fixed windows) | **179.75** | **161.69** |
| final train loss | 5.298 | 5.192 |
| wall | 8 586.6 s (2.39 h) | 1 012.5 s |

**Gap = +11.17%. Preregistered threshold (user-signed): ≤ 10.0%.**

**→ FAIL. The branch dies. The negative is published with its manifest.**
No re-run with a tweaked threshold, no post-hoc rationalization: the number
was signed before step 0 (`manifests/phase1-final.json`, cited by RUN_START
in the ledger). This document IS the promised artifact: "un negativo con
manifiesto vale más que una narrativa" (spec v2.2 §12).

## The audit trail is impeccable — the loss is scientific, not procedural

- Differential gate on this silicon (H200): BIT_IDENTICAL, 200k cases.
- **T1: 351/351 audited microbatches consistent** (same-binary CPU replay).
- 20 000/20 000 optimizer steps gated by durable receipts; codebook chain
  (c_prev→c_next) continuous end to end; 19 SEALs of θ; EXPORT committed.
- 0 codebook resets (no collapse); last-step verdicts C=29 418 / O=167 127 /
  A=63 — the write path stayed alive to the end (P1's freeze stayed fixed).
- Ledger overhead: 813.6 s / 8 586.6 s = **9.5% wall** (< 10% budget) —
  but note the honest asymmetry below.
- Ledger size: ~186 GB for 20k steps (engine amplification ~3× over the
  3 MiB/step wire; wire itself within the manifest budget).

## Honest asymmetries to carry into the paper

1. **Catalog wall = 8.5× baseline wall.** compute_s 7 657.8 vs 1 012.5:
   the gap is NOT the ledger (813 s); it is pack's top-k sweep (N_f=32 768
   scores/token/layer, one thread per token) plus the still-serial spool.
   FLOPs accounting includes pack; wall accounting shows where engineering
   (fused kernels, ring seam) must go in phase 2.
2. **The +11.17% gap is a 20k-step number on a 124M toy.** The sweep showed
   the gap SHRINKS with training (300 steps: +20.6% P1 → +2.9% P2-300 →
   +5.0% P3-300; at 20k: +11.17%). No extrapolation is claimed either way:
   the preregistered scale was 20k and the number at 20k is the number.
3. Aux=0 was the best of the preregistered sweep; richer balance policies
   (per-slot top-k, learned homes) are named POLICY levers, not tested here.

## What survives the branch's death (everything that matters)

C1 (system) and C2 (verification) claims are UNTOUCHED — they are about the
harness, and the harness performed flawlessly for 20k steps on H200 after
doing so on H100, RTX 4000 Ada and MI355X. C3 (empirics) now has its honest
centerpiece: a preregistered, fully-audited, cross-referenced NEGATIVE with
a diagnosis trail (P1 freeze → P2 fix → P3 aux finding → 20k miss by 1.17).

## Files

- `run-c35f6284-catalog-1788045305.runlog.json` (full step log, 351 T1
  verdicts, timing splits, seals, export)
- `run-c35f6284-baseline-1788053933.runlog.json`
- `train.log` (gate verdict, build facts, data sha256)
- Ledger: was on droplet /var/lib/hytorch/hyphae (186 GB, torn down with the
  droplet after runlogs + spills verified; heads and receipts live in the
  runlog). Phase-2 note: pull receipts DB before teardown for full
  re-queryability.
