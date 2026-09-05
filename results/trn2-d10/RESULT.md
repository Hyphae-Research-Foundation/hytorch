# Trainium2 d10 — catalog (policy 7, two-phase) vs dense twin — 2026-09-04

**Where:** trn2.3xlarge (1 Trainium2, 4 logical NeuronCores at LNC=2),
sa-east-1b, capacity block `cr-0e1d67d2d6ab5868f`. torch-xla 2.9,
neuronx-cc 2.26.6360. Model: nanochat d10 a64 (d_model 640, 10 layers, 59.6M
params), seq 1024, DBS 4 per rank, TBS 32,768, 400 steps, wikitext-103.
Same tree for both arms; the twin is `HYTORCH_CATALOG` unset (bit-identical
to upstream). Catalog: S=64 slots × d_slot=10, N_f=32,768, k=8, M=32
candidates, policy 7 (two-phase top-k), proposal clip 32, STE mirror on host.

**Custody:** `s3://hytorch-custody/trn2-d10-20260904/`
(56 objects, 5.38 GB; `artifact-hashes.txt` = sha256 of every file).
Ledgers `d10/hyphae{5,6,7}` (Hyphae stores, run ids `trn2-d10-catalog` ×2 and
`trn2-d10-catalog-w200`), all launcher/arm logs, both manifests, the
catalog checkpoint `d10/catalog7-final.pt` (sha256 `086e2630e8cc…`, contains
`hytorch_codebook` [32768,10]). The twin checkpoint is `d10/out-vanilla7/final.pt`
(sha256 `3c62b8d19631…`, regenerated 16:12–16:16Z in 4 min from the NEFF cache
because `train.py` writes a single `out/final.pt` and the catalog run overwrote
the twin's; the recovery run reproduced the final loss 3.4142 exactly).

## Runs

| run | manifest | steps | end | loss@100 | loss@399 | tok/s | channel |
|---|---|---|---|---|---|---|---|
| vanilla (`vanilla6.log`) | — | 400/400 | rc=0 | 5.212 | **3.414** | **84,500** | — |
| catalog take 5 (`catalog5.log`, `hyphae5`) | `manifest-d10.json` | 149 | **guard kill** (entropy 6.90 b < 8 b × 50) | 5.420 | — | 2,900 | commit 12.5 % (=1/k) from step 0 — proposer tie-break bug (NOTES f.8) |
| catalog take 6 (`catalog6.log`, `hyphae6`) | `manifest-d10.json` | 149 | **guard kill** (entropy 7.36 b < 8 b × 50, rising) | 5.359 | — | 2,900 | commit 13 %, abort 0, entropy 5.0→6.1→7.4 b |
| catalog take 7 (`catalog7.log`, `hyphae7`) | `manifest-d10-w200.json` | 400/400 | rc=0, **0 guard warnings** | 5.359 | **4.667** | **2,813** | commit 13→14 %, overflow 86 %, abort 0 %, entropy 8.39 b @200 → 9.03 b @350, distinct 1,779 → 3,062 |

Matched-step losses, twin / catalog (same seed and data): 50: 5.542 / 5.521 ·
100: 5.212 / 5.359 · 150: 4.877 / 5.274 · 200: — / 5.181 · 399: 3.414 / 4.667.

Catalog channel per 50 steps (take 7, identical to take 6 up to 149 —
deterministic): step 0 commit 62.6 % (1,425 features, the tie-break working
on an all-zero step-0 proposal), then concentration (step 4: 15 %, 1.1 b),
then slow diversification: 50: 13.2 % / 5.04 b / 163 features · 100: 13.3 % /
6.12 b / 570 · 150: 13.1 % / 7.37 b / 1,192 · 200: 13.3 % / 8.39 b / 1,779 ·
250: 13.4 % / 8.82 b / 2,197 · 300: 13.6 % / 8.89 b / 2,658 · 350: 14.0 % /
9.03 b / 3,062. Receipts 400/400, STEP chain continuous (per-step heads in
`spool7/seam.stderr`), 5,242,880 facts persisted per step (160 layers ×
32,768 tokens... = 4 ranks × 2 microbatches × 20 write units), 2.2 GB ledger.

## What this establishes

1. **The typed, transactional residual stream trains on a third
   architecture at a non-toy size**: 400 steps, DDP over 4 NeuronCores,
   every step receipted, every write journalized, the same reference binary
   disposing the verdicts, the same executable guard watching the channel.
2. **The preregistered guard did its job twice**, live, on Trainium: take 5
   was a real collapse (a proposer tie-break bug funnelled the channel into
   one slot — C4 again, caught at step 149); take 6 was the declared
   negative for `manifest-d10.json` (entropy rising but < 8 b at 100+50).
   `manifest-d10-w200.json` (warm-up 200, threshold and window unchanged)
   was declared before take 7 and take 7 cleared 8 bits at step 200.
3. **The price at d10 on this stack:** +1.25 nats at step 399 (4.667 vs
   3.414) with a channel committing ~14 % of its k=8 candidates per token,
   and **29× lower throughput** (2.8k vs 84.5k tok/s) — the per-write-unit
   host round trip (20 units × 2 microbatches per step), not the device
   proposer. The two-pass forward (propose all units → one dispose → one
   apply) is the lever and remains future work.
4. **Six compiler limits had to be routed around** to get here (NOTES
   findings 4–8 + the stale-lock note): TopK over 32768 with k≥16, any
   `d_slot`-shaped tensor in the fwd+bwd graph, the codebook grad in the
   all-reduce bucket, the bucket's zero-fill, the two-stage tie-break, and
   the compile-cache lock left by a killed compile. None weakened the
   verification contract; all are in `python/hytorch/neuron_backend.py`,
   `python/hytorch/model.py`, `trn/trn_bridge.py`, `trn/apply-patch.sh`.

## Not done

- No CORE eval of either d10 checkpoint (needs the nanochat eval harness on
  a GPU/CPU box; both checkpoints are in S3).
- Wall-clock overhead of the seam itself is not separable here from the
  two-phase host round trip; the 9.5 % seam-overhead number in the paper is
  the AMD measurement.
