# Phase 5 RESULT — d20 catalog vs dense twin, full compute-optimal horizon (2026-09-03)

**Preregistered question** (manifests/phase5-nanochat-d20.json): does the typed
channel's capacity cost amortize, converge, or grow at the full horizon, now
that the channel is demonstrably ALIVE (commit 19–40%, abort 0.0%, usage
entropy >8 bits on every one of 4980 steps)?

**Answer: it grows. The live typed channel costs ~10× on downstream
capability at this scale.** Reported as-is per the preregistration.

## Headline numbers (same silicon, same data, same 4980 steps, same tokenizer)

| | catalog (policy 6 + clip 32, live) | dense twin | 
|---|---|---|
| final train loss | 4.08 | 2.39 |
| val bpb (final / min) | 1.245 / 1.227 | — (per-step log lost; see note) |
| **CORE (centered, 22 tasks)** | **0.0151** | **0.2359** |
| s/step, 8×MI355X | 15.0 | 1.2 |
| wall | 20.8 h | 1.7 h |

CORE per task (`results/models/d20-p5/eval/base_eval_{catalog,vanilla}.csv`):
the catalog is at or below chance on essentially every task that requires
knowledge or reasoning (jeopardy 0.0005 vs 0.070; wikidata QA 0.0008 vs
0.447; lambada 0.039 vs 0.418; squad 0.009 vs 0.389; arc_easy 0.355 vs
0.663; hellaswag 0.27 vs 0.52). The only tasks where it is not at chance are
the near-format ones (piqa, language identification).

## What this is and is not

**It is** the first measurement of a typed, transactionally-gated residual
channel at nanochat scale with the channel verifiably alive for the whole
run. Every prior "cost" number we had (3.4: +2.63 nats) was a DEAD-channel
number (PHASE5-CHANNEL-COLLAPSE); this one is real.

**It is not** a subtle gap. At k=8 commits per token per write unit over a
1280-dim residual, with each commit a quantized bf16 magnitude along one of
32,768 unit vectors of dimension 20 (one slot of 64), the channel writes
≤160 scalars-worth of direction per token per layer versus the dense
twin's 1280 free dimensions — and the training loss curve plateaued at
~4.4 from step 1000 to 3000 while the twin kept descending. The model
learned format (val bpb 1.24 is respectable for byte-level LM) and did not
learn facts. The bypass paths (resid/x0 lambdas) carried the trunk.

**Interpretation for the paper (C4 stands, the capacity story is now
honest and quantified):**
1. The receipts system operated at 3.4e8 facts/step for 22 hours with
   100% receipts and zero ledger-less steps — C1/C2 at scale, delivered.
2. The channel-liveness guard, the clip, and the telemetry worked: this is
   what "the channel is alive" looks like, and we can now say the earlier
   collapse was a distinct failure mode from "the channel is too narrow".
3. The typed channel as configured (policy 6, k=8, N_f=32768, d_slot=20)
   is far too low-bandwidth to carry a language model's residual stream.
   That is the measured capacity cost: CORE 0.015 vs 0.236.
4. Wall-clock cost with exact device kernels: 12.5× (15.0 vs 1.2 s/step).

## What we do NOT claim

- We do not claim the architecture is salvageable by tuning k/N_f/d_slot;
  we have one point. The obvious next experiments (k=64, d_slot=64→
  fewer/wider slots, or catalog-as-side-channel rather than sole writer)
  are future work and stated as such.
- We do not report a dense-twin val-bpb curve: the per-step log of the
  vanilla arm died with the droplet that ran it (custody bug, since fixed
  — the poller now pulls all stage logs). CORE, computed by nanochat's
  own base_eval on the final checkpoints of both arms, is the comparison.

## SFT of the catalog model — DONE (2026-09-03, after the STE fix)

Root cause of the SFT NaN: `SFT-NAN-ROOTCAUSE.md` (clip applied on the
detached input → unclipped STE gradient, 1e7×/layer amplification). With
`hyphae_write()` clipping in-graph: SFT ran 466/466 steps, loss 3.67 → 2.81,
val bpb 0.875, channel alive (commit 20–22%, entropy 8.2–8.7 bits, 0
aborts), 100% receipts, run `nano-d20-sft`. Model in S3
(`arm-catalog/chatsft_checkpoints/d20/model_000466.pt`, hash in
`arm-catalog/sft-hash.txt`).

chat_eval (catalog SFT): **ARC-Easy 25.6%, ARC-Challenge 24.4%, MMLU
25.5%** — chance on all three (4-way). Consistent with CORE 0.015: SFT
teaches format, and there is no knowledge underneath to surface. Vanilla
SFT chat_eval was produced in the first launch (log lost with that
droplet; re-runnable from the S3 checkpoint).

## Open item (was: SFT NaN — resolved)

(resolved above) The remaining open item is the corrected-mirror RERUN of
the d20 pretraining: phase 5 trained with the STE gradient over-weighted on
clipped slots (SFT-NAN-ROOTCAUSE.md). The CORE 0.015 result stands for
that run; whether the corrected gradient changes the picture is the next
experiment.

## Custody
S3 `s3://hytorch-custody/phase5-d20-20260903/` — models,
all checkpoints, tokenizer, evals, 1.09 TB ledger. Hashes in
`artifact-hashes.txt`, verified by re-download.
