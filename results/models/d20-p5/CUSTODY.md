# Phase 5 d20 — custody record (2026-09-03)

**S3 (versioned, private):** `s3://hytorch-custody/phase5-d20-20260903/`
Account <redacted>, us-east-1. Uploaded from a mem1 box with the volume
mounted read-only; hashes computed on the volume BEFORE upload
(`artifact-hashes.txt`), catalog model re-downloaded from S3 and re-hashed:
identical.

| Artifact | sha256 | S3 key |
|---|---|---|
| catalog base model (step 4980) | `2485aa4ae4382eb9b84eaf6bd1a6e09f8d1d87c61b4fa16a628514bf3e06bcf8` | `arm-catalog/base_checkpoints/d20/model_004980.pt` |
| vanilla base model (step 4980) | `97278cf6635e09ad8333cca0eaf46551dd0a5130effab26880ffde04c82ee484` | `arm-vanilla/base_checkpoints/d20/model_004980.pt` |
| vanilla SFT model (step 466) | see artifact-hashes.txt | `arm-vanilla/chatsft_checkpoints/d20/model_000466.pt` |
| all 25 catalog checkpoints + 8 optimizer shards each | artifact-hashes.txt | `arm-catalog/base_checkpoints/d20/` |
| tokenizer | artifact-hashes.txt | `tokenizer/` |
| stamps (stage completion) | — | `stamps/` |
| catalog SFT model (step 466) | `85429a9588c09721ed693103c64f0449835e1d9dbbdf48afdc341f2f6b278949` | `arm-catalog/chatsft_checkpoints/d20/model_000466.pt` |
| ledger (918 GB incl. SFT run `nano-d20-sft`; 4980+466 steps, wire every 100) | — | `nano-hyphae/` (complete) |
| evals + stage logs | — | `logs/`, `results/models/d20-p5/eval/` |

## Run facts
- Catalog: 4980/4980 steps, 8×MI355X spot, 22h wall, ZERO preemptions,
  15.0 s/step, val bpb final 1.2449 (min 1.2275 @4500), loss 4.08.
  Channel alive every step: commit 19–40%, abort 0.0%, usage entropy >8b.
- Vanilla: 4980/4980, loss 2.39; SFT vanilla 466 steps; base_eval both done.
- Torchrun exited rc=1 AFTER saving the final checkpoint (`munmap_chunk():
  invalid pointer` in teardown) — cosmetic; checkpoint + meta intact.
- SFT catalog: DONE after STE fix (466 steps, bpb 0.875); chat_eval ChatCORE 0.0016 (chance). See SFT-NAN-ROOTCAUSE.md.

## DO volume `hytorch-p5` — DELETED 2026-09-03 18:40 UTC
after S3 held everything (1.2 TB, hashes verified). Zero DO resources remain.
