# 3.3 PASS — DDP multi-rank validated: nanochat d12 on 8×MI355X spot (2026-08-31)

**The #1 remaining technical risk is retired: the collective seam works on
8 ranks.** torchrun --nproc_per_node=8, both arms, 300 steps, TBS=262144.

| | catalog | vanilla |
|---|---|---|
| loss 0 → 300 | 10.397 → 4.739 | 10.397 → 3.754 |
| receipts | 300/300 (192 frames/step = 8 ranks × 12 layers × 2 units) | — |
| wall | ~6.2 min | ~1.5 min |
| facts/step | 50.3M | — |

Multi-rank mechanics verified: every rank spooled its frames
(suffix -rRR-mMMM), rank 0 ran the seam + committed the step transaction,
the head broadcast gated all 8 optimizer.step()s, codebook chain continuous
(c_prev/c_next across DDP-averaged updates).

## The five failed attempts (all fixed, all committed)

1. tiktoken missing → wrong python targeted on the 2-python AMD image
   → single-$PY installs + fail-fast import check + pipefail.
2. SSH timeout at 5-min cap → 8x nodes boot slower → 20-min cap.
3. `-m: command not found` → unescaped inner quotes in the ssh heredoc
   → escaped + remote block now lint-simulated locally before launch.
4. pip RECORD error on debian-owned typing_extensions → --ignore-installed.
5. **NaN in BOTH arms from step 2** → isolated on a 1× MI355X (~$2):
   torch.compile/Inductor on ROCm 7.0/gfx950 produces silent NaNs (bf16 AND
   fp32; Muon/AdamW fused steps are @torch.compile'd) → TORCHDYNAMO_DISABLE=1
   on ROCm for both arms. Eager trains clean. Upstream-reportable finding.
6. (attempt 6 infra) 2/5 8× spots came up SSH-dead → create→verify→recreate
   loop (up to 3×).

## Cost of the validation: ~$25 total across 6 attempts. 3.4 unblocked.
