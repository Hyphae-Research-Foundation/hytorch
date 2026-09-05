# 3.2 — nanochat d12, both arms, H100, 1500 steps (2026-08-31)

Same tree, same data (24 fineweb shards, nanochat tokenizer), same TBS
(65 536 tok/step, dbs=16, grad_accum reduced 8× from nanochat's auto value
for journal tractability — SAME in both arms), 1500 steps.

## Result (train loss trajectory; no bpb eval in this signal run)

| step | catalog | vanilla | gap (nats) |
|---|---|---|---|
| 300 | 5.275 | 4.431 | +0.845 |
| 600 | 4.904 | 3.772 | +1.132 |
| 900 | 4.724 | 3.579 | +1.145 |
| 1200 | 4.634 | 3.451 | +1.184 |
| 1499 | **4.485** | **3.302** | **+1.183** |

Wall: catalog 107 min vs vanilla 4.1 min (26×). Ledger: 17 GB (wire
retained every 50 steps; T2 heads all 1500 steps), 12.58M facts/step,
1500/1500 receipts, chain continuous (verified by ledger query: RUN_START,
RECEIPT@1499 head 9c44ee…, STEP@1499 c_prev→c_next).

## Gap slope across scale (the question 3.2 existed to answer)

| depth | params | steps | catalog loss | vanilla loss | gap |
|---|---|---|---|---|---|
| d4 | 3.5M | 60 | 5.95 | 5.92 | +0.03 |
| d12 | 124M | 1500 | 4.485 | 3.302 | **+1.18** |

The gap GROWS with scale/horizon at fixed k=8: the vanilla model uses its
full 768-dim dense residual; the catalog writes k×d_slot = 96 of 768 dims
per unit (12.5%), quantized. At d4/60 steps neither model had learned enough
for the channel to bind; at d12/1500 the dense twin pulls away. Consistent
with the phase-1b lrsweep reversal: the typed channel costs real capacity
at speaking-model scale with phase-1 policy (global_topk, k=8, N_f=32k).

## Wall breakdown (why 26× — and why it is NOT the ledger)

~4.3 s/step catalog vs 0.16 s/step vanilla. The seam commits 48 frames/step
in well under a second (group commit); the dominant costs are (a) eager
execution (no torch.compile: dynamo cannot trace our pybind kernels),
(b) pack's top-k sweep per write unit (2 units × N_f=32k × d_slot scores
per token), (c) per-layer pinned D2H of the verdict stream. All engineering,
none fundamental; fused CUDA-graph-compatible kernels are the known fix.

## Decision for 3.3/3.4 (recorded before running them)

The d20 run's PRIMARY claims are C1/C2 at scale (full receipts on a model
that talks, 8×MI355X DDP, cross-vendor) and the phase-4 instrument. The
capacity gap is now an expected, measured cost — the d20 catalog arm will
be worse at CORE than vanilla; we run it anyway because the artifact's
value was never "wins PPL": it is "every state transition of a speaking
model, receipt-backed". Threshold for 3.4 stays honest: report CORE both
arms + the full cost table; no win claimed.
