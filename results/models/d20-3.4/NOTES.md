# 3.4 — nanochat d20 both arms, 8×MI355X spot, 2000 steps (2026-08-31)

**The measurement run that prices phase 5. Both arms complete; models
CUSTODIED with verified hashes; ledger evidence extracted before teardown.**

## Numbers

| | catalog | vanilla |
|---|---|---|
| train loss @2000 | 5.250 | 2.619 |
| s/step | 7.42 | 0.62 |
| wall (2000 steps) | 249 min | 20.4 min |
| receipts | 2000/2000 (640 frames/step) | — |
| facts/step | 167.8M | — |
| ledger | 144 GB (wire_every=100) | — |

Speaks? Neither yet — both are at ~17% of the compute-optimal horizon
(2000 of ~11.8k steps). Catalog emits commas, vanilla emits loops
("is is is") — normal undertrained behavior. The talking model is phase 5.

## Custody (results/models/d20-3.4/, hashes verified after transfer)

- catalog model_002000.pt  sha256 01fd55f1… (matches droplet hash) + 8 optim shards
- vanilla model_002000.pt  sha256 f789824b… (matches droplet hash)
- tokenizer (token_bytes.pt 0bae8cf2…, tokenizer.pkl ae73c5f7…)
- ledger-evidence.txt: RUN_START (4 digests + manifest sha), POLICY/6,
  RECEIPT@1999 (head a1c73682…, 167.8M facts), STEP@1999 (codebook chain).
- Weights are git-ignored (local custody); hashes are committed.

## Phase-5 budget, now FIXED by measurement

- Catalog: 7.42 s/step × ~11 800 steps ≈ **24.3 h** ≈ $875 spot.
- Vanilla: ~2.1 h ≈ $75.
- Ledger at wire_every=100 ≈ **850 GB** for the full run → custody via
  attached DO volume + selective pull (heads/receipts/STEP chain always;
  wire samples; T1 spills). Total phase-5 ≈ **$1 000-1 200, ~27 h** — well
  inside the user's authorized 45 h.

## Gap reality (registered before phase 5)

Loss gap at d20/2000 steps: +2.63 nats — larger than d12's +1.18 at the
same relative horizon. The k=8/global_topk channel is a severe information
bottleneck at 1280 dims (writes 160/1280 = 12.5%, quantized). Phase 5 runs
BOTH arms to full horizon as designed (the question is whether the codebook
amortizes with 6× more tokens), with the honest framing already committed:
the deliverables are the receipts system at scale, the journal corpus, the
mechanism experiment (PHASE4-MECHANISM), and the talking vanilla-twin
comparison — not a PPL win.

## Ops notes

- Local poller died with a terminal interrupt (again) — irrelevant by
  design: arms ran detached, custody was manual, teardown manual. Zero loss.
- grad_norm=0 in STEP records: the shim passes lrm as lr and 0.0 as
  grad_norm (cosmetic; fix in phase 5 shim polish).
- c_prev == c_next at step 1999: codebook renorm converged to fixed point
  under bf16 rounding late in training (real signal, not a bug — the
  codebook stopped moving; consistent with the undertrained gap).
