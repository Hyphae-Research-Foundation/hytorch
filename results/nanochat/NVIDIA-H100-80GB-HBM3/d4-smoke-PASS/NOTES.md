# 3.1 SMOKE PASS — nanochat d4 + hytorch catalog, H100 (2026-08-30)

**Both arms trained end-to-end on the same patched tree.**

| | catalog (HYTORCH_CATALOG=1) | vanilla (flag off) |
|---|---|---|
| 60 steps, d4, dbs=8, grad_accum=16 | loss 10.40 → 5.95 | loss 10.40 → 5.92 |
| wall | 4.45 min | 0.13 min |
| ledger | 45 GB (60 steps) | — |

**What the smoke proved:**
- The law-0 patch + bridge + seam shim work against REAL nanochat (pin
  92d63d4): Muon+AdamW optimizer, grad-accum 16, torch.compile off for the
  catalog arm (dynamo can't trace pybind kernels), bf16 training.
- Every optimizer.step() was receipt-gated (128 frames/step: 16 micro ×
  4 layers × 2 write units); codebook chain continuous; RUN_START committed
  with real digests.
- Loss parity at d4/60 steps: 5.95 vs 5.92 — the catalog does not break
  learning at smoke scale.

**Bugs found and fixed BY this smoke (its job):**
1. rustbpe/psutil deps missing → added.
2. `--` separator rejected by `python -m` (torchrun-only) → SEP conditional.
3. attach_codebook broke nanochat's num_scaling_params assert → moved after
   setup_optimizer.
4. dynamo graph-breaks on pybind kernels OOM'd 20GB Ada → catalog arm eager.
5. bpb eval at default eval_tokens OOMs small cards → eval knob.
6. **STE backward dtype mismatch under bf16 autograd** (fp32 mirror now).
7. **DuplicateDocumentKey: multi-frame (grad-accum) steps collided in the
   atomic batch** → LAYER key now includes frame index.

**Scaling notes for 3.2+ (why the 34× wall gap will collapse):**
- d4 = 3.5M params: the model is ~nothing, the ledger is everything.
  bytes/step = 128 frames × 2.1 MB = 268 MB/step (no elision at this LR yet).
- At d20 (8×GPU, dbs=16): compute grows ~100×, ledger/step grows ~4×
  (fewer grad-accum steps at 8 ranks), and the toy showed 9.5%/3.3% ledger
  wall at 124M scale.
- Real lever for 3.2: journal_sample_rate or per-unit k — preregister if
  needed; the smoke ran FULL fidelity (every fact of every microbatch).
