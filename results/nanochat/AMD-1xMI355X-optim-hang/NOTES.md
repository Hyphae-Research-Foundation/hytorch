# nanochat single-GPU MuonAdamW hang on ROCm 7.0/gfx950 (2026-08-31)

Killtest attempt #1 (1x MI355X spot, d12, eager/TORCHDYNAMO_DISABLE=1,
catalog arm): trainer wedged >20 min INSIDE optimizer.step() at step 0.

py-spy dump (process State R, GPU use 6%, no progress):
    _finish_gathers (nanochat/optim.py:426)   # torch._foreach_copy_
    step            (nanochat/optim.py:459)
    <module>        (base_train.py:606)

Path analysis: launched WITHOUT torchrun (nproc=1), so no process group;
MuonAdamW takes its single-rank shortcuts (future=None, stacked buffers,
_foreach_copy_ back). The 8-rank path on the SAME silicon/wheel ran 2000
steps of d20 the same day without issue (3.4). The hytorch barrier is not
involved (receipt for step 0 was already committed; the hang is inside
upstream optimizer code after our gate released).

Suspect: muon_step_fused/adamw_step_fused fused kernels or _foreach_copy_
on the single-rank path wedging the HIP stream. Upstream-reportable
(nanochat pin 92d63d4 + torch-2.10.0+rocm7.0 on gfx950). Not our seam.

Consequence for us: single-GPU TRAINING on ROCm is off the menu (inference
is unaffected — no optimizer). Kill-test and any small SFT jobs run on the
8x tier with torchrun, which is the phase-5 configuration anyway (testing
resume on 1x would have validated the wrong code path).

Cost of discovery: ~$4 (55 min of 1x spot).
