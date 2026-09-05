# SFT NaN root cause (2026-09-03) — and what it means for the phase-5 pretraining

## Symptom
Catalog SFT: loss NaN at step 1, abort 100%. Pretraining never NaN'd.

## Bisect (on the 8×MI355X, real d20 checkpoint, plain fwd+bwd, no optimizer)
- forward finite through all 20 blocks
- backward: grad magnitude by layer 19→0 with catalog ON:
  `4e-4, 6e5, 2e13, 2e21, 4e24, 1e32, 3e34 … nan` — **~1e7× amplification per
  layer**, inf by layer 13
- same model, catalog OFF (x+delta): `1e-10 … 2e-8` — sane
- catalog ON with grad_delta zeroed (only grad_h passthrough): sane
→ the amplification is entirely in the STE mirror's `grad_delta`.

## Root cause
`clip_proposal` was applied INSIDE `HyphaeWrite.forward` on the DETACHED
input. Autograd therefore never saw the clip: the mirror returns
`grad_delta` w.r.t. the *clipped* proposal, but it flows straight into the
*unclipped* block output. A proposal at 400× the clip (common at d20 —
that is why the clip exists) receives a gradient 400× too large; compounded
over 40 write units it overflows bf16.

Pretraining did not NaN because the base model's proposals grew INTO the
clip gradually under warmup (the guard shows commit 19–40% throughout), so
the amplification stayed bounded; SFT starts from the trained model whose
proposals sit far above the clip on the new data distribution → instant
overflow. Note this also means **the phase-5 pretraining trained with a
mis-scaled STE gradient on every clipped slot** — the channel's writes
were correct (facts are facts) but the learning signal through the channel
was systematically overweighted for over-clip proposals. The measured
CORE 0.015 stands as the result of that run; a rerun with the corrected
mirror is the obvious next experiment and we say so in the paper.

## Fix (commit this series)
`hyphae_write()` is the only entry point: it applies `clip_proposal`
IN-GRAPH (Jacobian reaches the proposing block), then calls the Function.
`HyphaeWrite.forward` no longer clips. Bridges (nanochat, trainium) and the
toy CatalogedTransformer all route through `hyphae_write`. Verified on the
real d20 model: backward magnitudes 4e-4 → 6e-5 across 20 layers, no
non-finite anywhere. Host gates (two_phase, runtime, inject) green.

## Lesson (RUNBOOK I11)
Any transformation of a proposal that the policy JUDGES must also be a
transformation autograd SEES. "Detached-norm scale" is fine for the scale
factor's own gradient; hiding the whole op from autograd is not.
