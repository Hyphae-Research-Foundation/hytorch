# results/ — custody of every result

Everything the paper cites lives here or in S3 with a sha256 list here. Prose in this
tree is CC BY-SA 4.0.

| path | what |
|---|---|
| `gates/<silicon>/` | differential-gate outputs per architecture; `gates/AWS-Trainium2/NOTES.md` has the eight Neuron findings |
| `inject/inject-summary.json` | fault injection, 36/36 |
| `train/`, `sweep/`, `gpu-rehearsal/`, `ci/` | phase 1–2 runs (NVIDIA H100/H200/RTX, AMD MI355X), LR/seed sweeps, CI |
| `nanochat/` | phase 3–5 nanochat runs incl. `PHASE5-CHANNEL-COLLAPSE.md` and `KILLTEST-PASS.md` |
| `models/d20-p5/` | the phase-5 d20 result (`RESULT.md`, CORE per task, SFT root cause, custody); weights and ledger in `s3://hytorch-custody/phase5-d20-20260903/` |
| `models/d20-3.4/` | the dead-channel d20 (notes only; weights git-ignored) |
| `trn2-d10/` | Trainium2 d10 catalog vs twin (`RESULT.md`, `curves.csv`, manifests, launchers, sha256 list); ledgers, logs and both checkpoints in `s3://hytorch-custody/trn2-d10-20260904/` |
| `probe/` | semantic probe |
| `hallu/` | phase-4a smoke only |
