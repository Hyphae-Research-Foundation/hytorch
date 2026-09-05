# infra/ — operating runs

Scripts for DigitalOcean (spot GPUs via `doctl`) and AWS (S3 custody). Read
[`RUNBOOK.md`](RUNBOOK.md) first: it lists the invariants (I1–I10) and the incidents that
produced them.

| script | purpose |
|---|---|
| `provision.sh`, `cloud-init.yml`, `common.sh` | create a droplet with the pinned toolchain |
| `capture-build.sh` | fill the manifest's `build` facts (apply_ref hash, harness commit, torch wheel) |
| `gate-diff.sh` | run the 200k-case differential gate on the target silicon |
| `train-run.sh`, `multi-run.sh`, `sweep.sh`, `ladder.sh` | phase 1–2 runs, LR/seed sweeps |
| `nanochat-run.sh`, `nanochat-supervise.sh`, `watchdog.sh`, `channel-health-monitor.sh` | phase 3–5 nanochat runs on 8×MI355X with save cadence, spot guard and the dead-channel monitor |
| `sft-probe.sh`, `probe-run.sh`, `hallu-run.sh` | SFT, semantic probe and phase-4 drivers |
| `ledger-peek.sh` | read the volume snapshot, never the live ledger |
| `custody-model.sh` | hash-verified custody to S3; exits nonzero on any gap |
| `teardown.sh` | destroy droplet and volume (after custody) |
| `ci-remote.sh` | the six CI suites on a fresh droplet |
