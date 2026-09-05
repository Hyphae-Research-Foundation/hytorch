# hytorch RUNBOOK — operate without repeating the incidents

*Written 2026-09-02 after an external review observed: "lessons are encoded
in commit messages and comments, not in executable guards; nobody but the
author can operate this without repeating the incidents." This file is the
fix for the second half; the guards listed in §3 are the fix for the first.*

## 1. Invariants (if you are about to violate one, stop)

| # | Invariant | Why (incident) |
|---|---|---|
| I1 | **Never `pkill -f <pattern>` from an ssh command whose text contains `<pattern>`.** Kill by pid file or `pgrep -x <binary>`. | Launches 2,3,5,6 (Sep 1) and the old custody-model.sh: the shell carrying the script matched its own pkill and died (rc=255, "mystery" teardowns). |
| I2 | **Never pause (SIGSTOP) the live seam from an interruptible session.** Reads go to the volume snapshot (`infra/ledger-peek.sh`); only the local sync daemon may STOP/CONT, with a `trap CONT EXIT` and a 60s cap. | INCIDENT step-293: ssh timeout orphaned a manual STOP; §5.4 correctly killed the run. INCIDENT-901: unbounded daemon pause after a 5.4GB wired step outlived the barrier. |
| I3 | **Nothing with unbounded duration may sit between the trainer and its receipt.** Barrier budget is 300–600s; every pause/IO on the seam path must be capped well below it. | Same two incidents. |
| I4 | **The ledger lives on local NVMe; the volume gets snapshots.** Never point `HYTORCH_DATA_DIR` at network block storage. | Launch 1: WAL fsync on the volume doubled step time (15.6 vs 7.4s) and the wired-step flush blew the barrier. |
| I5 | **On spot, `save_every` ≤ 250.** The launcher refuses otherwise (guard runs AFTER tier resolution). | INCIDENT-901: died at 901, first save at 1000 → full restart. |
| I6 | **A promised threshold must be executable code, not a manifest line.** Manifest `thresholds.*` are read by `seam_shim._guard_update` and kill the run. Adding a threshold = adding its reader in the same commit. | Phase-1 `codebook_min_usage_entropy_bits: 8.0` was never read; the d20 channel ran ~750 steps dead. |
| I7 | **Proposal clip comes from the manifest** (`catalog.proposal_clip`), applied inside `HyphaeWrite`. No per-bridge hardcodes. | Clip was hardcoded in one bridge, absent in model.py → phases 1/2/4a exposed. |
| I8 | **Law-0 exceptions are declared and journalized** (BYPASS fact), never omitted. | nanochat's resid/x0 lambdas mutate h outside the channel and were the bypass path of the collapse. |
| I9 | **Custody before teardown, and custody must fail loudly.** `infra/custody-model.sh` exits nonzero on any missing artifact or hash mismatch; do not tear down on a nonzero exit. | Old script swallowed errors with `\|\| true`, could produce an empty evidence file silently. |
| I10 | **Every wired step is 5.4GB of IO.** Anything that runs on the seam host at wire cadence (`HYTORCH_WIRE_EVERY`) must tolerate that burst. | Steps 100 and 900 were both wired steps and both killed a run. |

## 2. Standard operations

### Launch phase 5 (spot, DO)
```
HYTORCH_STAGES=speedrun HYTORCH_VOLUME=hytorch-p5 HYTORCH_VOLUME_GB=2000 \
HYTORCH_SAVE_EVERY=200 HYTORCH_EVAL_EVERY=250 HYTORCH_DBS=16 HYTORCH_SHARDS=170 \
HYTORCH_WIRE_EVERY=100 HYTORCH_CAPACITY_WAIT_MIN=600 HYTORCH_BARRIER_MS=600000 \
HYTORCH_MAX_RELAUNCH=40 bash infra/nanochat-supervise.sh amd8 20 -1 both
```
Then `setsid nohup bash infra/channel-health-monitor.sh &` (alerts to
`/tmp/hytorch-phase5-health.log`). Fresh run on a reused volume: add
`HYTORCH_RESET_ARMS=1` (keeps dataset/tokenizer, wipes checkpoints/ledger/stamps).

### Read the live run's ledger safely
`bash infra/ledger-peek.sh <droplet-id> get run/<run>/step/00000500/STEP`
(reads the volume snapshot; never touches the live dir).

### Preemption happened
Do nothing. The supervisor relaunches (exit 75), pins the volume's region,
polls capacity, stamps skip finished stages, the arm resumes from the last
checkpoint as a child run citing `parent_run/parent_head/resume_step`.

### Run finished (`/var/log/hytorch-arms-DONE` + all `rc=0` in
`/var/log/hytorch-arms-status`)
1. `bash infra/custody-model.sh <droplet-id> <tag>` — must exit 0.
2. Commit `results/models/<tag>/{artifact-hashes.txt,ledger-evidence.txt,*.log}` (weights are git-ignored).
3. `doctl compute droplet delete <id> --force`.
4. Volume: keep until weights are verified locally, then `bash infra/teardown.sh` (per-volume confirmation).

### Something looks wrong
- `hytorch: WARNING` lines in `hytorch-arm-catalog.log` → the preregistered guard is counting; it will kill the run itself after the window. Do not intervene mid-window.
- `SEAM-HEAD-STUCK` alert while the trainer advances → catalog arm is dead and vanilla is running; check `hytorch-arms-status`.
- Do NOT ssh in and kill/pause things "to look". Read the snapshot.

## 3. Executable guards (where each lesson lives in code)

| Lesson | Guard | File |
|---|---|---|
| channel collapse | commit-rate + usage-entropy kill, manifest thresholds | `nanochat/seam_shim.py::_guard_update` |
| proposal explosion | per-slot norm clip from manifest | `python/hytorch/model.py::clip_proposal` |
| spot without checkpoints | `spot_guard` after tier case + ladder PROC | `infra/nanochat-run.sh` |
| seam continuity across restart | rehydrate `last_c_next` on open | `crates/seam/src/store.rs::open` + test |
| missing policy in RUN_START | refuse (no toy defaults) | `crates/seam/src/main.rs` |
| self-killing pkill | pidfile / `pgrep -x` only | `nanochat-run.sh`, `custody-model.sh` |
| unbounded seam pause | 60s `timeout` + `trap CONT` in sync daemon | `nanochat-run.sh` (ledger-sync.sh) |
| silent custody failure | nonzero exit, hash verification | `infra/custody-model.sh` |
| dead catalog hidden by green DONE | per-stage rc status → exit 75 | `nanochat-run.sh` poller |
| Law-0 bypass | BYPASS fact per step | `seamclient.step_chain(bypass=)`, seam `store.rs` |

## 4. Incident index (chronological, all Sep 1–2 2026)

| Incident | Root cause | Fix commit |
|---|---|---|
| Launch 1 killed @100 | ledger on network volume; barrier 60s | 0e536b4 |
| Launches 2,3,5,6 rc=255 | `pkill -f ledger-sync` killed own shell | 54ae4b3, 827331a |
| step-293 §5.4 kill | manual seam STOP orphaned by ssh timeout | 9530a60 |
| INCIDENT-901 §5.4 kill + full restart | unbounded sync pause after wired step; save_every=1000 | b071069 |
| Channel collapse (launch 7) | proposals > cap → 0 commits → no grad; no guard | c8aa6e2, 616f63c |
| MuonAdamW 1-GPU hang (ROCm) | upstream single-rank path | documented; 8× only |
| Ghost creates (DO 422 but created) | API | detection + adopt |
| Review 2026-09-02 (this file) | promises not guards | this commit series |
