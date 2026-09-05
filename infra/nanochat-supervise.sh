#!/usr/bin/env bash
# infra/nanochat-supervise.sh — phase-5 supervisor: relaunch-on-preemption.
#
# Wraps nanochat-run.sh: exit 75 (droplet vanished = spot preemption) or a
# transient failure retriggers a relaunch; the volume carries all state, so
# every relaunch resumes (stamps + checkpoints + cited child run §5.4).
# Validated end-to-end by the 2026-08-31 kill-test (see NOTES).
#
# Usage: same args/env as nanochat-run.sh, plus:
#   HYTORCH_MAX_RELAUNCH  max relaunches (default 20)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
set +e   # common.sh sets -e; the whole point here is to OUTLIVE failures

MAX="${HYTORCH_MAX_RELAUNCH:-20}"
: "${HYTORCH_VOLUME:?supervisor requires volume mode (HYTORCH_VOLUME)}"

FAST_FAILS=0
for launch in $(seq 1 "$MAX"); do
  log "supervisor: launch $launch/$MAX"
  T0=$(date +%s)
  "$(dirname "${BASH_SOURCE[0]}")/nanochat-run.sh" "$@"
  rc=$?
  ELAPSED=$(( $(date +%s) - T0 ))
  # forensics: keep every attempt's create.err (was overwritten per attempt)
  [ -f "$STATE_DIR/create.err" ] && cp "$STATE_DIR/create.err" "$STATE_DIR/create.err.$launch.$(date -u +%H%M%S)" 2>/dev/null || true
  # circuit breaker (review 2026-09-02): 7 relaunches in 35 min at $36/h
  # happened once. Three consecutive failures under 10 minutes means a
  # deterministic bug, not preemption — stop and let a human read the log.
  if [ "$rc" -ne 0 ] && [ "$ELAPSED" -lt 600 ]; then
    FAST_FAILS=$((FAST_FAILS + 1))
    if [ "$FAST_FAILS" -ge 3 ]; then
      log "supervisor: CIRCUIT BREAKER — $FAST_FAILS consecutive failures under 10 min (rc=$rc). Not a preemption pattern; stopping."
      exit 3
    fi
  else
    FAST_FAILS=0
  fi
  # An adopted droplet is gone after its run: relaunches go via the ladder.
  unset HYTORCH_ADOPT_ID
  if [ "$rc" -eq 0 ]; then
    log "supervisor: run completed (launch $launch)"
    exit 0
  fi
  log "supervisor: run exited rc=$rc; relaunching in 60s (state on volume)"
  sleep 60
done
log "supervisor: gave up after $MAX launches"
exit 1
