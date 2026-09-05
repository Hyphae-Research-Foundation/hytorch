#!/usr/bin/env bash
# infra/ledger-peek.sh — SAFE deep-read of the LIVE run's ledger.
#
# Lesson (2026-09-01): pausing the seam via interactive ssh nearly froze the
# run — the ssh session timed out between STOP and CONT, leaving the seam
# stopped (T) against a 300s barrier. This script ships ONE remote script
# with `trap CONT EXIT`, so the seam can never stay paused, and the pause
# window only covers the rsync delta (~seconds), never the query.
#
# Usage: ledger-peek.sh <droplet-id> <query...>
#   e.g. ledger-peek.sh 596788253 get run/nano-d20-catalog/step/00000250/STEP
#        ledger-peek.sh 596788253 last-head nano-d20-catalog 5000
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ID="${1:?droplet id}"; shift
IP=$(droplet_ip "$ID")

# Read the VOLUME snapshot the ledger-sync daemon maintains (never touches
# the live ledger or the seam — the 2026-09-01 step-293 kill taught us that
# pausing the seam from an interruptible ssh session is russian roulette).
# shellcheck disable=SC2029
timeout 300 ssh "${SSH_OPTS[@]}" -o ConnectTimeout=15 "root@$IP" '
  set -u
  test -d /mnt/hyvol/nano-hyphae || { echo "no volume snapshot" >&2; exit 1; }
  rsync -a --delete /mnt/hyvol/nano-hyphae/ /tmp/led-peek/ 2>/dev/null
  rm -f /tmp/led-peek/LOCK
  echo "snapshot age: $(cat /mnt/hyvol/stamps/ledger.last_sync 2>/dev/null || echo unknown)" >&2
  /opt/hytorch/target/release/hytorch-ledger-query /tmp/led-peek '"$*"'
'
