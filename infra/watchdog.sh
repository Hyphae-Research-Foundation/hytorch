#!/usr/bin/env bash
# infra/watchdog.sh — runs on the T-CTL droplet (or anywhere with doctl).
# Polls run droplets; if a spot instance vanishes (preemption == destroy in
# DO semantics), relaunches via the ladder and signals a CHILD RUN (the spec's
# §5.4 resumption: a child manifest citing the last committed head).
#
# Phase-1 scope: detection + relaunch + marker file. The training harness
# reads the marker and starts from the last checkpoint as a child run.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WATCH_FILE="$STATE_DIR/watch.jsonl"   # lines: {"id":…,"tier":…,"name":…,"commit":…}
POLL_S="${HYTORCH_WATCH_POLL:-60}"

log "watchdog: polling every ${POLL_S}s over $WATCH_FILE"
while true; do
  [ -f "$WATCH_FILE" ] || { sleep "$POLL_S"; continue; }
  while IFS= read -r line; do
    id=$(echo "$line" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
    tier=$(echo "$line" | python3 -c 'import json,sys;print(json.load(sys.stdin)["tier"])')
    name=$(echo "$line" | python3 -c 'import json,sys;print(json.load(sys.stdin)["name"])')
    commit=$(echo "$line" | python3 -c 'import json,sys;print(json.load(sys.stdin)["commit"])')
    if ! doctl compute droplet get "$id" --format Status --no-header >/dev/null 2>&1; then
      log "PREEMPTED/GONE: $name ($id). Relaunching via ladder tier=$tier…"
      child="${name}-child-$(date -u +%s)"
      if out=$("$(dirname "${BASH_SOURCE[0]}")/ladder.sh" "$tier" "$child" run); then
        new_id=$(echo "$out" | cut -d' ' -f1)
        "$(dirname "${BASH_SOURCE[0]}")/provision.sh" "$new_id" "$commit"
        # Marker consumed by the harness: start as child run from last checkpoint.
        printf '{"parent":"%s","child_droplet":"%s","ts":"%s"}\n' \
          "$name" "$new_id" "$(date -u +%FT%TZ)" >> "$STATE_DIR/child-runs.jsonl"
        # Replace watch entry.
        grep -v "\"id\": *\"$id\"" "$WATCH_FILE" > "$WATCH_FILE.tmp" || true
        printf '{"id":"%s","tier":"%s","name":"%s","commit":"%s"}\n' \
          "$new_id" "$tier" "$child" "$commit" >> "$WATCH_FILE.tmp"
        mv "$WATCH_FILE.tmp" "$WATCH_FILE"
      else
        log "ladder exhausted for $name; will retry next poll"
      fi
    fi
  done < "$WATCH_FILE"
  sleep "$POLL_S"
done
