#!/usr/bin/env bash
# infra/ladder.sh — the procurement ladders (user directive, encoded):
#   NVIDIA citable runs:  B300 spot → B300 LC spot → H200 on-demand → H100 on-demand
#   AMD    citable runs:  MI355X spot → MI350X spot → MI300X on-demand
#   DEV (kernels/smoke):  RTX 4000 Ada → L40S
#   CPU (CI/verify):      c-8 → c-16
#
# Usage: ladder.sh <nv|amd|dev|cpu> <name> [role-tag]
# Prints "<droplet-id> <size> <region>" of the first rung that provisions.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TIER="${1:?tier: nv|amd|dev|cpu}" NAME="${2:?droplet name}" ROLE="${3:-run}"

# HYTORCH_FORCE_SIZE overrides the ladder with one exact size (controlled
# experiments need fixed silicon; genesis is not comparable across chips).
if [ -n "${HYTORCH_FORCE_SIZE:-}" ]; then
  case "$TIER" in
    amd) IMAGE="gpu-amd-base" ;;
    cpu) IMAGE="ubuntu-24-04-x64" ;;
    *)   IMAGE="gpu-h100x1-base" ;;
  esac
  SIZES=("$HYTORCH_FORCE_SIZE")
else
case "$TIER" in
  nv)  SIZES=(gpu-b300x1-288gb-spot gpu-b300x1-288gb-lc-spot gpu-h200x1-141gb gpu-h100x1-80gb)
       IMAGE="gpu-h100x1-base" ;;
  amd) SIZES=(gpu-mi355x1-288gb-spot gpu-mi350x1-288gb-spot gpu-mi300x1-192gb)
       IMAGE="gpu-amd-base" ;;
  dev) SIZES=(gpu-4000adax1-20gb gpu-l40sx1-48gb gpu-h100x1-80gb)
       IMAGE="gpu-h100x1-base" ;;
  cpu) SIZES=(c-8 c-16)
       IMAGE="ubuntu-24-04-x64" ;;
  *) echo "unknown tier $TIER" >&2; exit 2 ;;
esac
fi

for size in "${SIZES[@]}"; do
  for region in $(regions_for "$size"); do
    log "trying $size in $region…"
    if id=$(try_create "$NAME" "$size" "$IMAGE" "$region" "hytorch-$ROLE"); then
      procurement="ondemand"; [[ "$size" == *-spot ]] && procurement="spot"
      echo "$id $size $region $procurement"
      # Record for the watchdog + the infra.* manifest fields (SPEC_AMEND).
      printf '{"id":"%s","size":"%s","region":"%s","procurement":"%s","tier":"%s","name":"%s","ts":"%s"}\n' \
        "$id" "$size" "$region" "$procurement" "$TIER" "$NAME" "$(date -u +%FT%TZ)" \
        >> "$STATE_DIR/ladder.jsonl"
      exit 0
    fi
  done
done
log "LADDER EXHAUSTED for tier=$TIER — no capacity on any rung"
exit 1
