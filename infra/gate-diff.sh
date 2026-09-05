#!/usr/bin/env bash
# infra/gate-diff.sh — M3 gate: run the differential harness on a GPU droplet.
# The gate is PER-ARCHITECTURE: passing on Ada does not authorize Hopper or
# Blackwell (plan M3). Results land in results/gates/<arch>/<commit>/.
#
# Usage: gate-diff.sh <nv|amd|dev> [commit] [iters]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TIER="${1:?tier nv|amd|dev}"
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
COMMIT="${2:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
ITERS="${3:-200000}"
SHORT="${COMMIT:0:12}"
NAME="hytorch-gate-$TIER-$(date -u +%H%M%S)"

log "M3 gate: tier=$TIER commit=$SHORT iters=$ITERS"
read -r ID SIZE REGION PROC < <("$(dirname "${BASH_SOURCE[0]}")/ladder.sh" "$TIER" "$NAME" gate)
trap 'log "teardown $ID"; doctl compute droplet delete "$ID" --force || true' EXIT

"$(dirname "${BASH_SOURCE[0]}")/provision.sh" "$ID" "$COMMIT" >/dev/null
IP=$(droplet_ip "$ID")

ARCH=$(hyssh "root@$IP" 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr " " "-" || echo unknown')
OUT_DIR="$REPO_ROOT/results/gates/$ARCH/$SHORT"
mkdir -p "$OUT_DIR"

hyssh "root@$IP" "
  set -e
  cd /opt/hytorch
  export PATH=\$PATH:/usr/local/cuda/bin
  nvcc --version | tail -1
  nvcc -O2 -fmad=false -arch=native kernels/cuda/diff_harness.cu kernels/cuda/refkernels.cu -o /tmp/diff_harness -ldl
  /tmp/diff_harness /opt/hytorch/target/release/libapply_ref.so $ITERS
" | tee "$OUT_DIR/diff-harness.log"

cp "$STATE_DIR/ladder.jsonl" "$OUT_DIR/procurement.jsonl" 2>/dev/null || true
{
  echo "{"
  echo "  \"commit\": \"$COMMIT\","
  echo "  \"gpu\": \"$ARCH\","
  echo "  \"droplet\": {\"size\": \"$SIZE\", \"region\": \"$REGION\", \"procurement\": \"$PROC\"},"
  echo "  \"iters\": $ITERS,"
  echo "  \"verdict\": \"$(grep -o 'BIT_IDENTICAL\|BACKEND_REJECTED' "$OUT_DIR/diff-harness.log" | tail -1)\","
  echo "  \"finished_at\": \"$(date -u +%FT%TZ)\""
  echo "}"
} > "$OUT_DIR/gate-summary.json"
log "gate done → $OUT_DIR"
