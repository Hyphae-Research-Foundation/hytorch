#!/usr/bin/env bash
# infra/ci-remote.sh — official CI: ephemeral CPU droplet, full workspace
# tests, results archived to results/ci/, droplet destroyed. The laptop is
# control plane and custody; the droplet is the lab (user directive).
#
# Usage: ci-remote.sh [commit-sha]   (default: HEAD)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
COMMIT="${1:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
SHORT="${COMMIT:0:12}"
NAME="hytorch-ci-$SHORT-$(date -u +%H%M%S)"
OUT_DIR="$REPO_ROOT/results/ci/$SHORT"
mkdir -p "$OUT_DIR"

log "CI for $COMMIT on ephemeral CPU droplet…"
read -r ID SIZE REGION PROC < <("$(dirname "${BASH_SOURCE[0]}")/ladder.sh" cpu "$NAME" ci)
trap 'log "teardown $ID"; doctl compute droplet delete "$ID" --force || true' EXIT

"$(dirname "${BASH_SOURCE[0]}")/provision.sh" "$ID" "$COMMIT" > "$OUT_DIR/build-facts.json"

IP=$(droplet_ip "$ID")
hyssh "root@$IP" 'cd /opt/hytorch && cargo test --workspace --release 2>&1' \
  | tee "$OUT_DIR/cargo-test.log"

# Verifier CLI smoke on the droplet (same-binary path).
hyssh "root@$IP" 'cd /opt/hytorch && ./target/release/hytorch-verify /dev/null 2>&1; echo "exit=$?"' \
  | tail -2 >> "$OUT_DIR/cargo-test.log" || true

# Python suites — ALL of them (review 2026-09-02: only 2/6 ran in CI; the
# backward, codebook reset, runtime, two-phase and fault injection were
# hand-run only). torch (CPU wheel) is required for 4 of the 6.
hyssh "root@$IP" '
  set -e
  cd /opt/hytorch
  PIPQ="python3 -m pip install -q --break-system-packages"
  $PIPQ --help >/dev/null 2>&1 || PIPQ="python3 -m pip install -q"
  $PIPQ numpy 2>&1 | tail -1 || true
  python3 -c "import torch" 2>/dev/null || $PIPQ torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1
  fail=0
  for t in test_spill_roundtrip test_e2e_seam test_inject test_runtime test_two_phase probe_semantics; do
    echo "=== $t"
    if timeout 1800 python3 python/tests/$t.py; then echo "=== $t PASS"; else echo "=== $t FAIL"; fail=1; fi
  done
  exit $fail
' | tee "$OUT_DIR/python-tests.log"

{
  echo "{"
  echo "  \"commit\": \"$COMMIT\","
  echo "  \"droplet\": {\"size\": \"$SIZE\", \"region\": \"$REGION\", \"procurement\": \"$PROC\"},"
  echo "  \"finished_at\": \"$(date -u +%FT%TZ)\","
  echo "  \"result\": \"$(grep -c 'test result: ok' "$OUT_DIR/cargo-test.log" || true) suites ok\""
  echo "}"
} > "$OUT_DIR/ci-summary.json"

log "CI done → $OUT_DIR"
