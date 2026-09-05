#!/usr/bin/env bash
# infra/custody-model.sh — pull trained model artifacts + ledger evidence
# from a live nanochat droplet BEFORE teardown. The model is the deliverable
# (user directive): weights custodied locally, hashes committed to git.
#
# Usage: custody-model.sh <droplet-id> <tag> [run-id]   (tag e.g. d20-p5)
#
# Review fix (2026-09-02): the old version ran `pkill -f hytorch-seam` inside
# an ssh command whose own argv contained "hytorch-seam" — it killed its own
# shell (same class as INCIDENT-901 pkill), and swallowed the failure with
# `2>/dev/null || true`, producing an EMPTY evidence file silently, in the
# one script that extracts the deliverable. Now: kill by pid (pgrep -x on
# the binary NAME, which never matches a bash shell), evidence via the
# shipped hytorch-ledger-query binary (no ad-hoc crate), and every stage
# FAILS LOUDLY — the script exits nonzero and prints what is missing.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ID="${1:?droplet id}"; TAG="${2:?tag}"; RUN_ID="${3:-}"
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
IP=$(droplet_ip "$ID")
OUT="$REPO_ROOT/results/models/$TAG"
mkdir -p "$OUT"
FAILS=0
fail() { log "CUSTODY FAILURE: $*"; FAILS=$((FAILS + 1)); }

log "custody: hashing + pulling model artifacts from $ID → $OUT"

# 1. Hash everything on the droplet FIRST (EXPORT-grade evidence even if the
#    transfer dies), then pull. Arm dirs may be symlinks into the volume.
hyssh "root@$IP" '
for arm in catalog vanilla; do
  for d in /var/lib/hytorch/arm-$arm/base_checkpoints/*/ /var/lib/hytorch/arm-$arm/chatsft_checkpoints/*/; do
    [ -d "$d" ] || continue
    for f in "$d"model_*.pt "$d"meta_*.json; do
      [ -f "$f" ] && sha256sum "$f"
    done
  done
done
for f in /var/lib/hytorch/nanochat-cache/tokenizer/*; do
  [ -f "$f" ] && sha256sum "$f"
done' > "$OUT/artifact-hashes.txt"
[ -s "$OUT/artifact-hashes.txt" ] || fail "artifact-hashes.txt is empty (no checkpoints found?)"
cat "$OUT/artifact-hashes.txt"

# 2. Pull: LAST checkpoint per arm (+SFT if present), tokenizer, logs, seam stderr.
SCP=(scp -i "$SSH_ID" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
for arm in catalog vanilla; do
  for stage in base_checkpoints chatsft_checkpoints; do
    LAST=$(hyssh "root@$IP" "ls /var/lib/hytorch/arm-$arm/$stage/*/model_*.pt 2>/dev/null | sort | tail -1" || true)
    [ -n "$LAST" ] || continue
    D=$(dirname "$LAST"); STEP=$(basename "$LAST" | sed 's/model_//;s/\.pt//')
    DEST="$OUT/$arm/$stage/$(basename "$D")"
    mkdir -p "$DEST"
    "${SCP[@]}" "root@$IP:$D/model_${STEP}.pt" "root@$IP:$D/meta_${STEP}.json" "$DEST/" \
      || fail "pull $arm/$stage step $STEP"
    # verify against the on-droplet hash
    for f in "model_${STEP}.pt" "meta_${STEP}.json"; do
      REMOTE=$(grep " $D/$f\$" "$OUT/artifact-hashes.txt" | cut -d' ' -f1)
      LOCAL=$(sha256sum "$DEST/$f" | cut -d' ' -f1)
      [ "$REMOTE" = "$LOCAL" ] || fail "hash mismatch $arm/$stage/$f ($REMOTE vs $LOCAL)"
    done
    log "custodied $arm/$stage step $STEP (hash verified)"
  done
done
mkdir -p "$OUT/tokenizer"
"${SCP[@]}" -r "root@$IP:/var/lib/hytorch/nanochat-cache/tokenizer/." "$OUT/tokenizer/" || fail "tokenizer pull"
"${SCP[@]}" "root@$IP:/var/log/hytorch-*.log" "$OUT/" || fail "logs pull"
"${SCP[@]}" "root@$IP:/run/hytorch/nano-spool/seam.stderr" "$OUT/seam.stderr" 2>/dev/null || log "note: no seam.stderr (seam not running on this box)"

# 3. Ledger evidence pack via the shipped query binary against a CONSISTENT
#    copy (never the live dir with a LOCK; never pause the seam by hand —
#    stop it by pid only if this is post-DONE custody).
if [ -z "$RUN_ID" ]; then
  RUN_ID=$(hyssh "root@$IP" 'cat /mnt/hyvol/stamps/catalog.current_run 2>/dev/null || echo nano-d20-catalog')
fi
hyssh "root@$IP" '
set -u
Q=/opt/hytorch/target/release/hytorch-ledger-query
[ -x "$Q" ] || { echo "MISSING ledger-query binary" >&2; exit 3; }
# seam still alive? it must be DONE by now — stop by pid (never pkill -f).
for p in $(pgrep -x hytorch-seam); do kill "$p"; done; sleep 1
SRC=/var/lib/hytorch/nano-hyphae
[ -d /mnt/hyvol/nano-hyphae ] && [ -n "$(ls -A /mnt/hyvol/nano-hyphae 2>/dev/null)" ] && \
  rsync -a --delete "$SRC/" /mnt/hyvol/nano-hyphae/ 2>/dev/null || true
rm -rf /tmp/led-custody; cp -r "$SRC" /tmp/led-custody; rm -f /tmp/led-custody/LOCK
RUN='"$RUN_ID"'
echo "== run/$RUN/RUN_START"; $Q /tmp/led-custody get "run/$RUN/RUN_START"
for pid in 6 7; do $Q /tmp/led-custody get "run/$RUN/POLICY/$pid" 2>/dev/null && echo "== POLICY/$pid" ; done
LH=$($Q /tmp/led-custody last-head "$RUN" 60000); echo "== last-head: $LH"
S=$(echo "$LH" | cut -d" " -f1)
for k in RECEIPT STEP; do echo "== step/$S/$k"; $Q /tmp/led-custody get "run/$RUN/step/$(printf %08d "$S")/$k"; done
' > "$OUT/ledger-evidence.txt" || fail "ledger evidence query"
grep -q "RUN_START" "$OUT/ledger-evidence.txt" && grep -q "last-head: [0-9]" "$OUT/ledger-evidence.txt" \
  || fail "ledger-evidence.txt incomplete (see $OUT/ledger-evidence.txt)"

if [ "$FAILS" -gt 0 ]; then
  log "CUSTODY INCOMPLETE: $FAILS failure(s). DO NOT TEAR DOWN."
  exit 1
fi
log "custody complete → $OUT (weights local-only; hashes + evidence go to git)"
du -sh "$OUT"
