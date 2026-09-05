#!/usr/bin/env bash
# infra/channel-health-monitor.sh — continuous internal-state watch (phase 5).
#
# User directive (2026-09-01): "tienes que estar verificando el estado
# interno... a medida que los pasos avancen tienes que ver el log que deja
# Hyphae, solo de esa forma sabemos si lo estamos haciendo bien."
#
# Every CHECK_S seconds, for whatever droplet currently runs the arms:
#   - last channel telemetry line (commit/overflow/abort mix from the shim)
#   - last trainer step line + last val bpb
#   - last seam stderr line (receipts still flowing, heads advancing)
#   - ALERT if: WARNING in telemetry, commit < 5%, abort > 5%, or the seam
#     head stopped advancing while the trainer moved.
#
# Appends to $OUT (default /tmp/hytorch-phase5-health.log); prints alerts
# to stderr too. Exits only when killed.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
set +e

CHECK_S="${HYTORCH_HEALTH_EVERY_S:-600}"
OUT="${HYTORCH_HEALTH_LOG:-/tmp/hytorch-phase5-health.log}"
NAME_RE="${HYTORCH_HEALTH_NAME:-nano-d20}"

LAST_HEAD=""
LAST_STEP=""

while true; do
  TS=$(date -u +%FT%TZ)
  ID=$(doctl compute droplet list --format ID,Name --no-header 2>/dev/null | awk -v re="$NAME_RE" '$2 ~ re {print $1}' | head -1)
  if [ -z "$ID" ]; then
    echo "$TS NO-DROPLET (supervisor between launches?)" >> "$OUT"
    sleep "$CHECK_S"; continue
  fi
  IP=$(droplet_ip "$ID" 2>/dev/null)
  SNAP=$(timeout 60 ssh "${SSH_OPTS[@]}" -o ConnectTimeout=15 "root@$IP" '
    tel=$(grep -haE "hytorch: (step|WARNING)" /var/log/hytorch-arm-catalog.log 2>/dev/null | tail -1)
    warn=$(grep -hacE "hytorch: WARNING" /var/log/hytorch-arm-catalog.log 2>/dev/null)
    st=$(grep -haE "step [0-9]{5}/" /var/log/hytorch-arm-*.log 2>/dev/null | tail -1 | cut -c1-110)
    bpb=$(grep -haE "Validation bpb" /var/log/hytorch-arm-catalog.log 2>/dev/null | tail -1)
    seam=$(tail -1 /run/hytorch/nano-spool/seam.stderr 2>/dev/null | cut -c1-130)
    echo "TEL|$tel"; echo "WARN|$warn"; echo "STEP|$st"; echo "BPB|$bpb"; echo "SEAM|$seam"
  ' 2>/dev/null)
  if [ -z "$SNAP" ]; then
    echo "$TS $ID SSH-UNREACHABLE" >> "$OUT"
    sleep "$CHECK_S"; continue
  fi
  TEL=$(sed -n 's/^TEL|//p' <<<"$SNAP")
  WARN=$(sed -n 's/^WARN|//p' <<<"$SNAP")
  STEP=$(sed -n 's/^STEP|//p' <<<"$SNAP")
  BPB=$(sed -n 's/^BPB|//p' <<<"$SNAP")
  SEAM=$(sed -n 's/^SEAM|//p' <<<"$SNAP")

  ALERT=""
  # commit-rate check from the telemetry line
  RATE=$(grep -oE 'commit=[0-9.]+' <<<"$TEL" | cut -d= -f2)
  if [ -n "$RATE" ]; then
    LOW=$(python3 -c "print(1 if float('$RATE' or 100) < 5 else 0)" 2>/dev/null)
    [ "$LOW" = 1 ] && ALERT="$ALERT COMMIT-RATE-LOW($RATE%)"
  fi
  AB=$(grep -oE 'abort=[0-9.]+' <<<"$TEL" | cut -d= -f2)
  if [ -n "$AB" ]; then
    HIGH=$(python3 -c "print(1 if float('$AB' or 0) > 5 else 0)" 2>/dev/null)
    [ "$HIGH" = 1 ] && ALERT="$ALERT ABORT-HIGH($AB%)"
  fi
  [ "${WARN:-0}" != "0" ] && [ -n "$WARN" ] && ALERT="$ALERT SHIM-WARNINGS($WARN)"
  # seam-head stuck while trainer moves
  CUR_HEAD=$(grep -oE 'head [0-9a-f]{16}' <<<"$SEAM" | head -1)
  CUR_STEP=$(grep -oE 'step [0-9]{5}' <<<"$STEP" | head -1)
  if [ -n "$CUR_HEAD" ] && [ "$CUR_HEAD" = "$LAST_HEAD" ] && [ -n "$CUR_STEP" ] && [ "$CUR_STEP" != "$LAST_STEP" ]; then
    ALERT="$ALERT SEAM-HEAD-STUCK"
  fi
  LAST_HEAD="$CUR_HEAD"; LAST_STEP="$CUR_STEP"

  {
    echo "$TS $ID"
    echo "  $STEP"
    [ -n "$TEL" ] && echo "  $TEL"
    [ -n "$BPB" ] && echo "  $BPB"
    [ -n "$SEAM" ] && echo "  seam: $SEAM"
    [ -n "$ALERT" ] && echo "  *** ALERT:$ALERT ***"
  } >> "$OUT"
  [ -n "$ALERT" ] && echo "$(date -u +%FT%TZ) *** ALERT:$ALERT *** (see $OUT)" >&2
  sleep "$CHECK_S"
done
