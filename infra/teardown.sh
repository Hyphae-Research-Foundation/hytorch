#!/usr/bin/env bash
# infra/teardown.sh — destroy hytorch droplets by tag and VERIFY nothing is
# left billing. Default tag: hytorch (everything). Narrower: hytorch-<role>.
set -euo pipefail
TAG="${1:-hytorch}"

echo "Droplets tagged $TAG:"
doctl compute droplet list --tag-name "$TAG" --format ID,Name,Size,Region,Status --no-header || true

read -r -p "Destroy ALL of the above? [yes/N] " ans
[ "$ans" = "yes" ] || { echo "aborted"; exit 1; }

doctl compute droplet delete --tag-name "$TAG" --force

echo "Verifying…"
sleep 3
left=$(doctl compute droplet list --tag-name "$TAG" --format ID --no-header | wc -l)
if [ "$left" -eq 0 ]; then
  echo "OK: nothing tagged $TAG remains."
else
  echo "WARNING: $left droplet(s) still present — check manually." >&2
  exit 1
fi

# --- Volumes (review 2026-09-02: 2000GiB volumes were created by
# nanochat-run.sh and nothing ever deleted them — ~$200/month each). List
# every hytorch-* volume and offer deletion, one by one, with confirmation.
# NEVER auto-delete: a volume may hold the only copy of a run's checkpoints.
echo
echo "hytorch volumes (each ~\$0.10/GiB/month):"
doctl compute volume list --format ID,Name,Size,Region,DropletIDs --no-header | grep -E "hytorch" || { echo "  (none)"; exit 0; }
while read -r VID VNAME VSIZE VREG VATT; do
  [ -n "$VID" ] || continue
  if [ -n "$VATT" ] && [ "$VATT" != "[]" ]; then
    echo "  $VNAME is attached to $VATT — skip (detach or destroy the droplet first)"; continue
  fi
  read -r -p "Delete volume $VNAME ($VSIZE, $VREG)? Type its NAME to confirm: " ans
  if [ "$ans" = "$VNAME" ]; then
    doctl compute volume delete "$VID" --force && echo "  deleted $VNAME"
  else
    echo "  kept $VNAME"
  fi
done < <(doctl compute volume list --format ID,Name,Size,Region,DropletIDs --no-header | grep -E "hytorch")
