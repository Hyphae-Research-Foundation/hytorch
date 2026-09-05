#!/usr/bin/env bash
# infra/common.sh — shared helpers. All hytorch droplets carry the tag
# "hytorch" plus "hytorch-<role>"; teardown is by tag, accounting is by tag.
set -euo pipefail

TAG_BASE="hytorch"
# The DO key id AND the local private key must be the same pair. Default:
# Operator's DigitalOcean ssh key id and private key path — set via env, no defaults in git.
SSH_KEYS="${HYTORCH_SSH_KEYS:?set HYTORCH_SSH_KEYS to your DO ssh key id}"
SSH_ID="${HYTORCH_SSH_ID:-$HOME/.ssh/id_ed25519}"
# Ephemeral droplets recycle IPs: a stale known_hosts entry turns into
# "Host key verification failed" and looks like a dead machine (2026-08-31:
# cost us 6 ghost-hunting relaunches). No host identity to defend here.
SSH_OPTS=(-i "$SSH_ID" -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=8)
STATE_DIR="$(dirname "${BASH_SOURCE[0]}")/.state"
mkdir -p "$STATE_DIR"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

hyssh() { ssh "${SSH_OPTS[@]}" "$@"; }

# regions_for <size-slug> — regions where a size is actually offered.
# The sizes API leaves `regions` empty for several GPU slugs; fall back to the
# known GPU regions and let `droplet create` be the arbiter of capacity.
GPU_FALLBACK_REGIONS="${HYTORCH_GPU_REGIONS:-$(doctl compute region list --format Slug,Available --no-header | awk '$2=="true"{print $1}' | paste -sd' ' -)}"
regions_for() {
  local r
  r=$(doctl compute size list -o json |
    python3 -c "import json,sys;[print(*s.get('regions',[]),sep='\n') for s in json.load(sys.stdin) if s['slug']==sys.argv[1]]" "$1")
  if [ -n "$r" ]; then echo "$r"; else echo $GPU_FALLBACK_REGIONS | tr ' ' '\n'; fi
}

# try_create <name> <size> <image> <region> [extra-tags]
# Prints droplet ID on success, returns 1 on failure (e.g. no spot capacity).
try_create() {
  local name="$1" size="$2" image="$3" region="$4" extra="${5:-}"
  local tags="$TAG_BASE${extra:+,$extra}"
  doctl compute droplet create "$name" \
    --size "$size" --image "$image" --region "$region" \
    --ssh-keys "$SSH_KEYS" --tag-names "$tags" \
    --user-data-file "$(dirname "${BASH_SOURCE[0]}")/cloud-init.yml" \
    --format ID --no-header --wait 2>"$STATE_DIR/create.err" || {
      # GHOST CREATE (seen 2026-08-31): the API returns 422 but the droplet
      # IS created server-side. Look the name up before declaring failure —
      # an unnoticed ghost is a leaked $36/h machine.
      sleep 8
      local ghost
      ghost=$(doctl compute droplet list --format ID,Name --no-header 2>/dev/null | awk -v n="$name" '$2==n{print $1}' | head -1)
      if [ -n "$ghost" ]; then
        log "ghost create detected: $name exists as $ghost despite API error — adopting"
        echo "$ghost"
        return 0
      fi
      log "create failed for $size in $region: $(tail -1 "$STATE_DIR/create.err")"
      return 1
    }
}

droplet_ip() { doctl compute droplet get "$1" --format PublicIPv4 --no-header; }

wait_ssh() {
  # 8-GPU nodes take far longer to boot than 1-GPU ones (3.3 attempt 2 died
  # at the old 5-min cap while the MI355X-8x was still booting). 20 min cap.
  local ip="$1" tries=0
  until hyssh -o ConnectTimeout=5 "root@$ip" true 2>/dev/null; do
    tries=$((tries + 1)); [ "$tries" -gt 240 ] && { log "ssh timeout $ip"; return 1; }
    sleep 5
  done
}
