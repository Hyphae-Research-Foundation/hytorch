#!/usr/bin/env bash
# infra/provision.sh — push the repo at an EXACT commit to a droplet and
# capture the build.* facts there (spec §09: RUN_START cites four digests).
#
# Usage: provision.sh <droplet-id> <commit-sha>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ID="${1:?droplet id}" COMMIT="${2:?commit sha}"
IP=$(droplet_ip "$ID")
wait_ssh "$IP"

# Wait for cloud-init to finish the toolchain install.
hyssh "root@$IP" 'until [ -f /root/.hytorch-cloudinit-done ]; do sleep 5; done'

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"

# Refuse to ship a dirty tree: build.harness_commit must describe the CODE
# that runs (spec C10). results/ is excluded: live batteries stream logs
# there while other runs launch — outputs are not code, and the shipped
# artifact is `git archive <commit>` regardless.
if ! git -C "$REPO_ROOT" diff --quiet HEAD -- ':(exclude)results' 2>/dev/null; then
  log "REFUSED: working tree is dirty outside results/; commit first (spec C10)"
  exit 1
fi

git -C "$REPO_ROOT" archive --format=tar "$COMMIT" | hyssh "root@$IP" '
  set -e
  rm -rf /opt/hytorch && mkdir -p /opt/hytorch
  tar -x -C /opt/hytorch
'
hyssh "root@$IP" "cd /opt/hytorch && bash infra/capture-build.sh '$COMMIT' > /opt/hytorch/build-facts.json && cat /opt/hytorch/build-facts.json"
