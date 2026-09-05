#!/usr/bin/env bash
# infra/probe-run.sh — phase 2: catalog run with the journal probe + causal
# ablation, on a GPU droplet. Shorter than citable (probe needs signal, not
# a threshold): default 5000 steps.
#
# Usage: probe-run.sh <nv|amd|dev> [steps] [manifest]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TIER="${1:?tier}"; STEPS="${2:-5000}"; MANIFEST="${3:-phase1b-global.json}"
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT="${COMMIT:0:12}"
NAME="hytorch-probe-$(date -u +%H%M%S)"

log "probe run: tier=$TIER steps=$STEPS manifest=$MANIFEST"
read -r ID SIZE REGION PROC < <("$(dirname "${BASH_SOURCE[0]}")/ladder.sh" "$TIER" "$NAME" probe)
trap 'log "teardown $ID"; doctl compute droplet delete "$ID" --force || true' EXIT

"$(dirname "${BASH_SOURCE[0]}")/provision.sh" "$ID" "$COMMIT" >/dev/null
IP=$(droplet_ip "$ID")
ARCH=$(hyssh "root@$IP" 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr " " "-" || echo unknown')
OUT_DIR="$REPO_ROOT/results/probe/$ARCH/$SHORT"
mkdir -p "$OUT_DIR"

hyssh "root@$IP" "
  set -e
  cd /opt/hytorch
  export PATH=\$PATH:/usr/local/cuda/bin
  nvcc -O2 -fmad=false -arch=native kernels/cuda/diff_harness.cu kernels/cuda/refkernels.cu -o /tmp/diff_harness -ldl
  /tmp/diff_harness target/release/libapply_ref.so 50000

  if python3 -m pip install -q --help 2>/dev/null | grep -q break-system-packages; then
    PIPQ='python3 -m pip install -q --break-system-packages'
  else PIPQ='python3 -m pip install -q'; fi
  \$PIPQ numpy ninja tiktoken pyarrow 2>&1 | tail -1
  python3 -c 'import torch' 2>/dev/null || timeout 900 \$PIPQ torch 2>&1 | tail -1
  python3 -c 'import torch; assert torch.cuda.is_available()'
  cd python && python3 -m hytorch.data --out /data/wikitext103 2>&1 | tail -2 && cd ..

  SO_HASH=\$(sha256sum target/release/libapply_ref.so | cut -d' ' -f1)
  TORCH=\$(python3 -c 'import torch; print(f\"torch-{torch.__version__}\")')
  mkdir -p /var/lib/hytorch /run/hytorch
  printf '{\"build.apply_ref_hash\":\"%s\",\"build.harness_commit\":\"%s\",\"build.torch_wheel\":\"%s\",\"build.backend_wheel\":\"cuda\"}\n' \
    \"\$SO_HASH\" \"$COMMIT\" \"\$TORCH\" > /run/hytorch/build-facts.json

  cd python
  python3 -m hytorch.run \
    --manifest ../manifests/$MANIFEST \
    --data-dir /var/lib/hytorch/hyphae-probe --spool /run/hytorch/spool-probe \
    --seam-bin ../target/release/hytorch-seam \
    --build-facts /run/hytorch/build-facts.json \
    --device cuda --tokens /data/wikitext103 \
    --steps-override $STEPS --probe --eval-batches 48 \
    --out-dir /run/hytorch/probe
" | tee "$OUT_DIR/probe.log"

scp -i "$SSH_ID" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  "root@$IP:/run/hytorch/probe/*.runlog.json" "$OUT_DIR/" 2>/dev/null || true
log "probe run done → $OUT_DIR"
