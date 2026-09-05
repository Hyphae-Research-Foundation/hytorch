#!/usr/bin/env bash
# infra/hallu-run.sh — phase 4a: hallucination signature experiment on a GPU
# droplet. Preregistered threshold lives in the manifest (AUROC >= 0.65).
#
# Usage: hallu-run.sh <nv|amd|dev> [train_steps] [cloze_prompts]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TIER="${1:?tier}"; STEPS="${2:-5000}"; PROMPTS="${3:-2000}"
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT="${COMMIT:0:12}"
NAME="hytorch-hallu-$(date -u +%H%M%S)"

log "phase 4a: tier=$TIER steps=$STEPS prompts=$PROMPTS"
read -r ID SIZE REGION PROC < <("$(dirname "${BASH_SOURCE[0]}")/ladder.sh" "$TIER" "$NAME" hallu)
trap 'log "teardown $ID"; doctl compute droplet delete "$ID" --force || true' EXIT

"$(dirname "${BASH_SOURCE[0]}")/provision.sh" "$ID" "$COMMIT" >/dev/null
IP=$(droplet_ip "$ID")
ARCH=$(hyssh "root@$IP" 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr " " "-" || echo unknown')
OUT_DIR="$REPO_ROOT/results/hallu/$ARCH/$SHORT"
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
  cd python && python3 -m hytorch.data --out /data/wikitext103 2>&1 | tail -2

  python3 -m hytorch.hallu \
    --manifest ../manifests/phase4a-signature.json \
    --tokens /data/wikitext103 --device cuda \
    --train-steps $STEPS --cloze-prompts $PROMPTS \
    --out-dir /run/hytorch/hallu
" | tee "$OUT_DIR/hallu.log"

scp -i "$SSH_ID" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  "root@$IP:/run/hytorch/hallu/*" "$OUT_DIR/" 2>/dev/null || true
log "phase 4a done → $OUT_DIR"
