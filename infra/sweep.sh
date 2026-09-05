#!/usr/bin/env bash
# infra/sweep.sh — N catalog runs (different manifests) + 1 baseline on ONE
# GPU droplet. For policy/hyperparameter sweeps BEFORE freezing the citable
# manifest. Results → results/sweep/<arch>/<tag>-<commit>/.
#
# Usage: sweep.sh <nv|amd|dev> <tag> <steps> <manifest1> [manifest2] ...
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TIER="${1:?tier}"; TAG="${2:?tag}"; STEPS="${3:?steps}"; shift 3
MANIFESTS=("$@")
[ ${#MANIFESTS[@]} -ge 1 ] || { echo "need at least one manifest" >&2; exit 2; }

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT="${COMMIT:0:12}"
NAME="hytorch-sweep-$TAG-$(date -u +%H%M%S)"

log "sweep: tier=$TIER tag=$TAG steps=$STEPS manifests=${MANIFESTS[*]}"
read -r ID SIZE REGION PROC < <("$(dirname "${BASH_SOURCE[0]}")/ladder.sh" "$TIER" "$NAME" sweep)
trap 'log "teardown $ID"; doctl compute droplet delete "$ID" --force || true' EXIT

"$(dirname "${BASH_SOURCE[0]}")/provision.sh" "$ID" "$COMMIT" >/dev/null
IP=$(droplet_ip "$ID")
ARCH=$(hyssh "root@$IP" 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr " " "-" || echo unknown')
OUT_DIR="$REPO_ROOT/results/sweep/$ARCH/$TAG-$SHORT"
mkdir -p "$OUT_DIR"

RUNS=""
for m in "${MANIFESTS[@]}"; do
  RUNS+="
  echo \"--- catalog: $m ---\"
  rm -rf /var/lib/hytorch/hyphae-$m /run/hytorch/spool-$m
  python3 -m hytorch.run \
    --manifest ../manifests/$m \
    --data-dir /var/lib/hytorch/hyphae-$m --spool /run/hytorch/spool-$m \
    --seam-bin ../target/release/hytorch-seam \
    --build-facts /run/hytorch/build-facts.json \
    --device cuda --tokens /data/wikitext103 \
    --steps-override $STEPS --out-dir /run/hytorch
"
done

hyssh "root@$IP" "
  set -e
  cd /opt/hytorch
  export PATH=\$PATH:/usr/local/cuda/bin
  echo '=== gate ==='
  nvcc -O2 -fmad=false -arch=native kernels/cuda/diff_harness.cu kernels/cuda/refkernels.cu -o /tmp/diff_harness -ldl
  /tmp/diff_harness target/release/libapply_ref.so \${HYTORCH_GATE_ITERS:-50000}

  if python3 -m pip install -q --help 2>/dev/null | grep -q break-system-packages; then
    PIPQ='python3 -m pip install -q --break-system-packages'
  else PIPQ='python3 -m pip install -q'; fi
  \$PIPQ numpy ninja tiktoken pyarrow 2>&1 | tail -1
  python3 -c 'import torch' 2>/dev/null || timeout 900 \$PIPQ torch 2>&1 | tail -1
  python3 -c 'import torch; assert torch.cuda.is_available()'

  cd python && python3 -m hytorch.data --out /data/wikitext103 2>&1 | tail -2 && cd ..

  SO_HASH=\$(sha256sum target/release/libapply_ref.so | cut -d' ' -f1)
  TORCH=\$(python3 -c 'import torch; print(f\"torch-{torch.__version__}\")')
  BACKEND=\$(python3 -c 'import torch; print(f\"cuda-{torch.version.cuda}\")')
  mkdir -p /var/lib/hytorch /run/hytorch
  printf '{\"build.apply_ref_hash\":\"%s\",\"build.harness_commit\":\"%s\",\"build.torch_wheel\":\"%s\",\"build.backend_wheel\":\"%s\"}\n' \
    \"\$SO_HASH\" \"$COMMIT\" \"\$TORCH\" \"\$BACKEND\" > /run/hytorch/build-facts.json

  cd python
  $RUNS
  echo '--- baseline ---'
  python3 -m hytorch.run \
    --manifest ../manifests/${MANIFESTS[0]} \
    --build-facts /run/hytorch/build-facts.json \
    --device cuda --tokens /data/wikitext103 --baseline \
    --steps-override $STEPS --out-dir /run/hytorch
" | tee "$OUT_DIR/sweep.log"

scp -i "$SSH_ID" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  "root@$IP:/run/hytorch/*.runlog.json" "$OUT_DIR/" 2>/dev/null || true
log "sweep done → $OUT_DIR"
