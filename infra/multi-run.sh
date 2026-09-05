#!/usr/bin/env bash
# infra/multi-run.sh — run SEVERAL manifests (both arms each) sequentially on
# ONE GPU droplet, amortizing provision + dataset + gate. For robustness
# batteries (LR sweeps, seed batteries) where each manifest is 20k steps.
#
# Usage: multi-run.sh <nv|amd|dev> <tag> <manifest1> [manifest2] ...
# Results → results/train/<arch>/<tag>-<commit>/<manifest-stem>/
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TIER="${1:?tier}"; TAG="${2:?tag}"; shift 2
MANIFESTS=("$@")
[ ${#MANIFESTS[@]} -ge 1 ] || { echo "need at least one manifest" >&2; exit 2; }

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT="${COMMIT:0:12}"
NAME="hytorch-multi-$TAG-$(date -u +%H%M%S)"

log "multi-run: tier=$TIER tag=$TAG manifests=${MANIFESTS[*]}"
read -r ID SIZE REGION PROC < <("$(dirname "${BASH_SOURCE[0]}")/ladder.sh" "$TIER" "$NAME" train)
trap 'log "teardown $ID"; doctl compute droplet delete "$ID" --force || true' EXIT

"$(dirname "${BASH_SOURCE[0]}")/provision.sh" "$ID" "$COMMIT" >/dev/null
IP=$(droplet_ip "$ID")
ARCH=$(hyssh "root@$IP" 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr " " "-" || (rocminfo 2>/dev/null | grep "Marketing Name" | grep -i instinct | head -1 | sed "s/.*: *//;s/ /-/g") || echo unknown')
OUT_DIR="$REPO_ROOT/results/train/$ARCH/$TAG-$SHORT"
mkdir -p "$OUT_DIR"

RUNS=""
for m in "${MANIFESTS[@]}"; do
  stem="${m%.json}"
  RUNS+="
  echo \"=== manifest: $m (catalog) ===\"
  rm -rf /var/lib/hytorch/hyphae-$stem /run/hytorch/spool-$stem
  python3 -m hytorch.run \
    --manifest ../manifests/$m \
    --data-dir /var/lib/hytorch/hyphae-$stem --spool /run/hytorch/spool-$stem \
    --seam-bin ../target/release/hytorch-seam \
    --build-facts /run/hytorch/build-facts.json \
    --device cuda --tokens /data/wikitext103 \
    --out-dir /run/hytorch/$stem
  echo \"=== manifest: $m (baseline) ===\"
  python3 -m hytorch.run \
    --manifest ../manifests/$m \
    --build-facts /run/hytorch/build-facts.json \
    --device cuda --tokens /data/wikitext103 --baseline \
    --out-dir /run/hytorch/$stem
"
done

hyssh "root@$IP" "
  set -e
  cd /opt/hytorch
  export PATH=\$PATH:/usr/local/cuda/bin:/opt/rocm/bin
  export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-}

  echo '=== gate (this silicon, both selection modes) ==='
  if command -v nvcc >/dev/null; then
    nvcc -O2 -fmad=false -arch=native kernels/cuda/diff_harness.cu kernels/cuda/refkernels.cu -o /tmp/diff_harness -ldl
  else
    for i in \$(seq 1 60); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 5; done
    apt-get install -y -q amdrocm-core-dev7.14 amdrocm-runtime-dev7.14 2>&1 | tail -1 || true
    hipcc -x hip -O2 -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt \
      kernels/cuda/diff_harness.cu kernels/cuda/refkernels.cu -o /tmp/diff_harness -ldl
  fi
  /tmp/diff_harness target/release/libapply_ref.so 200000

  if python3 -m pip install -q --help 2>/dev/null | grep -q break-system-packages; then
    PIPQ='python3 -m pip install -q --break-system-packages'
  else PIPQ='python3 -m pip install -q'; fi
  \$PIPQ numpy ninja tiktoken pyarrow 2>&1 | tail -1
  if ! python3 -c 'import torch' 2>/dev/null; then
    if command -v rocminfo >/dev/null; then
      timeout 900 \$PIPQ torch --index-url https://download.pytorch.org/whl/rocm7.0 2>&1 | tail -1
    else
      timeout 900 \$PIPQ torch 2>&1 | tail -1
    fi
  fi
  python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'

  cd python && python3 -m hytorch.data --out /data/wikitext103 2>&1 | tail -2 && cd ..

  SO_HASH=\$(sha256sum target/release/libapply_ref.so | cut -d' ' -f1)
  TORCH=\$(python3 -c 'import torch; print(f\"torch-{torch.__version__}\")')
  BACKEND=\$(python3 -c 'import torch; print(f\"cuda-{torch.version.cuda}\" if torch.version.cuda else f\"rocm-{torch.version.hip}\")')
  mkdir -p /var/lib/hytorch /run/hytorch
  printf '{\"build.apply_ref_hash\":\"%s\",\"build.harness_commit\":\"%s\",\"build.torch_wheel\":\"%s\",\"build.backend_wheel\":\"%s\"}\n' \
    \"\$SO_HASH\" \"$COMMIT\" \"\$TORCH\" \"\$BACKEND\" > /run/hytorch/build-facts.json

  cd python
  $RUNS
" | tee "$OUT_DIR/multi.log"

# Custody: pull every runlog per manifest.
for m in "${MANIFESTS[@]}"; do
  stem="${m%.json}"
  mkdir -p "$OUT_DIR/$stem"
  scp -i "$SSH_ID" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    "root@$IP:/run/hytorch/$stem/*.runlog.json" "$OUT_DIR/$stem/" 2>/dev/null || true
done
log "multi-run done → $OUT_DIR"
