#!/usr/bin/env bash
# infra/train-run.sh — citable training run on a GPU droplet: wikitext-103,
# both arms (catalog + baseline), full protocol v2 against live Hyphae.
#
# Usage: train-run.sh <nv|amd|dev> <manifest-name> [steps-override] [commit]
#   manifest-name: file in manifests/ (e.g. phase1-k8-nf32768.json)
#   steps-override: cap steps for validation runs (0 = manifest value)
#
# Results → results/train/<arch>/<manifest>-<commit>/.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TIER="${1:?tier nv|amd|dev}"
MANIFEST="${2:?manifest name in manifests/}"
STEPS="${3:-0}"
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
COMMIT="${4:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
SHORT="${COMMIT:0:12}"
NAME="hytorch-train-$TIER-$(date -u +%H%M%S)"

log "train run: tier=$TIER manifest=$MANIFEST steps=$STEPS commit=$SHORT"
read -r ID SIZE REGION PROC < <("$(dirname "${BASH_SOURCE[0]}")/ladder.sh" "$TIER" "$NAME" train)
trap 'log "teardown $ID"; doctl compute droplet delete "$ID" --force || true' EXIT

"$(dirname "${BASH_SOURCE[0]}")/provision.sh" "$ID" "$COMMIT" >/dev/null
IP=$(droplet_ip "$ID")

ARCH=$(hyssh "root@$IP" 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr " " "-" || (rocminfo 2>/dev/null | grep "Marketing Name" | grep -i instinct | head -1 | sed "s/.*: *//;s/ /-/g") || echo unknown')
OUT_DIR="$REPO_ROOT/results/train/$ARCH/${MANIFEST%.json}-$SHORT"
mkdir -p "$OUT_DIR"

hyssh "root@$IP" "
  set -e
  cd /opt/hytorch
  export PATH=\$PATH:/usr/local/cuda/bin:/opt/rocm/bin
  export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-}

  echo '=== [1/5] differential gate (this silicon) ==='
  if command -v nvcc >/dev/null; then
    nvcc -O2 -fmad=false -arch=native kernels/cuda/diff_harness.cu kernels/cuda/refkernels.cu -o /tmp/diff_harness -ldl
  else
    for i in \$(seq 1 60); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 5; done
    apt-get install -y -q amdrocm-core-dev7.14 amdrocm-runtime-dev7.14 2>&1 | tail -1 || true
    hipcc -x hip -O2 -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt \
      kernels/cuda/diff_harness.cu kernels/cuda/refkernels.cu -o /tmp/diff_harness -ldl
  fi
  /tmp/diff_harness target/release/libapply_ref.so 200000

  echo '=== [2/5] python stack ==='
  # pip on Ubuntu 22.04 lacks --break-system-packages; 24.04 requires it.
  if python3 -m pip install -q --help 2>/dev/null | grep -q break-system-packages; then
    PIPQ='python3 -m pip install -q --break-system-packages'
  else
    PIPQ='python3 -m pip install -q'
  fi
  \$PIPQ numpy ninja tiktoken pyarrow 2>&1 | tail -1
  if ! python3 -c 'import torch' 2>/dev/null; then
    if command -v rocminfo >/dev/null; then
      timeout 900 \$PIPQ torch --index-url https://download.pytorch.org/whl/rocm7.0 2>&1 | tail -1
    else
      timeout 900 \$PIPQ torch 2>&1 | tail -1
    fi
  fi
  python3 -c 'import torch; assert torch.cuda.is_available(), \"no device visible\"; print(torch.__version__, torch.cuda.get_device_name(0))'

  echo '=== [3/5] dataset (wikitext-103 -> bins) ==='
  cd python
  python3 -m hytorch.data --out /data/wikitext103 2>&1 | tail -4
  cd ..

  echo '=== [4/5] build facts ==='
  SO_HASH=\$(sha256sum target/release/libapply_ref.so | cut -d' ' -f1)
  TORCH=\$(python3 -c 'import torch; print(f\"torch-{torch.__version__}\")')
  BACKEND=\$(python3 -c 'import torch; print(f\"cuda-{torch.version.cuda}\" if torch.version.cuda else (f\"rocm-{torch.version.hip}\" if getattr(torch.version,\"hip\",None) else \"cpu\"))')
  DRIVER=\$( (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/nvidia-/') || echo rocm )
  # Durability discipline: Hyphae data dir on REAL disk (the ledger must
  # survive; /run is tmpfs). The spool is transient IPC: tmpfs is right.
  mkdir -p /var/lib/hytorch /run/hytorch
  cat > /run/hytorch/build-facts.json <<EOF
{
  \"build.apply_ref_hash\": \"\$SO_HASH\",
  \"build.harness_commit\": \"$COMMIT\",
  \"build.torch_wheel\": \"\$TORCH\",
  \"build.backend_wheel\": \"\$BACKEND\",
  \"infra.driver\": \"\$DRIVER\"
}
EOF
  cat /run/hytorch/build-facts.json

  echo '=== [5/5] both arms ==='
  cd python
  echo '--- catalog arm ---'
  python3 -m hytorch.run \
    --manifest ../manifests/$MANIFEST \
    --data-dir /var/lib/hytorch/hyphae --spool /run/hytorch/spool \
    --seam-bin ../target/release/hytorch-seam \
    --build-facts /run/hytorch/build-facts.json \
    --device cuda --tokens /data/wikitext103 \
    --steps-override $STEPS --eval-every 0 --out-dir /run/hytorch
  echo '--- baseline arm ---'
  python3 -m hytorch.run \
    --manifest ../manifests/$MANIFEST \
    --build-facts /run/hytorch/build-facts.json \
    --device cuda --tokens /data/wikitext103 --baseline \
    --steps-override $STEPS --eval-every 0 --out-dir /run/hytorch
" | tee "$OUT_DIR/train.log"

# Pull the runlogs (custody: everything lands on this side).
scp -i "$SSH_ID" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  "root@$IP:/run/hytorch/*.runlog.json" "$OUT_DIR/" 2>/dev/null || true

{
  echo "{"
  echo "  \"commit\": \"$COMMIT\","
  echo "  \"manifest\": \"$MANIFEST\","
  echo "  \"steps_override\": $STEPS,"
  echo "  \"gpu\": \"$ARCH\","
  echo "  \"droplet\": {\"size\": \"$SIZE\", \"region\": \"$REGION\", \"procurement\": \"$PROC\"},"
  echo "  \"gate\": \"$(grep -o 'BIT_IDENTICAL\|BACKEND_REJECTED' "$OUT_DIR/train.log" | head -1)\","
  echo "  \"finished_at\": \"$(date -u +%FT%TZ)\""
  echo "}"
} > "$OUT_DIR/summary.json"
log "train run done → $OUT_DIR"
