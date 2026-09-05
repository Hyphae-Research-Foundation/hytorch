#!/usr/bin/env bash
# infra/gpu-rehearsal.sh — one droplet, full GPU validation ladder:
#   1. re-run the differential gate (arch of THIS silicon)
#   2. build the torch extension and run the device-backend smoke
#   3. run the manifest rehearsal with the DeviceCatalog backend
# Results → results/gpu-rehearsal/<arch>/<commit>/.
#
# Usage: gpu-rehearsal.sh <nv|amd|dev> [commit]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TIER="${1:?tier nv|amd|dev}"
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
COMMIT="${2:-$(git -C "$REPO_ROOT" rev-parse HEAD)}"
SHORT="${COMMIT:0:12}"
NAME="hytorch-reh-$TIER-$(date -u +%H%M%S)"

log "GPU rehearsal: tier=$TIER commit=$SHORT"
read -r ID SIZE REGION PROC < <("$(dirname "${BASH_SOURCE[0]}")/ladder.sh" "$TIER" "$NAME" rehearsal)
trap 'log "teardown $ID"; doctl compute droplet delete "$ID" --force || true' EXIT

"$(dirname "${BASH_SOURCE[0]}")/provision.sh" "$ID" "$COMMIT" >/dev/null
IP=$(droplet_ip "$ID")

ARCH=$(hyssh "root@$IP" 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr " " "-" || rocminfo 2>/dev/null | grep -m1 "Marketing Name" | sed "s/.*: *//;s/ /-/g" || echo unknown')
OUT_DIR="$REPO_ROOT/results/gpu-rehearsal/$ARCH/$SHORT"
mkdir -p "$OUT_DIR"

hyssh "root@$IP" "
  set -e
  cd /opt/hytorch
  export PATH=\$PATH:/usr/local/cuda/bin

  echo '=== [1/3] differential gate ==='
  if command -v nvcc >/dev/null; then
    nvcc -O2 -fmad=false -arch=native kernels/cuda/diff_harness.cu kernels/cuda/refkernels.cu -o /tmp/diff_harness -ldl
  else
    # Sources are HIP-portable via __HIP__ guards; no hipify.
    # DO AMD image (ROCm 7.14 pre) ships hipcc but NOT the HIP runtime dev
    # headers; install them, and put rocm lib on the loader path.
    for i in \$(seq 1 60); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 5; done
    apt-get install -y -q amdrocm-core-dev7.14 amdrocm-runtime-dev7.14 2>&1 | tail -1 || true
    export PATH=\$PATH:/opt/rocm/bin
    export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-}
    hipcc -x hip -O2 -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt \
      kernels/cuda/diff_harness.cu kernels/cuda/refkernels.cu -o /tmp/diff_harness -ldl
  fi
  export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-}
  /tmp/diff_harness /opt/hytorch/target/release/libapply_ref.so 200000

  echo '=== [2/3] python env + build facts ==='
  # Prefer a system python that already has torch (DO AI/ML images ship it);
  # otherwise build a venv (cloud-init installs python3-venv).
  if python3 -c 'import torch' 2>/dev/null; then
    mkdir -p /opt/venv/bin && ln -sf \$(which python3) /opt/venv/bin/python
    python3 -m pip install -q --break-system-packages numpy ninja 2>&1 | tail -1 || \
      python3 -m pip install -q numpy ninja 2>&1 | tail -1 || true
  elif command -v rocminfo >/dev/null; then
    # AMD image: torch rocm7.0 wheel recognizes gfx950 (rocm6.4 does not).
    python3 -m pip install -q --break-system-packages numpy ninja 2>&1 | tail -1
    timeout 900 python3 -m pip install -q --break-system-packages \
      torch --index-url https://download.pytorch.org/whl/rocm7.0 2>&1 | tail -1
    mkdir -p /opt/venv/bin && ln -sf \$(which python3) /opt/venv/bin/python
  else
    python3 -m venv /opt/venv
    /opt/venv/bin/pip install -q numpy torch ninja 2>&1 | tail -1
  fi
  SO_HASH=\$(sha256sum target/release/libapply_ref.so | cut -d' ' -f1)
  TORCH=\$(/opt/venv/bin/python -c 'import torch; print(f\"torch-{torch.__version__}\")')
  BACKEND=\$(/opt/venv/bin/python -c 'import torch; print(f\"cuda-{torch.version.cuda}\" if torch.version.cuda else (f\"rocm-{torch.version.hip}\" if getattr(torch.version,\"hip\",None) else \"cpu\"))')
  mkdir -p /tmp/reh
  cat > /tmp/reh/build-facts.json <<EOF
{
  \"build.apply_ref_hash\": \"\$SO_HASH\",
  \"build.harness_commit\": \"$COMMIT\",
  \"build.torch_wheel\": \"\$TORCH\",
  \"build.backend_wheel\": \"\$BACKEND\"
}
EOF
  cat /tmp/reh/build-facts.json

  echo '=== [3/3] manifest rehearsal on DEVICE backend ==='
  export PATH=/opt/venv/bin:\$PATH   # ninja for the torch JIT
  cd python
  /opt/venv/bin/python -m hytorch.run \
    --manifest ../manifests/rehearsal-cpu.json \
    --data-dir /tmp/reh/hyphae --spool /tmp/reh/spool \
    --seam-bin ../target/release/hytorch-seam \
    --build-facts /tmp/reh/build-facts.json \
    --device cuda
" | tee "$OUT_DIR/rehearsal.log"

{
  echo "{"
  echo "  \"commit\": \"$COMMIT\","
  echo "  \"gpu\": \"$ARCH\","
  echo "  \"droplet\": {\"size\": \"$SIZE\", \"region\": \"$REGION\", \"procurement\": \"$PROC\"},"
  echo "  \"gate\": \"$(grep -o 'BIT_IDENTICAL\|BACKEND_REJECTED' "$OUT_DIR/rehearsal.log" | head -1)\","
  echo "  \"t1\": $(grep -o 't1_green.: [0-9]*' "$OUT_DIR/rehearsal.log" | tail -1 | grep -o '[0-9]*$' || echo 0),"
  echo "  \"finished_at\": \"$(date -u +%FT%TZ)\""
  echo "}"
} > "$OUT_DIR/summary.json"
log "rehearsal done → $OUT_DIR"
