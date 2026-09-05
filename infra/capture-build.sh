#!/usr/bin/env bash
# infra/capture-build.sh — runs ON the droplet. Builds release artifacts and
# emits the build.* facts JSON (spec §09 / C10). Falls back gracefully for
# fields that only exist on GPU hosts (wheels, drivers).
#
# Usage: capture-build.sh <commit-sha>
set -euo pipefail
COMMIT="${1:?commit}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

cargo build --workspace --release >&2

SO="target/release/libapply_ref.so"
[ -f "$SO" ] || { echo "missing $SO" >&2; exit 1; }
APPLY_REF_HASH=$(sha256sum "$SO" | cut -d' ' -f1)

TORCH_WHEEL="absent"
BACKEND_WHEEL="absent"
if command -v python3 >/dev/null && python3 -c 'import torch' 2>/dev/null; then
  TORCH_WHEEL=$(python3 - <<'EOF'
import torch, hashlib, pathlib
p = pathlib.Path(torch.__file__).parent
print(f"torch-{torch.__version__}")
EOF
)
  BACKEND_WHEEL=$(python3 - <<'EOF'
import torch
if torch.version.cuda: print(f"cuda-{torch.version.cuda}")
elif getattr(torch.version, "hip", None): print(f"rocm-{torch.version.hip}")
else: print("cpu")
EOF
)
fi

DRIVER="none"
command -v nvidia-smi >/dev/null && DRIVER="nvidia-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
command -v rocm-smi  >/dev/null && DRIVER="rocm-$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"

RUSTC=$(rustc --version | tr -d '\n')

cat <<EOF
{
  "build.apply_ref_hash": "$APPLY_REF_HASH",
  "build.harness_commit": "$COMMIT",
  "build.torch_wheel": "$TORCH_WHEEL",
  "build.backend_wheel": "$BACKEND_WHEEL",
  "infra.driver": "$DRIVER",
  "infra.rustc": "$RUSTC",
  "infra.hostname": "$(hostname)",
  "captured_at": "$(date -u +%FT%TZ)"
}
EOF
