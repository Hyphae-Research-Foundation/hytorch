#!/usr/bin/env bash
# infra/sft-probe.sh — one-shot SFT-catalog diagnostic on an 8x box.
# Creates the droplet, mounts the volume (RW: SFT writes checkpoints), restores
# the ledger to NVMe, provisions the current commit, applies nanochat patch +
# SFT probe, launches ONLY the catalog SFT stage with HYTORCH_SFT_PROBE=1, and
# LEAVES THE DROPLET UP for inspection (teardown is manual: this is forensics).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
set +e
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
export HYTORCH_FORCE_SIZE="${HYTORCH_FORCE_SIZE:-gpu-mi355x8-2304gb-spot}"
export HYTORCH_GPU_REGIONS=mem1
NAME="hytorch-sftprobe-$(date -u +%H%M%S)"

ID=""
for i in $(seq 1 90); do
  if OUT=$("$(dirname "${BASH_SOURCE[0]}")/ladder.sh" amd "$NAME" nano 2>/dev/null); then
    read -r ID SIZE REGION PROC <<<"$OUT"; break
  fi
  log "no capacity; retry $i/90 in 120s"; sleep 120
done
[ -n "$ID" ] || { log "no capacity"; exit 1; }
IP=$(droplet_ip "$ID")
wait_ssh "$IP" || { log "ssh dead"; exit 1; }
log "droplet $ID @ $IP ($SIZE $REGION $PROC) — NOT auto-torn-down"

VOL_ID=$(doctl compute volume list --format ID,Name --no-header | awk '$2=="hytorch-p5"{print $1}')
ATT=$(doctl compute volume get "$VOL_ID" --format DropletIDs --no-header | tr -d '[]')
if [ -n "$ATT" ] && [ "$ATT" != "$ID" ]; then
  log "volume attached to $ATT (custody box?) — detaching"
  doctl compute volume-action detach "$VOL_ID" "$ATT" --wait >/dev/null || true
fi
doctl compute volume-action attach "$VOL_ID" "$ID" --wait >/dev/null
hyssh "root@$IP" '
set -e
mkdir -p /mnt/hyvol
DEV=/dev/disk/by-id/scsi-0DO_Volume_hytorch-p5
for i in $(seq 1 30); do [ -e "$DEV" ] && break; sleep 2; done
mount -o discard,defaults "$DEV" /mnt/hyvol
for i in $(seq 1 60); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 5; done
command -v rsync >/dev/null || apt-get install -y -q rsync >/dev/null 2>&1
mkdir -p /var/lib/hytorch/nano-hyphae
ln -sfn /mnt/hyvol/nanochat-cache /var/lib/hytorch/nanochat-cache
ln -sfn /mnt/hyvol/arm-catalog    /var/lib/hytorch/arm-catalog
ln -sfn /mnt/hyvol/arm-vanilla    /var/lib/hytorch/arm-vanilla
echo "restoring ledger (823GB) volume -> NVMe in background"
setsid nohup rsync -a /mnt/hyvol/nano-hyphae/ /var/lib/hytorch/nano-hyphae/ > /var/log/ledger-restore.log 2>&1 < /dev/null &
echo $! > /run/ledger-restore.pid
'
"$(dirname "${BASH_SOURCE[0]}")/provision.sh" "$ID" "$COMMIT" >/dev/null
log "provisioned $COMMIT"

hyssh "root@$IP" "
set -e
cd /opt/hytorch
export PATH=\$PATH:/opt/rocm/bin
for i in \$(seq 1 60); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 5; done
apt-get install -y -q amdrocm-core-dev7.14 amdrocm-runtime-dev7.14 2>&1 | tail -1 || true
PY=\$(command -v python3)
PIPQ=\"\$PY -m pip install -q --break-system-packages\"
\$PIPQ --ignore-installed numpy ninja tiktoken pyarrow wandb regex rustbpe psutil 2>&1 | tail -1
\$PY -c 'import torch' 2>/dev/null || timeout 900 \$PIPQ --ignore-installed torch --index-url https://download.pytorch.org/whl/rocm7.0 2>&1 | tail -1
bash /opt/hytorch/nanochat/apply-patch.sh /opt/nanochat-work
rm -rf /opt/nanochat && mv /opt/nanochat-work/nanochat /opt/nanochat
cd /opt/nanochat && python3 /opt/hytorch/nanochat/sft_probe.py
\$PIPQ rustbpe psutil datasets 2>&1 | tail -1 || true
cd /opt/hytorch
SO_HASH=\$(sha256sum target/release/libapply_ref.so | cut -d' ' -f1)
TORCH=\$(python3 -c 'import torch; print(f\"torch-{torch.__version__}\")')
BACKEND=\$(python3 -c 'import torch; print(f\"rocm-{torch.version.hip}\")')
mkdir -p /run/hytorch
printf '{\"build.apply_ref_hash\":\"%s\",\"build.harness_commit\":\"%s\",\"build.torch_wheel\":\"%s\",\"build.backend_wheel\":\"%s\"}\n' \"\$SO_HASH\" \"$COMMIT\" \"\$TORCH\" \"\$BACKEND\" > /run/hytorch/build-facts.json
python3 - <<PYEOF
import json
m = json.load(open('manifests/phase3-nanochat-template.json'))
m['model']['d_model'] = 1280; m['model']['d_slot'] = 20; m['model']['n_layers'] = 20; m['\$depth'] = 20
json.dump(m, open('manifests/phase3-nanochat-d20.json', 'w'), indent=2)
PYEOF
echo 'waiting for ledger restore…'
while kill -0 \$(cat /run/ledger-restore.pid) 2>/dev/null; do sleep 30; du -sh /var/lib/hytorch/nano-hyphae | cut -f1; done
echo 'restore done'
rm -rf /run/hytorch/nano-spool; mkdir -p /run/hytorch/nano-spool
cd /opt/nanochat
cat > /root/run-sft-probe.sh <<'SFTEOF'
#!/usr/bin/env bash
cd /opt/nanochat
export OMP_NUM_THREADS=1
env TORCHDYNAMO_DISABLE=1 NANOCHAT_BASE_DIR=/var/lib/hytorch/arm-catalog HYTORCH_CATALOG=1 HYTORCH_SFT_PROBE=1 \
  HYTORCH_WIRE_EVERY=100 HYTORCH_BARRIER_MS=600000 HYTORCH_ROOT=/opt/hytorch HYTORCH_D_SLOT=20 \
  HYTORCH_MANIFEST=/opt/hytorch/manifests/phase3-nanochat-d20.json HYTORCH_BUILD_FACTS=/run/hytorch/build-facts.json \
  HYTORCH_DATA_DIR=/var/lib/hytorch/nano-hyphae HYTORCH_SPOOL=/run/hytorch/nano-spool \
  HYTORCH_SEAM_BIN=/opt/hytorch/target/release/hytorch-seam HYTORCH_RUN_ID=nano-d20-sft-probe NANOCHAT_DTYPE=bfloat16 \
  torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=dummy --num-iterations=3 --eval-every=-1 --chatcore-every=-1 \
  > /var/log/hytorch-sft-probe.log 2>&1
echo "probe rc=\$?" >> /var/log/hytorch-sft-probe.log
SFTEOF
chmod +x /root/run-sft-probe.sh
setsid nohup /root/run-sft-probe.sh > /dev/null 2>&1 < /dev/null &
echo 'SFT probe launched'
"
log "probe running on $ID @ $IP — watch /var/log/hytorch-sft-probe.log"
echo "$ID $IP" > "$STATE_DIR/sft-probe.box"
