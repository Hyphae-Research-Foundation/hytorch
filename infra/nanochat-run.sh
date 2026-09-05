#!/usr/bin/env bash
# infra/nanochat-run.sh — phases 3/5: nanochat with the hytorch catalog.
#
# Usage: nanochat-run.sh <nv|amd|dev|nv8|amd8> <depth> <num_iterations> [arm]
#   arm: catalog (default) | vanilla | both
#   3.4 CIERRE:  nanochat-run.sh amd8 20 2000 both
#   FASE 5:      HYTORCH_STAGES=speedrun HYTORCH_VOLUME=hytorch-p5 \
#                HYTORCH_SAVE_EVERY=1000 nanochat-run.sh amd8 20 -1 both
#
# Phase-5 knobs (all resolved locally at launch):
#   HYTORCH_STAGES     base (default) | speedrun (base→eval→SFT→chat_eval)
#   HYTORCH_VOLUME     DO volume NAME for all run state (created if missing;
#                      survives spot preemption — droplets are disposable)
#   HYTORCH_VOLUME_GB  size when creating (default 2000)
#   HYTORCH_SAVE_EVERY --save-every for base_train (default -1)
#   Resume is AUTOMATIC in volume mode: stamps + checkpoints on the volume
#   tell the remote script what is done and where to resume from; child runs
#   cite the parent run + last receipt head (§5.4) via hytorch-ledger-query.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TIER="${1:?tier}"; DEPTH="${2:?depth}"; ITERS="${3:?num_iterations}"; ARM="${4:-catalog}"
K_WIRE="${HYTORCH_WIRE_EVERY:-50}"
K_EVAL="${HYTORCH_EVAL_EVERY:--1}"
K_DBS="${HYTORCH_DBS:-16}"
K_TBS="${HYTORCH_TBS:--1}"
K_SHARDS="${HYTORCH_SHARDS:-8}"
K_STAGES="${HYTORCH_STAGES:-base}"
# Barrier budget: wired steps (wire_every) fsync ~5.4GB of raw wire to the
# volume; the receipt can trail the 60s shim default by minutes. Phase-5
# failure at step 100 (first wired step) taught this: pass the budget
# THROUGH to the trainer env (it does not inherit launcher env).
K_BARRIER="${HYTORCH_BARRIER_MS:-300000}"
K_VOL="${HYTORCH_VOLUME:-}"
K_VOL_GB="${HYTORCH_VOLUME_GB:-2000}"
K_SAVE="${HYTORCH_SAVE_EVERY:--1}"
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SHORT="${COMMIT:0:12}"
NAME="hytorch-nano-d${DEPTH}-$(date -u +%H%M%S)"

case "$TIER" in
  nv8)  export HYTORCH_FORCE_SIZE="${HYTORCH_FORCE_SIZE:-gpu-h200x8-1128gb}"; LTIER=nv; NPROC=8 ;;
  amd8) export HYTORCH_FORCE_SIZE="${HYTORCH_FORCE_SIZE:-gpu-mi355x8-2304gb-spot}"; LTIER=amd; NPROC=8 ;;
  *)    LTIER="$TIER"; NPROC=1 ;;
esac

# Backup cadence guard — AFTER the tier case resolves HYTORCH_FORCE_SIZE
# (review 2026-09-02: it used to run before, so `amd8` — the INCIDENT-901
# scenario — sailed through unchecked). Spot tiers and the bare `amd`/`nv`
# ladders (spot rungs first) need save_every in 1..250; a preemption then
# costs at most save_every steps. The ladder's procurement is re-checked
# after creation too (a spot rung may be picked even when FORCE_SIZE is unset).
spot_guard() {
  if [ "$K_SAVE" -lt 1 ] || [ "$K_SAVE" -gt 250 ]; then
    echo "[guard] spot procurement requires HYTORCH_SAVE_EVERY in 1..250 (got $K_SAVE)" >&2
    exit 2
  fi
}
case "${HYTORCH_FORCE_SIZE:-}${LTIER}" in *-spot*|amd|nv) spot_guard ;; esac

log "nanochat run: tier=$TIER depth=$DEPTH iters=$ITERS arm=$ARM nproc=$NPROC stages=$K_STAGES vol=${K_VOL:-none}"

# Volumes are region-locked: if the volume already exists, the droplet MUST
# land in its region (relaunch after preemption pins to where the state is).
if [ -n "$K_VOL" ]; then
  VOL_REGION=$(doctl compute volume list --format Name,Region --no-header | awk -v n="$K_VOL" '$1==n{print $2}')
  if [ -n "$VOL_REGION" ]; then
    export HYTORCH_GPU_REGIONS="$VOL_REGION"
    log "volume $K_VOL exists in $VOL_REGION — pinning droplet region"
  fi
fi

# HYTORCH_ADOPT_ID: run on an EXISTING droplet (ghost creates, manual
# forensics, reuse). Skips the ladder; teardown still owned by this script.
CAP_WAIT_MIN="${HYTORCH_CAPACITY_WAIT_MIN:-90}"
ID=""
if [ -n "${HYTORCH_ADOPT_ID:-}" ]; then
  ID="$HYTORCH_ADOPT_ID"
  trap 'log "teardown $ID"; doctl compute droplet delete "$ID" --force || true' EXIT
  IP=$(droplet_ip "$ID")
  wait_ssh "$IP" || { log "adopted droplet $ID unreachable"; exit 1; }
  read -r SIZE REGION < <(doctl compute droplet get "$ID" --format SizeSlug,Region --no-header)
  PROC="ondemand"; [[ "$SIZE" == *-spot ]] && PROC="spot"
  log "adopted droplet $ID @ $IP ($SIZE, $REGION, $PROC)"
fi
# Some 8x spot instances come up with dead SSH (2/5 so far). Create, verify
# SSH, destroy-and-recreate up to 3 times before giving up. In volume mode
# the region is PINNED, so capacity may be transiently gone right after a
# preemption (someone else took the nodes): poll with backoff up to
# HYTORCH_CAPACITY_WAIT_MIN minutes (default 90) instead of exhausting.
[ -n "$ID" ] || for attempt in 1 2 3; do
  LADDER_OUT=""
  deadline=$(( $(date +%s) + CAP_WAIT_MIN * 60 ))
  while true; do
    if LADDER_OUT=$("$(dirname "${BASH_SOURCE[0]}")/ladder.sh" "$LTIER" "$NAME-a$attempt" nano); then
      break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      log "no capacity after ${CAP_WAIT_MIN}m of polling — giving up"
      exit 1
    fi
    log "no capacity (likely transient post-preemption); retrying in 120s"
    sleep 120
  done
  read -r ID SIZE REGION PROC <<<"$LADDER_OUT"
  [ "$PROC" = spot ] && spot_guard
  trap 'log "teardown $ID"; doctl compute droplet delete "$ID" --force || true' EXIT
  IP=$(droplet_ip "$ID")
  if wait_ssh "$IP"; then
    log "ssh up on attempt $attempt ($ID @ $IP)"
    break
  fi
  log "ssh dead on $ID (attempt $attempt); destroying and recreating"
  doctl compute droplet delete "$ID" --force || true
  ID=""
done
[ -n "$ID" ] || { log "no reachable droplet after 3 attempts"; exit 1; }

# --- Volume mode: all run state (dataset, tokenizer, checkpoints, ledger,
# stamps) lives on a DO volume that OUTLIVES the spot droplet. ---
if [ -n "$K_VOL" ]; then
  VOL_ID=$(doctl compute volume list --format ID,Name --no-header | awk -v n="$K_VOL" '$2==n{print $1}')
  if [ -z "$VOL_ID" ]; then
    log "creating volume $K_VOL (${K_VOL_GB}GiB, $REGION)"
    VOL_ID=$(doctl compute volume create "$K_VOL" --region "$REGION" \
      --size "${K_VOL_GB}GiB" --fs-type ext4 --format ID --no-header)
  fi
  # detach from any dead droplet first (preemption leaves it attached-to-ghost
  # only if the droplet still exists; normally destroy auto-detaches).
  ATT=$(doctl compute volume get "$VOL_ID" --format DropletIDs --no-header | tr -d '[]')
  if [ -n "$ATT" ] && [ "$ATT" != "$ID" ]; then
    log "volume attached to $ATT; detaching"
    doctl compute volume-action detach "$VOL_ID" "$ATT" --wait || true
  fi
  log "attaching volume $VOL_ID to $ID"
  doctl compute volume-action attach "$VOL_ID" "$ID" --wait
  hyssh "root@$IP" "
    set -e
    mkdir -p /mnt/hyvol
    DEV=/dev/disk/by-id/scsi-0DO_Volume_$K_VOL
    for i in \$(seq 1 30); do [ -e \"\$DEV\" ] && break; sleep 2; done
    mount -o discard,defaults \"\$DEV\" /mnt/hyvol
    mkdir -p /mnt/hyvol/stamps /mnt/hyvol/nano-hyphae /mnt/hyvol/nanochat-cache \
             /mnt/hyvol/arm-catalog /mnt/hyvol/arm-vanilla
    # HYTORCH_RESET_ARMS=1: fresh training state on a reused volume (keeps
    # dataset/tokenizer). Wipes checkpoints, ledger, stage stamps.
    if [ \"${HYTORCH_RESET_ARMS:-0}\" = 1 ]; then
      echo 'RESET_ARMS: wiping checkpoints + ledger + stage stamps'
      rm -rf /mnt/hyvol/arm-catalog/base_checkpoints /mnt/hyvol/arm-vanilla/base_checkpoints \
             /mnt/hyvol/arm-catalog/chatsft_checkpoints /mnt/hyvol/arm-vanilla/chatsft_checkpoints \
             /mnt/hyvol/nano-hyphae/* 
      find /mnt/hyvol/stamps -maxdepth 1 -type f ! -name 'data_*' ! -name 'tok.done' ! -name 'evalbundle.done' -delete
    fi
    # Layout (phase-5 launch-1 lesson): the LEDGER lives on LOCAL NVMe —
    # Hyphae's per-step WAL fsync on network block storage doubled the
    # step time (15.6 vs 7.4 s/step) and the 5.4GB wire flush at wired
    # steps blew the 60s barrier (§5.4 kill at step 100). The volume keeps
    # a SNAPSHOT (rsync daemon below); checkpoints/dataset stay on the
    # volume (written rarely, must survive preemption immediately).
    for i in \$(seq 1 60); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 5; done
    command -v rsync >/dev/null || { apt-get install -y -q rsync 2>&1 | tail -1; }
    mkdir -p /var/lib/hytorch
    rm -f /var/lib/hytorch/nano-hyphae   # drop launch-1 symlink if present
    mkdir -p /var/lib/hytorch/nano-hyphae
    # The ledger snapshot is only meaningful when a run will RESUME (a
    # checkpoint exists). Otherwise it is an orphan of a failed fresh start:
    # wipe it instead of restoring (launch-2 lesson: a 15GB pointless rsync
    # through a quiet ssh pipe killed the session, rc=255,三 times).
    if ls /mnt/hyvol/arm-catalog/base_checkpoints/*/model_*.pt >/dev/null 2>&1 \
       && [ -n \"\$(ls -A /mnt/hyvol/nano-hyphae 2>/dev/null)\" ]; then
      echo 'restoring ledger snapshot volume -> local NVMe'
      rsync -a --delete /mnt/hyvol/nano-hyphae/ /var/lib/hytorch/nano-hyphae/ &
      RSYNC_PID=\$!
      while kill -0 \$RSYNC_PID 2>/dev/null; do echo 'restore in progress…'; sleep 20; done
      wait \$RSYNC_PID
      echo 'restore done'
    else
      # INCIDENT-901 custody rule: before wiping a stale ledger, archive its
      # receipt/STEP chain tail (KBs) — never again a kill without a corpse.
      if [ -n \"\$(ls -A /mnt/hyvol/nano-hyphae 2>/dev/null)\" ]; then
        TS=\$(date -u +%Y%m%dT%H%M%SZ)
        AR=/mnt/hyvol/incident-archive/\$TS
        mkdir -p \"\$AR\"
        cp /mnt/hyvol/stamps/catalog.current_run \"\$AR/\" 2>/dev/null || true
        cp /mnt/hyvol/stamps/catalog.last_head \"\$AR/\" 2>/dev/null || true
        RUN=\$(cat /mnt/hyvol/stamps/catalog.current_run 2>/dev/null)
        if [ -n \"\$RUN\" ] && [ -x /opt/hytorch/target/release/hytorch-ledger-query ]; then
          cp -r /mnt/hyvol/nano-hyphae /tmp/stale-led 2>/dev/null; rm -f /tmp/stale-led/LOCK
          LH=\$(/opt/hytorch/target/release/hytorch-ledger-query /tmp/stale-led last-head \"\$RUN\" 6000 2>/dev/null)
          echo \"\$LH\" > \"\$AR/last-receipt\"
          STEPN=\$(echo \"\$LH\" | cut -d' ' -f1)
          for k in RECEIPT STEP; do
            /opt/hytorch/target/release/hytorch-ledger-query /tmp/stale-led get \
              \"run/\$RUN/step/\$(printf %08d \${STEPN:-0})/\$k\" > \"\$AR/last-\$k\" 2>/dev/null || true
          done
          rm -rf /tmp/stale-led
        fi
        echo \"stale ledger chain archived to \$AR\"
      fi
      echo 'no resumable run: clearing stale ledger snapshot'
      rm -rf /mnt/hyvol/nano-hyphae/* 2>/dev/null || true
    fi
    ln -sfn /mnt/hyvol/nanochat-cache  /var/lib/hytorch/nanochat-cache
    ln -sfn /mnt/hyvol/arm-catalog     /var/lib/hytorch/arm-catalog
    ln -sfn /mnt/hyvol/arm-vanilla     /var/lib/hytorch/arm-vanilla
    # Ledger sync daemon: local -> volume every 300s. A torn WAL tail in
    # the snapshot is fine (Hyphae truncates on open; resume is bounded by
    # durable receipts anyway).
    cat > /root/ledger-sync.sh <<'SYNCEOF'
#!/usr/bin/env bash
# Consistent ledger snapshots with a BOUNDED seam pause (INCIDENT-901: an
# unbounded pause after a 5.4GB wired-step ingest outlived the barrier
# budget; §5.4 correctly killed the run at step 901, before its first
# checkpoint). Rules now:
#   - paused delta rsync is hard-capped at 60s (timeout); if it cannot
#     finish, CONT and let the NEXT cycle converge (rsync is incremental)
#   - the smeared pre-pass (seam running) does the bulk every cycle
#   - the last receipt head is sidecar-copied from seam.stderr so resume
#     citation never depends on a fresh ledger snapshot
while true; do
  rsync -a /var/lib/hytorch/nano-hyphae/ /mnt/hyvol/nano-hyphae/ 2>/dev/null
  grep -haoE \"head [0-9a-f]{64}\" /run/hytorch/nano-spool/seam.stderr 2>/dev/null | tail -1 \
    > /mnt/hyvol/stamps/catalog.last_head.tmp && \
    mv /mnt/hyvol/stamps/catalog.last_head.tmp /mnt/hyvol/stamps/catalog.last_head
  SEAM=\$(pgrep -f \"hytorch-seam /var\" | head -1)
  if [ -n \"\$SEAM\" ]; then
    trap \"kill -CONT \$SEAM 2>/dev/null\" EXIT
    kill -STOP \"\$SEAM\" 2>/dev/null
    timeout 60 rsync -a --delete /var/lib/hytorch/nano-hyphae/ /mnt/hyvol/nano-hyphae/ 2>/dev/null \
      && date -u +%FT%TZ > /mnt/hyvol/stamps/ledger.last_sync \
      || echo \"\$(date -u +%FT%TZ) delta exceeded 60s cap; retry next cycle\" >> /var/log/hytorch-ledger-sync.log
    kill -CONT \"\$SEAM\" 2>/dev/null
    trap - EXIT
  else
    rsync -a --delete /var/lib/hytorch/nano-hyphae/ /mnt/hyvol/nano-hyphae/ 2>/dev/null
    date -u +%FT%TZ > /mnt/hyvol/stamps/ledger.last_sync
  fi
  sleep 300
done
SYNCEOF
    chmod +x /root/ledger-sync.sh
    # NO pkill by name here: this ssh shell's OWN cmdline contains the
    # literal path (cat/chmod lines above), so any -f pattern that matches
    # the daemon also matches this shell and kills the session (rc=255 —
    # launches 2,3,5,6 all died here; the bracket-trick only shields the
    # pkill argument, not the rest of the script text). Pidfile instead.
    [ -f /run/ledger-sync.pid ] && { kill \$(cat /run/ledger-sync.pid) 2>/dev/null || true; }
    setsid nohup /root/ledger-sync.sh > /var/log/hytorch-ledger-sync.log 2>&1 &
    echo \$! > /run/ledger-sync.pid
    df -h /mnt/hyvol | tail -1
  "
  log "volume mounted; ledger on NVMe with snapshot daemon"
fi

"$(dirname "${BASH_SOURCE[0]}")/provision.sh" "$ID" "$COMMIT" >/dev/null
ARCH=$(hyssh "root@$IP" 'nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr " " "-" || (rocminfo 2>/dev/null | grep "Marketing Name" | grep -i instinct | head -1 | sed "s/.*: *//;s/ /-/g") || echo unknown')
OUT_DIR="$REPO_ROOT/results/nanochat/$ARCH/d${DEPTH}-${ARM}-$SHORT-$(date -u +%m%dT%H%M)"  # per-launch (review: same-commit reruns overwrote)
mkdir -p "$OUT_DIR"

D_SLOT=$((DEPTH))  # n_embd = depth*64 = 64 slots * depth dims

TORCHRUN="python3 -m"
SEP=""
[ "$NPROC" -gt 1 ] && TORCHRUN="torchrun --standalone --nproc_per_node=$NPROC -m" && SEP="--"

# d12/3.4 lessons baked in: (a) arms run DETACHED on the droplet with
# persistent logs; (b) per-arm NANOCHAT_BASE_DIR with shared dataset/tokenizer
# symlinks; (c) the local script POLLS until done; (d) phase 5: stamps on the
# volume make every stage idempotent — a preempted run re-runs this same
# script and it continues where the volume says it left off.
ARMS_SCRIPT=$(cat <<ARMEOF
#!/usr/bin/env bash
# NO set -e: a failed stage must still record its rc and touch DONE, or the
# local poller waits forever. Failure shows up in the rc + the stage log.
set -uo pipefail
cd /opt/nanochat
export OMP_NUM_THREADS=1
SHARED=/var/lib/hytorch/nanochat-cache
STAMPS=/mnt/hyvol/stamps
mkdir -p "\$STAMPS" 2>/dev/null || STAMPS=/var/lib/hytorch/stamps; mkdir -p "\$STAMPS"
mk_arm_dir() {
  local d="\$1"
  mkdir -p "\$d"
  ln -sfn "\$SHARED/base_data_climbmix" "\$d/base_data_climbmix" 2>/dev/null || true
  ln -sfn "\$SHARED/tokenizer" "\$d/tokenizer" 2>/dev/null || true
  # eval bundle: download ONCE into the shared cache (idempotent via stamp);
  # a dangling symlink would make base_eval skip its own download and die.
  if [ ! -f "\$STAMPS/evalbundle.done" ]; then
    ( cd /tmp && curl -fsSL -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip \
      && python3 -c "
import zipfile, shutil, tempfile, os
with tempfile.TemporaryDirectory() as t:
    zipfile.ZipFile('eval_bundle.zip').extractall(t)
    dst = '\$SHARED/eval_bundle'
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(os.path.join(t, 'eval_bundle'), dst)
print('eval_bundle placed')" && rm -f eval_bundle.zip && touch "\$STAMPS/evalbundle.done" ) || echo "eval_bundle download failed (base_eval will retry itself)"
  fi
  [ -d "\$SHARED/eval_bundle" ] && ln -sfn "\$SHARED/eval_bundle" "\$d/eval_bundle" 2>/dev/null || true
}
# torch.compile (Inductor) on ROCm 7.0/gfx950 produces SILENT NaNs from
# step ~2 in BOTH arms. Disable dynamo on ROCm (upstream-reportable).
DYNO=""
command -v rocminfo >/dev/null && DYNO="TORCHDYNAMO_DISABLE=1"
# infra.driver for RUN_START (SPEC_AMEND-001): ROCm or CUDA driver version.
DRIVER=\$( (cat /opt/rocm/.info/version 2>/dev/null | head -1 | sed "s/^/rocm-/") || (nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | sed "s/^/nvidia-/") || echo unknown)
[ -n "\$DRIVER" ] || DRIVER=unknown

CAT_ENVS() {  # common seam envs for any journaled stage; \$1 = run id
  echo "NANOCHAT_BASE_DIR=/var/lib/hytorch/arm-catalog \
  HYTORCH_CATALOG=1 \
  HYTORCH_WIRE_EVERY=$K_WIRE \
  HYTORCH_BARRIER_MS=$K_BARRIER \
  HYTORCH_SLUG=$SIZE HYTORCH_REGION=$REGION HYTORCH_PROC=$PROC HYTORCH_DRIVER=\$DRIVER \
  HYTORCH_ROOT=/opt/hytorch \
  HYTORCH_D_SLOT=$D_SLOT \
  HYTORCH_MANIFEST=/opt/hytorch/manifests/phase3-nanochat-d${DEPTH}.json \
  HYTORCH_BUILD_FACTS=/run/hytorch/build-facts.json \
  HYTORCH_DATA_DIR=/var/lib/hytorch/nano-hyphae \
  HYTORCH_SPOOL=/run/hytorch/nano-spool \
  HYTORCH_SEAM_BIN=/opt/hytorch/target/release/hytorch-seam \
  HYTORCH_RUN_ID=\$1 \
  NANOCHAT_DTYPE=bfloat16"
}

fresh_spool() {
  # A seam orphaned by a finished stage must die BEFORE the next stage's
  # seam opens the same ledger (single-writer engine lock).
  # kill by binary NAME (pgrep -x): a -f pattern can match a shell carrying
  # this text (INCIDENT-901 class). Never pkill -f in this repo again.
  for p in \$(pgrep -x hytorch-seam); do kill "\$p" 2>/dev/null; done; sleep 2
  rm -rf /run/hytorch/nano-spool; mkdir -p /run/hytorch/nano-spool
}

run_catalog() {
  echo "=== STAGE: base_train catalog ==="
  # Stamp encodes the horizon: extending a finished run to more iterations
  # is a RESUME (from its last checkpoint), not a skip.
  [ -f "\$STAMPS/base_catalog_${ITERS}.done" ] && { echo "stamped done, skip"; return 0; }
  fresh_spool
  mk_arm_dir /var/lib/hytorch/arm-catalog
  # Resume: highest checkpoint on the (volume-backed) arm dir, if any.
  RESUME_ARGS=""
  RUN_ID="nano-d${DEPTH}-catalog"
  LAST=\$(ls /var/lib/hytorch/arm-catalog/base_checkpoints/d${DEPTH}/model_*.pt 2>/dev/null | sed 's/.*model_0*//;s/\.pt//' | sort -n | tail -1)
  if [ -n "\$LAST" ] && [ -f "\$STAMPS/catalog.current_run" ]; then
    PARENT=\$(cat "\$STAMPS/catalog.current_run")
    # Parent's receipts cannot exceed its horizon; bound the ledger scan.
    MAXSTEP=$(( ITERS > 0 ? ITERS + 100 : 50000 ))
    PHEAD=\$(/opt/hytorch/target/release/hytorch-ledger-query /var/lib/hytorch/nano-hyphae last-head "\$PARENT" \$MAXSTEP 2>/dev/null | cut -d' ' -f2)
    RUN_ID="nano-d${DEPTH}-catalog-r\$LAST"
    RESUME_ARGS="--resume-from-step=\$LAST"
    export HYTORCH_PARENT_RUN="\$PARENT" HYTORCH_PARENT_HEAD="\${PHEAD:-unknown}" HYTORCH_RESUME_STEP="\$LAST"
    echo "RESUME catalog from step \$LAST (parent \$PARENT head \${PHEAD:-?})"
  else
    # fresh start: the ledger dir must be empty (a stale ledger without a
    # current_run stamp is a broken custody state — refuse to clobber).
    if [ -z "\$LAST" ]; then rm -rf /var/lib/hytorch/nano-hyphae/* 2>/dev/null; fi
  fi
  echo "\$RUN_ID" > "\$STAMPS/catalog.current_run"
  env \$DYNO \$(CAT_ENVS "\$RUN_ID") \
  $TORCHRUN scripts.base_train $SEP --depth=$DEPTH --num-iterations=$ITERS \
    --core-metric-every=-1 --sample-every=-1 --eval-every=$K_EVAL \
    --save-every=$K_SAVE \$RESUME_ARGS \
    --device-batch-size=$K_DBS --total-batch-size=$K_TBS --run=dummy \
    > /var/log/hytorch-arm-catalog.log 2>&1
  rc=\$?
  echo "catalog rc=\$rc"
  [ \$rc -eq 0 ] && touch "\$STAMPS/base_catalog_${ITERS}.done"
  return \$rc
}
run_vanilla() {
  echo "=== STAGE: base_train vanilla ==="
  [ -f "\$STAMPS/base_vanilla_${ITERS}.done" ] && { echo "stamped done, skip"; return 0; }
  mk_arm_dir /var/lib/hytorch/arm-vanilla
  RESUME_ARGS=""
  LAST=\$(ls /var/lib/hytorch/arm-vanilla/base_checkpoints/d${DEPTH}/model_*.pt 2>/dev/null | sed 's/.*model_0*//;s/\.pt//' | sort -n | tail -1)
  [ -n "\$LAST" ] && RESUME_ARGS="--resume-from-step=\$LAST" && echo "RESUME vanilla from \$LAST"
  env \$DYNO NANOCHAT_BASE_DIR=/var/lib/hytorch/arm-vanilla \
  NANOCHAT_DTYPE=bfloat16 \
  $TORCHRUN scripts.base_train $SEP --depth=$DEPTH --num-iterations=$ITERS \
    --core-metric-every=-1 --sample-every=-1 --eval-every=$K_EVAL \
    --save-every=$K_SAVE \$RESUME_ARGS \
    --device-batch-size=$K_DBS --total-batch-size=$K_TBS --run=dummy \
    > /var/log/hytorch-arm-vanilla.log 2>&1
  rc=\$?
  echo "vanilla rc=\$rc"
  [ \$rc -eq 0 ] && touch "\$STAMPS/base_vanilla_${ITERS}.done"
  return \$rc
}
run_base_eval() {  # \$1 = arm
  echo "=== STAGE: base_eval \$1 ==="
  [ -f "\$STAMPS/base_eval_\$1.done" ] && { echo "stamped done, skip"; return 0; }
  EXTRA=""
  [ "\$1" = catalog ] && EXTRA="HYTORCH_CATALOG=1 HYTORCH_ROOT=/opt/hytorch HYTORCH_D_SLOT=$D_SLOT"
  env \$DYNO NANOCHAT_BASE_DIR=/var/lib/hytorch/arm-\$1 \$EXTRA \
  NANOCHAT_DTYPE=bfloat16 \
  $TORCHRUN scripts.base_eval $SEP --device-batch-size=$K_DBS \
    > /var/log/hytorch-baseeval-\$1.log 2>&1
  rc=\$?
  echo "base_eval \$1 rc=\$rc"
  [ \$rc -eq 0 ] && touch "\$STAMPS/base_eval_\$1.done"
  return \$rc
}
run_sft() {  # \$1 = arm
  echo "=== STAGE: chat_sft \$1 ==="
  [ -f "\$STAMPS/sft_\$1.done" ] && { echo "stamped done, skip"; return 0; }
  if [ "\$1" = catalog ]; then
    fresh_spool
    env \$DYNO \$(CAT_ENVS "nano-d${DEPTH}-sft") \
    $TORCHRUN scripts.chat_sft $SEP --run=dummy \
      > /var/log/hytorch-sft-catalog.log 2>&1
  else
    env \$DYNO NANOCHAT_BASE_DIR=/var/lib/hytorch/arm-vanilla \
    NANOCHAT_DTYPE=bfloat16 \
    $TORCHRUN scripts.chat_sft $SEP --run=dummy \
      > /var/log/hytorch-sft-vanilla.log 2>&1
  fi
  rc=\$?
  echo "sft \$1 rc=\$rc"
  [ \$rc -eq 0 ] && touch "\$STAMPS/sft_\$1.done"
  return \$rc
}
run_chat_eval() {  # \$1 = arm
  echo "=== STAGE: chat_eval \$1 ==="
  [ -f "\$STAMPS/chat_eval_\$1.done" ] && { echo "stamped done, skip"; return 0; }
  EXTRA=""
  [ "\$1" = catalog ] && EXTRA="HYTORCH_CATALOG=1 HYTORCH_ROOT=/opt/hytorch HYTORCH_D_SLOT=$D_SLOT"
  env \$DYNO NANOCHAT_BASE_DIR=/var/lib/hytorch/arm-\$1 \$EXTRA \
  NANOCHAT_DTYPE=bfloat16 \
  $TORCHRUN scripts.chat_eval $SEP -i sft \
    > /var/log/hytorch-chateval-\$1.log 2>&1
  rc=\$?
  echo "chat_eval \$1 rc=\$rc"
  [ \$rc -eq 0 ] && touch "\$STAMPS/chat_eval_\$1.done"
  return \$rc
}

case "$ARM" in
  catalog) ARMS="catalog" ;;
  vanilla) ARMS="vanilla" ;;
  both)    ARMS="catalog vanilla" ;;
esac
STATUS=/var/log/hytorch-arms-status
rm -f "\$STATUS"
st() { echo "stage=\$1 rc=\$2" >> "\$STATUS"; }
for a in \$ARMS; do
  if [ "\$a" = catalog ]; then run_catalog; st base_catalog \$?; else run_vanilla; st base_vanilla \$?; fi
done
if [ "$K_STAGES" = speedrun ]; then
  for a in \$ARMS; do run_base_eval "\$a"; st base_eval_\$a \$?; done
  for a in \$ARMS; do run_sft "\$a"; st sft_\$a \$?; done
  for a in \$ARMS; do run_chat_eval "\$a"; st chat_eval_\$a \$?; done
fi
echo "all stages attempted"
cat "\$STATUS"
touch /var/log/hytorch-arms-DONE
ARMEOF
)

hyssh "root@$IP" "
  set -e -o pipefail
  cd /opt/hytorch
  export PATH=\$PATH:/usr/local/cuda/bin:/opt/rocm/bin
  export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-}

  echo '=== [1/4] gate (this silicon, both selection modes) ==='
  if command -v nvcc >/dev/null; then
    nvcc -O2 -fmad=false -arch=native kernels/cuda/diff_harness.cu kernels/cuda/refkernels.cu -o /tmp/diff_harness -ldl
  else
    for i in \$(seq 1 60); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 5; done
    apt-get install -y -q amdrocm-core-dev7.14 amdrocm-runtime-dev7.14 2>&1 | tail -1 || true
    hipcc -x hip -O2 -ffp-contract=off -fhip-fp32-correctly-rounded-divide-sqrt \
      kernels/cuda/diff_harness.cu kernels/cuda/refkernels.cu -o /tmp/diff_harness -ldl
  fi
  /tmp/diff_harness target/release/libapply_ref.so 100000

  echo '=== [2/4] python stack + nanochat ==='
  PY=\$(command -v python3)
  echo \"PY=\$PY\"
  if \$PY -m pip install -q --help 2>/dev/null | grep -q break-system-packages; then
    PIPQ=\"\$PY -m pip install -q --break-system-packages\"
  else PIPQ=\"\$PY -m pip install -q\"; fi
  \$PIPQ --ignore-installed numpy ninja tiktoken pyarrow wandb regex rustbpe psutil 2>&1 | tail -1
  if ! \$PY -c 'import torch' 2>/dev/null; then
    if command -v rocminfo >/dev/null; then
      timeout 900 \$PIPQ --ignore-installed torch --index-url https://download.pytorch.org/whl/rocm7.0 2>&1 | tail -1
    else
      timeout 900 \$PIPQ --ignore-installed torch 2>&1 | tail -1
    fi
  fi
  \$PY - <<'PYCHK'
import torch, tiktoken, numpy, pyarrow, rustbpe, psutil, regex
assert torch.cuda.is_available(), 'no device visible'
print(torch.__version__, torch.cuda.device_count(), 'gpus; imports ok')
PYCHK

  bash /opt/hytorch/nanochat/apply-patch.sh /opt/nanochat-work
  rm -rf /opt/nanochat && mv /opt/nanochat-work/nanochat /opt/nanochat
  cd /opt/nanochat
  \$PIPQ rustbpe psutil datasets 2>&1 | tail -1 || true

  echo '=== [2b/4] dataset shards + tokenizer (idempotent via stamps) ==='
  export NANOCHAT_BASE_DIR=/var/lib/hytorch/nanochat-cache
  export OMP_NUM_THREADS=1
  mkdir -p \$NANOCHAT_BASE_DIR
  STAMPS=/mnt/hyvol/stamps; mkdir -p \$STAMPS 2>/dev/null || { STAMPS=/var/lib/hytorch/stamps; mkdir -p \$STAMPS; }
  if [ ! -f \$STAMPS/data_${K_SHARDS}.done ]; then
    python3 -m nanochat.dataset -n $K_SHARDS 2>&1 | tail -2
    touch \$STAMPS/data_${K_SHARDS}.done
  else echo 'dataset stamped done'; fi
  if [ ! -f \$STAMPS/tok.done ]; then
    python3 -m scripts.tok_train 2>&1 | tail -3
    python3 -m scripts.tok_eval 2>&1 | tail -3
    touch \$STAMPS/tok.done
  else echo 'tokenizer stamped done'; fi

  echo '=== [3/4] build facts + manifest ==='
  cd /opt/hytorch
  SO_HASH=\$(sha256sum target/release/libapply_ref.so | cut -d' ' -f1)
  TORCH=\$(python3 -c 'import torch; print(f\"torch-{torch.__version__}\")')
  BACKEND=\$(python3 -c 'import torch; print(f\"cuda-{torch.version.cuda}\" if torch.version.cuda else f\"rocm-{torch.version.hip}\")')
  mkdir -p /run/hytorch /var/lib/hytorch
  printf '{\"build.apply_ref_hash\":\"%s\",\"build.harness_commit\":\"%s\",\"build.torch_wheel\":\"%s\",\"build.backend_wheel\":\"%s\"}\n' \
    \"\$SO_HASH\" \"$COMMIT\" \"\$TORCH\" \"\$BACKEND\" > /run/hytorch/build-facts.json
  python3 - <<PYEOF
import json
m = json.load(open('manifests/phase3-nanochat-template.json'))
m['model']['d_model'] = $DEPTH * 64
m['model']['d_slot'] = $DEPTH
m['model']['n_layers'] = $DEPTH
m['\$depth'] = $DEPTH
json.dump(m, open('manifests/phase3-nanochat-d${DEPTH}.json', 'w'), indent=2)
print('manifest for d$DEPTH written')
PYEOF

  echo '=== [4/4] stages (detached; local script polls) ==='
  rm -f /var/log/hytorch-arms-DONE
  cat > /root/run-arms.sh <<'ARMS_SH'
$ARMS_SCRIPT
ARMS_SH
  chmod +x /root/run-arms.sh
  setsid nohup /root/run-arms.sh > /var/log/hytorch-arms.log 2>&1 &
  echo 'arms launched detached'
" | tee "$OUT_DIR/nano.log"

# Poll until the arms finish (survives local ssh hiccups; state is remote).
# Distinguish "ssh hiccup" from "droplet PREEMPTED": if doctl says the
# droplet is gone, exit 75 — the supervisor relaunches (volume has state).
log "polling for completion (logs live on the droplet in /var/log/hytorch-*)"
while true; do
  sleep 120
  if hyssh "root@$IP" 'test -f /var/log/hytorch-arms-DONE' 2>/dev/null; then
    break
  fi
  if ! doctl compute droplet get "$ID" --format ID --no-header >/dev/null 2>&1; then
    trap - EXIT
    log "droplet $ID VANISHED (spot preemption) — exit 75 for supervisor relaunch"
    exit 75
  fi
  hyssh "root@$IP" 'tail -1 /run/hytorch/nano-spool/seam.stderr 2>/dev/null; grep -haE "step [0-9]{5}/" /var/log/hytorch-arm-*.log 2>/dev/null | tail -1' 2>/dev/null | tail -2 || true
done

# Any failed stage (rc!=0 in the status file) → exit 75: the supervisor
# relaunches and stamps skip what succeeded. A rc=1 stage must NOT be
# swallowed by a green DONE (phase-5 launch 1: catalog died at step 100,
# vanilla kept going, DONE looked normal).
if hyssh "root@$IP" 'grep -q "rc=[^0]" /var/log/hytorch-arms-status' 2>/dev/null; then
  scp -i "$SSH_ID" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    "root@$IP:/var/log/hytorch-*.log" "$OUT_DIR/" 2>/dev/null || true
  hyssh "root@$IP" 'cat /var/log/hytorch-arms-status' | tee -a "$OUT_DIR/nano.log" || true
  trap - EXIT
  log "one or more stages FAILED (status above; logs in $OUT_DIR) — exit 75 for relaunch; droplet 'hytorch-nano' kept for 10min forensics? NO: volume has state, kill it"
  doctl compute droplet delete "$ID" --force || true
  exit 75
fi

# Custody: pull stage logs + seam stderr + checkpoints metadata
# (INCIDENT-901: seam.stderr lived in tmpfs and died with the droplet).
scp -i "$SSH_ID" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  "root@$IP:/var/log/hytorch-*.log" "$OUT_DIR/" 2>/dev/null || true
scp -i "$SSH_ID" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  "root@$IP:/run/hytorch/nano-spool/seam.stderr" "$OUT_DIR/seam.stderr" 2>/dev/null || true
hyssh "root@$IP" 'for a in catalog vanilla; do for m in /var/lib/hytorch/arm-$a/*_checkpoints/*/meta_*.json; do [ -f "$m" ] && echo "== $a $m" && cat "$m"; done; done' \
  > "$OUT_DIR/checkpoints-meta.txt" 2>/dev/null || true

log "nanochat run done → $OUT_DIR"
