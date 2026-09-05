#!/usr/bin/env bash
# d10 a64 (60M) — both arms, 400 steps each at TBS 32768 / DBS 4 (same shapes as run-d10.sh
# so the vanilla NEFF cache is reused). Differences vs run-d10.sh:
#   * waits for the old run-d10.sh (pid 32228) to exit before touching the cores
#   * --max-train-seconds 36000: budget_started precedes the loop, so compile time counts
#     (the first run's vanilla arm burned its 7200 s compiling one graph)
#   * catalog arm uses the two-stage proposer top-M (ISGV902 fix, commit dbc166e)
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
cd /opt/trn
OLD_PID=32228
while kill -0 $OLD_PID 2>/dev/null; do sleep 60; done
echo "old run exited at $(date -u +%FT%TZ)"
export PJRT_DEVICE=NEURON NEURON_LOGICAL_NC_CONFIG=2 NEURON_CC_FLAGS="--model-type=transformer --optlevel=1 --enable-saturate-infinity --lnc=2"
export EXP_DEPTH=10 EXP_ASPECT=64 EXP_TRAIN_SEQLEN=1024 EXP_DBS=4 EXP_TBS=32768 EXP_SPLIT_GRAPH=1 EXP_PROGRESS_MODE=steps
COMMON="--num-steps 400 --max-train-seconds 36000 --no-eval-public --device-type xla"
echo "=== ARM catalog (policy 7, two-stage top-M) $(date -u +%FT%TZ)"
env HYTORCH_CATALOG=1 HYTORCH_ROOT=/opt/hytorch HYTORCH_D_SLOT=10 HYTORCH_NF=32768 HYTORCH_K=8 HYTORCH_M_CANDIDATES=32 HYTORCH_POLICY_ID=7 HYTORCH_PROPOSE_TOPK=two_stage \
  HYTORCH_MANIFEST=/opt/hyt/manifest-d10.json HYTORCH_BUILD_FACTS=/opt/hyt/build-facts.json HYTORCH_DATA_DIR=/opt/hyt/d10/hyphae HYTORCH_SPOOL=/opt/hyt/d10/spool \
  HYTORCH_SEAM_BIN=/opt/hytorch/target/release/hytorch-seam HYTORCH_RUN_ID=trn2-d10-catalog HYTORCH_BARRIER_MS=600000 HYTORCH_WIRE_EVERY=50 \
  torchrun --standalone --nproc_per_node=4 train.py $COMMON > /opt/hyt/d10/catalog2.log 2>&1
echo "catalog rc=$? $(date -u +%FT%TZ)"
echo "=== ARM vanilla $(date -u +%FT%TZ)"
torchrun --standalone --nproc_per_node=4 train.py $COMMON > /opt/hyt/d10/vanilla2.log 2>&1
echo "vanilla rc=$? $(date -u +%FT%TZ)"
echo D10B-DONE
