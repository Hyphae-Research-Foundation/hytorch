#!/usr/bin/env bash
# d10 a64 (60M), take 7: catalog under the SECOND declared manifest (manifest-d10-w200.json: guard
# warm-up 100 -> 200, threshold and window unchanged; declared 13:20Z before this run). Runs AFTER
# run-d10f.sh (vanilla6) finishes on its own — no more interrupted compiles.
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
cd /opt/trn
F_PID=126933
while kill -0 $F_PID 2>/dev/null; do sleep 60; done
echo "run-d10f finished at $(date -u +%FT%TZ): $(tail -3 /opt/hyt/d10/arms6.log | tr '\n' ' ')"
export PJRT_DEVICE=NEURON NEURON_LOGICAL_NC_CONFIG=2 NEURON_CC_FLAGS="--model-type=transformer --optlevel=1 --enable-saturate-infinity --lnc=2"
export EXP_DEPTH=10 EXP_ASPECT=64 EXP_TRAIN_SEQLEN=1024 EXP_DBS=4 EXP_TBS=32768 EXP_SPLIT_GRAPH=1 EXP_PROGRESS_MODE=steps
COMMON="--num-steps 400 --max-train-seconds 36000 --no-eval-public --device-type xla"
mkdir -p /opt/hyt/d10/hyphae7 /opt/hyt/d10/spool7
echo "=== ARM catalog (manifest-d10-w200, run trn2-d10-catalog-w200) $(date -u +%FT%TZ)"
env HYTORCH_CATALOG=1 HYTORCH_ROOT=/opt/hytorch HYTORCH_D_SLOT=10 HYTORCH_NF=32768 HYTORCH_K=8 HYTORCH_M_CANDIDATES=32 HYTORCH_POLICY_ID=7 HYTORCH_PROPOSE_TOPK=two_stage \
  HYTORCH_MANIFEST=/opt/hyt/manifest-d10-w200.json HYTORCH_BUILD_FACTS=/opt/hyt/build-facts.json HYTORCH_DATA_DIR=/opt/hyt/d10/hyphae7 HYTORCH_SPOOL=/opt/hyt/d10/spool7 \
  HYTORCH_SEAM_BIN=/opt/hytorch/target/release/hytorch-seam HYTORCH_RUN_ID=trn2-d10-catalog-w200 HYTORCH_BARRIER_MS=600000 HYTORCH_WIRE_EVERY=50 \
  torchrun --standalone --nproc_per_node=4 train.py $COMMON > /opt/hyt/d10/catalog7.log 2>&1
echo "catalog rc=$? $(date -u +%FT%TZ)"
echo D10G-DONE
