#!/usr/bin/env bash
# d10 a64 (60M) — both arms, fixed 400 steps each at TBS 65536 (the trainium project sizing point)
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
cd /opt/trn
export PJRT_DEVICE=NEURON NEURON_LOGICAL_NC_CONFIG=2 NEURON_CC_FLAGS="--model-type=transformer --optlevel=1 --enable-saturate-infinity --lnc=2"
export EXP_DEPTH=10 EXP_ASPECT=64 EXP_TRAIN_SEQLEN=1024 EXP_DBS=4 EXP_TBS=32768 EXP_SPLIT_GRAPH=1 EXP_PROGRESS_MODE=steps
COMMON="--num-steps 400 --max-train-seconds 7200 --no-eval-public --device-type xla"
echo "=== ARM catalog (policy 7)"
env HYTORCH_CATALOG=1 HYTORCH_ROOT=/opt/hytorch HYTORCH_D_SLOT=10 HYTORCH_NF=32768 HYTORCH_K=8 HYTORCH_M_CANDIDATES=32 HYTORCH_POLICY_ID=7   HYTORCH_MANIFEST=/opt/hyt/manifest-d10.json HYTORCH_BUILD_FACTS=/opt/hyt/build-facts.json HYTORCH_DATA_DIR=/opt/hyt/d10/hyphae HYTORCH_SPOOL=/opt/hyt/d10/spool   HYTORCH_SEAM_BIN=/opt/hytorch/target/release/hytorch-seam HYTORCH_RUN_ID=trn2-d10-catalog HYTORCH_BARRIER_MS=600000 HYTORCH_WIRE_EVERY=50   torchrun --standalone --nproc_per_node=4 train.py $COMMON > /opt/hyt/d10/catalog.log 2>&1
echo "catalog rc=$?"
echo "=== ARM vanilla"
torchrun --standalone --nproc_per_node=4 train.py $COMMON > /opt/hyt/d10/vanilla.log 2>&1
echo "vanilla rc=$?"
echo D10-DONE
