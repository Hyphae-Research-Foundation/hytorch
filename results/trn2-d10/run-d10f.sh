#!/usr/bin/env bash
# d10 a64 (60M), take 6: catalog with explicit tie-break in the two-stage proposer (take 5 collapsed
# into slot 0 from step 0 — every score ties at nanochat's zero-init — and the preregistered guard
# killed it at step 149). Then vanilla (cached NEFFs).
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
cd /opt/trn
B_PID=115811; TR_PID=125408
NEFF=/var/tmp/neuron-compile-cache/neuronxcc-2.26.6360.0+6f180f47/MODULE_18108742649528384371+839ddec0/model.neff
while kill -0 $TR_PID 2>/dev/null && [ ! -f "$NEFF" ]; do sleep 30; done
echo "vanilla5 NEFF 1810... present=$([ -f $NEFF ] && echo yes || echo no) at $(date -u +%FT%TZ)"
# whatever vanilla5 is compiling right now will leave a stale lock once killed: note it by name
INFLIGHT=$(pgrep -af "neuronx-cc compile" | grep -v pgrep | grep -o "MODULE_[0-9]*+839ddec0" | sort -u)
echo "in-flight at kill: ${INFLIGHT:-none}; stopping vanilla5"
kill $B_PID 2>/dev/null; sleep 1; kill $TR_PID 2>/dev/null
for i in $(seq 1 60); do kill -0 $TR_PID 2>/dev/null || break; sleep 2; done
kill -0 $TR_PID 2>/dev/null && kill -9 $TR_PID
sleep 10
for m in $INFLIGHT; do
  pgrep -x walrus_driver >/dev/null || rm -v "/var/tmp/neuron-compile-cache/neuronxcc-2.26.6360.0+6f180f47/$m/model.hlo_module.pb.lock"
done
export PJRT_DEVICE=NEURON NEURON_LOGICAL_NC_CONFIG=2 NEURON_CC_FLAGS="--model-type=transformer --optlevel=1 --enable-saturate-infinity --lnc=2"
export EXP_DEPTH=10 EXP_ASPECT=64 EXP_TRAIN_SEQLEN=1024 EXP_DBS=4 EXP_TBS=32768 EXP_SPLIT_GRAPH=1 EXP_PROGRESS_MODE=steps
COMMON="--num-steps 400 --max-train-seconds 36000 --no-eval-public --device-type xla"
mkdir -p /opt/hyt/d10/hyphae6 /opt/hyt/d10/spool6
echo "=== ARM catalog (policy 7, two-stage top-M w/ tie-break, flat apply+clip, host cb grad) $(date -u +%FT%TZ)"
env HYTORCH_CATALOG=1 HYTORCH_ROOT=/opt/hytorch HYTORCH_D_SLOT=10 HYTORCH_NF=32768 HYTORCH_K=8 HYTORCH_M_CANDIDATES=32 HYTORCH_POLICY_ID=7 HYTORCH_PROPOSE_TOPK=two_stage \
  HYTORCH_MANIFEST=/opt/hyt/manifest-d10.json HYTORCH_BUILD_FACTS=/opt/hyt/build-facts.json HYTORCH_DATA_DIR=/opt/hyt/d10/hyphae6 HYTORCH_SPOOL=/opt/hyt/d10/spool6 \
  HYTORCH_SEAM_BIN=/opt/hytorch/target/release/hytorch-seam HYTORCH_RUN_ID=trn2-d10-catalog HYTORCH_BARRIER_MS=600000 HYTORCH_WIRE_EVERY=50 \
  torchrun --standalone --nproc_per_node=4 train.py $COMMON > /opt/hyt/d10/catalog6.log 2>&1
echo "catalog rc=$? $(date -u +%FT%TZ)"
echo "=== ARM vanilla $(date -u +%FT%TZ)"
torchrun --standalone --nproc_per_node=4 train.py $COMMON > /opt/hyt/d10/vanilla6.log 2>&1
echo "vanilla rc=$? $(date -u +%FT%TZ)"
echo D10F-DONE
