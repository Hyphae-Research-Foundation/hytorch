#!/usr/bin/env bash
# Re-run the vanilla twin ONLY to recover its final.pt (train.py writes out/final.pt and the catalog
# run overwrote the twin's). All 6 vanilla NEFFs are cached: ~30 min. Own --out-dir this time.
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
cd /opt/trn
export PJRT_DEVICE=NEURON NEURON_LOGICAL_NC_CONFIG=2 NEURON_CC_FLAGS="--model-type=transformer --optlevel=1 --enable-saturate-infinity --lnc=2"
export EXP_DEPTH=10 EXP_ASPECT=64 EXP_TRAIN_SEQLEN=1024 EXP_DBS=4 EXP_TBS=32768 EXP_SPLIT_GRAPH=1 EXP_PROGRESS_MODE=steps
COMMON="--num-steps 400 --max-train-seconds 36000 --no-eval-public --device-type xla"
echo "=== ARM vanilla (checkpoint recovery) $(date -u +%FT%TZ)"
torchrun --standalone --nproc_per_node=4 train.py $COMMON --out-dir /opt/hyt/d10/out-vanilla7 > /opt/hyt/d10/vanilla7.log 2>&1
echo "vanilla rc=$? $(date -u +%FT%TZ)"
sha256sum /opt/hyt/d10/out-vanilla7/final.pt | tee /opt/hyt/d10/vanilla7-final.sha256
echo D10H-DONE
