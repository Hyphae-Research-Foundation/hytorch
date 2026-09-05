# trn/ — AWS Trainium2 integration

Same shape as `nanochat/`: an additive bridge plus guarded edits to the Neuron training
tree, applied by `apply-patch.sh` against a pinned commit. With `HYTORCH_CATALOG` unset
the tree is bit-identical to upstream and is the dense-twin arm.

- `trn_bridge.py` — `catalog_write` (the only writer when active), codebook attachment, `sync_codebook_grad` (host-accumulated codebook gradient, gloo all-reduce, uploaded once after the flat bucket)
- `apply-patch.sh` — six edits to `train.py`: block seam, codebook param group, seam handle, barrier/chain around `optimizer.step`, codebook-grad sync at both `GradBucket.all_reduce` call sites, codebook excluded from the bucket
- `tools/` — compile-only bisection of Neuron compiler failures without a NeuronCore: trace under `PJRT_DEVICE=CPU`, dump the HLO proto, feed `neuronx-cc`. `hlo_dump.py` decodes a cached `model.hlo_module.pb`; `isgv_bisect.py`, `topk_probe.py`, `d10_probe.py` are the probes that found findings 4–8 in `results/gates/AWS-Trainium2/NOTES.md`
- `runs/` — the d10 launchers, takes b–h (each header says what changed and why)
