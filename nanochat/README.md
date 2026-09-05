# nanochat/ — integration with a real trainer

hytorch enters [nanochat](https://github.com/karpathy/nanochat) by an additive patch
against a pinned upstream commit (`apply-patch.sh`): a bridge module plus guarded edits to
`scripts/base_train.py`. With `HYTORCH_CATALOG` unset the tree is bit-identical to upstream
— the dense-twin arm is this same tree.

- `hytorch_bridge.py` — `catalog_write(x, delta, layer_idx, unit)`: the only writer of the residual when active; codebook attachment; journal drain
- `seam_shim.py` — DDP-aware seam client: rank-collective barrier gating `optimizer.step()`, STEP chain, `BYPASS` facts for nanochat's residual scalars, the executable preregistered guard (`_guard_update`: commit rate + usage entropy from the manifest, kills all ranks)
- `sft_probe.py` — SFT stage with the channel live
- `INTEGRATION.md` — the phase-3 integration plan (Spanish)
