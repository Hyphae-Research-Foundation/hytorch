#!/usr/bin/env bash
# nanochat/apply-patch.sh — clones nanochat at a pinned commit and applies
# the hytorch law-0 integration (run ON the droplet; the laptop only edits).
#
# Usage: bash nanochat/apply-patch.sh <workdir>
set -euo pipefail
WORK="${1:?workdir}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN="${NANOCHAT_PIN:-92d63d4e8bb4df75c3b71618f31ddde2378b2bcd}"  # pinned (review 2026-09-02: was master)

mkdir -p "$WORK"
if [ ! -d "$WORK/nanochat" ]; then
  git clone -q https://github.com/karpathy/nanochat "$WORK/nanochat"
fi
cd "$WORK/nanochat"
git fetch -q origin "$PIN" 2>/dev/null || true
git checkout -q "$PIN"
NANOCHAT_COMMIT=$(git rev-parse HEAD)
echo "nanochat pinned at $NANOCHAT_COMMIT"

# The patch is additive: nanochat/hytorch_bridge.py + a guarded edit to
# gpt.py Block.forward. Vanilla behavior is bit-unchanged when
# HYTORCH_CATALOG is unset (the baseline arm IS this same tree).
cp "$PATCH_DIR/hytorch_bridge.py" nanochat/hytorch_bridge.py

python3 - <<'EOF'
import re

p = "nanochat/gpt.py"
s = open(p).read()
old = """    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x"""
new = """    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # hytorch law-0 seam: when the catalog is active, blocks PROPOSE and
        # HyphaeWrite is the only writer of x (two write units per layer).
        # With HYTORCH_CATALOG unset this is bit-identical to vanilla.
        from nanochat.hytorch_bridge import catalog_write
        d_attn = self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = catalog_write(x, d_attn, self.layer_idx, unit=0)
        d_mlp = self.mlp(norm(x))
        x = catalog_write(x, d_mlp, self.layer_idx, unit=1)
        return x"""
assert old in s, "nanochat Block.forward changed upstream; re-pin and update patch"
s = s.replace(old, new)
# Block needs layer_idx (upstream passes it to attn only).
s = s.replace("""class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)""",
"""class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config, layer_idx)""")
open(p, "w").write(s)
print("gpt.py patched (law-0 seam, vanilla-identical when disabled)")
EOF

cp "$PATCH_DIR/seam_shim.py" nanochat/seam_shim.py

python3 - <<'EOF'
# Patch scripts/base_train.py: attach codebook + seam, gate optimizer.step().
# All guarded: HYTORCH_CATALOG unset => vanilla bit-identical.
p = "scripts/base_train.py"
s = open(p).read()

# 1. Attach the codebook BEFORE optimizer setup + compile (it must be a
#    registered parameter so Muon/AdamW grouping sees it — we add it to the
#    AdamW 'embedding-like' path via setup_optimizer's assert bypass: nanochat
#    asserts total param count, so we attach AFTER setup_optimizer and give
#    it its own AdamW group instead).
old = """orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe"""
new = """orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
import os as _os
if _os.environ.get("HYTORCH_CATALOG", "") == "1":
    # dynamo cannot trace the hytorch pybind kernels: every layer graph-breaks
    # and the dual (compiled+eager) buffers OOM. The catalog arm runs eager —
    # our kernels are native anyway; declared in the manifest.
    print("hytorch: catalog arm runs EAGER (torch.compile skipped)")
else:
    model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe"""
assert old in s, "compile anchor drifted"
s = s.replace(old, new)

# 1b. Resume path: the catalog checkpoint carries hytorch_codebook, but the
#     bare GPT has no such param at load time — pop it BEFORE the strict
#     load, re-attach with init_from after setup_optimizer (phase 5).
old = """    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)"""
new = """    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    import os as _os0
    if _os0.environ.get("HYTORCH_CATALOG", "") == "1":
        from nanochat.hytorch_bridge import pop_codebook
        pop_codebook(model_data)   # re-attached after setup_optimizer
    model.load_state_dict(model_data, strict=True, assign=True)"""
assert old in s, "resume load anchor drifted"
s = s.replace(old, new)

# 2. Give the codebook its own AdamW param group after setup_optimizer.
old = """if resuming:
    optimizer.load_state_dict(optimizer_data)"""
new = """# --- hytorch: attach codebook AFTER num_scaling_params/setup_optimizer
# (nanochat asserts full param coverage in both; the codebook gets its own
# AdamW group here instead). On resume, init_from restores the checkpointed
# codebook; the group is added BEFORE optimizer.load_state_dict so the group
# counts line up (11 == 11). ---
if _os.environ.get("HYTORCH_CATALOG", "") == "1":
    from nanochat.hytorch_bridge import attach_codebook, stashed_codebook
    attach_codebook(orig_model, orig_model.config.n_embd,
                    device=device, dtype=torch.float32,
                    init_from=stashed_codebook())
    optimizer.add_param_group(dict(
        kind='adamw', params=[orig_model.hytorch_codebook],
        lr=args.embedding_lr * 0.1, betas=(0.9, 0.95), eps=1e-10,
        weight_decay=0.0, initial_lr=args.embedding_lr * 0.1))

if resuming:
    optimizer.load_state_dict(optimizer_data)"""
assert old in s, "optimizer anchor drifted"
s = s.replace(old, new)

# 3. Seam creation right before the training loop (after dist init exists).
old = """# -----------------------------------------------------------------------------
# Compile the model"""
new = """# -----------------------------------------------------------------------------
# hytorch seam shim (created lazily at first step; import here)
from nanochat.seam_shim import SeamShim as _HytorchSeam
_hytorch_seam = None

# -----------------------------------------------------------------------------
# Compile the model"""
assert old in s
s = s.replace(old, new)

# 3b. Phase 5: save_every checkpoints on the catalog arm carry the codebook
#     automatically (registered param). Nothing to patch for saving.

# 4. Gate optimizer.step() with the collective barrier + step chain after.
old = """    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)"""
new = """    # --- hytorch: collective barrier gates the optimizer (spec §5.4) ---
    if _os.environ.get("HYTORCH_CATALOG", "") == "1":
        from nanochat.hytorch_bridge import drain_journal
        if _hytorch_seam is None:
            import hashlib as _hl, json as _json
            _man_path = _os.environ["HYTORCH_MANIFEST"]
            _man_sha = _hl.sha256(open(_man_path, "rb").read()).hexdigest()
            _build = _json.load(open(_os.environ["HYTORCH_BUILD_FACTS"]))
            _pol_env = dict(policy_id=int(_os.environ.get("HYTORCH_POLICY_ID", "6")),
                            k=int(_os.environ.get("HYTORCH_K", "8")),
                            s_slots=64,
                            n_features=int(_os.environ.get("HYTORCH_NF", "32768")),
                            mag_max=float(_os.environ.get("HYTORCH_MAG_MAX", "64.0")),
                            selection="global_topk")
            from nanochat.hytorch_bridge import resolve_proposal_clip as _rpc
            _pol_env["proposal_clip"] = _rpc() or 0.0
            _hytorch_seam = _HytorchSeam.maybe_create(
                run_id=_os.environ.get("HYTORCH_RUN_ID", "nanochat-run"),
                data_dir=_os.environ["HYTORCH_DATA_DIR"],
                spool=_os.environ["HYTORCH_SPOOL"],
                seam_bin=_os.environ["HYTORCH_SEAM_BIN"],
                manifest_sha=_man_sha, build=_build, policy=_pol_env)
        _entries = drain_journal()
        _hytorch_seam.step_barrier(step, _entries, orig_model.hytorch_codebook)
        # Codebook grad norm for the STEP record (phase 5). nanochat reduces
        # grads INSIDE optimizer.step(), so pre-step grads are rank-local;
        # for the one parameter the ledger governs we all-reduce its grad
        # explicitly (2.6MB fp32 — trivial) and record the EXACT norm of the
        # averaged gradient that will move C from c_prev to c_next.
        with torch.no_grad():
            _cbg = orig_model.hytorch_codebook.grad
            if _cbg is None:
                _hytorch_grad_norm = 0.0
            else:
                _cbg = _cbg.float().clone()
                if is_ddp_initialized():
                    dist.all_reduce(_cbg, op=dist.ReduceOp.AVG)
                _hytorch_grad_norm = float(_cbg.norm().item())
    if scaler is not None:
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if _hytorch_seam is not None:
        import sys as _sys
        _sys.path.insert(0, _os.path.join(_os.environ["HYTORCH_ROOT"], "python"))
        from hytorch.model import renorm_or_reset as _rr
        import numpy as _np
        _dead = _rr(orig_model.hytorch_codebook,
                    float(_os.environ.get("HYTORCH_MIN_NORM", "0.00390625")),
                    _np.random.default_rng(1337 + step))
        # STEP record carries the ACTUAL lr of the codebook's own group (as
        # scheduled this step) and the reduced codebook grad norm — read from
        # the optimizer, never recomputed (phase 5: no cosmetic zeros).
        _cb_lr = next(g["lr"] for g in optimizer.param_groups
                      if any(p is orig_model.hytorch_codebook for p in g["params"]))
        # Law-0 bypass declaration (review 2026-09-02): nanochat mutates h
        # outside the catalog (per-layer resid/x0 lambdas + smear/backout
        # gates). They are journalized as a BYPASS fact every step.
        _bypass = {"resid_lambdas": orig_model.resid_lambdas,
                   "x0_lambdas": orig_model.x0_lambdas}
        for _n in ("smear_lambda", "backout_lambda"):
            if hasattr(orig_model, _n):
                _bypass[_n] = getattr(orig_model, _n).reshape(-1)
        _hytorch_seam.step_chain(step, _cb_lr, _hytorch_grad_norm,
                                 orig_model.hytorch_codebook, resets=_dead,
                                 bypass=_bypass)
    model.zero_grad(set_to_none=True)"""
assert old in s, "optimizer.step anchor drifted"
s = s.replace(old, new)
open(p, "w").write(s)
print("base_train.py patched: codebook param group + collective barrier + step chain")
EOF

python3 - <<'EOF'
# Patch nanochat/checkpoint_manager.py build_model: catalog checkpoints carry
# hytorch_codebook; the bare GPT can't strict-load it. Pop + stash before the
# load; inference entry points get the codebook via
# restore_codebook_for_inference (eval, SFT warm-start, chat_cli — phase 5).
p = "nanochat/checkpoint_manager.py"
s = open(p).read()
old = """    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)"""
new = """    model_config = GPTConfig(**model_config_kwargs)
    import os as _os
    if _os.environ.get("HYTORCH_CATALOG", "") == "1":
        from nanochat.hytorch_bridge import pop_codebook
        pop_codebook(model_data)   # stashed; consumers re-attach/restore
    _patch_missing_keys(model_data, model_config)"""
assert old in s, "build_model anchor drifted"
s = s.replace(old, new)

# Inference/eval consumers never call attach_codebook (no optimizer): the
# forward would die with codebook=None. Restore it right after the load.
old = """    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode"""
new = """    model.load_state_dict(model_data, strict=True, assign=True)
    if _os.environ.get("HYTORCH_CATALOG", "") == "1":
        from nanochat.hytorch_bridge import restore_codebook_for_inference
        restore_codebook_for_inference(device)
    # Put the model in the right training phase / mode"""
assert old in s, "build_model load anchor drifted"
s = s.replace(old, new)
open(p, "w").write(s)
print("checkpoint_manager.py patched: codebook pop on load (all loaders)")
EOF

python3 - <<'EOF'
# Patch scripts/chat_sft.py: SFT of a cataloged base model, fully journaled.
# Same seam protocol as base_train (§5.4): codebook re-attached from the
# base checkpoint, its own AdamW group, barrier gates optimizer.step(),
# STEP chain after. Run id distinguishes the stage: nano-<tag>-sft.
p = "scripts/chat_sft.py"
s = open(p).read()

# 0. eager on the catalog arm (dynamo can't trace the pybind kernels).
old = """orig_model = model
model = torch.compile(model, dynamic=False)"""
new = """orig_model = model
import os as _os
if _os.environ.get("HYTORCH_CATALOG", "") == "1":
    print0("hytorch: catalog SFT runs EAGER (torch.compile skipped)")
else:
    model = torch.compile(model, dynamic=False)"""
assert old in s, "sft compile anchor drifted"
s = s.replace(old, new)

# 1. attach codebook (from the base checkpoint, popped by build_model) with
#    its own AdamW group, BEFORE the optional optimizer warm-start load so
#    group counts match the base optimizer state (11 == 11).
old = """base_dir = get_base_dir()
if args.load_optimizer:"""
new = """base_dir = get_base_dir()
if _os.environ.get("HYTORCH_CATALOG", "") == "1":
    from nanochat.hytorch_bridge import attach_codebook, stashed_codebook
    _cb0 = stashed_codebook()
    assert _cb0 is not None, "catalog SFT requires the base codebook in the checkpoint"
    attach_codebook(orig_model, orig_model.config.n_embd,
                    device=device, dtype=torch.float32, init_from=_cb0)
    optimizer.add_param_group(dict(
        kind='adamw', params=[orig_model.hytorch_codebook],
        lr=args.embedding_lr * 0.1, betas=(0.9, 0.95), eps=1e-10,
        weight_decay=0.0, initial_lr=args.embedding_lr * 0.1))
if args.load_optimizer:"""
assert old in s, "sft optimizer anchor drifted"
s = s.replace(old, new)

# 2. seam import site.
old = """# SFT data mixture and DataLoader"""
new = """# hytorch seam shim (lazily created at first step)
from nanochat.seam_shim import SeamShim as _HytorchSeam
_hytorch_seam = None

# SFT data mixture and DataLoader"""
assert old in s
s = s.replace(old, new)

# 3. barrier + chain around optimizer.step() (single anchor in chat_sft).
old = """    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)"""
new = """    if _os.environ.get("HYTORCH_CATALOG", "") == "1":
        from nanochat.hytorch_bridge import drain_journal
        if _hytorch_seam is None:
            import hashlib as _hl, json as _json
            _man_path = _os.environ["HYTORCH_MANIFEST"]
            _man_sha = _hl.sha256(open(_man_path, "rb").read()).hexdigest()
            _build = _json.load(open(_os.environ["HYTORCH_BUILD_FACTS"]))
            _pol_env = dict(policy_id=int(_os.environ.get("HYTORCH_POLICY_ID", "6")),
                            k=int(_os.environ.get("HYTORCH_K", "8")),
                            s_slots=64,
                            n_features=int(_os.environ.get("HYTORCH_NF", "32768")),
                            mag_max=float(_os.environ.get("HYTORCH_MAG_MAX", "64.0")),
                            selection="global_topk")
            from nanochat.hytorch_bridge import resolve_proposal_clip as _rpc
            _pol_env["proposal_clip"] = _rpc() or 0.0
            _hytorch_seam = _HytorchSeam.maybe_create(
                run_id=_os.environ.get("HYTORCH_RUN_ID", "nanochat-sft"),
                data_dir=_os.environ["HYTORCH_DATA_DIR"],
                spool=_os.environ["HYTORCH_SPOOL"],
                seam_bin=_os.environ["HYTORCH_SEAM_BIN"],
                manifest_sha=_man_sha, build=_build, policy=_pol_env)
        _entries = drain_journal()
        _hytorch_seam.step_barrier(step, _entries, orig_model.hytorch_codebook)
        with torch.no_grad():
            _cbg = orig_model.hytorch_codebook.grad
            if _cbg is None:
                _hytorch_grad_norm = 0.0
            else:
                _cbg = _cbg.float().clone()
                if is_ddp_initialized():
                    dist.all_reduce(_cbg, op=dist.ReduceOp.AVG)
                _hytorch_grad_norm = float(_cbg.norm().item())
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if _hytorch_seam is not None:
        import sys as _sys
        _sys.path.insert(0, _os.path.join(_os.environ["HYTORCH_ROOT"], "python"))
        from hytorch.model import renorm_or_reset as _rr
        import numpy as _np
        _dead = _rr(orig_model.hytorch_codebook,
                    float(_os.environ.get("HYTORCH_MIN_NORM", "0.00390625")),
                    _np.random.default_rng(1337 + step))
        _cb_lr = next(g["lr"] for g in optimizer.param_groups
                      if any(p is orig_model.hytorch_codebook for p in g["params"]))
        _bypass = {"resid_lambdas": orig_model.resid_lambdas,
                   "x0_lambdas": orig_model.x0_lambdas}
        _hytorch_seam.step_chain(step, _cb_lr, _hytorch_grad_norm,
                                 orig_model.hytorch_codebook, resets=_dead,
                                 bypass=_bypass)
    model.zero_grad(set_to_none=True)"""
assert old in s, "sft step anchor drifted"
s = s.replace(old, new)
open(p, "w").write(s)
print("chat_sft.py patched: journaled SFT (barrier + chain + codebook group)")
EOF

echo "$NANOCHAT_COMMIT" > .hytorch-nanochat-pin
echo "patch applied"
