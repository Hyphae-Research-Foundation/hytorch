#!/usr/bin/env bash
# trn/apply-patch.sh — hytorch law-0 integration into the TRAINIUM train.py
# (the NKI-frontier tree at ~/Documents/trainium, pinned by commit).
#
# Same shape as nanochat/apply-patch.sh: additive bridge module + guarded
# edits. Vanilla behavior is bit-unchanged when HYTORCH_CATALOG is unset —
# the baseline arm IS this same tree.
#
# Usage: bash trn/apply-patch.sh <trainium-repo-dir>
set -euo pipefail
WORK="${1:?trainium repo dir}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$WORK"
TRN_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unpinned")
echo "trainium tree at $TRN_COMMIT"

cp "$PATCH_DIR/../nanochat/seam_shim.py" seam_shim.py
cp "$PATCH_DIR/trn_bridge.py" trn_bridge.py

python3 - <<'EOF'
import re

p = "train.py"
s = open(p).read()

# 1. Block.forward: catalog seam — the ONLY writer of x when active.
old = """    def forward(self, x: torch.Tensor, x0: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                ve: torch.Tensor | None = None) -> torch.Tensor:
        x = self.resid_lambda * x + self.x0_lambda * x0
        gated_ve = self.ve_lambda * ve if (ve is not None and self.ve_lambda is not None) else None
        x = x + self.attn(norm(x), cos, sin, ve=gated_ve)
        x = x + self.mlp(norm(x))
        return x"""
new = """    def forward(self, x: torch.Tensor, x0: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                ve: torch.Tensor | None = None) -> torch.Tensor:
        # hytorch law-0 seam (SPEC_AMEND-004 two-phase on Trainium): blocks
        # PROPOSE; the catalog is the only writer of x. HYTORCH_CATALOG unset
        # => bit-identical to vanilla (this same tree is the baseline arm).
        from trn_bridge import catalog_write
        x = self.resid_lambda * x + self.x0_lambda * x0
        gated_ve = self.ve_lambda * ve if (ve is not None and self.ve_lambda is not None) else None
        d_attn = self.attn(norm(x), cos, sin, ve=gated_ve)
        x = catalog_write(x, d_attn, self.layer_idx, unit=0)
        d_mlp = self.mlp(norm(x))
        x = catalog_write(x, d_mlp, self.layer_idx, unit=1)
        return x"""
assert old in s, "Block.forward anchor drifted"
s = s.replace(old, new)

# 1b. Block needs layer_idx.
old = """class Block(nn.Module):
    def __init__(self, config: GPTConfig, has_ve: bool = False):
        super().__init__()
        self.attn = CausalSelfAttention(config)"""
new = """class Block(nn.Module):
    def __init__(self, config: GPTConfig, has_ve: bool = False, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = CausalSelfAttention(config)"""
assert old in s, "Block.__init__ anchor drifted"
s = s.replace(old, new)
old = """                Block(config, has_ve=(self.ve_map[i] is not None))
                for i in range(config.n_layer)"""
new = """                Block(config, has_ve=(self.ve_map[i] is not None), layer_idx=i)
                for i in range(config.n_layer)"""
assert old in s, "Block construction anchor drifted"
s = s.replace(old, new)

# 2. Codebook param group after setup_optimizer (own AdamW group).
old = """    optimizer = model.setup_optimizer()
    optimizer.init_state()          # all state buffers exist before step 0
    sched = Schedule(device)"""
new = """    optimizer = model.setup_optimizer()
    import os as _os
    if _os.environ.get("HYTORCH_CATALOG", "") == "1":
        from trn_bridge import attach_codebook
        attach_codebook(orig_model, config.n_embd, device=device,
                        dtype=torch.float32)
        optimizer.add_param_group(dict(
            kind="adamw", params=[orig_model.hytorch_codebook],
            lr=EMBEDDING_LR * 0.1, betas=(0.9, 0.95), eps=1e-10,
            weight_decay=0.0, initial_lr=EMBEDDING_LR * 0.1))
    optimizer.init_state()          # all state buffers exist before step 0
    sched = Schedule(device)"""
assert old in s, "optimizer anchor drifted"
s = s.replace(old, new)

# 3. Seam import + lazy handle before the training loop.
old = """    total_train_tokens = 0
    step = 0"""
new = """    from seam_shim import SeamShim as _HytorchSeam
    _hytorch_seam = None
    # Control plane for the seam on XLA: gloo group over the same ranks
    # (xla PG lacks barrier/broadcast_object_list semantics we rely on).
    _hytorch_pg = None
    if _os.environ.get("HYTORCH_CATALOG", "") == "1" and dist.is_initialized():
        _hytorch_pg = dist.new_group(backend="gloo")

    total_train_tokens = 0
    step = 0"""
assert old in s, "loop-state anchor drifted"
s = s.replace(old, new)

# 4. Barrier gates optimizer.step(); chain after. The trainium loop has a
#    single optimizer.step(sched) call site.
old = """        optimizer.step(sched)
        # Cut the lazy graph exactly once per optimizer step: fwd+bwd+allreduce
        # +optim compile into ONE cached NEFF executed repeatedly.
        step_boundary(device)"""
new = """        if _os.environ.get("HYTORCH_CATALOG", "") == "1":
            from trn_bridge import drain_journal
            if _hytorch_seam is None:
                import hashlib as _hl, json as _json
                _man_path = _os.environ["HYTORCH_MANIFEST"]
                _man_sha = _hl.sha256(open(_man_path, "rb").read()).hexdigest()
                _build = _json.load(open(_os.environ["HYTORCH_BUILD_FACTS"]))
                _pol_env = dict(policy_id=int(_os.environ.get("HYTORCH_POLICY_ID", "7")),
                                k=int(_os.environ.get("HYTORCH_K", "8")),
                                s_slots=64,
                                n_features=int(_os.environ.get("HYTORCH_NF", "32768")),
                                mag_max=float(_os.environ.get("HYTORCH_MAG_MAX", "64.0")),
                                selection="two_phase_topk")
                from trn_bridge import _resolve_clip as _rpc
                _pol_env["proposal_clip"] = _rpc() or 0.0
                _hytorch_seam = _HytorchSeam.maybe_create(
                    run_id=_os.environ.get("HYTORCH_RUN_ID", "trn-run"),
                    data_dir=_os.environ["HYTORCH_DATA_DIR"],
                    spool=_os.environ["HYTORCH_SPOOL"],
                    seam_bin=_os.environ["HYTORCH_SEAM_BIN"],
                    manifest_sha=_man_sha, build=_build, policy=_pol_env,
                    control_pg=_hytorch_pg)
            # Facts must be REAL before spooling: cut the lazy graph so the
            # forward that produced the journal has executed (XLA).
            step_boundary(device)
            _entries = drain_journal()
            _hytorch_seam.step_barrier(step, _entries, orig_model.hytorch_codebook)
        optimizer.step(sched)
        # Cut the lazy graph exactly once per optimizer step: fwd+bwd+allreduce
        # +optim compile into ONE cached NEFF executed repeatedly.
        step_boundary(device)
        if _hytorch_seam is not None:
            import sys as _sys
            _sys.path.insert(0, _os.path.join(_os.environ["HYTORCH_ROOT"], "python"))
            from hytorch.model import renorm_or_reset as _rr
            import numpy as _np
            _dead = _rr(orig_model.hytorch_codebook,
                        float(_os.environ.get("HYTORCH_MIN_NORM", "0.00390625")),
                        _np.random.default_rng(1337 + step))
            _cb_lr = next(g["lr"] * float(sched.lrm.item()) if hasattr(sched.lrm, "item") else g["lr"]
                          for g in optimizer.param_groups
                          if any(p is orig_model.hytorch_codebook for p in g["params"]))
            with torch.no_grad():
                _cbg = orig_model.hytorch_codebook.grad
                _gn = float(_cbg.float().norm().item()) if _cbg is not None else 0.0
            _bypass = {"resid_lambda": torch.stack([b.resid_lambda.reshape(()) for b in orig_model.transformer.h]),
                       "x0_lambda": torch.stack([b.x0_lambda.reshape(()) for b in orig_model.transformer.h])}
            _hytorch_seam.step_chain(step, _cb_lr, _gn,
                                     orig_model.hytorch_codebook, resets=_dead,
                                     bypass=_bypass)"""
assert old in s, "optimizer.step anchor drifted"
s = s.replace(old, new)
# 5. Codebook gradient: host-accumulated on XLA (a [N_f, d_slot] grad in the
#    fwd+bwd+all-reduce NEFF is linearized row by row by neuronx-cc: 6M
#    instructions at d10, NCC_EXTP003). Reduced over gloo and uploaded once,
#    AFTER the bucket all-reduce and BEFORE optimizer.step, at both call sites
#    (compile warm-up loop + training loop). No-op for the vanilla arm.
n = s.count("grad_bucket.all_reduce()\n")
assert n == 2, f"grad_bucket.all_reduce anchor drifted ({n})"
s = re.sub(r"^( *)grad_bucket\.all_reduce\(\)\n",
           lambda m: m.group(0) + m.group(1)
           + "__import__(\"trn_bridge\").sync_codebook_grad()  # hytorch: host codebook grad, after the bucket\n",
           s, flags=re.M)
# 6. The codebook is NOT in the flat all-reduce bucket: its grad is None at
#    bucket time (host-accumulated, edit 5) and the zero_() fill of its slice
#    is the one structural difference from the vanilla bucket graph, which
#    neuronx-cc rejects (NCC_IMPR902 MaskPropagation isl_set_union). Without
#    it the catalog bucket graph is byte-for-byte the vanilla one.
old = """        self.params = [p for p in model.parameters() if p.requires_grad]
        self.numel = sum(p.numel() for p in self.params)"""
new = """        self.params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.endswith("hytorch_codebook")]
        self.numel = sum(p.numel() for p in self.params)"""
assert old in s, "GradBucket anchor drifted"
s = s.replace(old, new)
open(p, "w").write(s)
print("train.py patched: catalog seam + codebook group + barrier + chain + host codebook grad + bucket excl. codebook")
EOF

echo "$TRN_COMMIT" > .hytorch-trn-pin
echo "trn patch applied"
