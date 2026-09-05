# Instrumentation for the SFT NaN hunt (applied on top of apply-patch.sh).
# After the FIRST optimizer.step(), on rank 0: report per-group grad norms,
# non-finite params, and codebook/state sanity. Exits with code 42 after
# reporting so we get one clean diagnostic per launch.
import re
p = "scripts/chat_sft.py"
s = open(p).read()
old = """    if _hytorch_seam is not None:
        import sys as _sys
        _sys.path.insert(0, _os.path.join(_os.environ["HYTORCH_ROOT"], "python"))
        from hytorch.model import renorm_or_reset as _rr"""
new = """    if _os.environ.get("HYTORCH_SFT_PROBE", "") == "1" and step <= 2:
        import math as _m
        _bad = [n for n, p_ in orig_model.named_parameters() if not torch.isfinite(p_).all()]
        _badg = [n for n, p_ in orig_model.named_parameters() if p_.grad is not None and not torch.isfinite(p_.grad).all()]
        _gn = {}
        for _gi, _g in enumerate(optimizer.param_groups):
            _norms = [p_.grad.float().norm().item() for p_ in _g["params"] if p_.grad is not None]
            _gn[f"g{_gi}_{_g['kind']}"] = (round(sum(x*x for x in _norms) ** 0.5, 4), round(_g["lr"], 6))
        _cb = orig_model.hytorch_codebook
        _st = optimizer.state.get(_cb, {})
        _stinfo = {k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in _st.items()}
        _stfin = all(torch.isfinite(v).all().item() for k, v in _st.items() if hasattr(v, "shape"))
        print(f"[probe r{ddp_rank}] step {step} POST-STEP loss={float(train_loss.item()) if hasattr(train_loss,'item') else train_loss} "
              f"nonfinite_params={_bad[:6]} nonfinite_grads={_badg[:6]}", flush=True)
        print(f"[probe r{ddp_rank}] grad norms/lr per group: {_gn}", flush=True)
        print(f"[probe r{ddp_rank}] codebook finite={torch.isfinite(_cb).all().item()} norm_p50={_cb.norm(dim=1).median().item():.4f} "
              f"state={_stinfo} state_finite={_stfin} resid_lambdas={orig_model.resid_lambdas.tolist()[:4]} x0={orig_model.x0_lambdas.tolist()[:4]}", flush=True)
        # which param does the codebook's optimizer state ACTUALLY belong to? (index mapping check)
        _idx = {id(p_): i for i, p_ in enumerate(pp for g_ in optimizer.param_groups for pp in g_["params"])}
        print(f"[probe r{ddp_rank}] codebook param index in optimizer = {_idx.get(id(_cb))} (base_train saved it as 147)", flush=True)
        if step == 2:
            print(f"[probe r{ddp_rank}] probe complete — exiting 42", flush=True)
            _os._exit(42)
    if _hytorch_seam is not None:
        import sys as _sys
        _sys.path.insert(0, _os.path.join(_os.environ["HYTORCH_ROOT"], "python"))
        from hytorch.model import renorm_or_reset as _rr"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
# also probe BEFORE step 0's optimizer.step: are grads finite going in?
old2 = """        _entries = drain_journal()
        _hytorch_seam.step_barrier(step, _entries, orig_model.hytorch_codebook)
        with torch.no_grad():
            _cbg = orig_model.hytorch_codebook.grad"""
new2 = """        _entries = drain_journal()
        if _os.environ.get("HYTORCH_SFT_PROBE", "") == "1" and step <= 2:
            _badg0 = [n for n, p_ in orig_model.named_parameters() if p_.grad is not None and not torch.isfinite(p_.grad).all()]
            print(f"[probe r{ddp_rank}] step {step} PRE-STEP loss={float(train_loss.item())} nonfinite_grads={_badg0[:6]} "
                  f"lrm={lrm}", flush=True)
        _hytorch_seam.step_barrier(step, _entries, orig_model.hytorch_codebook)
        with torch.no_grad():
            _cbg = orig_model.hytorch_codebook.grad"""
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, new2)
open(p, "w").write(s)
print("sft probe patched")
