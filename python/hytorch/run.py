"""Manifest runner v2 — the citable harness (M7/M8).

Protocol v2 against the seam (two phases per step, spec §06–07):
  Phase A: spool layer frames → barrier → RECEIPT gates opt.step().
  Phase B: after opt.step()+renorm → STEP chain with REAL c_prev/c_next,
           CODEBOOK_RESET features, SEAL at cadence → chainack.
RUN_START + POLICY are committed INSIDE Hyphae by the seam before step 0.

Occupancy contract (Lámina C): zero per-layer syncs on the device path —
verdicts stay on GPU, one async D2H per layer into pinned memory, ONE
synchronize per step (before spooling), audit replay + verify OFFLINE after
the run ("entrenar no es verificar; verificar es otro proceso").

Usage (cataloged run):
  python -m hytorch.run --manifest ../manifests/phase1-k8-nf32768-p2.json \
      --data-dir /run/hyphae --spool /run/spool \
      --seam-bin ../target/release/hytorch-seam \
      --build-facts build-facts.json --device cuda \
      --tokens /data/wikitext103

Usage (baseline arm): same + --baseline (no seam args needed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from .applyref import ApplyRef, VERDICT_COMMIT, VERDICT_OVERFLOW, VERDICT_ABORT, fp32_to_bf16_bits
from .model import CatalogedTransformer, CatalogPolicy, renorm_or_reset
from .seam import StepChain
from .seamclient import SeamClient


def h_canonical(codebook: torch.Tensor) -> bytes:
    """H_canónico(C) = SHA-256 of shape || dtype || raw bf16 row-major (§4.3)."""
    bits = fp32_to_bf16_bits(codebook.detach().to(torch.float32).cpu().numpy())
    h = hashlib.sha256()
    h.update(np.array(bits.shape, dtype="<i8").tobytes())
    h.update(b"bf16")
    h.update(bits.astype("<u2").tobytes())
    return h.digest()


def theta_sha(model: torch.nn.Module) -> bytes:
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().to(torch.float32).cpu().numpy().tobytes())
    return h.digest()


def t1_selected(run_id: str, step: int, mb: int, inv_n: int) -> bool:
    key = f"{run_id}:{step}:{mb}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % inv_n == 0


def flops_per_token(man: dict) -> dict:
    """FLOPs accounting per §11: layer + pack (N_f sweep) + apply."""
    m, c = man["model"], man["catalog"]
    d, L = m["d_model"], m["n_layers"]
    seq = man["training"]["seq_len"]
    block = L * (24 * d * d + 4 * d * seq)
    head_lm = 2 * d * 50257 if man["training"]["tokenizer"].startswith("gpt2") else 2 * d * 256
    pack = L * 2 * c["n_features"] * m["d_slot"]
    apply_f = L * 2 * c["k"] * m["d_slot"]
    return {
        "block": block, "head": head_lm, "pack": pack, "apply": apply_f,
        "catalog_total": block + head_lm + pack + apply_f,
        "baseline_total": block + head_lm + L * d,
    }


@torch.no_grad()
def eval_ppl(model, batches, vocab: int, device: str) -> float:
    model.eval()
    total_nll, total_tok = 0.0, 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        journal: list = []
        logits, _ = model(x, journal)
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1), reduction="sum"
        )
        total_nll += float(nll.item())
        total_tok += y.numel()
    model.train()
    return math.exp(total_nll / max(total_tok, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-dir")
    ap.add_argument("--spool")
    ap.add_argument("--seam-bin")
    ap.add_argument("--verify-bin", default=None)
    ap.add_argument("--so", default=None)
    ap.add_argument("--build-facts", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tokens", default=None, help="dir with train.bin/val.bin")
    ap.add_argument("--baseline", action="store_true", help="dense control arm")
    ap.add_argument("--steps-override", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0, help="0 = only at end")
    ap.add_argument("--eval-batches", type=int, default=24)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--spill-dir", default=None, help="keep audit spills here")
    ap.add_argument("--probe", action="store_true",
                    help="phase 2: token->feature probe from the journal + causal ablation at the end")
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..", "..")
    so = args.so or os.path.join(root, "target", "release", "libapply_ref.so")
    verify = args.verify_bin or os.path.join(root, "target", "release", "hytorch-verify")

    with open(args.manifest, "rb") as f:
        manifest_bytes = f.read()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    man = json.loads(manifest_bytes)
    with open(args.build_facts) as f:
        build = json.load(f)

    arm = "baseline" if args.baseline else "catalog"
    run_id = args.run_id or f"run-{manifest_sha[:8]}-{arm}-{int(time.time())}"
    m_model, m_cat, m_tr, m_bud = man["model"], man["catalog"], man["training"], man["budgets"]

    sel_name = m_cat.get("selection", "global_topk")
    pol = CatalogPolicy(
        s_slots=m_model["s_slots"], d_slot=m_model["d_slot"],
        n_features=m_cat["n_features"], k=m_cat["k"], mag_max=m_cat["mag_max"],
        policy_id=int(m_cat.get("policy_id", 1)),
        selection={"global_topk": 0, "slot_topk": 1}[sel_name],
    )
    steps = args.steps_override or m_tr["steps"]
    seed = m_tr["seeds"][0]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    is_cuda = args.device.startswith("cuda")

    # ---- data ----
    if args.tokens:
        from .data import BinDataset
        vocab = 50257
        train_ds = BinDataset(os.path.join(args.tokens, "train.bin"),
                              m_tr["seq_len"], m_tr["batch_seq"], seed)
        val_ds = BinDataset(os.path.join(args.tokens, "val.bin"),
                            m_tr["seq_len"], m_tr["batch_seq"], seed + 1)
        eval_set = val_ds.eval_batches(args.eval_batches)
        data_manifest = json.load(open(os.path.join(args.tokens, "data-manifest.json")))

        def batch():
            x, y = train_ds.sample()
            return x.to(args.device), y.to(args.device)
    else:
        vocab = 256
        eval_set = []
        data_manifest = {"dataset": "synthetic-shift1"}

        def batch():
            x = torch.randint(0, vocab, (m_tr["batch_seq"], m_tr["seq_len"]),
                              device=args.device)
            return x, torch.roll(x, 1, dims=1)

    # ---- C10 gate ----
    ref = ApplyRef.load(so)
    for k in ("build.apply_ref_hash", "build.harness_commit",
              "build.torch_wheel", "build.backend_wheel"):
        v = str(build.get(k, ""))
        if not v or "FILLED" in v or v == "absent":
            print(f"RUN REFUSED: {k} missing ({v!r}) — C10", file=sys.stderr)
            return 2
    if build["build.apply_ref_hash"] != ref.sha256:
        print(f"RUN REFUSED: build.apply_ref_hash != loaded .so "
              f"({build['build.apply_ref_hash'][:12]} vs {ref.sha256[:12]})",
              file=sys.stderr)
        return 2

    # ---- backend + seam ----
    seam_proc = None
    client = None
    if not args.baseline:
        if not (args.data_dir and args.spool and args.seam_bin):
            print("catalog arm requires --data-dir --spool --seam-bin", file=sys.stderr)
            return 2
        if is_cuda:
            from .devext import DeviceCatalog
            backend = DeviceCatalog()
        else:
            backend = ref
        os.makedirs(args.spool, exist_ok=True)
        seam_proc = subprocess.Popen(
            [args.seam_bin, args.data_dir, run_id, args.spool],
            stderr=subprocess.DEVNULL,
        )
        client = SeamClient(args.spool, policy_id=pol.policy_id,
                            elide_zeros=m_bud["wal_elide_zero_commits"])
        client.run_start(
            manifest_sha, build,
            infra={"device_slug": man["infra"].get("device_slug", "local"),
                   "region": man["infra"].get("region", "local"),
                   "procurement": man["infra"].get("procurement", "local"),
                   "driver": build.get("infra.driver", "unknown")},
            policy={"policy_id": pol.policy_id, "k": pol.k, "s_slots": pol.s_slots,
                    "n_features": pol.n_features, "mag_max": pol.mag_max,
                    "selection": sel_name},
        )
    else:
        backend = None

    model = CatalogedTransformer(
        vocab, m_model["d_model"], n_heads=m_model["n_heads"],
        n_layers=m_model["n_layers"], pol=pol, backend=backend,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=m_tr["lr"])
    n_params = sum(p.numel() for p in model.parameters())

    spill_dir = args.spill_dir or tempfile.mkdtemp(prefix=f"{run_id}-spills-")
    os.makedirs(spill_dir, exist_ok=True)

    probe = None
    if args.probe and not args.baseline:
        from .probe import TokenFeatureProbe
        probe = TokenFeatureProbe(vocab, pol.n_features)

    run_log = {
        "run_id": run_id, "arm": arm, "manifest_sha256": manifest_sha,
        "build": build, "data": data_manifest, "n_params": n_params,
        "flops_per_token": flops_per_token(man),
        "steps": [], "t1": [], "evals": [], "codebook_resets": [],
        "timing": {"compute_s": 0.0, "ledger_s": 0.0, "audit_capture_s": 0.0},
    }
    inv_n = man["t1"]["sample_rate_inv_n"]
    budget_ms = m_bud["barrier_budget_ms"]
    seal_every = m_bud["seal_every_k_steps"]
    alias_mode = man["harness"]["alias_check"]

    c_next_prev_step = h_canonical(model.codebook)
    pending_audits: list[dict] = []  # verified offline after the run
    t_start = time.time()
    ret = 1
    try:
        for step in range(steps):
            t0 = time.time()
            x, y = batch()
            audited = (not args.baseline) and t1_selected(run_id, step, 0, inv_n)
            alias = alias_mode == "always" or (alias_mode == "t1_rate" and audited)

            c_prev = c_next_prev_step
            cb_step_bits = None
            capture = None
            if audited:
                cb_step_bits = fp32_to_bf16_bits(
                    model.codebook.detach().to(torch.float32).cpu().numpy()
                )
                capture = {}

            journal: list = []
            logits, aux = model(x, journal, alias_check=alias, capture=capture)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab), y.reshape(-1)
            )
            total = loss + m_tr["aux_balance_coeff"] * aux if not args.baseline else loss

            opt.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())

            step_rec = {"step": step, "loss": float(loss.item()),
                        "aux": float(aux.item()) if not args.baseline else 0.0,
                        "audited": audited}
            run_log["timing"]["compute_s"] += time.time() - t0

            if not args.baseline:
                # ---- ONE sync per step: pinned journal copies are done ----
                t1_ = time.time()
                if is_cuda:
                    torch.cuda.synchronize()
                recs_per_layer = [e.recs() for e in journal]
                if probe is not None:
                    probe.observe(step, x.reshape(-1).cpu().numpy(), recs_per_layer)

                # ---- Phase A: barrier gates opt.step ----
                chain = StepChain(step_id=step, mb=0, policy_id=pol.policy_id,
                                  elide_zeros=m_bud["wal_elide_zero_commits"])
                for layer_idx, recs in enumerate(recs_per_layer):
                    client.spool_layer(step, layer_idx, 0, recs, device=0)
                    chain.add_layer(layer_idx, recs, device=0)
                head = client.barrier(step, budget_ms=budget_ms)
                if head != chain.head.hex():
                    print(f"RECEIPT MISMATCH step {step}", file=sys.stderr)
                    return 1
                step_rec["head"] = head
                run_log["timing"]["ledger_s"] += time.time() - t1_

                t2_ = time.time()
                opt.step()
                dead = renorm_or_reset(model.codebook, man["codebook"]["min_norm"], rng)
                run_log["timing"]["compute_s"] += time.time() - t2_
                if dead:
                    run_log["codebook_resets"].append({"step": step, "features": dead})

                t3_ = time.time()
                c_next = h_canonical(model.codebook)
                seal = theta_sha(model) if (step % seal_every == 0 and step > 0) else None
                client.step_chain(step, m_tr["lr"], grad_norm, c_prev, c_next,
                                  dead, seal, budget_ms=budget_ms)
                c_next_prev_step = c_next
                if seal:
                    run_log.setdefault("seals", []).append(
                        {"step": step, "theta_sha256": seal.hex()})
                run_log["timing"]["ledger_s"] += time.time() - t3_

                v = np.concatenate(recs_per_layer)
                step_rec.update({
                    "commit": int((v["verdict"] == VERDICT_COMMIT).sum()),
                    "overflow": int((v["verdict"] == VERDICT_OVERFLOW).sum()),
                    "abort": int((v["verdict"] == VERDICT_ABORT).sum()),
                })

                # ---- T1 capture (verify OFFLINE after the run) ----
                # h_final is the DEVICE'S OWN CLAIM of the residual after its
                # writes (captured from the forward), not a re-derivation: if
                # the device apply diverged from apply_ref, T1 catches it as
                # residual_mismatch.
                if audited:
                    t4_ = time.time()
                    B, T = x.shape
                    h0_bits = fp32_to_bf16_bits(
                        capture["h0"].to(torch.float32).cpu().numpy()
                    ).reshape(B * T, pol.s_slots, pol.d_slot)
                    hf_bits = fp32_to_bf16_bits(
                        capture["h_final"].to(torch.float32).cpu().numpy()
                    ).reshape(-1)
                    data = chain.spill_bytes(
                        h0_bits.reshape(-1), cb_step_bits,
                        pol.s_slots, pol.d_slot, hf_bits,
                    )
                    spath = os.path.join(spill_dir, f"step-{step:08d}.spill")
                    with open(spath, "wb") as f:
                        f.write(data)
                    pending_audits.append({"step": step, "path": spath})
                    run_log["timing"]["audit_capture_s"] += time.time() - t4_
            else:
                opt.step()
                run_log["timing"]["compute_s"] += 0.0

            run_log["steps"].append(step_rec)

            if args.eval_every and step > 0 and step % args.eval_every == 0 and eval_set:
                ppl = eval_ppl(model, eval_set, vocab, args.device)
                run_log["evals"].append({"step": step, "val_ppl": ppl})
                print(f"[eval] step {step} val_ppl {ppl:.2f}", flush=True)

        # ---- OFFLINE verification of captured audits (CPU, apply_ref) ----
        t5_ = time.time()
        for audit in pending_audits:
            out = subprocess.run([verify, audit["path"], "--json"],
                                 capture_output=True, text=True)
            verdict = json.loads(out.stdout) if out.stdout.strip() else {}
            run_log["t1"].append({"step": audit["step"], **verdict})
            if out.returncode != 0:
                print(f"T1 RED at step {audit['step']}: {out.stdout.strip()}",
                      file=sys.stderr)
                return 1
        run_log["timing"]["verify_offline_s"] = time.time() - t5_

        # ---- phase-2 probe report + causal ablation (eval-time POLICY) ----
        if probe is not None and eval_set:
            from .probe import ablation_eval
            assoc = probe.report()
            run_log["probe"] = {"associations": assoc}
            if assoc:
                top = assoc[0]
                run_log["probe"]["ablation"] = ablation_eval(
                    model, eval_set, vocab, args.device,
                    deny_features=[top["feature"]],
                    target_token=top["token_id"],
                )
                print("[probe] top:", top, flush=True)
                print("[probe] ablation:", run_log["probe"]["ablation"], flush=True)

        # ---- final eval + EXPORT ----
        final_ppl = eval_ppl(model, eval_set, vocab, args.device) if eval_set else None
        if final_ppl is not None:
            run_log["evals"].append({"step": steps, "val_ppl": final_ppl})
        run_log["wall_s"] = time.time() - t_start

        theta_final = theta_sha(model).hex()
        c_final = h_canonical(model.codebook).hex()
        final_head = run_log["steps"][-1].get("head", "-")
        run_log["export"] = {"theta_sha256": theta_final, "c_sha256": c_final,
                             "final_head": final_head, "val_ppl": final_ppl}
        if not args.baseline:
            client.export(steps - 1, theta_final, c_final, final_head)
        ret = 0
    finally:
        if seam_proc:
            seam_proc.terminate()
            try:
                seam_proc.wait(timeout=10)
            except Exception:
                seam_proc.kill()

    out_dir = args.out_dir or (os.path.dirname(args.spool) if args.spool else ".")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.runlog.json")
    with open(out_path, "w") as f:
        json.dump(run_log, f, indent=2)

    t1_green = sum(1 for t in run_log["t1"] if t.get("outcome") == "consistent")
    print(json.dumps({
        "run_id": run_id, "arm": arm, "steps": steps,
        "loss_first": run_log["steps"][0]["loss"],
        "loss_last": run_log["steps"][-1]["loss"],
        "val_ppl": run_log["export"]["val_ppl"] if ret == 0 else None,
        "t1_audits": len(run_log["t1"]), "t1_green": t1_green,
        "wall_s": round(run_log.get("wall_s", 0), 1),
        "timing": {k: round(v, 2) for k, v in run_log["timing"].items()},
        "runlog": out_path,
    }, indent=2))
    if ret == 0 and t1_green != len(run_log["t1"]):
        ret = 1
    return ret


if __name__ == "__main__":
    raise SystemExit(main())
