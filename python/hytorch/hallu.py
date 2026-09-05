"""Phase 4a — does hallucination have a journal signature? (preregistered)

Protocol (docs/phases/PHASE4-HALLUCINATIONS.md):
1. Train the cataloged toy on wikitext (or reuse steps), accumulating
   TrainingPriors from the training journal.
2. Cloze generation on held-out val: prompt = real context, model generates
   next token greedily. Label = correct (matches the real next token) vs
   wrong (a measurable confabulation against ground truth).
3. Per generated token, compute F1-F5 from the GENERATION journal.
4. Train LogisticDetector on half, report AUROC on the other half.
   PREREGISTERED THRESHOLD: AUROC >= 0.65 -> signature exists (4b unlocks).
   Below -> published negative, phase dies.
5. Baseline to beat (reported either way): logit entropy.

Run on a GPU droplet via infra/hallu-run.sh. CPU-feasible at small steps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from .applyref import ApplyRef, fp32_to_bf16_bits
from .data import BinDataset
from .model import CatalogedTransformer, CatalogPolicy, renorm_or_reset
from .signatures import (LogisticDetector, TrainingPriors, auroc,
                         logit_entropy_baseline, token_signature)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--so", default=None)
    ap.add_argument("--train-steps", type=int, default=5000)
    ap.add_argument("--cloze-prompts", type=int, default=2000)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..", "..")
    so = args.so or os.path.join(root, "target", "release", "libapply_ref.so")
    man = json.load(open(args.manifest))
    m_model, m_cat, m_tr = man["model"], man["catalog"], man["training"]
    thresholds = man.get("thresholds", {})
    auroc_min = float(thresholds.get("hallu_auroc_min", 0.65))

    torch.manual_seed(m_tr["seeds"][0])
    rng = np.random.default_rng(m_tr["seeds"][0])
    pol = CatalogPolicy(
        s_slots=m_model["s_slots"], d_slot=m_model["d_slot"],
        n_features=m_cat["n_features"], k=m_cat["k"], mag_max=m_cat["mag_max"],
        policy_id=int(m_cat.get("policy_id", 4)),
        selection={"global_topk": 0, "slot_topk": 1}[m_cat.get("selection", "global_topk")],
    )
    vocab = 50257
    ref = ApplyRef.load(so)
    if args.device.startswith("cuda"):
        from .devext import DeviceCatalog
        backend = DeviceCatalog()
    else:
        backend = ref

    model = CatalogedTransformer(
        vocab, m_model["d_model"], n_heads=m_model["n_heads"],
        n_layers=m_model["n_layers"], pol=pol, backend=backend,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=m_tr["lr"])
    train_ds = BinDataset(os.path.join(args.tokens, "train.bin"),
                          m_tr["seq_len"], m_tr["batch_seq"], m_tr["seeds"][0])
    val = np.memmap(os.path.join(args.tokens, "val.bin"), dtype=np.uint16, mode="r")

    priors = TrainingPriors(pol.n_features)
    n_layers = m_model["n_layers"]

    # ---- 1. train + accumulate priors from the training journal ----
    t0 = time.time()
    for step in range(args.train_steps):
        x, y = train_ds.sample()
        x, y = x.to(args.device), y.to(args.device)
        journal: list = []
        logits, aux = model(x, journal)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        renorm_or_reset(model.codebook, man["codebook"]["min_norm"], rng)
        if step % 4 == 0:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            priors.observe_step(x.reshape(-1).cpu().numpy(),
                                [e.recs() for e in journal])
        if step % 500 == 0:
            print(f"[train] step {step} loss {loss.item():.3f}", flush=True)
    print(f"[train] done in {time.time()-t0:.0f}s; priors total={priors.usage.sum():,}",
          flush=True)

    # ---- 2+3. cloze generation with journal capture ----
    model.eval()
    X_rows, labels, prevs, logit_ents = [], [], [], []
    stride = args.prompt_len + 1
    with torch.no_grad():
        for i in range(args.cloze_prompts):
            base = (i * 977) % (len(val) - stride - 2)  # deterministic spread
            ctx = np.asarray(val[base:base + args.prompt_len], dtype=np.int64)
            target = int(val[base + args.prompt_len])
            x = torch.from_numpy(ctx[None, :]).to(args.device)
            journal: list = []
            logits, _ = model(x, journal)
            last = logits[0, -1]
            pred = int(last.argmax().item())
            label_wrong = 1 if pred != target else 0  # 1 = confabulation

            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            # facts for the LAST position only (the generated token's writes)
            pos_last = args.prompt_len - 1
            recs_tok = np.concatenate([
                e.recs()[e.recs()["pos"] == pos_last] for e in journal
            ])
            sig = token_signature(int(ctx[-1]), recs_tok, priors,
                                  pol.k, n_units=n_layers)
            X_rows.append(sig)
            labels.append(label_wrong)
            prevs.append(int(ctx[-1]))
            logit_ents.append(float(logit_entropy_baseline(
                last.float().cpu().numpy()[None, :])[0]))
            if i % 400 == 0:
                print(f"[cloze] {i}/{args.cloze_prompts}", flush=True)

    X = np.stack(X_rows)
    y = np.asarray(labels, dtype=np.float64)
    ents = np.asarray(logit_ents)
    err_rate = float(y.mean())
    print(f"[cloze] error rate {err_rate:.3f} over {len(y)} prompts", flush=True)

    # ---- 4. split, fit, AUROC vs preregistered threshold ----
    idx = np.arange(len(y))
    rng.shuffle(idx)
    half = len(y) // 2
    tr, te = idx[:half], idx[half:]
    det = LogisticDetector().fit(X[tr], y[tr])
    scores = det.score(X[te])
    a_facts = auroc(scores, y[te])
    a_logit = auroc(ents[te], y[te])  # entropy high -> wrong: same direction
    # Per-feature ablation of the detector (which fact matters):
    per_feature = {}
    names = ["F1_familiarity", "F2_context_fit", "F3_energy",
             "F4_contention", "F5_entropy"]
    for j, nm in enumerate(names):
        a = auroc(X[te][:, j], y[te])
        per_feature[nm] = round(float(a if a == a else 0.5), 4)

    verdict = "SIGNATURE_EXISTS" if a_facts >= auroc_min else "NEGATIVE"
    out = {
        "phase": "4a",
        "threshold_auroc_min": auroc_min,
        "auroc_facts": round(float(a_facts), 4),
        "auroc_logit_entropy_baseline": round(float(a_logit), 4),
        "per_feature_auroc": per_feature,
        "detector_weights": {n: round(float(w), 4)
                             for n, w in zip(names, det.w)},
        "cloze_prompts": len(y),
        "cloze_error_rate": round(err_rate, 4),
        "train_steps": args.train_steps,
        "verdict": verdict,
        "facts_beat_logits": bool(a_facts > a_logit),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "phase4a-result.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(args.out_dir, "phase4a-data.npz"),
                        X=X, y=y, logit_entropy=ents,
                        prev_tokens=np.asarray(prevs))
    priors.save(os.path.join(args.out_dir, "training-priors.npz"))
    print(json.dumps(out, indent=2))
    return 0 if verdict == "SIGNATURE_EXISTS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
