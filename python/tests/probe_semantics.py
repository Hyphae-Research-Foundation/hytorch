"""Semantic probe v0 — the first question of phase 2, asked ONLY of the journal.

The user's question, verbatim: "lo que pasa dentro de un modelo cuando piensa
son una especie de +1743 -290834 que puede significar 'hola' — ¿eso lo
estamos entendiendo ahora?"

Phase 1 changed the CATEGORY of that question, not its answer. Before: the
+1743/-290834 were anonymous float soup — "which write meant X?" was not even
well-formed. Now every write is a fact (pos, slot, feature, mag, verdict), so
the question becomes a QUERY:

    "When the model reads token T, does some feature f fire for it
     disproportionately — consistently, across steps, layers, positions?"

This probe trains the cataloged toy on a synthetic task, records ONLY the
journal (never activations), and measures token→feature mutual association:

    lift(f, T) = P(feature=f | token=T) / P(feature=f)

plus a purity/coverage table for the top features. High lift + high purity =
that feature is a NAME the model consistently uses when processing that
token. Low = superposition/polysemanticity persists (the honest expected
default — this probe MEASURES it instead of guessing).

What this is NOT (lista negra): a claim that features "mean" concepts, an
SAE, or interpretability of thoughts. It is the demonstration that the
question is now formulable against facts. Run: CPU, ~2 min.

Usage: .venv/bin/python python/tests/probe_semantics.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hytorch.applyref import ApplyRef, VERDICT_COMMIT  # noqa: E402
from hytorch.model import CatalogedTransformer, CatalogPolicy, renorm_or_reset  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SO = os.path.join(ROOT, "target", "release", "libapply_ref.so")

# Tiny world: 8 "words". The model learns next-token on sequences drawn from
# a fixed bigram chain, so tokens have stable, learnable identities.
VOCAB = 8
NAMES = ["hola", "mundo", "sol", "luna", "pan", "agua", "rio", "mar"]
S, D, NF, K = 16, 8, 256, 4
SEQ, BATCH, STEPS, WARMUP = 32, 8, 400, 200


def main() -> int:
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    ref = ApplyRef.load(SO)
    pol = CatalogPolicy(s_slots=S, d_slot=D, n_features=NF, k=K, mag_max=64.0,
                        policy_id=4, selection=0)
    model = CatalogedTransformer(VOCAB, S * D, n_heads=4, n_layers=2,
                                 pol=pol, backend=ref)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # Deterministic bigram world: token t is followed by (t*3+1) % VOCAB
    # with p=0.8, uniform otherwise. Identities are stable -> learnable.
    def batch():
        x = np.zeros((BATCH, SEQ), dtype=np.int64)
        x[:, 0] = rng.integers(0, VOCAB, size=BATCH)
        for j in range(1, SEQ):
            nxt = (x[:, j - 1] * 3 + 1) % VOCAB
            rnd = rng.integers(0, VOCAB, size=BATCH)
            take = rng.random(BATCH) < 0.8
            x[:, j] = np.where(take, nxt, rnd)
        t = torch.from_numpy(x)
        return t, torch.roll(t, -1, dims=1)

    # token→feature commit counts, read ONLY from the journal (post-warmup).
    counts = defaultdict(lambda: np.zeros(NF, dtype=np.int64))
    total_by_feature = np.zeros(NF, dtype=np.int64)
    total_commits = 0

    losses = []
    for step in range(STEPS):
        x, y = batch()
        journal: list = []
        logits, aux = model(x, journal)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        renorm_or_reset(model.codebook, 2 ** -8, rng)
        losses.append(float(loss.item()))

        if step < WARMUP:
            continue
        flat_tokens = x.reshape(-1).numpy()  # pos -> token id
        for entry in journal:
            recs = entry.recs()
            commits = recs[recs["verdict"] == VERDICT_COMMIT]
            nz = commits[(commits["mag_bf16"].astype(int) & 0x7FFF) != 0]
            toks = flat_tokens[nz["pos"].astype(int)]
            feats = nz["feature"].astype(int)
            for t, f in zip(toks, feats):
                counts[int(t)][f] += 1
                total_by_feature[f] += 1
                total_commits += 1

    print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}  "
          f"(commits analizados post-warmup: {total_commits:,})\n")

    # lift(f,T) = P(f|T)/P(f); purity(f) = max_T count(f,T)/count(f).
    p_f = total_by_feature / max(total_commits, 1)
    report = []
    for t in range(VOCAB):
        c = counts[t]
        n_t = c.sum()
        if n_t == 0:
            continue
        p_f_given_t = c / n_t
        with np.errstate(divide="ignore", invalid="ignore"):
            lift = np.where(p_f > 0, p_f_given_t / p_f, 0.0)
        # Only features with real support for this token.
        mask = c >= 50
        if not mask.any():
            continue
        best = int(np.argmax(np.where(mask, lift, 0.0)))
        purity = counts[t][best] / max(total_by_feature[best], 1)
        report.append({
            "token": NAMES[t], "feature": best,
            "lift": round(float(lift[best]), 2),
            "purity_pct": round(100 * float(purity), 1),
            "n_commits_pair": int(c[best]),
        })

    # Global stats: how concentrated is each token's feature distribution?
    ents = []
    for t in range(VOCAB):
        c = counts[t].astype(float)
        if c.sum() == 0:
            continue
        p = c / c.sum()
        p = p[p > 0]
        ents.append(-(p * np.log2(p)).sum())
    uniform_bits = np.log2((total_by_feature > 0).sum())

    print("token   feature  lift   purity%  n_pair")
    for r in sorted(report, key=lambda r: -r["lift"]):
        print(f"{r['token']:<7} f{r['feature']:<7} {r['lift']:<6} "
              f"{r['purity_pct']:<8} {r['n_commits_pair']}")
    print(f"\nentropía media token→features: {np.mean(ents):.2f} bits "
          f"(uniforme sería {uniform_bits:.2f} bits)")
    print("\nLectura honesta:")
    print(" - lift >> 1 con purity alta  = ese feature ES un nombre que el modelo")
    print("   usa consistentemente para ese token (asociación, no 'significado').")
    print(" - lift ~1 / purity baja      = superposición: el feature es compartido.")
    print(" - Esto se calculó SIN mirar una sola activación: solo el journal.")

    out = {
        "task": "bigram-8tok", "steps": STEPS, "warmup": WARMUP,
        "commits_analyzed": int(total_commits),
        "associations": report,
        "mean_token_entropy_bits": round(float(np.mean(ents)), 2),
        "uniform_bits": round(float(uniform_bits), 2),
        "read_from": "journal only (facts), zero activations",
    }
    os.makedirs(os.path.join(ROOT, "results", "probe"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "probe", "semantics-v0.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
