"""Semantic probe + causal ablation over the REAL run's journal (phase 2).

probe: accumulates token→feature commit counts from the live journal during
training (zero activations read; facts only) and reports lift/purity.

ablation (the causal step Tamkin et al. do with activations; we do it as a
JOURNALIZED POLICY intervention): after training, pick the top-associated
feature f* for a probe token T, then run eval twice — normal, and with f*
FORCED TO ABORT (reason=policy) via an eval-time deny-list. The deny-list is
exactly the spec's ABORT(reason=policy) verdict: the non-fact is data. We
measure the per-token NLL delta for T vs all other tokens. If ablating f*
hurts T specifically, the association is causal, not just correlational.
"""

from __future__ import annotations

import numpy as np
import torch

from .applyref import VERDICT_COMMIT


class TokenFeatureProbe:
    """Streaming token→feature association from journal facts."""

    def __init__(self, vocab: int, n_features: int, sample_every: int = 8):
        self.vocab = vocab
        self.nf = n_features
        self.sample_every = sample_every
        # Sparse dict-of-arrays: token -> feature counts (vocab can be 50k).
        self.counts: dict[int, np.ndarray] = {}
        self.total_by_feature = np.zeros(n_features, dtype=np.int64)
        self.total = 0

    def observe(self, step: int, tokens_flat: np.ndarray, recs_per_layer: list):
        if step % self.sample_every != 0:
            return
        for recs in recs_per_layer:
            commits = recs[recs["verdict"] == VERDICT_COMMIT]
            nz = commits[(commits["mag_bf16"].astype(int) & 0x7FFF) != 0]
            toks = tokens_flat[nz["pos"].astype(int)]
            feats = nz["feature"].astype(int)
            for t, f in zip(toks.tolist(), feats.tolist()):
                arr = self.counts.get(t)
                if arr is None:
                    arr = np.zeros(self.nf, dtype=np.int64)
                    self.counts[t] = arr
                arr[f] += 1
                self.total_by_feature[f] += 1
                self.total += 1

    def report(self, top_tokens: int = 30, min_pair: int = 200) -> list[dict]:
        p_f = self.total_by_feature / max(self.total, 1)
        rows = []
        # Rank tokens by total journal mass (most-processed tokens first).
        by_mass = sorted(self.counts.items(), key=lambda kv: -int(kv[1].sum()))
        for t, c in by_mass[:top_tokens]:
            n_t = int(c.sum())
            if n_t < min_pair:
                continue
            p_f_given_t = c / n_t
            with np.errstate(divide="ignore", invalid="ignore"):
                lift = np.where(p_f > 0, p_f_given_t / p_f, 0.0)
            mask = c >= min_pair
            if not mask.any():
                continue
            best = int(np.argmax(np.where(mask, lift, 0.0)))
            rows.append({
                "token_id": int(t),
                "feature": best,
                "lift": round(float(lift[best]), 2),
                "purity_pct": round(100 * float(c[best]) / max(int(self.total_by_feature[best]), 1), 1),
                "n_pair": int(c[best]),
                "n_token": n_t,
            })
        rows.sort(key=lambda r: -r["lift"])
        return rows


@torch.no_grad()
def ablation_eval(model, batches, vocab: int, device: str,
                  deny_features: list[int], target_token: int) -> dict:
    """Eval twice: normal vs feature-denied. The deny-list rides on the
    CatalogPolicy as an eval-time POLICY intervention (ABORT reason=policy
    on the denied features — journalized like any other non-fact).

    Returns per-token NLL for the target vs the rest, both conditions.
    """
    def run(deny):
        model.pol.deny_features = deny  # consumed by HyphaeWrite forward
        tgt_nll, tgt_n, rest_nll, rest_n = 0.0, 0, 0.0, 0
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            journal: list = []
            logits, _ = model(x, journal)
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab), y.reshape(-1), reduction="none")
            is_tgt = (y.reshape(-1) == target_token)
            tgt_nll += float(nll[is_tgt].sum()); tgt_n += int(is_tgt.sum())
            rest_nll += float(nll[~is_tgt].sum()); rest_n += int((~is_tgt).sum())
        model.pol.deny_features = []
        return (tgt_nll / max(tgt_n, 1), rest_nll / max(rest_n, 1), tgt_n)

    base_tgt, base_rest, n_tgt = run([])
    abl_tgt, abl_rest, _ = run(deny_features)
    return {
        "target_token": target_token,
        "denied_features": deny_features,
        "n_target_positions": n_tgt,
        "nll_target_base": round(base_tgt, 4),
        "nll_target_ablated": round(abl_tgt, 4),
        "nll_target_delta": round(abl_tgt - base_tgt, 4),
        "nll_rest_base": round(base_rest, 4),
        "nll_rest_ablated": round(abl_rest, 4),
        "nll_rest_delta": round(abl_rest - base_rest, 4),
        "specific": bool((abl_tgt - base_tgt) > 3 * abs(abl_rest - base_rest)),
    }
