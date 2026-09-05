"""Phase 4 — hallucination signatures from journal facts (F1-F5).

The detector's inputs are FACTS ONLY: commits/overflows/aborts per generated
token, plus training-time priors (feature usage counts and token→feature
co-occurrence) that themselves come from the training journal (the phase-2
probe tables). No logits, no activations: the detector must be as auditable
as the ledger it reads.

F1 familiarity: mean log(1+train_usage[f]) over committed features.
F2 context_fit: mean log(1+cooc[prev_token, f]) over committed features
   (how often did f fire for this preceding token during training?).
F3 energy: sum |mag| over commits.
F4 contention: #OVERFLOW / (k * n_units).
F5 entropy: Shannon entropy (bits) of the committed feature multiset.

Detector v0: 5-parameter logistic regression, closed-form-ish training via
sklearn-free gradient descent (numpy only — droplets don't need sklearn).
"""

from __future__ import annotations

import numpy as np

from .applyref import VERDICT_COMMIT, VERDICT_OVERFLOW


class TrainingPriors:
    """Feature-usage and token→feature co-occurrence accumulated from the
    TRAINING journal (same accumulation as the phase-2 probe, kept sparse)."""

    def __init__(self, n_features: int):
        self.nf = n_features
        self.usage = np.zeros(n_features, dtype=np.int64)
        self.cooc: dict[int, np.ndarray] = {}   # prev_token -> feature counts

    def observe_step(self, tokens_flat: np.ndarray, recs_per_layer: list):
        for recs in recs_per_layer:
            commits = recs[recs["verdict"] == VERDICT_COMMIT]
            nz = commits[(commits["mag_bf16"].astype(int) & 0x7FFF) != 0]
            pos = nz["pos"].astype(int)
            feats = nz["feature"].astype(int)
            self.usage[feats] += 1
            # prev-token context: pos 0 has no prev; skip it.
            has_prev = pos > 0
            prevs = tokens_flat[pos[has_prev] - 1]
            for pt, f in zip(prevs.tolist(), feats[has_prev].tolist()):
                arr = self.cooc.get(pt)
                if arr is None:
                    arr = np.zeros(self.nf, dtype=np.int32)
                    self.cooc[pt] = arr
                arr[f] += 1

    def save(self, path: str):
        np.savez_compressed(
            path, usage=self.usage,
            cooc_keys=np.array(list(self.cooc.keys()), dtype=np.int64),
            cooc_vals=np.stack(list(self.cooc.values())) if self.cooc else
            np.zeros((0, self.nf), dtype=np.int32),
        )

    @classmethod
    def load(cls, path: str) -> "TrainingPriors":
        z = np.load(path)
        p = cls(len(z["usage"]))
        p.usage = z["usage"]
        for k, v in zip(z["cooc_keys"].tolist(), z["cooc_vals"]):
            p.cooc[k] = v
        return p


def bf16_abs(bits: np.ndarray) -> np.ndarray:
    return np.abs((bits.astype(np.uint32) << 16).view(np.float32))


def token_signature(prev_token: int, recs_for_token: np.ndarray,
                    priors: TrainingPriors, k: int, n_units: int) -> np.ndarray:
    """F1-F5 for ONE generated token from its journal rows (all layers/units).
    recs_for_token: structured array already filtered to this token's pos."""
    v = recs_for_token["verdict"]
    commits = recs_for_token[v == VERDICT_COMMIT]
    nz = commits[(commits["mag_bf16"].astype(int) & 0x7FFF) != 0]
    feats = nz["feature"].astype(int)
    n_over = int((v == VERDICT_OVERFLOW).sum())

    if len(feats) == 0:
        return np.array([0.0, 0.0, 0.0, n_over / max(k * n_units, 1), 0.0])

    f1 = float(np.mean(np.log1p(priors.usage[feats])))
    co = priors.cooc.get(int(prev_token))
    f2 = float(np.mean(np.log1p(co[feats]))) if co is not None else 0.0
    f3 = float(bf16_abs(nz["mag_bf16"]).sum())
    f4 = n_over / max(k * n_units, 1)
    _, counts = np.unique(feats, return_counts=True)
    p = counts / counts.sum()
    f5 = float(-(p * np.log2(p)).sum())
    return np.array([f1, f2, f3, f4, f5])


class LogisticDetector:
    """5-param logistic regression, numpy-only. Standardizes features."""

    def __init__(self):
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sd = None

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1,
            steps: int = 2000, seed: int = 0) -> "LogisticDetector":
        rng = np.random.default_rng(seed)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        self.w = rng.normal(0, 0.01, d)
        self.b = 0.0
        for _ in range(steps):
            z = Xs @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-z))
            g = p - y
            self.w -= lr * (Xs.T @ g) / n
            self.b -= lr * float(g.mean())
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b)))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (ties handled by average rank)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def logit_entropy_baseline(logits: "np.ndarray") -> np.ndarray:
    """Reference baseline the facts must beat (reported either way):
    per-token predictive entropy from the softmax."""
    x = logits - logits.max(-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(-1, keepdims=True)
    return -(p * np.log2(np.clip(p, 1e-12, 1))).sum(-1)


# ---------------------------------------------------------------------------
# Signatures v2 (PHASE4-MECHANISM.md): what the hallucination theory demands.
#
# Anthropic's circuit story (Biology of an LLM §Entity Recognition): the
# misfire is a weakly-activated "known entity" suppressing the refusal
# default while a SEPARATE guessing circuit produces the answer. Translated
# to write-space, three testable signatures:
#   F6 cross_layer_coherence — does the guessing circuit fail to converge?
#   F7 familiarity_context_divergence — "know the name, not the answer"
#   F8 specificity — head-feature (generic) vs tail-feature (content) mass
#   F9 unit_skew — attn-writes vs mlp-writes energy (ours alone to measure)
# ---------------------------------------------------------------------------

def token_signature_v2(prev_token: int, recs_for_token: np.ndarray,
                       priors: TrainingPriors, k: int, n_layers: int,
                       n_units: int = 2,
                       head_cutoff_pct: float = 1.0) -> np.ndarray:
    """F1-F9 for ONE generated token.

    recs_for_token must carry 'layer' (frame id = 2*L + unit when the
    nanochat bridge is in play; unit = layer%2) in addition to the usual
    fields. Returns [F1..F5, F6, F7, F8, F9].
    """
    base = token_signature(prev_token, recs_for_token, priors, k,
                           n_units=n_layers * n_units)
    v = recs_for_token["verdict"]
    commits = recs_for_token[v == VERDICT_COMMIT]
    nz = commits[(commits["mag_bf16"].astype(int) & 0x7FFF) != 0]
    if len(nz) == 0:
        return np.concatenate([base, [0.0, 0.0, 0.0, 0.0]])

    feats = nz["feature"].astype(int)
    mags = bf16_abs(nz["mag_bf16"])

    # F6: mean Jaccard of committed feature sets between consecutive layers.
    f6 = 0.0
    if "layer" in nz.dtype.names:
        L = nz["layer"].astype(int) // n_units   # frame id -> layer
        per_layer = [set(feats[L == l].tolist()) for l in range(n_layers)]
        per_layer = [s for s in per_layer if s]
        if len(per_layer) >= 2:
            js = []
            for a, b in zip(per_layer[:-1], per_layer[1:]):
                inter = len(a & b)
                union = len(a | b)
                js.append(inter / union if union else 0.0)
            f6 = float(np.mean(js))

    # F7: familiarity minus context-fit, each z-scored against the priors'
    # own scale so the difference is comparable across runs.
    usage_log = np.log1p(priors.usage[priors.usage > 0])
    if len(usage_log):
        mu_u = float(usage_log.mean())
        # degenerate priors (uniform usage) have no scale: clamp so F7 stays
        # bounded instead of exploding by 1/eps
        sd_u = max(float(usage_log.std()), 0.1)
    else:
        mu_u, sd_u = 0.0, 1.0
    f1_raw = float(np.mean(np.log1p(priors.usage[feats])))
    co = priors.cooc.get(int(prev_token))
    f2_raw = float(np.mean(np.log1p(co[feats]))) if co is not None else 0.0
    f7 = (f1_raw - mu_u) / sd_u - (f2_raw - mu_u) / sd_u

    # F8: |mag| mass on TAIL features (outside the global top head_cutoff_pct%
    # by training usage). Generic/head features carry format; tail features
    # carry content. Confabulation ≈ head-dominated writing (H1).
    n_head = max(1, int(len(priors.usage) * head_cutoff_pct / 100.0))
    head_ids = np.argpartition(priors.usage, -n_head)[-n_head:]
    head_mask = np.isin(feats, head_ids)
    total_mass = float(mags.sum())
    f8 = float(mags[~head_mask].sum() / total_mass) if total_mass > 0 else 0.0

    # F9: (attn energy − mlp energy) / total. Only meaningful with the
    # nanochat 2-unit bridge (frame id = 2L+unit; unit 0=attn, 1=mlp).
    f9 = 0.0
    if "layer" in nz.dtype.names and n_units == 2:
        unit = nz["layer"].astype(int) % 2
        e_attn = float(mags[unit == 0].sum())
        e_mlp = float(mags[unit == 1].sum())
        tot = e_attn + e_mlp
        f9 = (e_attn - e_mlp) / tot if tot > 0 else 0.0

    return np.concatenate([base, [f6, f7, f8, f9]])


SIGNATURE_V2_NAMES = [
    "F1_familiarity", "F2_context_fit", "F3_energy", "F4_contention",
    "F5_entropy", "F6_xlayer_coherence", "F7_fam_ctx_divergence",
    "F8_specificity", "F9_unit_skew",
]
