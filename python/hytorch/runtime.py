"""Journaled inference runtime — Hyphae inside the model at generation time.

User directive (2026-08-31): "si tan solo pudiéramos introducir Hyphae
dentro del modelo y cuando alucine o cualquier cosa mala quedara documentada
(post-training), eso sería lo mejor."

This module makes generation leave facts, exactly like training does:

  Per generated token t:
    - the forward's catalog writes (COMMIT/OVERFLOW/ABORT per layer×unit)
      are captured from the journal — the model's internal acts for THIS token;
    - the signature F1-F5 (signatures.py) is computed live from those facts
      against the TRAINING priors (familiarity/context-fit of the features
      it cited);
    - a GENERATION fact is committed to Hyphae:
        run/<chat-id>/turn/<n>/token/<t> →
          {token_id, text, score, F1..F5, n_commit, n_overflow, n_abort,
           head-of-token-facts}
    - if score < τ (from phase-4a held-out calibration), a LOW_CONFIDENCE
      fact is committed BEFORE the token is emitted — the "something bad"
      is documented as a first-class non-fact, in the same ledger, at the
      moment it happens, not post-hoc.

  Per turn: a TURN receipt seals the token-fact chain (same T2 discipline).

The auditor story: "the model said X" → query the ledger → every internal
write behind X, its confidence signature, and whether the runtime flagged
it — receipts, not logs.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch

from .applyref import VERDICT_COMMIT, VERDICT_OVERFLOW, VERDICT_ABORT
from .signatures import TrainingPriors, token_signature, token_signature_v2


def _facts_head(prev: bytes, payload: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(prev)
    h.update(payload)
    return h.digest()


class GenerationLedger:
    """Writes generation facts through the seam spool (GENFACT protocol).

    Reuses the live hytorch-seam process: we spool small JSON fact files;
    the seam commits them into Hyphae under run/<run_id>/gen/... keys.
    Fail-stop discipline is softer than training (a chat should not die if
    the ledger hiccups for one token) BUT: an unacknowledged fact makes the
    token UNCITABLE and the runtime says so — the degradation is visible,
    never silent.
    """

    def __init__(self, spool_dir: str, budget_ms: int = 2000):
        self.spool = spool_dir
        self.budget_ms = budget_ms
        self.seq = 0
        os.makedirs(spool_dir, exist_ok=True)

    def commit_fact(self, key_suffix: str, payload: dict) -> bool:
        """Spool one generation fact; wait for ack. Returns citability."""
        self.seq += 1
        name = f"genfact-{self.seq:08d}"
        path = os.path.join(self.spool, name + ".json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"key": key_suffix, "value": payload}, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp, path)
        ack = path.replace(".json", ".ack")
        deadline = time.monotonic() + self.budget_ms / 1000.0
        while time.monotonic() < deadline:
            if os.path.exists(ack):
                os.remove(ack)
                return True
            time.sleep(0.001)
        return False  # visible degradation: token stays, marked uncitable


class JournaledGenerator:
    """Greedy/temperature generation with per-token facts + live signature.

    model: CatalogedTransformer-style (forward(tokens, journal) -> logits[,aux])
           or nanochat GPT patched with the hytorch bridge.
    priors: TrainingPriors accumulated during training (phase-4 tables).
    detector: object with .score(X)->[0..1] (LogisticDetector) or None
              (then raw F-vector is journaled without a score).
    tau: LOW_CONFIDENCE threshold from phase-4a held-out calibration.
    """

    def __init__(self, model, pol, ledger: GenerationLedger | None,
                 priors: TrainingPriors | None = None, detector=None,
                 tau: float = 0.5, n_units: int | None = None,
                 turn_id: str = "turn-0", device: str = "cpu",
                 signature_version: int = 1, n_layers: int | None = None):
        self.model = model
        self.pol = pol
        self.ledger = ledger
        self.priors = priors
        self.detector = detector
        self.tau = tau
        self.n_units = n_units or getattr(model, "n_layers", 2)
        self.turn_id = turn_id
        self.device = device
        # v2 (PHASE4-MECHANISM F6-F9) needs the layer structure: n_layers
        # transformer layers × 2 write units (frame id = 2L+unit).
        self.signature_version = signature_version
        self.n_layers = n_layers or getattr(model, "n_layers", None)
        self.head = b"\x00" * 32
        self.flagged: list[int] = []
        self.uncitable: list[int] = []

    def _signature(self, prev_token: int, recs_tok):
        if self.signature_version == 2 and self.n_layers:
            return token_signature_v2(prev_token, recs_tok, self.priors,
                                      self.pol.k, self.n_layers, n_units=2)
        return token_signature(prev_token, recs_tok, self.priors,
                               self.pol.k, self.n_units)

    @torch.no_grad()
    def generate(self, prompt_tokens: list[int], max_new: int = 64,
                 temperature: float = 0.0, eos: int | None = None) -> dict:
        toks = list(prompt_tokens)
        emitted, sigs, scores = [], [], []
        t0 = time.time()

        for step_i in range(max_new):
            x = torch.tensor([toks], dtype=torch.long, device=self.device)
            journal: list = []
            out = self.model(x, journal)
            logits = out[0] if isinstance(out, tuple) else out
            last = logits[0, -1]
            if temperature > 0:
                probs = torch.softmax(last / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            else:
                nxt = int(last.argmax().item())

            # ---- facts for the LAST position (the acts behind this token) ----
            # Each journal entry is one (layer, unit) write; v2 signatures
            # (F6/F9) need that structure, so recs are widened with a
            # 'layer' column carrying the frame id (2*L + unit).
            pos_last = len(toks) - 1
            parts = []
            for e in journal:
                r = e.recs()
                r = r[r["pos"] == pos_last]
                if not len(r):
                    continue
                meta = getattr(e, "meta", None)
                frame = 2 * meta[0] + meta[1] if meta else 0
                w = np.zeros(len(r), dtype=r.dtype.descr + [("layer", "<u2")])
                for name in r.dtype.names:
                    w[name] = r[name]
                w["layer"] = frame
                parts.append(w)
            recs_tok = np.concatenate(parts) if parts else np.zeros(0, dtype=[("verdict", "u1")])
            n_c = int((recs_tok["verdict"] == VERDICT_COMMIT).sum()) if len(recs_tok) else 0
            n_o = int((recs_tok["verdict"] == VERDICT_OVERFLOW).sum()) if len(recs_tok) else 0
            n_a = int((recs_tok["verdict"] == VERDICT_ABORT).sum()) if len(recs_tok) else 0

            sig = None
            score = None
            if self.priors is not None and len(recs_tok):
                sig = self._signature(toks[-1], recs_tok)
                sigs.append(sig)
                if self.detector is not None:
                    score = float(self.detector.score(sig[None, :])[0])
                    scores.append(score)

            # chain the token facts (T2-style, over the fact payload)
            payload = {
                "token_id": nxt, "index": step_i,
                "n_commit": n_c, "n_overflow": n_o, "n_abort": n_a,
                "signature": [round(float(v), 5) for v in sig] if sig is not None else None,
                "score": round(score, 5) if score is not None else None,
                "flagged": bool(score is not None and score >= self.tau),
            }
            self.head = _facts_head(self.head, json.dumps(payload, sort_keys=True).encode())
            payload["head"] = self.head.hex()

            citable = True
            if self.ledger is not None:
                citable = self.ledger.commit_fact(
                    f"gen/{self.turn_id}/token/{step_i:05d}", payload)
                if not citable:
                    self.uncitable.append(step_i)
                # The "something bad documented" moment: the flag is a FACT
                # committed before emission, not a post-hoc log line.
                if payload["flagged"]:
                    self.flagged.append(step_i)
                    self.ledger.commit_fact(
                        f"gen/{self.turn_id}/token/{step_i:05d}/LOW_CONFIDENCE",
                        {"score": payload["score"], "tau": self.tau,
                         "signature": payload["signature"]})

            toks.append(nxt)
            emitted.append(nxt)
            if eos is not None and nxt == eos:
                break

        # TURN receipt seals the chain.
        turn = {
            "n_tokens": len(emitted),
            "final_head": self.head.hex(),
            "n_flagged": len(self.flagged),
            "flagged_at": self.flagged,
            "n_uncitable": len(self.uncitable),
            "wall_s": round(time.time() - t0, 2),
        }
        if self.ledger is not None:
            self.ledger.commit_fact(f"gen/{self.turn_id}/TURN", turn)
        return {"tokens": emitted, "turn": turn,
                "signatures": [s.tolist() for s in sigs],
                "scores": scores}
