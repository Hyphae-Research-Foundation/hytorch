"""DDP-aware seam shim for nanochat training (rank-collective barrier).

Usage in the trainer loop (scripts/base_train.py patch, step 3.1+):

    seam = SeamShim.maybe_create(...)      # after dist init
    ...
    entries = drain_journal()              # from hytorch_bridge
    head = seam.step_barrier(step, entries)  # ALL ranks: gate optimizer.step()
    optimizer.step()
    seam.step_chain(step, lr, grad_norm, model.hytorch_codebook, resets=[])

Semantics (spec §5.4 + INTEGRATION.md):
- Every rank spools its OWN frames (microbatch_id=rank; frame file suffix
  -rank-R). Only rank 0 runs the hytorch-seam process and waits receipts.
- The receipt head is broadcast; a BarrierTimeout on rank 0 aborts ALL ranks
  (dist.broadcast of a poison value) — no partial survivors, no ledger-less
  degradation.
"""

from __future__ import annotations

import hashlib
import os
import subprocess

import numpy as np
import torch
import torch.distributed as dist


def _h_canonical(codebook: torch.Tensor) -> bytes:
    import sys
    sys.path.insert(0, os.path.join(os.environ["HYTORCH_ROOT"], "python"))
    from hytorch.applyref import fp32_to_bf16_bits
    bits = fp32_to_bf16_bits(codebook.detach().to(torch.float32).cpu().numpy())
    h = hashlib.sha256()
    h.update(np.array(bits.shape, dtype="<i8").tobytes())
    h.update(b"bf16")
    h.update(bits.astype("<u2").tobytes())
    return h.digest()


class SeamShim:
    POISON = "0" * 64

    @classmethod
    def maybe_create(cls, run_id: str, data_dir: str, spool: str, seam_bin: str,
                     manifest_sha: str, build: dict, policy: dict,
                     budget_ms: int | None = None, control_pg=None):
        if os.environ.get("HYTORCH_CATALOG", "") != "1":
            return None
        # Phase 5: the ledger lives on a DO volume (~300MB/s); wired steps
        # fsync ~3.5GB. Default 60s; HYTORCH_BARRIER_MS overrides.
        if budget_ms is None:
            budget_ms = int(os.environ.get("HYTORCH_BARRIER_MS", "60000"))
        return cls(run_id, data_dir, spool, seam_bin, manifest_sha, build,
                   policy, budget_ms, control_pg)

    def __init__(self, run_id, data_dir, spool, seam_bin, manifest_sha, build,
                 policy, budget_ms, control_pg=None):
        import sys
        sys.path.insert(0, os.path.join(os.environ["HYTORCH_ROOT"], "python"))
        from hytorch.seamclient import SeamClient

        # Control-plane process group (barriers + head broadcast). The XLA
        # backend (Trainium) lacks broadcast_object_list/barrier semantics we
        # rely on; trainers there pass a gloo group. None = default group
        # (CUDA/ROCm path, unchanged).
        self.pg = control_pg
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.budget_ms = budget_ms
        # ---- preregistered guard thresholds (manifest > env > defaults) ----
        th = {}
        try:
            import json as _json
            _man = _json.load(open(os.environ["HYTORCH_MANIFEST"]))
            th = _man.get("thresholds", {}) or {}
        except Exception:
            pass
        self.g_min_entropy = float(os.environ.get(
            "HYTORCH_GUARD_MIN_ENTROPY_BITS", th.get("codebook_min_usage_entropy_bits", 8.0)))
        self.g_min_commit = float(os.environ.get(
            "HYTORCH_GUARD_MIN_COMMIT_RATE", th.get("channel_min_commit_rate", 0.02)))
        self.g_window = int(os.environ.get(
            "HYTORCH_GUARD_WINDOW_STEPS", th.get("channel_guard_window_steps", 50)))
        self.g_warmup = int(os.environ.get(
            "HYTORCH_GUARD_WARMUP_STEPS", th.get("channel_guard_warmup_steps", 100)))
        self._bad_streak = 0
        self._guard_kill = None
        self.client = SeamClient(spool, policy_id=policy["policy_id"])
        self._chain_prev = None
        self.proc = None
        if self.rank == 0:
            os.makedirs(spool, exist_ok=True)
            env = dict(os.environ)
            env.setdefault("HYTORCH_WIRE_EVERY", "1")
            self.proc = subprocess.Popen([seam_bin, data_dir, run_id, spool],
                                         env=env,
                                         stderr=open(spool + "/seam.stderr", "ab"))
            infra = {"device_slug": os.environ.get("HYTORCH_SLUG", "unknown"),
                     "region": os.environ.get("HYTORCH_REGION", "unknown"),
                     "procurement": os.environ.get("HYTORCH_PROC", "unknown"),
                     "driver": os.environ.get("HYTORCH_DRIVER", "unknown")}
            # Child-run citation (§5.4): a resumed run is a NEW run_id whose
            # RUN_START cites the parent run, its last receipt head, and the
            # checkpoint step. The launcher injects these on resume.
            for k, env in (("parent_run", "HYTORCH_PARENT_RUN"),
                           ("parent_head", "HYTORCH_PARENT_HEAD"),
                           ("resume_step", "HYTORCH_RESUME_STEP")):
                v = os.environ.get(env, "")
                if v:
                    infra[k] = v
            # RUN_START budget = barrier budget (review incident 2026-09-03):
            # opening a 439GB ledger (post-pretraining SFT) took >30s and the
            # hardcoded 30s handshake killed the SFT stage twice. The seam is
            # single-writer and opens the WHOLE run's engine; the wait must
            # scale with the ledger, not with a toy default.
            self.client.run_start(manifest_sha, build, infra=infra,
                                  policy=policy, budget_ms=max(self.budget_ms, 120000))
        if dist.is_initialized():
            dist.barrier(group=self.pg)

    def _guard_update(self, step: int, commit_rate: float, entropy_bits: float):
        """Rank-0 only. Sustained breach of a preregistered threshold arms the kill."""
        if step < self.g_warmup:
            return
        breach = []
        if commit_rate < self.g_min_commit:
            breach.append(f"commit_rate {commit_rate:.3%} < {self.g_min_commit:.1%}")
        if entropy_bits < self.g_min_entropy:
            breach.append(f"usage_entropy {entropy_bits:.2f}b < {self.g_min_entropy}b")
        if breach:
            self._bad_streak += 1
            if self._bad_streak in (1, self.g_window // 2):
                print(f"hytorch: WARNING step {step}: {'; '.join(breach)} "
                      f"(streak {self._bad_streak}/{self.g_window})", flush=True)
            if self._bad_streak >= self.g_window:
                self._guard_kill = (f"PREREGISTERED GUARD: {'; '.join(breach)} sustained "
                                    f"{self._bad_streak} steps (manifest thresholds) — "
                                    f"channel collapse, killing the run (spec: crossing kills the branch)")
        else:
            self._bad_streak = 0

    def step_barrier(self, step: int, entries, codebook: torch.Tensor) -> str:
        """entries: [(frame_layer_id, JournalEntry)] from drain_journal().
        Every rank spools; rank 0 signals + waits; head broadcast gates all.
        Captures c_prev HERE (pre-optimizer) for the step chain."""
        self._c_prev_step = _h_canonical(codebook)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # PREREGISTERED CHANNEL GUARD (review 2026-09-02). Phase-1 manifests
        # promised "codebook_min_usage_entropy_bits: 8.0 — crossing it kills
        # the branch" and NO code ever read it; the d20 collapse ran ~750
        # steps dead at $36/h. Now the guard is executable: rank 0 computes
        # commit rate + feature-usage entropy of this step's commits EVERY
        # step, keeps a window, and if either stays below its manifest
        # threshold for `window` consecutive steps the run dies on ALL ranks
        # (decision rides on the same broadcast as the receipt head).
        # Warmup steps are exempt (proposals are near-zero at init).
        verdict_stats = None
        if self.rank == 0 and entries:
            n_c = n_o = n_a = 0
            feat_counts = {}
            for _, entry in entries:
                r = entry.recs()
                v = r["verdict"]
                n_c += int((v == 0).sum())
                n_o += int((v == 1).sum())
                n_a += int((v == 2).sum())
                if n_c:
                    f, c = np.unique(r["feature"][v == 0], return_counts=True)
                    for fi, ci in zip(f.tolist(), c.tolist()):
                        feat_counts[fi] = feat_counts.get(fi, 0) + ci
            tot = max(1, n_c + n_o + n_a)
            rate = n_c / tot
            if feat_counts:
                cnt = np.array(list(feat_counts.values()), dtype=np.float64)
                pr = cnt / cnt.sum()
                entropy = float(-(pr * np.log2(pr)).sum())
            else:
                entropy = 0.0
            verdict_stats = (rate, n_o / tot, n_a / tot, entropy, len(feat_counts))
            if step % 50 == 0 or step < 5:
                print(f"hytorch: step {step} channel commit={rate:.1%} "
                      f"overflow={n_o/tot:.1%} abort={n_a/tot:.1%} "
                      f"usage_entropy={entropy:.2f}b distinct={len(feat_counts)}",
                      flush=True)
            self._guard_update(step, rate, entropy)
        # entries may repeat (layer, unit) across grad-accum micro-steps;
        # suffix -rRR-mMMM keeps filenames unique and the T2 order
        # deterministic (seam sorts filenames: layer, then rank, then micro).
        seen: dict[int, int] = {}
        for frame_id, entry in entries:
            m = seen.get(frame_id, 0)
            seen[frame_id] = m + 1
            self.client.spool_layer(
                step, frame_id, self.rank, entry.recs(), device=0,
                suffix=f"-r{self.rank:02d}-m{m:03d}")
        if dist.is_initialized():
            # a rank that raised during spooling must not hang the others until
            # the NCCL default; monitored_barrier surfaces the missing rank.
            _be = dist.get_backend(self.pg) if self.pg is not None else dist.get_backend()
            if _be == "gloo":
                import datetime as _dt
                dist.monitored_barrier(group=self.pg, timeout=_dt.timedelta(milliseconds=self.budget_ms))
            else:
                dist.barrier(group=self.pg)          # all frames on disk before the signal

        head_hex = None
        if self.rank == 0:
            from hytorch.seamclient import BarrierTimeout
            try:
                head_hex = self.client.barrier(step, budget_ms=self.budget_ms)
            except BarrierTimeout:
                head_hex = self.POISON
        if dist.is_initialized():
            obj = [head_hex]
            dist.broadcast_object_list(obj, src=0, group=self.pg)
            head_hex = obj[0]
        if head_hex == self.POISON:
            raise RuntimeError(
                f"no CommitReceipt for step {step} — KILL THE RUN on all "
                f"{self.world} ranks (spec §5.4)")
        # Guard decision travels with the head so every rank dies together
        # (a rank-0-only exception would hang the others in the next collective).
        kill = [self._guard_kill if self.rank == 0 else None]
        if dist.is_initialized():
            dist.broadcast_object_list(kill, src=0, group=self.pg)
        if kill[0]:
            raise RuntimeError(f"hytorch: {kill[0]}")
        return head_hex

    def step_chain(self, step: int, lr: float, grad_norm: float,
                   codebook: torch.Tensor, resets, seal: bytes | None = None,
                   bypass: dict | None = None):
        """AFTER optimizer.step() (+renorm): c_prev was captured at the
        barrier (pre-step); c_next is the post-step hash. The seam enforces
        c_prev(t) == c_next(t-1) — codebook moved outside the ledger = death.

        bypass: {name: tensor|list} of residual scalars that mutate h OUTSIDE
        the catalog write (Law-0 declared exceptions). Journalized per step."""
        c_next = _h_canonical(codebook)
        if self.rank == 0:
            bp = None
            if bypass:
                bp = {k: (v.detach().float().cpu().tolist() if hasattr(v, "detach") else list(v))
                      for k, v in bypass.items()}
            self.client.step_chain(step, lr, grad_norm, self._c_prev_step,
                                   c_next, resets, seal,
                                   budget_ms=self.budget_ms, bypass=bp)

    def close(self):
        if self.proc:
            self.proc.terminate()
