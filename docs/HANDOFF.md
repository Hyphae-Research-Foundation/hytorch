# HANDOFF — hytorch, for the next operator (Fable 5.1 / any agent)

*Written 2026-09-03 22:40 UTC; updated through 2026-09-05 (Trainium d10 done,
repo reorganized, paper v1.0). This file is the single source of truth for
"where we are and how". Read it fully before touching anything. Companion
docs: `README.md` (project overview, in English), `infra/RUNBOOK.md`
(invariants + incidents), `results/models/d20-p5/*.md` and
`results/trn2-d10/RESULT.md` (results), `paper/main.tex` (v1.0).*

---

*Note: commit hashes cited in this file (e.g. `dbc166e`, `45dcc17`, `c09092a`) refer to the private development history; the public repository was published as a single release commit on 2026-09-05 after sanitizing operator identifiers.*

## 0. TL;DR

- **Phase 5 on AMD is CLOSED and fully custodied in S3.** Catalog d20
  trained 4980/4980 with the channel alive; CORE 0.015 vs dense twin 0.236;
  SFT done (bpb 0.875), chat_eval at chance (ChatCORE 0.0016). Everything
  (1.2 TB: models, 25 checkpoints, SFT, evals, full ledger) is in
  `s3://hytorch-custody/phase5-d20-20260903/` with verified
  hashes. **Zero DigitalOcean resources remain** (droplets and volume deleted).
- **Trainium2 d10 (60M) catalog-vs-twin is DONE and custodied** (2026-09-04):
  catalog 400/400 steps under the declared `manifest-d10-w200.json`, loss
  4.667 vs twin 3.414, channel alive (commit 13-14 %, abort 0, entropy 9.0 b),
  400/400 receipts, 2.8k vs 84.5k tok/s. Two guard kills at step 149 before
  it (one proposer bug = C4 live, one declared negative). Everything (both
  checkpoints, 3 ledgers, all logs, both manifests; 5.37 GB, sha256 list) in
  `s3://hytorch-custody/trn2-d10-20260904/`. Details:
  `results/trn2-d10/RESULT.md`, `results/gates/AWS-Trainium2/NOTES.md`
  (findings 4-8). The trn2 box is idle; its capacity block ends
  **2026-09-07 11:30 UTC** (nothing further is needed from it).
- **The paper is v1.1 (10 pp, compiles, 23 refs)**: reframed after four external
  reviews as an instrumentation/verification result, not a training architecture
  (`paper/notes/REVIEWS-2026-09.md`). What a top venue will still ask for needs
  compute and is listed in §5 and in the paper's own Limitations.
- **Two real bugs were found and fixed today** that reinterpret earlier
  numbers; both are disclosed in the paper (§5 below).

---

## 1. Credentials & access

Not in this repository. Operators: see `docs/private/OPERATOR-ACCESS.md` on the
operator's machine (git-ignored). Custody buckets are private; access on request.

## 2. What the project is (one paragraph)

hytorch makes the transformer residual stream a *typed, transactional
channel*: blocks propose deltas, a pinned policy (Rust `apply-ref`) packs
them against a learned codebook, allocates one winner per slot, journalizes
every verdict (COMMIT/OVERFLOW/ABORT) as a 16-byte fact, and
`optimizer.step()` is gated by a durable receipt from an embedded Hyphae
ledger. A CPU verifier replays audited microbatches bit-for-bit. The same
reference binary gates NVIDIA, AMD and (as of today) Trainium2 kernels.

---

## 3. State of each workstream

### 3.1 AMD / phase 5 — DONE
- Run: `nano-d20-catalog` (4980 steps, 8×MI355X spot, 22 h, zero
  preemptions, 15.0 s/step) + `nano-d20-sft` (466 steps). Vanilla twin 4980
  steps (1.2 s/step). Everything in S3.
- Results: `results/models/d20-p5/RESULT.md` (CORE table, per-task CSVs in
  `eval/`), `SFT-NAN-ROOTCAUSE.md`, `CUSTODY.md`.
- Interpretation: the k=8 / 32768×20 / 64-slot channel is far too narrow to
  be the *sole* writer of an LM residual (≤160 DoF vs 1280). The model
  learned form, not facts. Reported as-is per the preregistered manifest.

### 3.2 Trainium2 — DONE: d2 smoke + d10 catalog-vs-twin (see results/trn2-d10/RESULT.md)
What works (all committed):
- `python/tests/test_two_phase_device.py` on trn2: D1 exact mags, D2 apply ==
  reference replay, D3 miss-rate 0.000%, D4 propose 1.5–2 ms/128 tok.
- DDP training (4 logical cores, LNC=2) with seam + receipts + BYPASS facts
  at d2: 30 steps rc=0, 30/30 receipts, graph-stable after 45 NEFF compiles
  (0.1 s/step steady). Launcher: `/opt/hyt/run-smoke.sh` on the box.
Three findings (in `results/gates/AWS-Trainium2/NOTES.md`): Neuron compiler
drifts 1–4 ulp on the bf16 apply (→ apply moved to host reference); STE
backward SIGSEGVs under PJRT (→ host mirror); shape-varying device ops
fragment the lazy graph (→ fixed-shape host apply + host-side fields).

**Blocker (d10 a64, 60M) — ROOT-CAUSED AND FIXED 2026-09-04 (commit dbc166e):**
`/opt/hyt/run-d10.sh` failed in the Neuron compiler with `[NCC_INAS001] ...
error code ISGV902 does not exist` on the catalog arm. Compile-only bisection
(scripts in `/opt/hyt/isgv/` on the box; method in
`results/gates/AWS-Trainium2/NOTES.md` finding 4) showed the culprit is
`AwsNeuronTopK` over a 32768-wide row with k ≥ 16 — not the einsum, not the
transpose, not the shapes of d10. The proposer now does the top-M in two 2-D
topks (per slot, then over survivors); same candidate set, G5 gate in
`test_two_phase.py` proves it. The vanilla arm at DBS=16 hit `NCC_EOOM002`
(27.5 GB > 24 GB HBM per core) — DBS=4 fits.
Relaunch 2 (`trn/runs/run-d10b.sh`) then hit a SECOND compiler limit on the catalog
arm's fwd+bwd graph: `NCC_EXTP003` (6M instructions > 300k) from
`transpose_1x10` — `[.,10]`-shaped device tensors are linearized row by row.
Two commits: 45dcc17 (flat apply + flat clip, removes the `[nt,S,d]` tensors)
did not change the count; c09092a (codebook gradient kept on the HOST,
gloo-reduced, uploaded once after the bucket — `trn_bridge.sync_codebook_grad`,
`apply-patch.sh` edit #5) removes the actual culprit. NOTES findings 5-6.
Relaunch 4 then hit `NCC_IMPR902` (MaskPropagation) on the bucket's zero-fill
of the codebook slice → edit #6: codebook excluded from `GradBucket` (finding
7). Relaunch 5 (`trn/runs/run-d10e.sh`) TRAINED: 149 steps with receipts, ~2,900 tok/s,
then the preregistered guard killed it (usage entropy 6.9 b < 8 b × 50 steps)
— the two-stage proposer funnelled the step-0 all-zero tie into slot 0
(finding 8). Fixed with an explicit composite-key tie-break. Relaunch 6 (`trn/runs/run-d10f.sh`): tie-break correct, guard kill again at 149 with
entropy RISING (7.36 b) — the declared negative for `manifest-d10.json`. A
second manifest `manifest-d10-w200.json` (guard warm-up 200, rest identical,
declared 13:20Z) is queued as take 7 (`trn/runs/run-d10g.sh`, run id
`trn2-d10-catalog-w200`, `catalog7.log`, `hyphae7/`) to run AFTER the vanilla
arm (`vanilla6.log`, second half of run-d10f) finishes on its own. Take 7 COMPLETED 16:07Z (400/400, rc=0, 0 guard warnings); the twin's
checkpoint was regenerated (`trn/runs/run-d10h.sh`, `out-vanilla7/final.pt`, same final
loss 3.4142) because `train.py` writes one `out/final.pt`. All custodied. 400 steps, `--max-train-seconds 36000`. Progress:
`cat /opt/hyt/d10/arms6.log`. If a run says "Another process must be
compiling" for long with no `neuronx-cc` process, it is a stale `.lock` from
a killed compile (finding 6) — remove that one lock file.
Compile-only bisection recipe (no NeuronCore needed): trace under
`PJRT_DEVICE=CPU`, `torch_xla._XLAC._get_xla_tensors_hlo_proto([out])`,
`neuronx-cc compile --framework=XLA x.pb --target=trn2 <prod flags>`; decode a
cached `model.hlo_module.pb` with `trn/tools/hlo_dump.py`; the failing compile's
`log-neuron-cc.txt` under `/tmp/ubuntu/neuroncc_compile_workdir/*/` has a
"20 MACROS WITH LARGEST INSTRUCTION COUNTS" section that names the culprit.
Budget: the box costs nothing extra until 09-07 11:30Z. **Stop/terminate it
before the CB ends is not necessary; it just dies with the block.** If the
instance goes `impaired` again (happened once after 328 compiles filled
memory), `AWS_PROFILE=<profile> aws ec2 reboot-instances --instance-ids
i-0c8de80a3f6139aa1` (Reboot is in the IAM policy now). `/tmp` is wiped on
reboot; everything persistent is under `/opt/hyt`, `/opt/hytorch`, `/opt/trn`.
`/opt/hytorch` is an rsync copy of the repo (no `.git`): deploy with `scp`.

### 3.3 Paper — v1.1, 10 pages, compiles (`cd paper && ~/.local/bin/tectonic main.tex`)
Sections: abstract (leads with prereg-caught-our-headline + silent collapse +
the measured price), 5 laws (Law 0 now "single writer, declared exceptions"),
gates, seam (9.5% seam overhead vs 12× channel cost side by side; receipts =
tamper-evident not unforgeable), prereg reversal, the collapse finding (C4),
§The price of a live typed channel (CORE table + per-task fig), two-phase /
Trainium, journaled inference, related work (thin), limitations (discloses the
STE bug), evidence table. Figures regenerate from `results/` via
`paper/make_figs.py`.

### 3.4 Review fixes — ALL 7 done (commits 92d8f13 … ca86a05 + follow-ups)
custody-model.sh rewritten (no pkill -f, fail-loud, hash verify); executable
preregistered guard in `seam_shim._guard_update` (commit-rate + usage-entropy,
manifest thresholds, kills all ranks); BYPASS facts for resid/x0 lambdas;
proposal_clip from manifest via `hyphae_write()`; SeamStore rehydrates chain
continuity on open; RUN_START refuses incomplete policy; spot guard fixed;
CI runs 6 suites; RUNBOOK.md; volume delete in teardown; PIN fixed; circuit
breaker; kernel bounds. Minor leftovers: tests for the 3 Rust binaries,
`t2_walk` unreachable in verify, `common.sh` doctl-at-source, provision
timeout, `knowledge_boundary.py` has no consumer yet.

---

## 4. The two bugs found today — read before interpreting ANY number

**A. Silent channel collapse (PHASE5-CHANNEL-COLLAPSE.md).** At d20 the
proposals exceeded the magnitude cap → 0 commits → no STE gradient → frozen
codebook, while the loss kept falling via nanochat's bypass scalars. Every
"cost" number before 2026-09-01 (3.4's +2.63 nats, phase-1 "LR robustness")
was a dead-channel number. Fixed by the proposal clip + the executable guard.

**B. STE gradient mis-scaled on clipped slots (SFT-NAN-ROOTCAUSE.md).** The
clip was applied to the *detached* input inside `HyphaeWrite.forward`, so
autograd never saw it: the mirror returned gradients for the clipped
proposal that flowed into the unclipped block output (up to 400× too large),
compounding to inf in 13 layers on the SFT distribution. Fixed:
`hyphae_write()` clips in-graph; all callers route through it. **The phase-5
pretraining ran with the buggy gradient.** Facts/receipts are unaffected
(they record what was written). CORE 0.015 is the result of *that* run; a
rerun with the corrected mirror is the next experiment the paper promises.

---

## 5. Open experiments, ranked by value

1. **Rerun d20 pretraining with the corrected STE mirror** (the paper's
   stated next step). Needs 8×MI355X spot on DO (~$800–900, ~24 h; use
   `infra/nanochat-supervise.sh` with `HYTORCH_SAVE_EVERY=200`, see RUNBOOK
   §2) or wider Trainium once ISGV902 is understood.
2. ~~Trainium d10 both arms~~ **DONE 2026-09-04** (`results/trn2-d10/RESULT.md`).
   Follow-ups if the box is used again before 09-07 11:30Z: CORE eval of the
   two d10 checkpoints (needs the nanochat eval harness; not on the trn box),
   or the two-pass forward to cut the 29× host round-trip cost.
3. **Phase 4 / H0 (hallucination = bypass at inference).** Design in
   `docs/phases/PHASE4-MECHANISM.md`; instrument `python/hytorch/signatures.py`
   (F1–F9) + `runtime.py` (journaled generation) + `knowledge_boundary.py`.
   Caveat: the catalog model has CORE 0.015 — no knowledge to confabulate
   about; the detector must be validated on the *twin* (dense) or on a
   wider-channel model. Model A (3.4 dead channel) vs model B (p5 live) is a
   free contrast for "absence of facts as signature".
4. ~~Paper related-work section with a real bibliography~~ **DONE 2026-09-04**
   (commit ced3468: 21 entries — Codebook Features, VQ-VAE/STE/collapse, MoE
   balancing, PoL + its attacks, zkPoT, SAEs/transcoders/attribution graphs,
   steering, OLMo, EU AI Act, preregistration). `paper/notes/STATE-OF-THE-ART.md`
   has the notes. Paper is now 7 pp (the bibliography spills onto p.\,7).

---

## 6. Operating rules that were learned in blood (see RUNBOOK for all 10)

- Never `pkill -f <pattern>` from an ssh command containing `<pattern>`.
- Never pause the live seam by hand; read the volume snapshot.
- Nothing unbounded may sit between the trainer and its receipt.
- On spot, `save_every ≤ 250` (enforced).
- A promised threshold must exist as code (guard) in the same commit.
- Any transform the policy judges must be a transform autograd sees.
- Custody before teardown; custody must exit nonzero on any gap.
- After a reboot of the trn2 box `/tmp` is gone; `/opt/hyt` persists.

---

## 7. Quick status commands

```bash
# Trainium box
AWS_PROFILE=<profile> aws ec2 describe-instances --instance-ids i-0c8de80a3f6139aa1 \
  --query "Reservations[].Instances[].[State.Name,PublicIpAddress]" --output text
ssh -i <trn2-key> ubuntu@<trn2-ip> 'cat /opt/hyt/d10/arms.log; tail -2 /opt/hyt/d10/spool/seam.stderr'
# S3 custody
AWS_PROFILE=<profile> aws s3 ls s3://hytorch-custody/phase5-d20-20260903/ --recursive --summarize | tail -2
# DO (should be empty)
doctl compute droplet list; doctl compute volume list
# tests
.venv/bin/python python/tests/test_two_phase.py; cargo test --release
# paper
cd paper && ~/.local/bin/tectonic main.tex
```
