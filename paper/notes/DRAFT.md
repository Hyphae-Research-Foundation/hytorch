# The Missing Medium: A Transactional Residual Stream for Transformer Training

*Draft v0.1 — 2026-08-30. Target: MLSys. Every number in this draft traces to
a file in `results/` or a record key in a Hyphae ledger; the mapping is the
evidence table in §8. Spanish spec lineage: `docs/spec/spec_v2_2.pdf` and the full
correction registry (E1–E13, A1–A6, C1–C15, D1–D5, SPEC_AMEND-001/002).*

---

> **REVISION NOTE (2026-08-30, phase 1b).** Phase 1a (all runs below marked
> "1a") carried a ReZero-style scalar gate on the write path in BOTH arms.
> SPEC_AMEND-003 removed it (user directive: the ledger records the negative
> space regardless; the model need not be born mute — and the gate distorted
> the measurement: the P1 freeze was a gate×mag_max interaction, and the
> gate did heavy stabilization work for the dense baseline). Phase 1b reruns
> the same signed protocol without the gate, standard GPT-2-scaled init in
> both arms: **the branch now PASSES — catalog 167.8 val PPL vs dense twin
> 304.7 (−44.9%) on H200, 309/309 T1 green.** The 1a negative remains
> archived as the with-gate record; §7 is being rewritten around the 1b
> result with the 1a history as the diagnosis arc. **AMD replication
> CONFIRMS: MI355X catalog 169.68 vs twin 298.78 (−43.2%), 317/317 T1 —
> the PASS replicates cross-vendor.**
>
> **SECOND REVISION (2026-08-30, preregistered LR sweep): the headline
> REVERSES under best-vs-best.** Sweeping BOTH arms over {1e-4, 3e-4, 6e-4,
> 1.2e-3} at 20k steps: baseline@3e-4 = 49.33 PPL vs catalog@3e-4 = 144.13
> — the properly-tuned dense twin is ~2.9× better; the −44.9% figure was an
> artifact of the dense twin destabilizing at 6e-4. What survives, receipt-
> backed: the catalog is ~6× more LR-robust (144–188 across the sweep vs
> 49→304 for the twin), 7× more seed-stable (spread 1.0% vs 7.5% over three
> seeds), and 1,869 additional T1 audits are green. §7 will lead with the
> best-vs-best table. C1/C2 never depended on winning PPL; C3's centerpiece
> is now that the preregistered protocol caught OUR OWN too-good headline —
> twice in one project, once against us, once against our celebration.
> [results/train/NVIDIA-H200/lrsweep-441b9d2f6a4f/NOTES.md, seeds-441b.../NOTES.md]

## Abstract

Modern training treats the transformer residual stream as an anonymous
accumulator: any kernel may execute `h += d`, and no record of the write
survives the step. We present **hytorch**, the first training loop in which
the residual stream is a *typed, transactional channel*. Every write is a
16-byte fact — a *binding* with slot, feature, magnitude, and verdict; every
contention (OVERFLOW) and rejection (ABORT) is journalized as a first-class
non-fact; every optimizer step is gated by a durable commit receipt from an
embedded transactional store (Hyphae); and a CPU-only verifier replays audited
microbatches bit-for-bit against a pinned software reference. The same
reference binary gates NVIDIA (sm_89/sm_90/H200) and AMD (gfx950) silicon:
200 000 adversarial cases, bit-identical verdict streams and residuals, with
no vendor branch in the seam. Fault injection over real training spills
detects 32/32 mutations (100%) with the correct failure class each. The
ledger costs 9.5% of wall clock at 20 000 steps. Under a **preregistered**
evaluation — PPL gap ≤ 10% vs a FLOPs-matched dense twin on wikitext-103 —
the cataloged model *misses the threshold* (+11.17%); per the preregistration
contract we kill the branch and publish the negative with its complete audit
trail, including a training pathology (catalog freeze) that only the journal
could see, and a load-balancing result (differentiable aux loss hurts) found
because the gradient contract is itself an audited object. The model may
remain opaque; its state transitions cannot.

## 1. Introduction

The residual stream already is "the medium": every layer reads and writes a
shared channel. The field observed this (superposition, polysemanticity) and
chose to *read* the channel post-hoc rather than *type* it. The consequence
is not just interpretive difficulty; it is the absence of a category: between
geometry (tensors) and claims (model outputs) there is nothing that is a
*thing* — with identity, history, and the possibility of rejection.

This work does not propose interpretability. It asks a narrower, mechanical
question: **who has the right to say `h = h + d`?** In every mainstream
stack the answer is "any anonymous `+=` in any kernel." We change the answer:
the only kernel with the right to write `h` executes a policy constituted by
a transactional authority, and nothing becomes durable, exportable, or
citable without receipts.

Contributions:

1. **A bit-exact write policy as a pinned artifact** (§3): `apply_ref`, a
   dependency-free reference for pack→allocate→apply over bf16, whose `.so`
   SHA-256 is a fact every run cites (C10). Signed-zero application is a
   no-op *by rule* (D1) — IEEE would flip `-0.0` leaves and kill honest runs.
2. **Per-silicon differential gates** (§4): no backend enters the harness
   until its kernels match the reference bit-for-bit on 200k adversarial
   cases *on the exact silicon of the run*. This gate caught a real
   cross-vendor determinism bug (NaN total-order divergence, SPEC_AMEND-002).
3. **Authority ≠ durability** (§5): a seam architecture in which allocate
   verdicts are computed on-device with zero per-layer synchronization, the
   journal leaves as one async D2H per layer into pinned memory, and the
   durable step transaction (embedded Hyphae, group commit) gates
   `opt.step()` with fail-stop semantics — no ledger-less degradation path
   exists, by test.
4. **Two-tier verification** (§6): T2, a per-layer hash chain over persisted
   facts (elision-aware); T1, same-binary CPU replay of audited microbatches
   closing apply↔journal↔codebook via a residual hash. 100% injected-fault
   detection with correct failure classes.
5. **A preregistered negative** (§7): the catalog costs +11.17% PPL at 20k
   steps on a 124M-parameter model vs its dense twin; the signed threshold
   was 10%. The branch dies; the audit trail — including the diagnosis chain
   P1→P2→P3 — is the artifact.

## 2. Design: the five laws

(From spec v2.2 §01; stated here as the system's invariants.)

- **Law 0 (functional):** blocks propose; they never write `h`. Attention and
  MLP read the same normalized `h` and sum into their own buffer (a parallel
  block, GPT-J/PaLM style). Enforced by byte-comparison alias checks.
- **Law 1 (identity):** the write path is born at zero (ReZero-style gate);
  `H(h₀)` anchors each audited microbatch.
- **Law 2 (nameable delta):** every write is a binding `(slot, feature, mag,
  policy_id)`; candidates cross, never whole tensors.
- **Law 3 (visible overflow):** slot contention is a fact (OVERFLOW), not
  silence; rejection is a fact (ABORT) with a reason. Silence (slot not
  proposed) is not a journalizable non-fact.
- **Law 4 (commit or silence):** only `apply()` writes `h`; durable /
  exportable / citable requires receipts.

The catalog: `d_model = S × d_slot` (64 slots × 12 dims), features are rows
of a learned codebook `C ∈ R^{N_f × d_slot}` (N_f = 32 768), home slot
`σ(f) = f mod S`, top-k global per token (k = 8), winner per (pos, slot) by
|mag| bits, abort rules on the bf16 bits (`nonfinite`, `|mag| > mag_max`).
Zero-magnitude commits keep the straight-through gradient alive (C8) and are
elided from the WAL before the chain hash (D1).

## 3. The pinned reference

`apply_ref` (Rust, zero dependencies, cdylib) defines: bf16→fp32 promotion by
mantissa extension (exact); fp32→bf16 RNE downcast with canonical quiet NaN;
sequential no-FMA reductions; ε = 2⁻¹⁴ normalization inside apply (so
‖Δ‖ = |mag| and the binding self-describes, C1); the §04 total order (slots
ascending within token, tokens row-major); and the signed-zero no-op rule.
Property tests include exhaustive 65 536-pattern promotion round-trip, RNE
ties at bit 16, and `-0.0` landmines. The wire format (BindingMin, 16 bytes,
explicit padding, verdict+reason on the wire per D3/D4) and the T2 chain live
beside it.

**NaN canonicalization (SPEC_AMEND-002).** The first H100 gate REJECTED the
backend: 53 869/200 000 pack mismatches. Root cause: x86 emits −qNaN for
invalid operations, CUDA emits +qNaN; under the IEEE total-order key they
sort at opposite ends, so identical inputs selected different features on
host vs device. Both sides now canonicalize NaN scores to +qNaN before
ordering; NaN still surfaces as a journalized ABORT(nonfinite) — now
deterministically. The determinism the spec promised was false as written;
the gate — not review — found it.

## 4. Per-silicon differential gates

The gate loads the pinned `.so`, generates adversarial families (leaves
salted with −0.0, denormals, NaN/inf, zero delta_hat for step-0 dynamics,
dense random), and compares verdict streams and post-apply residuals
byte-for-byte. Verdicts (results table in §8): **BIT_IDENTICAL on NVIDIA
H100 (sm_90), RTX 4000 Ada (sm_89), H200, and AMD MI355X (gfx950)** — the
AMD path uses plain IEEE ops under `-ffp-contract=off
-fhip-fp32-correctly-rounded-divide-sqrt` (the reference's pre-authorized
fallback), the NVIDIA path explicit `__f*_rn` intrinsics under
`-fmad=false`. There is no `if nvidia / if amd` in the seam; the vendor is a
field in the binding, not a control branch.

## 5. Authority ≠ durability: the seam

Per layer (device): pack scores all N_f legal pairs per token (sequential
fp32 dots), selects top-k with total tiebreak, allocates one winner per
(pos, slot), applies committed bindings to disjoint leaves — the parallel
result equals the sequential §04 order because writes are disjoint. Verdicts
stay on device; field extraction is int32 word arithmetic on the raw verdict
tensor; the journal leaves as ONE async D2H per layer into pinned memory,
materialized after backward (`d2h.overlap = backward`). One CUDA/HIP
synchronize per step.

Per step (host): Phase A spools the elided wire frames and blocks on the
barrier — the seam (Rust) walks the T2 chain and commits one atomic batch
(LAYER facts + RECEIPT) into embedded Hyphae; the receipt gates
`opt.step()`. Phase B, after `opt.step()` and codebook renorm (dead-row
reset by pre-renorm norm, D2), commits the STEP record with the *real*
`c_prev/c_next` (H_canonical of the bf16 codebook), CODEBOOK_RESET features,
and SEAL of θ at cadence. The store enforces receipt-before-chain and
codebook chain continuity `c_prev(t) = c_next(t−1)`: a codebook mutated
outside the ledger gets no chainack and the run dies. RUN_START refuses
placeholder digests and a `.so` hash that does not match the loaded
reference (C10).

Failure semantics are load-bearing and tested: seam killed mid-run →
BarrierTimeout → no `opt.step()`; chain broken → no chainack → fail-stop.
An amusing instance: an early seam retry loop spammed stderr at 1 ms,
filled the pipe, and froze — the refusal path now logs once and stays
silent; the E2E refusal test caught it.

## 6. Verification: T1/T2 and fault injection

**T2 (always, all steps):** `head_{ℓ+1} = SHA-256(head_ℓ ‖ wire_ℓ ‖ meta_ℓ)`
over persisted (elided) facts; meta carries layer, n_persisted, n_elided,
policy_id so an empty step-0 layer is not an amorphous link. Any 1-bit flip
in the persisted WAL breaks the chain at the exact link.

**T1 (sampled 1/N):** for audited microbatches the trainer spills h₀, the
step's codebook, the per-layer wire, and the *device's own claim* of
h_final. The verifier — a separate CPU process linking the same `.so` —
replays and demands (a) the recomputed chain equals the receipt head and
(b) the replayed residual hashes to the claim. (b) closes
apply↔journal↔codebook: a tampered codebook replays to a different residual
(the v1 format could not see this; spill v2 can).

**Fault injection over real training spills** (not synthetic verdicts): six
families — mag bit-flip, dropped commit, injected commit, codebook tamper,
h_final lie, reorder. **32/32 detected (100%), each with the correct class**
(head_mismatch / residual_mismatch / apply_error); zero false positives on
honest spills including −0.0 leaves; bf16-rounding-absorbed codebook tampers
are counted honestly as "same computation" rather than missed detections.

Threat model (C6, declared): implementation faults, not adversarial
trainers — T1 selection is predictable from step 0; commit-then-reveal is
future work.

## 7. Preregistered evaluation — and the negative

**Protocol.** Signed before step 0 (manifest sha in RUN_START): PPL gap
≤ 10.0% vs a FLOPs-matched dense twin *of the same parallel topology* (C2),
20 000 steps, wikitext-103 (sha256-pinned bins, gpt2 BPE), seq 512, batch 4,
seed-paired, 124M params, k=8, N_f=32 768, mag_max=64, aux=0.
FLOPs accounting includes pack's N_f-sweep and apply.

**Policy history (all journalized):**
- **P1 (mag_max=8): catalog freeze, found BY the journal.** At step 299:
  COMMIT=2, OVERFLOW=118 986, ABORT=77 620 (exact k-partition). Loss kept
  dropping via embed+head — without the ledger the freeze is invisible.
- **P2 (mag_max=64): freeze resolved** (+2.9% gap at 300 steps).
- **P3 aux sweep on fixed silicon:** the differentiable Switch-style balance
  term *hurts* (coeff 0 → +5.0%; 0.002 → +16.6%; 0.02 → +23.3%); collapse
  remains guarded by the preregistered entropy floor. (The previous aux had
  zero gradient — detached counts — which the gradient-contract audit
  caught; both bugs are part of the record.)

**Result (20k steps, both arms same seed, BOTH vendors):**

| | H200 catalog | H200 twin | MI355X catalog | MI355X twin |
|---|---|---|---|---|
| val PPL | **179.75** | 161.69 | **185.15** | 158.21 |
| gap | **+11.17%** | — | **+17.03%** | — |
| wall | 8 586.6 s | 1 012.5 s | 10 661.1 s | 509.7 s |
| T1 | **351/351** | — | **291/291** | — |
| receipts | 20 000/20 000 | — | 20 000/20 000 | — |
| ledger wall | 9.5% | — | 3.3% | — |

Gap > 10% on both vendors. **The branch dies — replicated.** No post-hoc
threshold motion. The negative ships with: 642 green audits total, continuous
codebook chains, 19 θ-seals each, EXPORT records, write path alive to the end
on both (H200 step 19 999: C=29 418/O=167 127/A=63; MI355X:
C=29 899/O=166 574/A=135 — same verdict regime, P1's pathology stayed fixed).
Absolute PPLs do NOT replicate across vendors and were never promised to
(§04: genesis is vendor-local; the twin baselines differ too, 161.69 vs
158.21). What replicates is the *verdict* and the *audit*; what is
bit-identical is the application of facts and their verification.

**Honest asymmetries.** (1) Catalog wall is 8.5× the twin's — pack's top-k
sweep and the serial spool, not the ledger (813 s of 8 587 s); this is
phase-2 engineering, and FLOPs-matching already accounts pack. (2) 124M/20k
is toy scale; the gap trajectory across policies (+20.6 → +2.9@300 → +11.17@20k)
is reported without extrapolation claims in either direction. (3) OVERFLOW
at ~85% of verdicts names slot concentration as the dominant open phenomenon;
per-slot top-k and learned homes are future POLICY records.

## 8. Evidence table

| claim | where |
|---|---|
| Gate BIT_IDENTICAL H100 | `results/gates/NVIDIA-H100-80GB-HBM3/3820e6370fa3/` |
| Gate BIT_IDENTICAL MI355X | `results/gates/AMD-Instinct-MI355X/9819b6d795ca/` |
| Gate BIT_IDENTICAL H200 + citable run | `results/train/NVIDIA-H200/phase1-final-2698931d5cd7/` |
| Gate BIT_IDENTICAL Ada + P2/P3 sweeps | `results/train/NVIDIA-RTX-4000-Ada-Generation/`, `results/sweep/` |
| Cross-vendor rehearsals 9/9 T1 | `results/gpu-rehearsal/{NVIDIA-H100…, AMD-Instinct-MI355X}/` |
| Fault injection 32/32 | `results/inject/inject-summary.json` |
| P1 freeze / P2 fix | `results/train/NVIDIA-H100…/phase1-k8-nf32768*/NOTES.md` |
| NaN determinism bug | `docs/spec/SPEC_AMEND-002-nan-canonical.md` + first-gate log |
| 1a preregistered negative (with gate) | `results/train/NVIDIA-H200/phase1-final-2698931d5cd7/NOTES.md` |
| 1a AMD replication (negative replicates) | `results/train/AMD-Instinct-MI355X-phase1-final-66f0f86cfed3/NOTES.md` |
| SPEC_AMEND-003 (no ReZero) | `docs/spec/SPEC_AMEND-003-no-rezero.md` |
| 1b validation 300 steps (both selections) | `results/sweep/NVIDIA-H100-80GB-HBM3/p1b-validation-ae8f892ee098/` |
| **1b citable PASS (−44.9%, H200)** | `results/train/NVIDIA-H200/phase1b-global-26d294a3e2a1/NOTES.md` |
| **1b AMD replication PASS (−43.2%, MI355X)** | `results/train/AMD-Instinct-MI355X-phase1b-global-3eddb290eb42/NOTES.md` |
| Correction registry | `docs/spec/el_medio_que_falta_v2*.md`, `docs/spec/SPEC_AMEND-00{1,2}*` |

## 9. Related work (researched 2026-08-30; full analysis in STATE-OF-THE-ART.md)

**Codebook Features (Tamkin, Taufeeque & Goodman, ICML 2024)** is the
closest neighbor on the catalog side: VQ bottlenecks per layer yield sparse,
discrete, interpretable and steerable hidden states — but as *finetuning*
of a trained model, reporting modest *degradation*, and with no notion of
facts: no verdicts, no negative space (OVERFLOW/ABORT do not exist), no
receipts, no gated optimizer, no bit-exact replay, no preregistration. We
train from scratch, journal the negative space, and measure *improvement*
over the dense twin under a signed threshold.

**Proof-of-Learning (Jia et al., IEEE S&P 2021)** is the closest neighbor on
the verification side: checkpoint logs + approximate re-execution with a
tolerance band — a band that later spoofing attacks exploited. Our replay is
bit-exact (pinned bit policy + per-silicon gates), our grain is the
individual residual write (16 B) rather than weight checkpoints, and our
ledger has authority (no receipt → no step), not just posthoc verification.
**zkPoT (Abbaszadeh et al., CCS 2024)** proves training cryptographically at
~15 min *per iteration* (VGG-11); we occupy a different design point —
complete facts, no privacy, 9.5% wall — and the spec blacklists zkML-per-
matmul by construction.

**Backpack LMs (Hewitt et al., ACL 2023)** share the interpretable-by-design
spirit (from-scratch, named units) without ledger, verification, or negative
space. Mechanistic interpretability (SAEs, transcoders, crosscoders, circuit
tracing; Anthropic 2024-25) reads a trained field post-hoc; our semantic
probe reads the journal — orthogonal and composable. MoE routing (Shazeer
2017; Fedus 2021) and VQ-VAE (van den Oord 2017) supply the algorithmic
precedents for selection and quantization, without facts or receipts.
Fully-open training efforts (e.g. OLMo) publish artifacts (checkpoints,
data, logs) — openness, not verifiability of state transitions. On the
demand side, EU AI Act Art. 53/Annex XI already obliges GPAI providers to
document the training process; to our knowledge no existing system produces
machine-checkable evidence of it.

**The conjunction — typed residual facts with negative space + durable
receipts gating the optimizer + bit-exact cross-vendor replay + executable
preregistration — appears in none of the above, alone or pairwise.**

## 8b. Phase 3 (nanochat): the seam on a real training stack

We integrated the catalog into nanochat (Karpathy's full LLM pipeline) as an
additive patch: blocks propose, `catalog_write` is the only residual writer
(TWO write units per layer — first attn/mlp sublayer attribution), the
vanilla arm is the SAME tree with the flag off. Verified progression:

- **d4 smoke (H100)**: loss parity (5.95 vs 5.92 at 60 steps); 7 integration
  bugs found and fixed by the smoke.
- **d12 (H100, 1500 steps)**: gap +1.18 nats — the typed channel costs real
  capacity at scale with phase-1 policy, consistent with the LR-sweep
  reversal; 1500/1500 receipts, 17 GB ledger, chain verified by ledger query.
- **d12 DDP (8×MI355X spot)**: the collective seam works — 8 ranks spool,
  rank 0 commits, the receipt head broadcast gates all optimizer.step()s;
  300/300 steps, 192 frames/step, both arms clean.
- **Parallel pack kernel** (one block/token, packed (key,feature) comparator
  provably order-independent): BIT_IDENTICAL on 200k cases; catalog wall
  drops ~6× vs the serial kernel.
- **Upstream finding**: torch.compile/Inductor on ROCm 7.0 (gfx950) produces
  silent NaNs from step 2 in VANILLA nanochat (bf16 and fp32, Muon and
  AdamW fused steps are compiled); eager is clean. Both arms run dynamo-off
  on ROCm. Reproducible; upstream-reportable.

## 9b. Vendor scope: why two vendors, and what a third would require

The cross-vendor claim is *architectural*, not enumerative: the seam contains
no vendor branch, the reference binary is one, and **the differential gate is
the admission criterion** — any accelerator that can host bit-exact IEEE-754
fp32 scalar arithmetic (RNE, no contraction) and the bf16 bit policy may
enter the harness by passing 200k adversarial cases on its own silicon. Two
independent vendors (NVIDIA sm_89/sm_90/H200, AMD gfx950) suffice to prove
the design is not married to one; a third confirms rather than reveals.

Assessed and deferred (2026-08-30): **AWS Trainium** (NKI kernels exist, but
torch-neuronx's execution model conflicts with our eager per-layer kernels +
pinned async D2H journaling, and fp32 scalar rounding-mode control needs
per-silicon verification) and **Google TPU** (torch/XLA lazy tensors + XLA
fusion actively fight the no-FMA scalar contract; a Pallas port is a
research project of its own). Neither is offered by our execution substrate
(DigitalOcean). The gate remains open: the port cost is theirs to pay, the
admission test is already written.

## 10. Limitations & future work

Toy scale (124M/20k); threshold missed — phase 2 must either close the gap
under new preregistered policies (per-slot top-k, learned homes, richer
codebooks) or the idea is wrong at this design point. Wall overhead of pack.
Adversarial threat model. Resume-with-verification (Adam state, RNG) out of
scope in phase 1. Ledger post-run custody: pull the receipts DB before
teardown (currently runlogs carry heads/receipts; the 186 GB ledger died
with the droplet).

---

*Reproducibility: every run cites `build.apply_ref_hash`, `harness_commit`,
wheel digests, silicon, region, procurement; manifests are immutable and
sha-identified; data bins are sha256-pinned; the correction registry is in
the repo. The negative result is not an appendix — it is §7.*
