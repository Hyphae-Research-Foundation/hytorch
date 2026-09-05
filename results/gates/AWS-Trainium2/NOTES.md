# Trainium2 (trn2.3xlarge, torch_xla 2.9 / Neuron SDK 2.31) — 2026-09-03

*Note: commit hashes cited in this file (e.g. `dbc166e`, `45dcc17`, `c09092a`) refer to the private development history; the public repository was published as a single release commit on 2026-09-05 after sanitizing operator identifiers.*

**CB 96h `cr-0e1d67d2d6ab5868f`, instance i-0c8de80a3f6139aa1, sa-east-1b.**

## What PASSED on the third silicon

`python/tests/test_two_phase_device.py` (device proposal on TensorE, host
reference disposes, apply):

| gate | result |
|---|---|
| D1 every COMMIT mag == exact host recompute | PASS, 4 batteries |
| D2 device apply == apply_ref replay, bit-for-bit | PASS (see finding 1) |
| D3 device-proposal miss-rate vs exact policy-6 | **0.000%** (threshold 0.5%) |
| D4 propose() latency, d20-shaped policy, M=32 | 1.5–2.0 ms / 128 tokens (post-compile) |
| host gate (test_two_phase.py) on trn2 CPU | ALL PASS |
| HyphaeWrite forward+backward on XLA (d2) | PASS after finding 2 |

**Claim for the paper (widened, honest):** facts produced on AWS Trainium2
under policy 7 are bit-exactly verifiable and replayable by the same pinned
reference that gates NVIDIA and AMD. Third architecture, same evidence
contract.

## Findings (all upstream-relevant)

1. **Neuron/XLA compiler does not honor the bf16 apply contract.** The
   identical torch code that is bit-exact on CPU drifts 1–4 ulp on every
   touched leaf on device (fp32 FMA/rounding choices). Fix shipped: apply
   moves to the host reference by default (touched leaves only, 160
   scalars/token at d20); `HYTORCH_DEVICE_APPLY=1` re-enables device apply
   where a device passes D2 (CUDA/HIP kernels do). "Reference disposes" now
   covers selection AND apply.
2. **STE mirror backward SIGSEGVs under the Neuron PJRT backend** (boolean-
   mask gathers + uint16<<16 bit reinterpret in autograd). Fix shipped:
   host-side mirror (`HyphaeWrite._backward_host`) fed from the numpy
   commit facts; dense grads cross the boundary. d2 fwd+bwd verified
   finite, codebook grad nonzero.
3. **Lazy-tensor graph fragmentation (the blocker for full training):**
   the two-phase forward moves proposals to the host *inside* each write
   unit (`.cpu()` in dispose), which cuts the XLA graph 2×L times per
   forward. Every cut compiles a fresh NEFF: the d2 smoke produced **328
   unique modules in 23 minutes and never reached step 1** (compile ~5 s
   each, killed at 13:31Z). This is not a correctness issue — it is the
   architecture of policy 7 meeting a lazy compiler.

## UPDATE 21:50Z — TRAINING ON TRAINIUM2 WORKS (same window)

Finding 3 was fixed in ~2h without a two-pass redesign: the graph
fragmentation came from SHAPE-VARYING device ops (gather of `n_commits`
touched leaves, per-fact index uploads), not from the host round-trip per se.
Fixes: fixed-shape host apply (one full-residual D2H + dense `where` mask),
journal fields kept on host under XLA, STE mirror on host. Result:
- single-process d2: 45 NEFF compiles total, then ZERO new compiles,
  0.1 s/step steady state
- **DDP 4 logical cores + seam + receipts, d2, 30 steps: rc=0.** Loss
  9.01 → 6.16. 30/30 receipts, STEP chain continuous (c_prev==c_next),
  policy_id=7 in every STEP, **BYPASS facts** per step (resid/x0 lambdas —
  Law 0 with declared exceptions, live on a third architecture), channel
  commit 60–87%, abort 0%. Post-compile throughput 24.6k tok/s at
  TBS=16k (tiny model; compile-dominated warmup ~8 min).
- ledger: `run/trn2-d2-smoke/...` on the instance (`/opt/hyt/hyphae`).

Claim upgrade for the paper: not just "facts verifiable on a third silicon"
— the transactional residual stream TRAINS on Trainium2 under DDP with the
same seam, same receipts, same guard. Third architecture, full stack.

## UPDATE 2026-09-04 02:30Z — d10 blocker ISGV902 root-caused (compiler TopK limit)

`run-d10.sh` (d10 a64, 60M, DBS=4/TBS=32k) failed on the catalog arm with
`[NCC_INAS001] ... error code ISGV902 does not exist`, an internal assertion
in `SimplifyTongaTensor.buildAccessRanges → IntegerSetAnalysis` (neuronx-cc
2.26.6360). Bisected compile-only (trace under `PJRT_DEVICE=CPU`, dump the
HloModuleProto with `torch_xla._XLAC._get_xla_tensors_hlo_proto`, feed it to
`neuronx-cc compile --framework=XLA` with the production flags — no
NeuronCore needed; scripts in `/opt/hyt/isgv/` on the box):

4. **`AwsNeuronTopK` cannot be lowered for a 32768-wide row with k ≥ 16.**
   The proposer's `dot` ([nt,64,10]×[64,512,10]) and its 512 MB fp32
   transpose compile fine; `torch.topk(scores[nt, 32768], 32)` alone trips
   ISGV902 for any nt (512, 4096), any d_slot (2, 10), fp32 or bf16.
   Rows ≤ 16384 with k=32 compile; a 32768 row compiles only for k ≤ 8.
   Only a 2-D, last-dim, `sorted=True` topk is pattern-matched to the
   custom call at all — `sorted=False`, 3-D topk and `torch.sort` all lower
   to HLO `sort`, which trn2 rejects outright (`NCC_EVRF029`). The d2 smoke
   never saw this because it used `HYTORCH_NF=256`.
   Fix shipped (commit dbc166e): the proposer top-M is two 2-D topks —
   per-slot top-M over the 512 features of each slot, then top-M over the
   64·M survivors. Exact for the top-M *set* (an element of the global top-M
   has < M scores above it, hence is in its slot's top-M); ties may resolve
   differently, phase B re-ties exactly. Bonus: no transpose of the
   [nt, S, per_slot] scores. `HYTORCH_PROPOSE_TOPK=auto|global|two_stage`
   (auto → two_stage when NF > 16384). CPU gate G5 asserts two_stage ==
   global (ids and order) on 256 tokens × 32768 features. Isolated compile
   of the full proposer with two-stage: OK in 17 s.

Also learned on the vanilla arm: `--max-train-seconds` counts compile time
(`budget_started` precedes the loop), and the d10 vanilla step graph takes
~2 h per NEFF at DBS=4 — the first run spent its whole 7200 s budget
compiling. `run-d10b.sh` uses 36000 s.

## UPDATE 2026-09-04 04:45Z — second d10 blocker: NCC_EXTP003 (d_slot-shaped tensors)

With the TopK fixed, the catalog arm compiled 239 NEFFs in ~1 h and then
died on the fwd+bwd+all-reduce graph of the warm-up step:
`[NCC_EXTP003] Instructions generated by compiler 5997056 exceeds the
typical limit of 300000`. The compiler's own bottleneck report names the
macro: `transpose_1x10` — **10 = d_slot**. Histogram diff against the
vanilla fwd+bwd graph (which compiles in ~11 min): the only families
present in the catalog graph and absent in vanilla are `select/convert/
reshape bf16[4096,64,10]` (~180, the fixed-shape apply `where()` and the
in-graph proposal clip, recomputed inside the big graph because torch_xla
materializes only the tensors a `.cpu()` asked for) and the codebook grad
`f32[32768,10]` (41 uploads + 39 adds). Each in isolation compiles (probes
in `trn/tools/d10_probe.py`): the blow-up needs the real graph, where the layout
of `h` is fixed by the neighbouring matmuls and every reshape to
`[nt, S, d_slot]` becomes one 1×10 transpose per leaf row.

5. **No device tensor may carry `d_slot` as a dimension on XLA — and the
   one that mattered was the codebook gradient.** First fix (commit
   45dcc17): `apply_commits` expands the commit mask to leaves on the host
   and `where()`s on `h`'s own `[B,T,D]`; `clip_proposal` on XLA computes
   per-slot sums of squares and the scale expansion as one-hot matmuls on
   `[nt, D]` (CPU gate G6). That removed every `[4096,64,10]` tensor from the
   graph — and the instruction count stayed at exactly 5,997,056. The
   remaining `d_slot` tensors were the codebook grad `f32[32768,10]` (41
   host uploads from the STE mirror, 39 device adds, convert + reshape into
   the flat bucket, slice + reshape back out): neuronx-cc linearizes each
   `[32768,10]` with one `transpose_1x10` per row. Second fix (commit
   c09092a): the codebook gradient never enters the device graph. The host
   mirror accumulates it in `hytorch.model.HOST_CODEBOOK_GRAD` and returns
   `None` to autograd; `trn_bridge.sync_codebook_grad()` all-reduces the
   host sum over a gloo group (1.3 MB) and uploads it once as
   `codebook.grad` right after `GradBucket.all_reduce()` (both call sites,
   `apply-patch.sh` edit #5), so the optimizer graph sees it only as an
   elementwise `[32768,10]` input (compile-only probe `optim_cb`: OK). CPU
   gate G7: host accumulator == autograd grad bit-for-bit. Side effect: the
   logged bucket `gnorm` excludes the codebook (its bucket slice is zero);
   the seam's dead-channel guard reads `codebook.grad` directly, unaffected.
   Relaunched as `run-d10d.sh`.

6. **Killing torchrun mid-compile leaves a stale `model.hlo_module.pb.lock`
   in `/var/tmp/neuron-compile-cache/.../MODULE_*/`** and every later run
   that needs that module waits on it forever ("Another process must be
   compiling ... been waiting for N minutes" with no `neuronx-cc` alive).
   Check `pgrep neuronx-cc` before believing that message; remove the
   specific stale lock. (vanilla3 sat 10 min on the lock vanilla2's kill
   left behind.)

7. **The codebook must not be in the flat all-reduce bucket either.** With
   its grad host-resident (finding 5), `GradBucket` zero-fills the codebook's
   slice of the 60M-element buffer — the one structural difference from the
   vanilla bucket graph — and neuronx-cc dies in `MaskPropagation`
   (`[NCC_IMPR902] call to isl_set_union failed: spaces don't match`, take
   4). Fix: `GradBucket` skips `hytorch_codebook` (`apply-patch.sh` edit
   #6), so the catalog bucket graph is byte-for-byte the vanilla one
   (59637780 elements). Take 5 = `run-d10e.sh`.

## UPDATE 2026-09-04 10:40Z — d10 CATALOG TRAINS; the preregistered guard killed take 5 (correctly)

Take 5 (`run-d10e.sh`, `catalog5.log`, ledger `/opt/hyt/d10/hyphae5`,
run `trn2-d10-catalog`): warm-up 27 min (cache), then **149 training steps
on 4 logical NeuronCores with seam + receipts every step**, STEP chain
continuous (seam heads logged per step), ~2,900 tok/s at TBS 32k (≈11 s/step,
host dispose of 20 write units × 4,096 tokens included), loss 9.01 → 5.42 at
step 100. Killed at step 149 by the executable guard:
`PREREGISTERED GUARD: usage_entropy 6.90b < 8.0b sustained 50 steps —
channel collapse, killing the run`. The channel had been at **commit 12.5 %
= exactly 1/k, overflow 87.5 %, abort 0 %, distinct features = 512 = one
slot's worth, from step 0**: every token's 8 winners lived in ONE slot.

8. **The two-stage top-M was set-equivalent to the global top-M except
   under ties — and ties are the init condition.** nanochat zero-inits the
   output projections, so at step 0 every proposal is exactly 0 and every
   score ties. The exact policy (and a lowest-index-first topk) resolves the
   tie to features 0..k-1 = one feature in each of slots 0..k-1, and the
   zero-score "reservoir" (features asc) is what lets new slots enter the
   channel whenever fewer than k scores are positive. The slot-major second
   stage (`j = s·M + i`) resolved every tie into slot 0's candidates, so
   phase B could only ever commit slot 0, the STE mirror only ever trained
   slot 0's projection rows, and the collapse was self-sustaining — exactly
   the C4 failure, this time caught live by the guard on the third silicon.
   G5 had only tested Gaussian scores. Fix (commit pending): explicit,
   device-independent tie-break by composite key (`score − p·2^-40` in
   stage 1, `− s·2^-50` in stage 2 — below the fp32 ulp of any non-tiny
   score, so only exact ties are ordered, feature-ascending like the
   reference); neither CPU `torch.topk` nor the device TopK promises
   lowest-index-first. G5 now includes exact-tie batteries (zeros: reservoir
   == features 0..31, miss-rate 0; const; one live slot). Take 6 =
   `run-d10f.sh`.

Side observations from take 5: host memory peaked at 81/124 GB during the
step-1 compiles (5 `neuronx-cc` in flight) and settled at 20 GB; catalog
matrix-grad norm at step 0 is 2.6e-4 vs the twin's 1.7e-2 (the mirror only
returns gradient through committed leaves).

## UPDATE 2026-09-04 13:10Z — take 6: tie-break correct, guard kills again at 149 (declared result)

Take 6 (`run-d10f.sh`, `catalog6.log`, ledger `hyphae6`, run
`trn2-d10-catalog`, manifest `manifest-d10.json` sha `c2f3e86c…`): step 0
now commits across slots (commit 62.6 %, 1,425 distinct features — the
tie-break works), then the channel concentrates: step 4 commit 15 % /
entropy 1.11 b; step 50 commit 13.2 % / 5.04 b; step 100 13.3 % / 6.12 b;
step 149 7.36 b. Abort 0 % throughout, receipts every step, ~2,900 tok/s,
loss 9.01 → 5.36 at step 100. **Killed by the preregistered guard at step
149** (`usage_entropy 7.36b < 8.0b sustained 50 steps`, after the 100-step
warm-up). This is the declared outcome for `manifest-d10.json` and is
reported as such: at d10 under policy 7 the channel is alive (commit 13 %,
abort 0) but usage is too concentrated to clear 8 bits within 150 steps.
The entropy was rising monotonically (5.0 → 6.1 → 7.4 b), so a SECOND
manifest was declared before any further run: `manifest-d10-w200.json`
(sha `0f54144c…`), identical except `channel_guard_warmup_steps` 100 → 200
(threshold 8.0 b and window 50 unchanged; `$comment_w200` records the
declaration time and reason). Take 7 (`run-d10g.sh`, run id
`trn2-d10-catalog-w200`, `catalog7.log`, `hyphae7`) runs it AFTER the
vanilla arm (`run-d10f.sh`'s second half, `vanilla6.log`) finishes. Both
manifests are committed under `manifests/trn2/`.

## UPDATE 2026-09-04 14:50Z — vanilla twin d10 COMPLETE on Trainium2

`vanilla6.log` (second half of `run-d10f.sh`, same tree with `HYTORCH_CATALOG`
unset — bit-identical to upstream): **400/400 steps, final loss 3.414**,
steady **84.5k tok/s** (0.39 s/step at TBS 32k on 4 logical cores) once the
graphs stabilised; it needed 6 distinct large NEFFs in total (1–2 h each,
accumulated across runs 1–6 via the compile cache), then compiled nothing
new for 380 steps. 5,794 s wall including the last compile.

Matched-step losses (same seed and data order), vanilla / catalog (take 6):
step 50 5.542 / 5.521 · step 75 5.602 / 5.623 · step 100 5.212 / 5.359 ·
step 125 5.285 / 5.566. The catalog tracks the twin early and the gap opens
(0.02 → 0.28 nats) while its channel commits ~13 % with usage entropy
5–7 bits — the C4 pattern: the loss looks healthy through the declared
bypass scalars while the typed channel carries little.

Throughput: catalog ~2,900 tok/s vs twin 84,500 tok/s = **29× slower** on
this stack. The cost is the per-write-unit host round trip (20 units ×
D2H of 4,096 × 640 delta and residual + Rust dispose + H2D of the new leaves
+ mask, ×2 microbatches), not the device proposer. The two-pass forward
(propose all units → one dispose → one apply) sketched below is the lever.

## UPDATE 2026-09-04 16:20Z — take 7 COMPLETE: catalog 400/400 under manifest-d10-w200

`catalog7.log`, ledger `hyphae7`, run `trn2-d10-catalog-w200`: 400/400 steps,
rc=0, **zero guard warnings**; loss 4.667 at step 399 (twin 3.414); channel
commit 13.2 → 14.0 %, overflow 86 %, abort 0 %, usage entropy 8.39 b at step
200 → 9.03 b at 350, distinct features 1,779 → 3,062; 2,813 tok/s steady;
400/400 receipts, chain continuous; 2.2 GB ledger. Full table in
`results/trn2-d10/RESULT.md`. Twin checkpoint regenerated in 4 min from the
NEFF cache (`run-d10h.sh`; identical final loss). Everything custodied in
`s3://hytorch-custody/trn2-d10-20260904/` (56 objects, 5.37 GB).

## What a LARGER Trainium run needs (next)

Restructure the forward so the device proposes for ALL 2L write units in
one graph, then ONE host round-trip disposes all units, then ONE device
apply graph. That requires a two-pass forward (propose pass → dispose →
apply pass) or a deferred-dispose design where the residual update is
applied at the next layer boundary. It is a real design change to
HyphaeWrite (a day of work + gates), not a flag. With the CB expiring
Sep 7 11:30 UTC and the AMD result already in hand, we record the gate
PASS + findings as the Trainium contribution and leave full training as
stated future work.

## Cost
CB 96h $214.56 (prepaid). Instance used ~2 h of the window for the above.
