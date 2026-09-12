# Proposed admission rule for the conditional factual causal pilot

Prepared 2026-09-10, before any causal localization, matched-control selection,
or treatment outcome has been inspected by this reviewer. This is a **concrete
proposal to freeze in the launch manifest**, not an executed localizer, a
validated matching method, or evidence that a catalog control is competent.
It complements [the causal protocol](FACTUAL-CAUSAL-PILOT-PROTOCOL.md). A change
to these constants after seeing localization or matching measurements requires
a new preserved protocol version and an exploratory label; it cannot silently
repair a failed match. No core or deployment change is made by this document.

## Separate the schedule from scientific admission

The isolated `causal_schedule` API under preparation is a mechanical contract:
`CausalSchedule(branch, target_unit, control_unit, n_units=4,
training_steps=1000, rescue_step=500)`, with branches H/T/C/R. It supplies the
complete intervals `[0,500)` and `[500,1000)`. Primary inference has no veto and
must reject an optimizer-step argument. T and R share a numerical prefix but
have different complete schedules; ordinary strict resume must reject T→R.

Those properties do not establish that the supplied target or control is
scientifically admissible. Before a scientific branch launches, require a
separate immutable admission record. Store its canonical contract and SHA256
under `training.causal_admission`, alongside `training.causal_write_schedule`.
`comparable_training_config` includes the full `training` object, so these
records become part of `training_contract_sha256`. A top-level key omitted from
that function, a mutable path, or an environment variable alone is insufficient.

The admission record binds:

- The frozen rule, constants, control-text/prompt inventories, source and
  hardware qualification, numerical profile and complete seed list.
- The competent healthy reference checkpoint, original initial-state identity,
  architecture/optimizer/data contracts, tokenizer and actual exposure trace.
- The world, discovery/held-out split identities, competence report, full
  localization table, full matching table and any repeated-measurement checks.
- The selected target/control IDs, or a terminal admission-failure reason.
  Every branch of a seed must bind the same admission record.

Recompute report hashes and selection from the complete tables before launch;
do not trust a standalone `eligible: true` or manually supplied unit IDs. No
validation/test labels or T/C/R endpoint, training-loss, gradient or accuracy
report is an input to selection. A mechanical fixture may construct schedules
without a scientific admission, but must remain explicitly a fixture and must
not pass the scientific launch gate.

The healthy reference used to localize is already trained. Its checkpoint may
not be relabeled afterward as a scheduled H run with newly appended admission
provenance. Start H/T/C/R from the same saved pre-acquisition numerical state;
budget any additional healthy-reference run explicitly. Reusing a numerically
equivalent reference as evidence requires a declared equivalence comparison,
not rewriting its original contract. On a fresh world or a new seed, competence
and localization must refer to that world's/seed's healthy reference.

## Healthy eligibility and immutable measurement sets

Keep the existing gate: a catalog reference reaches at least 0.8 common-discovery
macro accuracy and strictly beats both declared entity-free generation priors
in **each** of identity, heldout-shift8 and heldout-gaps4, with valid execution
and provenance. If several V2 catalog configurations qualify, freeze a selection
rule before localization; the first eligible catalog entry in the original
matrix-plan order is a simple deterministic proposal. Dense cells do not enter
this choice. Preserve all eligibility failures and attempted seeds.

The localization set contains **all predeclared common exposed discovery facts**,
with all declared discovery paraphrases and all three positions. Do not restrict
it to the facts H happened to answer correctly, the easiest relation, the best
paraphrase or the best position. Healthy competence is a separate aggregate gate.
Check actual exposure rather than treating a planned repetition count as proof.
Report common facts with zero realized exposure as an admission/data-contract
failure, not an opportunity to drop them from the denominator.

Freeze a control-text inventory before localization. A reproducible choice on
this synthetic task is all original training-text records belonging to discovery
entities, preserving their original multiplicities, factual/mention mix, and
valid-token/EOS weights. Apply all three position conditions with equal condition
weight. This is a narrow, often filler-heavy proxy for general disruption;
matching it does not establish equivalent general language capability. Include
its complete record list and membership hash, not only a generator seed.

Evaluation comparisons use the reserved validation entity/fact split. No
discovery-localizer contrast is a confirmatory effect estimate. The existing
test split remains reserved according to the parent protocol. Selection using
post-treatment loss, an endpoint with the best rescue, or individually
successful facts is forbidden even if it is described as calibration.

## Localizer and its numerical guard

Freeze the healthy parameters, codebook, policy, optimizer, RNG and sampler.
Use evaluation mode with no parameter updates. Evaluate no-veto and each of the
four singleton inference vetoes, with fixed order, prompts, actual position IDs,
reference/backend and computation dtype. Compare numerical and policy/RNG state
before and after; analysis is not permitted to mutate the reference.

For the existing factual tokenizer, each visible `Value_*` is one lexical token.
Assert that property for every selected fact. At the public answer boundary,
append only a **fixed public scaffold**, such as the response-leading whitespace
declared by the protocol. The true Value may index the NLL target but may not
appear anywhere in its predictor input. Check tokenization and the exact input
prefix; do not use the next logit after the Value was already supplied. Compute
`-log_softmax(full_vocabulary_logits)[true_value_token]`. Do not renormalize over
the 16 possible Values or combine this NLL with space, punctuation or EOS.

For each unit, first average paired veto-minus-healthy Value NLL across the
declared prompt/position combinations within each fact, then average equally
across facts. Preserve the per-fact and per-position table. Missing, duplicate,
nonfinite or failed forwards invalidate the battery; none are omitted. Vetoes
still execute the ordinary quantization path and preserve every unit frame.

Proposed localizer constants:

| Field | Proposed value | Meaning |
|---|---:|---|
| `minimum_delta_value_nats` | 0.05 | Aggregate target effect must exceed this fixed floor. |
| `repeat_max_abs_nll_drift` | 0.00001 | Maximum per-query NLL drift on a second fixed battery. |
| `repeat_max_rel_rms_drift` | 0.000001 | Maximum relative drift of each nonzero RMS matching measurement. |
| `measurement_repetitions` | 2 | No retry-until-positive or retry-until-stable loop. |
| `tie_break` | lowest unit index | Break exact ties in the measured unit means. |

Select the largest first-battery effect; require the positive floor in both
batteries, the same argmax/tie-break result, and the stated per-query drift bound.
Report failure as `no-positive-target` or `unstable-localizer`.
The 0.05-nat floor is an explicit pilot design choice (about a 5% geometric
correct-token probability change), not a power calculation, confidence bound,
or established practically meaningful threshold. Report all four effects and
the runner-up gap. It does not justify a claim that only the chosen unit stores
the facts, or that a frozen-model access effect will survive compensation during
training from a pre-acquisition state.

## Pretreatment matching: proposed fixed calipers

Compute every matching quantity on the **same healthy reference** and frozen
control-text battery, before training any T/C/R branch. Singleton veto forwards
for localization/matching are explicitly pretreatment design measurements;
they are distinct from the eventual training-treatment outcomes.

For each unit `u`, record:

```text
g_u   = original valid-token-weighted NLL(singleton veto u) - NLL(no veto)
m_u   = RMS(h_after_ordinary_write_u - h_after_zero_commit_quantized_write_u)
r_u   = RMS(h_after_zero_commit_quantized_write_u)
rho_u = m_u / r_u
p_u   = healthy admitted candidates / healthy total candidates
```

Magnitude measurements are the immediate local applied write contribution on
the healthy trajectory, not the final downstream residual difference after
ablating a unit. Use the same nonpadding valid-target mask and feature dimensions
for both RMS terms; include zeros on valid positions with no write. Compute RMS
from global sums of squares and exact element counts, not an unweighted mean of
batch RMS values. Aggregate the declared three equally weighted positions with
identical membership; they therefore have equal denominators. Record raw sums,
counts and the position-specific values.

For `p_u`, retain the full candidate denominator, including commits, overflows
and original aborts, restricted to those same valid positions. Do not count
only successful writes or merge experimental denials with numerical failures.
Also retain all-position counts, so dropping padded positions from matching
cannot conceal a missing frame. Reject zero/nonfinite `m_u`, `r_u` or `p_u` for
a proposed target or control; no hidden epsilon makes a dead unit match a live
one. Reject unexpected magnitude/nonfinite aborts according to the already
declared qualification rules.

Let `t` be the selected target. Freeze the following constants before looking
at any of these tables:

| Caliper | Proposed value/rule |
|---|---|
| General-loss scale `s_g` | `max(0.02, 0.20 * abs(g_t))` nats per valid token |
| General-loss difference | `abs(g_u - g_t) <= s_g` |
| Absolute update RMS ratio | `1/1.5 <= m_u/m_t <= 1.5` |
| Residual-normalized update RMS ratio | `1/1.5 <= rho_u/rho_t <= 1.5` |
| Admission fraction difference | `abs(p_u-p_t) <= 0.10` (10 percentage points) |

The target-dependent `s_g` is a **predeclared formula using only pretreatment
measurements**; it may not be replaced after seeing a result. Preserve the sign
of `g`: do not match absolute magnitudes of opposite-direction loss effects.
For small effects, the explicit 0.02-nat floor bounds the permitted difference;
it is not an assertion that either loss change is significantly nonzero.

Among nontarget units satisfying **all** calipers, minimize:

```text
D(u,t) = ((g_u-g_t)/s_g)^2
       + (log(m_u/m_t)/log(1.5))^2
       + (log(rho_u/rho_t)/log(1.5))^2
       + ((p_u-p_t)/0.10)^2
```

Break exact ties by lowest unit index. Preserve all three candidates, each
measurement, every individual caliper outcome, distance and the selected ID.
Repeat the matching battery under the same fixed two-pass plan: require the
NLL/RMS drift bounds above, identical integer admission counts, the same admissible
candidate set and the same selected control in both passes. A borderline result
that changes admissibility or selection is `unstable-matching`; do not average
away the failed boundary, widen a caliper or add passes until it qualifies.
Do not estimate normalizing standard deviations from only four units, select
the candidate with the smallest factual effect, drop a failed caliper, tune the
veto strength, or replace a seed to make a comparison admissible.

If none qualifies, record `no-control-match`. Do not launch or label an unmatched
four-branch comparison as the planned matched specificity experiment. An
unmatched ablation requires a separately declared exploratory analysis and
cannot establish factual damage beyond comparable general disruption. These
calipers are a bounded operational proposal, not proof of functional equivalence;
report actual imbalances and limitations even when a candidate passes.

## Runtime checks that must consume the admission

- Runtime architecture must match the selected unit table and `n_units=4`.
  Booleans, duplicate/out-of-range unit IDs and inconsistent target/control IDs
  are rejected; the table and schedule must agree exactly.
- Training step is the zero-based optimizer update. R denies the target at
  steps 0–499 and reactivates before step 500. A generation token index never
  acts as a training step. Localization and diagnostic inference require their
  own explicit policies; primary inference is always unblocked.
- H uses the same scheduler/instrumentation route with an empty veto. Preserve
  independently qualified sham equality on the actual numerical/backend path.
  All branches share initial model/optimizer/RNG/data bytes and example order.
- Record schedule/admission hashes, phase, step, selected units and pre-/post-veto
  candidate counts in each update's provenance. Original denials, overflows and
  new experimental denials remain distinguishable; no losing candidate is
  promoted. Missing frames and numerical failures remain failures.
- The experiment declares all four complete schedules before training. Ordinary
  T→R resume remains invalid despite a shared prefix. A shared checkpoint fork
  needs an explicit fork contract covering parent bytes and the only permitted
  schedule differences; the H/T/C/R labels cannot bypass resume identity.

This admission proposal does not introduce a causal runner, a localization tool,
new treatment data, or cloud execution. Code under preparation still requires
its own mechanical and actual-hardware qualification before scientific use.
