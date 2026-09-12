# Conditional factual causal pilot

Prepared on 10 September 2026 for the approved next step. **This document does
not execute the study. Causal training remains conditional on a healthy catalog
control meeting competence.** V2's objective/position factorial is an engineering
experiment; its success would not establish a dead-channel explanation of factual
error or authorize that explanation by itself.

## Entry conditions and scope

Use a frozen catalog configuration that reaches at least 80% common-discovery
macro accuracy in **identity, heldout-shift8 and heldout-gaps4**, beats both declared
entity-free generation priors in each condition, and passes execution/provenance
checks. Preserve invalid, missing, abstention, rare and unexposed outcomes. A dense
success cannot substitute for catalog competence. If no catalog cell meets those
conditions, retain this protocol without running its causal branches.

Freeze the selected architecture, objective, position schedule, tokenizer, optimizer,
data policy, decoding and numerical profile before localization or treatment
comparisons. There are four ParallelBlock layers and one catalog write per layer:
units U0–U3 are those four write sites. They share dimensions but differ in depth;
equal shape alone does not establish functional equivalence.

The operational question is whether preventing one selected unit's admitted
writes **during training** impairs later factual performance more than a comparable
control perturbation, and whether restoring those writes restores performance
within a fixed horizon. It does not isolate a pure plasticity intervention: write
blocking changes forward communication and the associated backward contribution.

## Localization using discovery only

For each eligible training seed, use its healthy reference model to measure a
frozen discovery battery with no parameter updates. Keep prompts, tokenization,
decoding, positions and all nonintervention settings fixed. Never select units
using treatment outcomes, validation/test scores or the largest retrospective
contrast after training.

For unit `u`, calculate:

```text
Delta_Value(u) = mean_fact[mean_prompt,position(
    Value_NLL(healthy model, writes of u vetoed)
    - Value_NLL(healthy model, no veto))]
```

The primary localization stratum is common exposed discovery facts, where the
healthy control has demonstrated competence. Each fact has equal weight, then its
declared paraphrases and position conditions have equal weight. Measure Value-only
NLL over the full vocabulary at the specified answer boundary; no true answer is
placed before its prediction. Response scaffolding/EOS loss must not replace this
quantity. Report rare facts separately and do not select a route based on guessing
unexposed facts.

Choose `u_target = argmax Delta_Value(u)` over U0–U3, breaking exact ties by unit
index. Record the complete four-unit table. A nonpositive or numerically unstable
maximum does not identify an informative target under this rule. This localization
establishes a contribution to **access/prediction in the healthy frozen model**;
it does not prove that the unit is uniquely necessary for acquiring those facts.
Other units may compensate when training begins with the target unavailable.

No feature ID is treated as a stable semantic label. Shared-codebook rows can
change meaning over training and seeds. The declared treatment is selection of
a write site by this algorithm, with the selected anatomical index recorded for
each seed; it is not an assertion that the same semantic circuit was found in
all seeds.

## A comparable control unit

On that same healthy model, measure for every unit:

- The change in original, token-weighted global NLL on a frozen discovery control
  text set, with factual/mention/context components also reported.
- The RMS magnitude of its applied channel update at nonpadding target positions,
  measured relative to the ordinary zero-commit quantized output, plus its ratio
  to residual RMS and its admitted-write rate.

Choose a nontarget unit using only those pretreatment matching measurements, not
its eventual treatment outcome or the smallest factual effect. The launch manifest
must fix the distance, scales, tie-break and matching calipers before the treatment
results exist. A concrete distance is the sum of squared differences in global-NLL
change and log update magnitude, each divided by a declared scale. Select the
nearest admissible nontarget unit and record all candidates and distances.

With only three alternatives, an adequate match may not exist. Do not silently
relax the calipers or call an unmatched unit equivalent. Such a comparison can be
reported as exploratory unmatched ablation, but cannot support the planned claim
of a factual effect beyond comparable general damage. Matching aggregate loss
and magnitude also does not establish that the units perform identical functions;
the filler-heavy task's global loss is a particularly limited proxy for general
capability.

## Four training branches and a fixed rescue boundary

Within each seed, start every branch from the same saved **pre-acquisition** model,
optimizer, RNG and sampler state and feed exactly the same complete examples in
the same order. Do not start an acquisition experiment from the V2 checkpoint
that already learned the target facts. A fresh causal world is preferable; if the
V2 world is reused, retain the exploratory label and still restart before exposure
to its facts. Freeze the world choice before the branches start.

| Branch | Updates 1–500 | Updates 501–1,000 |
|---|---|---|
| H: healthy with sham instrumentation | No writes vetoed | No writes vetoed |
| T: target unit blocked | Veto Utarget | Veto Utarget |
| C: comparable control unit blocked | Veto Ucontrol | Veto Ucontrol |
| R: rescue | Veto Utarget | No writes vetoed |

All branches finish at 1,000 optimizer updates with equal data and horizon. The
rescue does not receive extra examples, updates, LR tuning, donor weights or oracle
information. In zero-based code, R vetoes steps 0–499 and reactivates before step
500, the 501st optimizer update. R and T must have identical numerical states and
consumed-data state at the 500-update boundary before their schedules diverge.

The sham executes the same scheduling/instrumentation path with an empty veto.
It must be numerically bit-identical to the unmodified healthy route in the
declared qualification: model, optimizer, data, RNG and logits. Compare numerical
states while retaining distinct run IDs and provenance. If sham changes the
trajectory, the control is not valid and must be repaired before interpretation.
Budget any duplicate healthy/sham qualification explicitly rather than treating
it as another independent research seed.

A complete experimental schedule belongs in the resolved contract before training.
If implementation forks from a shared checkpoint with a different future schedule,
that requires the explicit experimental-fork contract, including parent identities
and allowed differences. Do not bypass strict-resume checks to obtain a new
intervention or horizon silently.

## What a unit veto means numerically

Preserve proposal calculation, policy validation and allocation. At the selected
write call, convert only would-be admitted writes into explicitly recorded policy
denials; keep original overflow and other abort outcomes distinct. Do not promote
losing candidates after denying the winner. Every candidate and every one of the
four unit frames remains accounted for. Record the schedule, unit, step and
pre-/post-veto counts and commitments so intended denial cannot be confused with
a dropped frame or an arbitrary rewrite.

The blocked call contributes no admitted channel update. **Preserve its ordinary
quantization path.** In the current FP32 host path, zero commits still quantize and
promote the incoming residual through bf16; returning the original FP32 `h` directly
would additionally remove a rounding operation. The blocked forward must match
the declared zero-commit reference path, not an invented FP32 identity shortcut.

Under the declared STE contract, the residual VJP remains the identity, while the
blocked call's direct proposal and codebook VJP contributions are zero. Verify
this locally with isolated-call gradient checks, including finite edge cases.
The aggregate shared-codebook gradient may remain nonzero from the other three
units. AdamW moments and weight decay may move parameters even with zero current
gradient; a shared C can change through other units. Thus neither unchanged weights
nor zero total C gradient is the definition of successful blocking. Freezing C
alone does not implement this experiment.

Keep auxiliary objectives and optimizer behavior fixed and account explicitly for
any other gradient route. Prefer an immutable per-call unit policy over mutation
of a process-global feature-denial list. The existing global `deny_features`
capability is not a layer- and phase-scoped training schedule.

## Inference-only checks and interpretation of rescue

The primary final comparison evaluates all trained parameter states under the same
unblocked inference policy and fixed decoding/position conditions. Separately
evaluate:

- The healthy checkpoint with target-only and control-only inference vetoes.
- T and C both with and without their training veto at inference.
- The declared R checkpoints around reactivation, using a fixed diagnostic schedule
  applied to the other branches as well.

These checks distinguish a route's contribution to access from the consequences
of training without its writes. Inference masking of a healthy model alone cannot
show that facts were never learned. Conversely, reactivating a route that was
blocked during training can inject an unadapted signal; a failure immediately
after reactivation is not automatically evidence of forgetting. A rescue failure
after only 500 active updates does not prove irreversibility or unlimited failure
to learn. Report its learning curve and the finite rescue window.

Training and inference masks must be phase-explicit. A generation token index must
never be interpreted as an optimizer step and accidentally activate a training
schedule. The final primary endpoint is fixed at 1,000 updates; diagnostic curves
do not permit selecting a convenient checkpoint afterward.

## Expected inactivity does not suspend integrity

The manifest identifies exactly which unit is intentionally inactive at each
training step. Keep complete producer inventories, ACK-before-update ordering,
STEP/checkpoint bindings, byte custody and the declared T1 scope. Missing frames,
unacknowledged updates, corruption and unexpected numerical failures are run
failures, not consequences that can be excused as ablation.

Retain finite checks and distinguish pre-existing nonfinite/magnitude aborts from
experimental policy denials. Do not skip computations merely to hide a nonfinite
proposal behind the veto. Activity thresholds can recognize the declared inactive
unit only under an explicit rule frozen before the run; they must continue to
detect unexpected collapse in the remaining active units. Preserve global counts
as well as activity over eligible active units, with both denominators stated.

Qualify the veto's forward/STE behavior, sham equality, the 499/500 boundary,
phase-scoped inference, complete frames and failure paths on CPU and then the
actual hardware. Preserve the exact binaries and schedules. T1 verifies its
declared apply intervals; it does not establish answer truth or replay all
attention, optimizer and backward arithmetic.

## Outcomes, contrasts and falsification

Use a reserved factual evaluation set for the treatment comparisons after
localization. Discovery-localizer effects are selection-biased and are not the
confirmatory effect estimates. Keep the primary parser, all questions and the
declared positions; invalid outputs and missing predictions remain failures.
Report original global NLL, Value NLL, factual accuracy, false assertions,
abstention, invalid outputs, actual exposures, write activity and gradients as
different measurements. A low optimized loss or a nonzero channel is not a
knowledge label.

Primary contrasts at the common horizon are T−H, C−H and the specificity contrast
(T−H)−(C−H), computed on the same held-out facts within a seed. The rescue contrast
is R−T with identical total data and updates. Compare false-assertion risk at a
predeclared comparable-coverage rule; do not gain apparent safety by abstaining
always or generating unparseable text. Report unadjusted accuracy and all output
categories alongside any coverage-controlled metric.

Evidence supporting the **bounded proposed mechanism** would be: an informative
discovery target; a larger factual deficit and increase in false assertions under
T than under the adequately matched control; concurrent improvement of the original
global-loss metric; and a rescue benefit relative to T under the fixed budget,
with inference-only checks limiting an access-only interpretation. This remains
an effect of the declared training intervention on this task and architecture.

Results that weaken or refute the tested version include:

- No positive localization effect or no adequate control match.
- Compensation by other units, leaving T factually competent.
- T and C causing comparable damage, or apparent specificity explained by unequal
  general disruption or intervention magnitude.
- Only formatting failures or abstentions increasing, rather than false assertions.
- Effects reproduced entirely by inference-only access restriction, without evidence
  of impaired acquisition under the common inference policy.
- No benefit in the declared rescue window. This is a negative rescue result for
  that window, not proof that recovery is impossible.

Unexpected invariant, finite or T1 failures invalidate the affected comparison;
they are neither support nor refutation of a semantic hypothesis.

## Replication and conditional execution

Use three **predeclared exploratory training seeds** only if measured resource use
supports them. Report every attempted seed and every healthy-control eligibility
failure. Do not silently replace unsuccessful seeds. If different units are selected
per seed, the estimand is the effect of the declared selection rule, not the effect
of one supposedly invariant semantic feature.

The training run/seed is the primary replication unit. Facts are nested in entities
and repeated across templates/positions; tokens, paraphrases and model forwards are
not independent training replications. Report per-seed paired contrasts and their
range/mean. Three seeds do not turn this exploratory study into a population-level
confirmation or establish how often the mechanism occurs in ordinary transformers.

Before execution, freeze the seed list, healthy configuration, initial-state source,
world and splits, localization rule, control-matching rule/calipers, four schedules,
inference policies, scored checkpoints, outcome definitions and resource/retention
budget. Include localization, sham qualification, inference and custody costs.
No causal model, core change or cloud action is executed by this protocol document.
