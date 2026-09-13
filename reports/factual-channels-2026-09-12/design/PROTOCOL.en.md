# V6 Protocol: Persistent Blocking and Distributed Blocking

**English edition: September 13, 2026. Translation of the unchanged [Spanish source](PROTOCOLO.md), SHA256 `6bae2857c9bf6910aa728be2cd65da5f510400190f2e45a1d909194bd48ea195`.** This translation preserves the same design, scope, numerical parameters, and conditions. It creates no new protocol version, result, qualification, or execution authorization. The Spanish source remains unchanged and retains its original hash; this English translation has a separate file hash.

[English](PROTOCOL.en.md) · [Español](PROTOCOLO.md) · [English report](../README.en.md) · [Informe en español](../README.md)

Status: a prospective design for an exploratory experiment, prepared after the V5 `no-control-match` result. V6 identifies the experiment version; it does not assert that a V6 curriculum has been implemented. This document is neither an execution nor a qualification of the runner.

## Question and change of estimand

The question is whether concentrating a persistent write block in one unit harms factual retrieval and increases false assertions, while aggregate loss still appears favorable, compared with distributing interruptions across units. Restoration is included to study the effect of removing both regimes within the same training budget.

The V5 result remains `no-control-match`, with null final target/control IDs. U0 was the localizer candidate; no other site passed every caliper. Those limits are not relaxed, U0 is not replaced by another target, and no mixture of controls is computed to obtain a positive match. U1 is not treated as a dead channel either.

The new control is valid for a **contrast of persistent/concentrated blocking policies against distributed blocking**, with balanced marginal site assignments. This changes the estimand relative to the earlier comparison between a selected unit and another unit with comparable general damage. It does not promise equal general damage during training or identify a direct effect at constant loss. An exploratory unmatched comparison of U0 and U1 does not substitute for this panel.

## Complete panel

There are 17 separately executed runs starting from the same A0 bytes within one world/seed. Each performs 4,000 acquisition updates and receives the same examples in the same order. H is declared as an experimental arm from the outset; an old reference is not relabeled H.

| Family | Arms | Updates `[0,2000)` | Updates `[2000,4000)` |
| --- | ---: | --- | --- |
| H | 1 | Empty sham | Empty sham |
| P_u, u=0..3 | 4 | Permanent veto of Uu | Permanent veto of Uu |
| D_s, s=0..3 | 4 | Distributed veto defined below | The same distributed schedule |
| RP_u, u=0..3 | 4 | The same schedule as P_u | Empty sham |
| RD_s, s=0..3 | 4 | The same schedule as D_s | Empty sham |

The first reactivated update has index 2,000 and produces state A2001. P_u/RP_u and D_s/RD_s must match exactly in numerical states and consumed data through A2000, inclusive. Their distinct contracts and identities are preserved. P/D equality is not required after their policies diverge. A P or D run may not be converted into rescue through strict resume, and a prefix may not be shared without a new fork contract; this proposal uses separate complete executions.

The same site is vetoed in every forward and microbatch within an update. Generation and diagnostics have separate policies: a token index never selects the training site.

## Distributed schedule and balance guarantee

For t in `[0,4000)`, let b=`t//4` and k=`t%4`. Let pi_b be a permutation of U0..U3 constructed by sorting SHA256 hashes of a canonical JSON encoding with an explicit domain, schedule seed, index b, and unit index; ties are resolved by unit index. The exact encoding and domain belong to the contract of `control_schedule.py` and the complete generated plan. The model, sampler, dropout, and data-generator RNGs are not used.

Arm D_s vetoes `pi_b[(k+s)%4]`. RP/RD replace their mask with the sham from t=2000 onward. The permutations and complete schedules are fixed before generating or evaluating the new world; favorable schedule seeds are not sought.

At each t, the four P arms veto each site exactly once, as do the four D arms. Within each block of four updates, each D arm visits each site once. Each P arm accumulates 4,000 blocks at one site; each D arm accumulates 1,000 per site. During the first half, each RD arm accumulates 500 per site; each RP arm accumulates 2,000 at its site. This difference in concentration within a model is part of the intervention. The claim is not that only autocorrelation changes while each model's per-site dose remains fixed.

For any function f(u,x,t) evaluated at a common healthy state and the same input, the mean of f over the sites assigned to the four P arms equals the mean over the four D arms exactly at each t. The identity holds by permutation, including by position and text type. It does not depend on interpolating observed NLLs or assuming additive effects across sites.

This proof is limited to assignment. Once parameters, AdamW, the shared codebook, and representations diverge, effective COMMITs, their energy, gradients, compensation, and NLL may differ. Those differences are retained as outcomes and possible mechanisms; masks are not adjusted, arms are not reweighted, and runs are not excluded to equalize them.

The fixed cycle `(t+s)%4` is avoided: the current curriculum divides an auxiliary presentation into four slices through `t%4`, which would couple a site to the same slice within each arm. Per-block permutations avoid that permanent coupling; they do not prove the absence of every temporal interaction. The specific policy is the object of the experiment.

## New world, exposure, and reserved sets

The entire historical discovery world has already been observed. Although the V5 measurement did not open validation/test, validation was scored in V2; it is not presented as a globally untouched reserved set. Test has a declared reservation without a universal access audit. Neither is reused as new confirmation.

A single new exploratory realization is proposed; it has not yet been generated:

| Identity | Fixed seed |
| --- | ---: |
| World truth | 2026091301 |
| Exposure/split assignment | 2026091302 |
| Model, optimizer, and stream | 2026091303 |
| Blocking schedule | 2026091304 |
| Execution order after H | 2026091305 |

The generator retains 96 entities, two relations, 16 values, common_repeats=8, rare_repeats=1, low/high mentions=16/64, and unseen_entity_every=8. The seeds are prospective choices, not seeds found through search or a sample sufficient for population inference. Sharing vocabulary and templates with the historical benchmark limits novelty: the proposal is a new assignment of facts/exposure, not a new architecture or a natural-language corpus.

The new complete benchmark identity must differ from all known historical identities before training. The first generated world and every failure are retained; the world is not replaced for lack of competence or an unfavorable effect. The export separates public TRAIN, discovery/validation/test prompts, and the private oracle. Splits are assigned by entity; labels, reserved answers, and scores do not reach the model or schedule. Exposed facts belonging to validation entities may occur in TRAIN: this evaluates retrieval through reserved prompts, not generalization to facts never exposed during training.

A new W640/A0 is required under the final source: auxiliary warmup and a state preceding acquisition of the new world. B48/T128, the original32 + auxiliary8 + canonical8 domains, vocabulary size 428, the objective recipe, and V5 geometry are retained. The canonical bank is derived from all relevant public records of the new world. Its size and the cardinality of each stratum are checked during export; neither the historical size of 503 nor the historical count of 19 common facts is imposed.

The current V5 code fixes seed=1337 and other historical identities. Adapting it requires new source and qualification; editing those identities in a manifest does not turn the previous code into a V6 runner.

## Intervention semantics

The complete write at the assigned site is vetoed after proposals, validation, and allocation. Only surviving COMMITs are converted into experimental denials. Normal zero-commit quantization, the declared residual VJP, every frame, and original aborts/overflows are preserved; losing candidates are not promoted, and an invented FP32 identity is not returned.

Parameters are not frozen. The existing semantics block that call's direct VJP to proposals/codebook, but regularization, other calls to the shared codebook, AdamW moments, and weight decay may move parameters. Write blocking does not isolate a pure plasticity intervention or guarantee the absence of knowledge elsewhere in the model.

For each unit/update, retain total opportunities, COMMITs before the veto, experimental denials, original overflows and aborts, valid/padding positions, erased contribution, and denominators. Schedule balance is not confused with equality of effectively erased writes. The actual DATA sequence is authenticated and compared across all 17 arms.

## Measurements and estimands

H is trained first under its final contract. The discovery competence gate retains ≥80% common-fact macro accuracy and strict superiority over the four declared priors in each map. If it fails, the attempt ends as a competence failure: the world, objectives, horizon, and seeds are not changed to obtain approval. This gate does not use validation. If it passes, the other 16 arms execute in the fixed hash order, without intermediate selection based on results.

Primary evaluation uses every validation prompt at A4000, an unblocked policy, fixed parser and greedy decoding, a maximum of 16 tokens, and the three maps identity/shift8/gaps4. Every stratum is reported: common, rare, and unexposed. The primary stratum comprises all common validation facts without filtering on H correctness. Its membership and actual exposure are checked before interpretation. Test remains closed.

Primary capture and scoring begin only after all 17 arms have closed and been verified. First, all their generations and full 428-class logits at the public answer boundary are captured and sealed, using a fixed scaffold and without supplying the correct Value. Only then does the private scorer open validation targets, index the sealed logits, and classify the responses. No validation result feeds back into the schedule, training, H gate, or arm selection. Capture failures are not hidden by decoding again.

The new export must physically separate private labels by split. H and the discovery curves may access only the private discovery package; public capture does not open targets. The historical `load_world` loads a combined oracle and does not satisfy this separation merely by subsequently restricting a `split` argument: a new qualified reader is required. Opaque custody hashes are distinguished from semantic access to labels. The reservation is operational; public seeds for a deterministic generator do not constitute a cryptographic barrier.

For each arm, stratum, and map, retain accuracy by fact, false-assertion rate, abstentions, unevaluable outputs, and Value NLL. These factual metrics average paraphrases within each fact and then give each fact equal weight. The three maps have equal weight and are also reported separately.

Original NLL is a separate global outcome without stratification by fact: the sum of losses over every valid/EOS token in every original TRAIN occurrence belonging to validation entities, divided by the total count of those tokens within each map; the three maps are then averaged equally. Multiplicities are retained, and this reduction is not replaced by an average over records, batches, or facts. This is a task metric on known text, not universal language competence or a new language test. Each factual component and this global NLL receive the following contrasts with their own denominators.

For each component of the vector, with means taken over the four arms of a family:

```
tau   = mean(P) - mean(D)
rho_P = mean(RP) - mean(P)
rho_D = mean(RD) - mean(D)
kappa = rho_P - rho_D
      = (mean(RP) - mean(RD)) - (mean(P) - mean(D))
```

H provides context for competence and deterioration; all 16 individual differences from H are also reported. H cancels algebraically in tau and kappa. For error rates, a positive tau means more error under persistent blocking; a negative rho means improvement after removing the block. Signs are not silently reversed, and effects from different sites are not summed as though they were additive.

The world/seed is the replication unit. The 17 arms form a paired policy panel, not 17 independent seeds; facts, tokens, maps, and shifts are not training replications either. The complete exploratory realization is reported, without p-values or population intervals manufactured from those denominators. Expansion requires a list of worlds/seeds and its own resources fixed before execution.

The curves use fixed checkpoints A0, A1000, A2000, A2001, A3000, and A4000 in every arm, with no veto during readout. Their fixed factual population contains all common discovery facts from the new world, every discovery paraphrase, and all three maps, without filtering on correctness; generations and logits for Value NLL are captured. The fixed text-NLL population contains every original TRAIN occurrence belonging to discovery entities, using the same global token/EOS reduction defined above. Both inventories are sealed before fact acquisition and remain complete at every checkpoint. Validation is not introduced into these curves.

Separately, retain at every update the optimized objective on the actual stream, `Loriginal + 0.25 Laux + 0.25 Lcan + 0.001 Pglobal`, its V5 numerical grouping, and every component. Descriptively compare the mean of the first 100 updates with the mean of the last 100, recording that those windows may contain different examples. These online losses use the arm's training policy; the fixed curves above use unblocked inference. Those policies are not interchanged during interpretation, and a checkpoint is not chosen by loss or score.

At A4000, add unblocked inference and each of the four singleton inference vetoes for all 17 arms, only on discovery, using Value NLL at the public boundary without including the true Value in the input. This is an access diagnostic, separate from primary evaluation; it does not select a unit or recalibrate dose. Its cost is included before execution. Readouts preserve model state, RNG, mode, and gradients.

## Prospective interpretation and failures

The complete contrasts are retained regardless of sign. To summarize whether loss hides deterioration in this realization, the descriptive margin for original NLL is fixed at 0.02 nats per token, approximately a 2% relative change in perplexity. Report whether, in each map, `abs(mean(P)-mean(D)) <= 0.02` and neither mean exceeds H by more than 0.02, alongside differences in false assertions/accuracy and the evolution of the optimized objective. Each arm is also reported so that averages do not hide heterogeneity.

This margin is an exploratory design choice, not a power calculation or a statistical test of functional equivalence. It is an outcome-interpretation criterion, **not a gate for admitting, matching, filtering, repeating, or adjusting arms after training**. If it is not met, the total policy effect can still be estimated, but it is not summarized as factual damage with loss comparable under that criterion. Alternative margins are not searched to obtain a favorable result.

Unconditional rates include every question. Risk conditional on answering is compared only when coverage is exactly equal across the involved arms in each map and the denominator is nonzero; decoder, thresholds, or subsets are not adjusted to equalize it. Unevaluable outputs remain separate. A prediction missing because of an execution failure is not converted into an abstention or false assertion.

A deficit under unblocked inference does not by itself demonstrate that a fact was never acquired: reactivating a poorly adapted unit may introduce noise, and the rest of the model may compensate. The access diagnostic limits interpretations without guaranteeing separation of all these mechanisms. Rescue measures removal of the veto with 2,000 updates remaining; it may combine restored access and additional learning. A negative rescue result does not prove irreversibility.

Missing artifacts, corruption, sham divergence, different data, incorrect boundaries, nonfinite proposals, or verification failures invalidate the affected contrast. The panel is not salvaged by averaging three of four arms. The failure is retained, and there is no automatic retry. The guard considers the units active in the update under rules fixed before execution; its streak is not reset when switching units or restoring writes.

## Qualification required for execution

The code prepared at this stage tests schedules and the contract without NN, optimizer, Native, oracle, or cloud operations. It is not a scientific runner.

Before execution, the following are required:

1. New pinned source supporting the new seeds, world, and bank while preserving the intended bytes/semantics. Authenticated public/private export, exposure, vocabulary, and geometry.
2. W640/A0 initialization and shared sampler, optimizer, and RNG qualified under that source. Sham H must match the healthy route; the contracts remain distinct.
3. The new P/D/RP/RD schedule family in the kernel, guard, client, Native registration/reader, and replay. The V5 state machine recognizes H/T/C/R with two halves; D is not relabeled C, and contracts are not mutated per update.
4. Tests of the same actual execution path: update-level site changes, blocks of four, R1999/2000, ACK before parameter mutation, post-STEP, every frame, and rejection paths. Qualification of backward/clipping/AdamW and paired prefixes; T1 retains its declared scope.
5. A measured budget for TRAIN, checkpoints, diagnostics, inference, retention, verification, cleanup, and free storage. START/terminal fix the effective deadline; an authorization JSON does not start processes.

These are future execution requirements, distinct from completing the present control design. Until they are met: `runtime_qualified=false`, `empirical_control_validated=false`, `scientific_admission=false`, `training_authorized=false`.

## Resources and deliverables for this step

The extrapolation from eight verified V5 CPU terminal records is 11,925.299 seconds per arm for acquisition, verification, three maps, and postrun. For 17 arms, this is **56 h 18 min 50 s**; the new W640/initialization adds approximately 14 min 30 s. H is already included: no separate reference is added for localization or unit selection.

This is neither a sufficient cap nor a benchmark of the new runner. It excludes adaptation/qualification, new snapshots, curves at six checkpoints, access diagnostics, differences in the new world, retention, and contingency. Retaining the complete pre/post snapshots in the V5 example would cost 23.4375 GiB per arm, **398.4375 GiB for 17**, for those snapshots alone; other captures, weights, ledgers, copies, and filesystem overhead are additional. That retention policy is not proposed without measuring its cost.

No savings are assumed from parallel execution or shared prefixes. GPU/AMD requires its own qualification and measured estimate; the old 90-minute authorization is closed and does not fund these runs.

This step delivers the protocol, a machine-readable contract, complete schedules and their mechanical audit, evidence bindings, and an independent review. It establishes the construction of the new control and makes its limits explicit; it presents neither training results nor an empirically qualified control.
