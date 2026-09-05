# External reviews of v1.0 (September 2026) and what changed in v1.1

Four independent readers reviewed `main.pdf` v1.0 (8 pp) as if for a systems / ML venue.
This note records their consensus, the points we acted on, and the points we could not
act on without compute. No experiment was added between v1.0 and v1.1; every change is
framing, ordering or scoping.

## Consensus across the four reviews

**Kept as strengths.** The preregistered protocol reversing our own −44.9 % headline; the
cross-vendor bit-exactness (200k cases, 36/36 injection) and the Trainium2 port that moved
the verification boundary without weakening it; the silent-channel-collapse finding (C4)
as the result that matters; the evidence table; the honesty about tamper-evident ≠
unforgeable.

**Criticised.**
1. The abstract and title read as "we supplied the missing medium"; the evidence supports
   "we built an instrument that measured this medium does not suffice". Volume too high.
2. `CORE 0.015` is the most-cited number and comes from a run with a later-found STE
   gradient bug; this was disclosed in §11, far from Table 1. Readers who reach it late
   re-read §7 with suspicion.
3. "The first training loop…" is unprovable priority language.
4. "Unfalsifiable unless the writes are witnessed" is too absolute: patching, ablation and
   causal tracing establish mechanism without a write log.
5. Observability, auditability, replayability, interpretability and causality are mixed.
6. "Facts" does semantic work the object (a binding + verdict) does not earn; "the model
   learned to route around its channel" is interpretation.
7. Generality of C4 is open: nanochat's bypass scalars are architecture, not nature.
8. Missing controls for any capacity claim: k sweep, matched dense bottleneck,
   catalog-as-side-channel. One policy point, chosen to be verifiable not sufficient.
9. 3.4×10⁸ "receipted facts/step" reads as a full transcript; the raw wire is sampled.
10. The lost dense-twin validation log is an incident for a paper built on custody.
11. §9 anticipates that the instrument measures confabulation; it does not.
12. EU AI Act phrased as a requirement; too strong.
13. Related work lacks residual-stream instrumentation (TransformerLens hooks, profilers /
    flight recorders); the differential should be "the write is the transaction and the
    step is illegal without it".
14. The semantic probe is in the evidence table and absent from the body.
15. Theatrical bold headings; dense jargon.
16. Too many ideas for one paper; suggested spine: C1 transactional stream · C2
    deterministic cross-silicon semantics · C3 mechanical observability · C4 silent
    channel death · C5 what it costs. Protect "witnessed training process".

## What v1.1 changed (all in `main.tex`, commit 0048dcb and follow-ups)

- Abstract rewritten around the instrument-not-architecture thesis; "first" removed;
  the contaminated headline number and the missing rerun stated in the abstract; closes
  on the witnessed-process claim scoped to the typed channel (Law 0).
- Introduction: priority claim reduced to "to our knowledge, prior systems do not combine
  … at this granularity"; a paragraph defining observability / auditability /
  replayability and disclaiming interpretability / causality beyond one probe and one
  intervention; contributions reordered C1–C5 with the negative-space accounting promoted.
- §Seam: what is durable per step vs what is sampled, stated where the record counts are.
- §Collapse: consequences rewritten as prose (no bold labels); "unfalsifiable" → "cannot be
  established from observational activation evidence alone", with interventional methods
  acknowledged; a paragraph on what the finding is not (not a discovery of collapse, not yet
  a law; Law 0 vs VQ open); interpretive sentences marked as interpretation.
- §What it costs (was "The price…"): opens with the STE bug and the lost twin log as
  incidents with cause and fix; Table 1 captioned as the contaminated run; the capacity
  conclusion rests on degrees of freedom, the loss plateau and the bypass trunk, scoped to
  "the configuration tested"; the missing controls are named; a full-width cost table
  separates model compute, selection work, seam, host round trips, retention and wall.
- §Journaled inference retitled "(instrument only)"; nothing in the paper measures
  confabulation.
- Related work: residual-stream instrumentation added with the sharpened differential;
  the semantic probe and its journalized ablation (lift 90–178, purity ≤ 99.8 %, deny one
  feature → +0.0032 NLL on its token, −0.0003 elsewhere, specificity 10.7×) given as the
  one interpretability data point, with its size stated; EU AI Act softened to "may
  provide machine-checkable evidence that could complement".
- Limitations itemised with names: corrected rerun, one policy point, generality,
  throughput, retention, integrity-not-non-repudiation, planned confabulation evaluation,
  the SFT that exposed the bug.
- Fig. 6 caption: "learned form, not facts" removed.
- Length 8 → 10 pp (cost table, limitations, definitions).

## Not changed — requires compute we did not spend

- d20 rerun with the in-graph clip (the experiment that would confirm or change Table 1).
- Capacity curve k = 8 → 16 → 32 → 64, or fewer/wider slots, or catalog as side-channel.
- Dense bottleneck of matched degrees of freedom.
- The confabulation study (needs a cataloged model that knows something).
- Two-pass forward for XLA (the 29×).

## Prepared answers

- *"You didn't discover collapse; you built an instrument that made a known collapse
  observable."* — Correct. The novelty is distinguishing, during training and from
  write-level evidence, a dead internal pathway from an expensive functioning one.
- *"Is this a practical training system?"* — No. It is an instrumentation and
  verification framework; the paper measures what that costs as configured.
- *"Is the receipt a proof of training?"* — No. It is tamper-evident provenance
  (integrity), not non-repudiation.
