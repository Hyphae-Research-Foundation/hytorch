These portable data support the figures in the technical report. They preserve measured V5 evidence and the prospective V6 design; they do not present a confirmed causal result.

[English data guide](README.en.md) · [Guía de datos en español](README.md) · [English report](../README.en.md) · [Informe en español](../README.md)

From the repository root, using only the Python standard library:

```bash
python3 -B reports/factual-channels-2026-09-12/data/verify_selection.py
python3 -B reports/factual-channels-2026-09-12/data/export_data.py --check
```

The first command reconstructs the complete selector record from the committed reduced tables and requires the same `no-control-match`. It checks the hashes of the tables, result, rule, and exact selector copy before executing those verified bytes. It needs no `results/`, `.worktrees/`, Torch, checkpoint, oracle, or Native installation.

The second command verifies the curated dataset and every original source that is available. If all originals are present, it reconstructs the export from them and compares the bytes. If some are missing, it reports that limitation while retaining the portable selector check and its links to the tables used by the figures. The figures need only `report-data.json`. The provenance manifest identifies 42 original sources; the English and Spanish editions share these same data and sources.

To regenerate the extraction when **all** pinned originals are available:

```bash
python3 -B reports/factual-channels-2026-09-12/data/export_data.py --refresh
```

| File | Contents and scope |
| --- | --- |
| `report-data.json` | Small plotting dataset: competence by map/stratum with denominators; two batteries, four units, effects by fact/position; matching, repetitions, counts, timings, and prospective design. |
| `provenance.json` | SHA256 for each original and each export, repository-relative paths, source fields, and exact methodology. IDs E01–E06 refer to original sources, not new experiments. |
| `selection-tables.json.gz` | Complete `assembly.tables`, without text or tokens. Canonical JSON compressed with gzip, an empty filename, and `mtime=0`. It preserves NLL by coordinate and already reduced magnitude sums/counts. |
| `selection-result.json` | The original inner record `selection.result.selection`, including its hash and complete result, without the runtime provenance envelope. |
| `causal_selection.py` | An exact copy of the frozen pure selector; standard library only, with no model APIs. |
| `FACTUAL-CAUSAL-ADMISSION-V1.md` and `FACTUAL-CAUSAL-PILOT-PROTOCOL.md` | Exact copies of the historical rule documents, already in English. Their proposals do not provide current execution authorization or describe the new V6 schedule. |

The plotting interface has `schema_version=1`:

- `reference.competence`: rows `{position,stratum,facts,queries,correct,false_assertions,abstentions,invalid,missing,accuracy,strict_accuracy,false_assertion_rate,coverage}`. Accuracy is macro-averaged by fact; rates are fractions, not percentages.
- `reference.priors`: the four common-fact priors by position, with denominators.
- `measurement.batteries`: two `{battery,units}` objects. Each unit retains `delta_value_nats,g,m,r,rho,p`, `fact_effects`, `positions`, and `totals`. The positions include the same metrics derived from their closed sums and counts.
- `measurement.matching`, `rankings`, `rule`, `repeat_checks`, `cardinality`, `population`, `timing`, `custody`: the complete phase result and its limits. Custody includes bytes read by the closed verifier, file/directory counts, the metadata recheck result, and supervisor closure; it does not describe the size of the complete archive or a new raw replay.
- `measurement.geometry`: a summary of equivalence between sources at the same B16 geometry, the B1/B16 diagnostic by map, and the closed decision-preservation gate. Logits can differ across geometries even when the argmax is unchanged on the examined archived prefixes.
- `design_v6.calendar_examples`: ranges `[0,16)` and `[1992,2008)` from the 17 fixed schedules. Indices are zero-based and `null` means sham. `arm_summaries` retains total and half-horizon opportunity counts, without conflating them with observed erased writes.

Reproducing the reductions and selector does not revalidate raw captures or rerun the model. The package includes no checkpoints, raw tensors, answers, tokens, oracle, ledgers, absolute local paths, or infrastructure addresses. The two batteries repeat the same cases and model; V6 still has no generated world, qualified runtime, or empirically validated control.
