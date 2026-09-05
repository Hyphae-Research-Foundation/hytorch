# python/ — trainer side

`python/hytorch/` is importable from a checkout (tests insert `python/` on `sys.path`);
`pyproject.toml` carries the metadata. Requires `torch numpy`, plus `libapply_ref.so`
built by `cargo build --release` in `../target/release/`.

| module | role |
|---|---|
| `model.py` | Law-0 block and `HyphaeWrite`: the only writer of the residual. Forward = pack → allocate → apply via the pinned policy; backward = closed-form STE mirror over the committed set. `clip_proposal` (in-graph, from the manifest), the host mirror and host codebook-gradient accumulator used on XLA. |
| `applyref.py` | ctypes binding to `libapply_ref.so`, bf16 bit helpers. |
| `devext.py` | Loads the gate-authorized CUDA/HIP kernels as a torch extension (`-fmad=false`); zero-sync layer loop. |
| `neuron_backend.py` | Policy 7 (two-phase) backend for XLA/Trainium: device proposes (two-stage top-M with explicit tie-break), host reference disposes, fixed-shape host apply. |
| `seamclient.py`, `seam.py` | Trainer-side seam protocol v2 (RUN_START, per-step barrier → RECEIPT, STEP chain) and the audited-spill writer mirrored from `crates/verifier`. |
| `run.py` | Manifest runner: the citable single-GPU harness (phases 1–2). |
| `data.py` | wikitext-103 → token bins with sha256 manifest. |
| `probe.py`, `ledger_analysis.py` | Semantic probe and causal ablation from the journal; wire-dump analysis. |
| `runtime.py`, `signatures.py`, `knowledge_boundary.py`, `hallu.py` | Journaled inference (facts + `LOW_CONFIDENCE` before emission), confidence signatures F1–F9, the knowledge-boundary task builder and the phase-4 driver. Designed and tested; the study itself is not run (see README §What is not done). |

## Tests (`python/tests/`)

| suite | what it proves |
|---|---|
| `test_spill_roundtrip.py` | Python spill writer ≡ Rust reader, byte for byte |
| `test_e2e_seam.py` | trainer ↔ `hytorch-seam` ↔ Hyphae: receipts, chain, guard |
| `test_inject.py` | 36 mutations over real spills, each caught with its class |
| `test_runtime.py` | journaled generation: citable tokens, dead-seam degradation, verifier |
| `test_two_phase.py` | the policy-7 gate on CPU (G1–G7, see module docstring) |
| `test_two_phase_device.py` | the same gate on an accelerator (CUDA / XLA) |
| `probe_semantics.py` | semantic-probe smoke |

Run: `cd python && ../.venv/bin/python -m pytest tests/<suite>.py` (needs `cargo` on PATH for the verifier parts).
