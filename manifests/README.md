# manifests/ — preregistration as code

Each manifest fixes, before step 0: the model and catalog policy, the data (sha256 of the
token bins), the comparison (topology-matched dense twin), the thresholds that kill the
branch, and the `build` facts. The seam records the manifest's sha256 in `RUN_START` and
reads the guard thresholds from it; a threshold that is not implemented as a guard in the
same commit is not allowed to exist.

- `phase1*.json`, `sweep-*.json` — single-GPU phases and the LR/seed sweep
- `phase3-nanochat-template.json`, `phase5-nanochat-d20.json` — nanochat integration and the d20 run
- `phase4a-signature.json` — the hallucination-signature study (designed, not run)
- `trn2/manifest-d10.json`, `trn2/manifest-d10-w200.json` — the Trainium2 d10 manifests; the second differs only in the guard warm-up (100 → 200 steps) and records its own declaration time and reason
- `rehearsal-cpu.json` — CI rehearsal
