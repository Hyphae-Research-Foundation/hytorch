# SPEC_AMEND-002 · Canonicalización de NaN en el score de pack (2026-08-29)

**Origen:** el gate diferencial M3 en H100 (`results/gates/NVIDIA-H100-80GB-HBM3/`)
rechazó el backend: `pack_mismatch=53869/200000`, `apply_mismatch=0`.

**Causa raíz:** la spec §03 ordena los scores por el total order de IEEE-754
sobre los bits de fp32, y declara que un NaN «se selecciona y luego aborta como
nonfinite». Pero los bits de un NaN no son canónicos entre plataformas: x86
produce `-qNaN` (0xFFC00000) para operaciones inválidas (`inf×0`) mientras CUDA
produce `+qNaN` (0x7FC00000). Bajo la clave de orden total, `-qNaN` ordena
DEBAJO de todo y `+qNaN` ENCIMA de todo: el mismo `delta_hat` seleccionaba
features distintas en host y en device. El determinismo prometido por la spec
era falso tal como estaba escrita.

**Enmienda (gana sobre §03):** antes de ordenar y de derivar `mag`, todo score
NaN se canonicaliza a `+qNaN` = `0x7FC00000`. Consecuencias:

- Un score NaN ordena encima de +inf en ambos lados → se selecciona → su `mag`
  bf16 es el qNaN canónico `0x7FC0` → ABORT `nonfinite` journalizado. La
  semántica declarada por la spec («la locura aflora como hecho») se conserva;
  ahora además es determinista cross-plataforma.
- El desempate entre múltiples NaN es por `feature` ascendente (todas las
  claves NaN son iguales tras canonicalizar), consistente con el tie-break
  general.

**Implementación:** `crates/apply-ref/src/pack.rs` (referencia) y
`kernels/cuda/refkernels.cu` (device), mismo predicado, mismos bits.

**Estado del gate tras la enmienda:** ver `results/gates/*/gate-summary.json`.
