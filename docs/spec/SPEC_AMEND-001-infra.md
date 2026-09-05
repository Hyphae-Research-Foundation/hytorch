# SPEC_AMEND-001 · infra.* fields (2026-08-29)

**Choque journalizado** (per spec: "si choca, gana este documento y el choque se
journaliza como SPEC_AMEND"). La spec v2.2 asume el device como campo binario
(`cuda|rocm`) y no contempla procurement. La ejecución real es 100 % DigitalOcean
vía doctl, con spot preemptible (directiva del usuario, 2026-08-29):

- NVIDIA ladder: `gpu-b300x1-288gb-spot` → `gpu-b300x1-288gb-lc-spot` →
  `gpu-h200x1-141gb` (on-demand) → `gpu-h100x1-80gb` (on-demand).
- AMD ladder: `gpu-mi355x1-288gb-spot` → `gpu-mi350x1-288gb-spot` →
  `gpu-mi300x1-192gb` (on-demand).

## Campos añadidos al manifiesto (aditivo, no cambia el binding)

| Campo | Qué registra |
|---|---|
| `infra.device_slug` | Size slug exacto de DO del droplet del run |
| `infra.region` | Región DO |
| `infra.procurement` | `spot` \| `ondemand` |
| `infra.preemption_count` | Nº de preempciones absorbidas (runs hijo) |
| `infra.driver` | Versión de driver NVIDIA/ROCm capturada en el droplet |

El binding sigue declarando solo `device = cuda|rocm` (§08). El silicio exacto
vive en RUN_START, no por hecho.

## Preempción ↔ §5.4 (mapa, no cambio)

La preempción spot de DO **es** el abort del run de §5.4: el watchdog detecta el
droplet desaparecido, relanza por la escalera, y el harness arranca un **run
hijo** cuyo manifiesto cita `parent_run_id` + último head commiteado + hash del
checkpoint (θ, C, Adam, RNG — resume **evidenciado**, no verificado: Adam/RNG
están declarados fuera de la cadena en fase 1, §07).
