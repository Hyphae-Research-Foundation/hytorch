# SPEC_AMEND-004 — POLICY 7: two-phase selection (device proposes, policy disposes)

*2026-09-01. Motivo: port a AWS Trainium2 (fase 5). El systolic array
(TensorE) no ejecuta la reducción secuencial fp32-no-FMA del reference —
ningún acelerador matricial lo hace a throughput útil. La respuesta no es
renunciar a la exactitud: es mover la frontera de qué está verificado.*

## La regla

**Policy 7 (`selection = two_phase_topk`)** divide pack/allocate en:

- **Fase A — PROPUESTA (device, NO verificada):** scores aproximados por el
  hardware que sea (TensorE bf16, matmul XLA, lo que el chip haga rápido).
  Top-M candidatos por token (M declarado en POLICY, default 32 = 4k).
  La propuesta es *hint*, no hecho.
- **Fase B — POLÍTICA (exacta, verificada, replayable):** el reference
  (apply-ref, nuevo entry `pack_allocate_candidates`) recomputa los scores
  de SOLO los M candidatos con la matemática pinned (secuencial fp32
  no-FMA, NaN canónico SPEC_AMEND-002), y ejecuta el top-k + tiebreak +
  colisión + abort rules EXACTOS de policy 6 sobre ese conjunto. Los
  veredictos, mags y el orden canónico salen de la fase B únicamente.

Costo exacto por token: M·d mults (32×20 = 640) vs N_f·d (32768×20 = 655k)
— 1000× menos; cabe en los 12 vCPU del host trn2 sin despeinarse.

## Qué se verifica y qué se declara (honestidad del contrato)

| Propiedad | Estado bajo policy 7 |
|---|---|
| mag del veredicto = score exacto del feature citado | **VERIFICADO** (T1 spill igual que siempre: recompute bit-exacto desde leaf+C) |
| orden canónico, colisiones, aborts, #C+#O+#A=k | **VERIFICADO** (misma máquina de veredictos, mismo reference) |
| apply bit-exacto + replay CPU del residual | **VERIFICADO** (apply no cambia) |
| T2/receipts/chain | **VERIFICADO** (sin cambios) |
| "ningún feature fuera de los M propuestos tenía score mayor" | **NO verificable offline — DECLARADO** como device-proposed en POLICY |

La última fila es la novedad y va declarada en el registro POLICY
(`selection="two_phase_topk", m_candidates=32, proposal="device"`), en el
manifiesto y en el paper. La integridad del ledger es idéntica; la
*completitud de la propuesta* pasa a ser una propiedad medible del device
(miss-rate vs escaneo exacto, medible en CPU y auditable por muestreo),
no un axioma.

## Por qué esto no rompe la tesis (y la mejora)

1. El sistema SIEMPRE fue "el device propone, el ledger dispone". Policy 7
   hace esa frontera explícita también dentro de la selección.
2. Un miss en la propuesta (feature verdadero-top-k fuera del top-M bf16)
   solo ocurre con scores dentro del ruido de redondeo bf16 del corte — el
   sustituto tiene score casi idéntico. Es un cambio de *qué* se escribe
   (como cambiar una seed), jamás de la *veracidad* de lo escrito.
3. El claim cross-vendor se re-enuncia: policies ≤6 = hechos bit-idénticos
   NVIDIA/AMD (medido, sigue en pie); policy 7 = hechos bit-exactamente
   verificables y replayables en cualquier silicio, con propuesta declarada
   (NVIDIA/AMD/**Trainium**). Ambos claims van juntos al paper.
4. Bonus de escala: la fase A es el 99.97% del FLOP y ahora corre a la
   velocidad nativa del acelerador — el catalog deja de pagar el escaneo
   exacto completo en device (la causa del 20× de 3.4).

## Reglas de implementación

- `pack_allocate_candidates(delta_hat, codebook, cands[nt][M], k, mag_max,
  selection_base)` entra al reference object (apply-ref). El build hash
  cambia → build.apply_ref_hash nuevo, declarado en RUN_START como siempre.
- Equivalencia obligatoria (gate): con cands = TODOS los features (M=N_f,
  orden natural), policy 7 ≡ policy 6 BIT_IDENTICAL en toda la batería
  adversarial. Ese es el puente formal entre políticas.
- El `cand` field del VerdictRec registra el rank en la PROPUESTA (0..M-1)
  — auditoría del proposer gratis en cada hecho.
- Miss-rate: herramienta de medición (CPU bf16-sim vs exacto) con umbral
  prerregistrado en el manifiesto de fase 5-trn (<0.5% de tokens con
  cualquier miss en el top-k efectivo; si excede, subir M y re-medir).
