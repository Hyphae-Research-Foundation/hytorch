# FASE 5 HALLAZGO MAYOR: colapso total y silencioso del canal (2026-09-01)

**El ledger diagnosticó en 30 minutos lo que ninguna curva de loss puede
decir: el canal tipado no "cuesta capacidad" — llevaba MUERTO desde el
paso ~150 en todos los runs a escala, y el modelo aprende rodeándolo.**

## La evidencia (wire retenido de los pasos 100/200/300/500, run
nano-d20-catalog, fase 5 launch 7, 335M hechos/paso)

| step | commit | overflow | abort(reason=2) | codebook grad | c_prev vs c_next |
|---|---|---|---|---|---|
| 100 | 0.066% | 83.1% | 16.8% | 8.3e-6 | distintos |
| 200 | **0** | 83.4% | 16.6% | 0 | distintos (renorm) |
| 300 | **0** | 83.4% | 16.6% | 0 | — |
| 500 | **0** | 83.3% | 16.7% | 0 | **IDÉNTICOS (congelado)** |

- Step 100, mags de los commits supervivientes: p10=51.8, p50=58.2,
  p90=63.0, 6.8% en el cap — **la distribución entera pegada a mag_max=64**.
- Steps 200+: las propuestas piden p50≈1.2-1.7e5, p99≈1.9e6, max 3.7e6 —
  2000-58000× el cap. Nada puede committear.
- Con 0 commits: h'=h en las 40 unidades de escritura, el mirror STE no
  propaga gradiente (solo fluye por commits), attn/mlp/codebook quedan sin
  señal. El codebook queda bit-idéntico paso a paso.
- El loss BAJA igual (8.72→5.44; val bpb 1.666): nanochat tiene bypass
  estructural (x0_lambdas, resid_lambdas, value embeddings) — **el modelo
  aprende a ignorar su propio residual stream**.

## Mecánica del colapso

El bridge pasaba los outputs CRUDOS de attn/mlp como propuesta. A d20
(d_model=1280) la escala natural por slot en init ya es ~55 — al borde del
cap 64 heredado del toy. El warmup del LR empuja la distribución a través
del cap entre los pasos 100-150 y el canal entra en muerte total: overflow/
abort → sin gradiente → sin corrección → las propuestas crecen sin freno
(el sub-módulo que las produce solo recibe señal del bypass) → muerte
permanente. Un feedback loop de un solo sentido.

## Qué reinterpreta esto (honestidad retroactiva)

1. **3.4 (+2.63 nats a d20)**: no midió "costo del canal tipado"; midió un
   modelo bypass. El c_prev==c_next del step 1999 que anotamos como
   "convergencia del codebook" era rigor mortis. (El grad_norm=0 de esos
   STEP records era cosmético — el shim viejo pasaba 0.0 — así que no
   distinguía; AHORA el shim registra el norm real, y por eso step 100
   muestra 8.3e-6 y después 0 exacto.)
2. **1b/lrsweep "catalog 6× más robusto a LR"**: un modelo bypass es
   insensible al LR del canal por construcción.
3. **d12 3.2 (+1.18 nats)**: presumiblemente el mismo colapso (misma
   política, escala natural ~sqrt(768/64)·algo — a verificar contra su
   ledger si el wire retenido lo permite).
4. **El toy (fases 1-2) NO colapsó**: allí los deltas cabían bajo 64 y el
   canal mostró commits, semántica de features (f20338), ablación causal.
   La ciencia del toy sigue en pie.
5. **Nunca hemos medido un catálogo VIVO a escala nanochat.** Fase 5
   re-lanzada con el fix será la primera vez.

## Por qué esto es LA demo de C1/C2

Una curva de loss no puede distinguir "canal caro" de "canal muerto con
bypass". Los recibos lo hicieron con precisión mecánica: qué verdicto, qué
razón, qué magnitud pedía cada escritura, en qué paso murió el último
commit, y que el codebook no se movió un bit desde entonces — todo
auditable offline, del run real, sin re-ejecutar nada. **Hyphae hizo su
trabajo: lo malo quedó documentado.**

## El fix (implementado en este commit)

1. **Clip de norma por slot en el bridge** (`hytorch_bridge.py`): la
   propuesta se proyecta al presupuesto de la política ANTES del pack:
   `delta_slot *= min(1, clip / ||delta_slot||)` con clip = mag_max/2 = 32.
   Dirección preservada, magnitud acotada, mag_max=64 queda como backstop
   de abort (nonfinite/pathological). Declarado en manifiesto como
   `model.proposal_clip`. El factor de clip se computa sobre norma
   detached (estilo grad-clipping); el gradiente fluye por la dirección.
2. **Telemetría que grita** (`seam_shim.py`): rank 0 imprime
   `hytorch: step N commit_rate=X% overflow=Y% abort=Z%` cada 50 pasos y
   un WARNING rojo si commit_rate < 1% — el colapso nunca más será
   silencioso en el log del trainer (además de estar en el ledger).
3. **Evidencia custodiada**: verdict mixes + STEP records de este run en
   este directorio; el ledger completo del run muerto queda en el volumen
   hasta el RESET_ARMS del relanzamiento.

## Predicción prerregistrada para el relanzamiento (antes de correr)

Con clip=32: commit_rate ≥ 30% sostenido tras warmup (los k=8 ganadores
por slot caben bajo el cap por construcción; overflow queda para
contención real de slots), codebook grad > 0 durante todo el run, y el
gap vs vanilla se REDUCE respecto a los +2.63 nats del canal muerto. Si el
gap NO mejora con el canal demostrablemente vivo, el costo de capacidad es
real y el paper lo reporta con el canal vivo como calificador — ese
negativo también es nuevo conocimiento (nadie ha medido esto).
