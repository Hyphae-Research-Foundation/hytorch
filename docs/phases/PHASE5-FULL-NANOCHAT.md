# Fase 5 — nanochat COMPLETO con HyTorch: la data se pone seria

*Directiva del usuario (2026-08-31): "quiero al final de fase 4 una fase 5
con la creación completa de nanochat, no importa si toma 45 horas, estoy
dispuesto a asumirlo. A mayor volumen, más data con la que podemos trabajar.
Ya no sería un toy. Se debe reestudiar todo, incluyendo hallucinations."*

## Qué es fase 5

El **speedrun completo de nanochat** (el tier "$100": d20 o d24-GPT-2-grade,
según validación de costo en 3.4) entrenado end-to-end con el catálogo:
**pretraining completo compute-óptimo + midtraining + SFT + evals CORE/ARC/
GSM8K/HumanEval + chat CLI** — cada etapa con receipts. El twin vanilla es
el mismo árbol con el flag apagado, mismo horizonte completo.

Diferencia con 3.4: 3.4 es el speedrun a horizonte reducido (validar
sistema+costo); **fase 5 es el horizonte COMPLETO de nanochat** (~6.7B-11B
tokens según depth) — el modelo real, no señal.

## Presupuesto (asumido explícitamente por el usuario)

| Concepto | Estimado | Base |
|---|---|---|
| Brazo catalog (kernel paralelo, gated BIT_IDENTICAL) | 12-20 h en 8×MI355X spot | d12 medido post-fix + escala |
| Peor caso asumido | hasta 45 h | directiva del usuario |
| Brazo vanilla completo | 2-3.5 h | speedrun oficial |
| Costo total estimado | $500-800 (peor caso ~$1 700) | $36/h spot |
| Ledger | 2-8 TB según wire_retention (cadencia prerregistrada) | 17 GB medidos en d12/1500 pasos |
| Custodia | volumen DO adjunto + pull selectivo (heads, receipts, spills, STEP chain completo) ANTES del teardown | deuda saldada |

## Lo que la data seria habilita (el re-estudio completo)

1. **El gap a horizonte completo** — la pregunta abierta de 3.2: ¿+1.18 nats
   a 1500 pasos converge, crece o se cierra con 100× más tokens? Con el
   horizonte completo el codebook tiene por primera vez tokens suficientes
   para amortizar el STE. Umbral prerregistrado antes del run.
2. **Journal masivo**: ~10¹¹ veredictos de un modelo real. La sonda
   semántica pasa de 8 palabras de juguete a vocabulario completo:
   mapa token→feature de 65k tokens, purity/lift por capa Y por unidad
   (attn vs mlp — la atribución de subcapa que nadie tiene).
3. **Hallucinations RE-ESTUDIADA (fase 4 se repite aquí, como pediste)**:
   - 4a sobre el d20/d26 completo: la firma (F1-F5) con un modelo que
     REALMENTE sabe cosas (el toy no sabía nada que confabular).
   - 4b el freno selectivo en chat real: SimpleQA-style, precisión con/sin
     freno a cobertura fija.
   - 4c la demo que nadie puede hacer: preguntarle al chat, ver la firma
     por token EN VIVO desde el journal, y ablar el feature del concepto.
4. **Capacidades de Hyphae demostradas en serio**: sostener ~10⁷ hechos/seg
   agregados, TB-scale WAL, receipts a 8 ranks, consultas sobre el corpus
   completo de hechos. Números de ingeniería citables para el paper.
5. **CORE/ARC/GSM8K de ambos brazos**: el costo del canal tipado en
   benchmarks estándar, no solo loss.

## Prerequisitos (el orden que ya corre)

- **3.3** (en curso): DDP 8-rank validado — barrera colectiva, poison
  broadcast, seam en rank 0.
- **3.4**: speedrun a horizonte reducido — mide el s/paso REAL del catalog
  con el kernel paralelo a d20 en el 8×; ese número fija el presupuesto
  exacto de fase 5 y el depth (d20 vs d26).
- **Fase 4 sobre el d20 de 3.4**: primera pasada del instrumento.
- **FASE 5**: el run completo + re-estudio de todo.

## Riesgos declarados

- **Preempción spot en un run de 20-45h es CASI SEGURA** → checkpointing
  nanochat (save-every) + resume + run-hijo con manifiesto que cita el
  último head (protocolo §5.4 ya diseñado; el watchdog existe). Se valida
  el resume ANTES del run largo (kill-test de 3.4).
- El catalog arm puede seguir perdiendo en benchmarks — el valor del run es
  la data + el sistema a escala, y así está registrado en el manifiesto.
- Ledger TB-scale: wire_retention_every prerregistrado (candidato: 100) +
  spills T1 completos + STEP chain completo; el 100% de heads/receipts
  siempre durable.
