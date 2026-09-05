# Fase 4 — Hallucinations: del bug invisible al hecho journalizado

*Diseño prerregistrado (2026-08-30, directiva del usuario: "quiero una fase
para verificar hallucinations, creo que podemos arreglar este bug").
Honestidad primero: qué es alcanzable, qué es hipótesis, qué es humo.*

> **ORDEN (directiva del usuario, 2026-08-30): esta fase se ejecuta AL
> FINAL, sobre el nanochat d20 entrenado con HyTorch (fase 3).** Un modelo
> que habla es el sujeto correcto para confabulación; el toy solo validaría
> el instrumento. El código (signatures.py, hallu.py, manifiesto 4a) queda
> listo en el repo y NO se lanza hasta que fase 3 cierre. El run 4a lanzado
> prematuramente el 2026-08-30 se abortó (~2 min de GPU); ningún resultado
> se registró.

## La reformulación que nuestro sistema permite

La literatura trata la alucinación como un problema de OUTPUT: el modelo
dice algo falso y se detecta después (fact-checkers, RAG, calibración de
logits). Nuestro sistema permite una pregunta anterior, mecánica:

> Cuando el modelo escribe un token CORRECTO, sus escrituras al residual
> citan features con historia para ese contexto. Cuando ALUCINA, ¿qué
> escribió? ¿Features de baja historia? ¿Más OVERFLOW? ¿Menos energía
> (|mag|)? ¿Commits de features "genéricas" en vez de específicas?

Hipótesis central (falsable): **la alucinación tiene una firma en el
journal** — el acto de confabular se distingue del acto de recordar EN LOS
HECHOS DE ESCRITURA, antes de mirar el output. Si existe, tenemos:

1. **Detector**: score de confianza mecánico (no logits — hechos) por token
   generado, computable en vivo desde el journal.
2. **Freno**: una POLICY que degrada a "no sé" cuando la firma aparece
   (ABORT del acto de confabular — journalizado, como todo).

## Por qué nosotros podemos preguntar esto y nadie más

- Cada token generado deja k×L hechos (commit/overflow/abort con feature,
  mag, slot). La generación con catálogo produce un TRACE AUDITADO del
  estado interno por token — eso no existe en ningún stack.
- La sonda de fase 2 ya probó que features nombran tokens/estructuras con
  lift alto y que la ablación es quirúrgica (10.7× específica). La firma de
  alucinación es la misma maquinaria apuntada a la generación.
- El "no-hecho es dato": OVERFLOW/ABORT durante generación son exactamente
  la señal de "el modelo quiso escribir algo que la política rechazó" —
  candidato natural a proxy de inseguridad interna.

## Métricas de firma (F1-F5, prerregistradas)

Por token generado t, de los hechos del journal:

| # | Métrica | Intuición |
|---|---|---|
| F1 | `familiarity(t)` = media de log(uso_training(f)) sobre commits | ¿cita features con historia o rarezas? |
| F2 | `context_fit(t)` = lift medio de (f, token_prev) vs tabla del training | ¿las features citadas SUELEN aparecer en este contexto? |
| F3 | `energy(t)` = Σ\|mag\| de commits | ¿escritura firme o débil? |
| F4 | `contention(t)` = #OVERFLOW / k | ¿el top-k peleó? |
| F5 | `entropy(t)` = entropía de features commiteadas en el paso | ¿escritura enfocada o dispersa? |

Detector v0 = regresión logística sobre F1-F5 (5 parámetros, sin red
adicional — el detector debe ser tan auditable como el ledger).

## Protocolo (dos etapas, ambas con manifiesto)

### 4a — ¿Existe la firma? (toy 1b, ya entrenado, barato)

1. **Task con ground truth mecánico**: cloze de wikitext val — prompt =
   contexto real, target = token real siguiente. Generamos con el modelo
   catalogado; cada token generado se etiqueta `correct` (== target o en
   top-k del contexto real) / `wrong` (confabulación medible).
   Sin LLM-judge: la etiqueta es exact-match contra el corpus.
2. Capturamos F1-F5 por token desde el journal de generación.
3. **Umbral prerregistrado**: AUROC(detector, correct-vs-wrong) ≥ 0.65 en
   held-out → la firma existe y la fase sigue. < 0.65 → negativo publicado
   (la firma no está en estos hechos a esta escala) y la fase muere.
   0.65 es deliberadamente modesto: buscamos existencia, no producto.

### 4b — ¿El freno funciona? (solo si 4a pasa)

1. POLICY de generación: si score < τ (τ del held-out de 4a), el paso se
   marca LOW_CONFIDENCE en el journal y el sampler degrada (p.ej. fuerza
   <eos> o token de abstención en el task QA).
2. Métrica: precisión del modelo CON freno vs SIN freno a cobertura fija —
   curva selective-prediction (riesgo/cobertura), estándar y comparable.
3. Umbral prerregistrado: a 80% de cobertura, reducción relativa ≥25% de
   la tasa de error. Menos → el freno no paga su costo; se publica igual.

### 4c — Escala real (post fase 3): nanochat d20

El mismo detector sobre un modelo que HABLA, con SimpleQA-style prompts:
"deny the confabulation act" en un chat real. Solo si 4a+4b pasan en toy.

## Qué NO prometemos (lista negra de la fase)

- "Arreglar" la alucinación en general: la confabulación de un LM incluye
  factualidad del mundo que un toy de 124M ni posee. Atacamos la VERSIÓN
  MECÁNICA: detectar/frenar la escritura-sin-historia. Si el bug entero
  cae, caerá a escala — esta fase construye el instrumento y la evidencia.
- Cero claims sin el AUROC prerregistrado. El negativo es un resultado.
- El detector no usa logits ni probes de activaciones: SOLO hechos del
  journal. (Comparación contra baseline de logit-entropy se reporta como
  referencia — si los hechos no superan a los logits, se dice.)

## Entregables

- `python/hytorch/generate.py` — generación autoregresiva con journal
  capture por token (reusa el forward de eval + kv-cache simple).
- `python/hytorch/signatures.py` — F1-F5 desde recs + tablas del training
  (la tabla token→feature de la sonda ES el prior de familiaridad).
- `manifests/phase4a-signature.json` — umbral AUROC 0.65 firmado antes de
  ver el número.
- `results/hallu/...` — dataset etiquetado, curvas, veredicto.
