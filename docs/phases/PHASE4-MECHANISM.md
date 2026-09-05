# Fase 4 — MECANISMO: por qué alucina, atacado de frente

## H0 — La homología bypass↔confabulación (directiva del usuario, 2026-09-01)

*"Cuando el loss baja pero igual no está escribiendo — eso es lo que
sucede. Inventa la información porque la PERDIÓ en el training."*

El colapso del canal (PHASE5-CHANNEL-COLLAPSE) y la confabulación son el
MISMO fenómeno estructural a dos escalas de tiempo:

```
TRAINING (medido, fase 5 launch 7):        INFERENCE (hipótesis H0):
canal tipado MUERTO ──X                    vía de conocimiento AUSENTE ──X
        │                                          │
input → modelo                             prompt → modelo
        │                                          │
        └─ bypass → loss BAJA ✓                    └─ bypass → texto FLUIDO ✓
   "todo bien, sigo aprendiendo"              "todo bien, sé la respuesta"
```

En ambos casos un observador externo (curva de loss / fluidez del texto) ve
salud mientras el camino que debía llevar la carga está muerto. La
confabulación es la firma a tiempo de inferencia de información que el
training PERDIÓ (o nunca escribió por el canal auditable): el modelo
aprendió caminos de forma sin caminos de contenido, y en el hueco, inventa.

**La trampa de interpretabilidad que esto expone (y que nuestro ledger
esquiva):** observar una feature/mecanismo y asumir "esto explica el
comportamiento" es unfalsifiable si el modelo puede estar usando OTRO
camino. La curva de loss de launch 7 era indistinguible de la de un canal
vivo; solo los receipts (qué se escribió, qué se rechazó, qué gradiente
movió el codebook) separaron "canal caro" de "canal muerto con bypass".
Attribution graphs sin testigos de escritura tienen exactamente este
problema (~25% cobertura, error nodes). Nosotros no inferimos el camino:
lo journalizamos.

**El experimento natural que ya tenemos custodiado (H0-test, gratis):**
- Modelo A = d20 de 3.4 (canal MUERTO desde ~step 150): TODA su generación
  es bypass puro — a inferencia su journal muestra ~0 commits; cada token
  es "no atestiguado" por el canal tipado. El caso límite: 100% bypass.
- Modelo B = d20 de fase 5 (canal VIVO, commit 24-37% medido): generación
  atestiguada — tokens con hechos de escritura reales detrás.
- Predicción H0: sobre el MISMO task de frontera de conocimiento
  (knowledge_boundary.py), A confabula MÁS y con MENOS abstención que B a
  fluidez comparable; y el detector F1-F9 solo puede funcionar en B (en A
  no hay hechos que firmar — la ausencia misma es la firma).
- Corolario para el paper: "un modelo puede aprender a rodear su propio
  residual stream" es publicable por sí solo; que la confabulación sea la
  versión inference-time del mismo bypass es la tesis H0.


*Investigación 2026-08-31 (fuentes primarias: Anthropic "Biology of an LLM"
mar-2025 §Entity Recognition and Hallucinations; Farquhar et al. Nature 2024
semantic entropy; nuestra fase 1-3). Este doc convierte la teoría en
hipótesis falsables SOBRE HECHOS DE ESCRITURA — el instrumento que solo
nosotros tenemos.*

## Lo que el campo ya sabe (y sus límites)

**Anthropic (circuit tracing, Claude 3.5 Haiku):** la alucinación de
entidades es un **fallo de un circuito inhibitorio**. El DEFAULT del modelo
fine-tuneado es rechazar ("can't answer" se activa por defecto en prompts
Human/Assistant; "names are assumed to be unfamiliar unless proven
otherwise"). Responder requiere que features "known entity/known answer"
**inhiban** ese default. El misfire: el modelo reconoce el NOMBRE (Karpathy)
→ "known answer" se activa *débilmente pero lo suficiente* → apaga el
rechazo → y el circuito paralelo que computa la respuesta —que es OTRO
circuito— solo puede adivinar. Cita clave: *"the circuits determining if
the model believes it knows the answer may be different from those actually
computing the answer."* La confabulación es un **umbral mal calibrado de
auto-conocimiento**, no ausencia de señal.

**Límites de su método (nuestro hueco):** attribution graphs sobre un
replacement model (CLT) — indirecto, post-hoc, con error nodes no
interpretables, "insight satisfactorio en ~25% de los prompts", e
intervenciones que requieren magnitudes antinaturales. Ellos INFIEREN el
acto; **nosotros lo tenemos journalizado**: cada escritura al residual es un
hecho con veredicto, magnitud y feature, del modelo REAL, sin replacement.

**Farquhar (Nature 2024):** la confabulación es detectable por entropía
semántica sobre N generaciones (¿el modelo dice cosas distintas cada vez?).
Caro (5-10× compute) y de caja negra. Es nuestro baseline a batir junto a
logit entropy.

## Las tres hipótesis mecanicistas, en NUESTRO espacio de escrituras

El mecanismo de Anthropic vive en features de activación. Nuestro modelo
escribe el residual por un catálogo tipado — si el mecanismo es real y
general, debe dejar huella en los HECHOS:

**H1 — El default y su supresión son visibles como régimen de escritura.**
Si "responder" requiere suprimir un default, los tokens *recordados* deben
mostrar escrituras firmes y familiares (features de alta historia, energía
alta, entropía baja de features) y los *confabulados* escrituras del régimen
"adivinanza": features genéricas de formato/sintaxis (las que disparan para
TODO) en vez de features de contenido específicas. Métrica nueva:
**especificidad** = masa de commits en features de cola (baja frecuencia
global) vs features de cabeza. Predicción: confabulación ≈ escritura
dominada por features de cabeza.

**H2 — El misfire es de UMBRAL, no de ausencia (la firma borderline).**
Anthropic: "known entity se activa *débilmente* en el caso Karpathy". En
hechos: los tokens confabulados sobre entidades *vistas pero poco* en
training deben mostrar familiarity INTERMEDIA — ni la alta del recuerdo ni
la baja del OOD puro — con context_fit bajo (las features citadas no suelen
aparecer en ESTE contexto). Métrica nueva: **divergencia F1−F2**
(familiar-en-general pero no-en-este-contexto = la firma exacta del "conozco
el nombre pero no la respuesta"). Predicción: F1−F2 separa confabulación de
recuerdo mejor que F1 o F2 solos.

**H3 — La confabulación es inestable capa-a-capa (el guessing circuit no
converge).** Si el circuito de contenido adivina, sus escrituras tempranas y
tardías no deben reforzarse: predicción de baja **coherencia cross-layer**
(¿las capas tardías escriben las MISMAS features que las tempranas
prepararon, o cada capa tira para un lado?). En recuerdo genuino, el
refinamiento iterativo del residual (early layers proponen, late layers
confirman) debe verse como re-cita de features. Bonus exclusivo nuestro: la
señal por UNIDAD (attn vs mlp) — ¿la confabulación es mlp-dominante
(recuerdo asociativo fallando) o attn-dominante (copia de contexto
fallando)? Nadie puede medir eso; nosotros lo tenemos gratis en el wire.

## El experimento que distingue las tres (diseño, prerregistrable)

**Task — frontera de conocimiento mecánica (sin LLM-judge):**
1. Del propio training set del d20 (fineweb shards CITADOS por sha256),
   extraer entidades por frecuencia: E_alta (≥100 apariciones), E_baja
   (2-10), E_cero (entidades reales del mundo AUSENTES del corpus — se
   verifica por grep sobre los shards, que TENEMOS hasheados).
2. Prompts cloze/QA idénticos en forma sobre los tres grupos. Labels:
   correcto-contra-corpus / confabulado / (si el modelo aprende a
   abstenerse post-SFT: rechazo).
3. Capturar F1-F9 por token generado (F1-F5 actuales + los nuevos abajo).

**Signatures v2 (lo que la teoría pide medir):**
- F6 `cross_layer_coherence`: Jaccard medio de features commiteadas entre
  capas consecutivas para el mismo token (H3).
- F7 `familiarity_context_divergence`: F1 normalizada − F2 normalizada (H2).
- F8 `specificity`: fracción de masa |mag| en features fuera del top-1% de
  uso global (H1).
- F9 `unit_skew`: (energía attn − energía mlp) / total (H3-bonus).

**Predicciones firmadas ANTES de correr (fase 4a-v2, sobre el d20):**
- P1: AUROC(F1-F9) ≥ AUROC(F1-F5) + 0.03 (la teoría añade señal).
- P2: F7 solo > F1 solo y > F2 solo (el misfire es de umbral → H2).
- P3: en E_baja (borderline), F7 es el feature dominante del detector
  (coeficiente máximo) — el caso Karpathy reproducido en hechos.
- P4: F6(confabulado) < F6(correcto) con d de Cohen ≥ 0.3 (H3).
- Los cuatro con su negativo publicable: si P2/P3 fallan, el misfire de
  umbral NO se transfiere del espacio de features al de escrituras — un
  resultado citable contra la generalización del mecanismo de Anthropic.

**El paso causal que nadie puede dar (4b-v2):** si H2 gana, el freno deja
de ser un score opaco: **deny-list dinámica** — cuando F7 supera τ en un
token de entidad, se fuerza ABORT(reason=policy) de las features genéricas
de cabeza que están supliendo la escritura (H1) y se mide si el modelo,
privado de la vía de adivinanza, degrada a abstención. Eso es *interviниr
el mecanismo del misfire como acto journalizado* — Anthropic inyecta
activaciones en un replacement model; nosotros negamos hechos en el modelo
real, con recibo.

## Por qué esto puede ser lo que nadie ha logrado

Anthropic INFIERE el circuito del misfire con un modelo sustituto y ~25% de
cobertura. Farquhar lo detecta desde FUERA con 10× compute. Nosotros
tenemos, del modelo real y por construcción: cada acto de escritura, su
magnitud, su historia, su contexto, su capa y su subcapa — y la capacidad
de NEGAR actos específicos como política auditada. Si la firma del misfire
existe en el espacio de escrituras, la encontramos con recibos; si no
existe ahí, ese negativo también es nuevo conocimiento. En ambos casos es
terreno que ningún otro grupo puede pisar hoy.
