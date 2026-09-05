# El medio que falta — v2 (corregida)

**Hyphae en el residual · de la crítica al contrato de desarrollo**

- **Qué es:** la misma spec viva del 2026-08-29, con los errores técnicos corregidos y el registro de cada corrección.
- **Qué no es:** interpretabilidad post-hoc, un dashboard, un manifiesto contra las GPU.
- **Estado:** fase de desarrollo. El objeto interno tiene nombre y ahora tiene matemática que cierra.
- **Invariante (reformulado):** CUDA y ROCm proponen. El único kernel con derecho a escribir `h` ejecuta política constituida por Hyphae. Ningún `optimizer.step()` ni export de checkpoint ocurre sin los receipts del paso.

---

## 0 · ERRATA — qué estaba mal y cómo se arregló

| # | Error en v1 | Por qué es un error | Arreglo en v2 |
|---|---|---|---|
| E1 | `H(h_t) + binding = H(h_t+1)` como prueba offline | Los hashes no son homomórficos: `H(a) + Δ ≠ H(a+Δ)`. Tal como estaba escrita, la verificación es matemáticamente imposible sin el estado crudo, que la propia spec prohíbe persistir. | Replay del residual (no de las capas): para microbatches auditados se derrama `h_0` (≈3 MB) y el verificador CPU reconstruye `h_final` aplicando solo los bindings commiteados. Ver §3.3. |
| E2 | Contrato de gradiente ausente («Autograd intacto») | Si `apply()` escribe solo los candidatos commiteados (top-k / cuantizados), el grafo que ve autograd **no** es el de `delta_hat`. Top-k enmascara gradientes; la cuantización a codebook no es diferenciable. Y con ABORT=identidad la capa rechazada no recibe señal: capas muertas. | Contrato de gradiente explícito: STE para la cuantización, máscara top-k estilo MoE, gradiente directo a `mag`, pérdida auxiliar de balanceo para enseñar al modelo a no chocar slots. Ver §3.4. |
| E3 | ACK síncrono de Hyphae por capa en el hot path | 32 capas × (RTT UDS + group-commit ~200 µs–10 ms) ≈ ≥8 ms de stall por microbatch, y el device-host sync por capa rompe CUDA graphs y el pipelining asíncrono — exactamente la occupancy que la spec dice proteger. Contradice su propia lista negra («esperar el proof en el hot path» es la misma clase de stall). | Autoridad ≠ durabilidad. Allocate determinista con espejo del catálogo residente en device (decisión local, sin round-trip); journaling asíncrono; **barrera de commit por paso**, solapada con el backward, antes de `opt.step()`. Ver §3.5. |
| E4 | `next: hash1` dentro del binding | Un objeto no puede contener el hash de un estado que aún no existe. Además contradice el propio ejemplo del journal de v1 (`COMMIT id=… prev=9f3a head=4c11`), que es la forma correcta. | El binding lleva `prev`. El `head` lo emite el COMMIT receipt. La cadena vive en el journal append-only, no en punteros mutables. Ver §3.2. |
| E5 | Catálogo de 64 slots sin matemática de capacidad; «slot» y «feature» sin definir | La superposición existe porque el modelo necesita más features que direcciones. Si el número de hechos nombrables se capa en 64, la capacidad colapsa y el experimento muere por diseño, no por hipótesis. | Separación formal: **slot** = puerto de escritura (subespacio, S=64 particiona d_model), **feature** = entrada de un codebook aprendido (N_f = 4k–32k, barrido prerregistrado). Colisión = dos features contendiendo el mismo (pos, slot). Ver §3.1. |
| E6 | Ley 1 vacua («el residual empieza siendo x») | Eso ya es cierto en todo transformer pre-norm. Lo no-trivial es otra cosa. | Ley 1 reescrita: **zero-init del camino de escritura** (gates/escala a 0, precedente: ReZero/Fixup) + `H(h_0)` commiteado como hecho ancla. Ver §3.2. |
| E7 | Mutación de pesos invisible | La tesis es «ningún estado que el modelo trata como real muta sin dejar un hecho» — y los pesos son el estado más real que existe. v1 los ignoraba; además el problema del export («al salir el .pt vuelve el molde») queda sin mecanismo. | Un binding de paso por `optimizer.step()`: `(step_id, lr, grad_norm, weight_hash_prev, weight_hash_next)`. Barato (1 tx/paso) y es lo que hace verificable el checkpoint exportado. Ver §3.6. |
| E8 | Grano de ABORT inconsistente | El pseudocódigo de v1 aborta la capa entera (`else: h = h`) mientras la Lámina B aborta por slot. Un slot contendido no puede matar la escritura completa de la capa. | Commit mask por candidato: `h = apply(h, ack.committed)`; los rechazados se journalizan individualmente como OVERFLOW/ABORT. Ver §3.5. |
| E9 | API de Hyphae parcialmente imaginada | Verificado contra el source de **v2.2.0** (`release-v2.2.0-crates`): no existe `allocate` como primitiva; los proofs son síncronos (`get_record_with_proof`), sin batch asíncrono; los records no tienen prev/next hash nativo (solo `previous_digest` por bloque WAL). **Rectificación de la primera versión de esta errata:** `hyphae.tx()` no era imaginada — existe la API de transacciones explícitas cliente-visible (`transaction_begin/stage_*/commit/rollback`, opcodes 32–40; `ProductOperation::TransactionCommit`); y el verbo `journal` sí existe (`hyphae_memory_journal`): los 5 verbos de v1 eran correctos. | Se marca explícitamente qué existe y qué hay que construir (§V, §VI). La tx de v1 mapea a `begin_transaction + stage + commit` bajo un solo CSN (2.2.0 lo llama «transactional agent memory» — misma tesis). `allocate` = CAS de aplicación sobre MVCC first-committer-wins. Proof batch asíncrono = capacidad nueva. |
| E10 | Codebook sin versionar rompe el verify | El codebook es aprendido: cambia en cada paso. Verificar `mag · codebook[feature]` requiere el codebook exacto de ese paso. | El binding referencia `step_id`; el codebook queda cubierto por la cadena de hash de pesos (E7). Ver §3.3. |
| E11 | Riesgo de colapso de codebook no contemplado | Fallo clásico de VQ: la mayoría de features muere, pocas concentran el tráfico; el catálogo «funciona» mientras la capacidad se desploma. | Métrica prerregistrada obligatoria: entropía de uso / perplexity del codebook + mitigación (EMA, reinicio de códigos muertos). Ver §6. |
| E12 | «0,1%» sin definir; sobreclamas menores | Jerga de la conversación original sin referente; «identity-init como único init que respeta una ontología» ignora precedente (ReZero, Fixup, DeepNet). | Se elimina el «0,1%». Se citan precedentes: la fuerza de la spec es componer piezas probadas (VQ-VAE, MoE routing, SAE, ledger), no la novedad de cada pieza. |
| E13 | Umbral falsable débil | «Mover la tasa de overflow visible» es trivial: overflow es un parámetro de diseño que tú controlas. No falsa nada. | Umbrales reales en §6: gap de perplexity vs baseline a FLOPs iguales, detección del 100 % de adulteraciones de WAL de 1 bit con localización, Jaccard de bindings seed-a-seed, entropía de uso del catálogo. |

Lo que v1 tenía **correcto** y se conserva: el residual como canal compartido que el campo se negó a tipar (lectura correcta de superposición); overflow y abort como hechos de primera clase («el no-hecho es dato; el álgebra cruda no»); verificar transiciones de estado y no MACs; dos backends o no cuenta; manifiesto inmutable y umbral antes del número; un negativo publicado es un resultado.

---

## I · ARCO

Sin cambios sustantivos: la historia de cómo se llegó aquí (MLP → tradeoff-o-bug → cruzada del borde → el objeto que falta → las GPU no salen → runtime, no laboratorio) es registro histórico y se mantiene como está en v1. Dos notas:

- El paso 3 queda reforzado, no debilitado, por E1: el objeto que falta existía, pero su prueba estaba mal escrita. Ahora está bien escrita.
- El paso 4 se mantiene íntegro: el vendor no es la tesis; el `+=` anónimo sí. Si el diseño solo cierra en NVIDIA, no se rompió el molde: se casó con un vendor.

## II · DIAGNÓSTICO

Se mantiene la tabla de tres síntomas con una corrección de rigor (E6):

| Síntoma | Lectura barata | Lectura de esta spec |
|---|---|---|
| Init aleatorio | Tunear Xavier/He | Nacer sin constructor. Lo accionable no es «el residual empieza en x» (eso ya es verdad en todo pre-norm); es **el camino de escritura nace en cero** y `h_0` se commitea como ancla. |
| Superposición / polisemanticidad | Abrir neuronas | Bus sin catálogo. Más features que direcciones, colisión invisible. (Se añade: la respuesta no es capar features — es nombrar la contención.) |
| CUDA / ROCm asíncronos | NVIDIA es el error | Estado sin historia. El no-determinismo duele solo si existe un mismo objeto entre t y t+1. |

La interpretabilidad mecánica (SAE, dictionary learning, probes) sigue sin ser esta spec: esta spec no pregunta cómo leer el campo; pregunta cómo prohibir que el campo mute sin dejar un hecho.

## III · TESIS (corregida)

> El forward es ilegal si no factoriza por un sustrato de objetos identificables. No es interpretabilidad: es hacer que la red no pueda cambiar su estado sin emitir un hecho, ni volverse durable sin receipt.

### 3.1 Catálogo: slots y features (E5)

- `d_model = S × d_slot`. Ejemplo: 768 = 64 slots × 12 dims. El **slot** es un subespacio contiguo del residual: una dirección de escritura.
- **Feature** = fila de un codebook aprendido `C ∈ R^{N_f × d_slot}`, `N_f ∈ {4096 … 32768}` (barrido prerregistrado, análogo al factor de expansión de un SAE).
- Slot hogar: `σ(f) = f mod S` en fase 1 (hash-home determinista); aprendido en fases posteriores.
- **Colisión**: dos candidatos commiteables contendiendo el mismo `(pos, slot)`. Política determinista: gana `|mag|` mayor; el perdedor se journaliza como `OVERFLOW`. La política es un objeto commiteado en Hyphae — cambiarla es un hecho, no un hotfix.

### 3.2 El hecho interno (E4, E6)

```
binding {
  id:       b_{run}_{step}_{layer}_{pos}_{k}
  slot:     s ∈ [0, S)          # puerto de escritura
  feature:  f ∈ [0, N_f)        # entrada del codebook, versionada por step
  mag:      a                   # bf16, tal como se aplicó
  device:   cuda | rocm
  prev:     hash del head anterior de su cadena (layer, microbatch)
  # `next` NO existe: el head posterior lo emite el COMMIT receipt.
  # opcional para localización de fallas: H(pre_leaf) del slot tocado
}
```

Delta aplicado: `d = bf16_mul(a, C_step[f])`, escrito en el subespacio del slot: `post_leaf = bf16_add(pre_leaf, d)`. Suma y multiplicación IEEE-754 escalares, sin FMA ni fusión: bit-exactas entre CPU, CUDA y ROCm. (La reproducibilidad cruzada de `delta_hat` sigue sin prometerse — lo reproducible es la **aplicación** del hecho, no su génesis. El binding declara de qué device nació.)

**Cuatro leyes (reescritas, ejecutables):**

| Ley | Enunciado v2 | Si se viola |
|---|---|---|
| 1. Identidad | El camino de escritura nace en cero (zero-init de gates/escala; precedente ReZero/Fixup). `H(h_0)` se commitea como hecho ancla del microbatch. | Init gaussiano como constructor: no hay sujeto ni ancla que verificar. |
| 2. Delta nombrable | Toda escritura al residual es un binding `(slot, feature, mag)`. Cruzan candidatos, no tensores enteros. | Smearing anónimo en R^n. El medio vuelve a ser sopa. |
| 3. Overflow visible | El bus tiene slots; la colisión se declara y se journaliza con ID. El no-hecho es dato. | Superposición invisible. El invariante es decorativo. |
| 4. Commit o silencio | El único kernel que escribe `h` es `apply()`, gobernado por política commiteada. Ningún `opt.step()`, export ni emisión de claim sin los receipts del paso. ABORT por candidato = ese delta no existe. | El modelo avanza (o se exporta) en silencio. El molde ganó. |

### 3.3 Verificación (E1, E10) — la ecuación corregida

La v1 escribía `H(h_t) + binding = H(h_t+1)`, que no es computable. La forma correcta, CPU-only, sin GPU:

```
T1 · Replay de residual (microbatches auditados, muestreo 1/N):
    se derrama h_0 crudo (~3 MB para 2048×768 bf16) y el stream completo
    de bindings del microbatch. El verificador reconstruye:

        h_final' = h_0 ⊕ ⨁_{b ∈ commits, en orden} apply(b, C_{step(b)})

    y comprueba  H(h_final') == head commiteado.
    Costo: ~50M sumas bf16 ≈ decenas de ms de CPU. No prueba cada MAC.
    Prueba que ningún byte del residual cambió fuera de los hechos emitidos.

T1.5 · Localización (opcional): si el binding lleva H(pre_leaf), una
    divergencia se bisecta a la capa y slot exactos.

T2 · Cadena: head_{l+1} = H(head_l || bindings ordenados de la capa || meta).
    Cualquier adulteración del WAL de 1 bit rompe la cadena y se localiza.
```

Requisitos que esto impone: (a) el codebook se versiona por paso vía la cadena de pesos (§3.6) — el binding referencia `step_id`; (b) `apply` es aritmética IEEE definida (§3.2); (c) `h_0` de microbatches auditados se derrama crudo — para los no auditados solo viaja su hash.

### 3.4 Contrato de gradiente (E2) — sección nueva, faltaba entera

- **mag** es un escalar continuo: gradiente directo (diferenciable).
- **Selección top-k**: gradiente enmascarado a los candidatos commiteados (mismo sesgo aceptado que el routing de MoE; Shazeer 2017, Fedus 2021).
- **Codebook**: straight-through estimator o actualización EMA estilo VQ-VAE (van den Oord 2017) + commitment loss.
- **OVERFLOW/ABORT**: el candidato rechazado recibe gradiente cero **más** una pérdida auxiliar de balanceo de carga (análoga a la aux loss de MoE) para que el modelo aprenda a vivir dentro del catálogo. Sin esta señal, los rechazos son ruido no estacionario y las capas castigadas mueren.
- **Riesgo declarado**: colapso de codebook. Mitigación: EMA + reinicio de códigos muertos; métrica de entropía de uso prerregistrada (§6).

### 3.5 Autoridad ≠ durabilidad (E3, E8) — la costura corregida

La pregunta sigue siendo una sola: **quién tiene derecho a decir `h = h + d`**. La respuesta v2 tiene tres piezas en vez de un round-trip:

1. **Espejo del catálogo en device.** La política de allocate es determinista y su estado es minúsculo (bitmap de S slots × posiciones). El kernel `apply()` decide COMMIT/OVERFLOW/ABORT **localmente**, por candidato, ejecutando la política que Hyphae constituyó. Cero stalls; en training hay un solo escritor, así que no hay conflictos que arbitrar en caliente.
2. **Journal asíncrono.** Los veredictos fluyen a Hyphae por ring buffer (D2H async, ~MB/paso): catálogo + WAL + MVCC reales. El MVCC no arbitra al escritor único; da snapshot isolation a los lectores concurrentes (verificador, monitor) mientras el training escribe.
3. **Barrera por paso.** `opt.step()` espera los CommitReceipts del paso — espera que se solapa con el backward (los bindings del forward ya están en vuelo). Con group commit de 200 µs–1 ms amortizado contra 20–100 ms de paso: <1–5 % de overhead, contra el ≥8 ms/microbatch + ruptura de CUDA graphs del diseño v1.

El invariante sobrevive intacto donde importa: el único código que muta `h` es `apply()`; `apply()` solo ejecuta política commiteada; nada se vuelve durable, exportable ni citable sin receipts. La Lámina C queda enmendada en una línea: donde decía `ack = hyphae.tx(pack(delta_hat))` en el loop de capa, la transacción es por paso y el ACK por capa es un token local del espejo.

### 3.6 La cadena de pesos (E7)

Un binding por `optimizer.step()`:

```
STEP  step_id=t  lr=…  grad_norm=…  w_prev=H(θ_t)  w_next=H(θ_{t+1})
```

Cierra tres agujeros a la vez: el codebook queda versionado (E10), la mutación de pesos deja de ser silenciosa, y el `.pt` exportado es el eslabón final de una cadena verificable — el molde ya no «vuelve» en el export. Hash incremental (solo filas tocadas del codebook + resumen por tensor) para no pagar el hash completo por paso.

## IV · LÁMINAS

Las tres láminas de v1 siguen siendo la spec dibujada, con dos enmiendas declaradas:

- **Lámina B:** la rama triste (F12 ocupado → OVERFLOW journalizado) se decide en el espejo de device y se journaliza async; sigue siendo tan importante como la feliz. El verificador ya no «reconstruye H(h_t) + binding» — ejecuta el replay T1 de §3.3.
- **Lámina C:** el device espera token local, no ACK remoto; la barrera de receipts es por paso. «Una transacción por unidad de hecho» se conserva; la unidad durable es el paso, el hecho sigue siendo el binding.

## V · RUNTIME Y MEMORIA (corregido)

Hyphae es parte de la AI: training, inferencia y el tiempo entre turnos. La corrección es separar autoridad (siempre, barata, local) de durabilidad (por clases, real en Hyphae: `Strict | Group | Memory`):

| Tiempo | GPU | Hyphae — autoridad | Hyphae — durabilidad |
|---|---|---|---|
| Training | propone `delta_hat`, backward, step | espejo de catálogo gobierna `apply()`; barrera de receipts antes de `opt.step()` | Group: bindings de paso, STEP chain, todo OVERFLOW/ABORT |
| Inferencia | propone `delta_hat` del forward | mismo espejo, mismos veredictos | Memory + spill async; flush obligatorio al emitir un claim citable |
| Entre turnos | nada | — | Strict: memoria de agente — los 5 verbos de v1 existen en 2.2.0 (`store / journal / recall / forget / status`), y desde 2.2.0 `store` comitea documento + lifecycle + TTL bajo un solo CSN (transactional agent memory) |

**El loop, corregido (E2, E3, E8):**

```python
for batch in data:
    h = embed(batch)                       # GPU (CUDA o ROCm)
    anchor = commit_async(H(h))            # Ley 1: ancla del microbatch
    for layer in model:
        delta_hat = layer(h)               # 99% FLOPs en device
        cand   = topk_quantize(delta_hat)  # device: (slot, feature, mag)
        verd   = allocate_local(cand)      # device: espejo del catálogo,
                                           #   COMMIT/OVERFLOW/ABORT por candidato
        h      = apply(h, verd.committed)  # único kernel con derecho a escribir h
        journal_async(verd)                # ring buffer → Hyphae (catalog+WAL+MVCC)
    loss = head(h) + aux_balance(verd_all) # contrato de gradiente §3.4
    loss.backward()                        # STE + máscara sobre committed (§3.4; el addendum v2.1 §1.5 enmendó este comentario, que decía “top-k” y contradecía §3.4)
    wait_receipts(step)                    # barrera: solapada con backward
    opt.step()                             # ilegal sin receipts
    commit_step_chain(θ)                   # §3.6
```

Reglas de occupancy v2: la unidad durable es el paso; el device no espera nada remoto por capa; un solo código de costura en host, sin `if nvidia / if amd`; el proof batch es asíncrono y el verify es otro proceso, CPU, offline; NVIDIA y AMD no son reproducibles cruzados y el binding declara su device — lo que sí es bit-exacto entre devices es `apply()` (§3.2).

**Lo que no tiene sentido, bien guardado** (sin cambios de fondo — era correcto):

```
OVERFLOW  run=… step=… layer=3 pos=41 slot=F12 feature=084 mag=0.61 backend=rocm
ABORT     run=… step=… layer=3 pos=41 reason=policy
COMMIT    id=b_…  prev=9f3a  head=4c11
STEP      t=1042  w_prev=…  w_next=…
FORGET    id=b_…  by=policy
```

No se persiste el hidden crudo (salvo `h_0` de microbatches auditados, T1). Se persiste el acto — incluido el absurdo, si el absurdo fue un acto.

## VI · FASE DE DESARROLLO (corregida)

El primer objeto: un transformer pequeño, presupuesto fijo, escritura zero-init, `S=64` slots como partición (no como cap de features), `N_f` barrido {4k, 16k, 32k}, dos backends, un Hyphae, seeds pareados, manifiesto inmutable, umbral antes del número.

**Qué tiene que existir, en este orden** (con lo verificado contra el source de Hyphae v2.2.0):

1. Schema del binding como record nativo — los records tipados existen; los campos de cadena (`prev`/head) son de aplicación, no nativos: se construyen sobre el journal.
2. tx de capa sobre la API real de transacciones explícitas: `transaction_begin` → `stage_*` (bindings del paso) → `commit` bajo un solo CSN (opcodes 32–40, existentes en 2.2.0). `allocate` como CAS de aplicación sobre el record de catálogo — no existe como primitiva; el first-committer-wins del MVCC real lo soporta.
3. Espejo de catálogo en device + kernel `apply()` gateado — el `+=` solo corre bajo veredicto.
4. `pack()`/`topk_quantize` en device; D2H async por ring buffer; sin sync por capa.
5. Contrato de gradiente: STE + máscara + aux loss (§3.4). Sin esto no hay training viable, punto.
6. Barrera de receipts por paso + STEP chain (§3.6), con `DurabilityClass::Group`.
7. Proof batch **asíncrono** — capacidad nueva: hoy los proofs de Hyphae son síncronos por consulta. Verify offline sellado ya existe y se reutiliza.
8. Verificador T1/T1.5/T2 (§3.3) en proceso aparte, CPU, sin GPU.
9. Journal de OVERFLOW/ABORT first-class y consultable — si el no-hecho no se puede consultar, el invariante no existe.
10. El mismo harness en CUDA y ROCm. Si solo cierra en uno, se eligió vendor.

**Qué se mide, qué no (E13 — umbrales falsables de verdad):**

| Se mide (con umbral prerregistrado) | No se mide (todavía) |
|---|---|
| Gap de perplexity vs baseline denso a FLOPs iguales (matar la rama si > X% prerregistrado) | MMLU, chatbot arena, «transparencia percibida» |
| verify() detecta 100 % de adulteraciones de 1 bit en WAL y las localiza a binding | Que una neurona «signifique» un concepto humano |
| Jaccard de conjuntos de bindings commiteados seed-a-seed vs similitud de activaciones del baseline | Paridad numérica NVIDIA vs AMD |
| Entropía de uso / perplexity del codebook (colapso = matar la rama) | zkML de cada matmul |
| Tasa y localización de OVERFLOW/ABORT por capa a lo largo del training | Que Hidden Layer 2 se vuelva un mapa semántico |
| Overhead de pared vs baseline (barrera + journaling) ≤ presupuesto prerregistrado | |
| Un humano cita un binding y reencuentra su proof | |

**Lista negra (v1 + una entrada nueva):** loguear cada MAC o activación «por si acaso»; un slot basura donde verter sopa para que el loss baje; `if nvidia / if amd` en la costura; esperar el proof en el hot path; **esperar ACK remoto por capa en el hot path (el error E3 de esta misma spec)**; fsync por forward; mezclar este runtime con el foro «internet muerto»; prometer que las elipses del MLP se vuelven palabras; vender Hyphae como el modelo.

**Contrato de la fase** (se conserva íntegro): la unidad semántica es el binding o el span citado; nada irreproducible cuenta; el modelo opaco puede existir pero no hablar solo; todo experimento nace con manifiesto y umbral; un negativo publicado es un resultado; si el invariante no duele, no se rompió el molde.

**Definición de hecho** (se conserva, con E7): un hecho interno es un binding commiteado, un intento que el catálogo rechazó, o una transición de pesos sellada. Lo que no pasó por allocate no obliga a persistir un vector; sí obliga a persistir el evento. Esa asimetría es el borde entre memoria y profiler.

## CIERRE

Se quiere un modelo que siga entrenando a escala en B300 y MI355X, y que no pueda cambiar su estado interno — residual **ni pesos** — sin dejar un objeto citable, hasheable y abortable. La v1 tenía razón en la categoría que falta y se equivocaba en tres lugares donde el papel aguantaba y la física no: una ecuación de verificación no computable, un contrato de gradiente ausente y una autoridad síncrona que mataba la occupancy que juraba proteger. Las tres están corregidas sin ceder la tesis: CUDA y ROCm siguen siendo el músculo; Hyphae sigue siendo quien firma; y la firma ahora tiene una matemática que un proceso CPU puede ejecutar en frío.

Lo que falta no es otra lámina. Es el schema (§3.2), el espejo y su kernel (§3.5), la cadena de pesos (§3.6), el verificador (§3.3) y el primer run que pueda perder.

---

| Campo | Valor |
|---|---|
| Título | El medio que falta · Hyphae en el residual — v2 corregida |
| Tipo | Spec de trabajo / no es un paper |
| Fecha | 29 de agosto de 2026 |
| Base | `hyphae_el_medio_que_falta_spec.pdf` (v1) + errata E1–E13 |
| Hyphae verificado | v2.2.0 (`release-v2.2.0-crates`, commit `5bd8afb`) — tx explícitas y 5 verbos de memoria existen; `allocate`, proof batch async y prev/next hash por record siguen siendo capacidades a construir |
| Invariante | CUDA y ROCm proponen. `apply()` ejecuta política constituida por Hyphae. Nada es durable, exportable ni citable sin receipts. |
| Precedentes citados | VQ-VAE (van den Oord 2017) · MoE routing/aux loss (Shazeer 2017, Fedus 2021) · ReZero/Fixup · SAE/dictionary learning (superposición: Elhage 2021/2022) |
| Siguiente acto | schema del binding + espejo de catálogo + kernel `apply()` + STEP chain + verificador T1 |

> Un sistema no es confiable porque dice lo correcto. Es confiable cuando sus límites, su historia y su evidencia pueden examinarse. — aplicado ahora también a esta spec: los errores de v1 quedan en la errata, no se borran.
