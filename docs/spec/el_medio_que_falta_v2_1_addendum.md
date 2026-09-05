# El medio que falta — v2.1 addendum (corregido)

**Cierra lo que v2 dejó como fontanería: `pack`, `apply`, ley 0, cadena de `C`/`policy`, T1 y presupuesto de WAL.**

- **Qué es:** spec ejecutable de las seis costuras que v2 nombró y no cerró, con las correcciones C1–C7 aplicadas y journalizadas.
- **Qué no es:** errata de v1, reinterpretación de la tesis, ni un paper.
- **Base:** `el_medio_que_falta_v2.md` (2026-08-29) + borrador del addendum (`~/Downloads/el_medio_que_falta_v2_1_addendum.md`). No relitiga E1–E13.
- **Estado:** addendum de fase de desarrollo. Si choca con v2, gana este documento en los seis puntos de §0 y se journaliza el choque como `SPEC_AMEND`.

---

## 0.a · CORRECCIONES sobre el borrador del addendum

El borrador cerraba A1–A6 correctamente en estructura, e incluso detectó una inconsistencia interna de v2 (máscara sobre *committed* en §3.4 vs «top-k» en el comentario del loop — v2 queda enmendada). Pero introducía tres errores sustantivos y cuatro imprecisiones. Registro:

| # | Error en el borrador | Por qué es un error | Arreglo aquí |
|---|---|---|---|
| C1 | §1.2 puntúa con `Ĉ[f]` (normalizado) pero §3.2 aplica `C_fp32[f]` (crudo) | `‖delta‖ = |mag| · ‖C[f]‖`: el `mag` del binding deja de ser la magnitud del hecho — el humano que cita el binding necesita `‖C[f]‖` para saber cuánto cambió `h`, que es exactamente la opacidad que la spec prohíbe. El top-k selecciona por alineamiento pero el impacto lo domina la norma. Y un colapso **por norma** (pocas features acaparando energía vía `‖C‖` creciente) es invisible a la entropía de frecuencias de §6.4 y al aux-loss, que ven conteos, no energía. | `apply` usa `Ĉ[f]`: la normalización es parte de `apply_ref` (mismo `ε`, orden de reducción fijo, §3.2). `C` se hashea crudo como siempre (`H_canónico`), pero la semántica del hecho es `leaf' = leaf + mag · Ĉ[f]`, de modo que `‖delta‖ = |mag|` a redondeo bf16. Score y aplicación quedan en la misma geometría. |
| C2 | Ley 0 cambia la arquitectura y el baseline de §6.4 no se entera | Attn y mlp leyendo **el mismo** `h` normalizado (sin `h = h + attn(...)` intermedio) es un *parallel block* (GPT-J/PaLM), no el pre-norm secuencial estándar. Comparar contra un denso secuencial mezcla dos variables: catálogo y topología del bloque. El gap medido no atribuye. | El baseline denso obligatorio es **parallel-block** con la misma topología funcional (misma lectura de `h`, suma densa sin catálogo). Manifiesto: `baseline.block=parallel`. Si además se quiere el secuencial clásico como referencia externa, se reporta aparte y no participa del umbral. |
| C3 | `w_prev/w_next = H(θ\C)` por paso, con «nada de resumen por tensor» | El borrador da algoritmo para `C` (768 KB → <1 ms, viable) pero exige implícitamente hashear todo `θ` (≈10⁸ parámetros, cientos de MB) **cada paso**: decenas–cientos de ms/paso de CPU/PCIe — se come el presupuesto de pared que §5 jura proteger. v2 decía «hash incremental»; el borrador lo prohíbe sin dar sustituto computable. | La cadena de pesos se sella por época de sellado: `SEAL step_id=t H_canónico(θ)` cada `seal.every_k_steps` prerregistrado, y siempre en `EXPORT`. Entre sellos, la mutación de `θ` está **evidenciada** (STEPs con lr, grad_norm, policy_id encadenados) pero no **hasheada** — límite declarado en el manifiesto, igual que Adam `(m,v)` y el RNG quedan fuera de fase 1 (§4.4). `C` sí se hashea cada paso: es chico y es el diccionario que T1 necesita exacto. |
| C4 | `prev` obligatorio en cada binding | Bajo T2 (cadena por capa) + orden total de aplicación (§3.2 regla 5), el `prev` por binding es información redundante: la posición del binding en la cadena ya es única y verificable. A `β ≤ 64` bytes, un hash de 32 bytes es la mitad del presupuesto de WAL — pagada en cada hecho del run. | `prev` pasa a **opcional** (perfil `t1_5` lo activa junto con `H(pre_leaf)`). El encadenamiento canónico es T2: `head_{ℓ+1} = H(head_ℓ ‖ bindings_ℓ ‖ meta)`. El binding mínimo baja a `β ≤ 32`. |
| C5 | «El invariante en el complemento es el binario» — sin identidad de binario | Si la honestidad de los microbatches no auditados descansa en el binario (ley 0 + único `apply`), el binario es parte de la evidencia y tiene que ser citable. El borrador no lo registra en ningún hecho. | Manifiesto: `build.artifact_hash` (hash del binario/imagen del trainer y del verificador, mismo `apply_ref` enlazado). El `RUN_START` journalizado cita ese hash. Sin él, §6.3 es prosa. |
| C6 | Selección T1 `hash(run_id, step_id, mb) mod N` vendida como no manipulable | Es determinista y conocida **antes** de correr: un trainer adversarial sabe desde el step 0 qué microbatches serán auditados y puede desviarse solo en el complemento. Para fase 1 (auto-auditoría de un harness propio) es válida — pero hay que declarar el modelo de amenaza, no insinuar más. | §6.2 declara: fase 1 audita **fallos de implementación, no adversarios**. La variante adversarial (semilla de auditoría revelada post-hoc, commit-then-reveal) queda nombrada como fase posterior. |
| C7 | Menores: (a) test de no-mutación con `id()` de Python; (b) orden de reducción del score sin fijar; (c) `abort_rules` sin instanciar; (d) D2H del anchor fuera de presupuesto | (a) `id()` no detecta aliasing de storage en torch (`.view()`, `.data`): dos tensores distintos pueden compartir memoria. (b) `⟨leaf, Ĉ[f]⟩` en fp32 depende del orden de suma; si host y device reducen distinto, el mismo `delta_hat` produce candidatos distintos y el desempate «bit-reproducible» de §1.2 no lo es. (c) Una política sin reglas instanciadas hace ABORT invocable pero nunca ejercido — y un `mag` no finito commiteado corrompe `h` **reproduciblemente** (T1 verifica el replay del NaN, no lo detecta). (d) `commit_async(H(h))` viaja por el mismo ring y no estaba en `d2h.max_*`. | (a) El test compara **bytes del snapshot** (`h.clone()` pre-layer vs `h` post-layer), no identidades. (b) Reducción secuencial por índice creciente en fp32, especificada en §1.2 como parte de la política. (c) `abort_rules` mínimas de fase 1: `nonfinite(mag) → ABORT`, `mag == 0 → ABORT` (un hecho sin efecto no es un hecho), `|mag| > mag_max → ABORT` con `mag_max` en `POLICY`. (d) El anchor entra en `d2h.max_bytes_per_step`. |

---

## 0 · Alcance

v2 corrigió el ledger. Este addendum fija el modelo que escribe en ese ledger.

| # | Agujero en v2 | Cierre aquí |
|---|---|---|
| A1 | `pack()` / `topk_quantize` no definido | §1 interfaz, k, métrica, slots vacíos |
| A2 | `layer(h)` puede mutar `h` | §2 ley 0: bloque funcional |
| A3 | bf16 escalar bit-exacto CPU/CUDA/ROCm | §3 `apply()` software, T1 same-binary |
| A4 | EMA de `C` fuera del `STEP`; policy sin id | §4 `policy_id` + `C` dentro de la cadena del paso |
| A5 | «~MB/paso» sin presupuesto | §5 presupuesto WAL / D2H en manifiesto |
| A6 | T1 al 1/N hablado como omnisciencia | §6 T1 = test de consistencia, tasa prerregistrada |

Invariante intacto: CUDA y ROCm proponen. El único kernel con derecho a escribir `h` es `apply()`. Nada durable, exportable ni citable sin receipts.

---

## 1 · `pack` — de `delta_hat` a candidatos (A1)

### 1.1 Forma

El bloque no escribe el residual. Devuelve un tensor denso de propuesta:

```
delta_hat ∈ bf16^{B × T × d_model}
d_model   = S × d_slot          # partición contigua, v2 §3.1
```

`pack` es total y determinista sobre `(delta_hat, C_step, k, métrica)`:

```
pack(delta_hat, C_step) → candidatos[B, T, k]
candidato = (slot, feature, mag, score)
```

No hay otra vía de candidato. Nada que no salga de `pack` entra a `allocate_local`.

### 1.2 Fase 1 (cerrada; no se "decide en el run")

| Parámetro | Valor fase 1 | Notas |
|---|---|---|
| Home de feature | `σ(f) = f mod S` | Hash-home. Routing aprendido = fase posterior, cambio de política, hecho nuevo. |
| Métrica de vecino | `score(s, f) = ⟨ leaf_s , Ĉ[f] ⟩` | `leaf_s = delta_hat[..., s·d_slot : (s+1)·d_slot]` en fp32 canónico (§3). `Ĉ[f] = C[f] / (‖C[f]‖₂ + ε)`, `ε = 2⁻¹⁴`. **Reducción secuencial por índice creciente, fp32, sin FMA** — el orden es parte de la política (C7b). |
| `mag` propuesto | `⟨ leaf_s , Ĉ[f] ⟩` (el score) | Escalar fp32; se almacena en el binding como bf16 *tal como se aplicará* (§3.4). Con C1, `|mag|` **es** la norma del delta aplicado. |
| `k` | prerregistrado en el manifiesto, `k ∈ {1, 2, 4, 8}` | Barrido. Un run, un `k`. |
| Top-k | por token, **global** sobre el pool de `N_f` pares legales `(s, f)` con `σ(f)=s` | No hay top-k por slot en fase 1. |
| Desempate | mayor `score`; si empate, menor `feature`; si empate, menor `slot` | Total. Bit-reproducible porque la reducción está fijada (C7b). |

Un par `(s, f)` es legal iff `σ(f) = s`. `pack` no emite ilegales.

### 1.3 Qué pasa con un slot vacío

Después de top-k global, un slot `s` en `(b, t)` puede tener 0, 1 o varios candidatos.

| Caso | Veredicto de `allocate_local` | Gradiente (§1.5) | Journal |
|---|---|---|---|
| 0 candidatos | silencio. No hay ABORT. El slot no fue propuesto. | cero a ese slot | nada |
| 1 candidato | `COMMIT` si la política vigente lo admite; si no, `ABORT` con `reason` | COMMIT: STE a `mag` y a `C[f]`; ABORT: 0 + aux | COMMIT o ABORT |
| ≥2 candidatos | gana `‖mag‖` mayor (desempate: menor `feature`). Ganador `COMMIT` o `ABORT` por política. Perdedores `OVERFLOW` | ganador según fila de arriba; perdedores 0 + aux | un COMMIT/ABORT + n OVERFLOW |

`ABORT` no es "slot vacío". `ABORT` es "hubo candidato y la política lo rechazó". El silencio no es un no-hecho journalizable: no hubo acto. Eso distingue E8 de un profiler.

**Reglas de ABORT de fase 1, instanciadas (C7c):**

```
abort_rules (POLICY p_fase1):
  nonfinite(mag)        → ABORT reason=nonfinite
  mag == 0              → ABORT reason=null_effect    # DEROGADA por v2.2 C8: choca con ley 1
  |mag| > mag_max       → ABORT reason=mag_overflow   # mag_max en POLICY
```

Sin la primera regla, un NaN se commitea, corrompe `h`, y T1 lo **reproduce fielmente** en vez de detectarlo: el replay verifica consistencia, no cordura. La cordura es política.

**`SPEC_AMEND` (v2.2 C8):** la regla `null_effect` era un error de esta corrección C7c — con zero-init (ley 1) todos los `mag` iniciales son 0; abortarlos vacía el mask de backward y el camino de escritura no despierta jamás. v2.2 §1 la deroga: el cero es COMMIT, identidad bit-idéntica por regla, elidido del WAL por defecto. `nonfinite` y `mag_overflow` sobreviven.

### 1.4 Presupuesto de escritura

Por token, por capa:

```
#commits ≤ min(k, S)
#OVERFLOW ≤ max(0, k − #commits_y_aborts_de_slots_contendidos)
#hechos journalizados = #commits + #OVERFLOW + #ABORT
```

Con home `f mod S`, cada hoja `d_slot` queda cuantizada a un código del subdiccionario `⌊N_f / S⌋` (más el resto euclidiano en los primeros `N_f mod S` slots). Eso *es* la capacidad de escritura. No es un denso disfrazado.

### 1.5 Backward de `pack`

- `mag` es continuo → gradiente directo al score, por STE a través del bf16 del binding (§3.4).
- Índice `feature` es discreto → STE: el forward usa `Ĉ[f]`; el backward deposita el gradiente de la hoja en `C[f]` (a través de la normalización, que es diferenciable) y no en las otras filas.
- Top-k y colisión son no diferenciables → mask duro sobre el conjunto **commiteado**, no sobre el conjunto top-k. v2 §3.4 manda; el comentario del loop de v2 queda enmendado (ya aplicado en v2).
- Aux de balanceo (MoE-style) sobre la distribución empírica de `feature` y de `slot` en el microbatch. Coeficiente prerregistrado. Sin aux, OVERFLOW es ruido no estacionario.
- **Métrica de energía (C1):** además de la entropía de frecuencias, se monitorea `Σ|mag|` por feature — con `apply` normalizado, la energía del catálogo es visible en los bindings mismos, sin consultar `‖C‖`.

### 1.6 Lo que fase 1 no es

No es un SAE sobre activaciones post-hoc. No es un router con softmax aprendido. No es VQ sobre `d_model` entero. Es VQ por subespacio, k-sparse, con colisión explícita.

Cambio de métrica, de home, o de k = cambio de política = `SPEC_AMEND` + nuevo manifiesto. No es un hotfix de training.

---

## 2 · Ley 0 — el bloque no escribe `h` (A2)

### 2.1 Enunciado

```
Ningún op dentro de layer recibe h mutable.
layer : h ↦ delta_hat
h'    = apply(h, allocate_local(pack(delta_hat, C_step)))
```

`h` entra por valor (o por referencia read-only). Cualquier `+=` anónimo dentro de attn, mlp, norm residual, dropout-on-residual o fused block es violación de invariante, no "detalle de framework".

### 2.2 Consecuencias de implementación

- El bloque fase 1 es **parallel block funcional** (GPT-J/PaLM): LN lee `h`; attn y mlp leen el normalizado; ambos contribuyen a `delta_hat` por suma fp32 canónica *en un buffer propio*; ese buffer es la entrada de `pack`. **Esto es una elección de topología, no un accidente de la ley 0 — y el baseline lo comparte (C2, §6.4).**
- No hay `h = h + attn(...)` intermedio. Si se quiere residual interno attn-antes-de-mlp, es otro objeto: o se nombra como segundo `apply` (dos catálogos por capa, manifiesto distinto) o no existe.
- Autograd ve `h` como input del grafo y `h'` como output de `apply`. No hay in-place sobre el tensor que T1 va a reconstruir.
- Tests de harness (obligatorios, no "nice to have"):
  1. **Snapshot bit a bit (C7a):** `snap = h.clone()` pre-layer; post-layer, `h` y `snap` comparados por bytes. Nada de `id()` — la identidad de objeto Python no prueba no-aliasing de storage en torch.
  2. Un único writer de `h` en el stack: el símbolo `apply`.
  3. El mismo test en CUDA y en ROCm.

### 2.3 Leyes, ahora cinco

| Ley | Enunciado |
|---|---|
| 0. Funcional | `layer` propone. No escribe `h`. |
| 1. Identidad | Camino de escritura zero-init; `H(h_0)` ancla del microbatch. |
| 2. Delta nombrable | Toda escritura a `h` es un binding `(slot, feature, mag, policy_id)`. |
| 3. Overflow visible | Colisión = hecho. Silencio (slot no propuesto) ≠ OVERFLOW. |
| 4. Commit o silencio | Solo `apply()` escribe `h`. Durable/export/cita exige receipts. |

---

## 3 · `apply` — semiring de software, no la ALU del vendor (A3)

### 3.1 Lo que v2 no podía prometer

bf16 en CUDA y ROCm promociona, fusiona y redondea distinto que un CPU. "IEEE-754 escalar sin FMA" no es lo que ejecuta el tensor core ni el packed-bf16 del driver. T1 cross-vendor sobre la ALU nativa era el nuevo E1. Este contrato lo cierra.

### 3.2 Contrato

`apply` es una función pura, implementada **una vez** (referencia C/Rust compilada al binario de verify y enlazada o reimplementada bit-idéntica en el kernel de device):

```
n̂[f]  = rnd_fp32( C_fp32[f] / (‖C_fp32[f]‖₂ + ε) )      # C1: normalización DENTRO de apply_ref
leaf' = rnd_bf16( leaf_fp32 + rnd_fp32(mag_bf16) * n̂[f] )
```

Reglas:

1. Hoja y fila de codebook se leen como bits bf16 y se **promueven a fp32** por extensión de mantissa (no por round-trip).
2. `mag` se promociona igual.
3. La norma `‖C[f]‖₂` se computa en fp32 con **reducción secuencial por índice creciente, sin FMA** (mismo orden que §1.2); `ε = 2⁻¹⁴`; la división se redondea a fp32 por componente (C1).
4. Producto y suma son fp32 IEEE, **rounding to nearest even**, **sin FMA**. El producto se redondea a fp32; después la suma; después el downcast a bf16 (round to nearest even, flush denormals-to-zero off en fase 1; si el vendor no puede, el kernel de apply no usa la ALU nativa para esta línea).
5. Slots no commiteados no se tocan: copia bit-idéntica de la hoja.
6. Orden de aplicación en un `(b, t)`: slots crecientes. Entre tokens: layout `(b, t)` row-major. Ese orden es parte del COMMIT receipt de la capa.

El kernel de device puede vectorizar solo si el resultado es bit-idéntico a esta referencia. Si no puede, corre escalar. Occupancy no autoriza otro redondeo.

Con C1: `‖leaf' − leaf‖ = |mag|` a redondeo bf16. El binding se autodescribe — citar un hecho ya no requiere conocer `‖C[f]‖`.

### 3.3 T1 es same-binary

El verificador CPU enlaza **la misma referencia** `apply`. No reimplementa "bf16 a ojo".

- Reproducibilidad **prometida**: `apply_ref(h, bindings, C_step)` en el proceso verify == `apply_device` del run, bit a bit, porque ambos son la referencia.
- Reproducibilidad **no prometida**: génesis de `delta_hat` entre NVIDIA y AMD; matmuls del bloque; TF32/tensor cores. El binding sigue declarando `device`.
- Si un backend no puede hospedar la referencia, ese backend no entra al harness. Elegir vendor por ALU es exactamente lo que la spec prohíbe.
- **Identidad del binario (C5):** el manifiesto registra `build.artifact_hash` del trainer y del verificador; `RUN_START` lo cita. "El invariante en el complemento es el binario" solo es citable si el binario es un hecho.

### 3.4 Qué bits viajan en el binding

```
binding {
  id, slot, feature,
  mag:        bits bf16 aplicados,     # no el fp32 interno
  policy_id:  id del objeto de política commiteado,
  step_id,
  device,
  # C4: `prev` es OPCIONAL (perfil t1_5, junto con H(pre_leaf)).
  #     El encadenamiento canónico es T2 por capa; la posición en la
  #     cadena + el orden total de §3.2 regla 6 identifican el hecho.
}
```

`mag` en el hecho es el bf16 que `apply` multiplicó. Si el encoder pensó en fp32 y downcasteó, el binding guarda el downcast. Verify no adivina. Binding mínimo: `β ≤ 32` bytes; perfil `t1_5`: `β ≤ 96` (añade `prev` + `H(pre_leaf)`).

---

## 4 · Cadena del paso: `policy_id` y `C` sin excepciones (A4)

### 4.1 Política como objeto

La política de allocate fase 1 es un record Hyphae, no un `if` en el kernel:

```
POLICY  policy_id=p_…  step_from=t0  k=…  home=mod_S
        metric=cosine_l2normed  tie=feature_then_slot
        reduction=seq_fp32_nofma  mag_max=…
        abort_rules=[nonfinite, null_effect, mag_overflow]
        schema_hash=…
```

El espejo en device carga exactamente esos campos. Cambiar k, métrica, home o abort_rules = insertar un `POLICY` nuevo. Bindings posteriores citan el `policy_id` nuevo. Un binding sin `policy_id` es inválido.

### 4.2 Codebook dentro de `θ_C`

`C` es estado real. Toda mutación de `C` entra en la cadena del paso.

Fase 1 elige **un** mecanismo y lo escribe en el manifiesto. No los dos.

| Mecanismo | Forward | Update de `C` | Qué hashea el `STEP` |
|---|---|---|---|
| STE + optimizer | `apply` usa `Ĉ_t[f]` | `opt.step()` sobre `C` como parámetro | `H_canónico(C_{t+1})` |
| EMA estilo VQ-VAE | `apply` usa `Ĉ_t[f]` | `C_{t+1} = (1-λ)C_t + λ batch` **antes** del `STEP`, como acto explícito `CODEBOOK` | `H_canónico(C_{t+1})` más el record `CODEBOOK λ …` |

Prohibido: EMA silencioso en un hook de PyTorch que no emite hecho. Si `C` se mueve y el `STEP` no lo cubre, T1 de un microbatch posterior verifica contra el diccionario equivocado y el invariante miente.

### 4.3 `STEP` enmendado (con C3)

```
STEP     step_id=t  lr=…  grad_norm=…
         policy_id=p_…
         c_prev=H_canónico(C_t)  c_next=H_canónico(C_{t+1})
CODEBOOK step_id=t  mechanism=ste|ema  λ=…        # solo si ema
SEAL     step_id=t  w=H_canónico(θ_t \ C)          # cada seal.every_k_steps, y siempre pre-EXPORT
```

- `H_canónico(X)` = SHA-256 de `shape || dtype || raw_row_major_little_endian(X)`. Nada de pickle.
- **`C` se hashea cada paso** (chico — p.ej. 16k × 12 × 2 B = 384 KB, <1 ms — y es el diccionario exacto que T1 necesita).
- **`θ` se sella cada `seal.every_k_steps`** (C3): hashear cientos de MB por paso destruye el presupuesto de §5. Entre sellos, la mutación de `θ` está evidenciada por la cadena de STEPs (lr, grad_norm, orden), no hasheada. Límite declarado, no escondido.

Fuera de alcance fase 1 (declarado, no olvidado): estado Adam `(m, v)`, RNG del sampler, orden del dataloader, y verificación de `θ` **entre** sellos. Un resume verificado no es un entregable de esta fase. El `.pt` exportado *de pesos y codebook* sí es el último eslabón de `SEAL` y `c_*`.

### 4.4 Export

El artefacto citable no es el `.pt` crudo. Es:

```
EXPORT  step_id=t  w=H_canónico(θ)  c=H_canónico(C)  head=…  receipt=…
```

El fichero de tensores se serializa en un layout documentado (longitud, dtype, row-major, little-endian). Si alguien reexporta con `torch.save` y el hash cambia, el export no está en la cadena: es otro objeto.

---

## 5 · Presupuesto de WAL y D2H, en el manifiesto (A5)

### 5.1 Cota, no prosa

Sea `B` el batch del paso (tokens totales = `B_seq × T`), `L` capas, `k` de §1, `β` bytes del record de binding empaquetado (fase 1: `β ≤ 32` mínimo, `β ≤ 96` perfil `t1_5` — C4).

```
N_hechos_fwd ≤ L × N_tokens × k          # COMMIT+OVERFLOW+ABORT
bytes_bindings ≤ N_hechos_fwd × β
bytes_step     = bytes_bindings + bytes(STEP) + bytes(anchor)          # C7d
               + bytes(POLICY si cambió) + bytes(CODEBOOK si ema)
               + bytes(SEAL si toca)                                    # C3
```

D2H: el ring buffer transporta el packed binding **y el anchor `H(h_0)`** (C7d), no el hidden. Hidden crudo solo viaja en microbatches auditados (§6).

### 5.2 Campos obligatorios del manifiesto

```
wal.max_bytes_per_token      # cota de bytes_bindings / N_tokens
wal.max_bytes_per_step       # cota de bytes_step
d2h.max_bytes_per_step       # igual o estrictamente mayor; incluye framing y anchor
d2h.overlap                  # "backward" | "none"  — fase 1: backward
barrier.budget_ms            # espera de receipts; fail-stop si se excede
seal.every_k_steps           # C3: cadencia del SEAL de θ
t1.sample_rate               # ver §6
t1.max_h0_bytes_per_day      # spill de h_0
build.artifact_hash          # C5: binario del trainer y del verificador
baseline.block               # C2: "parallel" — el denso comparte topología
```

Si el run viola `wal.max_*` o `d2h.max_*`, el harness **mata la rama**. No se "optimiza después". El número se elige antes del primer step, a partir de `L, k, β, N_tokens` del budget de GPU.

### 5.3 Ejemplo de cota (no es el umbral; es aritmética)

`L=12`, `T=512`, `B_seq=4` → `N_tokens=2048`, `k=4`:

```
N_hechos_fwd ≤ 12 × 2048 × 4 = 98 304
β=32 (mínimo):  bytes_bindings ≤ 3.0 MiB / paso
β=96 (t1_5):    bytes_bindings ≤ 9.0 MiB / paso
```

Con `k=8`, `L=32`, `N_tokens=8192`, β=32 ya son ~64 MiB/paso (192 MiB en t1_5). Por eso k, L y el perfil entran al manifiesto juntos: no se escala el modelo y se finge el mismo WAL.

### 5.4 Fail-stop de la barrera

`wait_receipts(step)`:

- Timeout = `barrier.budget_ms` prerregistrado.
- Si Hyphae no entrega el `CommitReceipt` del paso: **abort del run**, dump del ring buffer a spill, no-op sobre `opt.step()`. No hay degradación a "seguir sin ledger".
- Reanudación = otro run con manifiesto hijo que cita el `head` último commiteado. No hay "retry del paso" en fase 1.

Hyphae 2.2.0 es un proceso, un data directory, lock de OS. El harness no inventa un cluster. Un escritor de training por instancia.

---

## 6 · T1 es un test, no un oráculo (A6)

### 6.1 Qué prueba T1

T1 prueba:

```
apply_ref(h_0, bindings_commiteados_en_orden, C_step)  tiene hash  head
```

Eso es consistencia **apply ↔ journal ↔ codebook del STEP**. No prueba que todos los microbatches no auditados fueran honestos. Esa honestidad, en los no auditados, es *por construcción del binario* (ley 0 + un solo `apply` + `build.artifact_hash` citado en `RUN_START` — C5), no por observación.

### 6.2 Protocolo

| Pieza | Regla fase 1 |
|---|---|
| Unidad auditada | microbatch |
| Tasa | `t1.sample_rate = 1/N`, `N` prerregistrado, `N ≥ 1` |
| Selección | `hash(run_id, step_id, microbatch_id) mod N == 0` — no un flag del trainer |
| **Modelo de amenaza (C6)** | **Fase 1 audita fallos de implementación, no adversarios.** La selección es predecible desde el step 0; un trainer malicioso podría desviarse solo en el complemento. Auditoría adversarial (commit-then-reveal de la semilla de selección) = fase posterior, nombrada aquí para no sobrevender. |
| Spill | `h_0` crudo bf16 del microbatch (~ `N_tokens_mb × d_model × 2` B) + stream de bindings + `c_next` del STEP |
| Verify | proceso aparte, CPU, misma `apply_ref` (same-binary, C5), sin GPU |
| Fallo | cualquier bit de diferencia → mata el run, localiza con T1.5 si el perfil lleva `H(pre_leaf)` |
| T2 | siempre, todos los pasos: cadena `head_{ℓ+1} = H(head_ℓ ‖ bindings_ℓ ‖ meta)`. Adulteración de 1 bit del WAL de bindings se detecta al 100 % *del WAL persistido*. Eso no se vende como T1 universal |

### 6.3 Lo que el manifiesto tiene que decir en una línea

> T1 cubre la fracción `1/N` de microbatches. El invariante en el complemento es el binario citado en `build.artifact_hash`, no la muestra. Un T1 verde no es una prueba de que el campo entero fue observado, ni de que un adversario no existió (C6).

### 6.4 Métricas que sí falsan (hereda E13, con las correcciones)

| Métrica | Corrección v2.1 |
|---|---|
| Gap de perplexity vs denso a FLOPs iguales | **El denso es parallel-block (C2)** — misma topología funcional, suma densa sin catálogo. FLOPs cuentan `layer` + `pack` + `apply_ref`; si `pack` barre `N_f` scores en device, eso entra. Umbral X% prerregistrado. |
| T1 detecta 100 % de las mutaciones inyectadas en `apply` o en el stream *de la muestra* | No se reclama sobre el complemento. Inyección de fallos = parte del harness, no del run. |
| T2 detecta 100 % de flips de 1 bit en el WAL persistido y localiza el eslabón | Integración del ledger. No mide el modelo. |
| Jaccard de bindings seed-a-seed | Solo después de alineación de codebook (Hungarian sobre filas por slot hogar, o no se reporta). Sin alinear, la métrica es ruido de permutación. |
| Entropía de uso de `C` / perplexity del codebook | Colapso = mata la rama. Umbral prerregistrado. **Más energía por feature `Σ|mag|` (C1): el colapso por norma ya no es invisible.** |
| Overhead de pared (D2H + barrera + journal + SEAL) | Contra `d2h.max_*`, `barrier.budget_ms` y el costo amortizado del SEAL (C3). |
| Humano cita un binding y reencuentra su proof | Conservado. Con C1, el binding además se autodescribe: `|mag|` es la magnitud real del hecho. |

---

## 7 · Loop enmendado

```python
for batch in data:
    h = embed(batch)                              # no escribe residual de capa
    commit_async(H(h), tag="anchor")              # ley 1; cuenta en d2h.max (C7d)
    verd_all = []
    for layer in model:
        snap = h.clone()                          # C7a: snapshot bit a bit
        delta_hat = layer(h)                      # ley 0: h intacto
        assert_bytes_equal(h, snap)               # harness test 1 (debug/CI)
        cand = pack(delta_hat, C_step, k, policy) # §1; reducción fijada
        verd = allocate_local(cand, policy)       # espejo device; abort_rules §1.3
        h    = apply(h, verd.committed)           # §3, único writer, Ĉ (C1)
        journal_async(verd)                       # packed bindings, no hidden
        verd_all.append(verd)
    loss = head(h) + aux_balance(verd_all)
    loss.backward()                               # mask = committed, no top-k
    wait_receipts(step)                           # fail-stop §5.4
    maybe_ema_codebook(C)                         # solo si manifiesto = ema; emite CODEBOOK
    opt.step()                                    # ilegal sin receipts
    commit_step_chain(C, policy_id)               # §4.3: STEP con c_prev/c_next
    if step % seal_every_k == 0:
        commit_seal(θ)                            # §4.3 SEAL (C3)
    if t1_selected(step, microbatch):
        spill_h0_and_bindings(...)                # §6
```

Un solo código de costura. Cero `if nvidia / if amd` alrededor de `apply`. El device del binding es un campo, no una rama de control. (`assert_bytes_equal` corre en CI y en runs de validación; en producción de fase 1 puede muestrearse a la misma tasa que T1.)

---

## 8 · Orden de construcción, actualizado

El §VI de v2 se conserva. Se insertan, *antes* del primer run que pueda perder:

0. Ley 0 en el bloque + test de no-mutación de `h` **por bytes** (C7a).
1. Referencia `apply_ref` **con normalización interna** (C1) + kernel device bit-idéntico, o el backend no entra.
2. Schema del binding con `policy_id`, `mag` bf16 aplicado, `prev` opcional (C4).
3. `pack` fase 1 (§1) con k, métrica y **orden de reducción** (C7b) del manifiesto.
4. Record `POLICY` con `abort_rules` instanciadas (C7c) + espejo.
5. `STEP`/`CODEBOOK` con `H_canónico(C)` por paso + `SEAL` de `θ` por cadencia (C3).
6. Presupuestos `wal.*`, `d2h.*`, `barrier.budget_ms`, `seal.every_k_steps`, `t1.sample_rate`, `build.artifact_hash`, `baseline.block` escritos **antes** del step 0 (C2, C3, C5).
7. Verificador T1 same-binary + T2.
8. El mismo harness en CUDA y ROCm.

Sin 0–3 no hay experimento. Hay un ledger esperando un modelo que todavía hace `+=`.

---

## 9 · Lista negra, entradas nuevas

Además de la de v2:

- `apply` nativo del vendor "porque es más rápido" sin match bit a bit con `apply_ref`.
- EMA o `register_buffer` de `C` que no pasa por `CODEBOOK`/`STEP`.
- Binding sin `policy_id`.
- **Aplicar `C` crudo cuando el score usó `Ĉ` (C1): el hecho deja de autodescribirse.**
- **Comparar contra un baseline secuencial cuando el bloque es paralelo (C2): el gap no atribuye.**
- **Hashear `θ` completo por paso "para más seguridad" (C3): destruye el presupuesto que §5 protege.**
- Reportar Jaccard de features sin alinear el codebook.
- Vender T1 1/N como observación del residual en todo el run — **o como defensa adversarial (C6)**.
- Residual interno attn→mlp sin un segundo `apply` declarado.
- Contar FLOPs del baseline omitiendo el barrido de `pack`.
- Serializar el export con pickle y llamarlo eslabón de la cadena.
- **Test de no-mutación por `id()` de Python (C7a): no prueba nada sobre storage.**

---

## CIERRE

v2 hizo que el papel no mintiera sobre hashes, occupancy y gradientes. El borrador del addendum hizo que el primer run no mintiera sobre qué se escribe, con qué redondeo, bajo qué política y con qué fracción del residual se observó. Esta versión hace que el hecho no mienta sobre sí mismo: `|mag|` es la magnitud real del delta (C1), el baseline mide el catálogo y no la topología (C2), la cadena de pesos cabe en el presupuesto que jura respetar (C3), y lo que descansa en el binario cita al binario (C5).

Lo que falta para perder de verdad: el schema con `policy_id`, `apply_ref` con normalización interna, `pack` con k y reducción fijos, el manifiesto con las cotas de §5 y §6, y un bloque que no toque `h`.

| Campo | Valor |
|---|---|
| Título | El medio que falta · addendum v2.1 (corregido) |
| Tipo | Spec de costura / no es un paper |
| Fecha | 29 de agosto de 2026 |
| Padre | `el_medio_que_falta_v2.md` |
| Base | borrador `~/Downloads/el_medio_que_falta_v2_1_addendum.md` + correcciones C1–C7 |
| Enmienda | A1–A6 sobre pack, ley 0, apply_ref, cadena C/policy, WAL, T1 — con C1 (geometría de apply), C2 (baseline parallel), C3 (SEAL de θ), C4 (prev opcional), C5 (hash del binario), C6 (modelo de amenaza), C7 (reducción, bytes-equal, abort_rules, anchor) |
| Invariante | Sin cambio |
| Siguiente acto | `apply_ref` (con Ĉ) + bloque funcional + schema del binding con `policy_id` |

> Un sistema no es confiable porque dice lo correcto. Es confiable cuando sus límites, su historia y su evidencia pueden examinarse. — aplicado dos veces: los seis agujeros de v2 quedaron en A1–A6; los siete del borrador quedan en C1–C7. Ninguno en una nota al pie.
