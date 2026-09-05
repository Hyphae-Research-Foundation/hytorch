# El medio que falta — v2.2 addendum (corregido)

**Tres parches de manifiesto sobre v2.1 corregido, y los flecos que todavía podían hacer mentir al primer run.**

- **Qué es:** cierre de C8–C15, con las correcciones D1–D5 aplicadas y journalizadas. No relitiga E1–E13 ni C1–C7.
- **Qué no es:** una lámina nueva, un paper, ni un cambio de tesis.
- **Base:** `el_medio_que_falta_v2.md` + `el_medio_que_falta_v2_1_addendum.md` (corregido) + borrador `~/Downloads/el_medio_que_falta_v2_2_addendum.md`.
- **Estado:** si choca con v2.1 en C8–C15, gana este documento y el choque se journaliza como `SPEC_AMEND`.

---

## 0.a · CORRECCIONES sobre el borrador (D1–D5)

Primero lo honesto: **C8 corrige un error introducido por la corrección C7c** de v2.1. La regla `mag == 0 → ABORT reason=null_effect` era mía, y el diagnóstico del borrador es correcto: con zero-init (ley 1), todos los `mag` iniciales son 0; abortarlos vacía el mask de backward, STE nunca llega a `mag`, y el camino de escritura no puede dejar de ser cero. ReZero arranca precisamente porque aplica `α·F` con `α=0` y `∂L/∂α` vive. C7c habría producido un ledger impecable de un modelo congelado en el step 0. La regla queda derogada; `nonfinite` y `mag_overflow` sobreviven.

Dicho eso, el borrador introducía dos errores de bits, un gap de schema y dos imprecisiones:

| # | Error en el borrador | Por qué es un error | Arreglo aquí |
|---|---|---|---|
| D1 | «T1 no lo necesita: replay sin ese binding = replay con él» — la elisión vendida como bit-segura por accidente IEEE | Falso con ceros con signo. `mag=0` da componentes de delta `±0.0` según el signo de `n̂`; IEEE round-to-nearest: `(-0.0) + (+0.0) = +0.0`. Si la hoja contiene `-0.0`, el device (que **sí** aplica el zero-commit) produce `+0.0`; el WAL lo elide; el verificador no lo aplica y conserva `-0.0` → hash distinto → **T1 mata un run honesto**. La spec exige «cualquier bit de diferencia → mata el run»: no puede depender de que ningún leaf sea `-0.0`. | `apply` define `mag_bf16 == ±0` como **copia bit-idéntica por regla**: la suma no se ejecuta (extensión de la regla «slots no commiteados no se tocan»). El zero-commit es COMMIT semántico, identidad aritmética *por definición, no por accidente*. Con eso la elisión es bit-segura siempre. Además: la elisión ocurre **antes** del hash T2 — la cadena `head_{ℓ+1}` cubre los hechos persistidos, o T2 fallaría por su cuenta. Ver §1. |
| D2 | §2.3: feature muerto si «`‖C[f]‖₂ < min_norm` al cierre del paso» | Bajo `post_step=renorm`, la norma al cierre del paso es ≈1 **siempre** (acaba de renormalizarse): la condición sería incumplible y el reset por norma, letra muerta. El pseudocódigo de §2.1 ya implica lo correcto; el enunciado no. | La condición se evalúa sobre la norma **pre-renorm**, medida inmediatamente después de `opt.step()` / update `CODEBOOK` y antes de renormalizar. Ver §2.3. |
| D3 | `f16 mag` en `BindingMin` | `f16` es IEEE binary16 (5 bits de exponente); `mag` es **bf16** (8 bits de exponente) en toda la spec: «los bits bf16 que `apply` multiplicó». Guardar bits bf16 en un campo declarado f16 — o peor, convertir — es corrupción de wire o pérdida de rango de exponente. Exactamente la clase de bug que «verify no adivina» prohíbe. | El campo es `u16 mag_bf16` — bits crudos con interpretación declarada bf16 en el schema. Ver §6. |
| D4 | `BindingMin` sin veredicto ni `reason` | El struct de 16 B no puede representar OVERFLOW ni ABORT(`nonfinite`\|`mag_overflow`): el packed record era solo-COMMIT y los no-hechos — «el no-hecho es dato» — no tenían wire format. Los 2 bytes de `_pad` estaban libres. | `u8 verdict` (COMMIT=0, OVERFLOW=1, ABORT=2) + `u8 reason` (0=none, 1=nonfinite, 2=mag_overflow, 3=policy) en lugar del pad. 16 bytes intactos, sin padding muerto. Ver §6. |
| D5 | Menores: (a) dinámica de init no declarada; (b) cota C14 con `≤` donde fase 1 da igualdad; (c) `slot` redundante sin nota | (a) En el step 0 todos los scores son exactamente 0 (delta_hat=0); el desempate determinista (menor feature, menor slot) comitea **los mismos features 0..k−1 para todos los tokens**: distribución maximalmente desbalanceada que el aux ve de golpe. No es un bug — es una consecuencia del empate total — pero callarla invita a «arreglarla» con un hotfix. (b) `pack` emite exactamente `k` candidatos por token (top-k global con desempate total); la cota es igualdad en fase 1. (c) Con `σ(f)=f mod S`, `slot` es derivable de `feature`: un byte tautológico según el propio criterio de C13. | (a) Declarada en §1.5: el aux la corrige; si un run quiere diversidad de arranque, `write_path.init=eps_bf16` o tie-break por hash son **otro POLICY**, no un hotfix. (b) §7: `#candidatos = k` en fase 1 (`≤ k` solo si `N_f < k`, que el manifiesto prohíbe). (c) §6: `slot` se **conserva deliberadamente** — estabilidad de schema para routing aprendido (fase 2), verificable como invariante `slot == feature mod S` en fase 1; la tautología se declara, no se esconde. |

---

## 0 · Alcance

v2.1 C1–C7 dejaron el hecho autodescrito, el baseline atribuible y el WAL pagable. Quedaban un choque de leyes, una invariancia de escala, un hash de binario que no pinneaba nada, y cuatro imprecisiones de harness.

| # | Error que quedaba | Por qué es un error | Cierre |
|---|---|---|---|
| C8 | `mag == 0 → ABORT reason=null_effect` (C7c) vs ley 1 | Zero-init ⇒ los primeros `mag` bf16 *son* 0. ABORT saca el candidato del mask; STE no llega a `mag`; el camino de escritura no despierta jamás. | El cero es COMMIT. Identidad bit-idéntica por regla (D1). Elidido del WAL por defecto. §1 |
| C9 | C1 normaliza en `apply` y deja `‖C[f]‖` libre | Forward (salvo `ε`) invariante a la norma: el optimizer puede inflar `C` (drift de `H_canónico(C)` que no es el modelo) o matarla (`‖C‖→0` ⇒ `Ĉ≈C/ε`, dirección basura). Entropía y `Σ\|mag\|` no ven un muerto por norma. El Jacobiano `(I−ûûᵀ)/‖c‖` explota cerca de 0. | `codebook.post_step=renorm` + `codebook.min_norm` en `POLICY`; reset como hecho `CODEBOOK_RESET`. §2 |
| C10 | `build.artifact_hash` como un solo SHA | El trainer no es un binario: un hash «del proceso Python» no pinnea `apply_ref`. El complemento de T1 seguía siendo prosa. | Cuatro campos citables en `RUN_START`. §3 |
| C11 | Hueco entre SEALs tratado como si el `STEP` sellara `θ` | Entre `seal.every_k_steps`, T2 no pesca un `opt.step()` distinto que loguee los mismos escalares. Misma amenaza que C6, no dicha. | §4: entre sellos, el modelo de amenaza es C6. |
| C12 | Un solo `apply` por capa sin decir qué nombra el hecho | En parallel block, `delta_hat = attn(LN(h)) + mlp(LN(h))`: contribuciones opuestas se cancelan *antes* de `pack`. Fase 1 no atribuye subcapa. | §5: el hecho nombra la suma. Segundo `apply` = otro manifiesto. |
| C13 | `β ≤ 32` sin wire struct | El id-string de v2 no cabe en 32 B. El presupuesto de §5 (v2.1) era un deseo. | §6: layout packed de 16 B (con D3/D4); `policy_id` vive en el `STEP`. |
| C14 | Cota de OVERFLOW opaca (v2.1 §1.4) | Cada candidato termina en exactamente un veredicto; la fórmula con `max(0, k−…)` oscurecía una partición. | §7: `#COMMIT + #OVERFLOW + #ABORT = k` por token por capa, antes de elidir. |
| C15 | `h.clone()` + assert por capa en el loop | Pelea con occupancy si queda en always. (`id()` ya estaba prohibido: C7a.) | §8: `harness.alias_check = off \| t1_rate \| always`. |

Invariante intacto: CUDA y ROCm proponen. El único kernel con derecho a escribir `h` es `apply()`. Nada durable, exportable ni citable sin receipts.

---

## 1 · C8 — el cero no es ABORT

### 1.1 Regla

`abort_rules` de fase 1, instanciadas otra vez (deroga la lista de v2.1 §1.3):

```
abort_rules (POLICY p_fase1):
  nonfinite(mag_bf16)   → ABORT reason=nonfinite
  |mag_bf16| > mag_max  → ABORT reason=mag_overflow
  # mag_bf16 == ±0 NO aborta
```

El predicado se evalúa sobre el **bf16 del binding**, no sobre el score fp32. Host y device downcastean con la misma RNE de `apply_ref` antes de juzgar.

### 1.2 Semántica del cero (con D1)

| Capa | Qué pasa con `mag_bf16 == ±0` |
|---|---|
| `allocate_local` | `COMMIT` (entra al mask de backward) |
| `apply` | **copia bit-idéntica de la hoja, por regla — la suma no se ejecuta** (D1). No es «`h+0` que da igual»: con ceros con signo, `(-0.0)+(+0.0)=+0.0` flippea un bit. La identidad es definición, no accidente IEEE. |
| autograd | STE vivo: `∂L/∂mag` existe, igual que `α` en ReZero. El espejo simbólico del forward (`h' = h + mag·Ĉ`) conserva el gradiente aunque el kernel no toque bits. |
| journal | **elidido** si `wal.elide_zero_commits=true` (default fase 1). La elisión ocurre **antes** del hash T2: la cadena `head_{ℓ+1} = H(head_ℓ ‖ bindings_ℓ ‖ meta)` cubre los hechos persistidos (D1). |
| T1 | replay sin el zero-commit == replay con él, **porque `apply` lo define como no-op bit a bit**, no porque la aritmética lo prometa. |

Un cero journalizado es legal si el manifiesto pone `wal.elide_zero_commits=false` (perfil de debug). No es el default: no se paga WAL por la identidad.

### 1.3 Lo que queda prohibido

- ABORT del cero + mask commiteado ⇒ gradiente 0 en el camino de escritura.
- Aplicar el cero y no meterlo en el mask ("write-path sin hecho de backward").
- Evaluar reglas de abort en fp32 y abortar en device con otro redondeo.
- **Ejecutar la suma para `mag==0` "porque total da lo mismo" (D1): con `-0.0` no da lo mismo, y T1 mata runs honestos.**

Ley 1 se conserva: el camino *nace* en cero, se *aplica* en cero (como no-op declarado), y puede dejar de ser cero porque el loss lo ve.

### 1.4 `ε_init` (opcional, no default)

Si un run concreto no arranca, el manifiesto *puede* fijar

```
write_path.init = eps_bf16    # p.ej. 2⁻⁷
```

Eso es otro `POLICY`, no un hotfix. Default fase 1: init = 0, C8 como arriba.

### 1.5 Dinámica del step 0, declarada (D5a)

Con init = 0 exacto, todos los scores del step 0 son 0 y el desempate total (mayor score → menor feature → menor slot) comitea **los mismos `k` features (0..k−1) para todos los tokens**. Con `k ≤ S` caen en slots distintos (0..k−1): no hay OVERFLOW, pero la distribución de uso arranca maximalmente desbalanceada y el aux de balanceo la ve entera en el primer backward. Es la consecuencia esperada del empate total, no un bug. Si un run quiere diversidad de arranque: `write_path.init=eps_bf16` (§1.4) o tie-break por hash — ambos son **otro `POLICY`**, journalizado, no un hotfix del trainer.

---

## 2 · C9 — norma de `C` es política

### 2.1 Campos nuevos de `POLICY`

```
codebook.post_step = renorm        # fase 1: obligatorio
codebook.min_norm  = 2⁻⁸           # fp32, prerregistrado
codebook.reset     = ema_dead      # ver §2.3
```

Después de `opt.step()` o del update `CODEBOOK` (EMA), **antes** de `H_canónico(C_{t+1})`:

```
para cada f:
  n = ‖C[f]‖₂          # pre-renorm; misma reducción seq fp32 no-FMA que §1.2 / §3.2 de v2.1
  si n < min_norm: marcar f muerto (no se renormaliza; se resetea, §2.3)
  si n ≥ min_norm y post_step=renorm:
      C[f] ← C[f] / n   # bits bf16 vía rnd_bf16 de cada componente
```

`H_canónico(C_{t+1})` hashea el codebook **ya renormalizado (y ya reseteado)**. El drift de norma deja de ser un hecho falso.

### 2.2 Por qué `renorm` es el default

C1 metió `Ĉ` en `apply`. Sin `renorm`, `C` tiene un grado de libertad que el forward no usa y el hash sí. Con `renorm`, `C` y `Ĉ` coinciden salvo `ε` y el downcast: el objeto hasheado *es* el diccionario que T1 aplica (T1 sigue usando `apply_ref`, que normaliza internamente — la coincidencia es redundancia sana, no dependencia).

`post_step=none` queda reservado a un manifiesto hijo que sepa por qué lo quiere. No es fase 1.

### 2.3 Reset de muertos (con D2)

Un feature está muerto en el paso `t` si, sobre una ventana prerregistrada `codebook.dead_window` (default: 100 steps):

- no recibió ningún COMMIT con `mag ≠ 0`, **o**
- su norma **pre-renorm** — medida en §2.1, después del update y antes de renormalizar — cayó bajo `min_norm` (D2: la norma post-renorm es ≈1 por construcción; evaluada ahí, la condición sería incumplible y el reset por norma, letra muerta).

Acción, un hecho:

```
CODEBOOK_RESET  step_id=t  features=[…]  method=ema_dead
```

`ema_dead` fase 1: reescribir la fila con una muestra del batch (media de hojas del slot hogar, unit-norm) o con un vector aleatorio unitario de semilla `hash(run_id, step_id, f)`. El método va en `POLICY`. Sin este record, el reset es mutación silenciosa de `C` y C9 no sirve. (El verificador no recomputa el reset — data-dependent —; lo evidencia la transición `c_prev → c_next` del `STEP`, misma epistemología que `opt.step()`.)

El Jacobiano `(I−ûûᵀ)/‖c‖` deja de explotar porque `‖c‖ < min_norm` no entra a STE: el feature se resetea, no se actualiza.

---

## 3 · C10 — el binario como hecho, en piezas

`build.artifact_hash` de v2.1 se parte. `RUN_START` cita los cuatro; falta uno ⇒ el run no arranca.

```
build.apply_ref_hash     SHA-256 del objeto/so de apply_ref
                         (misma ligadura en trainer y verificador)
build.harness_commit     commit git del harness (árbol limpio, no working dirty)
build.torch_wheel        nombre+versión+digest del wheel
build.backend_wheel      nombre+versión+digest (cuda / rocm)
```

Reglas:

- Trainer y verificador **comparten** `apply_ref_hash`. Si no, T1 same-binary es teatro.
- `harness_commit` sucio ⇒ no hay `RUN_START`. El experimento nace de un árbol.
- Los wheels se pinnean por digest, no por `pip freeze` textual.
- Un quinto campo opcional `build.image_digest` si el run va en contenedor; no sustituye los cuatro.

El complemento de T1 y el hueco entre SEALs descansan en *estos* objetos, no en "el binario".

---

## 4 · C11 — entre SEALs, la amenaza es C6

§4.3 de v2.1 se enmienda con una línea de amenaza, no con más hash:

> Entre dos `SEAL`, `θ` está evidenciada por la cadena de `STEP` (lr, grad_norm, policy_id, orden) y por `build.*`. T2 no detecta un `opt.step()` distinto que loguee los mismos escalares. Modelo de amenaza: **fallo de implementación bajo el artefacto citado, no adversario** — el mismo C6. Auditoría adversarial de pesos = fase posterior (sello por paso, o Merkle de tensores), nombrada para no vender el `STEP` como hash de `θ`.

`EXPORT` sigue exigiendo `SEAL` inmediato. Un `.pt` sin `SEAL` contemporáneo no es citable.

---

## 5 · C12 — el hecho nombra la suma

En el parallel block de fase 1:

```
delta_hat = attn(LN(h)) + mlp(LN(h))     # fp32 canónico, buffer propio
pack(delta_hat)                          # un solo catálogo por capa
```

Attn y mlp que se cancelan en un slot no dejan dos hechos: dejan *menos* energía en `delta_hat`. Fase 1 no atribuye subcapa. Quien quiera esa atribución declara **dos** `apply` (dos catálogos, dos `POLICY`, manifiesto hijo). No se "inspecciona" después.

Esto va en el manifiesto:

```
layer.write_units = 1          # fase 1
layer.sublayer_facts = false
```

---

## 6 · C13 — wire struct, entonces `β` (con D3, D4, D5c)

El binding mínimo fase 1 **no lleva string de id**. La identidad es la posición en T2.

```
# packed, little-endian, β = 16 bytes (mínimo)
struct BindingMin {
  u16  feature;       # 2   N_f ≤ 32768
  u8   slot;          # 1   S ≤ 64; en fase 1 verifica slot == feature mod S (D5c)
  u8   device;        # 1   cuda=0, rocm=1
  u16  mag_bf16;      # 2   bits bf16 crudos, tal cual los multiplicó apply (D3 — NO f16/IEEE-half)
  u16  layer;         # 2
  u32  pos;           # 4   índice plano en el microbatch
  u16  cand;          # 2   0..k-1, desempate estable
  u8   verdict;       # 1   COMMIT=0, OVERFLOW=1, ABORT=2   (D4)
  u8   reason;        # 1   none=0, nonfinite=1, mag_overflow=2, policy=3   (D4)
};                    # = 16, sin padding muerto
```

- **D3:** el campo de magnitud es `u16` con interpretación declarada bf16 en el schema. `f16` (IEEE binary16, 5 bits de exponente) es otro formato: guardar ahí bits bf16 (8 bits de exponente) es corrupción de wire, y convertir es perder rango. Verify no adivina — tampoco adivina el dtype.
- **D4:** sin `verdict`/`reason`, OVERFLOW y ABORT no tenían representación packed y «el no-hecho es dato» era prosa. Los 2 bytes que el borrador gastaba en `_pad` los pagan.
- **D5c:** `slot` es derivable de `feature` bajo el home fijo de fase 1 (`σ(f)=f mod S`) — un byte tautológico según el criterio del propio C13. Se **conserva deliberadamente**: estabilidad de schema para el routing aprendido de fase 2, y en fase 1 es un invariante verificable por record (`slot == feature mod S`, se chequea en ingest). La tautología se declara, no se esconde.

Fuera del struct, en el `STEP` / head de capa (una vez por capa, no por hecho):

```
step_id, policy_id, run_id, microbatch_id
```

Perfil `t1_5` añade 32+32 bytes opcionales *después* del min:

```
struct BindingT15 {
  BindingMin base;
  u8   prev[32];         # head previo, si se quiere cadena por hecho
  u8   pre_leaf_h[32];   # H(pre_leaf) del slot tocado
};                       # = 80
```

Presupuesto:

```
β_min  = 16
β_t1_5 = 80
wal.max_bytes_per_token se calcula con el β del perfil elegido
```

Aritmética de ejemplo actualizada (sustituye la de v2.1 §5.3, que usaba β=32/96): `L=12, N_tokens=2048, k=4` → 98 304 hechos → **1.5 MiB/paso** (β=16) o **7.5 MiB** (t1_5); el caso grande `L=32, N_tokens=8192, k=8` → 2 097 152 hechos → **32 MiB/paso** (β=16). La elisión de ceros (§1.2) solo puede bajar estas cotas.

`policy_id` en cada binding era redundante con el `STEP`. C4 quitó `prev` del mínimo; C13 quita el resto de tautologías (menos la declarada en D5c). Si `k≤8`, `S≤64`, `N_f≤32768`, los anchos de arriba bastan. Subir `N_f` o `S` es cambio de schema = `SPEC_AMEND`.

---

## 7 · C14 — una cota que se puede leer (con D5b)

Por token, por capa, **antes** de elidir ceros:

```
#COMMIT + #OVERFLOW + #ABORT = #candidatos = k        # fase 1 (D5b)
#hechos_persistidos = k − #ceros_elididos
```

Cada salida de `pack` recibe exactamente un veredicto. No hay cuarto cajón. La igualdad es de fase 1: `pack` emite exactamente `k` candidatos por token (top-k global con desempate total); `< k` solo sería posible con `N_f < k`, que el manifiesto prohíbe. La línea de v2.1 §1.4 sobre OVERFLOW queda sustituida por esta.

---

## 8 · C15 — alias check es un perfil

```
harness.alias_check = off | t1_rate | always
```

| Valor | Cuándo | Qué hace |
|---|---|---|
| `always` | CI | `snap = h.clone()` pre-layer; compara bytes post-`layer`, pre-`apply` |
| `t1_rate` | run fase 1 default | misma comparación, solo en microbatches T1 |
| `off` | prohibido en CI; permitido en un manifiesto hijo que cite por qué | — |

Nada de `id()` (C7a). Nada de assert en el hot path de un run que no lo pidió.

---

## 9 · Manifiesto, campos que se añaden

Además de los de v2.1 §5.2:

```
wal.elide_zero_commits       # default true          (C8)
write_path.init              # 0 | eps_bf16          (C8)
codebook.post_step           # renorm                (C9)
codebook.min_norm            # 2⁻⁸ o el del POLICY   (C9)
codebook.dead_window         # steps                 (C9)
layer.write_units            # 1                     (C12)
layer.sublayer_facts         # false                 (C12)
binding.profile              # min | t1_5            (C13)
harness.alias_check          # t1_rate               (C15)
build.apply_ref_hash         #                       (C10)
build.harness_commit
build.torch_wheel
build.backend_wheel
```

`build.artifact_hash` monolítico queda deprecado. Si aparece solo, el harness rechaza el manifiesto.

---

## 10 · Loop, una vez más (solo lo que cambia)

```python
# ... layer loop igual que v2.1, con alias_check según perfil (C15) ...
verd = allocate_local(cand, policy)       # C8: mag==±0 → COMMIT, no ABORT
h    = apply(h, verd.committed)           # mag==±0: copia bit-idéntica por regla (D1)
journal_async(elide_zeros(verd))          # C8/D1: elisión ANTES del hash T2
# ...
opt.step()
renorm_or_reset_C(policy)                 # C9/D2: norma pre-renorm decide muerte;
                                          #        emite CODEBOOK_RESET si toca
commit_step_chain(C, policy_id)           # hashea C ya renormalizado y reseteado
```

`RUN_START` al abrir el run cita los cuatro `build.*` (C10). `SEAL` no cambia de cadencia; cambia de pretensión (C11).

---

## 11 · Lista negra, entradas nuevas

Además de v2 y v2.1:

- `mag==0 → ABORT` como regla de fase 1 (C8 — deroga C7c parcialmente).
- STE apagado sobre el camino zero-init.
- **Ejecutar la suma IEEE para `mag==0` y llamar «identidad» al resultado (D1): `-0.0` existe.**
- **Elidir zero-commits después de computar el head T2 (D1): la cadena debe cubrir lo persistido.**
- `C` sin `renorm` post-step mientras `apply` usa `Ĉ` (C9).
- Reset de filas muertas sin `CODEBOOK_RESET`.
- **Evaluar la muerte por norma sobre la norma post-renorm (D2): siempre ≈1, nunca muere nadie.**
- Un solo `build.artifact_hash` que no pinnea `apply_ref` (C10).
- Vender el `STEP` como sello de `θ` entre SEALs (C11).
- Atribuir un binding a attn o a mlp en fase 1 (C12).
- Presupuestar `β ≤ 32` sin el struct de §6 (C13).
- **Declarar `mag` como `f16` — o cualquier dtype que no sean los bits bf16 aplicados (D3).**
- **Un wire format sin veredicto ni `reason`: OVERFLOW/ABORT sin representación packed (D4).**
- `harness.alias_check=always` en un run de occupancy sin decirlo (C15).

---

## CIERRE

C1 hizo que `|mag|` fuera la magnitud del hecho. C8 hace que esa magnitud pueda *nacer* en cero sin matar el gradiente — y D1 hace que ese cero sea identidad por definición, no por fe en los ceros con signo de IEEE. C9 hace que `C` hasheado sea el `C` que T1 aplica; D2, que sus muertos puedan morir. C10 hace que "el binario" tenga cuatro objetos. D3 y D4 hacen que el wire diga la verdad: los bits correctos del `mag` y un lugar para el no-hecho. El resto es no volver a pagar WAL por tautologías ni atribuir lo que un solo `apply` no vio.

Lo que falta para perder de verdad no cambió de categoría: `apply_ref` con `Ĉ`, `renorm` y el no-op del cero; bloque que no toca `h`; `BindingMin` de §6 con `verdict/reason`; manifiesto con C8–C10 escritos **antes** del step 0.

| Campo | Valor |
|---|---|
| Título | El medio que falta · addendum v2.2 (corregido) |
| Tipo | Spec de costura / no es un paper |
| Fecha | 29 de agosto de 2026 |
| Padre | `el_medio_que_falta_v2_1_addendum.md` (corregido) |
| Base | borrador `~/Downloads/el_medio_que_falta_v2_2_addendum.md` + correcciones D1–D5 |
| Enmienda | C8–C15 (del borrador, aceptadas) + D1 (cero = no-op por regla; elisión pre-T2), D2 (muerte por norma pre-renorm), D3 (`mag` = bits bf16, no f16), D4 (`verdict`/`reason` en el wire), D5 (dinámica de init declarada; C14 como igualdad; `slot` tautológico declarado) |
| Deroga | C7c parcialmente: `null_effect` fuera; `nonfinite` y `mag_overflow` sobreviven |
| Invariante | Sin cambio |
| Siguiente acto | `apply_ref` (Ĉ + renorm + no-op del cero) + `BindingMin` con verdict/reason + `RUN_START` con los cuatro hashes |

> Un sistema no es confiable porque dice lo correcto. Es confiable cuando sus límites, su historia y su evidencia pueden examinarse. — cuarta pasada: el cero con signo, la norma muerta y el dtype del mag ya no viven en una nota al pie. Y la corrección C7c que este documento deroga queda registrada como el error que fue.
