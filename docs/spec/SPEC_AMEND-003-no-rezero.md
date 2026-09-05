# SPEC_AMEND-003 · Eliminación de ReZero; ley 1 reducida al ancla (2026-08-30)

**Directiva del usuario** (2026-08-30): «ReZero aquí no pinta nada. Está bien
que Hyphae deje prueba de lo que pasa en el área negativa, pero no es
necesario ReZero. Por eso nuestros resultados están mal.»

**Análisis que la confirma.** La ley 1 de v2.2 («el camino de escritura nace
en cero, precedente ReZero/Fixup») fusionaba dos necesidades distintas:

1. **Ancla de verificación** — que `H(h₀)` selle el estado inicial del
   microbatch auditado. Esto lo da el spill/T1 y NO requiere init especial.
2. **Truco de estabilización de init** — el gate escalar `α=0`. Esto era una
   herencia de la conversación original (E6 lo introdujo contra el «init
   gaussiano como constructor») y **distorsionó los resultados de fase 1**:

   - El gate escalar comprime TODAS las magnitudes por un factor común: el
     freeze de P1 fue una interacción gate×mag_max (scores crudos crecen con
     α y revientan el bound), no una propiedad del catálogo.
   - Fuerza la dinámica degenerada del step 0 (todos los scores exactamente
     0 → mismas features para todos los tokens, D5a) — un artefacto del
     gate, no del medio tipado.
   - **Rompe la equidad del baseline (C2):** el twin denso llevaba el mismo
     gate, pero GPT-J/PaLM reales no tienen gate — ambos brazos diferían del
     parallel block estándar, y el gap medido arrastraba esa distorsión en
     una dirección no caracterizada.
   - El caso C8/D1 (mag==±0 commit + elisión) nació para proteger un camino
     zero-init que no necesitamos. La regla se CONSERVA (los ±0 pueden
     ocurrir por redondeo bf16 y la elisión sigue siendo correcta), pero
     deja de ser la vía obligada del arranque.

**Enmienda (gana sobre v2.2 ley 1 y §03):**

- Ley 1 reescrita: *«El ancla es el hecho: `H(h₀)` se commitea por microbatch
  auditado. La inicialización es estándar del campo (proyecciones de salida
  escaladas 1/√(2L), estilo GPT-2), idéntica en ambos brazos.»*
- `ParallelBlock` pierde el parámetro `gate`; `delta_hat = attn(LN(h)) +
  mlp(LN(h))` con proyecciones escaladas.
- La dinámica del step 0 declarada en D5a (empate total) deja de aplicar: con
  init estándar los scores del step 0 ya son diversos.
- C8 (±0 → COMMIT, elidido) y D1 (no-op bit a bit) se conservan sin cambio.
- Los resultados de fase 1 (P1–P3, runs citables H200/MI355X) quedan
  archivados como **fase 1a (con ReZero)** — válidos como registro de esa
  política, no comparables con fase 1b.

**Fase 1b:** mismos umbrales firmados (gap ≤10%, 20k pasos, wikitext-103),
manifiestos nuevos (`phase1b-*.json`), sin ReZero en ambos brazos, con
`selection` como campo de POLICY (`global_topk` | `slot_topk`).
