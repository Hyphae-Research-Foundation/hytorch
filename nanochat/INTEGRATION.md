# nanochat × hytorch — plan de integración (fase 3, el cierre)

*Objetivo del usuario: entrenar un modelo pequeño REAL que hable, con el
medio tipado y el ledger completo, en 8×MI355X spot ($36/h). nanochat d20
("$100 tier") habla; d12 es el ensayo barato.*

## Por qué nanochat encaja

- **d_model = depth×64.** d12 → 768 = 64 slots × 12 d_slot: EXACTAMENTE la
  forma de catálogo de fase 1. d20 → 1280 = 64 × 20 (d_slot=20). El pack
  sweep (N_f=32k) pasa de ~40% de los FLOPs del toy a ~3-6% aquí: el
  overhead de pared se desploma con la escala — a favor nuestro.
- **Bloque secuencial** (`x = x + attn(norm x)`; `x = x + mlp(norm x)`): dos
  escrituras por capa. La spec lo tiene previsto: `layer.write_units = 2`,
  dos unidades de apply POR CAPA declaradas en manifiesto (C12: "quien
  quiera esa atribución declara dos apply"). BONUS científico: por primera
  vez el journal distingue attn-writes de mlp-writes — atribución de
  subcapa que fase 1 no tenía.
- **Un solo dial (`--depth`)**: todos los demás hiperparámetros derivados.
  Nuestro manifiesto envuelve el dial y congela el resto.
- **Speedrun probado**: d20 = ~2h en 8×H100 (~$48 on-demand). En 8×MI355X
  spot ($36/h): ~$70-90 con margen. El brazo baseline es nanochat vanilla
  MISMO commit — el twin perfecto.

## Arquitectura de integración

```
nanochat GPT.Block (sequential)          hytorch law-0 wrap
────────────────────────────────         ─────────────────────────────────
x = x + attn(norm(x), ...)         →     d_attn = attn(norm(x), ...)        # propone
                                          x = HyphaeWrite(x, d_attn, C, unit=0)
x = x + mlp(norm(x))               →     d_mlp = mlp(norm(x))
                                          x = HyphaeWrite(x, d_mlp, C, unit=1)
```

- `unit` viaja en el frame header (reutilizamos `layer = 2*layer_idx + unit`
  — sin cambio de wire format; el manifiesto declara el mapeo).
- Codebook compartido entre unidades (fase 3a) o por unidad (3b, manifiesto
  hijo). Empezamos compartido: menos parámetros nuevos.

## Multi-GPU (8 ranks, torchrun DDP)

El wire ya lo soporta: `microbatch_id = rank` en el frame header.
- Cada rank spoolea sus frames (`step-S-layer-L.frame` → añade `-rank-R`).
- **Solo rank 0 corre el seam client**; la barrera es colectiva:
  `dist.barrier()` tras spool, rank 0 espera receipt, broadcast del head,
  TODOS los ranks gatean su `optimizer.step()` en el head (el optimizer es
  réplica DDP — un receipt cubre el paso global).
- T2: la cadena ordena por (layer, rank) — determinista.
- Presupuesto WAL a d20: 8 ranks × 20 capas × 2 units × (32×2048 tokens/rank
  /paso)… → ~8× el wire de fase 1 por paso. Group commit de Hyphae a 1
  tx/paso sigue sobrado; el ring D2H por rank ya existe (pinned copies).

## Los tres entregables del cierre

1. **Un chat model auditado**: nanochat d20 entrenado con el medio tipado,
   speedrun completo (pretrain + mid + SFT), CORE/ARC/GSM8K evaluados, y
   puedes HABLAR con él — cada token de su pretraining respaldado por
   hechos.
2. **El twin vanilla** mismo commit/datos/LR schedule: el gap (o la mejora,
   como en 1b) a escala de modelo real.
3. **La sonda semántica sobre un modelo que habla**: token→feature en
   vocabulario real + ablación causal (apagar el feature de "Paris" y
   preguntarle al chat por Francia — la demo que nadie puede hacer).

## Fases y costos (todo en cajas DO)

| paso | qué | dónde | costo est. |
|---|---|---|---|
| 3.0 | patch law-0 de Block + shim DDP del seam client | CPU droplet (CI) | ~$1 |
| 3.1 | d4 smoke, 1×GPU dev tier, 200 pasos, T1 verde | dev tier | ~$3 |
| 3.2 | d12 single-GPU, 2k pasos, ambos brazos (señal de gap) | H100/H200 | ~$15 |
| 3.3 | d12 8×MI355X spot, speedrun corto — valida DDP+seam multi-rank | mi355x8 spot | ~$40 |
| 3.4 | **d20 speedrun completo, ambos brazos, 8×MI355X spot** | mi355x8 spot ($36/h) | ~$150-200 |
| 3.5 | probe+ablación sobre el d20 + reporte + paper §8 | CPU droplet | ~$2 |

Escalera AMD 8x: `gpu-mi355x8-2304gb-spot` → `gpu-mi350x8-2304gb-spot` →
`gpu-mi300x8-1536gb` (od). NVIDIA fallback: `gpu-h200x8-1128gb`.

## Riesgos declarados

- nanochat usa Muon + quirks (value embeddings, logit softcap, fp8 opcional):
  el catálogo solo toca los DOS puntos de escritura del residual; Muon
  actualiza C igual que AdamW actualizaría (STEP chain agnóstica al opt).
  fp8 se desactiva para el brazo catalogado en 3a (bf16, como fase 1).
- El d20 catalogado puede perder contra vanilla a esta escala — o ganar,
  como en 1b. Umbral prerregistrado ANTES del 3.4; ambos resultados se
  publican. El twin es el MISMO repo con un flag.
- torch rocm7.0 + torchrun en el 8x AMD: validar en 3.3 antes de pagar 3.4.
