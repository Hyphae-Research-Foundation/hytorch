# Kill-test (fase 5 gate): preemption→resume citado, PASS (2026-08-31)

**El seguro de $1000: un run spot de 27h sobrevive preempciones perdiendo
minutos, no dólares. Validado matando el droplet a mitad del run.**

## Protocolo ejecutado (8×MI355X spot, d12, 300 pasos, save-every 40)

1. Run fresco en volumen `hytorch-killtest8` (mem1). Training a 2.4 s/paso.
2. **Preempción simulada a step ~168**: `doctl droplet delete --force` sin
   aviso al trainer (equivalente duro a la preempción real). Checkpoints
   40-160 + ledger + stamps quedan en el volumen; el droplet muere.
3. Relanzamiento con MISMOS args: región pinneada a la del volumen (mem1),
   attach, stamps saltan dataset/tokenizer (~7 min ahorrados), gate
   BIT_IDENTICAL re-corre en el silicio nuevo.
4. **Resume citado**: el arm detecta checkpoint 160 + stamp del run padre,
   consulta el ledger con `hytorch-ledger-query last-head` y arranca run
   HIJO `nano-d12-catalog-r160` con RUN_START citando:
   `parent_run=nano-d12-catalog, parent_head=d815a418dc5f…, resume_step=160`.
5. Training resumió en 160 (optimizer 11 grupos + codebook + dataloader
   state), completó 300/300, `catalog rc=0`, custodia automática, teardown.

## Todo lo que el test cazó (arreglado en el mismo día)

- **Capacidad post-preempción**: mem1 sin capacidad 8× inmediatamente tras
  liberar el droplet → polling con backoff (`HYTORCH_CAPACITY_WAIT_MIN`,
  default 90 min). La capacidad volvió en <10 min en ambos ciclos.
- **Región pinneada**: volúmenes son region-locked; el relaunch fuerza
  `HYTORCH_GPU_REGIONS=<región del volumen>`.
- **Stamps con horizonte**: `base_catalog_300.done` (no `.done` a secas) —
  extender un run terminado a más pasos es RESUME, no skip.
- **Poller detecta preempción**: droplet desaparecido ≠ hiccup ssh → exit 75
  → `nanochat-supervise.sh` relanza (hasta HYTORCH_MAX_RELAUNCH).
- **Scan del ledger acotado**: last-head busca desde ITERS+100 hacia abajo,
  no desde 1M (el scan completo tardaba minutos en runs largos).

## Hallazgo colateral (upstream-reportable)

nanochat single-GPU (sin torchrun) en ROCm 7.0/gfx950: MuonAdamW se cuelga
en `_finish_gathers` (optim.py:426, `torch._foreach_copy_`) en el primer
`optimizer.step()`, GPU al 6%, para siempre. El path de 8 ranks corre
perfecto en el mismo silicio/wheel (2000 pasos de d20 el mismo día).
Consecuencia: NADA de training 1× en ROCm; kill-test y SFT van al 8×
(que es la config de fase 5 de todas formas). Costo del descubrimiento: $4.

## Evidencia

- results/nanochat/d12-catalog-b729dbbd92fa/ (run hijo: RESUME + rc=0 + 299/300 en logs)
- results/nanochat/d12-catalog-13e7b5e8d96d/ (run padre preemptado)
- Ledger en el volumen: RUN_START del hijo cita parent_head d815a418dc5f…
  (verificado en vivo antes del teardown; el volumen retiene el ledger)

## Estado del volumen tras el test

`hytorch-killtest8` (250 GiB, mem1) — se conserva unos días como plantilla
de verificación; borrar antes de fase 5 real (que usa su propio volumen
`hytorch-p5` de 2 TiB).
