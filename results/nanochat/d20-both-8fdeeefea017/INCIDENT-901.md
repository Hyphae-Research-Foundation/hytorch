# INCIDENTE step-901 (2026-09-01): kill §5.4 antes del primer checkpoint

## Qué vio el usuario
El run que iba en step ~900 (18%) desapareció y el droplet nuevo tenía 51
min corriendo desde step 0. Pregunta legítima: ¿por qué EMPEZAR DE NUEVO?

## Cronología (de los logs custodiados en este directorio)

1. Catalog corría sano: canal vivo (commit 21.9% @900, abort 0.0%), loss
   4.245, val bpb 1.319 @750 — mejor que el run del canal muerto.
2. **Step 900 = wired step** (wire_every=100): el seam ingesta 5.4GB de
   wire. dt del trainer: 32.5s (normal para wired, budget 300s sobra).
3. **Step 901: el seam nunca emitió el receipt.** Barrier agotó los 300s
   → `no CommitReceipt for step 901 — KILL THE RUN on all 8 ranks (§5.4)`.
   Kill CORRECTO por diseño: sin recibo no hay paso.
4. El script de stages siguió: vanilla base_train completó (4980/4980,
   stamped), evals vanilla OK; los stages catalog-dependientes fallaron
   (rc=1, esperado sin modelo catalog).
5. El poller detectó stage fallido → exit 75 → supervisor relanzó.
6. **El catalog murió en 901 y su primer checkpoint era en 1000**
   (save_every=1000) → no había NADA que resumir → restart desde 0.
   Eso es lo que el usuario vio.

## Causa raíz: probable, no probada (y por qué no)

**Sospechoso principal: el daemon ledger-sync que YO añadí.** Cada 300s
hace STOP del seam para el rsync delta al volumen. Tras el wired step 900
el delta era ≥5.4GB (y si Hyphae compactó tras 9 ingestas de 5.4GB, el
delta pudo ser el ledger entero reescrito). Un delta grande a un volumen
de red lento = pausa del seam > presupuesto del barrier → receipt tardío →
kill. El diseño era estructuralmente inaceptable: **una pausa de duración
NO acotada dentro de un sistema con deadline duro de 300s.**

Evidencia circunstancial: 3.4 corrió 20 wired steps idénticos SIN daemon y
jamás falló un barrier; este run pasó los wired 100-800 con el daemon (deltas
que cupieron) y murió exactamente en el wired 900.

**Por qué no es concluyente:** el seam.stderr del run muerto vivía en tmpfs
del droplet que mi propio poller destruyó tras custodiar solo /var/log; y
el ledger del run 900 (con su cadena de receipts) fue borrado del volumen
por la lógica RESET del relanzamiento ("no resumable run: clearing stale
ledger snapshot"). Dos agujeros de custodia MÍOS, ambos cerrados abajo.

## Los cinco fixes (commiteados con este doc)

1. **Pausa acotada**: el daemon limita el rsync-con-seam-parado a 60s
   (rsync --timeout + kill); si no termina, CONT y reintenta al siguiente
   ciclo (rsync es incremental — converge). Pausa máxima 60s << barrier.
2. **Sidecar de heads**: el head del último receipt se copia del
   seam.stderr a `/mnt/hyvol/stamps/catalog.last_head` cada ciclo — la
   citación del resume ya NO depende de un snapshot fresco del ledger.
3. **Barrier 600s** (belt-and-suspenders para wired steps + compactación).
4. **save_every=500**: died-at-901-first-save-at-1000 no se repite; a
   15s/paso un checkpoint cada ~2h de cómputo.
5. **Custodia pre-wipe**: antes de borrar un ledger "stale", el mount
   archiva su cadena RECEIPT/STEP (ledger-query, KBs) a
   /mnt/hyvol/incident-archive/<ts>/ — nunca más un kill sin cadáver.
   Y el poller custodia seam.stderr además de /var/log.

## Qué HAY guardado ahora mismo (custodia verificada)

- **Vanilla d20 COMPLETO** (4980 pasos, loss 2.39): modelo + checkpoints
  en el volumen, stamped (no se re-corre).
- **Ledger del intento actual** (158 pasos, canal vivo): sincronizado al
  volumen (16GB) antes de parar.
- **Logs de ambos intentos** de catalog (900-run y 158-run) en results/.
- **No existe checkpoint de catalog**: ambos intentos murieron antes del
  primer save. Los pesos de step 158 estaban solo en memoria del trainer
  (parado limpiamente, no guardables post-mortem con save_every).
- Dataset, tokenizer, eval bundle: en el volumen, stamped.

## Nota de honestidad

El §5.4 ha matado dos runs este día (293 y 901) y las dos veces el sistema
hizo lo correcto y el fallo fue MÍO alrededor del seam (pausa interactiva;
daemon sin cota). El freno funciona. El operador aprende.
