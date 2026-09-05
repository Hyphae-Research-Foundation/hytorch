# Plan Trainium (trn2, São Paulo) — sin más paradas

*2026-09-01. Contexto: DO spot quemó dinero en preempciones e infra frágil
sin créditos. AWS nos da créditos generosos. Decisión: mover fase 5 a
Trainium2 con ventanas RESERVADAS — cero preempción por diseño.*

## Recursos (verificados hoy por API)

| Qué | Detalle |
|---|---|
| Región | sa-east-1 (única con trn2 fuera de us-east-2; quota On-Demand=12 vCPU=1 instancia, us-east-* tienen quota 0) |
| Máquina | **trn2.3xlarge**: 1× Trainium2, 8 NeuronCores v3, **512 GB HBM**, 12 vCPU, 128 GB RAM |
| Precio | CB ~$2.24/h; on-demand similar — **16× más barato que el 8×MI355X de DO** |
| AMI | ami-01129f0838bcebf2b (DL Neuron PyTorch 2.9, Ubuntu 24.04, SDK 2.31.1) |
| Key | trainium-frontier (operator key) |
| **Comprado** | **CB 96h `cr-0e1d67d2d6ab5868f`: 2026-09-03 11:30 → 09-07 11:30 UTC, $214.56 upfront (credits)** |

## Hoy no hay silicio (verificado, 3 AZ × on-demand + spot: InsufficientInstanceCapacity)

La región tiene poquísimos trn2.3xlarge y el CB del otro proyecto
(trainium-frontier-dev, expira **mañana 09-02 11:30 UTC**) tiene uno.
**Poller corriendo** (/tmp/trn2-capacity-poller.sh, cada 10 min, 3 AZs):
agarra 1 on-demand en cuanto aparezca — típicamente mañana al liberarse el
CB vecino. Ese on-demand es el banco de pruebas puente hasta nuestro CB.

## Por qué esto NO se para (diferencias estructurales vs DO)

1. **CB = capacidad reservada**: dentro de la ventana nadie nos quita la
   máquina. El supervisor/resume queda como seguro, no como rutina.
2. **96h para un run de ~30h** = margen 3× para port + smoke + run + SFT +
   evals + custodia, todo dentro de UNA ventana.
3. Deadline duro único: custodia ANTES de 09-07 11:30 UTC (checkpoints a
   EBS/S3 durante el run, no al final).

## El port (el trabajo real, empieza HOY sin GPU)

Neuron no corre nuestros kernels CUDA/HIP. Plan B ya diseñado en specs:

- **pack/allocate/apply en PyTorch puro** (ops tensoriales: norms por slot,
  top-k global con desempate (score,feature,slot), veredictos, apply):
  XLA lo compila para NeuronCores. Sin pybind, sin kernels custom.
- **El gate sigue siendo el gate**: bit-exactitud verificada contra
  ApplyRef CPU (libapply_ref.so, ya BIT_IDENTICAL en 4 silicios). El claim
  del paper se AMPLÍA: NVIDIA + AMD + **Trainium**, mismos hechos.
- Riesgos conocidos a validar mañana en silicio real:
  a) determinismo bf16 de XLA en reducciones (nuestro T2 usa reducción
     secuencial fp32-nofma en CPU — el device solo PROPONE, así que el
     riesgo real es solo la reproducibilidad del forward, no de los hechos);
  b) top-k con desempate exacto en XLA (sort estable vs nuestro orden);
  c) throughput del canal (recs → CPU una vez por step, igual que hoy).
- nanochat en Neuron: torchrun idéntico (el vecino corre
  `torchrun --nproc_per_node=8 train.py` en esta misma AMI); dynamo OFF
  (ya lo hacemos en ROCm); atención vía SDPA.

## Cronograma

| Cuándo (UTC) | Qué |
|---|---|
| **HOY** | Port: `neuron_backend.py` (pack/allocate/apply en torch puro) + test CPU bit-exacto vs ApplyRef + adaptar bridge/launcher (aws en vez de doctl). Sin gastar un dólar. |
| **09-02 ~11:30** | Poller agarra on-demand → validar port en silicio: gate bit-exacto, d4 smoke ambos brazos, d12 corto, **medir s/paso** (fija el presupuesto real del d20). Apagar al terminar (~$60-100). |
| **09-03 11:30** | **Abre nuestro CB 96h** → d20 completo ambos brazos + SFT + evals + chat journalizado, checkpoints a S3 cada 500 pasos, telemetría de canal cada 50. |
| **antes 09-07 11:30** | Custodia total (modelos+ledger+evidencia) → cerrar. Fase 4 (hallucinations) corre después en CPU/local sobre lo custodiado. |

## Presupuesto

- CB 96h: $214.56 (pagado, credits)
- Puente on-demand mañana: ~$60-100
- **Total Trainium ≈ $300** — menos que UN día de los fallos de DO.

## Registro DO (cerrado)

Volumen hytorch-p5 (2TB, $200/mes) aún vive con el vanilla d20 completo +
ledger del canal vivo. Decisión pendiente del usuario: custodiar a local y
borrar, o mantener hasta re-validar vanilla en Trainium (los brazos deben
correr en el MISMO silicio para el gap — el vanilla de DO no sirve como
twin del catalog de Trainium; sirve como referencia de sanidad).
