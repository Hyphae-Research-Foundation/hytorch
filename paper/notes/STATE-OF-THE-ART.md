# Estado del arte — ¿alguien ha llegado tan lejos? (2026-08-30)

*Investigación previa a fase 2, pedida por el usuario. Método: búsqueda por
cada pieza del sistema por separado, y por la conjunción. Fuentes primarias
verificadas (arXiv/IACR/ACL/ICML). Conclusión al final.*

## Los vecinos más cercanos, pieza por pieza

### 1. Codebook Features (Tamkin, Taufeeque, Goodman — arXiv 2310.17230, ICML 2024)
**El vecino más cercano del lado del catálogo.** Meten cuellos de botella de
cuantización vectorial en cada capa de un transformer (hasta 410M params),
producen estados ocultos sparse y discretos, encuentran códigos
interpretables (conceptos, meses, emociones) y controlan el modelo
activando códigos. Stanford/FAR AI, código abierto.

**Qué NO tienen (todo lo nuestro):**
- Es **finetuning** de un modelo ya entrenado; nuestro catálogo entrena
  *from scratch* y el twin denso comparte manifiesto.
- Reportan "modest **degradation**"; nosotros medimos **mejora** (−44.9%/−43.2%
  PPL vs twin a 20k, dos vendors) bajo umbral prerregistrado — con la
  cautela de un solo LR/seed ya documentada.
- **Cero noción de hechos**: no hay veredictos, ni OVERFLOW/ABORT (el
  espacio negativo no existe para ellos), ni receipts, ni optimizer gateado,
  ni replay verificable, ni bit-exactitud cross-vendor, ni prerregistro.
- Su selección es top-k por distancia euclidiana clásica; no hay política
  como objeto versionado ni journal de colisiones.

→ **Cita obligada en el paper** (ya citamos VQ-VAE/MoE; hay que añadir esta
como el antecedente directo del catálogo). Nuestra diferencia: ellos hacen
el bottleneck *interpretable*; nosotros lo hacemos *transaccional* — y
encontramos que además gana al denso en vez de degradarlo.

### 2. Proof-of-Learning (Jia et al. — IEEE S&P 2021, arXiv 2103.05633)
**El vecino más cercano del lado de verificación.** Loguean checkpoints +
índices de datos durante el training; un verificador re-ejecuta segmentos
elegidos y comprueba que llega "cerca" del siguiente checkpoint.

**Diferencias de fondo:**
- Su verificación es **aproximada con tolerancia** (asumen no-determinismo
  de GPU como inevitable); la nuestra es **bit-exacta** porque fijamos la
  política de bits (apply_ref) y gateamos cada silicio. Publicaron ataques
  de spoofing contra PoL precisamente por esa tolerancia (línea de trabajos
  2022-2023); nuestro replay no tiene banda de tolerancia que explotar.
- Granularidad: checkpoints de pesos, nada del estado interno. Nuestro grano
  es la ESCRITURA individual al residual (16 bytes), con el no-hecho como
  dato.
- No hay autoridad: su training corre libre y se verifica después; el
  nuestro NO AVANZA sin receipt (fail-stop probado).

### 3. zkPoT (Abbaszadeh, Pappas, Katz, Papadopoulos — ACM CCS 2024, ia.cr/2024/162)
**El extremo criptográfico.** Prueba zero-knowledge de que un modelo salió
de gradient descent sobre un dataset commiteado. Estado del arte real.

**El número que nos separa:** prover de **15 minutos POR ITERACIÓN** para
VGG-11 (10M params, batch 16). Nuestro overhead: **9.5% de pared** en un
124M a 20k pasos. Son puntos de diseño distintos: ellos compran privacidad
y succinctness a coste de 3-4 órdenes de magnitud; nosotros compramos
practicidad y hechos completos sin privacidad. La spec ya lo tenía en la
lista negra: "zkML de cada matmul" no es el objetivo.

### 4. Backpack LMs (Hewitt et al. — ACL 2023)
Arquitectura interpretable-by-design entrenada from scratch (sense vectors,
170M). Pariente en espíritu ("cambiar la arquitectura para que el interior
tenga unidades con nombre"), pero: sin ledger, sin verificación, sin
espacio negativo, y con pérdida de capacidad vs transformer equivalente.

### 5. SAEs / transcoders / crosscoders / circuit tracing (Anthropic 2024-2025)
La línea dominante de interpretabilidad. **Post-hoc por diseño**: entrenan
un segundo modelo para descomponer activaciones de un modelo ya entrenado.
Impresionante para leer el campo; cero sobre autoridad, hechos en training,
o verificación. Complementaria, no competidora — nuestra sonda semántica
lee del journal lo que ellos estiman de activaciones.

### 6. OLMo / entrenamientos "fully open" (AI2 etc.)
Publican checkpoints, datos y logs — transparencia de *artefactos*. No hay
transaccionalidad, ni receipts, ni gate del optimizer, ni verificación de
transiciones. Es apertura, no verificabilidad.

### 7. Marco regulatorio (EU AI Act, Art. 53 / Annex XI, vigente para GPAI)
No es investigación sino demanda: documentación técnica del "proceso de
desarrollo y entrenamiento" es ya obligación legal para proveedores GPAI en
la UE (agosto 2025/2026). Todo lo que existe para cumplirla son documentos
narrativos (model cards, summaries). **Nadie tiene evidencia
machine-checkable del proceso.** Esto es viento de cola de mercado, no
competencia.

## Búsquedas de la conjunción (lo nuestro completo)

Cadenas como "transactional training residual stream ledger commit abort",
"training audit trail every weight update tamper-evident", "verifiable
training hash chain every step": **cero resultados relevantes.** La
conjunción no aparece en la literatura.

## Veredicto

**Nadie ha llegado a donde estamos en la conjunción.** Existen los vecinos
por pieza — Tamkin (catálogo interpretable), Jia (verificar training),
Abbaszadeh (probar training criptográficamente), Hewitt (arquitectura con
nombres), Anthropic (leer features post-hoc) — y ninguno tiene, ni en
combinación por pares:

1. Escrituras al residual como **hechos tipados** con veredicto, incluido el
   espacio negativo (OVERFLOW/ABORT como datos de primera clase).
2. **Autoridad** real: optimizer gateado por receipts durables, fail-stop
   sin camino de degradación.
3. Replay **bit-exacto** same-binary en CPU, con gates por silicio y una
   sola política de bits en dos vendors (NVIDIA + AMD).
4. **Prerregistro ejecutable** que ya corrió dos veces (una en contra, una a
   favor) sin mover el umbral.
5. Y el hallazgo empírico: el canal disciplinado **ganando** al denso from
   scratch, donde el antecedente más cercano (finetuning) reporta
   degradación.

**Riesgos de novedad a vigilar:** (a) Tamkin et al. es prior art fuerte del
lado catálogo — el paper debe posicionarse explícitamente ("transactional,
from-scratch, with negative-space facts and receipts" vs "interpretable
finetuning bottleneck"); (b) la línea PoL/zkPoT es prior art del lado
verificación — posicionar por el punto de diseño (bit-exact facts at 9.5%
vs approximate-or-cryptographic); (c) nuestro resultado de mejora necesita
la robustez de LR sweep antes de que un revisor la ataque.
