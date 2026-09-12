# Protocolo V6: bloqueo persistente y bloqueo distribuido

Estado: diseño prospectivo de un experimento exploratorio, preparado después del resultado V5 `no-control-match`. V6 identifica la versión del experimento; no afirma que exista un currículo V6 implementado. Este documento no es una ejecución ni una calificación del runner.

## Pregunta y cambio de estimando

La pregunta es si concentrar un bloqueo de escritura de forma persistente en una unidad perjudica la recuperación factual y aumenta las afirmaciones falsas, mientras la loss agregada sigue pareciendo favorable, frente a repartir interrupciones entre las unidades. Se incluye restauración para estudiar el efecto de retirar ambos regímenes dentro del mismo presupuesto de entrenamiento.

El resultado V5 permanece `no-control-match`, con objetivo/control finales nulos. U0 fue el candidato del localizador; ningún otro sitio pasó todos los calipers. No se relajan esos límites, no se sustituye U0 por otro objetivo y no se calcula una mezcla de controles para obtener un match positivo. U1 tampoco se considera un canal muerto.

El nuevo control es válido para un **contraste de políticas de bloqueo persistente/concentrado frente a bloqueo distribuido**, con asignación marginal de sitios balanceada. Cambia el estimando respecto de la antigua comparación de una unidad seleccionada con otra de daño general comparable. No promete igual daño general durante el entrenamiento ni identifica un efecto directo a loss constante. Una comparación exploratoria sin emparejamiento entre U0 y U1 no sustituye este panel.

## Panel completo

Hay 17 ejecuciones independientes desde los mismos bytes A0 dentro de un mundo/seed. Cada una realiza 4.000 updates de adquisición y recibe los mismos ejemplos, en el mismo orden. H se declara como rama experimental desde el inicio; un reference antiguo no se rebautiza H.

| Familia | Ramas | Updates `[0,2000)` | Updates `[2000,4000)` |
| --- | ---: | --- | --- |
| H | 1 | Sham vacío | Sham vacío |
| P_u, u=0..3 | 4 | Veto permanente de Uu | Veto permanente de Uu |
| D_s, s=0..3 | 4 | Veto distribuido definido abajo | Mismo calendario distribuido |
| RP_u, u=0..3 | 4 | Mismo calendario que P_u | Sham vacío |
| RD_s, s=0..3 | 4 | Mismo calendario que D_s | Sham vacío |

El primer update reactivado tiene índice 2000 y produce el estado A2001. P_u/RP_u y D_s/RD_s deben coincidir exactamente en estados numéricos y datos consumidos hasta A2000, inclusive. Se preservan sus contratos e identidades distintas. No se exige igualdad P/D después de divergir sus políticas. No se permite convertir una ejecución P o D en rescate mediante resume estricto ni compartir un prefijo sin un contrato de fork nuevo; aquí se proponen ejecuciones completas independientes.

El mismo sitio se veta en todos los forwards y microbatches de un update. Generación y diagnóstico tienen políticas separadas: un índice de token nunca selecciona el sitio de entrenamiento.

## Calendario distribuido y garantía de balance

Para t en `[0,4000)`, b=`t//4` y k=`t%4`. Sea pi_b una permutación de U0..U3 construida ordenando hashes SHA256 de una codificación JSON canónica, con dominio explícito, seed de calendario, índice b e índice de unidad; los empates se resuelven por índice. La codificación y el dominio exactos pertenecen al contrato del módulo `control_schedule.py` y al plan completo generado. No se usa el RNG del modelo, sampler, dropout ni generador de datos.

La rama D_s veta `pi_b[(k+s)%4]`. RP/RD sustituyen su máscara por el sham desde t=2000. Las permutaciones y calendarios completos se fijan antes de generar o evaluar el nuevo mundo; no se buscan seeds de calendario favorables.

En cada t, las cuatro ramas P vetan cada sitio exactamente una vez; las cuatro D también. En cada bloque de cuatro updates, cada D visita cada sitio una vez. Cada P acumula 4.000 bloqueos en un solo sitio; cada D acumula 1.000 por sitio. Durante la primera mitad, cada RD acumula 500 por sitio; cada RP acumula 2.000 en su sitio. Esta diferencia de concentración dentro de un modelo forma parte de la intervención. No se afirma que sólo cambie la autocorrelación manteniendo fija la dosis por sitio de cada modelo.

Para cualquier función f(u,x,t) evaluada en un estado sano común y el mismo input, la media de f sobre los sitios asignados a las cuatro P coincide exactamente con la media sobre las cuatro D, en cada t. La identidad vale por permutación, incluso por posición y tipo de texto. No depende de interpolar las NLL observadas ni de suponer aditividad de efectos entre sitios.

Esta prueba se limita a la asignación. Una vez que divergen los parámetros, AdamW, el codebook compartido y las representaciones, pueden diferir los COMMIT efectivos, su energía, gradientes, compensación y NLL. Esas diferencias se conservan como resultados y posibles mecanismos; no se ajustan las máscaras, se reponderan ramas ni se excluyen corridas para igualarlas.

Se evita el ciclo fijo `(t+s)%4`: el currículo actual corta una presentación auxiliar en cuatro slices mediante `t%4`, lo que acoplaría un sitio al mismo slice dentro de cada rama. Las permutaciones por bloque evitan ese acoplamiento permanente; no prueban ausencia de toda interacción temporal. La política concreta es el objeto del experimento.

## Mundo nuevo, exposición y reservas

El mundo histórico completo de discovery ya fue observado. Aunque la medición V5 no abrió validation/test, validation se puntuó en V2; no se presenta como una reserva globalmente intacta. Test tiene una reserva declarada, sin una auditoría universal de accesos. Ninguno se reutiliza como nueva confirmación.

Se propone una sola realización nueva, exploratoria, sin generación todavía:

| Identidad | Seed fijado |
| --- | ---: |
| Verdad del mundo | 2026091301 |
| Asignación de exposición/splits | 2026091302 |
| Modelo, optimizer y stream | 2026091303 |
| Calendario de bloqueos | 2026091304 |
| Orden de ejecución después de H | 2026091305 |

El generador conserva 96 entidades, dos relaciones, 16 valores, common_repeats=8, rare_repeats=1, menciones low/high=16/64 y unseen_entity_every=8. Los seeds son elecciones prospectivas, no semillas encontradas por búsqueda ni una muestra suficiente para inferencia poblacional. Compartir vocabulario y plantillas con el benchmark histórico limita la novedad: se propone una nueva asignación de hechos/exposición, no una nueva arquitectura o un corpus de lenguaje natural.

La nueva identidad completa del benchmark debe distinguirse de todas las identidades históricas conocidas antes de entrenar. Se conserva el primer mundo generado y cualquier fallo; no se reemplaza por falta de competencia o por un efecto desfavorable. El export separa TRAIN público, prompts discovery/validation/test y oracle privado. Los splits se asignan por entidad; etiquetas, respuestas reservadas y scores no llegan al modelo ni al calendario. Los hechos expuestos de entidades de validation pueden estar en TRAIN: se evalúa recuperación con prompts reservados, no generalización a hechos nunca expuestos.

Se requiere un W640/A0 nuevo bajo la fuente final: warmup auxiliar y estado anterior a adquisición del mundo nuevo. Se conservan B48/T128 y los dominios original32 + auxiliar8 + canónico8, vocabulario 428, receta de objetivo y geometría V5. El banco canónico se deriva de todos los registros públicos pertinentes del nuevo mundo. Su tamaño y las cardinalidades de cada estrato se verifican al exportar; no se fuerza el tamaño histórico 503 ni el número histórico 19 de hechos comunes.

El código V5 actual fija seed=1337 y otras identidades históricas. Adaptarlo requiere una nueva fuente y calificación; editar esas identidades en un manifest no convierte el código anterior en un runner V6.

## Semántica de la intervención

Se veta la escritura completa del sitio asignado, después de propuestas, validación y allocation. Sólo los COMMIT sobrevivientes se convierten en denegaciones experimentales. Se conserva la cuantización normal de cero commits, el residual VJP declarado, todos los frames y los abortos/overflow originales; no se promueven perdedores ni se devuelve una identidad FP32 inventada.

No se congelan parámetros. El VJP directo de esa llamada a propuestas/codebook queda bloqueado por la semántica existente, pero regularización, otras llamadas al codebook compartido, momentos AdamW y weight decay pueden mover parámetros. Bloquear escritura no aísla una intervención pura sobre plasticidad ni garantiza ausencia de conocimiento en el resto del modelo.

En cada unidad/update se retienen oportunidades totales, COMMIT antes del veto, denegaciones experimentales, overflow y abortos originales, posiciones válidas/padding, contribución borrada y denominadores. El balance del calendario no se confunde con igualdad de escrituras efectivamente borradas. La secuencia real de DATA se autentica y compara en las 17 ramas.

## Mediciones y estimandos

H se entrena primero con su contrato final. La competencia discovery conserva ≥80% common macro y superioridad estricta frente a los cuatro priors declarados en cada mapa. Si falla, termina el intento como fallo de competencia: no se cambian mundo, objetivos, horizonte o seeds para conseguir aprobación. Este gate no usa validation. Si pasa, las otras 16 ramas se ejecutan en el orden hash fijado, sin selección intermedia por resultados.

La evaluación primaria usa todos los prompts validation a A4000, política sin veto, parser y greedy fijos, máximo 16 tokens y los tres mapas identity/shift8/gaps4. Se reportan todos los estratos: comunes, raros y no expuestos. El estrato primario son todos los hechos comunes de validation, sin filtrar aciertos de H. Se comprueba su pertenencia y exposición real antes de interpretar. Test permanece cerrado.

La captura y puntuación primaria empiezan sólo después de cerrar y verificar las 17 ramas. Primero se capturan y sellan todas sus generaciones y los logits completos de 428 clases en la frontera pública de respuesta, con scaffold fijo y sin suministrar el Value correcto. Sólo después el scorer privado abre los targets validation, indexa los logits sellados y clasifica las respuestas. Ningún resultado validation retorna al calendario, entrenamiento, gate H o selección de ramas. Los fallos de captura no se ocultan con una nueva decodificación.

El export nuevo debe separar físicamente las etiquetas privadas por split. H y las curvas discovery sólo pueden acceder al paquete privado discovery; la captura pública no abre targets. El `load_world` histórico carga un oracle conjunto y no satisface esta separación por limitar luego un argumento `split`: se requiere un lector nuevo calificado. Los hashes opacos de custodia se distinguen de la lectura semántica de etiquetas. La reserva es operacional; las semillas públicas de un generador determinista no constituyen una barrera criptográfica.

Para cada rama, estrato y mapa se conservan accuracy por hecho, tasa de afirmaciones falsas, abstenciones, salidas no evaluables y Value NLL. Estas métricas factuales promedian las paráfrasis dentro del hecho y luego dan igual peso a cada hecho. Los tres mapas tienen igual peso y se reportan además por separado.

La NLL original es un outcome global separado, sin estratificación por hecho: suma de pérdidas sobre todos los tokens válidos/EOS de todas las ocurrencias TRAIN originales de entidades validation, dividida por el conteo total de esos tokens dentro de cada mapa; después se promedian los tres mapas por igual. Se conservan multiplicidades y no se sustituye esta reducción por un promedio de registros, batches o hechos. Es una métrica de la tarea y de texto conocido, no competencia lingüística universal ni un test de lenguaje nuevo. Cada componente factual y esta NLL global reciben los contrastes siguientes con su propio denominador.

Para cada componente del vector, con barras que promedian las cuatro ramas de una familia:

```
tau   = mean(P) - mean(D)
rho_P = mean(RP) - mean(P)
rho_D = mean(RD) - mean(D)
kappa = rho_P - rho_D
      = (mean(RP) - mean(RD)) - (mean(P) - mean(D))
```

H contextualiza competencia y deterioro; se reportan también las 16 diferencias individuales respecto de H. H se cancela algebraicamente en tau y kappa. Para tasas de error, tau positivo indica más error bajo bloqueo persistente; rho negativo indica mejora al retirar el bloqueo. No se invierten signos silenciosamente ni se suman efectos de distintos sitios como si fueran aditivos.

El mundo/seed es la unidad de replicación. Las 17 ramas son un panel emparejado de políticas, no 17 semillas independientes; hechos, tokens, mapas y shifts tampoco son réplicas de entrenamiento. Se reporta la realización exploratoria completa, sin p-values o intervalos poblacionales fabricados con esos denominadores. Una ampliación exige lista de mundos/seeds y recursos propios fijados antes de correrla.

Las curvas usan los checkpoints fijos A0, A1000, A2000, A2001, A3000 y A4000, en todas las ramas y sin veto durante la lectura. Su población factual fija contiene todos los hechos comunes discovery del mundo nuevo, todas sus paráfrasis discovery y los tres mapas, sin filtrar por acierto; se capturan generaciones y logits para Value NLL. La población fija de NLL de texto contiene todas las ocurrencias TRAIN originales de entidades discovery, con la misma reducción global por token/EOS definida arriba. Ambos inventarios se sellan antes de adquirir hechos y permanecen completos en todos los checkpoints. No se introduce validation en estas curvas.

Separadamente se conserva por update el objetivo optimizado sobre el stream real `Loriginal + 0.25 Laux + 0.25 Lcan + 0.001 Pglobal`, su agrupación numérica V5 y cada componente. Se compara descriptivamente la media de los primeros 100 con los últimos 100 updates, registrando que los ejemplos de esas ventanas pueden ser distintos. Estas pérdidas online usan la política de entrenamiento de la rama; las curvas fijas anteriores usan inferencia sin veto. No se intercambian esas políticas al interpretar ni se elige un checkpoint por loss o score.

A4000 añade, sólo en discovery, no-veto y los cuatro vetos singleton de inferencia para las 17 ramas, usando Value NLL en la frontera pública sin incluir el verdadero Value en el input. Es un diagnóstico de acceso, separado de la evaluación primaria; no selecciona una unidad ni recalibra dosis. Su coste se incluye antes de ejecutar. Las lecturas conservan el estado, RNG, modo y gradientes del modelo.

## Interpretación prospectiva y fallos

Se conservan los contrastes completos cualquiera que sea su signo. Para resumir si la loss oculta un deterioro en esta realización se fija el margen descriptivo de NLL original en 0,02 nats por token (aproximadamente 2% de ratio de perplexity). Se informará si, en cada mapa, `abs(mean(P)-mean(D)) <= 0.02` y ambas medias no superan H en más de 0,02, junto con diferencias de afirmaciones falsas/accuracy y la evolución del objetivo optimizado. También se reporta cada rama para no ocultar heterogeneidad con promedios.

Ese margen es una elección de diseño exploratoria; no es una potencia calculada ni una prueba estadística de equivalencia funcional. Es un criterio de interpretación de outcomes, **no un gate para admitir, emparejar, filtrar, repetir o retocar ramas después de entrenar**. Si no se cumple, puede estimarse el efecto total de las políticas, pero no se resume como factualidad dañada con esa loss comparable. No se busca otro margen hasta conseguir un resultado favorable.

Las tasas incondicionales incluyen todas las preguntas. Riesgo condicionado a respuestas sólo se compara si la cobertura es exactamente igual en las ramas involucradas, en cada mapa, y existe denominador no nulo; no se ajustan decoder, thresholds o subconjuntos para igualarla. Salidas no evaluables se conservan separadas. Una predicción faltante por fallo de ejecución no se convierte en abstención o afirmación falsa.

Un déficit bajo inferencia sin veto no demuestra por sí solo que un hecho nunca se adquirió: reactivar una unidad poco adaptada puede introducir ruido y el resto del modelo puede compensar. El diagnóstico de acceso limita interpretaciones, sin garantizar separar todos esos mecanismos. El rescate mide retirar el veto con 2.000 updates restantes; puede combinar recuperación de acceso y aprendizaje adicional. Un rescate negativo no prueba irreversibilidad.

Faltantes, corrupción, divergencia sham, datos distintos, fronteras incorrectas, propuestas no finitas o fallos de verificación invalidan el contraste afectado. No se promedia sobre tres de las cuatro ramas para salvar un panel. Se conserva el fallo y no se repite automáticamente. El guard considera las unidades activas del update con reglas fijadas antes de correr; no se reinicia su streak al cambiar de unidad o al rescatar.

## Calificación necesaria para ejecutar

El código preparado ahora prueba calendarios y el contrato, sin NN, optimizer, Native, oracle o cloud. No es un runner científico.

Antes de ejecutar se requieren:

1. Fuente nueva fijada que admita los seeds, mundo y banco nuevos, conservando los bytes/semántica previstos. Export público/privado, exposición, vocabulario y geometría autenticados.
2. Inicialización W640/A0, sampler, optimizer y RNG compartidos calificados bajo esa fuente. Sham H debe coincidir con la ruta sana; los contratos permanecen distintos.
3. Familia nueva de schedules P/D/RP/RD en kernel, guard, cliente, registro/lector Native y replay. La máquina V5 reconoce H/T/C/R con dos mitades; no se reetiqueta D como C ni se mutan contratos por update.
4. Pruebas de la misma ruta real: cambio de sitio cada update, bloques de cuatro, R1999/2000, ACK antes de modificar parámetros, post-STEP, todos los frames y rechazos. Calificación del backward/clipping/AdamW y de los prefijos pareados; T1 conserva su alcance declarado.
5. Presupuesto medido de TRAIN, checkpoints, diagnósticos, inferencia, retención, verificación, limpieza y espacio libre. START/terminal fijan el deadline efectivo; una autorización JSON no inicia procesos.

Estos son requisitos de ejecución futuros, distintos de la terminación del presente diseño de control. Hasta cumplirlos: `runtime_qualified=false`, `empirical_control_validated=false`, `scientific_admission=false`, `training_authorized=false`.

## Recursos y entrega de este paso

La extrapolación de ocho terminales CPU V5 verificados es 11.925,299 segundos por rama para adquisición, verificación, tres mapas y postrun. Para 17 ramas son **56 h 18 min 50 s**; W640/inicialización nuevos añaden aproximadamente 14 min 30 s. H ya está incluido: no se añade un reference separado para localizar o seleccionar unidades.

No es un cap suficiente ni un benchmark del nuevo runner. Excluye adaptación/calificación, snapshots nuevos, curvas en seis checkpoints, diagnósticos de acceso, diferencias del mundo nuevo, retención y contingencia. Retener los pre/post completos del ejemplo V5 costaría 23,4375 GiB por rama, **398,4375 GiB para 17**, sólo para esos snapshots; otras capturas, pesos, ledgers, copias y filesystem son adicionales. No se propone esa política de retención sin medir su coste.

No se descuentan paralelización ni prefijos compartidos. GPU/AMD requiere calificación propia y una estimación medida; la autorización antigua de 90 minutos está cerrada y no financia estas ejecuciones.

La entrega de este paso consiste en este protocolo, un contrato legible por máquina, los calendarios completos y su auditoría mecánica, los vínculos de evidencia y una revisión independiente. Demuestra la construcción del nuevo control y deja explícitos sus límites; no presenta resultados de entrenamiento ni un control empíricamente calificado.
