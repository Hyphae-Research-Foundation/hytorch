# El medio que falta — explicado sin tecnicismos

*Material de divulgación. Para prensa, inversores, familia, o cualquier
persona que pregunte "¿y ustedes qué hicieron?". La versión técnica vive en
`paper/notes/DRAFT.md`; cada afirmación de aquí tiene su evidencia allá.*

---

## El problema, en una imagen

Una inteligencia artificial, mientras aprende, es como una **oficina con una
pizarra gigante en el centro**. Miles de "empleados" (las capas del modelo)
pasan millones de veces por segundo y escriben, borran y reescriben cosas en
esa pizarra. Todo lo que la IA "sabe" en un momento dado está ahí.

El problema: **nadie firma nada.** Nadie anota quién escribió qué, ni cuándo,
ni qué se borró, ni qué dos empleados pelearon por el mismo espacio. Al final
del día solo ves el resultado — y si algo salió raro, no hay forma de saber
qué pasó. Así se entrenan hoy TODAS las IAs del mundo: la pizarra se edita a
sí misma billones de veces y no queda ni un recibo.

Durante años la industria intentó resolverlo con "intérpretes": gente que
mira la pizarra terminada e intenta adivinar qué significan los garabatos.
Nosotros hicimos otra cosa.

## Lo que construimos

No intentamos leer los garabatos. **Cambiamos las reglas de la oficina:**

1. **Nadie escribe en la pizarra sin un ticket.** Cada escritura es ahora un
   pequeño recibo firmado (16 bytes): quién, dónde, cuánto, y el veredicto.
2. **Los "no" también quedan registrados.** Si dos escrituras chocaron por el
   mismo espacio, o si una fue rechazada por inválida, eso queda en el libro.
   El intento fallido es un dato, no un silencio.
3. **Hay un notario con sello.** Un sistema de registro (Hyphae) guarda cada
   recibo en un libro contable a prueba de alteraciones. Y aquí lo fuerte:
   **el entrenamiento no puede avanzar al siguiente paso sin el sello del
   notario.** No es un logger opcional — es una barrera. En nuestro
   entrenamiento real eso pasó 40.000 veces sin excepción.
4. **Hay un auditor independiente.** Un programa aparte — que corre en un
   computador normal, sin necesidad de las GPUs carísimas — toma el libro y
   **re-ejecuta** partes del entrenamiento para comprobar, bit por bit, que
   lo registrado es exactamente lo que ocurrió. Si un solo bit no cuadra, el
   entrenamiento se declara inválido. Lo probamos saboteándonos a nosotros
   mismos 32 veces de 6 formas distintas: el auditor cazó las 32.

En un solo entrenamiento el libro registró **casi 8.000 millones de
recibos**, y mantenerlo costó menos del 10% del tiempo total. Y funciona
idéntico en los chips de NVIDIA y en los de AMD — mismo libro, mismo
auditor, mismo veredicto.

## ¿Y sirvió de algo? Tres historias reales del proyecto

**1. El libro detectó un problema que era invisible.**
En nuestra primera prueba, el marcador de progreso (la "nota" del modelo)
mejoraba sin parar. Todo parecía perfecto. Pero el libro contable mostraba
otra cosa: de ~196.000 intentos de escritura por paso, solo **2** se estaban
aprobando. La pizarra estaba congelada — el modelo aprendía por un atajo
lateral, y el mecanismo central que estábamos probando llevaba muerto desde
el principio. **Sin el libro, ese experimento se habría publicado como
exitoso.** Es como una empresa que reporta ganancias mientras la fábrica
lleva meses parada: solo la contabilidad lo revela.

**2. El árbitro falló contra nosotros — y eso es una función, no un fallo.**
Antes de entrenar, firmamos una regla: "si el modelo disciplinado es más de
10% peor que el libre, el experimento fracasa". La primera versión perdió
por 1,17 puntos. La regla ejecutó: fracaso, publicado, con todos los
recibos. Después encontramos el defecto (un componente heredado que no
pintaba nada ahí), lo quitamos dejando constancia escrita del cambio, y
repetimos con la MISMA regla firmada.

**3. El final que no esperábamos.**
En la repetición, el modelo disciplinado no solo cumplió la regla — **le
ganó al modelo libre por más de 40%**, en los dos fabricantes de chips. La
disciplina que pusimos para poder auditar (escrituras acotadas, con límites,
con veredictos) resultó además estabilizar el aprendizaje: el modelo "libre"
se descarriló a mitad del entrenamiento y el disciplinado siguió mejorando.
El corsé resultó ser columna vertebral.

## Antes → Después

| | **Antes** | **Después** |
|---|---|---|
| El estado interno de la IA cambia… | …sin dejar rastro | …solo con recibo firmado |
| Un intento de escritura rechazado… | …no existe para nadie | …es un dato consultable |
| Si el entrenamiento se corrompe… | …nadie se entera | …se detiene solo (40.000 barreras) |
| Verificar qué pasó exige… | …confiar en quien entrenó | …un computador normal y el libro |
| Detectar un fallo interno… | …suerte, o nunca | …está en la contabilidad |
| ¿Sabemos qué "significa" cada dato? | No | **Tampoco** — eso no lo prometimos |

La última fila importa: esto no hace que la IA sea "explicable" en el
sentido de entender sus pensamientos. Hace algo más básico y que no existía:
**que su historia sea un hecho verificable en vez de un acto de fe.**

## El pitch de ascensor (30 segundos)

> Hoy, cuando se entrena una IA, su memoria interna se reescribe miles de
> millones de veces sin que quede ningún registro — es una caja negra que se
> edita a sí misma. Nosotros construimos el primer entrenamiento donde cada
> cambio deja un recibo, los rechazos también cuentan, el proceso no avanza
> sin el sello de un notario digital, y un computador corriente puede
> auditarlo todo después, bit por bit. Lo probamos en chips de NVIDIA y de
> AMD: detectó un fallo invisible que habría arruinado el experimento, y el
> modelo entrenado con disciplina terminó aprendiendo mejor que el que
> entrenó a sus anchas. En una frase: **convertimos el "confía en mí" del
> entrenamiento de IA en un "compruébalo tú mismo".**

## Preguntas que siempre hacen

**¿Esto hace la IA más segura?** Hace *verificable* una parte que antes era
invisible. La seguridad total es más grande que esto, pero no hay seguridad
sin registro: es el prerequisito.

**¿No la hace más lenta?** El notario costó ~10% del tiempo. Y en nuestro
caso el modelo disciplinado aprendió mejor, no peor.

**¿Por qué nadie lo había hecho?** Porque la industria decidió hace una
década que la velocidad importaba más que el registro, y toda la
infraestructura se construyó sobre esa decisión. Nosotros demostramos que el
registro cabe — cuesta un 10%, no un 10×.

**¿Y ahora qué?** Escala (esto fue un modelo pequeño), y la pregunta
siguiente: ya que cada escritura tiene nombre, ¿podemos empezar a preguntar
qué *dicen*?
