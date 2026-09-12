Estos datos portables sostienen las gráficas del informe técnico. Conservan evidencia V5 medida y diseño V6 prospectivo; no presentan un resultado causal confirmado.

Desde la raíz del repositorio, sólo con Python estándar:

```bash
python3 -B reports/factual-channels-2026-09-12/data/verify_selection.py
python3 -B reports/factual-channels-2026-09-12/data/export_data.py --check
```

El primer comando recompone el registro completo del selector desde las tablas reducidas comprometidas y exige el mismo `no-control-match`. Verifica los hashes de tablas, resultado, regla y copia exacta del selector antes de ejecutarlo desde esos bytes. No necesita `results/`, `.worktrees/`, Torch, checkpoint, oracle o Native.

El segundo verifica el conjunto curado y todos los originales que estén disponibles. Si están todos, lo reconstruye desde ellos y compara los bytes. Si faltan, lo indica y conserva la comprobación portable del selector y de sus vínculos con las tablas usadas por las figuras. Las figuras sólo necesitan `report-data.json`.

Para regenerar la extracción cuando estén disponibles **todos** los originales fijados:

```bash
python3 -B reports/factual-channels-2026-09-12/data/export_data.py --refresh
```

| Archivo | Contenido y alcance |
| --- | --- |
| `report-data.json` | Datos pequeños para gráficos: competencia por mapa/estrato con denominadores; dos baterías, cuatro unidades, efectos por hecho/posición; matching, repeticiones, conteos, tiempos y diseño prospectivo. |
| `provenance.json` | SHA256 de cada original y de cada export, paths relativos al repositorio, campos fuente y metodología exacta. Los IDs E01–E06 remiten a fuentes originales, no a nuevos experimentos. |
| `selection-tables.json.gz` | `assembly.tables` completas, sin textos ni tokens. JSON canónico comprimido con gzip, nombre vacío y `mtime=0`. Conserva NLL por coordenada y sumas/conteos de magnitud ya reducidos. |
| `selection-result.json` | Registro interno original `selection.result.selection`, con su hash y resultado completo, sin el envelope de procedencia del runtime. |
| `causal_selection.py` | Copia exacta del selector puro congelado; biblioteca estándar, sin APIs de modelo. |
| `FACTUAL-CAUSAL-ADMISSION-V1.md` y `FACTUAL-CAUSAL-PILOT-PROTOCOL.md` | Copias exactas de documentos históricos de la regla. Sus propuestas no constituyen autorización vigente ni describen el nuevo calendario V6. |

La interfaz de gráficos tiene `schema_version=1`:

- `reference.competence`: filas `{position,stratum,facts,queries,correct,false_assertions,abstentions,invalid,missing,accuracy,strict_accuracy,false_assertion_rate,coverage}`. Accuracy es macro por hecho; las tasas usan fracciones, no porcentajes.
- `reference.priors`: los cuatro priors comunes por posición, con denominadores.
- `measurement.batteries`: dos objetos `{battery,units}`. Cada unidad conserva `delta_value_nats,g,m,r,rho,p`, `fact_effects`, `positions` y `totals`. Las posiciones incluyen las mismas métricas derivadas de sus sumas y conteos cerrados.
- `measurement.matching`, `rankings`, `rule`, `repeat_checks`, `cardinality`, `population`, `timing`, `custody`: resultado completo de la fase y sus límites. Custodia incluye los bytes leídos por el verificador cerrado, conteos de archivos/directorios, resultado del recheck de metadata y cierre del supervisor; no describe el tamaño del archivo completo ni un nuevo replay raw.
- `measurement.geometry`: resumen de equivalencia entre fuentes con la misma geometría B16, diagnóstico B1/B16 por mapa y gate cerrado de preservación de decisiones. Los logits pueden diferir entre geometrías aunque el argmax no cambie en los prefijos archivados examinados.
- `design_v6.calendar_examples`: rangos `[0,16)` y `[1992,2008)` de los 17 calendarios fijados. Los índices son cero-based y `null` significa sham. `arm_summaries` conserva dosis de oportunidades totales y por mitad, sin confundirlas con escrituras borradas observadas.

Reproducir las reducciones y el selector no revalida las capturas raw ni vuelve a ejecutar el modelo. No se incluyen checkpoints, tensores crudos, respuestas, tokens, oracle, ledgers, rutas locales absolutas o direcciones de infraestructura. Las dos baterías repiten los mismos casos/modelo; V6 sigue sin mundo generado, runtime calificado o control validado empíricamente.
