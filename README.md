# BTCAgent - Replicación M-DQN para Trading de Bitcoin

## 1. Título y propósito del proyecto
Este repositorio implementa una réplica operativa del paper **"Multi-level deep Q-networks for Bitcoin trading strategies"** (Scientific Reports, DOI: `10.1038/s41598-024-51408-w`) y agrega una mejora experimental en el módulo principal.

El objetivo es entrenar y evaluar una arquitectura de trading en múltiples niveles que:
- combine precio histórico de Bitcoin y sentimiento de Twitter,
- optimice retorno (ROI), riesgo (Sharpe Ratio), y actividad de trading,
- produzca tablas y gráficas comparables con el paper.

Además, el proyecto incluye:
- orquestación con presupuesto de tiempo,
- reanudación de ejecuciones,
- modo de emergencia de cierre rápido,
- scripts para corridas controladas por fases.

---

## 2. Resumen ejecutivo de arquitectura
### Flujo de alto nivel
El pipeline principal (`main.py`) se organiza en fases:

1. **Fase A (prep):** limpieza y alineación de datos.
2. **Fase B (pre-modelos):** entrenamiento de Trade-DQN y Predictive-DQN para generar señales `x1` y `x2`.
3. **Fase C/D (Main):** entrenamiento de Main-DQN en grilla de `alpha` (riesgo) y `omega` (máx. trades/día), incluyendo baseline y mejora.
4. **Fase E (eval/report):** generación de tablas, comparativas, plots y reportes finales.

### Diagrama textual
`data/raw -> preprocessing -> integrated_base -> (Trade-DQN => x1) + (Predictive-DQN => x2) -> merge_x1_x2 -> Main (paper + mejora) -> evaluation -> results/<run>`

### Entradas y salidas clave
- **Entradas principales:**
  - `data/raw/bitcoin_price/btc_hourly.csv`
  - `data/raw/twitter/engtweetsbtc_vader_sentiment.csv`
- **Salidas intermedias:**
  - `data/processed/x1_trade_recommendations.csv`
  - `data/processed/x2_price_predictions.csv`
  - `data/processed/integrated_dataset.csv`
- **Salidas finales:**
  - `results/<run>/table4_risk30.csv`
  - `results/<run>/table5_risk55.csv`
  - `results/<run>/table6_risk80.csv`
  - `results/<run>/table_improvement_delta.csv`
  - `results/<run>/table_improvement_summary.csv`
  - plots y reportes markdown/json.

---

## 3. Estructura de carpetas (exhaustiva)
### `config/`
Propósito: configuración central del sistema.

Archivos clave:
- `hyperparameters.py`: hiperparámetros globales, rutas, defaults de budgets, variantes, `alpha/omega` base.
- `emergency_profile.py`: perfil acotado para ejecución de emergencia.

Interacción:
- Es consumido por `main.py`, `main_emergency_2h.py`, `training/*`.

### `data/`
Propósito: datasets raw y procesados.

Subcarpetas:
- `data/raw/bitcoin_price/`: serie horaria BTC.
- `data/raw/twitter/`: datasets de tweets y variantes limpiadas.
- `data/processed/`: artefactos listos para entrenamiento y evaluación.

Archivos importantes en `processed`:
- `btc_price_clean.csv`
- `tweet_sentiment_scores.csv`
- `integrated_base.csv`
- `x1_trade_recommendations.csv`
- `x2_price_predictions.csv`
- `integrated_dataset.csv`

### `preprocessing/`
Propósito: limpieza y construcción de datasets.

Archivos:
- `btc_cleaner.py`: limpieza de precios BTC.
- `sentiment_analyzer.py`: agregación de sentimiento por hora.
- `data_integrator.py`: alineación temporal, ventana adaptada, split train/test, merge de señales.
- `tweet_cleaner.py`: limpieza textual (si aplica en flujo local).

### `models/`
Propósito: definición de agentes, entornos y redes por nivel.

Submódulos:
- `models/trade_dqn/`
- `models/predictive_dqn/`
- `models/main_dqn/`

Patrón en cada submódulo:
- `agent.py`: política, acción, entrenamiento.
- `enviroment.py` (nota: nombre histórico con typo): dinámica de entorno RL.
- `model.py`: red neuronal base.
- `replay_buffer.py`: memoria de experiencia.

Main contiene además:
- `model_dueling_double.py` (histórico de variantes),
- `reward_function.py`.

### `training/`
Propósito: runners de entrenamiento por modelo.

Archivos:
- `train_trade_dqn.py`
- `train_predictive_dqn.py`
- `train_main_dqn.py`

Responsabilidad:
- controlar episodios, evaluación periódica, early-stop, checkpoints y payload de retorno.

### `evaluation/`
Propósito: métricas y tablas.

Archivos:
- `metrics.py`: ROI, Sharpe.
- `evaluate_thresholds.py`: tablas por riesgo (`table4/5/6`).
- `evaluate_improvement.py`: delta vs baseline.
- `evaluate_reward_functions.py`: comparativa por reward (si se usa).
- `finalize_emergency_report.py`: cierre de modo emergencia.

### `visualization/`
Propósito: plots para reportes.

Archivos:
- `plot_price_history.py`
- `plot_trading_signals.py`
- `plot_improvement_comparison.py`

### `utils/`
Propósito: infraestructura de ejecución.

Archivos:
- `logger.py`: logger unificado de pipeline y entrenamiento.
- `dqn_runtime.py`: `StopConfig`, seed global, selección GPU.
- `pipeline_state.py`: estado persistente (`run_manifest`, `run_state`).
- `parallel_executor.py`: ejecución serial/paralela de jobs.
- `run_manager.py`: control de budget por etapa.
- `checkpoint.py`: utilidades de checkpoints.
- `utils/checkpoints/`: pesos, optimizadores, replay buffers, metadata/progress.

### `scripts/`
Propósito: comandos reproducibles de alto nivel.

Archivos:
- `run_10h.sh`: perfil quality-first (~10h), con resume y restart de C/D.
- `run_emergency_2h.sh`: emergencia conservadora.
- `run_emergency_2h_aggressive.sh`: emergencia agresiva.

### `tests/`
Propósito: pruebas de entorno, modelos, evaluación y estado de pipeline.

Archivos:
- `test_enviroments.py`
- `test_models.py`
- `test_preprocessing.py`
- `test_pipeline_state.py`
- `test_evaluate_improvement.py`

### `results/`
Propósito: artefactos versionados por corrida.

Formato recomendado:
- `results/<YYYYMMDD_HHMMSS>/...`

Contenido típico:
- tablas CSV (`table4/5/6`, `table_improvement_*`, `table8_sota_comparison.csv`),
- plots PNG,
- `deadline_report*.md`,
- `pipeline_payload*.json`,
- `run_manifest.json`, `run_state.json`.

### `logs/`
Propósito: observabilidad.

Archivos:
- `training.log`: progreso por fases y jobs.
- `runtime_audit.jsonl`: auditoría temporal por etapa.

### `BTCenv/`
Propósito: entorno virtual local del proyecto.

Contiene:
- binarios Python/pip,
- librerías instaladas,
- dependencias de runtime para TensorFlow y stack RL.

---

## 4. Pipeline principal (`main.py`)
### Fase A - Preparación y caché
- Limpia BTC, agrega sentimiento horario y construye `integrated_base` con split train/test.
- Puede usar caché validada si ya existen artefactos y no se fuerza rebuild.

### Fase B - Trade/Predictive
- Entrena `Trade-DQN` -> genera `x1`.
- Entrena `Predictive-DQN` -> genera `x2`.
- Integra `x1` + `x2` + precio para dataset final Main.

### Fase C/D - Main grid
- Entrena baseline y mejora sobre combinaciones `alpha/omega`.
- Baseline:
  - `main_classic_paper_*`
  - `main_proposed_paper_*`
- Mejora:
  - `main_proposed_policy_gradient_*`

### Fase E - Evaluación y reporte
- Genera tablas por riesgo, tabla delta de mejora, resumen y plots.
- Escribe reporte final y payload JSON en la carpeta de resultados.

### Reglas de operación
- `resume`: `off`, `auto`, `force`.
- `freeze-policy`: `graceful` o `hard` al alcanzar deadline.
- `stage-budget`: presupuesto por etapa en minutos.
- `parallel-policy` y `max-workers`: control de paralelismo.

---

## 5. Pipeline de emergencia (`main_emergency_2h.py`)
Objetivo:
- cerrar resultados rápido cuando queda muy poco tiempo.

Características:
- apunta a nodos Main pendientes,
- aplica caps agresivos de episodios/tiempo,
- incluye fallback cuando `x1/x2` colapsan,
- finaliza con tablas/plots/reporte de emergencia.

Cuándo usarlo:
- ventanas operativas cortas (2h) para completar entregables mínimos.

Limitaciones:
- más riesgo de underfitting,
- mayor varianza de métricas,
- no sustituye una corrida quality-first para benchmark serio.

---

## 6. Modelos y entrenamiento
### Trade-DQN
- Entrada principal: dinámica de precio.
- Salida: señal discreta `x1` (recomendación de trading).

### Predictive-DQN
- Entrada: precio + sentimiento.
- Salida: `x2` (predicción de movimiento/porcentaje futuro discretizado según configuración).
- Implementación actualizada:
  - la predicción de `t` se calcula desde `ap_{t-1}` (no desde `ap_t`) para evitar solución trivial constante,
  - el checkpoint de inferencia usa el mejor estado entrenado (o el mejor estado diverso si aplica),
  - se monitorea diversidad de acciones para prevenir colapso.

### Main-DQN (paper)
- Integra `x1`, `x2` y precio.
- Aprende política final de buy/sell/hold.
- Evalúa ROI/SR/trades en test.

### Main policy-gradient (mejora)
- Variante del Main con enfoque de gradiente de política.
- Comparación directa contra baseline `paper` con mismos `alpha/omega`.

### Checkpoints y progreso
En `utils/checkpoints/` se guardan:
- `*.weights.h5`
- `*_optimizer.npy`
- `*_replay.pkl` (DQN)
- `*_metadata.json`
- `*_progress.json`

Criterios de parada:
- `time_budget`,
- `episode_cap`,
- `early_stop`,
- `deadline_freeze`.

---

## 7. Configuración y CLI (referencia práctica)
### Flags principales de `main.py`
- `--total-budget-hours`
- `--deadline-hours`
- `--stage-budget "prep=...,trade=...,pred=...,main=...,eval=..."`
- `--gpu-mode {single,dual,auto}`
- `--parallel-policy {safe_adaptive,always_single,always_dual}`
- `--resume {off,auto,force}`
- `--results-dir <ruta>`

### Variantes y grilla
- `--main-variant-list "paper,policy_gradient"`
- `--predictive-variant-list "paper"`
- `--main-alpha-list "30,55,80"` o `"0.3,0.55,0.8"`
- `--main-omega-list "16"` o `"8,16,24"`

### Controles anti-colapso / anti-`best=0`
- `--early-stop-metric {roi,sr}`
- `--early-stop-patience`
- `--eval-every-n-episodes`
- `--min-improvement-delta`
- `--min-episodes-before-early-stop`
- `--min-trades-for-best`
- `--x2-min-unique` (gate mínimo de diversidad para `x2` antes de Main)
- `--x2-min-std` (gate mínimo de desviación estándar para `x2` antes de Main)

### `scripts/run_10h.sh` (interfaces públicas)
- `--resume-from <results_dir>`
- `--restart-main-from-b`
- `--restart-predictive`

Con `--restart-main-from-b`:
- limpia estado `main_*` de manifest/state,
- limpia checkpoints `main_dqn_*` para evitar arrastre.

Con `--restart-predictive`:
- limpia estado `pred_*` de manifest/state,
- limpia checkpoints `predictive_dqn*`,
- fuerza reentrenamiento de Predictive en una corrida de resume.

---

## 8. Ejecución práctica (comandos listos)
### A) Corrida limpia quality-first (~10h)
```bash
bash scripts/run_10h.sh
```

### B) Reanudar desde A/B ya completadas
```bash
bash scripts/run_10h.sh --resume-from results/<run_dir>
```

### C) Reiniciar solo C/D desde B (Main limpio)
```bash
bash scripts/run_10h.sh --resume-from results/<run_dir> --restart-main-from-b
```

### D) Correr solo paper-core (`omega=16`)
```bash
bash scripts/run_10h.sh --resume-from results/<run_dir> --main-omega-list 16
```

### D.1) Reentrenar solo Predictive (sin rehacer A)
```bash
bash scripts/run_10h.sh --resume-from results/<run_dir> --restart-predictive
```

### E) Completar omegas faltantes si sobra tiempo
```bash
bash scripts/run_10h.sh --resume-from results/<run_dir> --main-omega-list 8
bash scripts/run_10h.sh --resume-from results/<run_dir> --main-omega-list 24
```

### F) Modo emergencia
```bash
bash scripts/run_emergency_2h.sh
# o
bash scripts/run_emergency_2h_aggressive.sh
```

---

## 9. Resultados y artefactos
En `results/<run>/` se espera:
- `table4_risk30.csv`
- `table5_risk55.csv`
- `table6_risk80.csv`
- `table_improvement_delta.csv`
- `table_improvement_summary.csv`
- `table8_sota_comparison.csv`
- `price_history.png`
- `trading_signals.png`
- `improvement_comparison.png`
- `deadline_report_*.md`
- `pipeline_payload*.json`
- `run_manifest.json`, `run_state.json`

Interpretación rápida:
- `table4/5/6`: desempeño por riesgo (`alpha`) y trade threshold (`omega`).
- `table_improvement_delta`: `improved - baseline` por combinación.
- `table_improvement_summary`: promedio de mejora por variante.

---

## 10. Logging y monitoreo
### Archivos
- `logs/training.log`: estado del pipeline y jobs.
- `logs/runtime_audit.jsonl`: start/end por etapa con tiempos.

### Señales de avance saludable
- actualizaciones periódicas de batches en Fase C/D,
- crecimiento de archivos en `utils/checkpoints/`,
- timestamps recientes en progress/metadata.

### Comandos útiles
```bash
tail -f logs/training.log
watch -n 30 'ls -lt utils/checkpoints | head -n 10'
watch -n 5 nvidia-smi
```

---

## 11. Troubleshooting
### Caso: `best=0`
Causas típicas:
- `time_budget` demasiado bajo por job,
- early-stop agresivo,
- política colapsada a `hold` (trades casi 0),
- arrastre de checkpoints Main degradados.

Acciones:
- ejecutar solo `omega=16` para aumentar profundidad por job,
- subir `main` en `--stage-budget`,
- usar `--min-trades-for-best` y `--min-episodes-before-early-stop`,
- reiniciar C/D con `--restart-main-from-b`.

### Caso: `time_budget` constante en C/D
- Reducir número de jobs (`--main-omega-list 16`),
- subir minutos de etapa `main`.

### Caso: colapso `x1/x2`
- Rehacer A/B (`--force-prep-rebuild` y `--resume off`),
- validar cardinalidad de señales en `data/processed`.
- `main.py` corta el pipeline antes de C/D si `x2` no supera mínimos de diversidad (`--x2-min-unique`, `--x2-min-std`).
- para recuperar rápido en runs con resume: `--restart-predictive` y luego `--restart-main-from-b`.
- si persiste el colapso en Predictive, verificar en logs `unique_actions` y `PRED-COLLAPSE`; el entrenamiento ahora aplica warmup de exploración y decay de epsilon por episodio.

### Caso: errores CUDA/libdevice (TensorFlow)
- usar scripts que setean `.cuda_stub` y `XLA_FLAGS`,
- confirmar `BTCenv` activo,
- usar `gpu-mode single` en servidor compartido.

---

## 12. Fidelidad al paper
### Alineado con paper
- Estructura multi-nivel: Trade-DQN + Predictive-DQN + Main.
- Evaluación por riesgo (`alpha`) y límites de actividad (`omega`).
- Comparativa `classic` vs `proposed`.
- Métricas ROI y SR.

### Adaptaciones por restricciones de operación
- presupuesto por tiempo y freeze de deadline,
- mecanismos de resume y reintento,
- modo emergencia para cierre rápido,
- mejora experimental `policy_gradient` en Main.

### Benchmark objetivo reportado por paper
- Mejor caso propuesto: `alpha=55%`, `omega=16`
  - ROI: **29.93%**
  - SR: ~**2.39** (y en comparación de reward reporta ~2.74)

---

## 13. Guía de operación recomendada (quality-first)
1. Confirmar A/B completos en `run_manifest` y outputs `x1/x2` existentes.
2. Ejecutar C/D en paper-core (`omega=16`) con Main limpio:
   - `--resume-from ... --restart-main-from-b --main-omega-list 16`
3. Validar calidad en `table5_risk55.csv` y delta vs baseline.
4. Solo si sobra tiempo, completar `omega=8` y `omega=24` en corridas separadas.
5. Cerrar con reporte y tablas en carpeta fechada única.

---

## 14. Glosario y convenciones
- **alpha (`α`)**: umbral de riesgo tolerado por reward/política.
- **omega (`ω`)**: límite de trades activos por día.
- **ROI**: retorno relativo sobre inversión inicial.
- **SR (Sharpe Ratio)**: retorno ajustado por riesgo.
- **classic method**: baseline con señales derivadas de precio sin pipeline propuesto completo.
- **proposed method**: baseline paper integrando pre-modelos (`x1/x2`).
- **variant**: variante de Main (`paper`, `policy_gradient`).
- **branch**: rol comparativo (`baseline`, `improved`).
- **run_manifest / run_state**: estado persistente de nodos y resultados por corrida.

---

## Apéndice A - Inventario rápido de raíz
- `main.py`: pipeline principal.
- `main_emergency_2h.py`: pipeline de emergencia.
- `scripts/`: runners operativos.
- `models/`: definición de agentes/redes/entornos.
- `training/`: entrenamiento por modelo.
- `evaluation/`: tablas y métricas.
- `visualization/`: plots.
- `preprocessing/`: preparación de datos.
- `config/`: hiperparámetros y perfiles.
- `utils/`: runtime, logging, estado y checkpoints.
- `data/`: inputs raw + procesados.
- `results/`: outputs por corrida.
- `logs/`: trazabilidad de ejecución.
