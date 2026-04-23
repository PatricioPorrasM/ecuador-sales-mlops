# ADR-003: RandomForestRegressor como Baseline y XGBoostRegressor como Challenger

## Metadatos

| Campo | Detalle |
|---|---|
| **ID** | ADR-003 |
| **Título** | Selección de RandomForestRegressor (v1) y XGBoostRegressor (v2) para la predicción de ventas provinciales del Ecuador |
| **Estado** | Aceptado |
| **Fecha** | 2025 |
| **Autores** | Equipo de Arquitectura |
| **Revisores** | — |

---

## Contexto

El sistema requiere un modelo de machine learning capaz de predecir el total de ventas y exportaciones de sociedades por provincia del Ecuador (`TOTAL VENTAS Y EXPORTACIONES (419)`) dado un mes fiscal y las variables de exportaciones de bienes y servicios de personas naturales y sociedades para esa provincia.

El dataset de entrenamiento proviene del Servicio de Rentas Internas (SRI) del Ecuador y contiene registros mensuales por provincia, con las siguientes características:

- **Tipo de dato:** tabular, estructurado, numérico
- **Dimensión temporal:** series de tiempo mensuales por provincia (12 meses × N años × 25 provincias)
- **Volumen:** dataset de tamaño pequeño-mediano (orden de miles de filas tras reshape)
- **Variable objetivo:** continua (valor en USD, rango desde millones hasta miles de millones)
- **Dominio:** economía fiscal, exportaciones y ventas declaradas al SRI

Se evaluaron cuatro familias de modelos:

1. **RandomForestRegressor** (scikit-learn) — ensemble de árboles de decisión, sin boosting
2. **XGBoostRegressor** (xgboost) — gradient boosting optimizado, estado del arte en datos tabulares
3. **Redes neuronales LSTM** (TensorFlow/Keras) — arquitectura recurrente para series de tiempo
4. **Linear Regression / Ridge** (scikit-learn) — modelo lineal como baseline estadístico

---

## Decisión

Se adopta **RandomForestRegressor con hiperparámetros por defecto como modelo v1** (baseline) y **XGBoostRegressor con features enriquecidas e hiperparámetros optimizados como modelo v2** (challenger).

---

## Justificación

### Por qué datos tabulares favorecen ensemble trees sobre redes neuronales

La literatura reciente en machine learning es consistente en este punto: para datos tabulares estructurados con features numéricas, los modelos de gradient boosting y random forest superan sistemáticamente a las redes neuronales. El paper "Why do tree-based models still outperform deep learning on tabular data?" (Grinsztajn et al., 2022, NeurIPS) documenta esta ventaja en benchmarks exhaustivos.

Las razones fundamentales son:

**Invarianza rotacional.** Las redes neuronales presuponen que transformaciones lineales de las features producen representaciones útiles. Los árboles de decisión no hacen esta suposición y capturan relaciones no lineales directamente mediante particiones del espacio de features, lo que es más adecuado cuando las variables tienen significados semánticos específicos (exportaciones de bienes, exportaciones de servicios).

**Tamaño del dataset.** Las redes neuronales necesitan grandes volúmenes de datos para aprender representaciones útiles sin overfitting. Con el dataset del SRI (tamaño pequeño-mediano), un Random Forest con regularización implícita (bagging, max_features) generaliza mejor que una red neuronal que tendería a memorizar los datos de entrenamiento.

**Interpretabilidad.** Los modelos de árboles proveen `feature_importances_` nativamente, lo que permite incluir en el trabajo académico un análisis de qué variables son más predictivas (¿importan más las exportaciones de bienes o de servicios? ¿cuál provincia tiene mayor varianza?). Este análisis tiene valor académico y es imposible de obtener de forma directa con una red neuronal.

### Por qué RandomForest como v1 (baseline)

**Solidez sin tuning.** RandomForest con parámetros por defecto (`n_estimators=100`, `max_features='sqrt'`) produce resultados razonables en prácticamente cualquier problema de regresión tabular sin necesidad de búsqueda de hiperparámetros. Esto lo hace ideal como baseline: establece un piso de rendimiento robusto con mínimo esfuerzo, y cualquier modelo más elaborado debe superarlo para justificar su complejidad adicional.

**Resistencia al overfitting.** El mecanismo de bagging (bootstrap aggregating) de RandomForest reduce la varianza del modelo. Con datasets pequeños, esto es una ventaja significativa sobre modelos más expresivos.

**Amplio soporte en la comunidad.** RandomForest es uno de los algoritmos más documentados y comprendidos en machine learning, lo que facilita la justificación de su uso ante el jurado evaluador.

**Cálculo de confianza.** La arquitectura de ensemble de RandomForest permite calcular el score de confianza de cada predicción como la dispersión entre las predicciones de los `n_estimators` árboles individuales, sin necesidad de técnicas adicionales como conformal prediction o Monte Carlo Dropout.

### Por qué XGBoost como v2 (challenger)

**Estado del arte en Kaggle y la industria para datos tabulares.** XGBoost ha sido el algoritmo ganador o subcampeón en la mayoría de competencias de Kaggle con datos tabulares desde su popularización en 2016. Para predicción de ventas con features temporales, XGBoost con lag features y rolling statistics es la elección profesional estándar.

**Features de ingeniería temporal.** A diferencia de v1 que usa solo las features brutas del CSV, v2 incorpora:
- `lag_1` y `lag_2`: ventas del mes anterior y dos meses antes de la misma provincia, capturando momentum temporal
- `rolling_mean_3` y `rolling_std_3`: media y desviación estándar móvil de 3 meses, capturando tendencia y volatilidad local
- `mes_sin` y `mes_cos`: codificación cíclica del mes (`sin(2π*mes/12)`, `cos(2π*mes/12)`), permitiendo que el modelo aprenda la estacionalidad sin asumir relación lineal con el mes

Estas features son estándar en forecasting de series de tiempo y su inclusión en v2 justifica académicamente la diferencia de rendimiento esperada.

**Regularización explícita.** XGBoost incorpora términos de regularización L1 (`alpha`) y L2 (`lambda`) directamente en la función objetivo, lo que permite un control más fino del overfitting que RandomForest.

**Velocidad de entrenamiento.** XGBoost está implementado en C++ con soporte para paralelización, lo que resulta en tiempos de entrenamiento menores que RandomForest para datasets de tamaño comparable.

### Por qué no LSTM u otras redes neuronales

**Sobrediseño para el tamaño del dataset.** Una arquitectura LSTM agrega complejidad de implementación (TensorFlow, preprocesamiento de secuencias, normalización, callbacks de entrenamiento, manejo de GPU en contenedores) sin garantía de mejora en rendimiento sobre XGBoost con el volumen de datos disponible.

**Prioridad del sistema sobre el modelo.** El objetivo académico principal de este trabajo es demostrar el pipeline MLOps completo. Invertir tiempo en arquitecturas de redes neuronales desenfoca la prioridad correcta: el sistema, no el modelo.

**Imágenes Docker más pesadas.** TensorFlow añade aproximadamente 1-2GB al tamaño de la imagen Docker, aumentando tiempos de build y de pull en CI/CD y consumiendo más almacenamiento en Minikube.

### Por qué no Linear Regression / Ridge

**Supuesto de linealidad inapropiado.** La relación entre las exportaciones de distintos tipos y el total de ventas puede incluir interacciones no lineales (efectos de estacionalidad, outliers de meses de alto comercio). Un modelo lineal no captura estas relaciones.

**Baseline estadístico insuficiente.** Linear Regression es útil como sanity check, pero no como baseline para un sistema en producción. RandomForest ofrece un punto de comparación más relevante industrialmente.

---

## Features por versión

### v1 — RandomForestRegressor (baseline)
```
- mes_fiscal
- exportaciones_bienes_pn      (Exportaciones de Bienes, Personas Naturales)
- exportaciones_servicios_pn   (Exportaciones de Servicios, Personas Naturales)
- exportaciones_bienes_soc     (Exportaciones de Bienes, Sociedades)
- exportaciones_servicios_soc  (Exportaciones de Servicios, Sociedades)
```

### v2 — XGBoostRegressor (challenger)
```
Todas las features de v1, más:
- lag_1          (total ventas mes anterior, misma provincia)
- lag_2          (total ventas 2 meses antes, misma provincia)
- rolling_mean_3 (media móvil 3 meses, misma provincia)
- rolling_std_3  (desviación estándar móvil 3 meses, misma provincia)
- mes_sin        (sin(2π * mes / 12))
- mes_cos        (cos(2π * mes / 12))
```

---

## Métricas de evaluación

| Métrica | Justificación |
|---|---|
| **RMSE** | Penaliza errores grandes, métrica principal de comparación para promoción |
| **MAE** | Interpretable en la misma unidad que la variable objetivo (USD) |
| **MAPE** | Error relativo porcentual, independiente de la escala de la provincia |
| **R²** | Proporción de varianza explicada, comparable entre provincias |

**Criterio de promoción:** v2 es promovido a producción si `RMSE_v2 < RMSE_v1 × 0.95`, es decir, si mejora el RMSE en al menos un 5%.

---

## Consecuencias

### Positivas
- Pipeline de entrenamiento simple, rápido y reproducible sin dependencias de GPU
- Imágenes Docker ligeras (python:3.11-slim + scikit-learn + xgboost ≈ 400MB)
- Cálculo de confianza nativo mediante dispersión del ensemble
- Feature importances disponibles para análisis e interpretabilidad
- Demostración clara del flujo MLOps: baseline → challenger → comparación → promoción

### Negativas
- Los modelos no capturan dependencias temporales de largo plazo (ej. efectos de crisis económicas multi-año) tan bien como un LSTM
- El score de confianza de XGBoost requiere una aproximación (predicciones individuales de árboles) menos directa que la de RandomForest

### Neutrales
- La decisión de usar v1 vs v2 en producción se toma automáticamente por el script `compare_and_promote.py` basado en métricas objetivas, no en criterio subjetivo

---

## Alternativas Descartadas

| Alternativa | Razón de descarte |
|---|---|
| LSTM (TensorFlow) | Sobrediseño para dataset pequeño, imágenes pesadas, desenfoca prioridad del sistema |
| Linear Regression | Supuesto de linealidad inapropiado, baseline industrialmente irrelevante |
| LightGBM | Similar a XGBoost en rendimiento; XGBoost tiene mayor adopción y documentación |
| Prophet (Meta) | Diseñado para forecasting univariado; el problema tiene múltiples features de entrada |
| ARIMA/SARIMA | Modelos estadísticos sin soporte nativo para features exógenas múltiples |
