# Ecuador Sales MLOps

Sistema MLOps de extremo a extremo para predicción de ventas y exportaciones de sociedades en Ecuador, basado en datos oficiales del SRI (Servicio de Rentas Internas).

---

## Descripción del sistema y arquitectura

El sistema responde preguntas en lenguaje natural sobre predicciones de ventas provinciales para el período **octubre 2025 – septiembre 2026**, utilizando un modelo de ML entrenado sobre 69 registros mensuales del SRI (enero 2020 – septiembre 2025).

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Usuario                                     │
│          "¿Cuánto venderán en Pichincha en marzo 2026?"             │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        web-ui  :5002                                 │
│              Flask · chat UI · Kafka producer                        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ HTTP POST /process
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ai-agent  :5001                                 │
│   ReAct loop · LiteLLM/Groq llama3-8b-8192 · 2 herramientas        │
│                                                                      │
│   [THINK] razonar → [ACT] llamar herramienta → [OBSERVE] resultado  │
│                                                                      │
│   Herramienta 1: get_province_data   → lee CSV del SRI              │
│   Herramienta 2: call_inference      → llama ml-inference           │
└─────────────────┬───────────────────────────────────────────────────┘
                  │ HTTP POST /predict
                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ml-inference  :5000                                │
│   Flask · RandomForest v1 / XGBoost v2 · /ready (503 sin modelo)   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         Kafka  :9092                                 │
│   Topics: user-requests · agent-actions · model-responses           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   kafka-consumer  :8003                              │
│            Agrega métricas → Prometheus :9090 → Grafana :3000       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    model-trainer  (Job)                              │
│   train_v1.py (RandomForest) → train_v2.py (XGBoost)               │
│   compare_and_promote.py → model_production.pkl si mejora ≥ 5%     │
└─────────────────────────────────────────────────────────────────────┘
```

### Servicios

| Servicio | Puerto | Descripción |
|---|---|---|
| `web-ui` | 5002 | Interfaz de chat Flask |
| `ai-agent` | 5001 | Agente ReAct con LiteLLM |
| `ml-inference` | 5000 | API de predicción (pickle bundle) |
| `kafka-consumer` | 8003 | Consumidor de métricas Kafka |
| `model-trainer` | — | Job de entrenamiento (único) |
| Kafka | 9092/29092 | Broker de mensajes |
| Prometheus | 9090 | Recolección de métricas |
| Grafana | 3000 | Dashboards de observabilidad |

---

## Prerequisitos

| Herramienta | Versión mínima | Notas |
|---|---|---|
| Docker Desktop | 24+ | Con BuildKit habilitado |
| Docker Compose | v2.24+ | Incluido en Docker Desktop |
| Minikube | 1.33+ | Solo para despliegue K8s |
| kubectl | 1.29+ | Solo para despliegue K8s |
| Python | 3.11+ | Solo para tests locales |
| Git | 2.40+ | — |

**Claves de API necesarias:**

- `GROQ_API_KEY` — obtener en [console.groq.com/keys](https://console.groq.com/keys) (gratuito)
- `WANDB_API_KEY` — obtener en [wandb.ai/settings](https://wandb.ai/settings) (gratuito)

---

## Levantar con docker-compose

### 1. Configuración inicial

```bash
git clone https://github.com/<tu-usuario>/ecuador-sales-mlops.git
cd ecuador-sales-mlops

cp .env.example .env
# Editar .env con las claves reales:
#   GROQ_API_KEY=gsk_...
#   WANDB_API_KEY=...
#   WANDB_ENTITY=tu_usuario_wandb
```

### 2. Colocar el dataset

```
data/Bdd_SRI_2025.csv
```

El directorio `data/` se monta como volumen de solo lectura en `ai-agent` y `model-trainer`.

### 3. Ejecutar el entrenamiento inicial

El modelo debe existir antes de que `ml-inference` pueda servir predicciones:

```bash
docker compose --profile train up model-trainer
```

Entrena v1 (RandomForest) y v2 (XGBoost), compara métricas y guarda el ganador
en el volumen `models-data` como `model_production.pkl`.

### 4. Levantar el stack completo

```bash
docker compose up -d
docker compose ps          # ver estado de cada contenedor
docker compose logs -f     # seguir logs en tiempo real
```

### 5. Verificar los endpoints

```bash
curl http://localhost:5000/ready    # {"status":"ready","model_version":"v2"}
curl http://localhost:5001/health   # {"status":"ok"}
curl http://localhost:5002/health   # {"status":"ok"}
```

Abrir el chat: **http://localhost:5002**
Prometheus: **http://localhost:9090**
Grafana: **http://localhost:3000** (usuario: `admin`, contraseña: `admin`)

### 6. Detener

```bash
docker compose down       # conserva volúmenes (modelo entrenado)
docker compose down -v    # destruye también los volúmenes
```

---

## Desplegar en Minikube

### 1. Preparar el entorno

```bash
minikube start --cpus=4 --memory=8192 --disk-size=30g
eval $(minikube docker-env)
```

O con el script automatizado:

```bash
chmod +x scripts/setup-minikube.sh
./scripts/setup-minikube.sh
```

### 2. Construir las imágenes

```bash
chmod +x scripts/build-images.sh
./scripts/build-images.sh
```

Construye las 5 imágenes directamente en el daemon Docker de Minikube.

### 3. Desplegar todos los manifiestos

```bash
chmod +x scripts/deploy-all.sh
./scripts/deploy-all.sh
```

El script aplica los manifiestos en orden:

1. Namespaces (`streaming`, `frontend`, `agent`, `inference`, `monitoring`, `training`)
2. ConfigMaps y Secrets (claves API como Kubernetes Secrets)
3. PVCs compartidos para el modelo (HostPath en Minikube)
4. Kafka + ZooKeeper (namespace `streaming`)
5. `ml-inference` (namespace `inference`)
6. `ai-agent` (namespace `agent`)
7. `web-ui` (namespace `frontend`)
8. `kafka-consumer` (namespace `streaming`)
9. Prometheus + Grafana (namespace `monitoring`)

### 4. Acceder a los servicios

```bash
minikube service web-ui     -n frontend   --url   # chat
minikube service prometheus -n monitoring --url   # métricas
minikube service grafana    -n monitoring --url   # dashboards
```

NodePorts configurados: web-ui → **30500**, Prometheus → **30900**, Grafana → **30300**

### 5. Ejecutar el entrenamiento en K8s

```bash
# Incluido automáticamente con --train:
./scripts/deploy-all.sh --train

# O manualmente:
kubectl apply -f k8s/training/model-trainer-job.yaml
kubectl wait --for=condition=complete job/model-trainer -n training --timeout=600s
kubectl logs -n training job/model-trainer -c compare-and-promote
```

El Job usa initContainers secuenciales: `train-v1` → `train-v2` → `compare-and-promote`.
El modelo ganador se escribe en el PVC compartido con `ml-inference`.

---

## Ejecutar el entrenamiento

### Con docker-compose

```bash
docker compose --profile train up model-trainer
```

### Localmente

```bash
cd services/model-trainer
pip install -r requirements.txt

export DATA_PATH=../../data/Bdd_SRI_2025.csv
export MODEL_DIR=./models
export WANDB_API_KEY=...
export WANDB_PROJECT=ecuador-sales-mlops

python train_v1.py
python train_v2.py
python compare_and_promote.py
```

### Criterio de promoción

```
RMSE(v2) < RMSE(v1) × PROMOTION_THRESHOLD   (default: 0.95)
```

v2 debe mejorar el RMSE al menos un 5% para reemplazar a v1 en producción.

---

## Ejemplos de preguntas para el chat

El agente acepta preguntas sobre el período **octubre 2025 – septiembre 2026**.

**Predicciones:**

```
¿Cuánto venderán las sociedades de Pichincha en marzo 2026?
```

```
Dame una predicción de ventas para Guayas en enero 2026.
```

```
¿Cuál será el total de ventas y exportaciones de sociedades en Azuay en julio 2026?
```

```
¿Cómo se espera que sea diciembre 2025 para las exportaciones de Manabí?
```

**Preguntas fuera del rango (rechazadas correctamente):**

```
¿Cuánto vendieron en Pichincha en marzo 2024?
```
→ Fecha dentro del dataset histórico; no predecible.

```
¿Cuánto venderán en Guayas en enero 2028?
```
→ Supera el horizonte máximo de 12 meses.

**Provincias válidas:**

`AZUAY` · `BOLIVAR` · `CANAR` · `CARCHI` · `CHIMBORAZO` · `COTOPAXI` · `EL ORO` · `ESMERALDAS` · `GALAPAGOS` · `GUAYAS` · `IMBABURA` · `LOJA` · `LOS RIOS` · `MANABI` · `MORONA SANTIAGO` · `NAPO` · `ND` · `ORELLANA` · `PASTAZA` · `PICHINCHA` · `SANTA ELENA` · `SANTO DOMINGO DE LOS TSACHILAS` · `SUCUMBIOS` · `TUNGURAHUA` · `ZAMORA CHINCHIPE`

---

## Dashboards de Grafana

Dashboard principal: **Ecuador Sales MLOps** (disponible en `http://localhost:3000` o NodePort 30300).

| Panel | Métrica clave | Qué monitorear |
|---|---|---|
| **1. Throughput de solicitudes** | `rate(*_total[5m])*60` | Picos de tráfico por servicio |
| **2. Latencia de inferencia** | `histogram_quantile(0.99, inference_latency_bucket)` | p99 debe ser < 500 ms |
| **3. Latencia end-to-end** | `histogram_quantile(0.95, pipeline_latency_bucket)` | p95 típico: 3–10 s (incluye LLM) |
| **4. Top 5 provincias** | `topk(5, requests_by_provincia_total)` | Demanda geográfica |
| **5. Valor de predicción** | `model_prediction_value` | Último USD predicho por provincia |
| **6. Confianza del modelo** | `histogram_quantile(0.5, model_confidence_score_bucket)` | Valores < 0.6 indican alta incertidumbre |
| **7. Consumer lag Kafka** | `kafka_consumer_lag` | Lag creciente = consumer retrasado |
| **8. Throughput Kafka** | `rate(kafka_messages_consumed_total[5m])*60` | Mensajes/min por topic |
| **9. CPU y memoria** | `process_cpu_seconds_total`, `process_resident_memory_bytes` | Fugas o cuellos de botella |
| **10. Tasa de errores** | `rate(errors_total[5m]) / rate(requests_total[5m])` | > 1% sostenido requiere atención |

---

## Estructura del repositorio

```
ecuador-sales-mlops/
├── services/
│   ├── web-ui/           # Chat Flask (puerto 5002)
│   ├── ai-agent/         # Agente ReAct LiteLLM (puerto 5001)
│   ├── ml-inference/     # API sklearn/XGBoost (puerto 5000)
│   ├── kafka-consumer/   # Consumidor de métricas (puerto 8003)
│   └── model-trainer/    # Job: train_v1, train_v2, compare_and_promote
├── k8s/                  # Manifiestos Kubernetes por namespace
│   ├── namespaces.yaml
│   ├── configmap-global.yaml
│   ├── secret-global.yaml
│   ├── pvc-models.yaml
│   ├── ai-agent/
│   ├── ml-inference/
│   ├── web-ui/
│   ├── kafka-consumer/
│   ├── kafka/
│   ├── monitoring/
│   └── training/
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/dashboards/main-dashboard.json
├── scripts/
│   ├── setup-minikube.sh
│   ├── build-images.sh
│   └── deploy-all.sh
├── .github/workflows/
│   ├── ci.yml            # lint → test → trivy → build
│   └── cd.yml            # push GHCR → kubectl apply → smoke test
├── docker-compose.yml         # Stack completo local
├── docker-compose.kafka.yml   # Solo Kafka+ZooKeeper
└── .env.example               # 20 variables documentadas
```

---

## CI/CD

| Workflow | Trigger | Pasos |
|---|---|---|
| `ci.yml` | Push / PR | Lint (flake8) → Tests (pytest, cobertura ≥ 70%) → Trivy HIGH/CRITICAL → Docker build |
| `cd.yml` | Push a `main` | Build + push a GHCR → `kubectl apply` → smoke tests |

Secretos requeridos en GitHub: `GROQ_API_KEY`, `WANDB_API_KEY`, `KUBE_CONFIG_DATA`.

---

## Variables de entorno

Ver [`.env.example`](.env.example) para la documentación completa.

| Variable | Predeterminado | Descripción |
|---|---|---|
| `GROQ_API_KEY` | — | Clave Groq para el LLM del agente |
| `WANDB_API_KEY` | — | Clave W&B para tracking de experimentos |
| `LITELLM_MODEL` | `groq/llama3-8b-8192` | Modelo LLM del agente |
| `MAX_PREDICTION_MONTHS` | `12` | Horizonte de predicción en meses |
| `PROMOTION_THRESHOLD` | `0.95` | Umbral RMSE para promover v2 a producción |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:9092` | Broker Kafka (Docker) / FQDN (K8s) |
| `DATA_PATH` | `/app/data/Bdd_SRI_2025.csv` | Ruta al CSV del SRI |
| `MODEL_PATH` | `/app/models/model_production.pkl` | Modelo cargado por ml-inference |
