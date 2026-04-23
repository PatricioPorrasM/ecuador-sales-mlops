"""
Servicio Flask del ai-agent — agente ReAct para predicción de ventas del SRI Ecuador.

Endpoints
─────────
  POST /process   Ejecuta el agente ReAct sobre una pregunta en lenguaje natural.
  GET  /health    Sonda de disponibilidad (liveness).
  GET  /ready     Sonda de preparación (readiness).
  GET  /metrics   Métricas Prometheus.

El agente utiliza LiteLLM (Groq / llama3-8b-8192) para orquestar dos herramientas:
  1. get_province_data  — lee datos históricos del CSV del SRI
  2. call_inference     — llama al servicio ml-inference POST /predict

Variables de entorno obligatorias:
  GROQ_API_KEY            Clave de API de Groq (leída automáticamente por LiteLLM)

Variables de entorno opcionales:
  LITELLM_MODEL           Predeterminado: groq/llama3-8b-8192
  ML_INFERENCE_URL        Predeterminado: http://ml-inference:5000
  DATA_PATH               Predeterminado: /app/data/Bdd_SRI_2025.csv
  KAFKA_BOOTSTRAP_SERVERS Predeterminado: kafka:9092
  AGENT_MAX_ITERATIONS    Predeterminado: 8
"""

from __future__ import annotations

import logging
import time

from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from agent import ReActAgent
from kafka_producer import publish_agent_action
from metrics import agent_latency_seconds, agent_requests_total

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Single agent instance — stateless, safe to share across threads
_agent = ReActAgent()


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────


@app.post("/process")
def process():
    """
    Ejecuta el agente ReAct sobre una pregunta en lenguaje natural acerca de ventas en Ecuador.

    Cuerpo:   {"pregunta": "¿Cuánto venderán las sociedades de Pichincha en marzo?"}
    Retorna: {respuesta, datos_usados, prediccion_raw, razonamiento}
    """
    body: dict = request.get_json(silent=True) or {}
    pregunta: str = str(body.get("pregunta", "")).strip()

    if not pregunta:
        return jsonify({"error": "El campo 'pregunta' es obligatorio y no puede estar vacío."}), 400

    agent_requests_total.labels(status="started").inc()
    t0 = time.perf_counter()

    try:
        result = _agent.run(pregunta)
    except Exception as exc:
        logger.exception("Unhandled agent error — pregunta: %s", pregunta)
        agent_requests_total.labels(status="error").inc()
        return jsonify({"error": f"Error interno del agente: {exc}"}), 500

    latency_s = time.perf_counter() - t0
    latency_ms = round(latency_s * 1000, 1)

    agent_requests_total.labels(status="success").inc()
    agent_latency_seconds.observe(latency_s)

    # Build Kafka event — extract province/month from inference payload or
    # fall back to datos_usados so the event is always populated.
    inf_payload: dict = result.get("_inference_payload", {})
    datos: dict = result.get("datos_usados", {})
    publish_agent_action(
        pregunta_original=pregunta,
        provincia_extraida=str(inf_payload.get("provincia") or datos.get("provincia", "")),
        mes_extraido=int(inf_payload.get("mes") or datos.get("mes", 0) or 0),
        payload_enviado=inf_payload,
        latencia_agente_ms=latency_ms,
    )

    return jsonify(
        {
            "respuesta": result["respuesta"],
            "datos_usados": result["datos_usados"],
            "prediccion_raw": result["prediccion_raw"],
            "razonamiento": result["razonamiento"],
        }
    )


@app.get("/health")
def health():
    """Sonda de liveness — retorna 200 mientras el proceso esté activo."""
    return jsonify({"status": "ok"})


@app.get("/ready")
def ready():
    """Sonda de readiness — retorna 200 una vez que el agente y sus herramientas están listos."""
    return jsonify({"status": "ready", "model": _agent.model})


@app.get("/metrics")
def metrics():
    """Métricas Prometheus en formato de exposición de texto."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# ─────────────────────────────────────────────────────────
# Punto de entrada para desarrollo (producción usa Gunicorn via Dockerfile)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
