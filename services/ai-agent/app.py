"""
ai-agent Flask service — ReAct agent for Ecuador SRI sales prediction.

Endpoints
─────────
  POST /process   Run the ReAct agent on a natural-language question.
  GET  /health    Liveness probe.
  GET  /ready     Readiness probe.
  GET  /metrics   Prometheus metrics.

The agent uses LiteLLM (Groq / llama3-8b-8192) to orchestrate two tools:
  1. get_province_data  — reads historical data from the SRI CSV
  2. call_inference     — calls the ml-inference service POST /predict

Required environment variables:
  GROQ_API_KEY            Groq API key (picked up automatically by LiteLLM)

Optional environment variables:
  LITELLM_MODEL           Default: groq/llama3-8b-8192
  ML_INFERENCE_URL        Default: http://ml-inference:5000
  DATA_PATH               Default: /app/data/Bdd_SRI_2025.csv
  KAFKA_BOOTSTRAP_SERVERS Default: kafka:9092
  AGENT_MAX_ITERATIONS    Default: 8
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
    Run the ReAct agent on a natural-language question about Ecuador sales.

    Body:   {"pregunta": "¿Cuánto venderán las sociedades de Pichincha en marzo?"}
    Returns: {respuesta, datos_usados, prediccion_raw, razonamiento}
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
    """Liveness probe — always 200 while the process is alive."""
    return jsonify({"status": "ok"})


@app.get("/ready")
def ready():
    """Readiness probe — 200 once the agent and tools are initialised."""
    return jsonify({"status": "ready", "model": _agent.model})


@app.get("/metrics")
def metrics():
    """Prometheus metrics in text exposition format."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# ─────────────────────────────────────────────────────────
# Dev server entry point (production uses Gunicorn via Dockerfile)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
