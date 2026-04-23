"""
web-ui Flask service — Conversational chat interface for Ecuador SRI sales predictions.

Flow per chat request:
  1. Validate input & extract session_id from the browser.
  2. Publish a 'user-requests' Kafka event (best-effort).
  3. Proxy the question to the ai-agent POST /process endpoint.
  4. Return the agent's JSON response to the browser.
  5. JavaScript updates the chat UI without a page reload.

Endpoints
─────────
  GET  /        Serve the chat HTML (index.html).
  POST /chat    Receive question, orchestrate the flow, return JSON.
  GET  /health  Liveness probe.
  GET  /metrics Prometheus metrics.

Environment variables
─────────────────────
  AI_AGENT_URL            Default: http://ai-agent:5001
  AGENT_TIMEOUT_SECS      Default: 120
  KAFKA_BOOTSTRAP_SERVERS Default: kafka:9092
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import requests as http_client
from flask import Flask, Response, jsonify, render_template, request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from kafka_producer import publish_user_request
from metrics import web_chat_latency_seconds, web_requests_total

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

AI_AGENT_URL: str = os.environ.get("AI_AGENT_URL", "http://ai-agent:5001")
AGENT_TIMEOUT: int = int(os.environ.get("AGENT_TIMEOUT_SECS", "120"))

app = Flask(__name__)


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/chat")
def chat():
    body: dict = request.get_json(silent=True) or {}
    pregunta: str = str(body.get("pregunta", "")).strip()
    session_id: str = str(body.get("session_id") or uuid.uuid4())

    if not pregunta:
        return jsonify({"error": "La pregunta no puede estar vacía."}), 400

    # Publish user request event to Kafka (non-blocking, best-effort)
    publish_user_request(
        session_id=session_id,
        pregunta=pregunta,
        user_agent=request.headers.get("User-Agent", ""),
    )

    web_requests_total.labels(endpoint="/chat", status="started").inc()
    t0 = time.perf_counter()

    try:
        resp = http_client.post(
            f"{AI_AGENT_URL}/process",
            json={"pregunta": pregunta},
            timeout=AGENT_TIMEOUT,
        )
        resp.raise_for_status()
        agent_data: dict = resp.json()

    except http_client.exceptions.Timeout:
        web_requests_total.labels(endpoint="/chat", status="timeout").inc()
        logger.warning("Agent timeout for session=%s question=%r", session_id, pregunta[:80])
        return jsonify({"error": "El agente tardó demasiado en responder. Intente de nuevo."}), 504

    except http_client.exceptions.ConnectionError:
        web_requests_total.labels(endpoint="/chat", status="connection_error").inc()
        logger.error("Cannot connect to ai-agent at %s", AI_AGENT_URL)
        return jsonify({"error": "No se pudo conectar al servicio de IA. Verifique que el agente esté activo."}), 503

    except http_client.exceptions.HTTPError as exc:
        web_requests_total.labels(endpoint="/chat", status="agent_error").inc()
        logger.error("Agent HTTP error %s for session=%s", exc.response.status_code, session_id)
        return jsonify({"error": f"El agente devolvió un error ({exc.response.status_code})."}), 502

    latency_s = time.perf_counter() - t0
    web_chat_latency_seconds.observe(latency_s)
    web_requests_total.labels(endpoint="/chat", status="success").inc()

    logger.info(
        "Chat OK — session=%s  latency=%.1fs  province=%s",
        session_id,
        latency_s,
        agent_data.get("datos_usados", {}).get("provincia", "?"),
    )

    return jsonify(agent_data)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# ─────────────────────────────────────────────────────────
# Dev entry point (Gunicorn used in production via Dockerfile)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)
