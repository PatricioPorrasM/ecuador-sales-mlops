"""
ml-inference Flask service — Ecuador SRI sales forecasting API.

Endpoints
─────────
  POST /predict   Run inference and return prediction + confidence.
  GET  /health    Liveness probe (always 200 while process is alive).
  GET  /ready     Readiness probe (200 only when model is loaded in memory).
  GET  /metrics   Prometheus metrics in text exposition format.

On each successful prediction the service:
  1. Increments Prometheus counters / histograms.
  2. Publishes a JSON event to the Kafka 'model-responses' topic (best-effort).

Gunicorn is the production server (see Dockerfile CMD).  A single worker is
used so the model stays in one process memory space; concurrency is achieved
via --threads.
"""

from __future__ import annotations

import logging
import time

from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from kafka_producer import publish_prediction
from metrics import (
    inference_latency_seconds,
    inference_requests_total,
    model_confidence_score,
    model_prediction_value,
    model_version_info,
)
from model_loader import (
    compute_confidence,
    get_bundle,
    is_model_loaded,
    prepare_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Eager model load at startup ───────────────────────────────────────────────
# A failed load is not fatal — /predict will retry lazily and /ready will
# return 503 until the PVC is populated by the training job.
_startup_bundle = get_bundle()
if _startup_bundle:
    model_version_info.labels(version=_startup_bundle["version"]).set(1)
    logger.info("Startup model load OK — version=%s", _startup_bundle["version"])
else:
    logger.warning(
        "Model not available at startup. Service will become ready once "
        "the training job writes model_production.pkl to the shared volume."
    )


# ─────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────


@app.post("/predict")
def predict():
    # ── Ensure model is available ─────────────────────────
    bundle = get_bundle()
    if bundle is None:
        return jsonify({"error": "Model not loaded. Run the training pipeline first."}), 503

    # ── Parse and validate request body ──────────────────
    data: dict = request.get_json(silent=True) or {}

    required_fields = {
        "provincia",
        "mes",
        "exportaciones_bienes_pn",
        "exportaciones_servicios_pn",
        "exportaciones_bienes_soc",
        "exportaciones_servicios_soc",
    }
    missing = required_fields - data.keys()
    if missing:
        return jsonify({"error": f"Missing required fields: {sorted(missing)}"}), 400

    provincia_raw: str = str(data["provincia"])

    # ── Feature engineering ───────────────────────────────
    t_start = time.perf_counter()

    try:
        X = prepare_features(bundle, data)
    except ValueError as exc:
        inference_requests_total.labels(provincia=provincia_raw, status="unknown_province").inc()
        return jsonify({"error": str(exc)}), 400

    # ── Inference ─────────────────────────────────────────
    prediction = float(bundle["model"].predict(X)[0])
    confidence = compute_confidence(bundle, X)

    latency_s = time.perf_counter() - t_start
    latency_ms = round(latency_s * 1000, 2)

    # ── Observability ─────────────────────────────────────
    inference_requests_total.labels(provincia=provincia_raw, status="success").inc()
    inference_latency_seconds.observe(latency_s)
    model_prediction_value.labels(provincia=provincia_raw).set(prediction)
    model_confidence_score.observe(confidence)

    logger.info(
        "Prediction — provincia=%s mes=%s value=%.2f confidence=%.4f latency=%.1fms",
        provincia_raw, data["mes"], prediction, confidence, latency_ms,
    )

    # ── Kafka event (non-blocking, best-effort) ───────────
    publish_prediction(
        provincia=provincia_raw,
        mes=int(data["mes"]),
        prediccion=prediction,
        modelo_version=bundle["version"],
        latencia_ms=latency_ms,
    )

    return jsonify(
        {
            "provincia": provincia_raw,
            "mes": int(data["mes"]),
            "prediccion_total_ventas": round(prediction, 2),
            "modelo_version": bundle["version"],
            "confianza": confidence,
        }
    )


@app.get("/health")
def health():
    """Liveness probe — always 200 while the process is running."""
    return jsonify({"status": "ok"})


@app.get("/ready")
def ready():
    """
    Readiness probe — 200 only when the model is loaded in memory.
    Kubernetes will stop routing traffic here until this returns 200,
    giving the training job time to produce model_production.pkl.
    """
    if not is_model_loaded():
        # One more attempt in case the file appeared since startup
        bundle = get_bundle()
        if bundle:
            model_version_info.labels(version=bundle["version"]).set(1)

    if is_model_loaded():
        return jsonify({"status": "ready", "model_version": get_bundle()["version"]})

    return (
        jsonify({"status": "not ready", "reason": "model not loaded"}),
        503,
    )


@app.get("/metrics")
def metrics():
    """Prometheus metrics in text exposition format."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


# ─────────────────────────────────────────────────────────
# Dev server entry point (production uses Gunicorn)
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
