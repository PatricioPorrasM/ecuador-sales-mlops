"""
Tests del servicio ml-inference.

Cubre:
  GET  /health  — sonda de liveness
  GET  /ready   — sonda de readiness (503 sin modelo, 200 con modelo)
  POST /predict — validación de campos, inferencia exitosa, provincia inválida
  GET  /metrics — métricas Prometheus

El bundle del modelo y las funciones de model_loader se mockean para
evitar dependencia en archivos de modelo en disco.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: bundle de modelo simulado
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_bundle():
    """Retorna un bundle de modelo mínimo compatible con ml-inference."""
    modelo = MagicMock()
    modelo.predict.return_value = np.array([1_200_000.0])
    return {
        "model": modelo,
        "version": "v2",
        "feature_cols": [
            "province_code", "ano_fiscal", "mes_fiscal",
            "exp_bienes_pn", "exp_servicios_pn",
            "exp_bienes_soc", "exp_servicios_soc",
        ],
        "label_encoder": MagicMock(),
        "provinces": ["PICHINCHA", "GUAYAS", "AZUAY"],
        "metrics": {"test_r2": 0.88},
    }


@pytest.fixture()
def client(mock_bundle):
    """Cliente de test con modelo y Kafka mockeados."""
    with patch("model_loader.get_bundle", return_value=None):
        with patch("kafka_producer.KafkaProducer"):
            import app as inf_app
            inf_app.app.config["TESTING"] = True
            with inf_app.app.test_client() as c:
                yield c, mock_bundle


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

def test_health_retorna_200(client):
    """El endpoint /health debe retornar 200 independientemente del estado del modelo."""
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# GET /ready
# ─────────────────────────────────────────────────────────────────────────────

def test_ready_sin_modelo_retorna_503(client):
    """Si el modelo no está cargado /ready debe retornar 503."""
    c, _ = client
    with patch("app.is_model_loaded", return_value=False):
        with patch("app.get_bundle", return_value=None):
            resp = c.get("/ready")
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["status"] == "not ready"


def test_ready_con_modelo_retorna_200(client):
    """Si el modelo está cargado /ready debe retornar 200 con la versión."""
    c, bundle = client
    with patch("app.is_model_loaded", return_value=True):
        with patch("app.get_bundle", return_value=bundle):
            resp = c.get("/ready")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ready"
    assert data["model_version"] == "v2"


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict — validación de campos
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_sin_modelo_retorna_503(client):
    """Si el modelo no está disponible /predict debe retornar 503."""
    c, _ = client
    with patch("app.get_bundle", return_value=None):
        resp = c.post("/predict", json={"provincia": "PICHINCHA", "mes": 3})
    assert resp.status_code == 503


def test_predict_campos_faltantes_retorna_400(client):
    """Una solicitud sin los campos obligatorios debe retornar 400."""
    c, bundle = client
    with patch("app.get_bundle", return_value=bundle):
        resp = c.post(
            "/predict",
            json={"provincia": "PICHINCHA", "mes": 3},  # faltan campos de exportación
        )
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data
    assert "Missing required fields" in data["error"]


def test_predict_provincia_invalida_retorna_400(client):
    """Una provincia desconocida debe retornar 400 con mensaje de error."""
    c, bundle = client
    payload_valido = {
        "provincia": "ATLANTIDA_INEXISTENTE",
        "mes": 3,
        "ano_fiscal": 2026,
        "exportaciones_bienes_pn": 100.0,
        "exportaciones_servicios_pn": 50.0,
        "exportaciones_bienes_soc": 200.0,
        "exportaciones_servicios_soc": 80.0,
    }
    with patch("app.get_bundle", return_value=bundle):
        with patch("app.prepare_features", side_effect=ValueError("Unknown province")):
            resp = c.post("/predict", json=payload_valido)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


# ─────────────────────────────────────────────────────────────────────────────
# POST /predict — predicción exitosa
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_exitoso_retorna_prediccion(client):
    """Una solicitud válida debe retornar la predicción con todos los campos esperados."""
    c, bundle = client
    features_mock = np.array([[1.0, 2026.0, 3.0, 100.0, 50.0, 200.0, 80.0]])
    payload = {
        "provincia": "PICHINCHA",
        "mes": 3,
        "ano_fiscal": 2026,
        "exportaciones_bienes_pn": 100.0,
        "exportaciones_servicios_pn": 50.0,
        "exportaciones_bienes_soc": 200.0,
        "exportaciones_servicios_soc": 80.0,
    }

    with patch("app.get_bundle", return_value=bundle):
        with patch("app.prepare_features", return_value=features_mock):
            with patch("app.compute_confidence", return_value=0.87):
                with patch("app.publish_prediction"):
                    resp = c.post("/predict", json=payload)

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["provincia"] == "PICHINCHA"
    assert data["mes"] == 3
    assert data["ano_fiscal"] == 2026
    assert data["modelo_version"] == "v2"
    assert data["prediccion_total_ventas"] == pytest.approx(1_200_000.0, rel=1e-3)
    assert data["confianza"] == pytest.approx(0.87, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_metrics_retorna_contenido_prometheus(client):
    """El endpoint /metrics debe retornar texto en formato Prometheus."""
    c, _ = client
    resp = c.get("/metrics")
    assert resp.status_code == 200
    assert b"inference_requests_total" in resp.data
