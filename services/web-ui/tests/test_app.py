"""
Tests del servicio web-ui.

Cubre los endpoints principales:
  GET  /health  — sonda de liveness
  POST /chat    — flujo de chat: validación, proxy al agente, manejo de errores
  GET  /metrics — exposición de métricas Prometheus

Las dependencias externas (Kafka, ai-agent HTTP) se reemplazan con mocks
para aislar el comportamiento del servicio.
"""
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def client():
    """Retorna un cliente de test Flask con Kafka mockeado."""
    with patch("kafka_producer.KafkaProducer"):
        import app as web_app
        web_app.app.config["TESTING"] = True
        with web_app.app.test_client() as c:
            yield c


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    """El endpoint /health debe retornar 200 con status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat — validación de entrada
# ─────────────────────────────────────────────────────────────────────────────

def test_chat_sin_pregunta_retorna_400(client):
    """Una solicitud de chat sin 'pregunta' debe retornar 400."""
    resp = client.post("/chat", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_chat_pregunta_vacia_retorna_400(client):
    """Una pregunta vacía o solo espacios debe retornar 400."""
    resp = client.post("/chat", json={"pregunta": "   "})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat — proxy exitoso al ai-agent
# ─────────────────────────────────────────────────────────────────────────────

def test_chat_exito_retorna_respuesta_del_agente(client):
    """Una pregunta válida debe devolver la respuesta JSON del ai-agent."""
    agente_respuesta = {
        "respuesta": "Las ventas de Pichincha en marzo 2026 serán de $1.200.000.",
        "datos_usados": {"provincia": "PICHINCHA", "mes": 3},
        "prediccion_raw": 1200000.0,
        "razonamiento": "[THINK] ...\n[ACT] ...\n[OBSERVE] ...",
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = agente_respuesta
    mock_resp.raise_for_status.return_value = None

    with patch("app.http_client.post", return_value=mock_resp):
        with patch("app.publish_user_request"):
            resp = client.post(
                "/chat",
                json={"pregunta": "¿Cuánto venderán en Pichincha en marzo 2026?"},
            )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["respuesta"] == agente_respuesta["respuesta"]
    assert data["prediccion_raw"] == 1200000.0


# ─────────────────────────────────────────────────────────────────────────────
# POST /chat — manejo de errores del ai-agent
# ─────────────────────────────────────────────────────────────────────────────

def test_chat_timeout_del_agente_retorna_504(client):
    """Si el ai-agent tarda demasiado debe retornar 504 Gateway Timeout."""
    import requests as req_lib

    with patch("app.http_client.post", side_effect=req_lib.exceptions.Timeout):
        with patch("app.publish_user_request"):
            resp = client.post(
                "/chat",
                json={"pregunta": "¿Cuánto venderán en Guayas?"},
            )

    assert resp.status_code == 504
    assert "error" in resp.get_json()


def test_chat_error_conexion_al_agente_retorna_503(client):
    """Si el ai-agent no está disponible debe retornar 503 Service Unavailable."""
    import requests as req_lib

    with patch("app.http_client.post", side_effect=req_lib.exceptions.ConnectionError):
        with patch("app.publish_user_request"):
            resp = client.post(
                "/chat",
                json={"pregunta": "¿Cuánto venderán en Azuay?"},
            )

    assert resp.status_code == 503
    assert "error" in resp.get_json()


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_metrics_retorna_contenido_prometheus(client):
    """El endpoint /metrics debe retornar texto en formato Prometheus."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"web_requests_total" in resp.data
