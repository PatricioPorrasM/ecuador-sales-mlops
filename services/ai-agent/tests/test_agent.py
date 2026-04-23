"""
Tests del servicio ai-agent.

Cubre:
  POST /process — validación de entrada y ejecución del agente ReAct
  GET  /health  — sonda de liveness
  GET  /ready   — sonda de readiness
  GET  /metrics — métricas Prometheus

Y la lógica interna del ReActAgent:
  - Respuesta final cuando el LLM no solicita herramientas
  - Comportamiento al alcanzar MAX_ITERATIONS sin respuesta
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: cliente Flask con LiteLLM y Kafka mockeados
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Cliente de test con LiteLLM y Kafka desactivados."""
    with patch("kafka_producer.KafkaProducer"):
        with patch("litellm.completion"):
            import app as agent_app
            agent_app.app.config["TESTING"] = True
            with agent_app.app.test_client() as c:
                yield c


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

def test_health_retorna_200(client):
    """El endpoint /health debe retornar 200 con status ok."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# GET /ready
# ─────────────────────────────────────────────────────────────────────────────

def test_ready_retorna_modelo_activo(client):
    """El endpoint /ready debe indicar que el agente está listo."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ready"
    assert "model" in data


# ─────────────────────────────────────────────────────────────────────────────
# POST /process — validación de entrada
# ─────────────────────────────────────────────────────────────────────────────

def test_process_sin_pregunta_retorna_400(client):
    """Una solicitud sin 'pregunta' debe retornar 400."""
    resp = client.post("/process", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_process_pregunta_vacia_retorna_400(client):
    """Una pregunta vacía debe retornar 400."""
    resp = client.post("/process", json={"pregunta": ""})
    assert resp.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# POST /process — ejecución exitosa del agente
# ─────────────────────────────────────────────────────────────────────────────

def test_process_exito_retorna_respuesta_del_agente(client):
    """Una pregunta válida debe devolver la respuesta estructurada del agente."""
    resultado_mock = {
        "respuesta": "Se predicen ventas de $1.500.000 para Guayas en enero 2026.",
        "datos_usados": {"provincia": "GUAYAS", "mes": 1},
        "prediccion_raw": 1500000.0,
        "razonamiento": "[THINK] ...\n[ACT] ...\n[OBSERVE] ...",
        "_inference_payload": {"provincia": "GUAYAS", "mes": 1, "ano": 2026},
    }

    with patch("app._agent.run", return_value=resultado_mock):
        with patch("app.publish_agent_action"):
            resp = client.post(
                "/process",
                json={"pregunta": "¿Cuánto venderán en Guayas en enero 2026?"},
            )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["respuesta"] == resultado_mock["respuesta"]
    assert data["prediccion_raw"] == 1500000.0
    assert "razonamiento" in data
    # _inference_payload es interno y no debe exponerse al cliente
    assert "_inference_payload" not in data


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_metrics_retorna_contenido_prometheus(client):
    """El endpoint /metrics debe retornar texto en formato Prometheus."""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"agent_requests_total" in resp.data


# ─────────────────────────────────────────────────────────────────────────────
# Lógica interna: ReActAgent
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm_response(content: str, tool_calls=None):
    """Construye un objeto de respuesta de LiteLLM simulado."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def test_react_agent_retorna_respuesta_final_sin_tool_calls():
    """El agente debe retornar cuando el LLM no solicita herramientas."""
    with patch("kafka_producer.KafkaProducer"):
        from agent import ReActAgent

    respuesta_esperada = "Las ventas predichas para octubre 2025 son $900.000."
    llm_resp = _make_llm_response(content=respuesta_esperada)

    with patch("litellm.completion", return_value=llm_resp):
        agente = ReActAgent()
        resultado = agente.run("¿Cuánto venderán en Pichincha en octubre 2025?")

    assert resultado["respuesta"] == respuesta_esperada
    assert resultado["prediccion_raw"] is None
    assert "[THINK]" in resultado["razonamiento"]


def test_react_agent_max_iterations_retorna_mensaje_de_fallback():
    """Al agotar las iteraciones el agente debe retornar el mensaje de fallback."""
    with patch("kafka_producer.KafkaProducer"):
        import agent as agent_module
        from agent import ReActAgent

    # Simula que el LLM siempre responde con un tool_call para forzar el bucle
    tool_call = SimpleNamespace(
        id="tc-001",
        function=SimpleNamespace(name="get_province_data", arguments='{"provincia":"GUAYAS","mes":1}'),
    )
    llm_resp = _make_llm_response(content="", tool_calls=[tool_call])

    observacion_mock = {"provincia": "GUAYAS", "mes": 1, "total_ventas_referencia": 0.0}

    with patch("litellm.completion", return_value=llm_resp):
        with patch.object(ReActAgent, "_dispatch", return_value=observacion_mock):
            agente = ReActAgent()
            original_max = agent_module.MAX_ITERATIONS
            agent_module.MAX_ITERATIONS = 2
            try:
                resultado = agente.run("pregunta de prueba")
            finally:
                agent_module.MAX_ITERATIONS = original_max

    assert "Lo siento" in resultado["respuesta"] or "máximo" in resultado["razonamiento"].lower()
    assert "ADVERTENCIA" in resultado["razonamiento"]
