"""
Productor Kafka para el servicio ai-agent.

Publica eventos de acciones del agente en el topic 'agent-actions'.
Degrada de forma segura si Kafka no está disponible — la respuesta del agente
nunca se bloquea por un fallo de Kafka.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS: str = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC: str = "agent-actions"

_producer: KafkaProducer | None = None


def _get_producer() -> KafkaProducer | None:
    """Retorna el productor en caché, creándolo en la primera llamada.

    Retorna None si Kafka no está disponible.
    """
    global _producer
    if _producer is not None:
        return _producer

    try:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
            acks="all",
            retries=3,
            request_timeout_ms=5000,
            api_version_auto_timeout_ms=5000,
        )
        logger.info("Productor Kafka conectado → %s", KAFKA_BOOTSTRAP_SERVERS)
    except (NoBrokersAvailable, KafkaError) as exc:
        logger.warning(
            "Kafka no disponible (%s) — los eventos del agente no serán transmitidos. "
            "El servicio continúa operando normalmente.",
            exc,
        )
    return _producer


def publish_agent_action(
    *,
    pregunta_original: str,
    provincia_extraida: str,
    mes_extraido: int,
    payload_enviado: dict[str, Any],
    latencia_agente_ms: float,
) -> None:
    """
    Publica un evento de acción del agente en el topic 'agent-actions'.
    Los errores se registran como advertencias y nunca se propagan al llamador.
    """
    producer = _get_producer()
    if producer is None:
        return

    event: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pregunta_original": pregunta_original,
        "provincia_extraida": provincia_extraida,
        "mes_extraido": mes_extraido,
        "endpoint_construido": "POST /predict",
        "payload_enviado": payload_enviado,
        "latencia_agente_ms": latencia_agente_ms,
    }

    try:
        future = producer.send(TOPIC, value=event)
        producer.flush(timeout=2.0)
        future.get(timeout=2.0)
        logger.debug("Evento agent-action publicado — %s/%d", provincia_extraida, mes_extraido)
    except KafkaError as exc:
        logger.warning("Error al publicar evento de acción del agente en Kafka: %s", exc)
