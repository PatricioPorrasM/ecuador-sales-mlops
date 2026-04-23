"""
Productor Kafka para el servicio web-ui.

Publica eventos de solicitudes de usuario en el topic 'user-requests' para que
los servicios posteriores y el kafka-consumer puedan observar patrones de tráfico
y construir registros de auditoría por sesión.

Degrada de forma segura si Kafka no está disponible — la UI nunca falla
por una interrupción de Kafka.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS: str = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC: str = "user-requests"

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
            "Kafka no disponible (%s) — los eventos de solicitud de usuario no serán transmitidos.",
            exc,
        )
    return _producer


def publish_user_request(
    *,
    session_id: str,
    pregunta: str,
    user_agent: str,
) -> None:
    """
    Publica un evento de solicitud de usuario en el topic 'user-requests'.
    Los errores se registran y se descartan para que las interrupciones de Kafka
    nunca degraden la experiencia del chat.
    """
    producer = _get_producer()
    if producer is None:
        return

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "pregunta": pregunta,
        "user_agent": user_agent,
    }

    try:
        future = producer.send(TOPIC, value=event)
        producer.flush(timeout=2.0)
        future.get(timeout=2.0)
        logger.debug("Evento user-request publicado — session=%s", session_id)
    except KafkaError as exc:
        logger.warning("Error al publicar evento de solicitud de usuario en Kafka: %s", exc)
