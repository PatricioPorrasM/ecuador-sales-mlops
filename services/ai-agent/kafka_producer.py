"""
Kafka producer for the ai-agent service.

Publishes agent action events to the 'agent-actions' topic.
Degrades gracefully if Kafka is unavailable — the agent response is
never blocked by a Kafka failure.
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
        logger.info("Kafka producer connected → %s", KAFKA_BOOTSTRAP_SERVERS)
    except (NoBrokersAvailable, KafkaError) as exc:
        logger.warning(
            "Kafka unavailable (%s) — agent events will not be streamed. "
            "Service continues operating normally.",
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
    Publish an agent action event to the 'agent-actions' topic.
    Failures are logged as warnings and never propagated to callers.
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
        logger.debug("Published agent-action event for %s/%d", provincia_extraida, mes_extraido)
    except KafkaError as exc:
        logger.warning("Failed to publish agent action to Kafka: %s", exc)
