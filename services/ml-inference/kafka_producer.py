"""
Kafka producer for streaming prediction events to the 'model-responses' topic.

Design: the producer is initialised lazily on the first publish call and
cached for reuse. If Kafka is unreachable the service degrades gracefully —
predictions are still returned to callers, only the stream event is lost.
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
TOPIC: str = "model-responses"

_producer: KafkaProducer | None = None


def _get_producer() -> KafkaProducer | None:
    """Return a cached producer, creating it on first call. Returns None if Kafka is unavailable."""
    global _producer
    if _producer is not None:
        return _producer

    try:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=3,
            request_timeout_ms=5000,
            api_version_auto_timeout_ms=5000,
        )
        logger.info("Kafka producer connected → %s", KAFKA_BOOTSTRAP_SERVERS)
    except (NoBrokersAvailable, KafkaError) as exc:
        logger.warning(
            "Kafka unavailable (%s) — prediction events will not be streamed. "
            "The inference service continues operating normally.",
            exc,
        )
    return _producer


def publish_prediction(
    *,
    provincia: str,
    mes: int,
    prediccion: float,
    modelo_version: str,
    latencia_ms: float,
) -> None:
    """
    Publish a prediction event to the 'model-responses' Kafka topic.
    Failures are logged as warnings; they never propagate to the caller.
    """
    producer = _get_producer()
    if producer is None:
        return

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provincia": provincia,
        "mes": mes,
        "prediccion": prediccion,
        "modelo_version": modelo_version,
        "latencia_ms": latencia_ms,
    }

    try:
        future = producer.send(TOPIC, value=event)
        producer.flush(timeout=2.0)
        future.get(timeout=2.0)
        logger.debug("Published prediction event for %s/%d", provincia, mes)
    except KafkaError as exc:
        logger.warning("Failed to publish prediction event to Kafka: %s", exc)
