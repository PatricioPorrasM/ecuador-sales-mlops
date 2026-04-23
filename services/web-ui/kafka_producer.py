"""
Kafka producer for the web-ui service.

Publishes user request events to the 'user-requests' topic so downstream
services and the kafka-consumer can observe traffic patterns and build
audit trails per session.

Degrades gracefully if Kafka is unreachable — the UI never fails because
of a Kafka outage.
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
            "Kafka unavailable (%s) — user request events will not be streamed.", exc
        )
    return _producer


def publish_user_request(
    *,
    session_id: str,
    pregunta: str,
    user_agent: str,
) -> None:
    """
    Publish a user request event. Failures are logged and silently swallowed
    so Kafka outages never degrade the chat experience.
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
        logger.debug("Published user-request event — session=%s", session_id)
    except KafkaError as exc:
        logger.warning("Failed to publish user-request event: %s", exc)
