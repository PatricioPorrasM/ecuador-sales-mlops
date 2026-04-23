"""
Productor Kafka para transmitir eventos de predicción al topic 'model-responses'.

Diseño: el productor se inicializa de forma diferida en la primera llamada
y se almacena en caché para reutilización. Si Kafka no está disponible, el
servicio degrada de forma segura — las predicciones se devuelven igualmente
al llamador; solo se pierde el evento de streaming.
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
    """Retorna el productor en caché, creándolo en la primera llamada.

    Retorna None si Kafka no está disponible.
    """
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
        logger.info("Productor Kafka conectado → %s", KAFKA_BOOTSTRAP_SERVERS)
    except (NoBrokersAvailable, KafkaError) as exc:
        logger.warning(
            "Kafka no disponible (%s) — los eventos de predicción no serán transmitidos. "
            "El servicio de inferencia continúa operando normalmente.",
            exc,
        )
    return _producer


def publish_prediction(
    *,
    provincia: str,
    mes: int,
    ano_fiscal: int,
    prediccion: float,
    modelo_version: str,
    latencia_ms: float,
) -> None:
    """
    Publica un evento de predicción en el topic 'model-responses' de Kafka.
    Los errores se registran como advertencias y nunca se propagan al llamador.
    """
    producer = _get_producer()
    if producer is None:
        return

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provincia": provincia,
        "ano_fiscal": ano_fiscal,
        "mes": mes,
        "prediccion": prediccion,
        "modelo_version": modelo_version,
        "latencia_ms": latencia_ms,
    }

    try:
        future = producer.send(TOPIC, value=event)
        producer.flush(timeout=2.0)
        future.get(timeout=2.0)
        logger.debug("Evento de predicción publicado — %s/%d", provincia, mes)
    except KafkaError as exc:
        logger.warning("Error al publicar evento de predicción en Kafka: %s", exc)
