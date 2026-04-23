"""
kafka-consumer: aggregates cross-service events and exposes Prometheus metrics.

Topics consumed
───────────────
  user-requests    {timestamp, session_id, pregunta, user_agent}
  agent-actions    {timestamp, pregunta_original, provincia_extraida,
                    mes_extraido, endpoint_construido, payload_enviado,
                    latencia_agente_ms}
  model-responses  {timestamp, provincia, mes, prediccion,
                    modelo_version, latencia_ms}

Metrics exposed at  http://0.0.0.0:8003/metrics
(prometheus_client's start_http_server handles the HTTP layer — no Flask needed)

Design notes
────────────
• Consumer lag is re-computed every LAG_UPDATE_SECS seconds by querying
  end_offsets() on the assigned partitions and comparing with position().
• pipeline_end_to_end_latency_seconds is sourced from agent-actions.latencia_agente_ms
  because model-responses events carry no session_id for cross-topic correlation.
  This captures the dominant cost: LLM calls + tool executions + ml-inference.
• Graceful shutdown on SIGTERM/SIGINT: the poll loop exits after at most
  POLL_TIMEOUT_MS milliseconds, then the consumer commits offsets and closes.
• Connection retries with exponential back-off (capped at 60 s) let the
  container survive Kafka startup races in docker-compose / K8s.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

from kafka import KafkaConsumer, TopicPartition
from kafka.errors import KafkaError, NoBrokersAvailable
from prometheus_client import start_http_server

from metrics import (
    kafka_consumer_lag,
    kafka_messages_consumed_total,
    pipeline_end_to_end_latency_seconds,
    requests_by_provincia,
)

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS: str = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
CONSUMER_GROUP: str          = os.environ.get("CONSUMER_GROUP", "mlops-metrics-consumer")
AUTO_OFFSET_RESET: str       = os.environ.get("AUTO_OFFSET_RESET", "latest")
METRICS_PORT: int            = int(os.environ.get("METRICS_PORT", "8003"))
POLL_TIMEOUT_MS: int         = 1_000   # max wait per poll call
LAG_UPDATE_SECS: float       = 30.0   # how often to recompute partition lag

TOPICS: list[str] = ["user-requests", "agent-actions", "model-responses"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Shutdown flag
# ─────────────────────────────────────────────────────────

_running = True


def _register_signals() -> None:
    def _handler(sig, _frame):
        global _running
        logger.info("Signal %s received — initiating graceful shutdown", sig)
        _running = False

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


# ─────────────────────────────────────────────────────────
# Kafka connection with retry
# ─────────────────────────────────────────────────────────

def _connect(max_retries: int = 15) -> KafkaConsumer:
    """
    Attempt to connect with exponential back-off (2 s → 60 s cap).
    Exits the process if all retries are exhausted — the container
    orchestrator will restart it.
    """
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            consumer = KafkaConsumer(
                *TOPICS,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id=CONSUMER_GROUP,
                value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
                auto_offset_reset=AUTO_OFFSET_RESET,
                enable_auto_commit=True,
                auto_commit_interval_ms=5_000,
                session_timeout_ms=30_000,
                heartbeat_interval_ms=10_000,
                max_poll_interval_ms=300_000,
            )
            logger.info(
                "Kafka consumer connected — group=%s  topics=%s", CONSUMER_GROUP, TOPICS
            )
            return consumer
        except (NoBrokersAvailable, KafkaError) as exc:
            logger.warning(
                "Connection attempt %d/%d failed (%s) — retrying in %.0f s",
                attempt, max_retries, exc, delay,
            )
            if attempt < max_retries:
                time.sleep(delay)
                delay = min(delay * 1.5, 60.0)

    logger.error("All Kafka connection attempts exhausted. Exiting.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────
# Lag computation
# ─────────────────────────────────────────────────────────

def _update_lag(consumer: KafkaConsumer) -> None:
    """
    Compute and record consumer lag for every assigned partition.
    Lag = end_offset - current_position  (0 means fully caught up).
    Errors are suppressed so a transient Kafka blip does not crash the loop.
    """
    for topic in TOPICS:
        try:
            partitions = consumer.partitions_for_topic(topic) or set()
            tps = [TopicPartition(topic, p) for p in partitions]
            if not tps:
                continue

            end_offsets: dict = consumer.end_offsets(tps)
            total_lag = sum(
                max(0, end_offsets[tp] - consumer.position(tp))
                for tp in tps
                if tp in end_offsets
            )
            kafka_consumer_lag.labels(topic=topic).set(total_lag)
            logger.debug("Lag updated — topic=%s  lag=%d", topic, total_lag)

        except Exception as exc:
            logger.debug("Lag computation skipped for %s: %s", topic, exc)


# ─────────────────────────────────────────────────────────
# Per-topic message handlers
# ─────────────────────────────────────────────────────────

def _handle_user_request(payload: dict) -> None:
    """
    user-requests: {timestamp, session_id, pregunta, user_agent}
    No metrics are updated here directly — province is not yet known.
    The record is available for future cross-topic correlation if
    session_id is added to downstream events.
    """
    logger.debug(
        "user-request — session=%s  pregunta=%r",
        payload.get("session_id", "?"),
        str(payload.get("pregunta", ""))[:60],
    )


def _handle_agent_action(payload: dict) -> None:
    """
    agent-actions: {timestamp, provincia_extraida, latencia_agente_ms, ...}

    Updates:
      • requests_by_provincia  — demand per province
      • pipeline_end_to_end_latency_seconds — agent processing time as
        the best available proxy for full pipeline latency
    """
    provincia = str(payload.get("provincia_extraida") or "unknown").strip().upper()
    requests_by_provincia.labels(provincia=provincia).inc()

    raw_ms = payload.get("latencia_agente_ms")
    if raw_ms is not None:
        try:
            pipeline_end_to_end_latency_seconds.observe(float(raw_ms) / 1_000.0)
        except (TypeError, ValueError) as exc:
            logger.warning("Invalid latencia_agente_ms value %r: %s", raw_ms, exc)

    logger.debug(
        "agent-action — provincia=%s  latency_ms=%s",
        provincia, raw_ms,
    )


def _handle_model_response(payload: dict) -> None:
    """
    model-responses: {timestamp, provincia, mes, prediccion, modelo_version, latencia_ms}

    No additional metrics are incremented here — prediction throughput is
    already captured by kafka_messages_consumed_total{topic='model-responses'}
    and province demand by requests_by_provincia (sourced from agent-actions).
    """
    logger.debug(
        "model-response — provincia=%s  mes=%s  prediccion=%s  model=%s",
        payload.get("provincia"), payload.get("mes"),
        payload.get("prediccion"), payload.get("modelo_version"),
    )


_HANDLERS: dict = {
    "user-requests":  _handle_user_request,
    "agent-actions":  _handle_agent_action,
    "model-responses":_handle_model_response,
}


# ─────────────────────────────────────────────────────────
# Main consumer loop
# ─────────────────────────────────────────────────────────

def _run(consumer: KafkaConsumer) -> None:
    processed = 0
    last_lag_ts = time.monotonic()

    # Initialise lag metrics to 0 so Prometheus shows them immediately
    for topic in TOPICS:
        kafka_consumer_lag.labels(topic=topic).set(0)

    logger.info("Consumer loop started — polling every %d ms", POLL_TIMEOUT_MS)

    try:
        while _running:
            # poll() returns a dict {TopicPartition: [ConsumerRecord, ...]}
            batch: dict = consumer.poll(timeout_ms=POLL_TIMEOUT_MS)

            for _tp, records in batch.items():
                for record in records:
                    topic = record.topic
                    try:
                        payload = record.value
                        if not isinstance(payload, dict):
                            logger.debug("Non-dict payload skipped on %s", topic)
                            continue
                    except Exception as exc:
                        logger.warning("Deserialise error on %s offset %d: %s", topic, record.offset, exc)
                        continue

                    kafka_messages_consumed_total.labels(topic=topic).inc()

                    handler = _HANDLERS.get(topic)
                    if handler:
                        handler(payload)
                    else:
                        logger.debug("No handler for topic %s", topic)

                    processed += 1

            # Periodic lag update (time-based, not message-count-based)
            now = time.monotonic()
            if now - last_lag_ts >= LAG_UPDATE_SECS:
                _update_lag(consumer)
                last_lag_ts = now

    finally:
        logger.info("Committing offsets and closing consumer (processed=%d)…", processed)
        try:
            consumer.commit()
        except Exception:
            pass
        consumer.close()
        logger.info("Consumer closed.")


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main() -> None:
    _register_signals()

    logger.info("Starting Prometheus HTTP server on port %d", METRICS_PORT)
    start_http_server(METRICS_PORT)

    logger.info("Connecting to Kafka at %s…", KAFKA_BOOTSTRAP_SERVERS)
    consumer = _connect()

    _run(consumer)
    logger.info("kafka-consumer service stopped.")


if __name__ == "__main__":
    main()
