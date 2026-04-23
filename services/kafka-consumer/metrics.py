"""
Prometheus metric definitions for the kafka-consumer service.

All metrics are module-level singletons; import and use them directly.
The prometheus_client HTTP server (started in consumer.py) exposes them
at GET http://0.0.0.0:8003/metrics.

Latency note
────────────
pipeline_end_to_end_latency_seconds is populated from the
`latencia_agente_ms` field of agent-actions events.  That field measures
wall-clock time from the moment the ai-agent received the user question to
the moment it returned the final answer — encompassing every LLM call and
tool execution (including the ml-inference round-trip).  This is the
dominant cost in the pipeline and the most meaningful latency to track.

True end-to-end (browser → response) would additionally include Flask
proxy overhead and network RTT, but those are not observable from the
Kafka stream alone because model-responses events carry no session_id to
correlate with user-requests.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Per-topic message throughput ──────────────────────────────────────────────
kafka_messages_consumed_total = Counter(
    "kafka_messages_consumed_total",
    "Total messages successfully consumed and processed per Kafka topic",
    ["topic"],
)

# ── Consumer health — partition lag ───────────────────────────────────────────
kafka_consumer_lag = Gauge(
    "kafka_consumer_lag",
    "Kafka consumer lag: sum of (end_offset - current_position) across all "
    "partitions for the topic.  Updated every 30 s.  Zero means the consumer "
    "is keeping up with producers.",
    ["topic"],
)

# ── Pipeline latency ──────────────────────────────────────────────────────────
pipeline_end_to_end_latency_seconds = Histogram(
    "pipeline_end_to_end_latency_seconds",
    "End-to-end prediction pipeline latency in seconds, sourced from "
    "agent-actions.latencia_agente_ms (agent processing time including "
    "all LLM calls and ml-inference round-trip).",
    buckets=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0],
)

# ── Business metric — demand by province ─────────────────────────────────────
requests_by_provincia = Counter(
    "requests_by_provincia",
    "Total prediction requests per province, extracted from agent-actions events.",
    ["provincia"],
)
