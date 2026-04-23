"""
Prometheus metrics for the ml-inference service.

All metrics are module-level singletons; import and use them directly.
The /metrics endpoint calls generate_latest() from prometheus_client to
serialise them into the Prometheus text exposition format.
"""

from prometheus_client import Counter, Gauge, Histogram

# Total requests by province and outcome (success | error | unknown_province)
inference_requests_total = Counter(
    "inference_requests_total",
    "Total number of prediction requests received",
    ["provincia", "status"],
)

# End-to-end latency from request receipt to response (seconds)
inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "End-to-end inference latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# Most recent raw prediction value per province (useful for drift detection)
model_prediction_value = Gauge(
    "model_prediction_value",
    "Most recent predicted total_ventas_sociedades value",
    ["provincia"],
)

# Distribution of confidence scores across all requests
model_confidence_score = Histogram(
    "model_confidence_score",
    "Distribution of model confidence scores [0, 1]",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Marks which model version is currently loaded (label carries the version string)
model_version_info = Gauge(
    "model_version_info",
    "Loaded model version (1 = active)",
    ["version"],
)
