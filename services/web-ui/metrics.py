"""
Prometheus metrics for the web-ui service.
"""

from prometheus_client import Counter, Histogram

# All HTTP requests by endpoint and final status
web_requests_total = Counter(
    "web_requests_total",
    "Total HTTP requests received by the web-ui",
    ["endpoint", "status"],  # status: success | timeout | agent_error | connection_error
)

# End-to-end latency of /chat (includes agent round-trip)
web_chat_latency_seconds = Histogram(
    "web_chat_latency_seconds",
    "End-to-end /chat request latency in seconds (includes ai-agent call)",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)
