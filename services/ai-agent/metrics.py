"""
Prometheus metrics for the ai-agent service.
"""

from prometheus_client import Counter, Histogram

# Total requests by final outcome
agent_requests_total = Counter(
    "agent_requests_total",
    "Total number of requests processed by the ReAct agent",
    ["status"],  # started | success | error
)

# End-to-end wall-clock latency (includes all LLM calls + tool executions)
agent_latency_seconds = Histogram(
    "agent_latency_seconds",
    "End-to-end agent processing latency in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

# Each call to litellm.completion() within the ReAct loop
agent_llm_calls_total = Counter(
    "agent_llm_calls_total",
    "Total LLM API calls made during agent processing",
)

# Each tool dispatched during the ReAct loop
agent_tool_calls_total = Counter(
    "agent_tool_calls_total",
    "Total tool executions by the ReAct agent",
    ["tool_name"],  # get_province_data | call_inference
)
