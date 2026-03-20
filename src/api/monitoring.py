"""Query monitoring and metrics collection for the RAG API."""

import statistics
from dataclasses import dataclass, field


@dataclass
class QueryMetrics:
    """Tracks query-level metrics for monitoring and alerting."""

    _latencies: list[float] = field(default_factory=list)
    _passed: list[bool] = field(default_factory=list)
    _empty: list[bool] = field(default_factory=list)

    def record(self, latency: float, passed: bool, empty: bool) -> None:
        """Record metrics for a single query."""
        self._latencies.append(latency)
        self._passed.append(passed)
        self._empty.append(empty)

    def summary(self) -> dict:
        """Return aggregate metrics summary."""
        n = len(self._latencies)
        if n == 0:
            return {
                "total_queries": 0,
                "guardrail_fail_rate": 0.0,
                "p50_latency_ms": 0.0,
            }

        fail_count = sum(1 for p in self._passed if not p)
        empty_count = sum(1 for e in self._empty if e)

        return {
            "total_queries": n,
            "guardrail_fail_rate": fail_count / n,
            "p50_latency_ms": statistics.median(self._latencies),
            "avg_latency_ms": statistics.mean(self._latencies),
            "empty_response_rate": empty_count / n,
        }
