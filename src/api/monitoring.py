"""Query monitoring and metrics collection for the RAG API."""

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Collects and summarises per-query metrics for the RAG API."""

    _latencies: deque = field(default_factory=lambda: deque(maxlen=1000), repr=False)
    _guardrail_failures: int = field(default=0, repr=False)
    _empty_results: int = field(default=0, repr=False)
    _total: int = field(default=0, repr=False)

    def record(self, latency: float, passed: bool, empty: bool) -> None:
        """Record metrics for a single query.

        Args:
            latency: End-to-end latency in milliseconds.
            passed: Whether all guardrail checks passed.
            empty: Whether the response was empty.
        """
        self._total += 1
        self._latencies.append(latency)
        if not passed:
            self._guardrail_failures += 1
        if empty:
            self._empty_results += 1

    def summary(self) -> dict:
        """Return an aggregate summary of collected metrics.

        Returns:
            Dict with total_queries, guardrail_fail_rate, p50_latency_ms.
        """
        if self._total == 0:
            return {
                "total_queries": 0,
                "guardrail_fail_rate": 0.0,
                "p50_latency_ms": 0.0,
            }

        latency_arr = np.array(self._latencies)
        return {
            "total_queries": self._total,
            "guardrail_fail_rate": self._guardrail_failures / self._total,
            "p50_latency_ms": float(np.percentile(latency_arr, 50)),
            "p95_latency_ms": float(np.percentile(latency_arr, 95)),
            "empty_result_rate": self._empty_results / self._total,
        }
