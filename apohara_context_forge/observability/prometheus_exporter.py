from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """Prometheus metrics exporter for INV-15 gate decisions and KV-cache events."""

    def __init__(self) -> None:
        self._active = False
        self._port: Optional[int] = None
        self._metrics_count = 0
        self._registry = None
        self._counters: dict = {}
        self._gauges: dict = {}
        self._prom_available = False
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        try:
            import prometheus_client as prom
            from prometheus_client import CollectorRegistry, Counter, Gauge
            self._prom = prom
            self._registry = CollectorRegistry()
            self._counters["jcr_decisions"] = Counter(
                "apohara_jcr_gate_decisions_total",
                "Total JCR gate decisions",
                ["action", "agent_id"],
                registry=self._registry,
            )
            self._gauges["inv15_risk_score"] = Gauge(
                "apohara_inv15_risk_score",
                "Last observed INV-15 risk score",
                registry=self._registry,
            )
            self._counters["anchor_match"] = Counter(
                "apohara_anchor_match_total",
                "Total anchor pool match attempts",
                ["result"],
                registry=self._registry,
            )
            self._counters["lmcache_hit"] = Counter(
                "apohara_lmcache_hit_total",
                "Total LMCache hits",
                registry=self._registry,
            )
            self._prom_available = True
        except ImportError:
            logger.warning(
                "prometheus_client not installed; Prometheus export disabled"
            )

    def start(self, port: int = 9090) -> Optional[object]:
        """Start the Prometheus HTTP server on the given port."""
        if not self._prom_available:
            logger.warning(
                "prometheus_client not available; cannot start metrics server"
            )
            return None
        try:
            self._prom.start_http_server(port, registry=self._registry)
            self._active = True
            self._port = port
            return self._registry
        except Exception as exc:
            logger.warning("Failed to start Prometheus server: %s", exc)
            return None

    def record_jcr_decision(
        self, agent_id: str, action: str, risk_score: float
    ) -> None:
        """Record one JCR gate decision (increments counter, sets risk gauge)."""
        self._metrics_count += 1
        if self._prom_available:
            self._counters["jcr_decisions"].labels(
                action=action, agent_id=agent_id
            ).inc()
            self._gauges["inv15_risk_score"].set(risk_score)

    def record_anchor_match(self, hit: bool) -> None:
        """Record an anchor pool match result."""
        self._metrics_count += 1
        if self._prom_available:
            result = "hit" if hit else "miss"
            self._counters["anchor_match"].labels(result=result).inc()

    def record_lmcache_hit(self) -> None:
        """Record a LMCache hit."""
        self._metrics_count += 1
        if self._prom_available:
            self._counters["lmcache_hit"].inc()

    def get_state(self) -> dict:
        """Return introspection state for testing and dashboards."""
        return {
            "active": self._active,
            "port": self._port,
            "metrics_count": self._metrics_count,
        }
