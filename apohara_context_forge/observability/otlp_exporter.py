from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OTLPExporter:
    """OpenTelemetry Protocol gRPC exporter for INV-15 gate decisions and KV-cache events.

    Optional dependency: opentelemetry-exporter-otlp-proto-grpc.
    If not installed, all methods are no-ops and start() returns None with a WARNING.
    """

    def __init__(self, endpoint: str = "localhost:4317", insecure: bool = False) -> None:
        # insecure default is False: TLS-by-default. For localhost collectors set insecure=True
        # explicitly. Remote collectors over plaintext gRPC leak INV-15 telemetry to any
        # in-path observer.
        self._endpoint = endpoint
        self._insecure = insecure
        if insecure and not self._endpoint.startswith(("localhost", "127.0.0.1", "[::1]")):
            logger.warning(
                "OTLPExporter: insecure=True with non-localhost endpoint %s — "
                "telemetry transmitted in plaintext. Use TLS for remote collectors.",
                self._endpoint,
            )
        self._active = False
        self._build_error: Optional[str] = None
        self._meter = None
        self._provider = None
        self._counters: dict = {}

    def start(self) -> None:
        """Set up the meter provider with a periodic reader (export interval 60s).

        Returns None if opentelemetry-exporter-otlp is not installed, with a single WARNING.
        """
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        except ImportError as exc:
            self._build_error = str(exc)
            logger.warning(
                "opentelemetry-exporter-otlp not installed; OTLP export disabled. "
                "Install with: pip install opentelemetry-exporter-otlp-proto-grpc"
            )
            return None

        try:
            exporter = OTLPMetricExporter(
                endpoint=self._endpoint,
                insecure=self._insecure,
            )
            reader = PeriodicExportingMetricReader(exporter, export_interval_millis=60_000)
            self._provider = MeterProvider(metric_readers=[reader])
            self._meter = self._provider.get_meter("apohara.contextforge")

            self._counters["jcr_decisions"] = self._meter.create_counter(
                name="apohara.jcr_gate_decisions",
                description="Total JCR gate decisions",
                unit="1",
            )
            self._counters["anchor_match"] = self._meter.create_counter(
                name="apohara.anchor_match",
                description="Total anchor pool match attempts",
                unit="1",
            )
            self._counters["lmcache_hit"] = self._meter.create_counter(
                name="apohara.lmcache_hit",
                description="Total LMCache hits",
                unit="1",
            )
            self._active = True
        except Exception as exc:
            self._build_error = str(exc)
            logger.warning("OTLPExporter failed to start: %s", exc)

        return None

    def record_jcr_decision(self, agent_id: str, action: str, risk_score: float) -> None:
        """Record one JCR gate decision into the OTLP counter."""
        if not self._active:
            return
        try:
            self._counters["jcr_decisions"].add(
                1,
                {"agent_id": agent_id, "action": action, "risk_score": str(risk_score)},
            )
        except Exception as exc:
            logger.warning("OTLPExporter.record_jcr_decision failed: %s", exc)

    def record_anchor_match(self, hit: bool) -> None:
        """Record an anchor pool match result into the OTLP counter."""
        if not self._active:
            return
        try:
            result = "hit" if hit else "miss"
            self._counters["anchor_match"].add(1, {"result": result})
        except Exception as exc:
            logger.warning("OTLPExporter.record_anchor_match failed: %s", exc)

    def record_lmcache_hit(self) -> None:
        """Record a LMCache hit into the OTLP counter."""
        if not self._active:
            return
        try:
            self._counters["lmcache_hit"].add(1)
        except Exception as exc:
            logger.warning("OTLPExporter.record_lmcache_hit failed: %s", exc)

    def get_state(self) -> dict:
        """Return introspection state for testing and diagnostics."""
        return {
            "active": self._active,
            "endpoint": self._endpoint,
            "build_error": self._build_error,
        }

    def shutdown(self) -> None:
        """Flush pending metrics and release resources. Idempotent."""
        if not self._active or self._provider is None:
            return
        try:
            self._provider.shutdown()
        except Exception as exc:
            logger.warning("OTLPExporter.shutdown failed: %s", exc)
        finally:
            self._active = False
            self._provider = None
            self._meter = None
            self._counters = {}
