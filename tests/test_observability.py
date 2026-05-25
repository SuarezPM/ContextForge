from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import directly from submodule paths to avoid triggering the heavy
# apohara_context_forge/__init__.py (which pulls in numpy, faiss, etc.)
from apohara_context_forge.observability.prometheus_exporter import PrometheusExporter
from apohara_context_forge.observability.audit_log import AuditLog
import apohara_context_forge.observability.recorders as recorders_mod


# ---------------------------------------------------------------------------
# Test 1: honest fallback when prometheus_client not installed
# ---------------------------------------------------------------------------

def test_prometheus_exporter_honest_fallback_when_missing(caplog):
    """PrometheusExporter.start() returns None + logs WARNING when prometheus_client absent."""
    import importlib
    import apohara_context_forge.observability.prometheus_exporter as prom_mod

    # Patch prometheus_client away inside the module under test.
    with patch.object(prom_mod, "logger") as mock_logger:
        exporter = PrometheusExporter.__new__(PrometheusExporter)
        exporter._active = False
        exporter._port = None
        exporter._metrics_count = 0
        exporter._registry = None
        exporter._counters = {}
        exporter._gauges = {}
        exporter._prom_available = False  # simulate absent prometheus_client

        result = exporter.start(port=9999)

    assert result is None
    mock_logger.warning.assert_called()


# ---------------------------------------------------------------------------
# Test 2: metrics_count increments with each record call
# ---------------------------------------------------------------------------

def test_prometheus_exporter_metrics_increment():
    """get_state()['metrics_count'] increments with each record call."""
    exporter = PrometheusExporter()
    assert exporter.get_state()["metrics_count"] == 0
    exporter.record_jcr_decision(agent_id="agent-1", action="allow", risk_score=0.3)
    exporter.record_jcr_decision(agent_id="agent-2", action="block", risk_score=0.9)
    exporter.record_jcr_decision(agent_id="agent-3", action="invalidate", risk_score=0.7)
    assert exporter.get_state()["metrics_count"] == 3


# ---------------------------------------------------------------------------
# Test 3: AuditLog writes JSONL and replay() returns all records with ts
# ---------------------------------------------------------------------------

def test_audit_log_writes_jsonl(tmp_path):
    """AuditLog writes records as JSONL; replay() returns all 5 with ts field."""
    log_path = tmp_path / "audit.jsonl"
    audit = AuditLog(log_path)
    for i in range(5):
        audit.record({"kind": "inv15_gate", "seq": i})
    records = list(audit.replay())
    assert len(records) == 5
    for rec in records:
        assert "ts" in rec
        assert "kind" in rec


# ---------------------------------------------------------------------------
# Test 4: disk error is swallowed — no exception, WARNING logged once
# ---------------------------------------------------------------------------

def test_audit_log_handles_disk_error_gracefully(tmp_path, caplog):
    """AuditLog.record() swallows OSError and warns once per session."""
    from pathlib import Path as _Path
    log_path = tmp_path / "audit.jsonl"
    audit = AuditLog(log_path)
    with patch.object(_Path, "open", side_effect=OSError("disk full")):
        with caplog.at_level(logging.WARNING, logger="apohara_context_forge.observability.audit_log"):
            audit.record({"kind": "inv15_gate"})
            audit.record({"kind": "inv15_gate"})  # second call should NOT warn again
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Test 5: record_inv15_decision fans out to both exporter and audit log
# ---------------------------------------------------------------------------

def test_record_inv15_decision_fans_out(tmp_path):
    """record_inv15_decision() calls both PrometheusExporter and AuditLog."""
    mock_exporter = MagicMock()
    mock_audit = MagicMock()

    # Inject mocks into singleton slots
    orig_exporter = recorders_mod._exporter
    orig_audit = recorders_mod._audit_log
    recorders_mod._exporter = mock_exporter
    recorders_mod._audit_log = mock_audit

    try:
        recorders_mod.record_inv15_decision(
            agent_id="critic",
            anchor_hash="abc123",
            risk_score=0.85,
            gate_action="block",
            predicted_jcr_delta=0.12,
            lmcache_consulted=True,
            lmcache_hit=True,
        )
    finally:
        recorders_mod._exporter = orig_exporter
        recorders_mod._audit_log = orig_audit

    mock_exporter.record_jcr_decision.assert_called_once_with(
        agent_id="critic", action="block", risk_score=0.85
    )
    mock_exporter.record_lmcache_hit.assert_called_once()
    mock_audit.record.assert_called_once()
    call_args = mock_audit.record.call_args[0][0]
    assert call_args["agent_id"] == "critic"
    assert call_args["gate_action"] == "block"
    assert call_args["kind"] == "inv15_gate"


# ---------------------------------------------------------------------------
# Test 6: replay() preserves insertion order
# ---------------------------------------------------------------------------

def test_audit_log_replay_preserves_order(tmp_path):
    """replay() returns records in the same order they were written."""
    log_path = tmp_path / "audit_order.jsonl"
    audit = AuditLog(log_path)
    for i in range(10):
        audit.record({"seq": i})
    records = list(audit.replay())
    assert len(records) == 10
    for idx, rec in enumerate(records):
        assert rec["seq"] == idx

# ---------------------------------------------------------------------------
# Test 7: record_agent_ttft handles large outliers correctly
# ---------------------------------------------------------------------------

def test_record_agent_ttft_large_outlier():
    """record_agent_ttft() handles extremely large outliers without throwing exceptions."""
    from apohara_context_forge.metrics.prometheus_metrics import record_agent_ttft
    from prometheus_client import REGISTRY

    agent_id = "outlier-agent"
    thinking_mode = "cot"
    outlier_val = 1e9  # 1 million seconds, well beyond the 10000ms max bucket

    # Should not throw an exception
    record_agent_ttft(agent_id=agent_id, thinking_mode=thinking_mode, ttft_ms=outlier_val)

    # Verify the sample hit the +Inf bucket
    found = False
    for metric_family in REGISTRY.collect():
        if metric_family.name == "contextforge_agent_ttft_ms":
            for sample in metric_family.samples:
                if sample.name == "contextforge_agent_ttft_ms_bucket":
                    if sample.labels.get("agent_id") == agent_id and sample.labels.get("thinking_mode") == thinking_mode:
                        if sample.labels.get("le") == "+Inf":
                            assert sample.value > 0
                            found = True
    assert found, "Expected to find a +Inf bucket entry for the outlier"
