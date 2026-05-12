from __future__ import annotations

import os
from typing import Optional

from apohara_context_forge.observability.prometheus_exporter import PrometheusExporter
from apohara_context_forge.observability.audit_log import AuditLog

_exporter: Optional[PrometheusExporter] = None
_audit_log: Optional[AuditLog] = None


def _get_exporter() -> PrometheusExporter:
    global _exporter
    if _exporter is None:
        _exporter = PrometheusExporter()
    return _exporter


def _get_audit_log() -> AuditLog:
    global _audit_log
    if _audit_log is None:
        audit_dir = os.environ.get("APOHARA_OBSERVABILITY_DIR", "./.apohara/audit")
        _audit_log = AuditLog(f"{audit_dir}/inv15.jsonl")
    return _audit_log


def record_inv15_decision(
    *,
    agent_id: str,
    anchor_hash: str,
    risk_score: float,
    gate_action: str,
    predicted_jcr_delta: float,
    lmcache_consulted: bool = False,
    lmcache_hit: bool = False,
) -> None:
    """Fan out an INV-15 gate decision to both PrometheusExporter and AuditLog."""
    exporter = _get_exporter()
    audit_log = _get_audit_log()

    exporter.record_jcr_decision(
        agent_id=agent_id, action=gate_action, risk_score=risk_score
    )
    if lmcache_hit:
        exporter.record_lmcache_hit()

    audit_log.record({
        "kind": "inv15_gate",
        "agent_id": agent_id,
        "anchor_hash": anchor_hash,
        "risk_score": risk_score,
        "gate_action": gate_action,
        "predicted_jcr_delta": predicted_jcr_delta,
        "lmcache_consulted": lmcache_consulted,
        "lmcache_hit": lmcache_hit,
    })
