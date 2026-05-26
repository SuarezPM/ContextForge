import pytest
pytest.importorskip("z3")
from apohara_context_forge.observability import recorders
from apohara_context_forge.observability.ledger import Ledger
from apohara_context_forge.safety.jcr_gate import JCRSafetyGate

def test_gate_emits_certified_ledger_when_flag_set(tmp_path, monkeypatch):
    monkeypatch.setenv("APOHARA_OBSERVABILITY_DIR", str(tmp_path))
    monkeypatch.setenv("APOHARA_FORGE_LEDGER", "1")
    recorders._reset_singletons()
    gate = JCRSafetyGate(jcr_threshold=0.7)
    d = gate.gate_decision(agent_role="critic", candidate_count=5,
                           reuse_rate=0.75, layout_shuffled=False)
    assert d.use_dense is True
    v = Ledger(tmp_path / "inv15_ledger.jsonl").verify()
    assert v["valid"] is True and v["entries"] == 1

def test_gate_no_ledger_when_flag_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("APOHARA_OBSERVABILITY_DIR", str(tmp_path))
    monkeypatch.delenv("APOHARA_FORGE_LEDGER", raising=False)
    recorders._reset_singletons()
    gate = JCRSafetyGate(jcr_threshold=0.7)
    gate.gate_decision(agent_role="critic", candidate_count=5,
                       reuse_rate=0.75, layout_shuffled=False)
    assert not (tmp_path / "inv15_ledger.jsonl").exists()
