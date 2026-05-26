import pytest
pytest.importorskip("z3")
from apohara_context_forge.safety.inv15_certifier import certify_decision

def test_correct_critic_dense_is_certified():
    # critic, 5 candidates, reuse 0.75 -> risk 0.9 > 0.7 -> INV-15 mandates dense=True
    c = certify_decision(agent_role="critic", candidate_count=5, reuse_rate=0.75,
                         layout_shuffled=False, use_dense=True)
    assert c["satisfies_inv15"] is True
    assert c["z3_status"] == "unsat"

def test_critic_skipping_dense_is_flagged():
    c = certify_decision(agent_role="critic", candidate_count=5, reuse_rate=0.75,
                         layout_shuffled=False, use_dense=False)
    assert c["satisfies_inv15"] is False

def test_nonjudge_neverdense_is_certified():
    c = certify_decision(agent_role="retriever", candidate_count=10, reuse_rate=1.0,
                         layout_shuffled=True, use_dense=False)
    assert c["satisfies_inv15"] is True

def test_out_of_domain_inputs_raise():
    with pytest.raises(ValueError):
        certify_decision(agent_role="critic", candidate_count=-5, reuse_rate=0.5,
                         layout_shuffled=False, use_dense=True)
    with pytest.raises(ValueError):
        certify_decision(agent_role="critic", candidate_count=5, reuse_rate=1.5,
                         layout_shuffled=False, use_dense=True)

def test_return_contract_keys():
    c = certify_decision(agent_role="CRITIC", candidate_count=5, reuse_rate=0.75,
                         layout_shuffled=False, use_dense=True)
    for k in ("satisfies_inv15","agent_role","candidate_count","reuse_rate",
              "layout_shuffled","observed_use_dense","z3_status","elapsed_ms","z3_version"):
        assert k in c
    assert c["agent_role"] == "critic" and c["observed_use_dense"] is True
