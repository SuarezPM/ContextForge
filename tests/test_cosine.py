import numpy as np
from apohara_context_forge.dedup.cosine import VectorizedSimilarity


def test_vectorized_similarity_clear():
    """Clearing the VectorizedSimilarity index resets its internal state."""
    similarity = VectorizedSimilarity(dim=384)
    agent_id = "agent1"
    embedding = np.random.randn(384).tolist()
    similarity.index(agent_id, embedding)

    assert similarity.size == 1
    assert similarity._candidates is not None
    assert similarity._candidate_ids == [agent_id]

    similarity.clear()

    assert similarity.size == 0
    assert similarity._candidates is None
    assert similarity._candidate_ids == []
