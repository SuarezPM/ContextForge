import numpy as np
import pytest
from apohara_context_forge.dedup.cosine import VectorizedSimilarity

def test_vectorized_similarity_clear():
    """Test that clearing the VectorizedSimilarity index resets its internal state."""
    # 1. Arrange: Create a VectorizedSimilarity object and add some embeddings to the index
    similarity = VectorizedSimilarity(dim=384)
    agent_id = "agent1"
    embedding = np.random.randn(384).tolist()
    similarity.index(agent_id, embedding)

    # Verify the initial state
    assert similarity.size == 1
    assert similarity._candidates is not None
    assert similarity._candidate_ids == [agent_id]

    # 2. Act: Call clear()
    similarity.clear()

    # 3. Assert: Verify the state has been correctly reset
    assert similarity.size == 0
    assert similarity._candidates is None
    assert similarity._candidate_ids == []
