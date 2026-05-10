"""Tests for EmbeddingEngine — TASK-001."""
import pytest
import numpy as np
from apohara_context_forge.embeddings.embedding_engine import EmbeddingEngine

faiss_spec = __import__('importlib').util.find_spec
pytestmark = pytest.mark.skipif(
    not faiss_spec('onnxruntime'),
    reason="onnxruntime not installed — GPU/DevCloud environment required"
)


@pytest.fixture
async def engine():
    """Get EmbeddingEngine singleton."""
    return await EmbeddingEngine.get_instance(dim=512, use_onnx=False)


class TestEmbeddingEngine:
    """Tests for EmbeddingEngine core functionality."""

    @pytest.mark.asyncio
    async def test_get_instance_returns_singleton(self, engine):
        """get_instance() returns the same instance on repeated calls."""
        engine2 = await EmbeddingEngine.get_instance()
        assert engine is engine2

    @pytest.mark.asyncio
    async def test_encode_returns_normalized_vector(self, engine):
        """encode() returns L2-normalized embedding."""
        embedding = await engine.encode("test prompt")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == 512  # dim=512
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_encode_batch_returns_list(self, engine):
        """encode_batch() returns list of embeddings."""
        texts = ["prompt one", "prompt two", "prompt three"]
        embeddings = await engine.encode_batch(texts)
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        for emb in embeddings:
            assert isinstance(emb, np.ndarray)
            assert emb.shape[0] == 512

    @pytest.mark.asyncio
    async def test_simhash_returns_int(self, engine):
        """simhash() returns 64-bit integer."""
        token_ids = [101, 2003, 1996, 3007, 102]
        h = await engine.simhash(token_ids)
        assert isinstance(h, int)
        assert h >= 0

    @pytest.mark.asyncio
    async def test_simhash_deterministic(self, engine):
        """simhash() is deterministic for same input."""
        token_ids = [101, 2003, 1996, 3007, 102]
        h1 = await engine.simhash(token_ids)
        h2 = await engine.simhash(token_ids)
        assert h1 == h2

    @pytest.mark.asyncio
    async def test_simhash_different_for_different_inputs(self, engine):
        """simhash() returns different values for different token sequences."""
        h1 = await engine.simhash([101, 2003, 1996])
        h2 = await engine.simhash([101, 3007, 102])
        assert h1 != h2

    @pytest.mark.asyncio
    async def test_encode_caching(self, engine):
        """Identical text returns cached embedding."""
        text = "shared system prompt"
        e1 = await engine.encode(text)
        e2 = await engine.encode(text)
        assert np.allclose(e1, e2)