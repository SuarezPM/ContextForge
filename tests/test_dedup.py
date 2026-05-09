"""Tests for SemanticDedupEngine."""
import pytest

from contextforge.dedup.dedup_engine import SemanticDedupEngine


@pytest.fixture
def dedup_engine():
    return SemanticDedupEngine()


class TestSemanticDedupEngine:
    """Tests for semantic deduplication."""

    async def test_embed(self, dedup_engine):
        embedding = await dedup_engine.embed("This is a test sentence")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    async def test_similarity_same_text(self, dedup_engine):
        text = "This is a test sentence"
        emb1 = await dedup_engine.embed(text)
        emb2 = await dedup_engine.embed(text)
        similarity = await dedup_engine.similarity(emb1, emb2)
        assert similarity > 0.99  # Nearly identical

    async def test_similarity_different_text(self, dedup_engine):
        emb1 = await dedup_engine.embed("Machine learning is great")
        emb2 = await dedup_engine.embed("The weather is nice today")
        similarity = await dedup_engine.similarity(emb1, emb2)
        assert 0 <= similarity <= 1.0

    async def test_find_shared_prefix(self, dedup_engine):
        shared = await dedup_engine.find_shared_prefix(
            "This is a test context with specific information",
            "This is a test context with different information",
        )
        assert shared.startswith("This is a")
        assert "different" not in shared

    async def test_find_shared_prefix_no_overlap(self, dedup_engine):
        shared = await dedup_engine.find_shared_prefix(
            "Hello world",
            "Goodbye world",
        )
        # Should find common prefix at start
        words = shared.split()
        assert len(words) <= 1 or "Hello" in shared or "Goodbye" in shared

    async def test_batch_deduplicate(self, dedup_engine):
        contexts = [
            "This is the first document about AI",
            "This is the first document about ML",
            "Completely different topic here",
        ]
        results = await dedup_engine.batch_deduplicate(contexts)
        assert isinstance(results, dict)
        assert "context_0" in results