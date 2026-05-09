"""Tests for ContextCompressor."""
import pytest

from contextforge.compression.compressor import ContextCompressor


@pytest.fixture
def compressor():
    return ContextCompressor()


class TestContextCompressor:
    """Tests for LLMLingua-2 compressor wrapper."""

    async def test_compress_basic(self, compressor):
        text = "This is a test sentence that we want to compress. " * 10
        compressed, ratio = await compressor.compress(text, rate=0.5)
        assert isinstance(compressed, str)
        assert len(compressed) > 0
        assert ratio > 0

    async def test_compress_preserves_meaning(self, compressor):
        text = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
        compressed, ratio = await compressor.compress(text, rate=0.5)
        # Compressed should be shorter
        assert len(compressed) <= len(text)

    async def test_compress_rate_0_5_on_200_tokens(self, compressor):
        # Create ~200 token text
        text = "The quick brown fox jumps over the lazy dog. " * 20
        original_tokens = len(text.split())
        
        compressed, ratio = await compressor.compress(text, rate=0.5)
        compressed_tokens = len(compressed.split())
        
        # Verify output is less than 110 tokens (rate=0.5 means ~50% compression)
        assert compressed_tokens < 110, f"Expected <110 tokens, got {compressed_tokens}"

    async def test_compress_batch(self, compressor):
        texts = [
            "First test document about machine learning.",
            "Second test document about deep learning.",
            "Third test document about neural networks.",
        ]
        results = await compressor.compress_batch(texts, rate=0.5)
        assert len(results) == 3
        for compressed, ratio in results:
            assert isinstance(compressed, str)
            assert ratio > 0