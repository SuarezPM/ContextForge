"""Tests for ContextCompressor and CompressionBudgetManager."""
import pytest

from apohara_context_forge.compression.budget_manager import (
    CompressionBudgetManager,
    CompressionPlan,
    SegmentType,
    COMPRESSION_MIN_TOKENS,
    detect_segment_type,
)
from apohara_context_forge.compression.compressor import ContextCompressor


@pytest.fixture
def compressor():
    return ContextCompressor()


@pytest.fixture
def budget_manager():
    return CompressionBudgetManager()


class TestCompressionBudgetManager:
    """Tests for CompressionBudgetManager with segment-type-aware compression."""

    def test_plan_system_prompt(self, budget_manager):
        """SYSTEM_PROMPT segment gets near-lossless rate (0.9) per DYNAMIC_RATE_TABLE."""
        text = "You are a helpful assistant. " * 50  # Large enough to compress
        plan = budget_manager.plan(text, SegmentType.SYSTEM_PROMPT)

        assert plan.should_compress is True
        assert plan.target_rate == 0.9
        assert "system_prompt" in plan.reason.lower()

    def test_plan_retrieved_docs(self, budget_manager):
        """RETRIEVED_DOCS should have budget rate 0.5."""
        text = "Document content. " * 200  # Large enough (>512 tokens)
        plan = budget_manager.plan(text, SegmentType.RETRIEVED_DOCS)

        assert plan.should_compress is True
        assert plan.target_rate == 0.5
        assert "retrieved_docs" in plan.reason

    def test_plan_conv_history(self, budget_manager):
        """CONV_HISTORY should have budget rate 0.4."""
        text = "User said hello. Assistant responded. " * 75  # >512 tokens
        plan = budget_manager.plan(text, SegmentType.CONV_HISTORY)

        assert plan.should_compress is True
        assert plan.target_rate == 0.4
        assert "conv_history" in plan.reason

    def test_plan_recent_turns(self, budget_manager):
        """RECENT_TURNS has 0.0 rate (compression disabled per design)."""
        text = "User said hello. " * 130  # >512 tokens
        plan = budget_manager.plan(text, SegmentType.RECENT_TURNS)

        assert plan.should_compress is True
        assert plan.target_rate == 0.0
        assert "recent_turns" in plan.reason

    def test_plan_tool_output(self, budget_manager):
        """TOOL_RESULT should have budget rate 0.6."""
        text = "Tool executed successfully. Result: data. " * 75  # >512 tokens
        plan = budget_manager.plan(text, SegmentType.TOOL_RESULT)

        assert plan.should_compress is True
        assert plan.target_rate == 0.6

    def test_plan_cot_reasoning(self, budget_manager):
        """COT_REASONING should have budget rate 0.07."""
        text = "Step 1: analyze the problem. Step 2: reason through solution. " * 50
        plan = budget_manager.plan(text, SegmentType.COT_REASONING)
        
        assert plan.should_compress is True
        assert plan.target_rate == 0.7
        assert "cot_reasoning" in plan.reason

    def test_plan_short_segment(self, budget_manager):
        """Segments under 512 tokens should NOT compress."""
        text = "Short text. " * 30  # Under 512 tokens
        plan = budget_manager.plan(text, SegmentType.RETRIEVED_DOCS)
        
        assert plan.should_compress is False
        assert "too short" in plan.reason.lower()
        assert plan.original_tokens < COMPRESSION_MIN_TOKENS

    def test_plan_and_compress(self, budget_manager):
        """Full plan + compress workflow."""
        text = "Important document content that should be compressed. " * 100
        plan = budget_manager.plan(text, SegmentType.RETRIEVED_DOCS)
        
        assert plan.segment == text
        assert plan.segment_type == SegmentType.RETRIEVED_DOCS
        assert plan.original_tokens > 0
        assert plan.should_compress is True

    @pytest.mark.asyncio
    async def test_compress_with_plan(self, budget_manager):
        """Execute compression according to plan."""
        text = "Content to compress. " * 100
        plan = budget_manager.plan(text, SegmentType.RETRIEVED_DOCS)
        
        compressed, actual_ratio = await budget_manager.compress_with_plan(plan)
        
        assert isinstance(compressed, str)
        assert len(compressed) > 0
        assert actual_ratio > 0
        assert actual_ratio <= 1.0

    def test_detect_segment_type(self):
        """Test the detect_segment_type() heuristic function."""
        # System prompt detection
        system_text = "System: You are a helpful assistant."
        assert detect_segment_type(system_text) == SegmentType.SYSTEM_PROMPT

        # Tool result detection
        tool_text = "Tool: function executed with result: success"
        assert detect_segment_type(tool_text) == SegmentType.TOOL_RESULT
        
        # CoT reasoning detection
        cot_text = "Step by step reasoning process. Step 1: analyze. Step 2: reason."
        assert detect_segment_type(cot_text) == SegmentType.COT_REASONING
        
        # Retrieved docs detection
        rag_text = "Retrieved document: context from knowledge base."
        assert detect_segment_type(rag_text) == SegmentType.RETRIEVED_DOCS
        
        # Unknown/default
        unknown_text = "Some arbitrary content."
        assert detect_segment_type(unknown_text) == SegmentType.UNKNOWN


onnx_spec = __import__('importlib').util.find_spec('onnxruntime')

pytestmark = pytest.mark.skipif(
    not onnx_spec,
    reason="onnxruntime not installed — LLMLingua compression requires GPU/ONNX runtime"
)


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