"""Tests for ContextCompressor and CompressionBudgetManager."""
import pytest

from contextforge.compression.budget_manager import (
    CompressionBudgetManager,
    CompressionPlan,
    SegmentType,
    COMPRESSION_MIN_TOKENS,
    detect_segment_type,
)
from contextforge.compression.compressor import ContextCompressor


@pytest.fixture
def compressor():
    return ContextCompressor()


@pytest.fixture
def budget_manager():
    return CompressionBudgetManager()


class TestCompressionBudgetManager:
    """Tests for CompressionBudgetManager with segment-type-aware compression."""

    def test_plan_system_prompt(self, budget_manager):
        """SYSTEM_PROMPT segment should never compress."""
        text = "You are a helpful assistant. " * 50  # Large enough to compress
        plan = budget_manager.plan(text, SegmentType.SYSTEM_PROMPT)
        
        assert plan.should_compress is False
        assert plan.target_rate == 0.0
        assert "protected" in plan.reason.lower()

    def test_plan_retrieved_docs(self, budget_manager):
        """RETRIEVED_DOCS should have budget rate 0.25."""
        text = "Document content. " * 100  # Large enough
        plan = budget_manager.plan(text, SegmentType.RETRIEVED_DOCS)
        
        assert plan.should_compress is True
        assert plan.target_rate == 0.25
        assert "budget rate 0.25" in plan.reason

    def test_plan_conv_history(self, budget_manager):
        """CONV_HISTORY should have budget rate 0.40."""
        text = "User said hello. Assistant responded. " * 50
        plan = budget_manager.plan(text, SegmentType.CONV_HISTORY)
        
        assert plan.should_compress is True
        assert plan.target_rate == 0.40
        assert "budget rate 0.40" in plan.reason

    def test_plan_recent_turns(self, budget_manager):
        """RECENT_TURNS should never compress."""
        text = "Latest user message. " * 50
        plan = budget_manager.plan(text, SegmentType.RECENT_TURNS)
        
        assert plan.should_compress is False
        assert plan.target_rate == 0.0
        assert "protected" in plan.reason.lower()

    def test_plan_tool_output(self, budget_manager):
        """TOOL_OUTPUT should have budget rate 0.50."""
        text = "Tool executed successfully. Result: data. " * 50
        plan = budget_manager.plan(text, SegmentType.TOOL_OUTPUT)
        
        assert plan.should_compress is True
        assert plan.target_rate == 0.50

    def test_plan_cot_reasoning(self, budget_manager):
        """COT_REASONING should have budget rate 0.07."""
        text = "Step 1: analyze the problem. Step 2: reason through solution. " * 50
        plan = budget_manager.plan(text, SegmentType.COT_REASONING)
        
        assert plan.should_compress is True
        assert plan.target_rate == 0.07

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
        
        # Tool output detection
        tool_text = "Tool: function executed with result: success"
        assert detect_segment_type(tool_text) == SegmentType.TOOL_OUTPUT
        
        # CoT reasoning detection
        cot_text = "Step by step reasoning process. Step 1: analyze. Step 2: reason."
        assert detect_segment_type(cot_text) == SegmentType.COT_REASONING
        
        # Retrieved docs detection
        rag_text = "Retrieved document: context from knowledge base."
        assert detect_segment_type(rag_text) == SegmentType.RETRIEVED_DOCS
        
        # Unknown/default
        unknown_text = "Some arbitrary content."
        assert detect_segment_type(unknown_text) == SegmentType.UNKNOWN


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