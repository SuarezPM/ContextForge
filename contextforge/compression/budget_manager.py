"""Adaptive Compression Budget Manager - IMPROVEMENT-003.

Replaces flat rate=0.5 with segment-type-aware compression budgets.
Critical rule: NEVER compress the shared system prefix (breaks vLLM prefix caching).

Compression budgets by segment type:
- SYSTEM_PROMPT: 0.0 (NO COMPRESSION - must be token-identical)
- RETRIEVED_DOCS: 0.25 (high info density, factual content)
- CONV_HISTORY: 0.40 (resolved context, safe to compress)
- RECENT_TURNS: 0.0 (NO COMPRESSION - immediate relevance)
- TOOL_OUTPUT: 0.50 (artifact refs break at high compression)
- COT_REASONING: 0.07 (LLMLingua-2 preserves reasoning well)
- RAG_CHUNK: 0.40 (already filtered by reranker)

Usage:
    manager = CompressionBudgetManager()
    plan = manager.plan(segment_text, SegmentType.RETRIEVED_DOCS)
    if plan.should_compress:
        compressed, ratio = await manager.compress_with_plan(plan)
"""
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from contextforge.token_counter import TokenCounter

logger = logging.getLogger(__name__)

# Minimum tokens before compression overhead is worthwhile
COMPRESSION_MIN_TOKENS = 512


class SegmentType(Enum):
    """Type of content segment for compression budget determination."""
    SYSTEM_PROMPT = "system_prompt"
    RETRIEVED_DOCS = "retrieved_docs"
    CONV_HISTORY = "conv_history"
    RECENT_TURNS = "recent_turns"
    TOOL_OUTPUT = "tool_output"
    COT_REASONING = "cot_reasoning"
    RAG_CHUNK = "rag_chunk"
    UNKNOWN = "unknown"


# Budget rates by segment type (lower = more aggressive compression)
COMPRESSION_BUDGET: dict[SegmentType, float] = {
    SegmentType.SYSTEM_PROMPT:  0.0,   # NO compression - prefix cache critical
    SegmentType.RETRIEVED_DOCS: 0.25,  # 4x compression - high info density
    SegmentType.CONV_HISTORY:   0.40,  # ~2.5x compression - resolved context
    SegmentType.RECENT_TURNS:    0.0,   # NO compression - recent relevance
    SegmentType.TOOL_OUTPUT:    0.50,  # 2x compression - artifact refs
    SegmentType.COT_REASONING:  0.07,  # ~14x compression - LLMLingua-2 handles well
    SegmentType.RAG_CHUNK:      0.40,  # ~2.5x compression - reranked content
    SegmentType.UNKNOWN:         0.50,  # Safe default
}


@dataclass
class CompressionPlan:
    """Compression plan for a single segment."""
    segment: str
    segment_type: SegmentType
    original_tokens: int
    target_rate: float  # 0.0 = no compression, 1.0 = most aggressive
    should_compress: bool
    reason: str


class CompressionBudgetManager:
    """
    Adaptive compression budget manager.
    Determines per-segment compression rates based on content type.
    Enforces no-compression for prefix-critical segments.
    
    Usage:
        manager = CompressionBudgetManager()
        plan = manager.plan(text, SegmentType.RETRIEVED_DOCS)
        if plan.should_compress:
            result = await manager.compress_with_plan(plan)
    """
    
    def __init__(self):
        self._token_counter = TokenCounter.get()
        self._compressor = None
        self._lock = asyncio.Lock()
    
    async def _ensure_compressor(self):
        """Lazy load the LLMLingua-2 compressor."""
        if self._compressor is None:
            async with self._lock:
                if self._compressor is None:
                    from contextforge.compression.compressor import ContextCompressor
                    self._compressor = ContextCompressor()
                    await self._compressor.load()
    
    def plan(self, segment: str, segment_type: SegmentType) -> CompressionPlan:
        """
        Create a compression plan for a segment.
        
        Args:
            segment: Text content to potentially compress
            segment_type: Type of content (determines budget)
        
        Returns:
            CompressionPlan with decision and parameters
        """
        token_count = self._token_counter.count(segment)
        rate = COMPRESSION_BUDGET.get(segment_type, COMPRESSION_BUDGET[SegmentType.UNKNOWN])
        
        # Hard rule: SYSTEM_PROMPT never compressed
        if rate == 0.0:
            return CompressionPlan(
                segment=segment,
                segment_type=segment_type,
                original_tokens=token_count,
                target_rate=0.0,
                should_compress=False,
                reason=f"{segment_type.value}: protected from compression (prefix cache critical)"
            )
        
        # Skip compression for too-short segments
        if token_count < COMPRESSION_MIN_TOKENS:
            return CompressionPlan(
                segment=segment,
                segment_type=segment_type,
                original_tokens=token_count,
                target_rate=0.0,
                should_compress=False,
                reason=f"too short ({token_count} tokens < {COMPRESSION_MIN_TOKENS} minimum)"
            )
        
        return CompressionPlan(
            segment=segment,
            segment_type=segment_type,
            original_tokens=token_count,
            target_rate=rate,
            should_compress=True,
            reason=f"budget rate {rate} for {segment_type.value}"
        )
    
    async def compress_with_plan(self, plan: CompressionPlan) -> tuple[str, float]:
        """
        Execute compression according to plan.
        
        Args:
            plan: CompressionPlan from .plan()
        
        Returns:
            Tuple of (compressed_text, actual_compression_ratio)
        """
        if not plan.should_compress:
            return plan.segment, 1.0
        
        await self._ensure_compressor()
        return await self._compressor.compress(
            plan.segment,
            rate=plan.target_rate
        )
    
    def plan_and_compress(
        self,
        segment: str,
        segment_type: SegmentType,
    ) -> tuple[CompressionPlan, Optional[tuple[str, float]]]:
        """
        Convenience: create plan and return (plan, None) or (plan, (compressed, ratio)).
        Synchronous version for non-async contexts.
        """
        plan = self.plan(segment, segment_type)
        if plan.should_compress:
            # Note: caller should await compress_with_plan for actual compression
            return plan, None
        return plan, None


def detect_segment_type(segment: str) -> SegmentType:
    """
    Heuristic segment type detection based on content patterns.
    Override with explicit type when known.
    
    Args:
        segment: Text content
    
    Returns:
        Detected SegmentType
    """
    # Check for system prompt indicators
    system_indicators = ["system:", "instructions:", "# system", "you are a "]
    for indicator in system_indicators:
        if indicator.lower() in segment.lower()[:100]:
            return SegmentType.SYSTEM_PROMPT
    
    # Check for tool output indicators
    tool_indicators = ["tool:", "function:", "execution result:", "output:"]
    for indicator in tool_indicators:
        if indicator.lower() in segment.lower()[:100]:
            return SegmentType.TOOL_OUTPUT
    
    # Check for CoT reasoning
    cot_indicators = ["step", "reasoning", "because", "therefore", "thus", "analysis"]
    if all(ind in segment.lower() for ind in ["step", "reasoning"]) or "step by step" in segment.lower():
        return SegmentType.COT_REASONING
    
    # Check for RAG/retrieved content
    rag_indicators = ["document", "retrieved", "context:", "reference:"]
    if any(ind in segment.lower()[:200] for ind in rag_indicators):
        return SegmentType.RETRIEVED_DOCS
    
    return SegmentType.UNKNOWN
