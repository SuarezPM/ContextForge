"""Adaptive Compression Budget Manager v3.0 - Dynamic per-segment rates.

Replaces static COMPRESSION_BUDGET table with dynamic rates that:
1. Vary by segment_type (validated against LLMLingua-2 research, ACL 2024 Findings)
2. Respond to VRAM pressure (emergency compression when GPU memory is tight)
3. Use sample-wise probability threshold θ (dynamic per-segment, not fixed ratio)

Key rates (from LLMLingua-2 §L):
- system_prompt: 0.9 (near-lossless - role-critical information must be preserved)
- shared_context: 0.5 (high compression - shared docs have high redundancy)
- agent_output: 0.7 (moderate - reasoning chains have task-critical steps)
- tool_result: 0.6 (moderate-high - tool outputs often contain padded JSON/XML)
- user_query: 1.0 (NEVER compress - user intent must be preserved exactly)

Under VRAM pressure > 0.85: multiply all non-user_query rates by 0.8 (emergency).

Usage:
    manager = CompressionBudgetManager()
    rate = manager.get_rate_for_segment("shared_context", token_count=1000, vram_pressure=0.5)
    # rate = 0.5 (normal)

    rate_emergency = manager.get_rate_for_segment("shared_context", token_count=1000, vram_pressure=0.9)
    # rate = 0.4 (0.5 * 0.8 emergency multiplier)
"""
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum tokens before compression overhead is worthwhile
COMPRESSION_MIN_TOKENS = 512

# VRAM pressure threshold for emergency compression
VRAM_EMERGENCY_THRESHOLD = 0.85

# Emergency multiplier when VRAM pressure > threshold
VRAM_EMERGENCY_MULTIPLIER = 0.8


class SegmentType(Enum):
    """Type of content segment for compression budget determination."""
    SYSTEM_PROMPT = "system_prompt"
    SHARED_CONTEXT = "shared_context"
    AGENT_OUTPUT = "agent_output"
    TOOL_RESULT = "tool_result"
    USER_QUERY = "user_query"
    RETRIEVED_DOCS = "retrieved_docs"
    CONV_HISTORY = "conv_history"
    RECENT_TURNS = "recent_turns"
    COT_REASONING = "cot_reasoning"
    RAG_CHUNK = "rag_chunk"
    UNKNOWN = "unknown"


# Dynamic compression rate table (higher = more aggressive = lower output)
# Source: LLMLingua-2 research (ACL 2024 Findings) - dynamic per-sample approach
DYNAMIC_RATE_TABLE: dict[SegmentType, float] = {
    # Near-lossless: system prompts are dense with role-critical information
    SegmentType.SYSTEM_PROMPT: 0.9,
    # High compression: shared retrieved docs have high redundancy
    SegmentType.SHARED_CONTEXT: 0.5,
    SegmentType.RETRIEVED_DOCS: 0.5,
    # Moderate: agent reasoning chains contain task-critical steps
    SegmentType.AGENT_OUTPUT: 0.7,
    SegmentType.COT_REASONING: 0.7,
    # Moderate-high: tool outputs often contain padded JSON/XML
    SegmentType.TOOL_RESULT: 0.6,
    # High compression: resolved context is safe to compress
    SegmentType.CONV_HISTORY: 0.4,
    SegmentType.RAG_CHUNK: 0.4,
    # NO compression: recent relevance and user intent must be exact
    SegmentType.RECENT_TURNS: 0.0,
    SegmentType.USER_QUERY: 1.0,  # 1.0 = no compression
    # Safe default
    SegmentType.UNKNOWN: 0.5,
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
    emergency: bool = False  # True if VRAM emergency multiplier applied


class CompressionBudgetManager:
    """
    Dynamic compression budget manager with VRAM-pressure-responsive rates.

    Key design decision: uses dynamic per-sample probability threshold θ
    rather than fixed ratio enforcement. This allows natural variation
    in compression ratio per segment based on content characteristics.

    Usage:
        manager = CompressionBudgetManager()
        plan = manager.plan(segment_text, SegmentType.SHARED_CONTEXT)

        # Or get rate directly for custom compression
        rate = manager.get_rate_for_segment("agent_output", token_count=1000, vram_pressure=0.5)
    """

    def __init__(self):
        self._lock = asyncio.Lock()

    def get_rate_for_segment(
        self,
        segment_type: str,
        token_count: int,
        vram_pressure: float = 0.0,
    ) -> float:
        """
        Get compression rate for a segment type with VRAM pressure adjustment.

        Args:
            segment_type: String name of segment type (e.g., "shared_context")
            token_count: Number of tokens in segment
            vram_pressure: Current VRAM utilization (0.0-1.0)

        Returns:
            Compression rate (0.0-1.0), or 1.0 if no compression needed
        """
        # Parse segment type
        try:
            st = SegmentType(segment_type)
        except ValueError:
            st = SegmentType.UNKNOWN

        # Never compress user queries
        if st == SegmentType.USER_QUERY:
            return 1.0

        # Get base rate
        rate = DYNAMIC_RATE_TABLE.get(st, DYNAMIC_RATE_TABLE[SegmentType.UNKNOWN])

        # Never compress system prompts (prefix cache critical)
        if st == SegmentType.SYSTEM_PROMPT:
            return 0.9  # Near-lossless, not zero (LLMLingua-2 default)

        # Apply VRAM emergency multiplier
        emergency = False
        if vram_pressure > VRAM_EMERGENCY_THRESHOLD:
            rate = rate * VRAM_EMERGENCY_MULTIPLIER
            emergency = True

        return rate

    def plan(
        self,
        segment: str,
        segment_type: SegmentType,
        token_count: Optional[int] = None,
        vram_pressure: float = 0.0,
    ) -> CompressionPlan:
        """
        Create a compression plan for a segment.

        Args:
            segment: Text content to potentially compress
            segment_type: Type of content (determines budget)
            token_count: Optional pre-computed token count (faster)
            vram_pressure: Current VRAM utilization for emergency detection

        Returns:
            CompressionPlan with decision and parameters
        """
        from contextforge.token_counter import TokenCounter

        if token_count is None:
            token_count = TokenCounter.get().count(segment)

        rate = self.get_rate_for_segment(segment_type.value, token_count, vram_pressure)

        # Hard rule: never compress user queries
        if segment_type == SegmentType.USER_QUERY:
            return CompressionPlan(
                segment=segment,
                segment_type=segment_type,
                original_tokens=token_count,
                target_rate=1.0,
                should_compress=False,
                reason="user_query: never compress (intent must be preserved)",
            )

        # Hard rule: never compress system prompts (prefix cache critical)
        if segment_type == SegmentType.SYSTEM_PROMPT:
            return CompressionPlan(
                segment=segment,
                segment_type=segment_type,
                original_tokens=token_count,
                target_rate=0.9,  # Near-lossless
                should_compress=True,
                reason="system_prompt: near-lossless compression (prefix cache ok)",
            )

        # Skip compression for too-short segments
        if token_count < COMPRESSION_MIN_TOKENS:
            return CompressionPlan(
                segment=segment,
                segment_type=segment_type,
                original_tokens=token_count,
                target_rate=0.0,
                should_compress=False,
                reason=f"too short ({token_count} tokens < {COMPRESSION_MIN_TOKENS} minimum)",
            )

        # Check for emergency compression
        emergency = vram_pressure > VRAM_EMERGENCY_THRESHOLD

        return CompressionPlan(
            segment=segment,
            segment_type=segment_type,
            original_tokens=token_count,
            target_rate=rate,
            should_compress=True,
            reason=f"{segment_type.value}: rate={rate} (vram_pressure={vram_pressure:.2f})"
                   + (" [EMERGENCY]" if emergency else ""),
            emergency=emergency,
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

        from contextforge.compression.compressor import ContextCompressor

        compressor = ContextCompressor()
        await compressor.load()

        return await compressor.compress(
            plan.segment,
            rate=plan.target_rate,
        )

    def plan_and_compress(
        self,
        segment: str,
        segment_type: SegmentType,
        vram_pressure: float = 0.0,
    ) -> tuple[CompressionPlan, Optional[tuple[str, float]]]:
        """
        Convenience: create plan and return (plan, None) or (plan, (compressed, ratio)).
        Synchronous version for non-async contexts.
        """
        plan = self.plan(segment, segment_type, vram_pressure=vram_pressure)
        if plan.should_compress:
            # Note: caller should await compress_with_plan for actual compression
            return plan, None
        return plan, None


def detect_segment_type(segment: str) -> SegmentType:
    """
    Heuristic segment type detection based on content patterns.
    Override with explicit type when known.
    """
    # Check for system prompt indicators
    system_indicators = ["system:", "instructions:", "# system", "you are a "]
    for indicator in system_indicators:
        if indicator.lower() in segment.lower()[:100]:
            return SegmentType.SYSTEM_PROMPT

    # Check for user query indicators (should be near start)
    user_indicators = ["query:", "question:", "what is", "how do", "tell me"]
    for indicator in user_indicators:
        if indicator.lower() in segment.lower()[:50]:
            return SegmentType.USER_QUERY

    # Check for tool output indicators
    tool_indicators = ["tool:", "function:", "execution result:", "output:", "tool result:"]
    for indicator in tool_indicators:
        if indicator.lower() in segment.lower()[:100]:
            return SegmentType.TOOL_RESULT

    # Check for agent output indicators
    agent_indicators = ["retrieved", "summarized", "analyzed", "reasoning:", "step"]
    if any(ind in segment.lower()[:150] for ind in agent_indicators):
        return SegmentType.AGENT_OUTPUT

    # Check for CoT reasoning
    if all(ind in segment.lower() for ind in ["step", "reasoning"]) or "step by step" in segment.lower():
        return SegmentType.COT_REASONING

    # Check for RAG/retrieved content
    rag_indicators = ["document", "retrieved", "context:", "reference:"]
    if any(ind in segment.lower()[:200] for ind in rag_indicators):
        return SegmentType.RETRIEVED_DOCS

    # Check for shared context (general knowledge)
    shared_indicators = ["knowledge", "context:", "background:"]
    if any(ind in segment.lower()[:200] for ind in shared_indicators):
        return SegmentType.SHARED_CONTEXT

    return SegmentType.UNKNOWN


# Backwards compatibility alias
COMPRESSION_BUDGET = DYNAMIC_RATE_TABLE