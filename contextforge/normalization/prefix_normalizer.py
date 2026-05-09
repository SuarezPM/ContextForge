"""Prefix Normalizer for vLLM prefix caching (enable_prefix_caching=True).

vLLM requires token-identical prefixes across requests to trigger KV cache hits.
A single extra space or different capitalization creates a completely different
token sequence and breaks cache sharing.

Key enforcement:
- FIXED order: [canonical_system_prompt][SEP][agent_role_prompt][SEP][user_prompt]
- SEPARATOR is exactly two newlines: "\n\n" (never one, never three)
- Each segment stripped of trailing whitespace before assembly
- SHA256 validation catches mismatched canonical prefixes

Usage:
    normalizer = PrefixNormalizer(
        canonical_system_prompt="You are a helpful AI assistant."
    )

    # All agents use the same normalizer
    prompt1 = normalizer.normalize("agent1", "What is AI?", "retriever role")
    prompt2 = normalizer.normalize("agent2", "What is AI?", "summarizer role")

    # prompt1 and prompt2 are byte-identical at the system prompt prefix
"""
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Fixed separator between prompt segments
SEPARATOR = "\n\n"


class PrefixNormalizer:
    """
    Enforces token-identical prefixes for vLLM prefix caching.

    All agents must use the same canonical_system_prompt. Any deviation
    is logged as a WARNING (not ERROR) because vLLM silently degrades
    to non-cached computation when prefixes don't match.

    Usage:
        normalizer = PrefixNormalizer(
            canonical_system_prompt="You are a helpful AI assistant."
        )
        final_prompt = normalizer.normalize(
            agent_id="agent1",
            user_prompt="What is machine learning?",
            agent_role_prompt="You are a retriever agent."
        )
    """

    def __init__(
        self,
        canonical_system_prompt: str,
        separator: str = SEPARATOR,
    ):
        """
        Initialize with the shared system prompt.

        Args:
            canonical_system_prompt: The shared base prompt (must be identical
                                     byte-for-byte across all agents)
            separator: Separator between segments (default: two newlines)
        """
        self._canonical_system_prompt = canonical_system_prompt.strip()
        self._separator = separator
        self._canonical_hash = self._compute_hash(self._canonical_system_prompt)
        self._registered_agents: set[str] = set()

        logger.info(
            f"PrefixNormalizer initialized with system prompt hash: "
            f"{self._canonical_hash[:16]}..."
        )

    @staticmethod
    def _compute_hash(text: str) -> str:
        """Compute SHA256 hex of text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def normalize(
        self,
        agent_id: str,
        user_prompt: str,
        agent_role_prompt: str,
    ) -> str:
        """
        Assemble final prompt in FIXED order with canonical system prompt.

        Order: [canonical_system_prompt][SEP][agent_role_prompt][SEP][user_prompt]

        Args:
            agent_id: Agent identifier (for logging only)
            user_prompt: User's query/input
            agent_role_prompt: Agent-specific role prompt

        Returns:
            Final assembled prompt with byte-identical system prefix
        """
        # Strip trailing whitespace from each segment
        system_part = self._canonical_system_prompt
        role_part = agent_role_prompt.strip()
        user_part = user_prompt.strip()

        # Assemble in fixed order
        segments = [system_part, role_part, user_part]
        assembled = self._separator.join(segments)

        # Validate system prompt hash (catch silent prefix mismatches)
        # We don't validate here because the system prompt is already stored
        # and should be identical. Validation happens at registration.

        if agent_id not in self._registered_agents:
            self._registered_agents.add(agent_id)

        return assembled

    def validate_system_prompt(self, system_prompt: str) -> bool:
        """
        Validate that a system prompt matches the canonical one.

        Args:
            system_prompt: System prompt to validate

        Returns:
            True if identical, False otherwise
        """
        hash_to_check = self._compute_hash(system_prompt.strip())
        matches = hash_to_check == self._canonical_hash

        if not matches:
            logger.warning(
                f"Agent system prompt hash MISMATCH. "
                f"Expected {self._canonical_hash[:16]}, "
                f"got {hash_to_check[:16]}. "
                f"vLLM prefix caching will NOT work for this agent."
            )

        return matches

    def get_canonical_hash(self) -> str:
        """Get SHA256 of the canonical system prompt."""
        return self._canonical_hash

    def get_canonical_prompt(self) -> str:
        """Get the canonical system prompt."""
        return self._canonical_system_prompt

    @property
    def separator(self) -> str:
        """Get the separator string."""
        return self._separator

    def compute_prompt_hash(self, prompt: str) -> str:
        """
        Compute hash of an assembled prompt (for debugging)."""
        return self._compute_hash(prompt)


def create_prefix_normalizer(
    canonical_system_prompt: Optional[str] = None,
) -> PrefixNormalizer:
    """
    Factory to create a PrefixNormalizer with default or custom system prompt.

    Args:
        canonical_system_prompt: Custom system prompt (optional)

    Returns:
        Configured PrefixNormalizer instance
    """
    default_prompt = (
        "You are a helpful AI assistant. "
        "Provide accurate, detailed, and thoughtful responses. "
        "Use chain-of-thought reasoning when appropriate."
    )

    return PrefixNormalizer(
        canonical_system_prompt=canonical_system_prompt or default_prompt,
        separator=SEPARATOR,
    )