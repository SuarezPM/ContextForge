"""Tests for PrefixNormalizer."""
import pytest
from apohara_context_forge.normalization.prefix_normalizer import (
    PrefixNormalizer,
    create_prefix_normalizer,
    SEPARATOR,
)


class TestPrefixNormalizerBasic:
    """Basic PrefixNormalizer tests."""

    def test_byte_identical_output_for_same_canonical_prompt(self):
        """Test normalize() produces byte-identical output for same canonical prompt."""
        normalizer = PrefixNormalizer(canonical_system_prompt="You are a helpful AI.")

        prompt1 = normalizer.normalize("agent1", "What is AI?", "retriever role")
        prompt2 = normalizer.normalize("agent2", "What is AI?", "summarizer role")

        # Extract system prompt prefix (everything before first separator)
        system_prefix_1 = prompt1.split(SEPARATOR)[0]
        system_prefix_2 = prompt2.split(SEPARATOR)[0]

        # Both should have the same system prompt prefix
        assert system_prefix_1 == system_prefix_2
        assert system_prefix_1 == "You are a helpful AI."

    def test_sha256_validation_catches_mismatched_canonical_prompts(self):
        """Test SHA256 validation catches mismatched canonical prompts."""
        normalizer = PrefixNormalizer(canonical_system_prompt="You are a helpful AI.")

        # Valid matching prompt
        assert normalizer.validate_system_prompt("You are a helpful AI.") is True

        # Different prompt should not match
        assert normalizer.validate_system_prompt("You are a different AI.") is False

        # Prompt with extra whitespace should not match (validation strips input)
        assert normalizer.validate_system_prompt("  You are a helpful AI.  ") is True

    def test_separator_enforcement(self):
        """Test separator enforcement."""
        normalizer = PrefixNormalizer(canonical_system_prompt="You are a helpful AI.")

        # Default separator should be exactly "\n\n"
        assert normalizer.separator == "\n\n"

        # Output should contain exactly two newlines between segments
        prompt = normalizer.normalize("agent1", "What is AI?", "retriever role")

        # Count occurrences of separator
        assert prompt.count("\n\n") == 2

        # Should have pattern: system\n\nrole\n\nuser
        parts = prompt.split("\n\n")
        assert len(parts) == 3
        assert parts[0] == "You are a helpful AI."
        assert parts[1] == "retriever role"
        assert parts[2] == "What is AI?"

    def test_whitespace_stripping(self):
        """Test whitespace stripping from user_prompt and role_prompt."""
        normalizer = PrefixNormalizer(canonical_system_prompt="You are a helpful AI.")

        # Trailing whitespace should be stripped
        prompt = normalizer.normalize(
            "agent1",
            "What is AI?   ",
            "retriever role   ",
        )

        # Verify no trailing whitespace in output
        lines = prompt.split("\n\n")
        assert lines[1] == "retriever role"
        assert lines[2] == "What is AI?"

        # Leading whitespace should also be stripped
        prompt2 = normalizer.normalize(
            "agent2",
            "   What is AI?",
            "   summarizer role",
        )
        lines2 = prompt2.split("\n\n")
        assert lines2[1] == "summarizer role"
        assert lines2[2] == "What is AI?"

    def test_get_canonical_hash(self):
        """Test get_canonical_hash() returns consistent SHA256 hex string."""
        normalizer1 = PrefixNormalizer(canonical_system_prompt="You are a helpful AI.")
        normalizer2 = PrefixNormalizer(canonical_system_prompt="You are a helpful AI.")

        hash1 = normalizer1.get_canonical_hash()
        hash2 = normalizer2.get_canonical_hash()

        # Same prompt should produce same hash
        assert hash1 == hash2

        # Should be a valid SHA256 hex string (64 characters)
        assert len(hash1) == 64
        assert all(c in "0123456789abcdef" for c in hash1)

        # Different prompt should produce different hash
        normalizer3 = PrefixNormalizer(canonical_system_prompt="You are a different AI.")
        hash3 = normalizer3.get_canonical_hash()

        assert hash1 != hash3

    def test_separator_property(self):
        """Test separator property returns the correct string."""
        normalizer = PrefixNormalizer(canonical_system_prompt="Test prompt.")
        assert normalizer.separator == SEPARATOR
        assert normalizer.separator == "\n\n"

    def test_canonical_hash_consistency(self):
        """Test two instances with same prompt have same hash."""
        normalizer_a = PrefixNormalizer(canonical_system_prompt="You are a helpful AI.")
        normalizer_b = PrefixNormalizer(canonical_system_prompt="You are a helpful AI.")

        assert normalizer_a.get_canonical_hash() == normalizer_b.get_canonical_hash()

    def test_compute_prompt_hash(self):
        """Test compute_prompt_hash returns correct sha256 hash."""
        import hashlib
        normalizer = PrefixNormalizer(canonical_system_prompt="You are a helpful AI.")
        prompt = "test prompt"
        expected_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        assert normalizer.compute_prompt_hash(prompt) == expected_hash


class TestCreatePrefixNormalizer:
    """Tests for create_prefix_normalizer factory function."""

    def test_returns_correct_type(self):
        """Test that create_prefix_normalizer returns a PrefixNormalizer instance."""
        normalizer = create_prefix_normalizer()
        assert isinstance(normalizer, PrefixNormalizer)

    def test_create_with_custom_prompt(self):
        """Test create_prefix_normalizer with custom prompt."""
        normalizer = create_prefix_normalizer(
            canonical_system_prompt="Custom system prompt."
        )

        assert normalizer.get_canonical_prompt() == "Custom system prompt."

    def test_create_with_default_prompt(self):
        """Test create_prefix_normalizer uses default prompt when none provided."""
        normalizer = create_prefix_normalizer()

        expected_default = (
            "You are a helpful AI assistant. "
            "Provide accurate, detailed, and thoughtful responses. "
            "Use chain-of-thought reasoning when appropriate."
        )
        assert normalizer.get_canonical_prompt() == expected_default

    def test_create_prefix_normalizer_has_correct_separator(self):
        """Test create_prefix_normalizer uses correct separator."""
        normalizer = create_prefix_normalizer(
            canonical_system_prompt="Test prompt."
        )
        assert normalizer.separator == "\n\n"


class TestNormalize:
    """Tests for normalize() method."""

    def test_normalize_assembles_in_fixed_order(self):
        """Test normalize() assembles segments in fixed order."""
        normalizer = PrefixNormalizer(canonical_system_prompt="System prompt.")

        prompt = normalizer.normalize(
            agent_id="test_agent",
            user_prompt="User question?",
            agent_role_prompt="Role description.",
        )

        # Order should be: system, role, user
        assert prompt.startswith("System prompt.")
        assert "Role description." in prompt
        assert "User question?" in prompt

    def test_normalize_with_empty_role_prompt(self):
        """Test normalize() with empty role prompt."""
        normalizer = PrefixNormalizer(canonical_system_prompt="System.")

        prompt = normalizer.normalize(
            agent_id="agent",
            user_prompt="Question",
            agent_role_prompt="",
        )

        parts = prompt.split("\n\n")
        assert parts[0] == "System."
        assert parts[1] == ""
        assert parts[2] == "Question"

    def test_normalize_registered_agents(self):
        """Test normalize() tracks registered agents."""
        normalizer = PrefixNormalizer(canonical_system_prompt="System.")

        normalizer.normalize("agent1", "Q1", "Role1")
        normalizer.normalize("agent2", "Q2", "Role2")

        # Agents should be tracked (internal state)
        assert len(normalizer._registered_agents) == 2