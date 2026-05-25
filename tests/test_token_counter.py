import pytest
from unittest.mock import patch

from apohara_context_forge.token_counter import TokenCounter, encode_tokens

def test_encode_tokens_error_path():
    """Test that encode_tokens falls back correctly when tokenizer fails to load."""
    TokenCounter.reset()

    text = "Hello world"
    expected_fallback = [hash(w) % 50000 for w in text.split()]

    with patch("apohara_context_forge.token_counter.logger.warning") as mock_warning:
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=Exception("Simulated load failure")):
            result = encode_tokens(text)

            assert result == expected_fallback
            mock_warning.assert_called_once()
            assert "Simulated load failure" in mock_warning.call_args[0][0]

    # Clean up singleton state
    TokenCounter.reset()
