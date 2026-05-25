"""Tests for token_counter helpers.

Combined from PRs #31 (compute_kv_gb happy/zero/negative/kwargs) and
#32 (encode_tokens fallback when the tokenizer fails to load).
"""
import pytest
from unittest.mock import patch

from apohara_context_forge.token_counter import TokenCounter, compute_kv_gb, encode_tokens


# --- compute_kv_gb (PR #31) ---

def test_compute_kv_gb_positive():
    """Happy path: 1024 tokens at defaults = 0.25 GB."""
    gb = compute_kv_gb(1024)
    assert gb > 0
    # 2 * 64 layers * 1024 * 8 kv-heads * 128 head_dim * 2 bytes = 268,435,456 B = 0.25 GB
    assert gb == 0.25


def test_compute_kv_gb_zero():
    assert compute_kv_gb(0) == 0.0


def test_compute_kv_gb_negative():
    with pytest.raises(ValueError, match="token_count must be non-negative"):
        compute_kv_gb(-10)


def test_compute_kv_gb_with_kwargs():
    # Override layers + head_dim: 2 * 32 * 1024 * 8 * 64 * 2 = 67,108,864 B = 0.0625 GB
    gb = compute_kv_gb(1024, n_layers=32, head_dim=64)
    assert gb == 0.0625


# --- encode_tokens fallback (PR #32) ---

def test_encode_tokens_error_path():
    """encode_tokens falls back to a hash-based encoding when the tokenizer fails to load."""
    TokenCounter.reset()
    text = "Hello world"
    expected_fallback = [hash(w) % 50000 for w in text.split()]
    with patch("apohara_context_forge.token_counter.logger.warning") as mock_warning:
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=Exception("Simulated load failure"),
        ):
            result = encode_tokens(text)
            assert result == expected_fallback
            mock_warning.assert_called_once()
            assert "Simulated load failure" in mock_warning.call_args[0][0]
    TokenCounter.reset()
