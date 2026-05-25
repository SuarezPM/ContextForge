import pytest
from apohara_context_forge.token_counter import count_tokens, encode_tokens, compute_kv_gb, TokenCounter

def test_count_tokens_null_bytes():
    """Test counting tokens with null bytes."""
    assert count_tokens("\0") > 0

def test_count_tokens_long_sequence():
    """Test counting tokens with an extremely long sequence."""
    # "a " * 10000 has 10000 "a"s, each should be at least a token, or the word pieces.
    # The heuristic in fallback calculates ~7500. So check that it's at least 5000.
    assert count_tokens("a " * 10000) > 5000

def test_encode_tokens():
    """Test encoding tokens."""
    assert len(encode_tokens("hello world")) > 0

def test_compute_kv_gb():
    """Test computing KV GB."""
    assert compute_kv_gb(100) > 0.0

def test_singleton_reset():
    """Test resetting the singleton."""
    TokenCounter.reset()
    assert TokenCounter._instance is None
