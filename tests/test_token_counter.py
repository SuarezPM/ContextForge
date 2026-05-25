import pytest
from apohara_context_forge.token_counter import compute_kv_gb

def test_compute_kv_gb_positive():
    """Test happy path for compute_kv_gb with a positive token count."""
    gb = compute_kv_gb(1024)
    assert gb > 0

    # 2 * 64 * 1024 * 8 * 128 * 2 = 268,435,456 bytes
    # 268,435,456 / (1024 ** 3) = 0.25 GB
    assert gb == 0.25

def test_compute_kv_gb_zero():
    """Test compute_kv_gb with zero tokens."""
    gb = compute_kv_gb(0)
    assert gb == 0.0

def test_compute_kv_gb_negative():
    """Test error path for compute_kv_gb with negative token count."""
    with pytest.raises(ValueError, match="token_count must be non-negative"):
        compute_kv_gb(-10)

def test_compute_kv_gb_with_kwargs():
    """Test with kwargs overrides."""
    # Override layers and head_dim
    gb = compute_kv_gb(1024, n_layers=32, head_dim=64)
    # 2 * 32 * 1024 * 8 * 64 * 2 = 67,108,864 bytes
    # 67,108,864 / (1024 ** 3) = 0.0625 GB
    assert gb == 0.0625
