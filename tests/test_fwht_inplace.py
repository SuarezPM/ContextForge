"""Tests for in-place torch FWHT and IFWHT."""
import pytest
import torch

from apohara_context_forge.quantization.fwht_inplace import fwht_inplace, ifwht_inplace


def test_fwht_inplace_shape_dtype_preserved():
    torch.manual_seed(0)
    x = torch.randn(8, dtype=torch.float32)
    # create a clone to avoid losing original
    x_input = x.clone()
    y = fwht_inplace(x_input)
    assert y is x_input
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_ifwht_inplace_round_trip_identity():
    torch.manual_seed(1)
    x = torch.randn(16, dtype=torch.float32)
    # Deep copy to check against the original
    original_x = x.clone()

    # Forward
    fwht_inplace(x)
    # In-place means x is modified
    assert not torch.allclose(x, original_x, atol=1e-5)

    # Inverse
    ifwht_inplace(x)

    # Should restore original
    assert torch.allclose(x, original_x, atol=1e-5)


def test_ifwht_inplace_return_value():
    torch.manual_seed(2)
    x = torch.randn(16, dtype=torch.float32)
    x_input = x.clone()
    y = ifwht_inplace(x_input)
    assert y is x_input
    assert y.shape == x.shape


def test_fwht_inplace_power_of_two_error():
    x = torch.randn(6, dtype=torch.float32)
    with pytest.raises(ValueError, match="FWHT inplace requires power-of-two last dim, got 6"):
        fwht_inplace(x)

def test_ifwht_inplace_power_of_two_error():
    x = torch.randn(14, dtype=torch.float32)
    with pytest.raises(ValueError, match="FWHT inplace requires power-of-two last dim, got 14"):
        ifwht_inplace(x)

def test_fwht_inplace_fp16():
    """Verify upcasting logic in inplace fwht works fine and restores fp16"""
    torch.manual_seed(3)
    x = torch.randn(8, dtype=torch.float16)
    x_input = x.clone()

    y = fwht_inplace(x_input)
    # The upcast to float32 makes it no longer strictly in-place in memory for fp16 inputs
    assert y.shape == x.shape
    assert y.dtype == torch.float16

def test_ifwht_inplace_fp16():
    """Verify upcasting logic in inplace ifwht works fine and restores fp16"""
    torch.manual_seed(4)
    x = torch.randn(8, dtype=torch.float16)
    x_input = x.clone()

    y = ifwht_inplace(x_input)
    # The upcast to float32 makes it no longer strictly in-place in memory for fp16 inputs
    assert y.shape == x.shape
    assert y.dtype == torch.float16

def test_fwht_inplace_batched():
    torch.manual_seed(5)
    x = torch.randn(2, 8, dtype=torch.float32)
    # Manually compute row by row
    row0 = x[0].clone()
    row1 = x[1].clone()
    fwht_inplace(row0)
    fwht_inplace(row1)

    y = fwht_inplace(x)
    assert torch.allclose(y[0], row0, atol=1e-6)
    assert torch.allclose(y[1], row1, atol=1e-6)
