"""Tests for in-place torch FWHT / IFWHT.

Combined from PRs #26 (fwht + ifwht round-trip, dtype, batched, errors) and
#28 (orthogonality vs the normalized Hadamard matrix, zero input).
"""
import numpy as np
import pytest
import torch

from apohara_context_forge.quantization.fwht_inplace import fwht_inplace, ifwht_inplace


HADAMARD_4 = np.array(
    [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ],
    dtype=np.float32,
)


def test_fwht_inplace_shape_dtype_preserved():
    torch.manual_seed(0)
    x = torch.randn(8, dtype=torch.float32)
    x_input = x.clone()
    y = fwht_inplace(x_input)
    assert y is x_input
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_ifwht_inplace_round_trip_identity():
    torch.manual_seed(1)
    x = torch.randn(16, dtype=torch.float32)
    original_x = x.clone()
    fwht_inplace(x)
    assert not torch.allclose(x, original_x, atol=1e-5)
    ifwht_inplace(x)
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
    """Upcast logic in in-place fwht works and restores fp16."""
    torch.manual_seed(3)
    x = torch.randn(8, dtype=torch.float16)
    y = fwht_inplace(x.clone())
    assert y.shape == x.shape
    assert y.dtype == torch.float16


def test_ifwht_inplace_fp16():
    """Upcast logic in in-place ifwht works and restores fp16."""
    torch.manual_seed(4)
    x = torch.randn(8, dtype=torch.float16)
    y = ifwht_inplace(x.clone())
    assert y.shape == x.shape
    assert y.dtype == torch.float16


def test_fwht_inplace_batched():
    torch.manual_seed(5)
    x = torch.randn(2, 8, dtype=torch.float32)
    row0 = x[0].clone()
    row1 = x[1].clone()
    fwht_inplace(row0)
    fwht_inplace(row1)
    y = fwht_inplace(x)
    assert torch.allclose(y[0], row0, atol=1e-6)
    assert torch.allclose(y[1], row1, atol=1e-6)


def test_fwht_inplace_orthogonality():
    """FWHT applied row-wise to I_4 yields the normalized Hadamard matrix H_4 / sqrt(4)."""
    I4 = torch.eye(4, dtype=torch.float32)
    Y = fwht_inplace(I4).cpu().numpy()
    expected = HADAMARD_4 / np.sqrt(4.0)
    assert np.allclose(Y, expected, atol=1e-5)


def test_fwht_inplace_zero_input():
    x = torch.zeros(8, dtype=torch.float32)
    y = fwht_inplace(x.clone())
    assert torch.allclose(y, torch.zeros(8, dtype=torch.float32))
