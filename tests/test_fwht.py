"""Tests for Fast Walsh-Hadamard Transform — V7 Sprint 1 Track 1."""
import numpy as np
import pytest
import torch

from apohara_context_forge.quantization.fwht import fwht, ifwht


HADAMARD_4 = np.array(
    [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ],
    dtype=np.float32,
)


def test_fwht_shape_preserved():
    x = torch.randn(8, dtype=torch.float32)
    y = fwht(x)
    assert y.shape == x.shape


def test_fwht_power_of_two_no_pad():
    torch.manual_seed(0)
    x = torch.randn(8, dtype=torch.float32)
    y = ifwht(fwht(x))
    assert torch.allclose(x, y, atol=1e-5)


def test_fwht_non_power_of_two_padding():
    x = torch.randn(6, dtype=torch.float32)
    y = fwht(x)
    assert y.shape[-1] == 8


def test_fwht_orthogonality():
    # FWHT applied row-wise to I_4 yields the normalized Hadamard matrix H_4 / sqrt(4).
    I4 = torch.eye(4, dtype=torch.float32)
    Y = fwht(I4).numpy()
    expected = HADAMARD_4 / np.sqrt(4.0)
    assert np.allclose(Y, expected, atol=1e-5)


def test_ifwht_round_trip_identity():
    torch.manual_seed(1)
    x = torch.randn(16, dtype=torch.float32)
    y = ifwht(fwht(x))
    assert torch.allclose(x, y, atol=1e-5)


def test_fwht_batched():
    torch.manual_seed(2)
    x = torch.randn(2, 8, dtype=torch.float32)
    y = fwht(x)
    # Each batch row transformed independently → equals fwht applied per-row.
    y0 = fwht(x[0])
    y1 = fwht(x[1])
    assert torch.allclose(y[0], y0, atol=1e-6)
    assert torch.allclose(y[1], y1, atol=1e-6)


def test_fwht_dtype_preservation():
    x32 = torch.randn(8, dtype=torch.float32)
    assert fwht(x32).dtype == torch.float32

    x16 = torch.randn(8, dtype=torch.float16)
    y16 = fwht(x16)
    assert y16.dtype == torch.float16
    # fp16 round-trip within fp16 epsilon (~1e-3 for d=8).
    rt = ifwht(fwht(x16))
    assert torch.allclose(rt.float(), x16.float(), atol=5e-3)


def test_fwht_zero_input():
    x = torch.zeros(8, dtype=torch.float32)
    y = fwht(x)
    assert torch.allclose(y, torch.zeros(8, dtype=torch.float32))
