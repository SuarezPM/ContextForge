"""Tests for in-place Fast Walsh-Hadamard Transform."""
import numpy as np
import pytest
import torch

from apohara_context_forge.quantization.fwht_inplace import fwht_inplace


HADAMARD_4 = np.array(
    [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ],
    dtype=np.float32,
)

DEVICES = ["cpu", "cuda"]

def skip_if_no_cuda(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

@pytest.mark.parametrize("device", DEVICES)
def test_fwht_inplace_shape_preserved(device):
    skip_if_no_cuda(device)
    x = torch.randn(8, dtype=torch.float32, device=device)
    y = fwht_inplace(x.clone())
    assert y.shape == x.shape

@pytest.mark.parametrize("device", DEVICES)
def test_fwht_inplace_raises_value_error_non_power_of_two(device):
    skip_if_no_cuda(device)
    x = torch.randn(6, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="FWHT inplace requires power-of-two last dim, got 6"):
        fwht_inplace(x)

@pytest.mark.parametrize("device", DEVICES)
def test_fwht_inplace_orthogonality(device):
    skip_if_no_cuda(device)
    # FWHT applied row-wise to I_4 yields the normalized Hadamard matrix H_4 / sqrt(4).
    I4 = torch.eye(4, dtype=torch.float32, device=device)
    Y = fwht_inplace(I4).cpu().numpy()
    expected = HADAMARD_4 / np.sqrt(4.0)
    assert np.allclose(Y, expected, atol=1e-5)

@pytest.mark.parametrize("device", DEVICES)
def test_fwht_inplace_round_trip_identity(device):
    skip_if_no_cuda(device)
    torch.manual_seed(1)
    x = torch.randn(16, dtype=torch.float32, device=device)
    orig_x = x.clone()
    # fwht_inplace is self-inverse, so applying it twice should return the original
    y = fwht_inplace(x)
    z = fwht_inplace(y)
    assert torch.allclose(orig_x, z, atol=1e-5)

@pytest.mark.parametrize("device", DEVICES)
def test_fwht_inplace_batched(device):
    skip_if_no_cuda(device)
    torch.manual_seed(2)
    x = torch.randn(2, 8, dtype=torch.float32, device=device)
    orig_x = x.clone()
    y = fwht_inplace(x)
    # Each batch row transformed independently
    y0 = fwht_inplace(orig_x[0].clone())
    y1 = fwht_inplace(orig_x[1].clone())
    assert torch.allclose(y[0], y0, atol=1e-6)
    assert torch.allclose(y[1], y1, atol=1e-6)

@pytest.mark.parametrize("device", DEVICES)
def test_fwht_inplace_dtype_preservation(device):
    skip_if_no_cuda(device)
    x32 = torch.randn(8, dtype=torch.float32, device=device)
    assert fwht_inplace(x32.clone()).dtype == torch.float32

    x16 = torch.randn(8, dtype=torch.float16, device=device)
    orig_x16 = x16.clone()
    y16 = fwht_inplace(x16)
    assert y16.dtype == torch.float16
    # fp16 round-trip within fp16 epsilon (~1e-3 for d=8).
    rt = fwht_inplace(y16)
    assert torch.allclose(rt.float(), orig_x16.float(), atol=5e-3)

@pytest.mark.parametrize("device", DEVICES)
def test_fwht_inplace_zero_input(device):
    skip_if_no_cuda(device)
    x = torch.zeros(8, dtype=torch.float32, device=device)
    y = fwht_inplace(x.clone())
    assert torch.allclose(y, torch.zeros(8, dtype=torch.float32, device=device))
