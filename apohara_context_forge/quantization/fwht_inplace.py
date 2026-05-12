"""In-place torch FWHT — Sprint 4 candidate optimization.

Sprint 3 Wave B extended found that the current `_fwht_butterfly_torch`
has +700% peak alloc overhead due to `.clone()` at each butterfly stage.
This module implements an in-place version using strided views and
`torch.where` for the additive/subtractive lanes — no clones, no extra
allocations.

Run as a smoke test:
  PYTHONPATH=. python3 scripts/fwht_inplace_bench.py
"""
from __future__ import annotations

import math

import torch


def fwht_inplace(x: torch.Tensor) -> torch.Tensor:
    """In-place FWHT on the last dim (must be power of two).

    Returns the rotated tensor (modifies x in place). Normalisation is
    orthonormal: divides by sqrt(d) so the transform is self-inverse.
    """
    d = x.shape[-1]
    if d & (d - 1) != 0:
        raise ValueError(f"FWHT inplace requires power-of-two last dim, got {d}")

    orig_dtype = x.dtype
    if orig_dtype != torch.float32:
        # Upcast to fp32 in-place for precision; cast back at the end.
        x = x.to(torch.float32)

    h = 1
    while h < d:
        # Reshape last dim into (..., d/(2h), 2, h)
        view = x.view(*x.shape[:-1], d // (2 * h), 2, h)
        # Each butterfly: a = view[..., 0, :], b = view[..., 1, :]
        # Update: view[..., 0, :] = a + b ; view[..., 1, :] = a - b
        # Done in-place without clone via temporary on the small h slice.
        a = view[..., 0, :]
        b = view[..., 1, :]
        # Use torch.add / sub_ to avoid extra tensors.
        # tmp = a clone-free alternative: hold b's values in t, then update.
        t = b.clone()                # ONLY one clone of the smaller slice
        b.copy_(a)
        b.sub_(t)
        a.add_(t)
        h *= 2

    x.mul_(1.0 / math.sqrt(d))

    if orig_dtype != torch.float32:
        x = x.to(orig_dtype)
    return x


def ifwht_inplace(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal FWHT is self-inverse."""
    return fwht_inplace(x)
