"""V8 codec — per-nibble independent scales.

Design rationale: see ``docs/v8-codec-design.md``.

V7 (current ``RotateKVQuantizer._quantize_block``) collapses the
(d_lo, d_hi) pair axis when computing the per-block min/max, so both
nibbles of every packed byte share a single ``(scale, zero_point)``.
This costs reconstruction fidelity when the two channels in a pair have
asymmetric dynamic range (typical for transformer KV projections near
attention-sink positions).

V8 keeps the pair axis. Each nibble of each packed byte gets its own
``(scale, zero_point)``. Storage cost: scales/zps shape grows by a
trailing factor of 2 (from ``(n_blocks, num_heads, packed_head_dim)``
to ``(n_blocks, num_heads, packed_head_dim, 2)``).

Honesty discipline: V8 numbers DO NOT enter paper Table 3 until they
are measured on real MI300X (Sprint 5). See
``docs/v8-codec-design.md`` § Acceptance criteria.

INVARIANT preservation: V8 inherits INV-10 (pre-RoPE quantization)
from V7. Channel reordering, sink-token FP16 protection, FWHT toggle,
and the QuantizedKVBlock packed-INT4 layout are all unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from apohara_context_forge.quantization.rotate_kv import (
    QuantizedKVBlock,
    RotateKVConfig,
    RotateKVQuantizer,
)


@dataclass
class CodecV8Config(RotateKVConfig):
    """V8 config. Identical to V7 except the codec marker."""

    codec_version: str = "v8"


class CodecV8Quantizer(RotateKVQuantizer):
    """V8 quantizer: per-nibble independent scales.

    Drop-in replacement for :class:`RotateKVQuantizer`. Subclasses ship
    the same ``quantize_pre_rope`` / ``dequantize`` public surface; only
    ``_quantize_block`` and ``_dequantize_block`` differ.

    Output ``QuantizedKVBlock`` reuses the V7 dataclass shape; the
    scales / zero_points arrays carry an additional trailing axis of
    size 2 (lower nibble first, upper nibble second).
    """

    def __init__(self, config: CodecV8Config | None = None) -> None:  # type: ignore[override]
        super().__init__(config or CodecV8Config())

    # -- override _quantize_block ----------------------------------------
    def _quantize_block(
        self, states: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize a block of states to INT4 with per-nibble scales.

        Mirrors the V7 implementation in :func:`RotateKVQuantizer._quantize_block`
        verbatim except for two changes flagged with ``# V8``:

        1. min/max computed per-pair (no ``axis=4`` collapse)
        2. scales / zero_points carry a trailing ``pair`` axis of size 2
        """
        cfg = self._config
        batch, seq, num_heads, head_dim = states.shape

        n_blocks = seq // cfg.group_size
        if seq % cfg.group_size != 0:
            n_blocks += 1

        packed_head_dim = head_dim // 2
        max_range = 15.0 if cfg.bits == 4 else 255.0

        keys_int4 = np.zeros(
            (n_blocks, cfg.group_size, num_heads, packed_head_dim), dtype=np.uint8
        )
        # V8: trailing pair axis on scales / zero_points.
        scales = np.zeros(
            (n_blocks, num_heads, packed_head_dim, 2), dtype=np.float32
        )
        zero_points = np.zeros(
            (n_blocks, num_heads, packed_head_dim, 2), dtype=np.float32
        )

        padded_seq = n_blocks * cfg.group_size
        valid_mask = np.zeros(padded_seq, dtype=bool)
        valid_mask[:seq] = True

        for b in range(batch):
            buf = np.zeros((padded_seq, num_heads, head_dim), dtype=states.dtype)
            buf[:seq] = states[b]

            blocks = buf.reshape(
                n_blocks, cfg.group_size, num_heads, packed_head_dim, 2
            )
            valid_blocks = valid_mask.reshape(n_blocks, cfg.group_size)

            # V8 difference: do NOT collapse the pair axis (axis=4).
            # min/max are computed over group_size only (axis=1).
            valid_4d = valid_blocks[:, :, None, None, None]
            masked_for_min = np.where(valid_4d, blocks, np.inf)
            masked_for_max = np.where(valid_4d, blocks, -np.inf)
            min_val = np.min(masked_for_min, axis=1).astype(np.float64)
            max_val = np.max(masked_for_max, axis=1).astype(np.float64)
            # min_val / max_val shape: (n_blocks, num_heads, packed_head_dim, 2)

            empty = ~valid_blocks.any(axis=1)  # (n_blocks,)

            range_ = max_val - min_val
            scale = np.where(range_ > 0, range_ / max_range, 1.0)
            zp = np.where(scale != 0, -np.rint(min_val / scale), 0.0)

            scale_f32 = scale.astype(np.float32)
            zp_f32 = zp.astype(np.float32)
            # Broadcast empty mask over the pair axis.
            scale_f32[empty] = 0.0
            zp_f32[empty] = 0.0

            # Quantize: shape (n_blocks, group_size, num_heads, packed_head_dim, 2).
            # Broadcast scale/zp over group_size only — pair axis is now
            # carried by scale_f32 / zp_f32 directly.
            scale_b = scale_f32[:, None, :, :, :]
            zp_b = zp_f32[:, None, :, :, :]
            safe_scale = np.where(scale_b == 0, 1.0, scale_b)
            q = np.clip(
                np.round(blocks / safe_scale + zp_b),
                0,
                max_range,
            ).astype(np.uint8)

            q_lo = q[..., 0]
            q_hi = q[..., 1]
            packed = (q_lo & 0xF) | ((q_hi & 0xF) << 4)

            packed = packed * valid_blocks[:, :, None, None].astype(np.uint8)

            keys_int4[:] = packed
            scales[:] = scale_f32
            zero_points[:] = zp_f32

        return keys_int4, scales, zero_points

    # -- override _dequantize_block --------------------------------------
    def _dequantize_block(
        self,
        packed_int4: np.ndarray,
        scales: np.ndarray,
        zero_points: np.ndarray,
        group_size: int,
    ) -> np.ndarray:
        """Dequantize INT4 with per-nibble scales back to FP32.

        Differs from V7 only in:
        - ``scales`` / ``zero_points`` carry a trailing pair axis (shape
          ``(n_blocks, num_heads, packed_head_dim, 2)``)
        - ``scale_lo`` / ``scale_hi`` and ``zp_lo`` / ``zp_hi`` are read
          per-nibble instead of a single shared scalar
        """
        n_blocks, _, num_heads, packed_head_dim = packed_int4.shape
        seq_len = n_blocks * group_size

        output = np.zeros(
            (1, seq_len, num_heads, packed_head_dim * 2), dtype=np.float32
        )

        for blk in range(n_blocks):
            start = blk * group_size
            for h in range(num_heads):
                for d in range(packed_head_dim):
                    scale_lo = scales[blk, h, d, 0]
                    scale_hi = scales[blk, h, d, 1]
                    zp_lo = zero_points[blk, h, d, 0]
                    zp_hi = zero_points[blk, h, d, 1]

                    for i in range(group_size):
                        if start + i >= seq_len:
                            break
                        byte = packed_int4[blk, i, h, d]
                        val_lo = byte & 0x0F
                        val_hi = (byte >> 4) & 0x0F

                        output[0, start + i, h, d * 2] = (val_lo - zp_lo) * scale_lo
                        output[0, start + i, h, d * 2 + 1] = (val_hi - zp_hi) * scale_hi

        return output
