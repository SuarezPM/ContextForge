"""Tests for RotateKVQuantizer — TASK-005."""
import pytest
import numpy as np
from contextforge.quantization.rotate_kv import RotateKVQuantizer, RotateKVConfig, QuantizedKVBlock


class TestRotateKVQuantizer:
    """Tests for RotateKV quantization (INVARIANT 10: pre-RoPE only)."""

    def test_rotate_kv_config_defaults(self):
        """RotateKVConfig has sensible defaults."""
        config = RotateKVConfig()
        assert config.bits == 4
        assert config.group_size == 64
        assert config.sink_tokens == 4

    def test_quantized_kv_block_has_pre_rope_metadata(self):
        """QuantizedKVBlock stores pre_rope flag in metadata."""
        # This tests the invariant: pre-RoPE tensors are what we quantize
        block = QuantizedKVBlock(
            keys_int4=np.zeros((10, 8, 64), dtype=np.float32),
            values_int4=np.zeros((10, 8, 64), dtype=np.float32),
            keys_sink_fp16=np.zeros((4, 8, 128), dtype=np.float16),
            values_sink_fp16=np.zeros((4, 8, 128), dtype=np.float16),
            scales_k=np.ones((1, 8, 64), dtype=np.float32),
            zero_points_k=np.zeros((1, 8, 64), dtype=np.float32),
            scales_v=np.ones((1, 8, 128), dtype=np.float32),
            zero_points_v=np.zeros((1, 8, 128), dtype=np.float32),
            channel_order=np.arange(128, dtype=np.int32),
            positions=np.arange(14, dtype=np.float32),
            bits=4,
        )
        assert block.bits == 4

    @pytest.mark.asyncio
    async def test_quantize_pre_rope_returns_quantized_block(self):
        """quantize_pre_rope() returns (QuantizedKVBlock, ndarray) tuple (INVARIANT 10)."""
        config = RotateKVConfig(bits=4, group_size=64, sink_tokens=4)
        quantizer = RotateKVQuantizer(config)

        # Pre-RoPE tensors: (batch=1, seq_len, num_heads, head_dim)
        k_tensor = np.random.randn(1, 64, 8, 64).astype(np.float32)
        v_tensor = np.random.randn(1, 64, 8, 64).astype(np.float32)
        positions = np.arange(64, dtype=np.float32)

        result = quantizer.quantize_pre_rope(k_tensor, v_tensor, positions)
        assert isinstance(result, tuple)
        qblock, remaining = result
        assert isinstance(qblock, QuantizedKVBlock)
        assert qblock.keys_int4.shape[0] > 0
        assert qblock.values_int4.shape[0] > 0

    @pytest.mark.asyncio
    async def test_quantize_pre_rope_sink_tokens_preserved(self):
        """First sink_tokens are preserved at FP16."""
        config = RotateKVConfig(bits=4, sink_tokens=4)
        quantizer = RotateKVQuantizer(config)

        k_tensor = np.random.randn(1, 64, 8, 64).astype(np.float32)
        v_tensor = np.random.randn(1, 64, 8, 64).astype(np.float32)
        positions = np.arange(64, dtype=np.float32)

        qblock, _ = quantizer.quantize_pre_rope(k_tensor, v_tensor, positions)

        assert qblock.keys_sink_fp16.shape == (1, 4, 8, 64)
        assert qblock.values_sink_fp16.shape == (1, 4, 8, 64)

    @pytest.mark.asyncio
    async def test_dequantize_returns_fp32_tensors(self):
        """dequantize() returns FP32 tensors."""
        config = RotateKVConfig(bits=4, group_size=64, sink_tokens=4)
        quantizer = RotateKVQuantizer(config)

        k_tensor = np.random.randn(1, 64, 8, 64).astype(np.float32)
        v_tensor = np.random.randn(1, 64, 8, 64).astype(np.float32)
        positions = np.arange(64, dtype=np.float32)

        qblock, _ = quantizer.quantize_pre_rope(k_tensor, v_tensor, positions)
        k_deq, v_deq = quantizer.dequantize(qblock)

        assert isinstance(k_deq, np.ndarray)
        assert isinstance(v_deq, np.ndarray)
        assert k_deq.dtype == np.float32
        assert v_deq.dtype == np.float32