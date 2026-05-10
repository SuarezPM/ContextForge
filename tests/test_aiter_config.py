"""Tests for AITERConfig.

Covers:
- All documented env vars are applied to os.environ
- get_expected_speedups returns the documented entries
- is_rocm_available is honest on this host
- status() round-trips correctly
"""
from __future__ import annotations

import os

import pytest

from apohara_context_forge.serving.aiter_config import AITERConfig


class TestAITERConfigDefaults:
    def test_default_env_vars(self):
        cfg = AITERConfig()
        assert cfg.AITER_ENV_VARS["VLLM_ROCM_USE_AITER"] == "1"
        assert cfg.AITER_ENV_VARS["VLLM_ROCM_USE_AITER_MOE"] == "1"
        assert cfg.AITER_ENV_VARS["VLLM_ROCM_USE_AITER_MHA"] == "1"
        assert cfg.AITER_ENV_VARS["VLLM_ROCM_USE_AITER_RMSNORM"] == "1"
        assert cfg.AITER_ENV_VARS["VLLM_ROCM_USE_AITER_LINEAR"] == "1"
        # AITER_ENABLE_VSKIP must be "0" — a "1" here is documented to crash.
        assert cfg.AITER_ENV_VARS["AITER_ENABLE_VSKIP"] == "0"
        assert cfg.AITER_ENV_VARS["NCCL_MIN_NCHANNELS"] == "112"


class TestAITERApply:
    @pytest.fixture(autouse=True)
    def cleanup_env(self):
        """Snapshot env before each test, restore after."""
        cfg = AITERConfig()
        prev = {k: os.environ.get(k) for k in cfg.AITER_ENV_VARS}
        yield
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_apply_writes_all_vars(self):
        cfg = AITERConfig()
        applied = cfg.apply()
        assert applied == cfg.AITER_ENV_VARS
        for k, v in cfg.AITER_ENV_VARS.items():
            assert os.environ.get(k) == v

    def test_apply_returns_independent_copy(self):
        cfg = AITERConfig()
        applied = cfg.apply()
        applied["VLLM_ROCM_USE_AITER"] = "tampered"
        # Mutating the return value should NOT change the dataclass state.
        assert cfg.AITER_ENV_VARS["VLLM_ROCM_USE_AITER"] == "1"


class TestAITERSpeedups:
    def test_documented_speedups(self):
        cfg = AITERConfig()
        sp = cfg.get_expected_speedups()
        assert "fused_moe" in sp
        assert "block_scale_gemm" in sp
        assert sp["fused_moe"] == "3x"
        assert "memory" in sp["fp8_quantization"].lower()


class TestAITERAvailability:
    def test_is_rocm_available_returns_bool(self):
        cfg = AITERConfig()
        assert isinstance(cfg.is_rocm_available(), bool)

    def test_status_dict_shape(self):
        cfg = AITERConfig()
        st = cfg.status()
        assert "rocm_available" in st
        assert "applied" in st
        assert "env" in st
        assert "expected_speedups" in st
        # env mirrors the documented keys.
        assert set(st["env"].keys()) == set(cfg.AITER_ENV_VARS.keys())


class TestAITERRepr:
    def test_repr_does_not_explode(self):
        cfg = AITERConfig()
        r = repr(cfg)
        assert "AITERConfig" in r
        assert "rocm_available" in r
