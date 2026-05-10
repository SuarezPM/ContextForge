"""AITERConfig — AMD AI Tensor Engine for ROCm configuration.

AITER provides fused GEMM/MoE/MHA kernels tuned for MI300X. On Qwen3.6-35B-A22B
(MoE) the documented gains are ~3x on the fused MoE kernel, ~2x on block-scaled
GEMM, and 2-4x memory reduction with FP8 quantization.

This module is a thin wrapper that sets the recommended environment variables
before vLLM starts up. The wrapper degrades gracefully on non-ROCm machines:
apply() still sets the env vars, but is_rocm_available() returns False so the
caller can decide whether to proceed.

References
----------
- AMD ROCm AITER docs (see ROCm 7.x release notes)
- vLLM 0.9.x AITER integration (vllm/model_executor/layers/quantization)
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass


@dataclass
class AITERConfig:
    """Apply AITER-recommended environment variables for MI300X inference.

    AITER provides:
    - 2x faster block-scaled GEMM (FP8)
    - 3x faster fused MoE (Qwen3.6-35B-A22B is MoE)
    - Fused MHA/MLA attention kernels
    """

    AITER_ENV_VARS: dict[str, str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.AITER_ENV_VARS is None:
            self.AITER_ENV_VARS = {
                "VLLM_ROCM_USE_AITER": "1",
                "VLLM_ROCM_USE_AITER_MOE": "1",      # Critical for Qwen3 MoE
                "VLLM_ROCM_USE_AITER_MHA": "1",      # Fused multi-head attention
                "VLLM_ROCM_USE_AITER_RMSNORM": "1",  # Accelerated normalization
                "VLLM_ROCM_USE_AITER_LINEAR": "1",   # Quantization + GEMM
                "AITER_ENABLE_VSKIP": "0",           # CRITICAL: prevents crashes
                "NCCL_MIN_NCHANNELS": "112",         # Multi-GPU RCCL optimization
            }

    # ------------------------------------------------------------------ #
    # Apply / inspect                                                      #
    # ------------------------------------------------------------------ #

    def apply(self) -> dict[str, str]:
        """Set all AITER env vars. Returns a copy of what was applied."""
        applied: dict[str, str] = {}
        for k, v in self.AITER_ENV_VARS.items():
            os.environ[k] = v
            applied[k] = v
        return applied

    def get_expected_speedups(self) -> dict[str, str]:
        """Documented speedups from AMD benchmarks (illustrative)."""
        return {
            "deepseek_v3_r1": "2.1x",
            "block_scale_gemm": "2x",
            "fused_moe": "3x",
            "fp8_quantization": "2x-4x memory reduction",
        }

    def is_rocm_available(self) -> bool:
        """Detect ROCm/HIP at runtime without importing torch.

        We check three independent signals so the answer is robust on
        DevCloud-style images:
        1. `rocminfo` on PATH (most reliable on bare metal)
        2. `/opt/rocm` directory exists
        3. HIP_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES env var set
        """
        if shutil.which("rocminfo"):
            return True
        if os.path.isdir("/opt/rocm"):
            return True
        if os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
            "ROCR_VISIBLE_DEVICES"
        ):
            return True
        return False

    def status(self) -> dict[str, object]:
        """Snapshot of current AITER state for the dashboard."""
        currently_set = {
            k: os.environ.get(k, "<unset>") for k in self.AITER_ENV_VARS
        }
        # Truthy if every documented var is set to its expected value.
        applied = all(
            os.environ.get(k) == v for k, v in self.AITER_ENV_VARS.items()
        )
        return {
            "rocm_available": self.is_rocm_available(),
            "applied": applied,
            "env": currently_set,
            "expected_speedups": self.get_expected_speedups(),
        }

    def __repr__(self) -> str:
        st = self.status()
        return (
            f"AITERConfig(rocm_available={st['rocm_available']}, "
            f"applied={st['applied']}, vars={len(self.AITER_ENV_VARS)})"
        )
