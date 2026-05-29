"""Tests for scripts/vram_ab_harness.py — import + dry-mode plumbing.

Confirms the harness imports with NEITHER vllm NOR lmcache installed (they are
not in the venv) and that ``--mode dry`` runs end-to-end with no GPU and no
server, returning HONEST placeholder numbers (``measured=False``). The real
A/B measurement (``--mode live``) is gated outside this workflow and is NOT
exercised here.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_HARNESS_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vram_ab_harness.py"


def _load_harness():
    """Import the harness as a proper module (it lives under scripts/)."""
    spec = importlib.util.spec_from_file_location("vram_ab_harness", _HARNESS_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection sees a real module object.
    sys.modules["vram_ab_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_harness_imports_without_vllm():
    """vllm/lmcache are absent in the venv; the harness must still import."""
    with pytest.raises(ModuleNotFoundError):
        import vllm  # noqa: F401
    mod = _load_harness()
    assert hasattr(mod, "run_dry")
    assert hasattr(mod, "ABConfig")
    assert hasattr(mod, "ABResult")


def test_dry_mode_runs_and_is_honest():
    """Dry mode runs with no GPU/server and flags numbers as NOT measured."""
    mod = _load_harness()
    cfg = mod.ABConfig(concurrency=4)
    res = mod.run_dry(cfg)
    assert res.measured is False
    assert res.vram_source == "dry"
    assert res.hardware_label == "dry-run (no GPU)"
    # Placeholder numbers, not a fabricated measurement.
    assert res.vram_off_gb == 0.0
    assert res.vram_on_gb == 0.0
    assert res.delta_gb == 0.0
    assert res.max_concurrency == 4


def test_dry_mode_exercises_normalizer_and_planner():
    """Dry mode builds ON prompts through the real PrefixNormalizer + planner."""
    mod = _load_harness()
    cfg = mod.ABConfig(concurrency=3)
    prompts_on = mod.build_prompts_on(cfg)
    prompts_off = mod.build_prompts_off(cfg)
    assert len(prompts_on) == 3
    assert len(prompts_off) == 3
    # ON prompts carry a planner cache_salt; OFF prompts carry none.
    assert all(p["cache_salt"] is not None for p in prompts_on)
    assert all(p["cache_salt"] is None for p in prompts_off)
    # Shared system prefix is byte-identical across ON prompts (the whole point).
    prefixes = {p["prompt"].split("\n\n")[0] for p in prompts_on}
    assert len(prefixes) == 1


def test_main_dry_returns_zero(capsys):
    """CLI entrypoint runs dry mode and emits JSON to stdout."""
    mod = _load_harness()
    rc = mod.main(["--mode", "dry", "--concurrency", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"measured": false' in out
    assert '"hardware_label": "dry-run (no GPU)"' in out


def test_hardware_label_honesty():
    """hardware_label is derived honestly from the vram_source backend."""
    mod = _load_harness()
    assert mod._hardware_label("cuda_nvml") == "NVIDIA/CUDA"
    assert mod._hardware_label("cuda_torch") == "NVIDIA/CUDA"
    assert mod._hardware_label("pyrsmi") == "AMD/ROCm"
    # The 192 GB default must be flagged as UNVERIFIED, never silently trusted.
    assert "UNVERIFIED" in mod._hardware_label("amd_default_192gb")


def test_cross_worker_dry_reports_two_separate_deltas():
    """Cross-worker dry run reports APC-only and +LMCache deltas separately."""
    mod = _load_harness()
    cfg = mod.ABConfig(concurrency=4)
    res = mod.run_cross_worker_dry(cfg)
    assert res.measured is False
    assert res.vram_source == "dry"
    assert res.hardware_label == "dry-run (no GPU)"
    # Two distinct deltas exist on the record (not a single blended number).
    assert hasattr(res, "apc_delta_gb")
    assert hasattr(res, "lmcache_delta_gb")
    assert res.apc_delta_gb == 0.0
    assert res.lmcache_delta_gb == 0.0
    assert res.max_concurrency == 4


def test_cross_worker_dry_wires_official_lmcache_connector():
    """The dry run records the EXACT official LMCacheConnectorV1Dynamic JSON."""
    mod = _load_harness()
    res = mod.run_cross_worker_dry(mod.ABConfig(concurrency=2))
    assert res.kv_transfer_config == {
        "kv_connector": "LMCacheConnectorV1Dynamic",
        "kv_role": "kv_both",
        "kv_connector_module_path": (
            "lmcache.integration.vllm.lmcache_connector_v1"
        ),
    }
    # chunk_size MUST equal block_size (alignment invariant) and both are 16.
    assert res.lmcache_chunk_size == res.vllm_block_size == 16


def test_main_cross_worker_dry_returns_zero(capsys):
    """CLI --cross-worker dry mode emits a JSON record with both deltas."""
    mod = _load_harness()
    rc = mod.main(["--mode", "dry", "--cross-worker", "--concurrency", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"measured": false' in out
    assert '"apc_delta_gb": 0.0' in out
    assert '"lmcache_delta_gb": 0.0' in out
    assert "LMCacheConnectorV1Dynamic" in out
