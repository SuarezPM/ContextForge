"""Tests for apohara_context_forge/serving/vllm_launch_config.py.

Pure-function tests: no vllm, no lmcache, no server. They pin the EXACT
``--kv-transfer-config`` JSON to the documented LMCacheConnectorV1Dynamic
invocation, the mandatory PYTHONHASHSEED=0 worker env, and the
block_size==chunk_size alignment invariant (including the loud failure
on misalignment).
"""
import json

import pytest

from apohara_context_forge.serving.vllm_launch_config import (
    DEFAULT_BLOCK_SIZE,
    LMCACHE_KV_CONNECTOR,
    LMCACHE_KV_CONNECTOR_MODULE_PATH,
    LMCACHE_KV_ROLE,
    build_kv_transfer_config,
    build_kv_transfer_config_json,
    build_vllm_serve_args,
    worker_env,
)

# The canonical LMCache/vLLM invocation, verbatim from the official docs.
_EXPECTED_KV_TRANSFER_CONFIG = {
    "kv_connector": "LMCacheConnectorV1Dynamic",
    "kv_role": "kv_both",
    "kv_connector_module_path": "lmcache.integration.vllm.lmcache_connector_v1",
}


def test_kv_transfer_config_dict_is_exact():
    """The dict matches the documented LMCacheConnectorV1Dynamic invocation."""
    assert build_kv_transfer_config() == _EXPECTED_KV_TRANSFER_CONFIG


def test_kv_transfer_config_module_constants():
    """The exported constants are the exact public LMCache contract strings."""
    assert LMCACHE_KV_CONNECTOR == "LMCacheConnectorV1Dynamic"
    assert LMCACHE_KV_ROLE == "kv_both"
    assert (
        LMCACHE_KV_CONNECTOR_MODULE_PATH
        == "lmcache.integration.vllm.lmcache_connector_v1"
    )


def test_kv_transfer_config_json_is_exact():
    """The JSON string round-trips to exactly the documented config."""
    s = build_kv_transfer_config_json()
    assert json.loads(s) == _EXPECTED_KV_TRANSFER_CONFIG
    # Keys are emitted in the documented order (not sorted): connector first.
    assert s.index('"kv_connector"') < s.index('"kv_role"')
    assert s.index('"kv_role"') < s.index('"kv_connector_module_path"')


def test_worker_env_pins_pythonhashseed_zero():
    """PYTHONHASHSEED=0 is mandatory for cross-worker prefix-hash agreement."""
    env = worker_env()
    assert env["PYTHONHASHSEED"] == "0"
    # v1 engine selection (the one the Dynamic connector drives).
    assert env["LMCACHE_USE_EXPERIMENTAL"] == "True"


def test_worker_env_extra_merges_without_dropping_seed():
    """Caller overrides merge but cannot accidentally drop the pinned seed."""
    env = worker_env({"FOO": "bar"})
    assert env["FOO"] == "bar"
    assert env["PYTHONHASHSEED"] == "0"


def test_worker_env_returns_fresh_dict():
    """worker_env must not return a shared/mutable singleton."""
    a = worker_env()
    a["PYTHONHASHSEED"] = "999"
    b = worker_env()
    assert b["PYTHONHASHSEED"] == "0"


def test_alignment_default_matches_block_size():
    """The default chunk_size equals the default block_size (16)."""
    assert DEFAULT_BLOCK_SIZE == 16
    # Aligned defaults must not raise.
    assert build_kv_transfer_config() == _EXPECTED_KV_TRANSFER_CONFIG


def test_aligned_custom_sizes_ok():
    """Equal custom block/chunk sizes are accepted."""
    assert build_kv_transfer_config(block_size=32, chunk_size=32) == (
        _EXPECTED_KV_TRANSFER_CONFIG
    )


def test_misaligned_sizes_fail_loud():
    """Misaligned chunk/block sizes raise ValueError, not silent degradation."""
    with pytest.raises(ValueError, match="must equal vLLM block_size"):
        build_kv_transfer_config(block_size=16, chunk_size=256)


def test_misaligned_sizes_fail_loud_in_json():
    """The JSON builder propagates the alignment failure."""
    with pytest.raises(ValueError, match="must equal vLLM block_size"):
        build_kv_transfer_config_json(block_size=16, chunk_size=8)


def test_serve_args_carry_block_size_and_transfer_config():
    """The CLI args wire --block-size and the exact --kv-transfer-config JSON."""
    args = build_vllm_serve_args("qwen3-embed", block_size=16, chunk_size=16)
    assert args[0] == "serve"
    assert args[1] == "qwen3-embed"
    assert "--block-size" in args
    assert args[args.index("--block-size") + 1] == "16"
    assert "--kv-transfer-config" in args
    cfg_json = args[args.index("--kv-transfer-config") + 1]
    assert json.loads(cfg_json) == _EXPECTED_KV_TRANSFER_CONFIG


def test_serve_args_reject_misalignment():
    """Building serve args with misaligned sizes fails loud too."""
    with pytest.raises(ValueError, match="must equal vLLM block_size"):
        build_vllm_serve_args("qwen3-embed", block_size=16, chunk_size=64)
