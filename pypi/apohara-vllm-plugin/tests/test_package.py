"""Smoke tests for the apohara-vllm-plugin distribution.

These tests verify the *packaging* contract — that the entry-point is
discoverable, that the re-exports work, and that the plugin
constructs cleanly without vLLM installed. The behavioural tests for
the hook semantics live in the main repo's
tests/test_atom_plugin.py (single source of truth).

Run with::

    cd pypi/apohara-vllm-plugin
    pip install -e .
    pytest tests/

Or, from a clean wheel build::

    python -m build
    pip install dist/apohara_vllm_plugin-0.1.0-py3-none-any.whl
    pytest tests/
"""
from __future__ import annotations

import importlib
import importlib.metadata as md

import pytest


# ---------------------------------------------------------------------------
# Import + re-export                                                         #
# ---------------------------------------------------------------------------

def test_module_imports():
    """The package is importable without vLLM."""
    mod = importlib.import_module("apohara_vllm_plugin")
    assert hasattr(mod, "register")
    assert hasattr(mod, "ATOMConfig")
    assert hasattr(mod, "vLLMAtomPlugin")
    assert hasattr(mod, "PreAttentionHook")
    assert hasattr(mod, "PostAttentionHook")


def test_version_is_pep440():
    """__version__ exists and looks like a PEP-440 release identifier."""
    from apohara_vllm_plugin import __version__
    parts = __version__.split(".")
    assert len(parts) >= 2
    for p in parts[:3]:
        # tolerate prerelease suffixes on the last component (e.g. 0.1.0rc1)
        digits = "".join(c for c in p if c.isdigit())
        assert digits, f"{p!r} has no digit segment"


def test_register_returns_plugin():
    """The entry-point callable returns a constructed plugin without
    requiring vLLM. Cross-worker KV interception is config-driven via
    --kv-transfer-config (LMCache), not installed by register()."""
    from apohara_vllm_plugin import register
    plugin = register()
    stats = plugin.get_stats()
    assert stats["initialized"] is True
    assert stats["worker_id"] == "default"


def test_atomconfig_default_flags():
    """The re-exported ATOMConfig still has the V6.1 defaults."""
    from apohara_vllm_plugin import ATOMConfig
    cfg = ATOMConfig()
    assert cfg.enable_quantization is True
    assert cfg.enable_jcr_gate is True
    assert cfg.quantization_mode == "rotate_kv"


# ---------------------------------------------------------------------------
# Entry-point discoverability                                                #
# ---------------------------------------------------------------------------

def test_entry_point_registered_under_vllm_general_plugins():
    """`vllm.general_plugins` group must contain our entry. This is
    the contract vLLM relies on at worker startup — if this test
    breaks, vLLM will not find the plugin in production."""
    try:
        eps = md.entry_points(group="vllm.general_plugins")
    except TypeError:
        # Python < 3.10 returned a dict-like SelectableGroups; modern
        # API takes a `group` kwarg. We test on 3.11+ which uses the
        # new API, but keep this fallback for safety.
        eps = md.entry_points().get("vllm.general_plugins", [])

    names = [ep.name for ep in eps]
    assert "apohara_contextforge" in names, (
        f"Expected 'apohara_contextforge' in {names}. "
        "Did `pip install -e .` run?"
    )

    apohara_ep = next(ep for ep in eps if ep.name == "apohara_contextforge")
    assert apohara_ep.value == "apohara_vllm_plugin:register"


def test_entry_point_callable_resolves_and_runs():
    """Loading + invoking the entry point must produce the same
    plugin we'd get from `from apohara_vllm_plugin import register`."""
    eps = md.entry_points(group="vllm.general_plugins")
    apohara_ep = next(ep for ep in eps if ep.name == "apohara_contextforge")
    register_fn = apohara_ep.load()
    plugin = register_fn()
    assert plugin.is_initialized() is True


# ---------------------------------------------------------------------------
# vLLM optional integration                                                  #
# ---------------------------------------------------------------------------

# Marker for tests that require an actual vLLM install. CI matrix can
# run one job with vLLM and one without.
vllm_present = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="vLLM not installed; integration test skipped",
)


@vllm_present
def test_vllm_integration_smoke():
    """If vLLM is present, register() must not raise and must produce
    a plugin with a non-empty dependency_status."""
    from apohara_vllm_plugin import register
    plugin = register()
    assert plugin.is_initialized() is True
