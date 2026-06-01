"""Apohara ContextForge plugin for vLLM V1.

This is a thin shim over the in-tree implementation at
:mod:`apohara_context_forge.serving.romy_plugin`. It exists so we can
publish a small, focused PyPI distributable whose only job is to be
discoverable through the ``vllm.general_plugins`` entry-point group.

Public surface
--------------

``register``         Entry-point callable invoked by vLLM at worker
                     startup. Returns a configured plugin instance.
``ROMYConfig``       Re-export, for users who want to construct a
                     custom-configured plugin.
``vLLMRomyPlugin``   Re-export of the plugin class itself.

Usage from vLLM (automatic)
---------------------------

Once ``pip install apohara-vllm-plugin`` has run inside the same
environment as vLLM, no further wiring is needed: vLLM enumerates the
``vllm.general_plugins`` entry-point group at startup and invokes
``apohara_vllm_plugin:register``.

Usage from outside vLLM (manual, for tests)
-------------------------------------------

>>> from apohara_vllm_plugin import register
>>> plugin = register()
>>> plugin.is_initialized()
True

The plugin is constructed even when vLLM is not importable, so the
import is safe in pure-Python test environments. Cross-worker KV
interception is not wired by this package; it is config-driven via
vLLM's --kv-transfer-config (LMCache) — see register().
"""
from __future__ import annotations

# Re-export the canonical implementations. Keeping the wire-level
# implementation in apohara_context_forge means there is exactly one
# source of truth for the hook semantics, and the same code is
# exercised by the main repo's 19 unit tests in tests/test_romy_plugin.py.
from apohara_context_forge.serving.romy_plugin import (
    ROMYConfig,
    PostAttentionHook,
    PreAttentionHook,
    register,
    vLLMRomyPlugin,
)

__all__ = [
    "ROMYConfig",
    "PostAttentionHook",
    "PreAttentionHook",
    "register",
    "vLLMRomyPlugin",
]

# Version mirrors apohara-context-forge's V6.1 baseline. Bumped
# independently from this point forward — V6.x #1 ships as 0.1.0.
__version__ = "0.1.0"
