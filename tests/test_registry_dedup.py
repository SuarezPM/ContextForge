from __future__ import annotations

from apohara_context_forge.registry.context_registry import ContextRegistry
from apohara_context_forge.compression.coordinator import CompressionCoordinator


def test_registry_exposes_real_dedup_by_default():
    reg = ContextRegistry()
    assert reg.dedup is not None
    assert reg.dedup.count_prefix_tokens("hello world") > 0
    assert (
        reg.dedup.find_shared_prefix("hello world foo", "hello world bar")
        == "hello world"
    )


def test_coordinator_reuses_registry_dedup():
    reg = ContextRegistry()
    coord = CompressionCoordinator(registry=reg)
    assert coord.dedup is reg.dedup
