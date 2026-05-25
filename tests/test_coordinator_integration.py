from __future__ import annotations

import pytest

# Exercises decide() against a REAL ContextRegistry (register_agent + FAISS).
# Skip hermetically when faiss isn't installed.
pytest.importorskip("faiss")

from apohara_context_forge.compression.coordinator import CompressionCoordinator
from apohara_context_forge.models import CompressionDecision
from apohara_context_forge.registry.context_registry import ContextRegistry


class _NoopCompressor:
    """Records calls; returns a short body. Keeps the e2e off the LLMLingua model."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    async def compress(self, context: str, rate: float = 0.5):
        self.calls.append((context, rate))
        return "COMPRESSED", 0.5


async def test_decide_against_real_registry_does_not_crash():
    reg = ContextRegistry()
    await reg.start()
    try:
        sys_prompt = "You are a helpful assistant. "  # short -> short ctx
        await reg.register_agent("a1", sys_prompt, "retriever role")
        await reg.register_agent("a2", sys_prompt, "summarizer role")

        coord = CompressionCoordinator(registry=reg, compressor=_NoopCompressor())
        # Before the fix this raised AttributeError (find_similar missing) and
        # the MCP endpoint 503'd. Now it must return a real decision.
        decision = await coord.decide("a3", sys_prompt + " new role")

        assert isinstance(decision, CompressionDecision)
        assert decision.strategy in {
            "apc_reuse", "compress", "compress_and_reuse", "passthrough",
        }
    finally:
        await reg.stop()
