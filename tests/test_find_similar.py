from __future__ import annotations

import pytest

# Exercises the real registry path (register_agent + FAISS ANN). Skip
# hermetically when faiss isn't installed (the embedder falls back to
# xorshift pseudo-embeddings, but FAISS is a hard requirement here).
pytest.importorskip("faiss")

from apohara_context_forge.models import ContextMatch
from apohara_context_forge.registry.context_registry import ContextRegistry


async def test_find_similar_returns_sorted_typed_matches():
    reg = ContextRegistry()
    await reg.start()
    try:
        sys_prompt = "You are a helpful assistant. " * 20
        await reg.register_agent("a1", sys_prompt, "retriever role")
        await reg.register_agent("a2", sys_prompt, "summarizer role")

        matches = await reg.find_similar(sys_prompt + " retriever role")

        assert isinstance(matches, list)
        assert all(isinstance(m, ContextMatch) for m in matches)
        sims = [m.similarity for m in matches]
        assert sims == sorted(sims, reverse=True)  # sorted desc (empty list ok)
    finally:
        await reg.stop()
