# MERGED: OpenCode (deep KV physics) + CC (surface coverage)
# All tests hermetic: no GPU, no TCP, no downloaded weights required
from __future__ import annotations

import pytest

# Optional dep guard — skip entire module if sentence_transformers not installed
# (CompressionCoordinator requires SemanticDedupEngine which requires sentence_transformers)
try:
    import sentence_transformers  # noqa: F401
    _HAS_SENTENCE_TRANSFORMERS = True
except ModuleNotFoundError:
    pytest.skip("sentence_transformers not installed", allow_module_level=True)

from typing import Any

from apohara_context_forge.compression.coordinator import CompressionCoordinator
from apohara_context_forge.config import settings
from apohara_context_forge.models import CompressionDecision, ContextMatch


# ---- Hermetic doubles (zero real model loads) ------------------------------------


class FakeDedup:
    """Deterministic stand-in for SemanticDedupEngine.

    ``count_prefix_tokens`` returns a value from a precomputed dict-by-string
    when present, else falls back to ``len(s) // 4`` so tests never depend on
    a real tokenizer. ``find_shared_prefix`` mirrors the real char-level loop.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def set_count(self, text: str, count: int) -> None:
        self._counts[text] = count

    def count_prefix_tokens(self, prefix: str) -> int:
        if prefix in self._counts:
            return self._counts[prefix]
        return len(prefix) // 4

    def find_shared_prefix(self, a: str, b: str) -> str:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        if i == n:
            return a[:i]
        j = a.rfind(" ", 0, i)
        return a[:j] if j > 0 else a[:i]


class FakeContextRegistry:
    """Stores a list of pre-canned ``ContextMatch`` objects and returns them
    sorted desc by similarity from ``find_similar`` (mirroring the real
    registry's contract). Exposes ``dedup`` so the coordinator's sharing rule
    can be exercised without loading a real embedder.
    """

    def __init__(self, dedup: FakeDedup | None = None) -> None:
        self._matches: list[ContextMatch] = []
        self.dedup: FakeDedup = dedup if dedup is not None else FakeDedup()
        self.find_similar_calls: list[tuple[str, float | None]] = []

    def set_matches(self, matches: list[ContextMatch]) -> None:
        self._matches = list(matches)

    async def find_similar(
        self, context: str, threshold: float | None = None
    ) -> list[ContextMatch]:
        self.find_similar_calls.append((context, threshold))
        return sorted(self._matches, key=lambda m: m.similarity, reverse=True)


class FakeCompressor:
    """Async double for ``ContextCompressor.compress``. Returns a deterministic
    ``(body, 0.5)`` tuple and records every call's (context, rate) pair so
    tests can assert which slice of the input was compressed.
    """

    def __init__(self, response: str | None = None) -> None:
        self.calls: list[tuple[str, float]] = []
        self._response = response

    async def compress(self, context: str, rate: float = 0.5) -> tuple[str, float]:
        self.calls.append((context, rate))
        body = (
            self._response
            if self._response is not None
            else f"COMPRESSED({len(context)})"
        )
        return body, 0.5


def make_coordinator(
    *,
    registry: FakeContextRegistry | None = None,
    compressor: FakeCompressor | None = None,
    dedup: FakeDedup | None = None,
) -> CompressionCoordinator:
    return CompressionCoordinator(
        registry=registry,
        compressor=compressor,
        dedup=dedup,
    )


# ---- Strategy branch tests --------------------------------------------------------


async def test_apc_reuse_strategy_when_strong_match_long_prefix_short_ctx():
    dedup = FakeDedup()
    registry = FakeContextRegistry(dedup=dedup)
    compressor = FakeCompressor()

    incoming = "hello agent_a body short enough to skip compression"
    dedup.set_count(incoming, 400)  # < COMPRESS_MIN_CONTEXT_TOKENS=500

    match = ContextMatch(
        agent_id="agent_a",
        similarity=0.95,
        shared_prefix="hello agent_a body",
        shared_prefix_tokens=300,  # > APC_REUSE_MIN_SHARED_PREFIX_TOKENS=200
    )
    registry.set_matches([match])

    coord = make_coordinator(registry=registry, compressor=compressor, dedup=dedup)
    decision = await coord.decide("agent_a", incoming)

    assert decision.strategy == "apc_reuse"
    assert decision.final_context == incoming
    assert decision.shared_prefix == "hello agent_a body"
    assert decision.original_tokens == 400
    assert decision.final_tokens == 400
    assert decision.tokens_saved == 0
    # apc_reuse must NEVER call the compressor.
    assert compressor.calls == []


async def test_compress_and_reuse_when_strong_match_long_prefix_long_ctx():
    dedup = FakeDedup()
    registry = FakeContextRegistry(dedup=dedup)
    compressor = FakeCompressor(response="COMPRESSED_TAIL")

    shared_prefix = "agent shared prefix portion"
    unique_tail = " tail body that is much longer and should be compressed"
    incoming = shared_prefix + unique_tail
    dedup.set_count(incoming, 800)
    dedup.set_count("COMPRESSED_TAIL", 50)

    match = ContextMatch(
        agent_id="agent_b",
        similarity=0.95,
        shared_prefix=shared_prefix,
        shared_prefix_tokens=300,
    )
    registry.set_matches([match])

    coord = make_coordinator(registry=registry, compressor=compressor, dedup=dedup)
    decision = await coord.decide("agent_b", incoming)

    assert decision.strategy == "compress_and_reuse"
    assert decision.final_context.startswith(shared_prefix)
    assert decision.final_context.endswith("COMPRESSED_TAIL")
    # Compressor MUST be called exactly once with the unique tail — not the
    # full incoming context — so KV-prefix reuse pays off downstream.
    assert len(compressor.calls) == 1
    assert compressor.calls[0][0] == unique_tail
    assert compressor.calls[0][1] == settings.CONTEXTFORGE_COMPRESSION_RATE
    assert decision.final_tokens == 350  # 300 (prefix) + 50 (compressed tail)
    assert decision.tokens_saved == 800 - 350
    assert decision.shared_prefix == shared_prefix


async def test_compress_when_no_long_prefix_long_ctx():
    dedup = FakeDedup()
    registry = FakeContextRegistry(dedup=dedup)
    compressor = FakeCompressor(response="WHOLE_COMPRESSED")

    incoming = (
        "long body that needs full-context compression because the prefix "
        "overlap is too small to bother with KV reuse"
    )
    dedup.set_count(incoming, 800)
    dedup.set_count("WHOLE_COMPRESSED", 100)

    # Strong sim but short prefix — has_long_prefix=False forces "compress".
    match = ContextMatch(
        agent_id="agent_c",
        similarity=0.90,
        shared_prefix="short",
        shared_prefix_tokens=50,
    )
    registry.set_matches([match])

    coord = make_coordinator(registry=registry, compressor=compressor, dedup=dedup)
    decision = await coord.decide("agent_c", incoming)

    assert decision.strategy == "compress"
    assert decision.final_context == "WHOLE_COMPRESSED"
    assert decision.shared_prefix == ""
    # Compressor MUST be called with the FULL incoming context.
    assert len(compressor.calls) == 1
    assert compressor.calls[0][0] == incoming
    assert decision.final_tokens == 100
    assert decision.tokens_saved == 700


async def test_passthrough_when_no_match_short_ctx():
    dedup = FakeDedup()
    registry = FakeContextRegistry(dedup=dedup)  # empty matches
    compressor = FakeCompressor()

    incoming = "tiny context that is just passed through"
    dedup.set_count(incoming, 200)

    coord = make_coordinator(registry=registry, compressor=compressor, dedup=dedup)
    decision = await coord.decide("agent_d", incoming)

    assert decision.strategy == "passthrough"
    assert decision.final_context == incoming
    assert decision.shared_prefix == ""
    assert decision.original_tokens == 200
    assert decision.final_tokens == 200
    assert decision.tokens_saved == 0
    assert compressor.calls == []


async def test_passthrough_when_short_ctx_with_weak_match():
    """Strong sim + short prefix + short ctx → passthrough preserves weak prefix."""
    dedup = FakeDedup()
    registry = FakeContextRegistry(dedup=dedup)
    compressor = FakeCompressor()

    incoming = "short body with a weak match preserved for observability"
    dedup.set_count(incoming, 200)  # short — long_enough=False

    match = ContextMatch(
        agent_id="agent_e",
        similarity=0.90,
        shared_prefix="short body",
        shared_prefix_tokens=50,  # weak — has_long_prefix=False
    )
    registry.set_matches([match])

    coord = make_coordinator(registry=registry, compressor=compressor, dedup=dedup)
    decision = await coord.decide("agent_e", incoming)

    assert decision.strategy == "passthrough"
    # Weak-match prefix surfaced to the caller for observability, even though
    # we did not act on it.
    assert decision.shared_prefix == "short body"
    assert decision.final_context == incoming
    assert decision.tokens_saved == 0
    assert compressor.calls == []


async def test_no_prior_contexts_returns_passthrough():
    dedup = FakeDedup()
    registry = FakeContextRegistry(dedup=dedup)
    compressor = FakeCompressor()

    incoming = "first context ever"
    dedup.set_count(incoming, 50)

    coord = make_coordinator(registry=registry, compressor=compressor, dedup=dedup)
    decision = await coord.decide("first_agent", incoming)

    assert decision.strategy == "passthrough"
    assert decision.shared_prefix == ""
    assert compressor.calls == []


async def test_decide_uses_registry_dedup_engine_by_default():
    """When only `registry` is provided, coordinator MUST reuse `registry.dedup`
    (identity check) so we never spin up a second embedder."""
    shared_dedup = FakeDedup()
    registry = FakeContextRegistry(dedup=shared_dedup)
    compressor = FakeCompressor()

    coord = CompressionCoordinator(registry=registry, compressor=compressor)

    assert coord.dedup is registry.dedup
    assert coord.dedup is shared_dedup

    # Exercising decide() must not allocate a new dedup somewhere internally.
    incoming = "x"  # ctx_tokens proxy = 0 → passthrough
    decision = await coord.decide("solo", incoming)
    assert isinstance(decision, CompressionDecision)
    assert coord.dedup is shared_dedup


async def test_compression_decision_strict_typing():
    """R014 strict-typing surface check: Pydantic round-trip is identity, and
    `strategy` is one of the four documented literals."""
    dedup = FakeDedup()
    registry = FakeContextRegistry(dedup=dedup)
    compressor = FakeCompressor()

    incoming = "anything"
    dedup.set_count(incoming, 100)

    coord = make_coordinator(registry=registry, compressor=compressor, dedup=dedup)
    decision = await coord.decide("agent_strict", incoming)

    assert type(decision) is CompressionDecision
    assert decision.strategy in {
        "apc_reuse",
        "compress",
        "compress_and_reuse",
        "passthrough",
    }

    payload: dict[str, Any] = decision.model_dump()
    rebuilt = CompressionDecision.model_validate(payload)
    assert rebuilt == decision
    assert rebuilt.model_dump() == payload


# ---- Boundary / negative tests (Q7) ----------------------------------------------


async def test_long_enough_uses_strict_greater_than_at_boundary():
    """ctx_tokens == COMPRESS_MIN_CONTEXT_TOKENS (=500) → long_enough is False."""
    dedup = FakeDedup()
    registry = FakeContextRegistry(dedup=dedup)
    compressor = FakeCompressor()

    incoming = "boundary body"
    dedup.set_count(incoming, settings.COMPRESS_MIN_CONTEXT_TOKENS)  # exactly 500

    coord = make_coordinator(registry=registry, compressor=compressor, dedup=dedup)
    decision = await coord.decide("agent_b1", incoming)

    # No match AND long_enough=False → passthrough; compressor untouched.
    assert decision.strategy == "passthrough"
    assert compressor.calls == []


async def test_has_long_prefix_uses_strict_greater_than_at_boundary():
    """shared_prefix_tokens == APC_REUSE_MIN_SHARED_PREFIX_TOKENS (=200) → not long enough."""
    dedup = FakeDedup()
    registry = FakeContextRegistry(dedup=dedup)
    compressor = FakeCompressor(response="WHOLE")

    incoming = "boundary prefix body for whole-context compression check"
    dedup.set_count(incoming, 800)  # long_enough=True
    dedup.set_count("WHOLE", 50)

    match = ContextMatch(
        agent_id="agent_b2",
        similarity=0.95,
        shared_prefix="boundary prefix",
        shared_prefix_tokens=settings.APC_REUSE_MIN_SHARED_PREFIX_TOKENS,
    )
    registry.set_matches([match])

    coord = make_coordinator(registry=registry, compressor=compressor, dedup=dedup)
    decision = await coord.decide("agent_b2", incoming)

    # has_long_prefix=False (strict >), long_enough=True → compress branch
    # with FULL context, not unique tail.
    assert decision.strategy == "compress"
    assert len(compressor.calls) == 1
    assert compressor.calls[0][0] == incoming


async def test_registry_results_already_sorted_picks_top_match():
    """Coordinator trusts registry-sort-desc — picks matches[0] as best."""
    dedup = FakeDedup()
    registry = FakeContextRegistry(dedup=dedup)
    compressor = FakeCompressor()

    incoming = "overlap body"
    dedup.set_count(incoming, 100)  # short — only apc_reuse can fire

    matches = [
        ContextMatch(
            agent_id="lower",
            similarity=0.86,
            shared_prefix="overlap",
            shared_prefix_tokens=50,  # weak — would NOT fire apc_reuse alone
        ),
        ContextMatch(
            agent_id="higher",
            similarity=0.95,
            shared_prefix="overlap body",
            shared_prefix_tokens=300,  # strong — fires apc_reuse
        ),
    ]
    registry.set_matches(matches)

    coord = make_coordinator(registry=registry, compressor=compressor, dedup=dedup)
    decision = await coord.decide("agent_pick", incoming)

    assert decision.strategy == "apc_reuse"
    assert decision.shared_prefix == "overlap body"
    assert compressor.calls == []
