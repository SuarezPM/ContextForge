"""LMCache V2 connector for ContextForge V6.x.

This is the **real** integration with the LMCache distributed KV store
(https://github.com/LMCache/LMCache). It replaces the V4-era
``LMCacheConnectorV1`` stub at :mod:`apohara_context_forge.serving.lmcache_bridge`,
which logged save/load calls but never invoked any LMCache API.

What it actually does
---------------------

* On ``store``: invokes ``engine.store(tokens, kv_tensors, ...)`` to
  push the KV tensors into the LMCache engine. Returns the chunk count
  written (engine returns ``None`` in async mode; we surface that).
* On ``retrieve``: invokes ``engine.retrieve(tokens, ...)`` and
  returns ``(kv_tensors, mask)`` if the lookup hit, else ``None``.
* On ``lookup``: invokes ``engine.lookup(tokens)`` and returns the
  number of tokens already cached (engine's contract). Used by the
  ROMY plugin's pre-attention hook to decide whether to read or
  materialise.
* On ``prefetch``: a thin convenience that calls ``lookup`` and then
  ``retrieve`` for every block in ``token_id_blocks``, returning the
  per-block hit status.

Late imports
------------

``lmcache.*`` is imported inside the constructor, not at module top
level, so this file is importable on a machine that does not have
LMCache installed (HF Spaces, CI without GPU, developer laptop on
Python 3.14, etc.). If LMCache is unavailable and no ``engine``
argument is supplied, the connector enters an honest fallback mode:
``is_active() -> False``, every call returns the documented null
value and logs a single WARNING.

Honest semantics
----------------

Every public method returns a value that reflects what *actually
happened* — not what the config flag asked for:

* ``store(...) -> Optional[int]`` — number of chunks written, or
  ``None`` if no engine.
* ``retrieve(...) -> Optional[tuple[Any, Any]]`` — the engine's
  return tuple iff a real fetch happened, else ``None``.
* ``lookup(...) -> int`` — number of cached tokens, ``0`` if no
  engine.

This is the same discipline applied across V6.1 (see ``AUDIT.md``):
state, not intent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class LMCacheConnectorConfig:
    """User-facing config for the V2 connector.

    The defaults reproduce LMCache's own recommended settings for a
    single-node deployment. For multi-node, point ``remote_url`` at a
    Redis instance reachable from every ContextForge worker.
    """

    instance_id: str = "apohara-contextforge"
    chunk_size: int = 256
    local_device: str = "cpu"
    remote_url: Optional[str] = None       # e.g. "redis://lmcache-redis:6379"
    blocking_store: bool = False           # async store by default
    blocking_retrieve: bool = True         # block on read (hot-path)
    enable_anchor_metadata: bool = True    # attach AnchorPool hints


class LMCacheConnectorV2:
    """Production connector between ContextForge and LMCache.

    Construct with an existing ``LMCacheEngine`` (if your host has
    already built one — typical when vLLM has spun the engine up
    through its own plugin path), or with a
    :class:`LMCacheConnectorConfig` and let this class build the
    engine for you::

        # vLLM-managed engine (preferred — single source of truth)
        engine = ... # from vllm's LMCacheConnectorV1
        conn = LMCacheConnectorV2(engine=engine)

        # Standalone build
        conn = LMCacheConnectorV2(
            config=LMCacheConnectorConfig(
                instance_id="apohara-contextforge",
                remote_url="redis://lmcache-redis:6379",
            ),
        )

    Either way the connector is safe to import on a host that does
    not have ``lmcache`` available — see module docstring.
    """

    def __init__(
        self,
        *,
        engine: Optional[Any] = None,
        config: Optional[LMCacheConnectorConfig] = None,
    ):
        self._config = config or LMCacheConnectorConfig()
        self._engine = engine
        self._build_error: Optional[BaseException] = None
        self._backend: str = "fallback"

        if self._engine is None:
            self._engine = self._try_build_engine()

        self._active = self._engine is not None
        self._stats = {
            "stores":   0,
            "retrieves_hit":  0,
            "retrieves_miss": 0,
            "lookups":  0,
        }

    # ------------------------------------------------------------------ #
    # Engine construction                                                 #
    # ------------------------------------------------------------------ #

    def _try_build_engine(self) -> Optional[Any]:
        """Best-effort LMCache engine construction. Returns the engine
        on success, ``None`` on import error or build error, with a
        single WARNING. Never raises.

        Tries two import strategies in order:
        1. Legacy path: ``lmcache.config`` + ``lmcache.experimental.cache_engine``
           (lmcache < 0.4.x, pre-v1 packaging).
        2. v1 path: ``lmcache.v1.config`` + ``lmcache.v1.cache_engine``
           (lmcache >= 0.4.x, current packaging, works on AMD ROCm without
           ``libcudart.so.12`` because lmcache auto-falls back to
           ``lmcache.non_cuda_equivalents`` for GPU tensor ops).
        """
        LMCacheEngineConfig = None
        LMCacheEngineBuilder = None
        detected_backend: str = "fallback"

        # --- Strategy 1: legacy path (lmcache < 0.4.x) ---
        try:
            from lmcache.config import LMCacheEngineConfig  # type: ignore  # noqa: PLC0415
            from lmcache.experimental.cache_engine import (  # type: ignore  # noqa: PLC0415
                LMCacheEngineBuilder,
            )
            detected_backend = "cuda"
        except ImportError:
            pass

        # --- Strategy 2: v1 path (lmcache >= 0.4.x, CUDA or non-CUDA) ---
        if LMCacheEngineConfig is None:
            try:
                from lmcache.v1.config import (  # type: ignore  # noqa: PLC0415
                    LMCacheEngineConfig,
                )
                from lmcache.v1.cache_engine import (  # type: ignore  # noqa: PLC0415
                    LMCacheEngineBuilder,
                )
                detected_backend = "non_cuda"
            except ImportError:
                pass

        if LMCacheEngineConfig is None:
            # Neither path worked — determine if lmcache is installed at all.
            try:
                import lmcache as _lmcache_probe  # noqa: PLC0415
                del _lmcache_probe
                exc: BaseException = ImportError(
                    "lmcache installed but neither lmcache.config nor "
                    "lmcache.v1.config could be imported"
                )
            except ImportError as _exc:
                exc = _exc
            self._build_error = exc
            logger.warning(
                "LMCacheConnectorV2: lmcache not importable (%s: %s); "
                "connector will run in honest-fallback mode "
                "(every call returns the documented null value).",
                type(exc).__name__, exc,
            )
            return None

        try:
            lmcache_config = LMCacheEngineConfig.from_defaults(
                chunk_size=self._config.chunk_size,
                local_device=self._config.local_device,
                remote_url=self._config.remote_url,
            )
            engine = LMCacheEngineBuilder.get_or_create(
                instance_id=self._config.instance_id,
                config=lmcache_config,
            )
            self._backend = detected_backend
            logger.info(
                "LMCacheConnectorV2: engine built via %s backend "
                "(instance_id=%s, chunk_size=%s, local=%s, remote=%s)",
                detected_backend,
                self._config.instance_id, self._config.chunk_size,
                self._config.local_device,
                self._config.remote_url or "<none>",
            )
            return engine
        except Exception as exc:  # noqa: BLE001
            self._build_error = exc
            logger.warning(
                "LMCacheConnectorV2: lmcache importable (%s backend) but "
                "engine build failed (%s: %s); falling back to no-op mode.",
                detected_backend, type(exc).__name__, exc,
            )
            return None

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def is_active(self) -> bool:
        """True iff a real LMCache engine is wired and ready."""
        return self._active

    def store(
        self,
        tokens: Any,
        kv_tensors: Any,
        *,
        metadata: Optional[dict] = None,
        blocking: Optional[bool] = None,
    ) -> Optional[int]:
        """Push KV tensors into the LMCache engine.

        Returns the number of chunks written, or ``None`` when the
        engine is unavailable / running async (LMCache's own contract
        — async stores return immediately with no count).
        """
        if not self._active:
            return None
        blocking = (
            blocking if blocking is not None else self._config.blocking_store
        )
        try:
            written = self._engine.store(
                tokens=tokens,
                kv_tensors_or_list_of_kv_tensors=kv_tensors,
                blocking=blocking,
            )
            self._stats["stores"] += 1
            return written
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LMCacheConnectorV2.store failed (%s: %s); reporting miss",
                type(exc).__name__, exc,
            )
            return None

    def retrieve(
        self,
        tokens: Any,
        *,
        blocking: Optional[bool] = None,
    ) -> Optional[tuple[Any, Any]]:
        """Try to fetch KV tensors for ``tokens`` from LMCache.

        Returns ``(kv_tensors, mask)`` on hit, ``None`` on miss or
        when the engine is unavailable. LMCache's mask is the per-token
        cached/uncached flag the caller uses to splice the partial hit
        with freshly-computed KV.
        """
        if not self._active:
            return None
        blocking = (
            blocking if blocking is not None else self._config.blocking_retrieve
        )
        try:
            kv, mask = self._engine.retrieve(tokens=tokens, blocking=blocking)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LMCacheConnectorV2.retrieve failed (%s: %s)",
                type(exc).__name__, exc,
            )
            self._stats["retrieves_miss"] += 1
            return None

        # LMCache uses ``mask`` to signal "no hit": typically a zero
        # mask or a None tensor. We treat both as miss.
        try:
            has_hit = bool(mask is not None and getattr(mask, "any", lambda: False)())
        except Exception:  # noqa: BLE001
            has_hit = mask is not None
        if has_hit:
            self._stats["retrieves_hit"] += 1
            return (kv, mask)
        self._stats["retrieves_miss"] += 1
        return None

    def lookup(self, tokens: Any) -> int:
        """How many of ``tokens`` are already cached. ``0`` when the
        engine is unavailable."""
        if not self._active:
            return 0
        try:
            n = int(self._engine.lookup(tokens=tokens))
            self._stats["lookups"] += 1
            return n
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LMCacheConnectorV2.lookup failed (%s: %s); reporting 0",
                type(exc).__name__, exc,
            )
            return 0

    def prefetch(
        self,
        token_id_blocks: Sequence[Any],
    ) -> list[dict]:
        """Drive the engine to warm itself on a batch of blocks.

        Returns one dict per block with ``cached_tokens`` (from
        ``lookup``) and ``retrieved`` (whether the subsequent fetch
        produced a non-trivial mask). Tolerates the no-engine case
        by returning empty stats with ``retrieved=False`` per block.
        """
        results: list[dict] = []
        for block in token_id_blocks:
            cached = self.lookup(block)
            fetched = self.retrieve(block) is not None if cached else False
            results.append({
                "cached_tokens": cached,
                "retrieved": fetched,
            })
        return results

    def close(self) -> None:
        """Release engine resources. Idempotent."""
        if self._engine is None:
            return
        close_fn = getattr(self._engine, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("LMCacheConnectorV2.close: %s", exc)
        self._engine = None
        self._active = False

    # ------------------------------------------------------------------ #
    # Telemetry                                                          #
    # ------------------------------------------------------------------ #

    @property
    def backend(self) -> str:
        """Engine import strategy actually used at construction time.

        One of:

        - ``"cuda"`` — legacy ``lmcache.experimental.cache_engine`` path
          (lmcache < 0.4.x, CUDA wheel).
        - ``"non_cuda"`` — current ``lmcache.v1.cache_engine`` path
          which falls back to ``lmcache.non_cuda_equivalents`` on
          machines without ``libcudart.so.12`` (AMD ROCm / CPU).
        - ``"fallback"`` — neither lmcache import strategy worked;
          the connector runs in honest-fallback mode.

        Exposed for the Grafana dashboard + deployment
        readiness checks: a multi-node MI300X cluster expects
        ``"non_cuda"`` (the v1 path), never ``"cuda"``.
        """
        return self._backend

    def get_stats(self) -> dict:
        return {
            "active": self._active,
            "backend": self._backend,
            "instance_id": self._config.instance_id,
            "chunk_size":  self._config.chunk_size,
            "local_device": self._config.local_device,
            "remote_url":  self._config.remote_url,
            "build_error": (
                None if self._build_error is None
                else f"{type(self._build_error).__name__}: {self._build_error}"
            ),
            **self._stats,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LMCacheConnectorV2(active={self._active}, "
            f"instance_id={self._config.instance_id!r}, "
            f"stores={self._stats['stores']}, "
            f"hits={self._stats['retrieves_hit']}, "
            f"misses={self._stats['retrieves_miss']})"
        )


# ---------------------------------------------------------------------------
# Legacy alias                                                               #
# ---------------------------------------------------------------------------
#
# Callers importing the V1 bridge under its historical name still work.
# The V1 class itself stays at apohara_context_forge.serving.lmcache_bridge
# for backwards compat with old tests / agents; new code should target V2.

__all__ = [
    "LMCacheConnectorConfig",
    "LMCacheConnectorV2",
]
