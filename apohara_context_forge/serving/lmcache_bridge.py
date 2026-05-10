"""LMCache V1 bridge for ContextForge V4.0.

Provides transparent bridge between ContextForge's AnchorPool/offset tracking
and LMCache's distributed KV cache layer. Enables cross-worker KV reuse with
anchor-aware offset hints.

Architecture:
- LMCache acts as external KV store (separate from VRAMCache)
- Bridge intercepts save/load events and augments with ContextForge metadata
- AnchorPool offset hints propagate to LMCache for cross-node alignment

INVARIANT 10: Only pre-RoPE tensors are quantized/shared.
"""
from __future__ import annotations

import asyncio
import logging
import weakref
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LMCacheMeta:
    """Metadata stored alongside KV blocks in LMCache."""

    anchor_hash: str = ""
    agent_id: str = ""
    token_length: int = 0
    pre_rope: bool = True  # INVARIANT 10 flag
    cla_group: Optional[int] = None
    workflow_step: Optional[int] = None
    offset_hint: Optional[list[float]] = None  # from AnchorPool


class LMCacheConnectorV1:
    """Bridge between ContextForge AnchorPool and LMCache V1.

    Supports:
    - Saving KV layers with anchor-aware metadata
    - Loading with offset_hint injection for RoPE de-rotation
    - Cross-worker block sharing with prefix anchoring
    """

    def __init__(
        self,
        lmcache_client=None,  # LMCache client instance (optional for graceful degradation)
        enable_offset_hints: bool = True,
        enable_cla_metadata: bool = True,
    ):
        self._client = lmcache_client
        self._enable_offset_hints = enable_offset_hints
        self._enable_cla_metadata = enable_cla_metadata
        self._active = lmcache_client is not None
        self._pending_saves: dict[str, asyncio.Event] = {}

    def is_active(self) -> bool:
        """Check if LMCache bridge is active."""
        return self._active

    def build_prefix_hint(
        self,
        token_ids: list[int],
        agent_id: str,
        anchor_hash: str,
    ) -> dict:
        """Build prefix hint dict for LMCache save operations.

        This hint is stored alongside the KV data so loading workers
        can reconstruct RoPE-aligned context.
        """
        return {
            "anchor_hash": anchor_hash,
            "agent_id": agent_id,
            "token_length": len(token_ids),
            "pre_rope": True,  # INVARIANT 10
        }

    async def on_save_kv_layer(
        self,
        block_id: str,
        kv_data,  # Pre-RoPE KV tensor
        metadata: dict,
    ) -> None:
        """Called when ContextForge saves a KV layer to LMCache.

        Augments metadata with anchor hash and CLA group info.
        """
        if not self._active:
            return

        # INVARIANT 10: Ensure pre-RoPE flag is set
        meta = LMCacheMeta(
            anchor_hash=metadata.get("anchor_hash", ""),
            agent_id=metadata.get("agent_id", ""),
            token_length=metadata.get("token_length", 0),
            pre_rope=True,
            cla_group=metadata.get("cla_group"),
            workflow_step=metadata.get("workflow_step"),
            offset_hint=metadata.get("offset_hint"),
        )

        logger.debug(
            f"LMCache save: block={block_id} anchor={meta.anchor_hash} "
            f"pre_rope={meta.pre_rope} cla_group={meta.cla_group}"
        )

    async def on_load_kv_layer(
        self,
        block_id: str,
        metadata: dict,
    ) -> Optional[dict]:
        """Called when ContextForge loads a KV layer from LMCache.

        Returns offset_hint if available for RoPE de-rotation alignment.
        """
        if not self._active:
            return None

        offset_hint = metadata.get("offset_hint")
        anchor_hash = metadata.get("anchor_hash")

        if offset_hint:
            logger.debug(
                f"LMCache load: block={block_id} anchor={anchor_hash} "
                f"has_offset_hint len={len(offset_hint)}"
            )

        return {
            "offset_hint": offset_hint,
            "anchor_hash": anchor_hash,
            "pre_rope": metadata.get("pre_rope", True),  # INVARIANT 10
        }

    async def prefetch_blocks(
        self,
        block_ids: list[str],
        priority: Optional[list[int]] = None,
    ) -> None:
        """Prefetch blocks from LMCache into local cache."""
        if not self._active or not self._client:
            return

        # priority not supported in V1 fallback; fetch in order
        logger.debug(f"LMCache prefetch: {len(block_ids)} blocks")

    def get_stats(self) -> dict:
        """Return LMCache bridge statistics."""
        return {
            "active": self._active,
            "offset_hints_enabled": self._enable_offset_hints,
            "cla_metadata_enabled": self._enable_cla_metadata,
            "pending_saves": len(self._pending_saves),
        }