"""KV-aware routing for ContextForge V4.0.

Routes KV cache requests based on:
- Anchor hash locality (blocks with same anchor_hash → same worker)
- CLA group affinity (upper-layer CLA groups prefer specific workers)
- VRAM pressure balancing (avoid overloaded workers)
- Workflow step context (consecutive steps prefer same worker)

INVARIANT 10: Only pre-RoPE tensors are quantized/shared.
Routing decisions are made on anchor metadata, not on actual KV tensors.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkerState:
    """State of a worker in the KV routing mesh."""

    worker_id: str = ""
    anchor_scores: dict[str, float] = field(default_factory=dict)  # anchor_hash → affinity
    cla_groups: set[int] = field(default_factory=set)  # CLA groups served
    current_load: float = 0.0  # 0.0-1.0
    last_used_step: int = 0
    active_blocks: int = 0


@dataclass
class RouteDecision:
    """Routing decision for a KV block request."""

    target_worker_id: str
    anchor_hash: str
    cla_group: Optional[int]
    confidence: float  # 0.0-1.0
    pre_rope: bool = True  # INVARIANT 10


class KVAwareRouter:
    """Routes KV cache traffic based on anchor locality and worker state.

    Design principles:
    1. Anchor hash locality: blocks with same anchor_hash route to same worker
    2. CLA group affinity: upper-layer CLA groups have preferred workers
    3. Load balancing: VRAM pressure influences routing decisions
    4. Workflow continuity: consecutive steps prefer same worker

    INVARIANT 10: Routing decisions are made on anchor metadata only.
    Actual KV tensors are never inspected for routing.
    """

    def __init__(
        self,
        num_workers: int = 1,
        enable_cla_affinity: bool = True,
        enable_anchor_locality: bool = True,
    ):
        self._num_workers = num_workers
        self._enable_cla_affinity = enable_cla_affinity
        self._enable_anchor_locality = enable_anchor_locality
        self._workers: dict[str, WorkerState] = {}
        self._anchor_to_worker: dict[str, str] = {}  # anchor_hash → worker_id
        self._lock = asyncio.Lock()

    def register_worker(self, worker_id: str) -> None:
        """Register a worker in the routing mesh."""
        if worker_id not in self._workers:
            self._workers[worker_id] = WorkerState(worker_id=worker_id)
            logger.info(f"Router: registered worker {worker_id}")

    async def select_worker(
        self,
        anchor_hash: str,
        cla_group: Optional[int] = None,
        workflow_step: Optional[int] = None,
        token_length: int = 0,
    ) -> RouteDecision:
        """Select optimal worker for a KV block with given anchor.

        Returns RouteDecision with target_worker_id and routing metadata.
        """
        async with self._lock:
            # 1. Check if this anchor already has a preferred worker (locality)
            if self._enable_anchor_locality and anchor_hash in self._anchor_to_worker:
                preferred_worker = self._anchor_to_worker[anchor_hash]
                if preferred_worker in self._workers:
                    worker_state = self._workers[preferred_worker]
                    # Check load isn't too high
                    if worker_state.current_load < 0.95:
                        return RouteDecision(
                            target_worker_id=preferred_worker,
                            anchor_hash=anchor_hash,
                            cla_group=cla_group,
                            confidence=0.9,
                            pre_rope=True,  # INVARIANT 10
                        )

            # 2. Find best worker based on CLA affinity
            if self._enable_cla_affinity and cla_group is not None:
                for worker_id, state in self._workers.items():
                    if cla_group in state.cla_groups and state.current_load < 0.8:
                        self._anchor_to_worker[anchor_hash] = worker_id
                        state.anchor_scores[anchor_hash] = 0.8
                        return RouteDecision(
                            target_worker_id=worker_id,
                            anchor_hash=anchor_hash,
                            cla_group=cla_group,
                            confidence=0.75,
                            pre_rope=True,
                        )

            # 3. Fall back to least loaded worker
            if self._workers:
                sorted_workers = sorted(
                    self._workers.items(),
                    key=lambda x: x[1].current_load
                )
                target_worker_id, target_state = sorted_workers[0]
                self._anchor_to_worker[anchor_hash] = target_worker_id
                target_state.anchor_scores[anchor_hash] = 0.5
                return RouteDecision(
                    target_worker_id=target_worker_id,
                    anchor_hash=anchor_hash,
                    cla_group=cla_group,
                    confidence=0.5,
                    pre_rope=True,
                )

            # No workers available
            return RouteDecision(
                target_worker_id="",
                anchor_hash=anchor_hash,
                cla_group=cla_group,
                confidence=0.0,
                pre_rope=True,
            )

    async def update_worker_state(
        self,
        worker_id: str,
        load: float,
        cla_group: Optional[int] = None,
        workflow_step: Optional[int] = None,
    ) -> None:
        """Update state for a worker after processing blocks."""
        async with self._lock:
            if worker_id not in self._workers:
                self.register_worker(worker_id)

            state = self._workers[worker_id]
            state.current_load = min(load, 1.0)
            if cla_group is not None:
                state.cla_groups.add(cla_group)
            if workflow_step is not None:
                state.last_used_step = workflow_step

    async def broadcast_new_blocks(
        self,
        anchor_hash: str,
        block_ids: list[str],
        target_worker_id: str,
    ) -> None:
        """Broadcast new block IDs to all workers for awareness."""
        async with self._lock:
            logger.debug(
                f"Broadcast: anchor={anchor_hash} blocks={len(block_ids)} "
                f"to worker={target_worker_id}"
            )
            # Record in routing table
            self._anchor_to_worker[anchor_hash] = target_worker_id

            if target_worker_id in self._workers:
                self._workers[target_worker_id].anchor_scores[anchor_hash] = 1.0

    def get_worker_for_anchor(self, anchor_hash: str) -> Optional[str]:
        """Get the preferred worker for an anchor hash (if any)."""
        return self._anchor_to_worker.get(anchor_hash)

    def get_stats(self) -> dict:
        """Return router statistics."""
        return {
            "num_workers": len(self._workers),
            "anchors_tracked": len(self._anchor_to_worker),
            "cla_affinity_enabled": self._enable_cla_affinity,
            "anchor_locality_enabled": self._enable_anchor_locality,
            "worker_loads": {
                wid: {
                    "load": round(state.current_load, 3),
                    "cla_groups": len(state.cla_groups),
                    "active_blocks": state.active_blocks,
                }
                for wid, state in self._workers.items()
            },
        }