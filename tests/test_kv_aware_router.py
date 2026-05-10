"""Tests for KVAwareRouter — TASK-009."""
import pytest
from apohara_context_forge.routing.kv_aware_router import KVAwareRouter, RouteDecision, WorkerState


class TestKVAwareRouter:
    """Tests for KV-aware routing."""

    def test_register_worker(self):
        """register_worker() adds worker to routing mesh."""
        router = KVAwareRouter(num_workers=2)
        router.register_worker("worker_0")
        stats = router.get_stats()
        assert stats["num_workers"] == 1

    def test_get_worker_for_anchor_unknown(self):
        """get_worker_for_anchor() returns None for unknown anchor."""
        router = KVAwareRouter()
        result = router.get_worker_for_anchor("unknown_anchor")
        assert result is None

    @pytest.mark.asyncio
    async def test_select_worker_returns_route_decision(self):
        """select_worker() returns RouteDecision."""
        router = KVAwareRouter(num_workers=2)
        router.register_worker("worker_0")
        router.register_worker("worker_1")

        decision = await router.select_worker("anchor_hash", cla_group=1)
        assert isinstance(decision, RouteDecision)
        assert decision.anchor_hash == "anchor_hash"
        assert decision.pre_rope == True  # INVARIANT 10

    @pytest.mark.asyncio
    async def test_select_worker_anchor_locality(self):
        """Same anchor_hash routes to same worker (locality)."""
        router = KVAwareRouter(num_workers=2, enable_anchor_locality=True)
        router.register_worker("worker_0")
        router.register_worker("worker_1")

        d1 = await router.select_worker("anchor_x", cla_group=1)
        d2 = await router.select_worker("anchor_x", cla_group=1)
        # Both should route to same worker
        assert d1.target_worker_id == d2.target_worker_id

    @pytest.mark.asyncio
    async def test_select_worker_load_balancing(self):
        """With no anchor history, routes to least loaded worker."""
        router = KVAwareRouter(num_workers=3)
        for i in range(3):
            router.register_worker(f"worker_{i}")

        decision = await router.select_worker("new_anchor", cla_group=None)
        assert decision.target_worker_id.startswith("worker_")

    @pytest.mark.asyncio
    async def test_update_worker_state(self):
        """update_worker_state() updates worker load and CLA groups."""
        router = KVAwareRouter(num_workers=2)
        router.register_worker("worker_0")

        await router.update_worker_state("worker_0", load=0.75, cla_group=2, workflow_step=5)

        stats = router.get_stats()
        assert stats["worker_loads"]["worker_0"]["load"] == 0.75

    @pytest.mark.asyncio
    async def test_broadcast_new_blocks(self):
        """broadcast_new_blocks() updates routing table."""
        router = KVAwareRouter(num_workers=2)
        router.register_worker("worker_0")

        await router.broadcast_new_blocks("anchor_abc", ["b0", "b1"], "worker_0")

        # Verify anchor now maps to worker
        worker = router.get_worker_for_anchor("anchor_abc")
        assert worker == "worker_0"

    def test_get_stats_returns_worker_states(self):
        """get_stats() returns worker loads and CLA groups."""
        router = KVAwareRouter(num_workers=2)
        router.register_worker("worker_0")
        router.register_worker("worker_1")

        stats = router.get_stats()
        assert "worker_loads" in stats
        assert "worker_0" in stats["worker_loads"]
        assert "worker_1" in stats["worker_loads"]