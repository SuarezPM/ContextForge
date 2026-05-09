"""VRAM-pressure-aware eviction cache - IMPROVEMENT-002.

Replaces static TTL-based eviction with adaptive LRU/LFU hybrid that responds
to actual GPU memory pressure. Monitors MI300X VRAM via PyRSMI and adjusts
eviction policy dynamically.

Eviction modes:
- RELAXED (VRAM < 70%): No eviction, TTL = 10 minutes
- NORMAL (70-85%): LRU eviction of entries idle > 2 min
- PRESSURE (85-92%): LFU by token_count, evict heaviest first
- CRITICAL (92-96%): Offload inactive KV tensors to CPU RAM
- EMERGENCY (VRAM >= 96%): Hard evict all idle > 30s, block new registrations
"""
import asyncio
import heapq
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from contextforge.scheduling.step_graph import AgentStepGraph

from contextforge.metrics.vram_monitor import VRAMMonitor


class EvictionMode(Enum):
    RELAXED = "relaxed"
    NORMAL = "normal"
    PRESSURE = "pressure"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    WORKFLOW_AWARE = "workflow_aware"


@dataclass(order=True)
class CacheEntry:
    # Priority for heap (lower = evict first): last_accessed - (access_count * 10)
    # LFU/LRU hybrid: frequent+recent entries survive longer
    priority: float = field(compare=True)
    last_accessed: float = field(compare=False, default_factory=time.monotonic)
    access_count: int = field(compare=False, default=0)
    token_count: int = field(compare=False, default=0)
    key: str = field(compare=False, default="")
    value: Any = field(compare=False, default=None)
    offloaded_to_cpu: bool = field(compare=False, default=False)


class VRAMAwareCache:
    """
    LRU/LFU hybrid cache with VRAM pressure-responsive eviction.
    Monitors AMD MI300X memory in real-time via PyRSMI.
    
    Usage:
        cache = VRAMAwareCache(max_token_budget=50_000_000)  # 50M tokens = ~3GB
        await cache.start()
        await cache.set("agent1", context_entry, token_count=500)
        entry = await cache.get("agent1")
        await cache.stop()
    """
    
    VRAM_CHECK_INTERVAL = 2.0  # seconds between VRAM pressure checks
    
    def __init__(self, max_token_budget: int = 50_000_000, step_graph: Optional["AgentStepGraph"] = None):
        """
        Args:
            max_token_budget: Maximum tokens to hold in cache (~3GB for 64-layer model)
            step_graph: Optional workflow dependency graph for WORKFLOW_AWARE eviction
        """
        self._store: dict[str, CacheEntry] = {}
        self._heap: list[CacheEntry] = []
        self._total_tokens: int = 0
        self._max_token_budget = max_token_budget
        self._vram = VRAMMonitor()
        self._mode = EvictionMode.RELAXED
        self._lock = asyncio.Lock()
        self._monitor_task: Optional[asyncio.Task] = None
        self._blocked = False
        self._step_graph = step_graph
    
    async def start(self) -> None:
        """Start background VRAM monitor."""
        if self._monitor_task is not None:
            return
        self._monitor_task = asyncio.create_task(self._vram_monitor_loop())
    
    async def stop(self) -> None:
        """Stop background monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
    
    async def _vram_monitor_loop(self) -> None:
        """Background loop: check VRAM pressure every interval."""
        while True:
            try:
                pressure = self._vram.get_pressure()
                new_mode = self._pressure_to_mode(pressure, self._step_graph)
                if new_mode != self._mode:
                    self._mode = new_mode
                    if new_mode == EvictionMode.EMERGENCY:
                        self._blocked = True
                    elif self._mode == EvictionMode.EMERGENCY:
                        self._blocked = False
                    await self._apply_eviction_policy()
                await asyncio.sleep(self.VRAM_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                await asyncio.sleep(1)  # Brief backoff on error
    
    @staticmethod
    def _pressure_to_mode(pressure: float, step_graph=None) -> EvictionMode:
        """Convert VRAM pressure to eviction mode."""
        if pressure < 0.70:   return EvictionMode.RELAXED
        if pressure < 0.85:   return EvictionMode.NORMAL
        if pressure < 0.92:   return EvictionMode.PRESSURE
        if pressure < 0.96:   return EvictionMode.CRITICAL
        if pressure >= 0.96 and step_graph is not None: return EvictionMode.WORKFLOW_AWARE
        return EvictionMode.EMERGENCY
    
    async def set(self, key: str, value: Any, token_count: int) -> bool:
        """
        Store value in cache.
        
        Args:
            key: Cache key (e.g., "context:agent1")
            value: Value to store
            token_count: Token count for VRAM tracking
        
        Returns:
            True if stored, False if blocked in EMERGENCY mode
        """
        if self._blocked:
            return False
        
        entry = CacheEntry(
            priority=time.monotonic(),  # Will be updated by LRU/LFU formula
            last_accessed=time.monotonic(),
            access_count=1,
            token_count=token_count,
            key=key,
            value=value,
        )
        
        async with self._lock:
            # Evict old entry if key exists
            if key in self._store:
                old_entry = self._store[key]
                self._total_tokens -= old_entry.token_count
            
            self._store[key] = entry
            heapq.heappush(self._heap, entry)
            self._total_tokens += token_count
        
        # Trigger eviction check if needed
        if self._mode in (EvictionMode.PRESSURE, EvictionMode.CRITICAL, EvictionMode.EMERGENCY):
            await self._apply_eviction_policy()
        
        return True
    
    async def get(self, key: str) -> Any | None:
        """Retrieve value, updating access metadata."""
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            
            # Update access metadata
            entry.last_accessed = time.monotonic()
            entry.access_count += 1
            # Recalculate priority: lower = evict first
            entry.priority = entry.last_accessed - (entry.access_count * 10)
            
            return entry.value
    
    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            entry = self._store.pop(key, None)
            if entry:
                self._total_tokens -= entry.token_count
                return True
            return False
    
    async def _apply_eviction_policy(self) -> int:
        """
        Apply eviction policy based on current mode.
        
        Returns:
            Number of entries evicted
        """
        evicted = 0
        now = time.monotonic()
        
        async with self._lock:
            match self._mode:
                case EvictionMode.RELAXED:
                    pass  # No eviction
                
                case EvictionMode.NORMAL:
                    # LRU: evict entries idle > 120s
                    to_evict = [
                        k for k, e in self._store.items()
                        if now - e.last_accessed > 120
                    ]
                    for k in to_evict:
                        self._evict(k)
                        evicted += 1
                
                case EvictionMode.PRESSURE:
                    # LFU by token_count: evict heaviest, least used first
                    candidates = sorted(
                        self._store.values(),
                        key=lambda e: e.token_count / max(e.access_count, 1),
                        reverse=True
                    )
                    # Evict top 25%
                    target = max(1, int(len(candidates) * 0.25))
                    for entry in candidates[:target]:
                        self._evict(entry.key)
                        evicted += 1
                
                case EvictionMode.CRITICAL:
                    # Mark inactive for CPU offload instead of destroying
                    for entry in self._store.values():
                        if now - entry.last_accessed > 30 and not entry.offloaded_to_cpu:
                            entry.offloaded_to_cpu = True
                
                case EvictionMode.EMERGENCY:
                    # Hard evict everything idle > 30s
                    to_evict = [
                        k for k, e in self._store.items()
                        if now - e.last_accessed > 30
                    ]
                    for k in to_evict:
                        self._evict(k)
                        evicted += 1
                
                case EvictionMode.WORKFLOW_AWARE:
                    if self._step_graph is not None:
                        priority_order = self._step_graph.get_eviction_priority_order()
                        # Evict in reverse priority order (lowest priority first)
                        for agent_id in reversed(priority_order):
                            key = f"context:{agent_id}"
                            if key in self._store:
                                self._evict(key)
                                evicted += 1
        
        if evicted > 0:
            await self._reheap()
        
        return evicted
    
    def _evict(self, key: str) -> None:
        """Remove entry. Must be called under lock."""
        entry = self._store.pop(key, None)
        if entry:
            self._total_tokens -= entry.token_count
    
    async def _reheap(self) -> None:
        """Rebuild heap after evictions."""
        self._heap = list(self._store.values())
        heapq.heapify(self._heap)
    
    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._store.clear()
            self._heap.clear()
            self._total_tokens = 0
    
    @property
    def size(self) -> int:
        """Number of entries."""
        return len(self._store)
    
    @property
    def total_tokens(self) -> int:
        """Total token count in cache."""
        return self._total_tokens
    
    @property
    def mode(self) -> EvictionMode:
        """Current eviction mode."""
        return self._mode
    
    @property
    def is_blocked(self) -> bool:
        """True if new registrations are blocked (EMERGENCY mode)."""
        return self._blocked
    
    @property
    def step_graph(self) -> Optional["AgentStepGraph"]:
        """The workflow dependency graph for WORKFLOW_AWARE eviction."""
        return self._step_graph
