## 2024-05-25 - Bolt: Concurrent asyncio operations for multi-agent workloads
**Learning:** Using `asyncio.gather` inside a tight loop querying context for many agents allows execution to complete concurrently in bounded O(1) latency rather than O(n) sequentially. For singletons, they need to be eagerly initialized prior to the gather to avoid concurrent initialization race conditions.
**Action:** Identify sequence loops of independent async queries/lookups, and replace them with a gathered map-reduce operation, ensuring singletons are pre-warmed safely.
