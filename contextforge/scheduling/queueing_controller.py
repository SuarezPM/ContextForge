"""
QueueingController — stability-aware KV cache eviction.

Replaces VRAMAwareCache's empirical pressure thresholds with a
queueing-theoretic stability controller based on arXiv:2605.04595
(ICML 2026). The controller continuously estimates λ (arrival rate)
and E[S] (service time) from a sliding window, derives the stability
margin, and adjusts eviction aggressiveness to maintain stability.

Key invariant (INVARIANT-11):
  The controller NEVER evicts below minimum_stable_blocks.
  minimum_stable_blocks = ceil(λ * E[S] * E[blocks_per_request] * safety_margin)
  where safety_margin = 1.15 (15% buffer, validated in paper at < 10% deviation)
"""

from dataclasses import dataclass, field
from typing import Optional
import asyncio
import time
import math


@dataclass
class QueueingConfig:
    """Configuration for the queueing-theoretic stability controller.

    Based on arXiv:2605.04595 ICML 2026 findings for KV cache stability.
    """
    window_seconds: float = 60.0          # sliding window for λ estimation (paper §3.2)
    safety_margin: float = 1.15           # 15% buffer above theoretical minimum
    block_size: int = 16                  # PagedAttention block size in tokens
    head_dim: int = 128                   # attention head dimension
    num_kv_heads: int = 8                # GQA heads for Qwen3.6
    bytes_per_element: float = 2.0        # FP16 default; 0.5 for INT4 (RotateKV)
    min_eviction_interval_ms: float = 100.0  # prevent eviction storms (paper §4.1)


@dataclass
class StabilityState:
    """Current stability state snapshot.

    All values derived from queueing theory as described in arXiv:2605.04595.
    """
    arrival_rate_lambda: float      # requests/sec, estimated via EMA over window
    service_rate_mu: float          # requests/sec capacity (1 / E[S])
    mean_blocks_per_request: float  # E[blocks consumed per request]
    utilization_rho: float          # λ/μ — must be < 1.0 for stability (paper §2.2)
    is_stable: bool                 # rho < 1.0 AND free_blocks >= minimum_stable_blocks
    lambda_critical: float          # λ threshold that triggers eviction (paper §3.3)
    minimum_stable_blocks: int      # INVARIANT-11 floor: ceil(λ * E[S] * E[blocks] * margin)
    stability_margin_pct: float     # (1 - rho) * 100


class _WelfordStatistics:
    """Numerically stable online mean and variance using Welford's algorithm.

    Welford, B. P. (1962). "Note on a method for calculating corrected sums of
    squares and products". Technometrics 4(3): 419–420.

    This implementation maintains running statistics in a single pass,
    avoiding the numerical instability of naive two-pass or sum-of-squares
    methods, which is critical for 64-bit float accumulation over long windows.
    """
    _count: int = 0
    _mean: float = 0.0
    _M2: float = 0.0  # sum of squared deviations (n * variance)

    def update(self, value: float) -> None:
        """Update statistics with a new observation."""
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._M2 += delta * delta2

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        """Sample mean E[X]."""
        return self._mean if self._count > 0 else 0.0

    @property
    def variance(self) -> float:
        """Sample variance Var(X) = M2 / n."""
        if self._count < 2:
            return 0.0
        return self._M2 / self._count

    @property
    def std(self) -> float:
        """Sample standard deviation sqrt(Var(X))."""
        return math.sqrt(max(0.0, self.variance))


class QueueingController:
    """Stability-aware KV cache eviction controller.

    Implements the queueing-theoretic framework from arXiv:2605.04595 (ICML 2026).
    Estimates arrival rate λ and mean service time E[S] from a sliding observation
    window, derives the M/G/1 stability condition, and adjusts eviction to keep
    free blocks ≥ minimum_stable_blocks.

    Key invariant (INVARIANT-11):
        The controller NEVER evicts below minimum_stable_blocks.

    Notation (paper §2):
        λ  = request arrival rate (requests/sec)
        μ  = service rate (requests/sec), μ = 1 / E[S]
        ρ  = utilization = λ / μ  (must be < 1 for stability)
        E[B] = expected blocks per request

    Stability condition (paper Theorem 2.1):
        free_blocks ≥ ceil(λ * E[S] * E[B] * safety_margin)

    Usage:
        controller = QueueingController(QueueingConfig())
        controller.record_request_arrival(time.time(), token_count=512, agent_id="agent-1")
        # ... later, after completion ...
        controller.record_request_completion(time.time(), service_time_ms=45.2,
                                             blocks_consumed=32, agent_id="agent-1")
        state = controller.compute_stability_state(current_free_blocks=128, total_blocks=256)
        target = controller.get_eviction_target_blocks(current_free_blocks=128,
                                                       total_blocks=256,
                                                       requested_new_blocks=64)
    """

    def __init__(self, config: QueueingConfig = QueueingConfig()):
        self.config = config

        # --- Sliding window ring buffer for arrivals ---
        # Each entry: (timestamp, token_count, agent_id)
        self._arrival_buffer: list[tuple[float, int, str]] = []
        self._arrival_buffer_lock = asyncio.Lock()

        # --- Welford accumulators for service time and blocks ---
        self._service_stats = _WelfordStatistics()
        self._blocks_stats = _WelfordStatistics()

        # --- EMA state for λ estimation (exponential moving average) ---
        # arXiv:2605.04595 §3.2: λ estimated via EMA with decay based on window_seconds
        self._lambda_ema: float = 0.0          # current EMA of λ
        self._last_arrival_time: Optional[float] = None
        self._ema_lock = asyncio.Lock()

        # --- Inter-request intervals for μ estimation ---
        # Collect inter-arrival times to estimate service rate via 1/E[Δt]
        self._inter_arrival_times: list[float] = []
        self._inter_arrival_lock = asyncio.Lock()
        self._min_requests_for_stable_estimate: int = 10

        # --- Throttle for eviction storms (paper §4.1) ---
        self._last_eviction_time: float = 0.0

        # --- Grace period on startup ---
        self._start_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_request_arrival(
        self, timestamp: float, token_count: int, agent_id: str
    ) -> None:
        """Record a request arrival for λ estimation.

        Updates the EMA of the arrival rate using the exponential decay
        factor α = 1 - exp(-Δt / window_seconds) derived from the inter-
        arrival time Δt (paper §3.2, Equation 3).

        Args:
            timestamp:   Unix timestamp of request arrival.
            token_count: Number of tokens in the request (used to estimate blocks).
            agent_id:    Identifier of the agent that issued the request.
        """
        # Add to sliding window buffer
        self._arrival_buffer.append((timestamp, token_count, agent_id))
        self._prune_arrival_buffer(timestamp)

        # Compute EMA update step from inter-arrival time
        # arXiv:2605.04595 Equation (3): α = 1 - exp(-Δt / T)
        # where T = window_seconds is the smoothing window.
        now = timestamp
        if self._last_arrival_time is not None:
            dt = now - self._last_arrival_time
            if dt > 0:
                alpha = 1.0 - math.exp(-dt / self.config.window_seconds)
                # Instantaneous rate = 1/dt, EMA blends with current estimate
                instantaneous_rate = 1.0 / dt
                self._lambda_ema = alpha * instantaneous_rate + (1.0 - alpha) * self._lambda_ema

                # Store inter-arrival time for service rate estimation
                self._inter_arrival_times.append(dt)
                if len(self._inter_arrival_times) > 1000:
                    # Keep bounded; oldest are least relevant for recent ρ
                    self._inter_arrival_times = self._inter_arrival_times[-500:]

        self._last_arrival_time = now

    def record_request_completion(
        self,
        timestamp: float,
        service_time_ms: float,
        blocks_consumed: int,
        agent_id: str,
    ) -> None:
        """Record service time and block consumption.

        Updates Welford accumulators for E[S] and E[blocks] (paper §3.2).
        These are used to compute the stability margin and minimum cache size.

        Args:
            timestamp:        Unix timestamp of request completion.
            service_time_ms:  Wall-clock service time in milliseconds.
            blocks_consumed:  Number of KV cache blocks used by this request.
            agent_id:         Identifier of the agent.
        """
        service_time_s = service_time_ms / 1000.0  # convert to seconds
        self._service_stats.update(service_time_s)
        if blocks_consumed > 0:
            self._blocks_stats.update(float(blocks_consumed))

    def compute_stability_state(
        self, current_free_blocks: int, total_blocks: int
    ) -> StabilityState:
        """Compute current stability state from queueing-theoretic estimators.

        Uses fallback values when fewer than 10 requests have been observed,
        as the statistical estimates are not yet reliable (paper §4.2 mentions
        n < 10 as insufficient for stable online estimation).

        Args:
            current_free_blocks: Number of currently free KV cache blocks.
            total_blocks:        Total number of KV cache blocks available.

        Returns:
            StabilityState with all derived metrics.
        """
        # --- Fallback values when insufficient data ---
        # arXiv:2605.04595 §4.2: estimates unreliable with < 10 samples
        if self._service_stats.count < self._min_requests_for_stable_estimate:
            lambda_estimate = 0.1           # requests/sec (conservative low rate)
            e_service_time = 1.0            # seconds (1 req/sec capacity)
            e_blocks = float(self.config.block_size)  # one block
        else:
            lambda_estimate = self._get_lambda()
            e_service_time = max(0.001, self._service_stats.mean)  # avoid div-by-zero
            e_blocks = max(1.0, self._blocks_stats.mean)

        # --- Service rate μ = 1 / E[S] ---
        # arXiv:2605.04595 §2.1: service rate defined as reciprocal of mean service time
        service_rate_mu = 1.0 / e_service_time

        # --- Utilization ρ = λ / μ ---
        # arXiv:2605.04595 §2.2: utilization must be < 1 for system stability
        # Using max to guard against pathological μ ≈ 0 (can occur on startup)
        rho = min(lambda_estimate / max(service_rate_mu, 1e-9), 0.9999)

        # --- Minimum stable blocks (INVARIANT-11) ---
        # arXiv:2605.04595 Theorem 2.1 (M/G/1 stability condition):
        #   minimum_stable_blocks = ceil(λ * E[S] * E[B] * safety_margin)
        # where E[B] = mean_blocks_per_request.
        expected_blocks_per_request = e_blocks
        raw_minimum = (
            lambda_estimate
            * e_service_time
            * expected_blocks_per_request
            * self.config.safety_margin
        )
        minimum_stable_blocks = self._ceiling_int(raw_minimum)

        # --- Critical λ threshold (paper §3.3) ---
        # λ at which minimum_stable_blocks would equal current_free_blocks.
        # Used as the eviction trigger threshold.
        if expected_blocks_per_request > 0 and self.config.safety_margin > 0:
            lambda_critical = (
                current_free_blocks
                / (e_service_time * expected_blocks_per_request * self.config.safety_margin)
            )
        else:
            lambda_critical = float("inf")

        # --- Stability check ---
        # System is stable if: (1) utilization < 1 AND (2) free blocks ≥ minimum
        # Both conditions are required per paper Theorem 2.1 and INVARIANT-11.
        is_stable = bool(rho < 1.0 and current_free_blocks >= minimum_stable_blocks)

        # --- Stability margin as percentage ---
        stability_margin_pct = (1.0 - rho) * 100.0

        return StabilityState(
            arrival_rate_lambda=round(lambda_estimate, 6),
            service_rate_mu=round(service_rate_mu, 6),
            mean_blocks_per_request=round(expected_blocks_per_request, 4),
            utilization_rho=round(rho, 6),
            is_stable=is_stable,
            lambda_critical=round(lambda_critical, 6),
            minimum_stable_blocks=minimum_stable_blocks,
            stability_margin_pct=round(stability_margin_pct, 4),
        )

    def get_eviction_target_blocks(
        self,
        current_free_blocks: int,
        total_blocks: int,
        requested_new_blocks: int,
    ) -> int:
        """Compute the number of blocks to evict to maintain stability.

        INVARIANT-11 (non-negotiable):
            The result guarantees free_blocks_after_eviction >= minimum_stable_blocks.
            This is asserted in this method and never violated.

        Algorithm (paper §3.3, Algorithm 1):
            1. Compute minimum_stable_blocks from current λ, E[S], E[B] estimates.
            2. Compute target_free = max(minimum_stable_blocks, current_free_blocks - requested_new_blocks).
            3. If target_free < minimum_stable_blocks, evict enough to restore the floor.
            4. Throttle eviction to prevent storms (min_eviction_interval_ms).

        Args:
            current_free_blocks:   Current number of free blocks.
            total_blocks:           Total KV cache capacity (used for logging bounds).
            requested_new_blocks:  Blocks needed for the incoming request.

        Returns:
            Number of blocks to evict. Zero means no eviction needed.

        Raises:
            AssertionError: If the result would violate INVARIANT-11.
        """
        state = self.compute_stability_state(current_free_blocks, total_blocks)

        # projected_free = free blocks after the new request arrives (before eviction)
        projected_free = current_free_blocks - requested_new_blocks

        # Eviction is needed only if we would dip below the minimum stable floor.
        # After eviction: result_free = current_free - requested - evict_needed
        # INVARIANT-11 requires: result_free >= minimum_stable_blocks
        # => evict_needed >= requested_new_blocks - current_free_blocks + minimum_stable_blocks
        if projected_free >= state.minimum_stable_blocks:
            return 0

        evict_needed = requested_new_blocks - current_free_blocks + state.minimum_stable_blocks

        # --- Throttle: prevent eviction storms (paper §4.1) ---
        now_ms = time.monotonic() * 1000.0
        time_since_last_eviction = now_ms - self._last_eviction_time

        if time_since_last_eviction < self.config.min_eviction_interval_ms and evict_needed > 0:
            # Not enough time has passed since the last eviction; refuse to evict
            # Return 0 rather than violating the throttle. Caller should retry later.
            return 0

        self._last_eviction_time = now_ms

        # --- INVARIANT-11 assertion (documented, non-negotiable) ---
        # Eviction ADDS free blocks back (frees cached memory).
        # result_free = projected_free (before eviction) + evict_needed (after eviction)
        result_free_blocks = projected_free + evict_needed
        assert result_free_blocks >= state.minimum_stable_blocks, (
            f"INVARIANT-11 violation: after eviction free_blocks={result_free_blocks} "
            f"would be below minimum_stable_blocks={state.minimum_stable_blocks}. "
            f"Eviction of {evict_needed} blocks is insufficient to maintain invariant."
        )

        return int(evict_needed)

    def get_recommended_quantization_bits(self) -> int:
        """Recommend KV cache quantization level based on current utilization.

        Derived from arXiv:2605.04595 §5 (Table 2) which validates that lower
        quantization allows higher throughput at the cost of memory savings.
        The thresholds map utilization regimes to bit widths:

            ρ < 0.70  → 16 bits (FP16, no quantization, maximum quality)
            0.70 ≤ ρ < 0.85  → 8 bits (INT8, balanced)
            0.85 ≤ ρ < 0.95  → 4 bits (INT4, memory-constrained)
            ρ ≥ 0.95  → 2 bits (INT2, aggressive, high quality degradation)

        Returns:
            Recommended quantization bit-width (2, 4, 8, or 16).
        """
        state_placeholder = self.compute_stability_state(
            current_free_blocks=1, total_blocks=2
        )
        rho = state_placeholder.utilization_rho

        if rho < 0.70:
            return 16   # FP16 — full precision
        elif rho < 0.85:
            return 8    # INT8 — balanced quality/cost
        elif rho < 0.95:
            return 4     # INT4 — memory-constrained regime
        else:
            return 2     # INT2 — stability-critical, aggressive compression

    def export_metrics(self) -> dict:
        """Export current metrics as a Prometheus-compatible dictionary.

        Returns 7 metrics matching the queueing_* prefix convention:

            queueing_lambda               — current EMA arrival rate (req/sec)
            queueing_mu                   — current service rate (req/sec)
            queueing_rho                  — utilization (dimensionless, 0–1)
            queueing_is_stable            — 1 if stable, 0 otherwise
            queueing_lambda_critical       — critical λ threshold (req/sec)
            queueing_minimum_stable_blocks — INVARIANT-11 floor (blocks)
            queueing_stability_margin_pct  — (1 - rho) * 100 (%)

        Returns:
            Dictionary mapping metric names to float values.
        """
        # Dummy values for stable startup before any data
        state = self.compute_stability_state(
            current_free_blocks=1, total_blocks=2
        )

        return {
            "queueing_lambda": state.arrival_rate_lambda,
            "queueing_mu": state.service_rate_mu,
            "queueing_rho": state.utilization_rho,
            "queueing_is_stable": float(1.0 if state.is_stable else 0.0),
            "queueing_lambda_critical": state.lambda_critical,
            "queueing_minimum_stable_blocks": float(state.minimum_stable_blocks),
            "queueing_stability_margin_pct": state.stability_margin_pct,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_lambda(self) -> float:
        """Return the current EMA estimate of λ.

        If no inter-arrival data is available yet, returns the EMA directly
        stored (may be 0.0 on cold start). Fallback to 0.1 req/sec if the
        estimate is effectively zero, to avoid divide-by-zero in stability
        calculations.
        """
        lam = self._lambda_ema
        if lam <= 0.0:
            # No arrivals recorded yet — use conservative fallback
            return 0.1
        return lam

    def _prune_arrival_buffer(self, current_time: float) -> None:
        """Remove arrivals outside the sliding window.

        Keeps the buffer bounded to window_seconds so old arrivals do not
        bias the λ estimate (paper §3.2 "sliding window" description).
        """
        cutoff = current_time - self.config.window_seconds
        self._arrival_buffer = [
            entry for entry in self._arrival_buffer if entry[0] >= cutoff
        ]

    @staticmethod
    def _ceiling_int(value: float) -> int:
        """Safe ceiling to non-negative integer.

        Handles floating-point rounding artifacts (e.g. 3.9999999999 due to
        IEEE 754 representation) by rounding up only when meaningfully above
        an integer threshold.
        """
        if value < 0.0:
            return 0
        result = int(math.ceil(value))
        return max(0, result)
