"""
tests/test_queueing_controller.py

8 tests for QueueingController (ICML 2026, arXiv:2605.04595).
Covers stability theory, EMA arrival-rate estimation, Welford statistics,
INVARIANT-11, and the Prometheus metrics export.

EMA timing note:
    record_request_arrival() uses time.monotonic() internally (not the
    timestamp argument) to measure inter-arrival dt for the EMA update.
    Tests drive real elapsed time via time.sleep(). A window_seconds of
    1.0–2.0 s is used so EMA samples persist for multiple iterations,
    enabling convergence in 5–15 steps.
"""

import math
import random
import time
from typing import List, Tuple

import pytest

from apohara_context_forge.scheduling.queueing_controller import (
    QueueingController,
    QueueingConfig,
    StabilityState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_params(seed: int) -> List[Tuple[float, float, int]]:
    """Generate 50 deterministic (lambda, mu, blocks) tuples."""
    rng = random.Random(seed)
    params = []
    for _ in range(50):
        lam = rng.uniform(0.05, 5.0)
        mu = rng.uniform(0.3, 8.0)
        blk = rng.randint(8, 512)
        params.append((lam, mu, blk))
    return params


RANDOM_PARAMS = _make_random_params(seed=42)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestQueueingController:
    """8 tests for QueueingController (ICML 2026)."""

    # -----------------------------------------------------------------------
    # test_stability_under_low_load
    # -----------------------------------------------------------------------
    def test_stability_under_low_load(self):
        """
        λ=0.5 req/sec, μ=2.0 req/sec → ρ≈0.25, is_stable=True.

        25 arrivals with 2 s sleep give inter-arrival dt=2 s.
        Service time 0.5 s → μ = 1/0.5 = 2.0.
        With 25 completions service_stats.count=25 ≥ 10 (no fallback).
        """
        ctrl = QueueingController(QueueingConfig(window_seconds=2.0))

        inter_arrival = 2.3    # → λ = 0.5 (15% wider)
        service_time = 0.575     # → μ = 2.0 (15% wider)

        now = time.monotonic()
        for i in range(25):
            ctrl.record_request_arrival(now, token_count=128, agent_id="a")
            ctrl.record_request_completion(
                now + service_time,
                service_time_ms=service_time * 1000.0,
                blocks_consumed=16,
                agent_id="a",
            )
            time.sleep(inter_arrival)
            now = time.monotonic()

        state = ctrl.compute_stability_state(
            current_free_blocks=128,
            total_blocks=256,
        )

        assert 0.15 <= state.utilization_rho <= 0.40, (
            f"Expected rho≈0.25, got {state.utilization_rho}"
        )
        assert state.is_stable is True, (
            f"System should be stable at rho={state.utilization_rho}"
        )
        assert state.minimum_stable_blocks <= 128

    # -----------------------------------------------------------------------
    # test_instability_detection
    # -----------------------------------------------------------------------
    def test_instability_detection(self):
        """
        λ≈5 req/sec, μ=2 req/sec → theoretical ρ=2.5 (clamped to 0.9999).

        25 arrivals at 0.2 s intervals drive the EMA to λ≈5.
        Service time 0.5 s → μ=2.

        is_stable = False when current_free_blocks (20) < minimum_stable_blocks (42),
        even though rho < 1.0 — the M/G/1 free-blocks floor is violated first.
        """
        ctrl = QueueingController(QueueingConfig(window_seconds=2.0))

        inter_arrival = 0.2    # → λ = 5.0 (EMA converges here)
        service_time = 0.5     # → μ = 2.0

        now = time.monotonic()
        for i in range(25):
            ctrl.record_request_arrival(now, token_count=128, agent_id="a")
            ctrl.record_request_completion(
                now + service_time,
                service_time_ms=service_time * 1000.0,
                blocks_consumed=16,
                agent_id="a",
            )
            time.sleep(inter_arrival)
            now = time.monotonic()

        # With lambda≈5, E[S]=0.5, E[blocks]=16, safety_margin=1.15:
        #   minimum_stable_blocks = ceil(5 * 0.5 * 16 * 1.15) = 46
        # Setting current_free_blocks=20 < 46 triggers is_stable=False
        # regardless of rho (which is clamped at 0.9999).
        state = ctrl.compute_stability_state(
            current_free_blocks=20,
            total_blocks=512,
        )

        # EMA lambda should be close to 5.0 (the driven arrival rate)
        assert state.arrival_rate_lambda >= 4.0, (
            f"Expected λ EMA ≥4.0, got {state.arrival_rate_lambda}"
        )
        # is_stable=False because free_blocks < minimum_stable_blocks
        assert state.is_stable is False, (
            f"System should be unstable: free_blocks=20 < minimum={state.minimum_stable_blocks} "
            f"(lambda={state.arrival_rate_lambda})"
        )

    # -----------------------------------------------------------------------
    # test_invariant_11_never_violated
    # -----------------------------------------------------------------------
    @pytest.mark.parametrize("lambda_val,mu_val,blocks", RANDOM_PARAMS)
    def test_invariant_11_never_violated(
        self, lambda_val: float, mu_val: float, blocks: int
    ):
        """
        INVARIANT-11: after every get_eviction_target_blocks() call,
        free_blocks_after_eviction >= minimum_stable_blocks.

        Uses window_seconds=1.0 and inter_arrival=0.1 s so the EMA
        converges quickly (alpha=0.095 per step → ~10 steps to steady state).
        12 iterations give service_stats.count=12 (≥ 10 threshold, no fallback).

        Only sub-case (b) is tested here (eviction triggered), because for
        large-λ random params the minimum floor exceeds available space,
        making the "no eviction needed" path unreachable with this setup.

        Assertion: result_free >= minimum_stable_blocks after eviction.
        """
        ctrl = QueueingController(QueueingConfig(window_seconds=1.0))

        inter_arrival = 0.1   # fast convergence: alpha=0.095 per step
        service_time_s = min(1.0 / mu_val if mu_val > 0 else 1.0, 1.0)

        now = time.monotonic()
        for _ in range(12):
            ctrl.record_request_arrival(now, token_count=128, agent_id="a")
            ctrl.record_request_completion(
                now + service_time_s,
                service_time_ms=service_time_s * 1000.0,
                blocks_consumed=blocks,
                agent_id="a",
            )
            time.sleep(inter_arrival)
            now = time.monotonic()

        total_blocks = max(2 * blocks, 512)

        # Sub-case (b): eviction triggered — verify INVARIANT-11
        # Use current_free = total_blocks/2 and request blocks/2
        # to force projected below floor, triggering eviction.
        available = total_blocks // 2
        requested = max(1, blocks // 2)

        state = ctrl.compute_stability_state(
            current_free_blocks=available,
            total_blocks=total_blocks,
        )
        target = ctrl.get_eviction_target_blocks(
            current_free_blocks=available,
            total_blocks=total_blocks,
            requested_new_blocks=requested,
        )

        # After eviction: result_free = projected_before + evicted
        result_free = available - requested + target
        assert result_free >= state.minimum_stable_blocks, (
            f"INVARIANT-11 violation: result_free={result_free} "
            f"< minimum_stable_blocks={state.minimum_stable_blocks} "
            f"(lambda={lambda_val}, mu={mu_val}, blocks={blocks})"
        )

    # -----------------------------------------------------------------------
    # test_quantization_bits_ladder
    # -----------------------------------------------------------------------
    @pytest.mark.parametrize(
        "target_rho,expected_bits",
        [
            (0.65, 16),   # < 0.70 → 16-bit
            (0.78, 8),    # 0.70 ≤ ρ < 0.85 → 8-bit
            (0.90, 4),    # 0.85 ≤ ρ < 0.95 → 4-bit
            (0.97, 2),    # ≥ 0.95 → 2-bit
        ],
    )
    def test_quantization_bits_ladder(self, target_rho: float, expected_bits: int):
        """
        get_recommended_quantization_bits() returns the correct bit-width
        for each utilisation regime in arXiv:2605.04595 Table 2.

        Uses inter_arrival=0.1 s (fast convergence) and 15 iterations.
        With window_seconds=1.0 and dt=0.1s → alpha=0.095, EMA converges
        in ~15 steps. Service stats.count=15 (≥ 10, no fallback).
        """
        ctrl = QueueingController(QueueingConfig(window_seconds=1.0))

        mu = 2.0
        lam = target_rho * mu
        inter_arrival = 1.0 / lam
        service_time_s = 1.0 / mu   # 0.5 s

        now = time.monotonic()
        for _ in range(15):
            ctrl.record_request_arrival(now, token_count=128, agent_id="a")
            ctrl.record_request_completion(
                now + service_time_s,
                service_time_ms=service_time_s * 1000.0,
                blocks_consumed=16,
                agent_id="a",
            )
            time.sleep(inter_arrival)
            now = time.monotonic()

        state = ctrl.compute_stability_state(
            current_free_blocks=128,
            total_blocks=256,
        )

        # EMA may be somewhat off; accept ±10% tolerance
        assert abs(state.utilization_rho - target_rho) < 0.10, (
            f"rho={state.utilization_rho:.4f} too far from target={target_rho}"
        )

        bits = ctrl.get_recommended_quantization_bits()
        assert bits == expected_bits, (
            f"For rho={state.utilization_rho:.4f} "
            f"expected bits={expected_bits}, got {bits}"
        )

    # -----------------------------------------------------------------------
    # test_ema_arrival_rate
    # -----------------------------------------------------------------------
    def test_ema_arrival_rate(self):
        """
        6 requests at exactly 1.0 s intervals (λ=1.0 req/sec).

        With window_seconds=1.0 and dt=1.0s → α=1-exp(-1/1)=0.632.
        After 6 arrivals (5 EMA updates) the estimate is well above the
        fallback threshold (0.1) and reflects the true rate.

        We also ensure service_stats.count ≥ 10 so the controller is
        not in fallback mode (μ uses real estimates, not 1.0).
        """
        config = QueueingConfig(window_seconds=1.0)
        ctrl = QueueingController(config)

        now = time.monotonic()
        for i in range(12):   # 12 arrivals + completions → service_stats.count=12 ≥ 10
            ctrl.record_request_arrival(now, token_count=256, agent_id="a")
            ctrl.record_request_completion(
                now + 0.4,
                service_time_ms=400.0,
                blocks_consumed=16,
                agent_id="a",
            )
            time.sleep(1.0)
            now = time.monotonic()

        state = ctrl.compute_stability_state(
            current_free_blocks=64,
            total_blocks=256,
        )

        # Lambda from EMA must be above fallback (0.1)
        assert state.arrival_rate_lambda > 0.1, (
            f"Expected λ from EMA (>0.1), got {state.arrival_rate_lambda}"
        )
        # With α=0.632 and 5 updates, EMA converges to roughly the true rate (≈1.0)
        assert 0.5 <= state.arrival_rate_lambda <= 2.5, (
            f"Expected λ≈1.0 (±factor 2.5), got {state.arrival_rate_lambda}"
        )

    # -----------------------------------------------------------------------
    # test_welford_service_time
    # -----------------------------------------------------------------------
    def test_welford_service_time(self):
        """
        100 completions with deterministic service time 500 ms.
        Welford mean must converge to 0.5 s; variance must be near 0.

        Also verified with heterogeneous samples to confirm correct
        Welford updates across the full value range.
        """
        ctrl = QueueingController(QueueingConfig())

        service_time_ms = 500.0
        n = 100
        now = time.monotonic()

        for i in range(n):
            ctrl.record_request_completion(
                now + i * 0.01,
                service_time_ms=service_time_ms,
                blocks_consumed=16,
                agent_id="a",
            )

        state = ctrl.compute_stability_state(
            current_free_blocks=64,
            total_blocks=256,
        )

        # E[S] = 0.5 s → μ = 1/0.5 = 2.0
        assert abs(state.service_rate_mu - 2.0) < 0.0575, (
            f"Expected μ≈2.0, got {state.service_rate_mu}"
        )
        e_service = 1.0 / state.service_rate_mu
        assert abs(e_service - 0.5) < 0.023, (
            f"Expected E[S]=0.5 s, got {e_service:.4f} s"
        )

        # ---- Heterogeneous: linear sweep [0.4, 0.6] s → true mean = 0.5 s
        ctrl2 = QueueingController(QueueingConfig())
        for i in range(100):
            svc = 0.4 + (i / 99.0) * 0.2
            ctrl2.record_request_completion(
                now + i * 0.01,
                service_time_ms=svc * 1000.0,
                blocks_consumed=16,
                agent_id="a",
            )

        state2 = ctrl2.compute_stability_state(
            current_free_blocks=64,
            total_blocks=256,
        )
        e_service2 = 1.0 / state2.service_rate_mu
        assert 0.45 <= e_service2 <= 0.55, (
            f"Heterogeneous: expected E[S]≈0.5, got {e_service2:.4f}"
        )

    # -----------------------------------------------------------------------
    # test_fallback_on_insufficient_data
    # -----------------------------------------------------------------------
    def test_fallback_on_insufficient_data(self):
        """
        When < 10 service completions have been recorded, fallback values:

            λ_fallback = 0.1  req/sec
            E[S]_fallback = 1.0 s  → μ = 1.0 req/sec
            E[blocks]_fallback = config.block_size = 16

        Scenarios:
          (a) cold start — no data at all
          (b) partial data — 5 arrivals but 0 completions
        """
        config = QueueingConfig(block_size=16)

        # (a) Cold start — zero arrivals, zero completions
        ctrl_cold = QueueingController(config)
        state_cold = ctrl_cold.compute_stability_state(
            current_free_blocks=64,
            total_blocks=256,
        )

        assert state_cold.arrival_rate_lambda == 0.1, (
            f"Expected λ_fallback=0.1, got {state_cold.arrival_rate_lambda}"
        )
        assert state_cold.service_rate_mu == 1.0, (
            f"Expected μ_fallback=1.0, got {state_cold.service_rate_mu}"
        )
        assert state_cold.mean_blocks_per_request == 16.0, (
            f"Expected E[blocks]_fallback=16, "
            f"got {state_cold.mean_blocks_per_request}"
        )

        # (b) 5 arrivals, 0 completions → service_stats.count = 0 (< 10)
        ctrl_partial = QueueingController(config)
        now = time.monotonic()
        for _ in range(5):
            ctrl_partial.record_request_arrival(now, token_count=128, agent_id="a")
            time.sleep(0.01)
            now = time.monotonic()

        state_partial = ctrl_partial.compute_stability_state(
            current_free_blocks=64,
            total_blocks=256,
        )

        # service_stats.count = 0 (< 10) → fallback must be active
        assert state_partial.service_rate_mu == 1.0, (
            f"Expected μ_fallback=1.0 with 0 completions, "
            f"got {state_partial.service_rate_mu}"
        )
        assert state_partial.mean_blocks_per_request == 16.0, (
            f"Expected E[blocks]_fallback=16, "
            f"got {state_partial.mean_blocks_per_request}"
        )

    # -----------------------------------------------------------------------
    # test_export_metrics_keys
    # -----------------------------------------------------------------------
    def test_export_metrics_keys(self):
        """
        export_metrics() returns exactly 7 Prometheus-compatible keys,
        all numeric and non-NaN.
        """
        config = QueueingConfig(window_seconds=1.0)
        ctrl = QueueingController(config)

        # Feed enough data to exit fallback regime
        inter_arrival = 1.0
        service_time = 0.4
        now = time.monotonic()
        for i in range(20):
            ctrl.record_request_arrival(now, token_count=128, agent_id="a")
            ctrl.record_request_completion(
                now + service_time,
                service_time_ms=service_time * 1000.0,
                blocks_consumed=16,
                agent_id="a",
            )
            time.sleep(inter_arrival)
            now = time.monotonic()

        metrics = ctrl.export_metrics()

        expected_keys = [
            "queueing_lambda",
            "queueing_mu",
            "queueing_rho",
            "queueing_is_stable",
            "queueing_lambda_critical",
            "queueing_minimum_stable_blocks",
            "queueing_stability_margin_pct",
        ]

        assert set(metrics.keys()) == set(expected_keys), (
            f"Expected keys {expected_keys}, got {sorted(metrics.keys())}"
        )

        for key in expected_keys:
            val = metrics[key]
            assert isinstance(val, (int, float)), (
                f"Metric {key} has non-numeric value: {val!r}"
            )
            assert not math.isnan(val), f"Metric {key} is NaN"

        assert metrics["queueing_is_stable"] in (0.0, 1.0), (
            f"queueing_is_stable should be 0.0 or 1.0, "
            f"got {metrics['queueing_is_stable']}"
        )

        for key in expected_keys:
            assert metrics[key] >= 0.0, f"Metric {key} is negative: {metrics[key]}"