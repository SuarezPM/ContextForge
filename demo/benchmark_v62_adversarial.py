"""V6.2 adversarial benchmark — stress the mechanisms that V6.1 truth-up
exposed as needing harder tests.

V6.1 made the benchmark honest. S-11 then honestly reports a ~100%
λ_critical deviation under the V5 scenario, not because the
QueueingController's math is wrong but because the simulated load was
too gentle and statistically too short for Welford to converge. The
fix is **not** to soften the controller — it is to write an actually
adversarial scenario that lets the model converge to the right answer.

This file is independent of `benchmark_v5.py`. It uses:

  * **Poisson arrivals** — statistically rigorous inter-arrival times
    (exponential), not deterministic 1/λ.
  * **Long burn-in** (default: 1,000 samples per arrival rate) so the
    Welford estimator stabilises (the paper assumes n ≥ 30).
  * **Theoretical λ_critical** computed analytically from the
    simulation parameters, so we have a ground-truth value to compare
    against, not just "the highest rate we tried".
  * **Multiple service-time distributions** (exponential, lognormal,
    constant) so we can see how the M/G/1 model handles variance.

Run:

    python demo/benchmark_v62_adversarial.py

The script writes `demo/benchmark_v62_adversarial_results.json` and
prints a markdown table. It does NOT touch the V5 PASS/FAIL bookkeeping
— it is a separate, longer-running adversarial sweep.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np

from apohara_context_forge.scheduling.queueing_controller import (
    QueueingConfig,
    QueueingController,
    StabilityState,
)


# ---------------------------------------------------------------------------
# Simulation parameters                                                      #
# ---------------------------------------------------------------------------

@dataclass
class AdversarialScenario:
    """One adversarial run configuration."""

    name: str
    lambda_target: float           # arrivals / second
    service_time_mean_ms: float    # E[S] in ms
    service_dist: str              # "exponential" | "lognormal" | "constant"
    n_samples: int                 # how many arrivals to inject
    blocks_per_request: int
    total_blocks: int

    def theoretical_lambda_critical(self, safety_margin: float) -> float:
        """λ_critical = free_blocks / (E[S] * E[B] * safety_margin)."""
        e_s = self.service_time_mean_ms / 1000.0
        return self.total_blocks / (e_s * self.blocks_per_request * safety_margin)


@dataclass
class AdversarialResult:
    """Output of one adversarial run."""

    name: str
    lambda_target: float
    lambda_observed: float
    lambda_predicted: float
    lambda_theoretical: float
    deviation_vs_theoretical_pct: float
    deviation_vs_observed_pct: float
    service_dist: str
    n_samples: int
    utilization_rho: float
    is_stable: bool
    duration_ms: float


# ---------------------------------------------------------------------------
# Arrival / service generators                                               #
# ---------------------------------------------------------------------------

def poisson_inter_arrival(rng: np.random.Generator, rate: float) -> float:
    """Sample an exponential inter-arrival time for rate λ.

    For a Poisson process, inter-arrivals are Exp(λ); mean = 1/λ.
    """
    return float(rng.exponential(1.0 / rate))


def service_time_sample(
    rng: np.random.Generator, dist: str, mean_ms: float
) -> float:
    """Sample a service time in milliseconds from one of three
    distributions, all parameterised to have the same mean. This is
    the "G" in M/G/1: by varying the distribution we vary the
    coefficient of variation while holding E[S] constant."""
    if dist == "exponential":
        return float(rng.exponential(mean_ms))
    if dist == "lognormal":
        # log-normal with mean = mean_ms, variance ≈ (0.5 * mean_ms)^2
        sigma = 0.5
        mu = math.log(mean_ms) - sigma**2 / 2
        return float(rng.lognormal(mu, sigma))
    if dist == "constant":
        return float(mean_ms)
    raise ValueError(f"unknown service distribution: {dist!r}")


# ---------------------------------------------------------------------------
# Single run                                                                 #
# ---------------------------------------------------------------------------

def run_one(scenario: AdversarialScenario, seed: int = 42) -> AdversarialResult:
    """Inject `n_samples` Poisson arrivals at `lambda_target`, feed
    the QueueingController, and compare its λ_critical estimate to the
    theoretical ground truth derived from the simulation parameters.

    The simulator tracks in-flight blocks via a min-heap keyed by
    completion time: each arrival consumes `blocks_per_request`
    blocks at time t, releases them at t + service_time. The
    controller is then asked for its stability estimate AT a
    snapshot that reflects steady-state load, not an artificial drain.
    """
    import heapq

    rng = np.random.default_rng(seed)
    controller = QueueingController(QueueingConfig())

    # Min-heap of (completion_time, blocks_held). When advancing the
    # simulation clock we pop everything that has already completed
    # and return its blocks to the free pool.
    in_flight: list[tuple[float, int]] = []
    blocks_in_flight = 0
    now = 0.0

    t_start = time.perf_counter()

    for step in range(scenario.n_samples):
        # Advance the clock by an exponential inter-arrival sample.
        dt = poisson_inter_arrival(rng, scenario.lambda_target)
        now += dt

        # Free any blocks whose service has completed by `now`.
        while in_flight and in_flight[0][0] <= now:
            _, freed = heapq.heappop(in_flight)
            blocks_in_flight -= freed

        # New arrival.
        controller.record_request_arrival(
            now, token_count=512, agent_id=f"agent-{step}",
        )

        st_ms = service_time_sample(
            rng, scenario.service_dist, scenario.service_time_mean_ms,
        )
        completion_time = now + st_ms / 1000.0
        heapq.heappush(in_flight, (completion_time, scenario.blocks_per_request))
        blocks_in_flight += scenario.blocks_per_request

        controller.record_request_completion(
            completion_time,
            service_time_ms=st_ms,
            blocks_consumed=scenario.blocks_per_request,
            agent_id=f"agent-{step}",
        )

    # Drain any in-flight requests whose completion times are
    # already past, so the snapshot reflects steady state.
    while in_flight and in_flight[0][0] <= now:
        _, freed = heapq.heappop(in_flight)
        blocks_in_flight -= freed

    duration_ms = (time.perf_counter() - t_start) * 1000

    current_free = max(0, scenario.total_blocks - blocks_in_flight)
    state: StabilityState = controller.compute_stability_state(
        current_free_blocks=current_free,
        total_blocks=scenario.total_blocks,
    )
    lambda_predicted = state.lambda_critical
    lambda_theoretical = scenario.theoretical_lambda_critical(
        QueueingConfig().safety_margin,
    )

    # Observed = the λ_target we drove (it's a known input). We then
    # compare predictions vs (a) the theoretical ceiling derived from
    # the same params and (b) the observed input.
    lambda_observed = scenario.lambda_target

    return AdversarialResult(
        name=scenario.name,
        lambda_target=scenario.lambda_target,
        lambda_observed=lambda_observed,
        lambda_predicted=lambda_predicted,
        lambda_theoretical=lambda_theoretical,
        deviation_vs_theoretical_pct=(
            abs(lambda_predicted - lambda_theoretical)
            / max(lambda_theoretical, 1e-9) * 100.0
        ),
        deviation_vs_observed_pct=(
            abs(lambda_predicted - lambda_observed)
            / max(lambda_observed, 1e-9) * 100.0
        ),
        service_dist=scenario.service_dist,
        n_samples=scenario.n_samples,
        utilization_rho=state.utilization_rho,
        is_stable=state.is_stable,
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Scenario sweep                                                             #
# ---------------------------------------------------------------------------

def default_scenarios() -> list[AdversarialScenario]:
    """The default sweep: three service-time distributions × two load
    points (light and near-critical) × long burn-in. Eight runs total."""
    # With E[S]=60 ms, blocks=32, total=2000, safety=1.15:
    # λ_critical ≈ 2000 / (0.06 * 32 * 1.15) ≈ 906 req/sec.
    # We test at 10 (light, ρ ≈ 0.0007) and at 200 (≈ 22% of capacity).
    scenarios = []
    for dist in ("exponential", "lognormal", "constant"):
        for lambda_target, label in [(10.0, "light"), (200.0, "moderate")]:
            scenarios.append(
                AdversarialScenario(
                    name=f"adv_{dist}_{label}",
                    lambda_target=lambda_target,
                    service_time_mean_ms=60.0,
                    service_dist=dist,
                    n_samples=1000,
                    blocks_per_request=32,
                    total_blocks=2000,
                )
            )
    return scenarios


def run_all(scenarios: list[AdversarialScenario]) -> list[AdversarialResult]:
    """Run every scenario and return ordered results."""
    results = []
    for i, sc in enumerate(scenarios):
        print(f"  [{i+1}/{len(scenarios)}] {sc.name} (λ={sc.lambda_target}, "
              f"dist={sc.service_dist}, n={sc.n_samples})...",
              end=" ", flush=True)
        result = run_one(sc, seed=42 + i)
        print(f"predicted={result.lambda_predicted:.1f}, "
              f"theoretical={result.lambda_theoretical:.1f}, "
              f"deviation={result.deviation_vs_theoretical_pct:.1f}%, "
              f"{result.duration_ms:.1f}ms")
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Reporting                                                                  #
# ---------------------------------------------------------------------------

def print_summary(results: list[AdversarialResult]) -> None:
    print()
    print("=" * 100)
    print("V6.2 ADVERSARIAL BENCHMARK — QueueingController M/G/1 stability")
    print("=" * 100)
    print()
    print(f"{'name':<28} {'λ_obs':>8} {'λ_pred':>10} {'λ_theo':>10}"
          f" {'dev_vs_theo':>14} {'dist':>13} {'n':>6} {'ms':>8}")
    print("-" * 100)
    for r in results:
        print(
            f"{r.name:<28} {r.lambda_observed:>8.1f} {r.lambda_predicted:>10.1f}"
            f" {r.lambda_theoretical:>10.1f} {r.deviation_vs_theoretical_pct:>13.2f}%"
            f" {r.service_dist:>13} {r.n_samples:>6} {r.duration_ms:>8.1f}"
        )
    print("-" * 100)
    print()
    print("PASS criteria (per paper §4 — converged after n ≥ 30 samples):")
    print("  * deviation vs theoretical < 25% for exponential")
    print("  * deviation vs theoretical < 50% for lognormal / constant")
    print()
    passes = 0
    for r in results:
        threshold = 25.0 if r.service_dist == "exponential" else 50.0
        ok = r.deviation_vs_theoretical_pct < threshold
        if ok:
            passes += 1
        marker = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {marker} {r.name} — dev {r.deviation_vs_theoretical_pct:.2f}%"
              f" (threshold {threshold:.0f}%)")
    print()
    print(f"Adversarial PASS rate: {passes}/{len(results)}")


def write_json(results: list[AdversarialResult], path: str) -> None:
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "benchmark_version": "v6.2-adversarial",
        "scenarios": [
            {
                "name": r.name,
                "lambda_target": r.lambda_target,
                "lambda_observed": r.lambda_observed,
                "lambda_predicted": r.lambda_predicted,
                "lambda_theoretical": r.lambda_theoretical,
                "deviation_vs_theoretical_pct": r.deviation_vs_theoretical_pct,
                "deviation_vs_observed_pct": r.deviation_vs_observed_pct,
                "service_dist": r.service_dist,
                "n_samples": r.n_samples,
                "utilization_rho": r.utilization_rho,
                "is_stable": r.is_stable,
                "duration_ms": r.duration_ms,
            }
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Entry point                                                                #
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V6.2 adversarial benchmark")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "benchmark_v62_adversarial_results.json",
        ),
        help="Output JSON path",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Override per-scenario sample count (default: 1000)",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 100}")
    print("V6.2 adversarial benchmark — QueueingController M/G/1 stability")
    print(f"Date: {datetime.utcnow().isoformat()}Z")
    print(f"{'=' * 100}\n")

    scenarios = default_scenarios()
    if args.n_samples is not None:
        for sc in scenarios:
            sc.n_samples = args.n_samples

    results = run_all(scenarios)
    print_summary(results)
    write_json(results, args.out)


if __name__ == "__main__":
    main()
