"""Benchmarking utilities for performance optimizations.

Provides tools to measure the effectiveness of various optimizations
and compare configurations.

Usage:
    from alpha_zero.optimizations.benchmarks import OptimizationBenchmark

    benchmark = OptimizationBenchmark()
    result = benchmark.run_playout_cap_benchmark(config, num_iterations=1000)
    print(result.summary())
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alpha_zero.optimizations.auxiliary_targets import (
    AuxiliaryLossCalculator,
    AuxiliaryTargetGenerator,
    AuxiliaryTargets,
    AuxiliaryTargetsConfig,
)
from alpha_zero.optimizations.cyclic_lr import CyclicLRConfig, CyclicLRScheduler
from alpha_zero.optimizations.playout_cap import PlayoutCapConfig, PlayoutCapRandomizer
from alpha_zero.optimizations.policy_pruning import PolicyPruningConfig, PolicyTargetPruner

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run.

    Attributes:
        name: Name of the benchmark
        iterations: Number of iterations run
        total_time_seconds: Total time taken
        mean_time_ms: Mean time per iteration in milliseconds
        std_time_ms: Standard deviation of time in milliseconds
        min_time_ms: Minimum time per iteration
        max_time_ms: Maximum time per iteration
        throughput_per_second: Operations per second
        additional_metrics: Additional benchmark-specific metrics
    """

    name: str
    iterations: int
    total_time_seconds: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_per_second: float
    additional_metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Get human-readable summary of results.

        Returns:
            Summary string
        """
        lines = [
            f"Benchmark: {self.name}",
            "-" * 40,
            f"Iterations: {self.iterations:,}",
            f"Total time: {self.total_time_seconds:.3f}s",
            f"Mean time: {self.mean_time_ms:.3f}ms ± {self.std_time_ms:.3f}ms",
            f"Range: [{self.min_time_ms:.3f}ms, {self.max_time_ms:.3f}ms]",
            f"Throughput: {self.throughput_per_second:,.1f} ops/sec",
        ]

        if self.additional_metrics:
            lines.append("")
            lines.append("Additional Metrics:")
            for key, value in self.additional_metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_seconds": self.total_time_seconds,
            "mean_time_ms": self.mean_time_ms,
            "std_time_ms": self.std_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "throughput_per_second": self.throughput_per_second,
            "additional_metrics": self.additional_metrics,
        }


@dataclass
class ComparisonResult:
    """Result from comparing two benchmark configurations.

    Attributes:
        baseline: Baseline benchmark result
        optimized: Optimized benchmark result
        speedup_factor: Speedup ratio (optimized / baseline throughput)
        time_reduction_percent: Percentage reduction in time
        additional_comparisons: Additional comparison metrics
    """

    baseline: BenchmarkResult
    optimized: BenchmarkResult
    speedup_factor: float
    time_reduction_percent: float
    additional_comparisons: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Get human-readable comparison summary.

        Returns:
            Summary string
        """
        lines = [
            "Benchmark Comparison",
            "=" * 40,
            "",
            "BASELINE:",
            self.baseline.summary(),
            "",
            "OPTIMIZED:",
            self.optimized.summary(),
            "",
            "COMPARISON:",
            "-" * 40,
            f"Speedup: {self.speedup_factor:.2f}x",
            f"Time reduction: {self.time_reduction_percent:.1f}%",
        ]

        if self.additional_comparisons:
            lines.append("")
            for key, value in self.additional_comparisons.items():
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.4f}")
                else:
                    lines.append(f"{key}: {value}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "baseline": self.baseline.to_dict(),
            "optimized": self.optimized.to_dict(),
            "speedup_factor": self.speedup_factor,
            "time_reduction_percent": self.time_reduction_percent,
            "additional_comparisons": self.additional_comparisons,
        }


class OptimizationBenchmark:
    """Benchmark suite for performance optimizations.

    Provides methods to benchmark each optimization component
    and compare different configurations.

    Example:
        >>> benchmark = OptimizationBenchmark(seed=42)
        >>> result = benchmark.run_playout_cap_benchmark(
        ...     PlayoutCapConfig(min_playouts=50),
        ...     num_iterations=1000
        ... )
        >>> print(result.summary())
    """

    def __init__(
        self,
        seed: int | None = None,
        warmup_iterations: int = 100,
    ) -> None:
        """Initialize benchmark suite.

        Args:
            seed: Random seed for reproducibility
            warmup_iterations: Number of warmup iterations before timing
        """
        self._rng = np.random.default_rng(seed)
        self._warmup_iterations = warmup_iterations
        self._results: list[BenchmarkResult] = []

    def _time_function(
        self,
        func: Callable[[], Any],
        num_iterations: int,
    ) -> tuple[list[float], list[Any]]:
        """Time a function over multiple iterations.

        Args:
            func: Function to benchmark
            num_iterations: Number of iterations

        Returns:
            Tuple of (times_ms, results)
        """
        # Warmup
        for _ in range(self._warmup_iterations):
            func()

        # Timed runs
        times: list[float] = []
        results: list[Any] = []

        for _ in range(num_iterations):
            start = time.perf_counter()
            result = func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
            results.append(result)

        return times, results

    def _create_benchmark_result(
        self,
        name: str,
        times: list[float],
        additional_metrics: dict[str, Any] | None = None,
    ) -> BenchmarkResult:
        """Create BenchmarkResult from timing data.

        Args:
            name: Benchmark name
            times: List of times in milliseconds
            additional_metrics: Extra metrics to include

        Returns:
            BenchmarkResult instance
        """
        times_array = np.array(times)
        total_time = times_array.sum() / 1000  # Convert to seconds

        return BenchmarkResult(
            name=name,
            iterations=len(times),
            total_time_seconds=total_time,
            mean_time_ms=float(times_array.mean()),
            std_time_ms=float(times_array.std()),
            min_time_ms=float(times_array.min()),
            max_time_ms=float(times_array.max()),
            throughput_per_second=len(times) / total_time if total_time > 0 else 0,
            additional_metrics=additional_metrics or {},
        )

    def run_playout_cap_benchmark(
        self,
        config: PlayoutCapConfig | None = None,
        num_iterations: int = 1000,
    ) -> BenchmarkResult:
        """Benchmark playout cap randomization.

        Args:
            config: Playout cap configuration (uses defaults if None)
            num_iterations: Number of iterations

        Returns:
            BenchmarkResult with timing and distribution statistics
        """
        config = config or PlayoutCapConfig()
        randomizer = PlayoutCapRandomizer(config, rng=self._rng)

        # Create sample observations
        observations = [
            {"action_mask": self._rng.random(config.max_playouts) > 0.5}
            for _ in range(num_iterations)
        ]

        idx = [0]

        def sample_cap() -> int:
            obs = observations[idx[0] % len(observations)]
            turn = idx[0] % 100
            idx[0] += 1
            return randomizer.sample_playout_cap(observation=obs, turn=turn)

        times, caps = self._time_function(sample_cap, num_iterations)

        # Additional metrics
        caps_array = np.array(caps)
        stats = randomizer.get_stats()

        # Compute effective speedup: max_playouts / average_cap
        effective_speedup = (
            config.max_playouts / stats.average_cap if stats.average_cap > 0 else 1.0
        )

        additional = {
            "mean_cap": float(caps_array.mean()),
            "std_cap": float(caps_array.std()),
            "min_cap": int(caps_array.min()),
            "max_cap": int(caps_array.max()),
            "distribution": config.cap_distribution,
            "effective_speedup": effective_speedup,
        }

        result = self._create_benchmark_result(
            name=f"PlayoutCap ({config.cap_distribution})",
            times=times,
            additional_metrics=additional,
        )
        self._results.append(result)
        return result

    def run_policy_pruning_benchmark(
        self,
        config: PolicyPruningConfig | None = None,
        action_space_size: int = 1000,
        num_iterations: int = 1000,
    ) -> BenchmarkResult:
        """Benchmark policy target pruning.

        Args:
            config: Policy pruning configuration
            action_space_size: Size of action space
            num_iterations: Number of iterations

        Returns:
            BenchmarkResult with timing and pruning statistics
        """
        config = config or PolicyPruningConfig()
        pruner = PolicyTargetPruner(config)

        # Create sample policies (Dirichlet-distributed for realism)
        policies = []
        masks = []
        for _ in range(num_iterations):
            alpha = self._rng.random(action_space_size) + 0.1
            policy = self._rng.dirichlet(alpha)
            mask = self._rng.random(action_space_size) > 0.3
            policies.append(policy)
            masks.append(mask)

        idx = [0]

        def prune_policy() -> np.ndarray:
            i = idx[0] % len(policies)
            idx[0] += 1
            return pruner.prune(policies[i], masks[i])

        times, pruned = self._time_function(prune_policy, num_iterations)

        # Calculate action reduction
        original_actions = [m.sum() for m in masks]
        pruned_actions = [(p > 0).sum() for p in pruned]

        # Get KL divergence from recorded stats
        stats = pruner.get_stats()

        additional = {
            "mean_original_actions": float(np.mean(original_actions)),
            "mean_pruned_actions": float(np.mean(pruned_actions)),
            "action_reduction_percent": float(
                100 * (1 - np.mean(pruned_actions) / np.mean(original_actions))
            ),
            "mean_kl_divergence": stats.avg_kl_divergence,
        }

        result = self._create_benchmark_result(
            name="PolicyPruning",
            times=times,
            additional_metrics=additional,
        )
        self._results.append(result)
        return result

    def run_cyclic_lr_benchmark(
        self,
        config: CyclicLRConfig | None = None,
        num_iterations: int = 10000,
    ) -> BenchmarkResult:
        """Benchmark cyclic learning rate computation.

        Args:
            config: Cyclic LR configuration
            num_iterations: Number of step lookups

        Returns:
            BenchmarkResult with timing and schedule statistics
        """
        config = config or CyclicLRConfig()
        scheduler = CyclicLRScheduler(config)

        steps = list(range(num_iterations))
        idx = [0]

        def get_lr() -> float:
            step = steps[idx[0]]
            idx[0] += 1
            return scheduler.get_lr(step)

        times, lrs = self._time_function(get_lr, num_iterations)

        lrs_array = np.array(lrs)
        additional = {
            "schedule_type": config.schedule_type,
            "mean_lr": float(lrs_array.mean()),
            "min_lr": float(lrs_array.min()),
            "max_lr": float(lrs_array.max()),
            "lr_std": float(lrs_array.std()),
        }

        result = self._create_benchmark_result(
            name=f"CyclicLR ({config.schedule_type})",
            times=times,
            additional_metrics=additional,
        )
        self._results.append(result)
        return result

    def run_auxiliary_targets_benchmark(
        self,
        config: AuxiliaryTargetsConfig | None = None,
        board_size: int = 16,
        action_space_size: int = 1000,
        num_iterations: int = 500,
    ) -> BenchmarkResult:
        """Benchmark auxiliary target generation.

        Args:
            config: Auxiliary targets configuration
            board_size: Size of game board
            action_space_size: Size of action space
            num_iterations: Number of iterations

        Returns:
            BenchmarkResult with timing statistics
        """
        config = config or AuxiliaryTargetsConfig()
        generator = AuxiliaryTargetGenerator(
            config, board_size, action_space_size, seed=self._rng.integers(0, 2**31)
        )

        # Create sample game states
        game_states = []
        for _ in range(num_iterations):
            state = {
                "board": self._rng.random((board_size, board_size)),
                "turn": int(self._rng.integers(0, 100)),
                "units": [
                    {
                        "position": (
                            int(self._rng.integers(0, board_size)),
                            int(self._rng.integers(0, board_size)),
                        )
                    }
                    for _ in range(5)
                ],
                "cities": [
                    {
                        "position": (
                            int(self._rng.integers(0, board_size)),
                            int(self._rng.integers(0, board_size)),
                        )
                    }
                    for _ in range(2)
                ],
                "action_mask": self._rng.random(action_space_size) > 0.5,
            }
            game_states.append(state)

        idx = [0]

        def generate_targets() -> AuxiliaryTargets:
            state = game_states[idx[0] % len(game_states)]
            idx[0] += 1
            return generator.generate_targets(state)

        times, _ = self._time_function(generate_targets, num_iterations)

        enabled = []
        if config.enable_attention_targets:
            enabled.append("attention")
        if config.enable_opponent_modeling:
            enabled.append("opponent")
        if config.enable_phase_prediction:
            enabled.append("phase")
        if config.enable_future_value:
            enabled.append("future_value")

        additional = {
            "enabled_targets": enabled,
            "total_weight": config.get_total_weight(),
            "board_size": board_size,
        }

        result = self._create_benchmark_result(
            name="AuxiliaryTargets",
            times=times,
            additional_metrics=additional,
        )
        self._results.append(result)
        return result

    def run_auxiliary_loss_benchmark(
        self,
        config: AuxiliaryTargetsConfig | None = None,
        board_size: int = 16,
        action_space_size: int = 1000,
        num_iterations: int = 1000,
    ) -> BenchmarkResult:
        """Benchmark auxiliary loss computation.

        Args:
            config: Auxiliary targets configuration
            board_size: Size of game board
            action_space_size: Size of action space
            num_iterations: Number of iterations

        Returns:
            BenchmarkResult with timing and loss statistics
        """
        config = config or AuxiliaryTargetsConfig()
        calculator = AuxiliaryLossCalculator(config)

        # Generate sample predictions and targets
        predictions_list = []
        targets_list = []

        for _ in range(num_iterations):
            attention_pred = self._rng.random((board_size, board_size))
            attention_pred /= attention_pred.sum()

            attention_target = self._rng.random((board_size, board_size))
            attention_target /= attention_target.sum()

            opponent_pred = self._rng.dirichlet(np.ones(action_space_size))
            opponent_target = self._rng.dirichlet(np.ones(action_space_size))

            phase_pred = self._rng.dirichlet(np.ones(3))
            phase_idx = int(self._rng.integers(0, 3))
            phase_target = np.zeros(3)
            phase_target[phase_idx] = 1.0

            predictions_list.append(
                {
                    "attention": attention_pred,
                    "opponent": opponent_pred,
                    "phase": phase_pred,
                    "future_value": float(self._rng.random() * 2 - 1),
                }
            )
            targets_list.append(
                AuxiliaryTargets(
                    attention_map=attention_target,
                    opponent_action=opponent_target,
                    game_phase=phase_target,
                    future_value=float(self._rng.random() * 2 - 1),
                )
            )

        idx = [0]

        def compute_loss() -> dict[str, float]:
            i = idx[0] % len(predictions_list)
            idx[0] += 1
            return calculator.compute_loss(predictions_list[i], targets_list[i])

        times, losses = self._time_function(compute_loss, num_iterations)

        # Aggregate loss statistics
        total_losses = [loss["total_auxiliary_loss"] for loss in losses]

        additional = {
            "mean_total_loss": float(np.mean(total_losses)),
            "std_total_loss": float(np.std(total_losses)),
        }

        result = self._create_benchmark_result(
            name="AuxiliaryLoss",
            times=times,
            additional_metrics=additional,
        )
        self._results.append(result)
        return result

    def compare_configurations(
        self,
        baseline_result: BenchmarkResult,
        optimized_result: BenchmarkResult,
    ) -> ComparisonResult:
        """Compare two benchmark results.

        Args:
            baseline_result: Baseline benchmark result
            optimized_result: Optimized benchmark result

        Returns:
            ComparisonResult with speedup metrics
        """
        speedup = optimized_result.throughput_per_second / max(
            baseline_result.throughput_per_second, 1e-10
        )
        time_reduction = (
            100
            * (baseline_result.mean_time_ms - optimized_result.mean_time_ms)
            / max(baseline_result.mean_time_ms, 1e-10)
        )

        return ComparisonResult(
            baseline=baseline_result,
            optimized=optimized_result,
            speedup_factor=speedup,
            time_reduction_percent=time_reduction,
        )

    def run_all_benchmarks(
        self,
        num_iterations: int = 1000,
    ) -> dict[str, BenchmarkResult]:
        """Run all optimization benchmarks.

        Args:
            num_iterations: Base number of iterations

        Returns:
            Dictionary of benchmark results
        """
        results = {}

        logger.info("Running playout cap benchmark...")
        results["playout_cap"] = self.run_playout_cap_benchmark(num_iterations=num_iterations)

        logger.info("Running policy pruning benchmark...")
        results["policy_pruning"] = self.run_policy_pruning_benchmark(num_iterations=num_iterations)

        logger.info("Running cyclic LR benchmark...")
        results["cyclic_lr"] = self.run_cyclic_lr_benchmark(
            num_iterations=num_iterations * 10  # LR lookups are very fast
        )

        logger.info("Running auxiliary targets benchmark...")
        results["auxiliary_targets"] = self.run_auxiliary_targets_benchmark(
            num_iterations=num_iterations // 2  # These are slower
        )

        logger.info("Running auxiliary loss benchmark...")
        results["auxiliary_loss"] = self.run_auxiliary_loss_benchmark(num_iterations=num_iterations)

        return results

    def get_all_results(self) -> list[BenchmarkResult]:
        """Get all benchmark results from this session.

        Returns:
            List of all BenchmarkResult instances
        """
        return self._results.copy()

    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results.clear()

    def summary(self) -> str:
        """Get summary of all benchmark results.

        Returns:
            Summary string
        """
        if not self._results:
            return "No benchmark results available."

        lines = [
            "Optimization Benchmark Summary",
            "=" * 50,
            "",
        ]

        for result in self._results:
            lines.extend(
                [
                    result.summary(),
                    "",
                ]
            )

        return "\n".join(lines)
