"""
Performance Benchmarks for Civ6-AlphaZero

Provides configurable benchmarks for measuring performance of key components:
- Neural network inference
- MCTS search
- Game mechanics operations
- State encoding/decoding

All thresholds and parameters are configurable.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""

    # Timing configuration
    warmup_iterations: int = 5
    benchmark_iterations: int = 100
    min_time_seconds: float = 0.1

    # Performance thresholds (ops/second)
    mcts_search_threshold: float = 10.0
    network_inference_threshold: float = 100.0
    state_encoding_threshold: float = 1000.0
    mechanics_update_threshold: float = 10000.0

    # Memory thresholds (MB)
    max_memory_mb: float = 1024.0

    # Enable detailed profiling
    enable_profiling: bool = False


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    total_time_seconds: float
    mean_time_seconds: float
    min_time_seconds: float
    max_time_seconds: float
    std_time_seconds: float
    ops_per_second: float
    passed_threshold: bool
    threshold_ops_per_second: float = 0.0
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_seconds": self.total_time_seconds,
            "mean_time_seconds": self.mean_time_seconds,
            "min_time_seconds": self.min_time_seconds,
            "max_time_seconds": self.max_time_seconds,
            "std_time_seconds": self.std_time_seconds,
            "ops_per_second": self.ops_per_second,
            "passed_threshold": self.passed_threshold,
            "threshold_ops_per_second": self.threshold_ops_per_second,
            "extra_info": self.extra_info,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        status = "PASS" if self.passed_threshold else "FAIL"
        return (
            f"{self.name}: {self.ops_per_second:.2f} ops/sec "
            f"(threshold: {self.threshold_ops_per_second:.2f}) [{status}]\n"
            f"  Mean: {self.mean_time_seconds*1000:.3f}ms, "
            f"Min: {self.min_time_seconds*1000:.3f}ms, "
            f"Max: {self.max_time_seconds*1000:.3f}ms"
        )


class Benchmarker:
    """Benchmarking utility for measuring performance."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        """Initialize benchmarker.

        Args:
            config: Benchmark configuration.
        """
        self.config = config or BenchmarkConfig()
        self._results: list[BenchmarkResult] = []

    def run_benchmark(
        self,
        name: str,
        func: Callable[[], T],
        threshold_ops_per_second: float = 0.0,
        iterations: int | None = None,
    ) -> BenchmarkResult:
        """Run a benchmark on a function.

        Args:
            name: Name of the benchmark.
            func: Function to benchmark (no arguments).
            threshold_ops_per_second: Minimum ops/second threshold.
            iterations: Number of iterations (uses config default if None).

        Returns:
            BenchmarkResult with timing statistics.
        """
        num_iterations = iterations or self.config.benchmark_iterations

        # Warmup
        for _ in range(self.config.warmup_iterations):
            func()

        # Benchmark
        times: list[float] = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append(end - start)

        # Calculate statistics
        total_time = sum(times)
        mean_time = total_time / len(times)
        min_time = min(times)
        max_time = max(times)

        # Standard deviation
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        std_time = variance ** 0.5

        ops_per_second = 1.0 / mean_time if mean_time > 0 else float("inf")
        passed = ops_per_second >= threshold_ops_per_second

        result = BenchmarkResult(
            name=name,
            iterations=num_iterations,
            total_time_seconds=total_time,
            mean_time_seconds=mean_time,
            min_time_seconds=min_time,
            max_time_seconds=max_time,
            std_time_seconds=std_time,
            ops_per_second=ops_per_second,
            passed_threshold=passed,
            threshold_ops_per_second=threshold_ops_per_second,
        )

        self._results.append(result)
        return result

    def run_batch_benchmark(
        self,
        name: str,
        func: Callable[[int], T],
        batch_sizes: list[int],
        threshold_ops_per_second: float = 0.0,
    ) -> list[BenchmarkResult]:
        """Run benchmarks across multiple batch sizes.

        Args:
            name: Base name of the benchmark.
            func: Function that takes batch size as argument.
            batch_sizes: List of batch sizes to test.
            threshold_ops_per_second: Minimum ops/second threshold.

        Returns:
            List of BenchmarkResults, one per batch size.
        """
        results = []
        for batch_size in batch_sizes:
            benchmark_name = f"{name} (batch={batch_size})"
            result = self.run_benchmark(
                name=benchmark_name,
                func=lambda bs=batch_size: func(bs),
                threshold_ops_per_second=threshold_ops_per_second,
            )
            result.extra_info["batch_size"] = batch_size
            results.append(result)
        return results

    def get_results(self) -> list[BenchmarkResult]:
        """Get all benchmark results."""
        return self._results.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all benchmarks."""
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed_threshold)
        failed = total - passed

        return {
            "total_benchmarks": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 1.0,
            "results": [r.to_dict() for r in self._results],
        }

    def print_summary(self) -> None:
        """Print a summary of all benchmarks."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        for result in self._results:
            print(result)
            print()

        summary = self.get_summary()
        print("-" * 60)
        print(
            f"Total: {summary['total_benchmarks']}, "
            f"Passed: {summary['passed']}, "
            f"Failed: {summary['failed']}, "
            f"Pass Rate: {summary['pass_rate']*100:.1f}%"
        )
        print("=" * 60)

    def reset(self) -> None:
        """Reset all benchmark results."""
        self._results.clear()


def benchmark_function(
    func: Callable[[], T],
    iterations: int = 100,
    warmup: int = 5,
) -> BenchmarkResult:
    """Convenience function to benchmark a single function.

    Args:
        func: Function to benchmark.
        iterations: Number of iterations.
        warmup: Number of warmup iterations.

    Returns:
        BenchmarkResult.
    """
    config = BenchmarkConfig(
        warmup_iterations=warmup,
        benchmark_iterations=iterations,
    )
    benchmarker = Benchmarker(config)
    return benchmarker.run_benchmark(
        name=func.__name__ if hasattr(func, "__name__") else "anonymous",
        func=func,
    )


def time_function(func: Callable[[], T]) -> tuple[T, float]:
    """Time a single function execution.

    Args:
        func: Function to time.

    Returns:
        Tuple of (result, elapsed_seconds).
    """
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    return result, elapsed


class BenchmarkSuite:
    """Suite of benchmarks for Civ6-AlphaZero components."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        """Initialize benchmark suite.

        Args:
            config: Benchmark configuration.
        """
        self.config = config or BenchmarkConfig()
        self.benchmarker = Benchmarker(config)
        self._benchmark_funcs: dict[str, Callable[[], BenchmarkResult]] = {}

    def register(
        self,
        name: str,
        threshold: float = 0.0,
    ) -> Callable[[Callable[[], None]], Callable[[], None]]:
        """Decorator to register a benchmark function.

        Args:
            name: Name of the benchmark.
            threshold: Minimum ops/second threshold.

        Returns:
            Decorator function.
        """
        def decorator(func: Callable[[], None]) -> Callable[[], None]:
            def run_benchmark() -> BenchmarkResult:
                return self.benchmarker.run_benchmark(
                    name=name,
                    func=func,
                    threshold_ops_per_second=threshold,
                )
            self._benchmark_funcs[name] = run_benchmark
            return func
        return decorator

    def run_all(self) -> dict[str, BenchmarkResult]:
        """Run all registered benchmarks.

        Returns:
            Dictionary mapping names to results.
        """
        results = {}
        for name, run_func in self._benchmark_funcs.items():
            results[name] = run_func()
        return results

    def run_subset(self, names: list[str]) -> dict[str, BenchmarkResult]:
        """Run a subset of benchmarks.

        Args:
            names: List of benchmark names to run.

        Returns:
            Dictionary mapping names to results.
        """
        results = {}
        for name in names:
            if name in self._benchmark_funcs:
                results[name] = self._benchmark_funcs[name]()
        return results


def create_mechanics_benchmarks(
    config: BenchmarkConfig | None = None,
) -> BenchmarkSuite:
    """Create a benchmark suite for game mechanics.

    Args:
        config: Benchmark configuration.

    Returns:
        Configured BenchmarkSuite.

    Note:
        Mechanics modules are imported lazily inside benchmark functions
        to avoid import errors when numpy is not available.
    """
    config = config or BenchmarkConfig()
    suite = BenchmarkSuite(config)

    # Diplomacy benchmarks
    @suite.register("diplomacy_get_relation", config.mechanics_update_threshold)
    def bench_diplomacy_get_relation() -> None:
        from alpha_zero.mechanics.diplomacy import DiplomacyManager
        manager = DiplomacyManager()
        _ = manager.get_relation(0, 1)

    @suite.register("diplomacy_process_turn", config.mechanics_update_threshold / 10)
    def bench_diplomacy_process_turn() -> None:
        from alpha_zero.mechanics.diplomacy import DiplomacyManager
        manager = DiplomacyManager()
        for i in range(4):
            for j in range(i + 1, 4):
                manager.get_relation(i, j)
        manager.process_turn(current_turn=0)

    @suite.register("diplomacy_vector", config.mechanics_update_threshold / 10)
    def bench_diplomacy_vector() -> None:
        from alpha_zero.mechanics.diplomacy import DiplomacyManager
        manager = DiplomacyManager()
        _ = manager.get_diplomatic_vector(0, num_players=8)

    # Religion benchmarks
    @suite.register("religion_add_faith", config.mechanics_update_threshold)
    def bench_religion_add_faith() -> None:
        from alpha_zero.mechanics.religion import ReligionManager
        manager = ReligionManager()
        manager.add_faith(0, 100)

    @suite.register("religion_vector", config.mechanics_update_threshold / 10)
    def bench_religion_vector() -> None:
        from alpha_zero.mechanics.religion import ReligionManager
        manager = ReligionManager()
        _ = manager.get_religious_vector(0, max_religions=7)

    # Culture benchmarks
    @suite.register("culture_tourism", config.mechanics_update_threshold / 10)
    def bench_culture_tourism() -> None:
        from alpha_zero.mechanics.culture import CultureManager, GreatWorkType
        manager = CultureManager()
        for i in range(5):
            manager.create_great_work(GreatWorkType.WRITING, f"Work {i}", 0)
        _ = manager.calculate_tourism_per_turn(0)

    @suite.register("culture_process_turn", config.mechanics_update_threshold / 100)
    def bench_culture_process_turn() -> None:
        from alpha_zero.mechanics.culture import CultureManager
        manager = CultureManager()
        manager.process_turn([0, 1, 2, 3])

    @suite.register("culture_vector", config.mechanics_update_threshold / 10)
    def bench_culture_vector() -> None:
        from alpha_zero.mechanics.culture import CultureManager
        manager = CultureManager()
        _ = manager.get_cultural_vector(0, num_players=4)

    return suite


def run_all_benchmarks(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    """Run all available benchmarks.

    Args:
        config: Benchmark configuration.

    Returns:
        Summary of all benchmark results.
    """
    config = config or BenchmarkConfig()

    # Run mechanics benchmarks
    mechanics_suite = create_mechanics_benchmarks(config)
    mechanics_results = mechanics_suite.run_all()

    # Combine results
    benchmarker = Benchmarker(config)
    for result in mechanics_results.values():
        benchmarker._results.append(result)

    benchmarker.print_summary()
    return benchmarker.get_summary()


if __name__ == "__main__":
    run_all_benchmarks()
