"""
Tests for the Benchmarks Module

Tests the benchmarking utilities for Civ6-AlphaZero.

Note: These tests require numpy to be installed (for logging utilities).
Tests will be skipped if numpy is not available.
"""

from __future__ import annotations

import time

import pytest

# Try to import benchmarks - skip tests if dependencies not available
try:
    from alpha_zero.utils.benchmarks import (
        BenchmarkConfig,
        Benchmarker,
        BenchmarkResult,
        BenchmarkSuite,
        benchmark_function,
        create_mechanics_benchmarks,
        time_function,
    )

    HAS_DEPENDENCIES = True
    IMPORT_ERROR = ""
except ImportError as e:
    HAS_DEPENDENCIES = False
    IMPORT_ERROR = str(e)
    # Placeholders for type checking
    BenchmarkConfig = None  # type: ignore[misc,assignment]
    BenchmarkResult = None  # type: ignore[misc,assignment]
    Benchmarker = None  # type: ignore[misc,assignment]
    BenchmarkSuite = None  # type: ignore[misc,assignment]
    benchmark_function = None  # type: ignore[assignment]
    create_mechanics_benchmarks = None  # type: ignore[assignment]
    time_function = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(
    not HAS_DEPENDENCIES, reason=f"Benchmark dependencies not available: {IMPORT_ERROR}"
)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BenchmarkConfig()
        assert config.warmup_iterations == 5
        assert config.benchmark_iterations == 100
        assert config.mcts_search_threshold > 0
        assert config.network_inference_threshold > 0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BenchmarkConfig(
            warmup_iterations=10,
            benchmark_iterations=50,
            mcts_search_threshold=20.0,
        )
        assert config.warmup_iterations == 10
        assert config.benchmark_iterations == 50
        assert config.mcts_search_threshold == 20.0


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_creation(self) -> None:
        """Test result creation."""
        result = BenchmarkResult(
            name="test",
            iterations=100,
            total_time_seconds=1.0,
            mean_time_seconds=0.01,
            min_time_seconds=0.005,
            max_time_seconds=0.02,
            std_time_seconds=0.003,
            ops_per_second=100.0,
            passed_threshold=True,
            threshold_ops_per_second=50.0,
        )
        assert result.name == "test"
        assert result.ops_per_second == 100.0
        assert result.passed_threshold is True

    def test_to_dict(self) -> None:
        """Test serialization."""
        result = BenchmarkResult(
            name="test",
            iterations=100,
            total_time_seconds=1.0,
            mean_time_seconds=0.01,
            min_time_seconds=0.005,
            max_time_seconds=0.02,
            std_time_seconds=0.003,
            ops_per_second=100.0,
            passed_threshold=True,
        )
        data = result.to_dict()
        assert data["name"] == "test"
        assert data["ops_per_second"] == 100.0

    def test_str_representation(self) -> None:
        """Test string representation."""
        result = BenchmarkResult(
            name="test",
            iterations=100,
            total_time_seconds=1.0,
            mean_time_seconds=0.01,
            min_time_seconds=0.005,
            max_time_seconds=0.02,
            std_time_seconds=0.003,
            ops_per_second=100.0,
            passed_threshold=True,
            threshold_ops_per_second=50.0,
        )
        s = str(result)
        assert "test" in s
        assert "PASS" in s


class TestBenchmarker:
    """Tests for Benchmarker."""

    def test_initialization(self) -> None:
        """Test benchmarker initialization."""
        benchmarker = Benchmarker()
        assert benchmarker.config is not None

    def test_run_benchmark(self) -> None:
        """Test running a benchmark."""
        config = BenchmarkConfig(warmup_iterations=2, benchmark_iterations=10)
        benchmarker = Benchmarker(config)

        def simple_function() -> int:
            return sum(range(100))

        result = benchmarker.run_benchmark(
            name="simple_sum",
            func=simple_function,
            threshold_ops_per_second=1.0,
        )

        assert result.name == "simple_sum"
        assert result.iterations == 10
        assert result.ops_per_second > 0
        assert result.passed_threshold is True

    def test_run_benchmark_with_threshold(self) -> None:
        """Test benchmark with threshold checking."""
        config = BenchmarkConfig(warmup_iterations=1, benchmark_iterations=5)
        benchmarker = Benchmarker(config)

        def slow_function() -> None:
            time.sleep(0.01)  # 10ms

        result = benchmarker.run_benchmark(
            name="slow",
            func=slow_function,
            threshold_ops_per_second=200.0,  # Impossible threshold
        )

        assert result.passed_threshold is False
        assert result.ops_per_second < 200.0

    def test_run_batch_benchmark(self) -> None:
        """Test batch benchmarking."""
        config = BenchmarkConfig(warmup_iterations=1, benchmark_iterations=5)
        benchmarker = Benchmarker(config)

        def batch_function(batch_size: int) -> list[int]:
            return list(range(batch_size))

        results = benchmarker.run_batch_benchmark(
            name="batch_test",
            func=batch_function,
            batch_sizes=[10, 100],
        )

        assert len(results) == 2
        assert results[0].extra_info["batch_size"] == 10
        assert results[1].extra_info["batch_size"] == 100

    def test_get_results(self) -> None:
        """Test getting all results."""
        config = BenchmarkConfig(warmup_iterations=1, benchmark_iterations=5)
        benchmarker = Benchmarker(config)

        benchmarker.run_benchmark("test1", lambda: None)
        benchmarker.run_benchmark("test2", lambda: None)

        results = benchmarker.get_results()
        assert len(results) == 2

    def test_get_summary(self) -> None:
        """Test summary generation."""
        config = BenchmarkConfig(warmup_iterations=1, benchmark_iterations=5)
        benchmarker = Benchmarker(config)

        benchmarker.run_benchmark("test1", lambda: None, threshold_ops_per_second=1.0)
        benchmarker.run_benchmark("test2", lambda: None, threshold_ops_per_second=1.0)

        summary = benchmarker.get_summary()
        assert summary["total_benchmarks"] == 2
        assert "passed" in summary
        assert "failed" in summary

    def test_reset(self) -> None:
        """Test resetting results."""
        config = BenchmarkConfig(warmup_iterations=1, benchmark_iterations=5)
        benchmarker = Benchmarker(config)

        benchmarker.run_benchmark("test", lambda: None)
        assert len(benchmarker.get_results()) == 1

        benchmarker.reset()
        assert len(benchmarker.get_results()) == 0


class TestBenchmarkFunctions:
    """Tests for benchmark helper functions."""

    def test_benchmark_function(self) -> None:
        """Test the benchmark_function helper."""

        def simple() -> int:
            return 42

        result = benchmark_function(simple, iterations=10, warmup=2)
        assert result.name == "simple"
        assert result.iterations == 10

    def test_time_function(self) -> None:
        """Test the time_function helper."""

        def compute() -> int:
            return sum(range(1000))

        result, elapsed = time_function(compute)
        assert result == sum(range(1000))
        assert elapsed > 0
        assert elapsed < 1.0  # Should be very fast


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_initialization(self) -> None:
        """Test suite initialization."""
        suite = BenchmarkSuite()
        assert suite.config is not None
        assert suite.benchmarker is not None

    def test_register_decorator(self) -> None:
        """Test registering benchmarks with decorator."""
        suite = BenchmarkSuite(
            BenchmarkConfig(
                warmup_iterations=1,
                benchmark_iterations=5,
            )
        )

        @suite.register("test_bench", threshold=1.0)
        def bench_test() -> None:
            _ = sum(range(100))

        assert "test_bench" in suite._benchmark_funcs

    def test_run_all(self) -> None:
        """Test running all benchmarks."""
        suite = BenchmarkSuite(
            BenchmarkConfig(
                warmup_iterations=1,
                benchmark_iterations=5,
            )
        )

        @suite.register("bench1")
        def bench1() -> None:
            pass

        @suite.register("bench2")
        def bench2() -> None:
            pass

        results = suite.run_all()
        assert "bench1" in results
        assert "bench2" in results

    def test_run_subset(self) -> None:
        """Test running subset of benchmarks."""
        suite = BenchmarkSuite(
            BenchmarkConfig(
                warmup_iterations=1,
                benchmark_iterations=5,
            )
        )

        @suite.register("bench1")
        def bench1() -> None:
            pass

        @suite.register("bench2")
        def bench2() -> None:
            pass

        results = suite.run_subset(["bench1"])
        assert "bench1" in results
        assert "bench2" not in results


class TestMechanicsBenchmarks:
    """Tests for mechanics benchmark suite."""

    def test_create_suite(self) -> None:
        """Test creating mechanics benchmark suite."""
        config = BenchmarkConfig(
            warmup_iterations=1,
            benchmark_iterations=5,
        )
        suite = create_mechanics_benchmarks(config)

        # Should have registered benchmarks
        assert len(suite._benchmark_funcs) > 0

    def test_run_mechanics_benchmarks(self) -> None:
        """Test running mechanics benchmarks."""
        config = BenchmarkConfig(
            warmup_iterations=1,
            benchmark_iterations=5,
            mechanics_update_threshold=1.0,  # Low threshold for testing
        )
        suite = create_mechanics_benchmarks(config)

        results = suite.run_all()
        assert len(results) > 0

        # All benchmarks should complete
        for _name, result in results.items():
            assert result.iterations == 5
            assert result.ops_per_second > 0


class TestBenchmarkStatistics:
    """Tests for benchmark statistics accuracy."""

    def test_timing_accuracy(self) -> None:
        """Test that timing statistics are reasonable."""
        config = BenchmarkConfig(warmup_iterations=2, benchmark_iterations=20)
        benchmarker = Benchmarker(config)

        # Function with predictable timing
        def timed_sleep() -> None:
            time.sleep(0.001)  # 1ms

        result = benchmarker.run_benchmark(
            name="sleep_test",
            func=timed_sleep,
        )

        # Mean should be around 1ms (0.001s)
        assert result.mean_time_seconds >= 0.001
        assert result.mean_time_seconds < 0.01  # Some overhead allowed

        # Min should not be less than sleep time
        assert result.min_time_seconds >= 0.001

        # Ops/sec should be around 1000 (1/0.001)
        assert result.ops_per_second > 100
        assert result.ops_per_second < 1000

    def test_std_deviation(self) -> None:
        """Test standard deviation calculation."""
        config = BenchmarkConfig(warmup_iterations=1, benchmark_iterations=50)
        benchmarker = Benchmarker(config)

        # Very consistent function
        def consistent() -> int:
            return 42

        result = benchmarker.run_benchmark("consistent", consistent)

        # Std should be relatively small for consistent operations
        # (relative to mean)
        if result.mean_time_seconds > 0:
            cv = result.std_time_seconds / result.mean_time_seconds
            assert cv < 10.0  # Coefficient of variation should be reasonable
