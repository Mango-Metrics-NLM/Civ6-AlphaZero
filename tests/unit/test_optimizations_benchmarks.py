"""Comprehensive tests for alpha_zero.optimizations.benchmarks.

Covers BenchmarkResult, ComparisonResult, and OptimizationBenchmark
including all benchmark methods, comparison, and summary generation.
"""

from __future__ import annotations

import pytest

from alpha_zero.optimizations.auxiliary_targets import AuxiliaryTargetsConfig
from alpha_zero.optimizations.benchmarks import (
    BenchmarkResult,
    ComparisonResult,
    OptimizationBenchmark,
)
from alpha_zero.optimizations.cyclic_lr import CyclicLRConfig
from alpha_zero.optimizations.playout_cap import PlayoutCapConfig
from alpha_zero.optimizations.policy_pruning import PolicyPruningConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED = 42
SMALL_ITERS = 20
WARMUP = 0  # Zero warmup avoids index-out-of-range in closure-based benchmarks


@pytest.fixture()
def benchmark():
    return OptimizationBenchmark(seed=SEED, warmup_iterations=WARMUP)


@pytest.fixture()
def sample_result():
    return BenchmarkResult(
        name="test",
        iterations=100,
        total_time_seconds=1.0,
        mean_time_ms=10.0,
        std_time_ms=2.0,
        min_time_ms=8.0,
        max_time_ms=15.0,
        throughput_per_second=100.0,
        additional_metrics={"extra_float": 0.5, "extra_int": 42},
    )


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_summary(self, sample_result):
        s = sample_result.summary()
        assert "test" in s
        assert "100" in s
        assert "10.000ms" in s
        assert "extra_float" in s
        assert "extra_int" in s

    def test_summary_no_additional(self):
        r = BenchmarkResult(
            name="simple",
            iterations=10,
            total_time_seconds=0.1,
            mean_time_ms=10.0,
            std_time_ms=1.0,
            min_time_ms=9.0,
            max_time_ms=12.0,
            throughput_per_second=100.0,
        )
        s = r.summary()
        assert "simple" in s
        assert "Additional" not in s

    def test_to_dict(self, sample_result):
        d = sample_result.to_dict()
        assert d["name"] == "test"
        assert d["iterations"] == 100
        assert d["additional_metrics"]["extra_float"] == 0.5


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------


class TestComparisonResult:
    def test_summary(self, sample_result):
        faster = BenchmarkResult(
            name="optimized",
            iterations=100,
            total_time_seconds=0.5,
            mean_time_ms=5.0,
            std_time_ms=1.0,
            min_time_ms=4.0,
            max_time_ms=7.0,
            throughput_per_second=200.0,
        )
        comp = ComparisonResult(
            baseline=sample_result,
            optimized=faster,
            speedup_factor=2.0,
            time_reduction_percent=50.0,
            additional_comparisons={"quality_delta": -0.01, "note": "good"},
        )
        s = comp.summary()
        assert "BASELINE" in s
        assert "OPTIMIZED" in s
        assert "2.00x" in s
        assert "50.0%" in s
        assert "quality_delta" in s

    def test_summary_no_extra_comparisons(self, sample_result):
        comp = ComparisonResult(
            baseline=sample_result,
            optimized=sample_result,
            speedup_factor=1.0,
            time_reduction_percent=0.0,
        )
        s = comp.summary()
        assert "1.00x" in s

    def test_to_dict(self, sample_result):
        comp = ComparisonResult(
            baseline=sample_result,
            optimized=sample_result,
            speedup_factor=1.0,
            time_reduction_percent=0.0,
        )
        d = comp.to_dict()
        assert "baseline" in d
        assert "optimized" in d
        assert d["speedup_factor"] == 1.0


# ---------------------------------------------------------------------------
# OptimizationBenchmark
# ---------------------------------------------------------------------------


class TestOptimizationBenchmark:
    def test_init(self, benchmark):
        assert benchmark._warmup_iterations == WARMUP
        assert len(benchmark._results) == 0

    def test_playout_cap_benchmark(self, benchmark):
        config = PlayoutCapConfig(min_playouts=10, max_playouts=50)
        result = benchmark.run_playout_cap_benchmark(config, num_iterations=SMALL_ITERS)
        assert result.iterations == SMALL_ITERS
        assert result.mean_time_ms > 0
        assert "mean_cap" in result.additional_metrics
        assert "effective_speedup" in result.additional_metrics
        assert result in benchmark.get_all_results()

    def test_playout_cap_benchmark_default_config(self, benchmark):
        result = benchmark.run_playout_cap_benchmark(num_iterations=SMALL_ITERS)
        assert result.iterations == SMALL_ITERS

    def test_policy_pruning_benchmark(self, benchmark):
        config = PolicyPruningConfig(min_probability=0.01, max_actions=50)
        result = benchmark.run_policy_pruning_benchmark(
            config, action_space_size=100, num_iterations=SMALL_ITERS
        )
        assert result.iterations == SMALL_ITERS
        assert "mean_original_actions" in result.additional_metrics
        assert "action_reduction_percent" in result.additional_metrics

    def test_policy_pruning_benchmark_default_config(self, benchmark):
        result = benchmark.run_policy_pruning_benchmark(
            action_space_size=50, num_iterations=SMALL_ITERS
        )
        assert result.iterations == SMALL_ITERS

    def test_cyclic_lr_benchmark(self, benchmark):
        config = CyclicLRConfig(base_lr=1e-5, max_lr=1e-3, step_size_up=100)
        result = benchmark.run_cyclic_lr_benchmark(config, num_iterations=SMALL_ITERS)
        assert result.iterations == SMALL_ITERS
        assert "mean_lr" in result.additional_metrics
        assert "schedule_type" in result.additional_metrics

    def test_cyclic_lr_benchmark_default_config(self, benchmark):
        result = benchmark.run_cyclic_lr_benchmark(num_iterations=SMALL_ITERS)
        assert result.iterations == SMALL_ITERS

    def test_auxiliary_targets_benchmark(self, benchmark):
        config = AuxiliaryTargetsConfig(
            enable_attention_targets=True,
            enable_opponent_modeling=True,
            enable_phase_prediction=True,
            enable_future_value=True,
        )
        result = benchmark.run_auxiliary_targets_benchmark(
            config, board_size=4, action_space_size=50, num_iterations=SMALL_ITERS
        )
        assert result.iterations == SMALL_ITERS
        assert "enabled_targets" in result.additional_metrics
        assert len(result.additional_metrics["enabled_targets"]) == 4

    def test_auxiliary_targets_benchmark_default(self, benchmark):
        result = benchmark.run_auxiliary_targets_benchmark(
            board_size=4, action_space_size=50, num_iterations=SMALL_ITERS
        )
        assert result.iterations == SMALL_ITERS

    def test_auxiliary_targets_benchmark_none_enabled(self, benchmark):
        config = AuxiliaryTargetsConfig(
            enable_attention_targets=False,
            enable_opponent_modeling=False,
            enable_phase_prediction=False,
            enable_future_value=False,
        )
        result = benchmark.run_auxiliary_targets_benchmark(
            config, board_size=4, action_space_size=50, num_iterations=SMALL_ITERS
        )
        assert result.additional_metrics["enabled_targets"] == []

    def test_auxiliary_loss_benchmark(self, benchmark):
        config = AuxiliaryTargetsConfig()
        result = benchmark.run_auxiliary_loss_benchmark(
            config, board_size=4, action_space_size=50, num_iterations=SMALL_ITERS
        )
        assert result.iterations == SMALL_ITERS
        assert "mean_total_loss" in result.additional_metrics

    def test_auxiliary_loss_benchmark_default(self, benchmark):
        result = benchmark.run_auxiliary_loss_benchmark(
            board_size=4, action_space_size=50, num_iterations=SMALL_ITERS
        )
        assert result.iterations == SMALL_ITERS

    def test_compare_configurations(self, benchmark):
        r1 = BenchmarkResult(
            name="baseline",
            iterations=100,
            total_time_seconds=1.0,
            mean_time_ms=10.0,
            std_time_ms=1.0,
            min_time_ms=9.0,
            max_time_ms=12.0,
            throughput_per_second=100.0,
        )
        r2 = BenchmarkResult(
            name="optimized",
            iterations=100,
            total_time_seconds=0.5,
            mean_time_ms=5.0,
            std_time_ms=0.5,
            min_time_ms=4.5,
            max_time_ms=6.0,
            throughput_per_second=200.0,
        )
        comp = benchmark.compare_configurations(r1, r2)
        assert comp.speedup_factor == pytest.approx(2.0)
        assert comp.time_reduction_percent == pytest.approx(50.0)

    def test_compare_configurations_zero_baseline(self, benchmark):
        """Edge case: baseline with zero throughput."""
        r1 = BenchmarkResult(
            name="baseline",
            iterations=0,
            total_time_seconds=0.0,
            mean_time_ms=0.0,
            std_time_ms=0.0,
            min_time_ms=0.0,
            max_time_ms=0.0,
            throughput_per_second=0.0,
        )
        r2 = BenchmarkResult(
            name="optimized",
            iterations=100,
            total_time_seconds=1.0,
            mean_time_ms=10.0,
            std_time_ms=1.0,
            min_time_ms=9.0,
            max_time_ms=12.0,
            throughput_per_second=100.0,
        )
        comp = benchmark.compare_configurations(r1, r2)
        assert comp.speedup_factor > 0  # Should not divide by zero

    def test_run_all_benchmarks(self, benchmark):
        results = benchmark.run_all_benchmarks(num_iterations=SMALL_ITERS)
        assert "playout_cap" in results
        assert "policy_pruning" in results
        assert "cyclic_lr" in results
        assert "auxiliary_targets" in results
        assert "auxiliary_loss" in results
        assert len(benchmark.get_all_results()) == 5

    def test_get_all_results_returns_copy(self, benchmark):
        benchmark.run_playout_cap_benchmark(num_iterations=SMALL_ITERS)
        r1 = benchmark.get_all_results()
        r2 = benchmark.get_all_results()
        assert r1 is not r2
        assert len(r1) == len(r2)

    def test_clear_results(self, benchmark):
        benchmark.run_playout_cap_benchmark(num_iterations=SMALL_ITERS)
        assert len(benchmark.get_all_results()) == 1
        benchmark.clear_results()
        assert len(benchmark.get_all_results()) == 0

    def test_summary_empty(self, benchmark):
        s = benchmark.summary()
        assert "No benchmark results" in s

    def test_summary_with_results(self, benchmark):
        benchmark.run_playout_cap_benchmark(num_iterations=SMALL_ITERS)
        s = benchmark.summary()
        assert "Optimization Benchmark Summary" in s
        assert "PlayoutCap" in s

    def test_time_function_returns_results(self, benchmark):
        counter = [0]

        def func():
            counter[0] += 1
            return counter[0]

        times, results = benchmark._time_function(func, 10)
        assert len(times) == 10
        assert len(results) == 10
        # Results should reflect function output after warmup
        assert all(r > 0 for r in results)

    def test_create_benchmark_result(self, benchmark):
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = benchmark._create_benchmark_result(
            name="test", times=times, additional_metrics={"key": "val"}
        )
        assert result.name == "test"
        assert result.iterations == 5
        assert result.mean_time_ms == pytest.approx(3.0)
        assert result.min_time_ms == pytest.approx(1.0)
        assert result.max_time_ms == pytest.approx(5.0)
        assert result.additional_metrics["key"] == "val"

    def test_create_benchmark_result_no_metrics(self, benchmark):
        times = [1.0, 2.0]
        result = benchmark._create_benchmark_result(name="bare", times=times)
        assert result.additional_metrics == {}

    def test_seed_reproducibility(self):
        b1 = OptimizationBenchmark(seed=123, warmup_iterations=2)
        b2 = OptimizationBenchmark(seed=123, warmup_iterations=2)
        r1 = b1.run_playout_cap_benchmark(num_iterations=10)
        r2 = b2.run_playout_cap_benchmark(num_iterations=10)
        assert r1.additional_metrics["mean_cap"] == pytest.approx(
            r2.additional_metrics["mean_cap"], rel=0.01
        )
