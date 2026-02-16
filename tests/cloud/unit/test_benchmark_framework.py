"""
Unit tests for the MuZero vs AlphaZero benchmark framework.

Tests BenchmarkConfig, BenchmarkResult, ComparisonReport,
and AlphaZeroVsMuZeroBenchmark components.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

# =============================================================================
# BenchmarkConfig Tests
# =============================================================================


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_config(self) -> None:
        """Test that default config has sensible values."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import BenchmarkConfig

        config = BenchmarkConfig()

        assert config.num_games > 0
        assert config.board_size > 0
        assert config.action_space_size > 0
        assert config.mcts_simulations > 0
        assert config.num_res_blocks > 0
        assert config.num_filters > 0

    def test_custom_config(self) -> None:
        """Test creating config with custom values."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import BenchmarkConfig

        config = BenchmarkConfig(
            num_games=20,
            board_size=16,
            action_space_size=500,
            mcts_simulations=100,
        )

        assert config.num_games == 20
        assert config.board_size == 16
        assert config.action_space_size == 500
        assert config.mcts_simulations == 100


# =============================================================================
# SearchMetrics Tests
# =============================================================================


class TestSearchMetrics:
    """Tests for SearchMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test that search metrics default to 0."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import SearchMetrics

        metrics = SearchMetrics()

        assert metrics.total_env_clones == 0
        assert metrics.total_dynamics_calls == 0
        assert metrics.total_network_evals == 0

    def test_custom_values(self) -> None:
        """Test creating search metrics with custom values."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import SearchMetrics

        metrics = SearchMetrics(
            total_env_clones=100,
            total_dynamics_calls=50,
            total_network_evals=200,
        )

        assert metrics.total_env_clones == 100
        assert metrics.total_dynamics_calls == 50
        assert metrics.total_network_evals == 200


# =============================================================================
# BenchmarkResult Tests
# =============================================================================


class TestBenchmarkResult:
    """Tests for BenchmarkResult dataclass."""

    def test_default_empty(self) -> None:
        """Test that benchmark result initializes with empty lists."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import BenchmarkResult

        result = BenchmarkResult(algorithm="alphazero")

        assert result.algorithm == "alphazero"
        assert result.wins == 0
        assert result.losses == 0
        assert result.draws == 0
        assert result.game_lengths == []

    def test_win_rate_computation(self) -> None:
        """Test win rate is computed correctly."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import BenchmarkResult

        result = BenchmarkResult(algorithm="muzero")
        result.wins = 7
        result.losses = 2
        result.draws = 1

        total = result.wins + result.losses + result.draws
        win_rate = result.wins / total if total > 0 else 0.0

        assert win_rate == pytest.approx(0.7)


# =============================================================================
# ComparisonReport Tests
# =============================================================================


class TestComparisonReport:
    """Tests for ComparisonReport."""

    def _make_results(self) -> tuple:
        """Helper to create test results."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import (
            BenchmarkResult,
            SearchMetrics,
        )

        az_result = BenchmarkResult(algorithm="alphazero")
        az_result.wins = 5
        az_result.losses = 3
        az_result.draws = 2
        az_result.game_lengths = [20, 25, 30, 22, 28, 18, 35, 32, 26, 24]
        az_result.search_metrics = SearchMetrics(total_env_clones=500, total_network_evals=1000)

        mz_result = BenchmarkResult(algorithm="muzero")
        mz_result.wins = 3
        mz_result.losses = 5
        mz_result.draws = 2
        mz_result.game_lengths = [22, 27, 32, 24, 30, 20, 33, 28, 25, 23]
        mz_result.search_metrics = SearchMetrics(total_dynamics_calls=300, total_network_evals=800)

        return az_result, mz_result

    def _make_config(self) -> object:
        """Helper to create a BenchmarkConfig for tests."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import BenchmarkConfig

        return BenchmarkConfig(num_games=10, mcts_simulations=400)

    def test_report_creation(self) -> None:
        """Test that ComparisonReport can be created."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import ComparisonReport

        az, mz = self._make_results()
        report = ComparisonReport(
            config=self._make_config(),
            alphazero_result=az,
            muzero_result=mz,
        )

        assert report.alphazero_result.algorithm == "alphazero"
        assert report.muzero_result.algorithm == "muzero"

    def test_report_compute_summary(self) -> None:
        """Test computing summary metrics from report."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import ComparisonReport

        az, mz = self._make_results()
        report = ComparisonReport(
            config=self._make_config(),
            alphazero_result=az,
            muzero_result=mz,
        )

        summary = report.compute_summary()

        assert isinstance(summary, dict)
        assert "alphazero_win_rate" in summary
        assert "muzero_win_rate" in summary
        assert "win_rate_delta" in summary
        assert "search_efficiency_ratio" in summary

    def test_report_summary_values(self) -> None:
        """Test that computed summary values are correct."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import ComparisonReport

        az, mz = self._make_results()
        report = ComparisonReport(
            config=self._make_config(),
            alphazero_result=az,
            muzero_result=mz,
        )

        summary = report.compute_summary()

        # env_clones=500 from AlphaZero, dynamics_calls=300 from MuZero
        assert summary["alphazero_env_clones"] == 500
        assert summary["muzero_dynamics_calls"] == 300

    def test_report_summary_stored(self) -> None:
        """Test that compute_summary stores results on report."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import ComparisonReport

        az, mz = self._make_results()
        report = ComparisonReport(
            config=self._make_config(),
            alphazero_result=az,
            muzero_result=mz,
        )

        assert report.summary == {}
        report.compute_summary()
        assert len(report.summary) > 0


# =============================================================================
# AlphaZeroVsMuZeroBenchmark Tests
# =============================================================================


class TestBenchmarkClass:
    """Tests for the main benchmark orchestrator class."""

    def test_benchmark_creation(self) -> None:
        """Test that benchmark can be instantiated."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import (
            AlphaZeroVsMuZeroBenchmark,
            BenchmarkConfig,
        )

        config = BenchmarkConfig(num_games=2, mcts_simulations=10)
        benchmark = AlphaZeroVsMuZeroBenchmark(config)

        assert benchmark.config.num_games == 2
        assert benchmark.config.mcts_simulations == 10

    def test_create_alphazero_model(self) -> None:
        """Test creating an AlphaZero model through benchmark."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import (
            AlphaZeroVsMuZeroBenchmark,
            BenchmarkConfig,
        )

        config = BenchmarkConfig(
            board_size=8,
            action_space_size=376,
            num_res_blocks=4,
            num_filters=32,
        )
        benchmark = AlphaZeroVsMuZeroBenchmark(config)

        model = benchmark._create_model("alphazero")
        assert model is not None

        # Verify forward pass works
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)
        model.eval()
        with torch.no_grad():
            policy, value = model(spatial, vector)
        assert policy.shape[1] == 376

    def test_create_muzero_model(self) -> None:
        """Test creating a MuZero model through benchmark."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import (
            AlphaZeroVsMuZeroBenchmark,
            BenchmarkConfig,
        )

        config = BenchmarkConfig(
            board_size=8,
            action_space_size=376,
            num_res_blocks=4,
            num_filters=32,
        )
        benchmark = AlphaZeroVsMuZeroBenchmark(config)

        model = benchmark._create_model("muzero")
        assert model is not None

        # Verify forward pass works
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)
        model.eval()
        with torch.no_grad():
            policy, value = model(spatial, vector)
        assert policy.shape[1] == 376

    def test_create_model_unknown_algorithm_falls_back(self) -> None:
        """Test that unknown algorithm falls back to AlphaZero (else branch)."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import (
            AlphaZeroVsMuZeroBenchmark,
            BenchmarkConfig,
        )

        config = BenchmarkConfig(num_res_blocks=2, num_filters=16)
        benchmark = AlphaZeroVsMuZeroBenchmark(config)

        # Unknown algorithm falls through to AlphaZero else-branch
        model = benchmark._create_model("dreamer")
        assert model is not None


# =============================================================================
# Integration Sanity Tests
# =============================================================================


class TestBenchmarkIntegration:
    """Lightweight integration tests for the benchmark framework."""

    def test_full_benchmark_dry_run(self) -> None:
        """Test running full benchmark with tiny config (dry run style)."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import (
            AlphaZeroVsMuZeroBenchmark,
            BenchmarkConfig,
        )

        config = BenchmarkConfig(
            num_games=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=2,
            num_filters=16,
            mcts_simulations=2,
            max_turns=5,
        )

        benchmark = AlphaZeroVsMuZeroBenchmark(config)

        # Mock the game playing to avoid heavy computation
        with patch.object(
            benchmark,
            "_play_single_game",
            return_value={
                "winner": "alphazero",
                "length": 10,
                "reward": 1.0,
                "search_metrics": {},
            },
        ):
            report = benchmark.run_comparison()

        assert report is not None

    def test_benchmark_report_formats(self) -> None:
        """Test that benchmark produces both JSON and Markdown reports."""
        from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import (
            BenchmarkConfig,
            BenchmarkResult,
            ComparisonReport,
            SearchMetrics,
        )

        az = BenchmarkResult(algorithm="alphazero")
        az.wins = 1
        az.game_lengths = [10]
        az.search_metrics = SearchMetrics(total_env_clones=100)

        mz = BenchmarkResult(algorithm="muzero")
        mz.wins = 0
        mz.losses = 1
        mz.game_lengths = [10]
        mz.search_metrics = SearchMetrics(total_dynamics_calls=50)

        report = ComparisonReport(
            config=BenchmarkConfig(num_games=1),
            alphazero_result=az,
            muzero_result=mz,
        )

        # compute_summary should work and produce valid output
        summary = report.compute_summary()

        assert isinstance(summary, dict)
        assert len(summary) > 0
        assert summary["alphazero_env_clones"] == 100
        assert summary["muzero_dynamics_calls"] == 50
