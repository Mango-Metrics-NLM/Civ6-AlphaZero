"""Tests for alpha_zero.evaluation.muzero_vs_alphazero_benchmark module.

Focuses on coverage of BenchmarkConfig, SearchMetrics, BenchmarkResult,
ComparisonReport, and AlphaZeroVsMuZeroBenchmark.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.evaluation.muzero_vs_alphazero_benchmark import (
    AlphaZeroVsMuZeroBenchmark,
    BenchmarkConfig,
    BenchmarkResult,
    ComparisonReport,
    SearchMetrics,
)

# ---------------------------------------------------------------------------
# Dataclass construction and default tests
# ---------------------------------------------------------------------------


class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.num_games == 50
        assert cfg.mcts_simulations == 100
        assert cfg.board_size == 8
        assert cfg.device == "cpu"
        assert cfg.wandb_enabled is False
        assert cfg.win_reward_threshold == 0.5
        assert cfg.loss_reward_threshold == -0.5

    def test_custom_values(self):
        cfg = BenchmarkConfig(num_games=10, board_size=4, device="cuda")
        assert cfg.num_games == 10
        assert cfg.board_size == 4
        assert cfg.device == "cuda"


class TestSearchMetrics:
    def test_defaults(self):
        m = SearchMetrics()
        assert m.total_env_clones == 0
        assert m.total_dynamics_calls == 0
        assert m.total_network_evals == 0
        assert m.avg_search_time_ms == 0.0

    def test_custom(self):
        m = SearchMetrics(total_env_clones=100, total_dynamics_calls=50)
        assert m.total_env_clones == 100
        assert m.total_dynamics_calls == 50


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


class TestBenchmarkResult:
    def test_compute_derived_metrics_with_data(self):
        r = BenchmarkResult(
            algorithm="alphazero",
            num_games=4,
            game_lengths=[10, 20, 30, 40],
            game_rewards=[1.0, -1.0, 0.0, 1.0],
            total_time_seconds=120.0,
        )
        r.compute_derived_metrics(win_threshold=0.5, loss_threshold=-0.5)
        assert r.avg_game_length == 25.0
        assert r.avg_reward == 0.25
        assert r.wins == 2
        assert r.losses == 1
        assert r.draws == 1
        assert r.win_rate == 0.5
        assert r.games_per_minute == pytest.approx(2.0)

    def test_compute_derived_metrics_empty(self):
        r = BenchmarkResult(algorithm="muzero", num_games=0)
        r.compute_derived_metrics()
        assert r.avg_game_length == 0.0
        assert r.avg_reward == 0.0
        assert r.games_per_minute == 0.0

    def test_compute_derived_metrics_zero_time(self):
        r = BenchmarkResult(
            algorithm="test",
            num_games=1,
            game_lengths=[5],
            game_rewards=[1.0],
            total_time_seconds=0.0,
        )
        r.compute_derived_metrics()
        assert r.games_per_minute == 0.0


# ---------------------------------------------------------------------------
# ComparisonReport
# ---------------------------------------------------------------------------


class TestComparisonReport:
    @pytest.fixture()
    def report(self):
        az = BenchmarkResult(
            algorithm="alphazero",
            num_games=10,
            win_rate=0.6,
            avg_game_length=20.0,
            games_per_minute=5.0,
            search_metrics=SearchMetrics(total_env_clones=100, total_network_evals=200),
        )
        mz = BenchmarkResult(
            algorithm="muzero",
            num_games=10,
            win_rate=0.7,
            avg_game_length=18.0,
            games_per_minute=4.0,
            search_metrics=SearchMetrics(total_dynamics_calls=150, total_network_evals=200),
        )
        return ComparisonReport(
            config=BenchmarkConfig(num_games=10),
            alphazero_result=az,
            muzero_result=mz,
            timestamp="2026-01-01T00:00:00",
        )

    def test_compute_summary(self, report):
        s = report.compute_summary()
        assert s["alphazero_win_rate"] == 0.6
        assert s["muzero_win_rate"] == 0.7
        assert s["win_rate_delta"] == pytest.approx(0.1)
        assert "search_efficiency_ratio" in s
        assert "speed_ratio" in s
        assert s["search_efficiency_ratio"] == pytest.approx(150 / 100)
        assert s["speed_ratio"] == pytest.approx(4.0 / 5.0)

    def test_compute_summary_zero_env_clones(self):
        """When alphazero has 0 env clones, the ratio divides by 1 instead."""
        az = BenchmarkResult(
            algorithm="alphazero",
            search_metrics=SearchMetrics(total_env_clones=0),
            games_per_minute=0.0,
        )
        mz = BenchmarkResult(
            algorithm="muzero",
            search_metrics=SearchMetrics(total_dynamics_calls=50),
            games_per_minute=0.0,
        )
        r = ComparisonReport(
            config=BenchmarkConfig(),
            alphazero_result=az,
            muzero_result=mz,
        )
        s = r.compute_summary()
        assert s["search_efficiency_ratio"] == 50.0
        # speed_ratio with zero games/min
        assert s["speed_ratio"] == pytest.approx(0.0 / 0.001)


# ---------------------------------------------------------------------------
# AlphaZeroVsMuZeroBenchmark
# ---------------------------------------------------------------------------


class TestAlphaZeroVsMuZeroBenchmark:
    @pytest.fixture()
    def config(self):
        return BenchmarkConfig(num_games=2, mcts_simulations=2, board_size=8, max_turns=5)

    @pytest.fixture()
    def benchmark(self, config):
        return AlphaZeroVsMuZeroBenchmark(config)

    # -- run_comparison with mocked _run_algorithm --
    def test_run_comparison(self, benchmark):
        fake_result = BenchmarkResult(
            algorithm="test",
            num_games=2,
            game_lengths=[3, 4],
            game_rewards=[1.0, -1.0],
            total_time_seconds=1.0,
            search_metrics=SearchMetrics(total_env_clones=10),
        )
        fake_result.compute_derived_metrics()

        with patch.object(benchmark, "_run_algorithm", return_value=fake_result):
            report = benchmark.run_comparison()

        assert isinstance(report, ComparisonReport)
        assert report.timestamp != ""
        assert report.summary  # should have been computed

    # -- _run_algorithm with mocked _play_single_game --
    def test_run_algorithm(self, benchmark):
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = None

        game_result = {
            "length": 5,
            "reward": 1.0,
            "env_clones": 2,
            "dynamics_calls": 0,
            "network_evals": 2,
        }
        with patch.object(benchmark, "_play_single_game", return_value=game_result):
            result = benchmark._run_algorithm("alphazero", model=model)

        assert result.num_games == 2
        assert result.game_lengths == [5, 5]
        assert result.game_rewards == [1.0, 1.0]
        assert result.search_metrics.total_env_clones == 4

    def test_run_algorithm_creates_model_when_none(self, benchmark):
        """When model=None, _create_model should be called."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None

        game_result = {
            "length": 3,
            "reward": 0.0,
            "env_clones": 0,
            "dynamics_calls": 0,
            "network_evals": 0,
        }
        with (
            patch.object(benchmark, "_create_model", return_value=mock_model) as cm,
            patch.object(benchmark, "_play_single_game", return_value=game_result),
        ):
            benchmark._run_algorithm("muzero", model=None)
            cm.assert_called_once_with("muzero")

    # -- _create_model --
    def test_create_model_alphazero(self, benchmark):
        with patch(
            "alpha_zero.evaluation.muzero_vs_alphazero_benchmark.AlphaZeroVsMuZeroBenchmark._create_model"
        ) as mock_cm:
            mock_cm.return_value = MagicMock()
            model = mock_cm("alphazero")
            assert model is not None

    def test_create_model_muzero(self, benchmark):
        with patch(
            "alpha_zero.evaluation.muzero_vs_alphazero_benchmark.AlphaZeroVsMuZeroBenchmark._create_model"
        ) as mock_cm:
            mock_cm.return_value = MagicMock()
            model = mock_cm("muzero")
            assert model is not None

    # -- _play_single_game --
    def test_play_single_game_alphazero(self, benchmark):
        mock_env = MagicMock()
        mock_env.reset.return_value = None
        mock_env.step.return_value = (None, 1.0, True, False, {})

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.5, 0.5], None)
        mock_mcts.select_action.return_value = 0

        model = MagicMock()

        with (
            patch(
                "alpha_zero.core.mcts_factory.create_mcts",
                return_value=mock_mcts,
            ),
            patch(
                "alpha_zero.envs.env_factory.create_game_env",
                return_value=mock_env,
            ),
        ):
            result = benchmark._play_single_game(model=model, algorithm="alphazero", game_idx=0)

        assert result["length"] == 1
        assert result["reward"] == 1.0
        assert result["env_clones"] == 2  # config.mcts_simulations
        assert result["dynamics_calls"] == 0

    def test_play_single_game_muzero(self, benchmark):
        mock_env = MagicMock()
        mock_env.reset.return_value = None
        mock_env.step.return_value = (None, -1.0, False, True, {})

        mock_mcts = MagicMock()
        mock_mcts.search.return_value = ([0.5, 0.5], None)
        mock_mcts.select_action.return_value = 0

        model = MagicMock()

        with (
            patch(
                "alpha_zero.core.mcts_factory.create_mcts",
                return_value=mock_mcts,
            ),
            patch(
                "alpha_zero.envs.env_factory.create_game_env",
                return_value=mock_env,
            ),
        ):
            result = benchmark._play_single_game(model=model, algorithm="muzero", game_idx=0)

        assert result["length"] == 1
        assert result["reward"] == -1.0
        assert result["dynamics_calls"] == 2
        assert result["env_clones"] == 0

    def test_play_single_game_mcts_exception(self, benchmark):
        """When MCTS raises, the game should break early."""
        mock_env = MagicMock()
        mock_env.reset.return_value = None

        mock_mcts = MagicMock()
        mock_mcts.search.side_effect = RuntimeError("MCTS failure")

        model = MagicMock()

        with (
            patch(
                "alpha_zero.core.mcts_factory.create_mcts",
                return_value=mock_mcts,
            ),
            patch(
                "alpha_zero.envs.env_factory.create_game_env",
                return_value=mock_env,
            ),
        ):
            result = benchmark._play_single_game(model=model, algorithm="alphazero", game_idx=0)

        assert result["length"] == 0
        assert result["reward"] == 0.0

    # -- generate_report --
    def test_generate_report(self, benchmark, tmp_path):
        az = BenchmarkResult(
            algorithm="alphazero",
            num_games=2,
            win_rate=0.5,
            avg_game_length=10.0,
            search_metrics=SearchMetrics(total_env_clones=10, total_network_evals=20),
        )
        mz = BenchmarkResult(
            algorithm="muzero",
            num_games=2,
            win_rate=0.5,
            avg_game_length=12.0,
            search_metrics=SearchMetrics(total_dynamics_calls=15, total_network_evals=20),
        )
        report = ComparisonReport(
            config=benchmark.config,
            alphazero_result=az,
            muzero_result=mz,
            timestamp="2026-01-01",
            summary={"key": "value"},
        )

        result_dict = benchmark.generate_report(report, output_dir=str(tmp_path))

        assert (tmp_path / "benchmark_report.json").exists()
        assert (tmp_path / "benchmark_report.md").exists()
        assert result_dict["timestamp"] == "2026-01-01"

        with open(tmp_path / "benchmark_report.json") as f:
            loaded = json.load(f)
        assert loaded["summary"]["key"] == "value"

    def test_generate_report_default_output_dir(self, benchmark, tmp_path):
        """When output_dir is None, uses config.output_dir."""
        benchmark.config.output_dir = str(tmp_path / "default_out")
        az = BenchmarkResult(algorithm="alphazero", search_metrics=SearchMetrics())
        mz = BenchmarkResult(algorithm="muzero", search_metrics=SearchMetrics())
        report = ComparisonReport(
            config=benchmark.config,
            alphazero_result=az,
            muzero_result=mz,
            summary={},
        )
        benchmark.generate_report(report, output_dir=None)
        assert (tmp_path / "default_out" / "benchmark_report.json").exists()

    # -- _generate_markdown --
    def test_generate_markdown(self, benchmark):
        az = BenchmarkResult(
            algorithm="alphazero",
            num_games=2,
            wins=1,
            losses=0,
            draws=1,
            win_rate=0.5,
            avg_game_length=10.0,
            total_time_seconds=60.0,
            games_per_minute=2.0,
            search_metrics=SearchMetrics(
                total_env_clones=10,
                total_network_evals=20,
                avg_search_time_ms=100.0,
            ),
        )
        mz = BenchmarkResult(
            algorithm="muzero",
            num_games=2,
            wins=1,
            losses=1,
            draws=0,
            win_rate=0.5,
            avg_game_length=12.0,
            total_time_seconds=30.0,
            games_per_minute=4.0,
            search_metrics=SearchMetrics(
                total_dynamics_calls=15,
                total_network_evals=20,
                avg_search_time_ms=50.0,
            ),
        )
        report = ComparisonReport(
            config=BenchmarkConfig(num_games=2, mcts_simulations=10, board_size=8),
            alphazero_result=az,
            muzero_result=mz,
            timestamp="2026-01-01",
            summary={"search_efficiency_ratio": 1.5, "speed_ratio": 2.0, "win_rate_delta": 0.0},
        )
        md = benchmark._generate_markdown(report)
        assert "# AlphaZero vs MuZero Benchmark Report" in md
        assert "Win Rates" in md
        assert "Search Efficiency" in md
        assert "Performance" in md
        assert "Key Insights" in md

    # -- log_to_wandb --
    def test_log_to_wandb_not_available(self, benchmark):
        """When wandb is unavailable, should log warning and return."""
        report = ComparisonReport(
            config=benchmark.config,
            alphazero_result=BenchmarkResult(algorithm="alphazero"),
            muzero_result=BenchmarkResult(algorithm="muzero"),
        )
        with patch("alpha_zero.evaluation.muzero_vs_alphazero_benchmark._WANDB_AVAILABLE", False):
            benchmark.log_to_wandb(report)  # should not raise

    def test_log_to_wandb_disabled(self, benchmark):
        """When wandb_enabled is False, should skip."""
        benchmark.config.wandb_enabled = False
        report = ComparisonReport(
            config=benchmark.config,
            alphazero_result=BenchmarkResult(algorithm="alphazero"),
            muzero_result=BenchmarkResult(algorithm="muzero"),
        )
        with patch("alpha_zero.evaluation.muzero_vs_alphazero_benchmark._WANDB_AVAILABLE", True):
            benchmark.log_to_wandb(report)  # should not raise

    def test_log_to_wandb_success(self, benchmark):
        """When wandb is available and enabled, should log metrics."""
        benchmark.config.wandb_enabled = True
        az = BenchmarkResult(
            algorithm="alphazero",
            win_rate=0.5,
            avg_game_length=10.0,
            games_per_minute=2.0,
            search_metrics=SearchMetrics(total_env_clones=10),
        )
        mz = BenchmarkResult(
            algorithm="muzero",
            win_rate=0.6,
            avg_game_length=12.0,
            games_per_minute=3.0,
            search_metrics=SearchMetrics(total_dynamics_calls=15),
        )
        report = ComparisonReport(
            config=benchmark.config,
            alphazero_result=az,
            muzero_result=mz,
            timestamp="2026-01-01",
            summary={"alphazero_win_rate": 0.5, "muzero_win_rate": 0.6},
        )

        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.init.return_value = mock_run
        mock_wandb.Table.return_value = MagicMock()

        with (
            patch("alpha_zero.evaluation.muzero_vs_alphazero_benchmark._WANDB_AVAILABLE", True),
            patch("alpha_zero.evaluation.muzero_vs_alphazero_benchmark.wandb", mock_wandb),
        ):
            benchmark.log_to_wandb(report)

        mock_wandb.init.assert_called_once()
        assert mock_wandb.log.call_count >= 3  # summary + 2 algos + table
        mock_run.finish.assert_called_once()

    def test_log_to_wandb_exception(self, benchmark):
        """When wandb raises an exception, should log error and not crash."""
        benchmark.config.wandb_enabled = True
        report = ComparisonReport(
            config=benchmark.config,
            alphazero_result=BenchmarkResult(algorithm="alphazero"),
            muzero_result=BenchmarkResult(algorithm="muzero"),
            summary={"key": 1.0},
        )

        mock_wandb = MagicMock()
        mock_wandb.init.side_effect = RuntimeError("W&B init failed")

        with (
            patch("alpha_zero.evaluation.muzero_vs_alphazero_benchmark._WANDB_AVAILABLE", True),
            patch("alpha_zero.evaluation.muzero_vs_alphazero_benchmark.wandb", mock_wandb),
        ):
            benchmark.log_to_wandb(report)  # should not raise
