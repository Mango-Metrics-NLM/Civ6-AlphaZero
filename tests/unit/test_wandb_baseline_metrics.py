"""Tests for WandB baseline evaluation metrics logging."""

from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.evaluation.evaluation_pipeline import (
    EvaluationResult,
    GameResult,
    WinRateResult,
)
from alpha_zero.utils.wandb_logger import WandbConfig, WandbLogger


class TestWandbBaselineLogging:
    """Tests for baseline evaluation logging methods."""

    @pytest.fixture
    def mock_wandb(self):
        """Create a mock wandb module."""
        with patch("alpha_zero.utils.wandb_logger.wandb") as mock:
            mock.init.return_value = MagicMock()
            mock.log = MagicMock()
            mock.define_metric = MagicMock()
            yield mock

    @pytest.fixture
    def enabled_logger(self, mock_wandb):
        """Create an enabled WandB logger with mocked run."""
        config = WandbConfig(enabled=True, mode="disabled")
        logger = WandbLogger(config)
        logger._run = MagicMock()
        logger._initialized = True
        return logger

    @pytest.fixture
    def disabled_logger(self):
        """Create a disabled WandB logger."""
        config = WandbConfig(enabled=False)
        return WandbLogger(config)

    @pytest.fixture
    def sample_evaluation_result(self):
        """Create a sample EvaluationResult for testing."""
        win_rate = WinRateResult(wins=7, losses=2, draws=1, total_games=10)
        game_results = [
            GameResult(
                winner=0,
                num_turns=50,
                player0_cities=3,
                player1_cities=1,
                player0_units=5,
                player1_units=2,
                game_duration_seconds=1.5,
            )
            for _ in range(10)
        ]
        return EvaluationResult(
            win_rate=win_rate,
            game_results=game_results,
            avg_game_length=50.0,
            avg_cities_won=3.0,
            avg_cities_lost=1.0,
            avg_units_won=5.0,
            avg_units_lost=2.0,
            total_evaluation_time=15.0,
            value_predictions=[],
            value_correlation=0.85,
        )

    def test_log_baseline_evaluation_single_opponent(
        self, enabled_logger, sample_evaluation_result, mock_wandb
    ):
        """Should log metrics for a single opponent."""
        results = {"random": sample_evaluation_result}

        enabled_logger.log_baseline_evaluation(results, iteration=100)

        mock_wandb.log.assert_called_once()
        call_args = mock_wandb.log.call_args
        data = call_args[0][0]

        # Verify all expected metrics are present
        assert data["eval/random/win_rate"] == 70.0
        assert data["eval/random/loss_rate"] == 20.0
        assert data["eval/random/draw_rate"] == 10.0
        assert data["eval/random/wins"] == 7
        assert data["eval/random/losses"] == 2
        assert data["eval/random/draws"] == 1
        assert data["eval/random/games_played"] == 10
        assert data["eval/random/avg_game_length"] == 50.0
        assert data["eval/random/value_correlation"] == 0.85

    def test_log_baseline_evaluation_multiple_opponents(
        self, enabled_logger, sample_evaluation_result, mock_wandb
    ):
        """Should log metrics for multiple opponents."""
        results = {
            "random": sample_evaluation_result,
            "heuristic": sample_evaluation_result,
        }

        enabled_logger.log_baseline_evaluation(results, iteration=50)

        mock_wandb.log.assert_called_once()
        call_args = mock_wandb.log.call_args
        data = call_args[0][0]

        # Verify both opponents have metrics
        assert "eval/random/win_rate" in data
        assert "eval/heuristic/win_rate" in data
        assert data["eval/random/win_rate"] == 70.0
        assert data["eval/heuristic/win_rate"] == 70.0

    def test_log_baseline_evaluation_with_iteration(
        self, enabled_logger, sample_evaluation_result, mock_wandb
    ):
        """Should pass iteration as step parameter."""
        results = {"random": sample_evaluation_result}

        enabled_logger.log_baseline_evaluation(results, iteration=200)

        call_args = mock_wandb.log.call_args
        assert call_args[1]["step"] == 200

    def test_log_baseline_evaluation_disabled_logger(
        self, disabled_logger, sample_evaluation_result
    ):
        """Should not log when logger is disabled."""
        results = {"random": sample_evaluation_result}

        # Should not raise and should not call wandb
        disabled_logger.log_baseline_evaluation(results, iteration=100)

    def test_log_baseline_evaluation_empty_results(self, enabled_logger, mock_wandb):
        """Should handle empty results dictionary."""
        enabled_logger.log_baseline_evaluation({}, iteration=100)

        mock_wandb.log.assert_called_once()
        call_args = mock_wandb.log.call_args
        data = call_args[0][0]
        assert data == {}


class TestLogReflectionStats:
    """Tests for reflection memory statistics logging."""

    @pytest.fixture
    def mock_wandb(self):
        """Create a mock wandb module."""
        with patch("alpha_zero.utils.wandb_logger.wandb") as mock:
            mock.log = MagicMock()
            yield mock

    @pytest.fixture
    def enabled_logger(self, mock_wandb):
        """Create an enabled WandB logger."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._run = MagicMock()
        logger._initialized = True
        return logger

    def test_log_reflection_stats_minimal(self, enabled_logger, mock_wandb):
        """Should log with only required parameter."""
        enabled_logger.log_reflection_stats(total_entries=100)

        mock_wandb.log.assert_called_once()
        data = mock_wandb.log.call_args[0][0]
        assert data["reflection/total_entries"] == 100
        assert "reflection/unique_patterns" not in data

    def test_log_reflection_stats_all_params(self, enabled_logger, mock_wandb):
        """Should log all optional parameters when provided."""
        enabled_logger.log_reflection_stats(
            total_entries=100,
            unique_patterns=25,
            recent_additions=5,
            avg_critique_length=150.5,
            iteration=50,
        )

        data = mock_wandb.log.call_args[0][0]
        assert data["reflection/total_entries"] == 100
        assert data["reflection/unique_patterns"] == 25
        assert data["reflection/recent_additions"] == 5
        assert data["reflection/avg_critique_length"] == 150.5

    def test_log_reflection_stats_with_iteration(self, enabled_logger, mock_wandb):
        """Should pass iteration as step parameter."""
        enabled_logger.log_reflection_stats(total_entries=50, iteration=300)

        call_args = mock_wandb.log.call_args
        assert call_args[1]["step"] == 300


class TestLogImitationTraining:
    """Tests for imitation pre-training logging."""

    @pytest.fixture
    def mock_wandb(self):
        """Create a mock wandb module."""
        with patch("alpha_zero.utils.wandb_logger.wandb") as mock:
            mock.log = MagicMock()
            yield mock

    @pytest.fixture
    def enabled_logger(self, mock_wandb):
        """Create an enabled WandB logger."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._run = MagicMock()
        logger._initialized = True
        return logger

    def test_log_imitation_training_required_params(self, enabled_logger, mock_wandb):
        """Should log with required parameters."""
        enabled_logger.log_imitation_training(
            epoch=1,
            loss=0.5,
            policy_loss=0.3,
            value_loss=0.2,
        )

        data = mock_wandb.log.call_args[0][0]
        assert data["imitation/epoch"] == 1
        assert data["imitation/loss"] == 0.5
        assert data["imitation/policy_loss"] == 0.3
        assert data["imitation/value_loss"] == 0.2

    def test_log_imitation_training_all_params(self, enabled_logger, mock_wandb):
        """Should log all optional parameters when provided."""
        enabled_logger.log_imitation_training(
            epoch=5,
            loss=0.25,
            policy_loss=0.15,
            value_loss=0.10,
            policy_accuracy=0.85,
            value_mse=0.05,
            games_used=1000,
            samples_per_epoch=50000,
        )

        data = mock_wandb.log.call_args[0][0]
        assert data["imitation/epoch"] == 5
        assert data["imitation/policy_accuracy"] == 0.85
        assert data["imitation/value_mse"] == 0.05
        assert data["imitation/games_used"] == 1000
        assert data["imitation/samples_per_epoch"] == 50000

    def test_log_imitation_training_disabled(self):
        """Should not log when disabled."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)

        # Should not raise
        logger.log_imitation_training(epoch=1, loss=0.5, policy_loss=0.3, value_loss=0.2)


class TestDefineBaselineCharts:
    """Tests for WandB chart definition."""

    @pytest.fixture
    def mock_wandb(self):
        """Create a mock wandb module."""
        with patch("alpha_zero.utils.wandb_logger.wandb") as mock:
            mock.define_metric = MagicMock()
            yield mock

    @pytest.fixture
    def enabled_logger(self, mock_wandb):
        """Create an enabled WandB logger."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._run = MagicMock()
        logger._initialized = True
        return logger

    def test_define_baseline_charts_calls_define_metric(self, enabled_logger, mock_wandb):
        """Should call define_metric for all expected metrics."""
        enabled_logger.define_baseline_charts()

        # Verify define_metric was called multiple times
        assert mock_wandb.define_metric.call_count >= 10

    def test_define_baseline_charts_defines_iteration(self, enabled_logger, mock_wandb):
        """Should define iteration as a metric."""
        enabled_logger.define_baseline_charts()

        # Find the call that defines iteration
        calls = mock_wandb.define_metric.call_args_list
        iteration_calls = [c for c in calls if c[0][0] == "iteration"]
        assert len(iteration_calls) >= 1

    def test_define_baseline_charts_opponent_metrics(self, enabled_logger, mock_wandb):
        """Should define metrics for each opponent type."""
        enabled_logger.define_baseline_charts()

        calls = mock_wandb.define_metric.call_args_list
        metric_names = [c[0][0] for c in calls]

        # Check win rate for each opponent
        assert "eval/random/win_rate" in metric_names
        assert "eval/heuristic/win_rate" in metric_names
        assert "eval/checkpoint/win_rate" in metric_names

    def test_define_baseline_charts_disabled_logger(self):
        """Should not call define_metric when disabled."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)

        # Should not raise
        logger.define_baseline_charts()

    def test_define_baseline_charts_uninitialized(self, mock_wandb):
        """Should not define metrics when not initialized."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        # Not initialized

        logger.define_baseline_charts()

        mock_wandb.define_metric.assert_not_called()

    def test_define_baseline_charts_handles_exception(self, enabled_logger, mock_wandb):
        """Should handle exceptions gracefully."""
        mock_wandb.define_metric.side_effect = Exception("WandB error")

        # Should not raise
        enabled_logger.define_baseline_charts()


class TestIntegrationWithEvaluationPipeline:
    """Integration tests for WandB logging with evaluation pipeline."""

    @pytest.fixture
    def mock_wandb(self):
        """Create a mock wandb module."""
        with patch("alpha_zero.utils.wandb_logger.wandb") as mock:
            mock.log = MagicMock()
            yield mock

    def test_log_zero_win_rate(self, mock_wandb):
        """Should correctly log 0% win rate scenario."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._run = MagicMock()
        logger._initialized = True

        # Simulate 0% win rate (the critical issue we're tracking)
        win_rate = WinRateResult(wins=0, losses=10, draws=0, total_games=10)
        result = EvaluationResult(
            win_rate=win_rate,
            game_results=[],
            avg_game_length=30.0,
            value_correlation=0.0,  # No correlation with random values
        )

        logger.log_baseline_evaluation({"random": result}, iteration=1)

        data = mock_wandb.log.call_args[0][0]
        assert data["eval/random/win_rate"] == 0.0
        assert data["eval/random/value_correlation"] == 0.0

    def test_log_improving_win_rate(self, mock_wandb):
        """Should correctly log improving win rates over iterations."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._run = MagicMock()
        logger._initialized = True

        # Simulate win rate improving over training
        for iteration in [10, 20, 30]:
            wins = iteration // 10 * 3  # 3, 6, 9 wins
            win_rate = WinRateResult(wins=wins, losses=10 - wins, draws=0, total_games=10)
            result = EvaluationResult(
                win_rate=win_rate,
                game_results=[],
                avg_game_length=30.0,
                value_correlation=iteration / 100,
            )

            logger.log_baseline_evaluation({"random": result}, iteration=iteration)

        # Verify last call has highest win rate
        last_call_data = mock_wandb.log.call_args_list[-1][0][0]
        assert last_call_data["eval/random/win_rate"] == 90.0


class TestDisabledLoggerNoOps:
    """Verify disabled logger methods are no-ops."""

    @pytest.fixture
    def disabled_logger(self):
        """Create a disabled logger."""
        return WandbLogger(WandbConfig(enabled=False))

    def test_log_baseline_evaluation_noop(self, disabled_logger):
        """log_baseline_evaluation should be a no-op when disabled."""
        win_rate = WinRateResult(wins=5, losses=5, draws=0, total_games=10)
        result = EvaluationResult(win_rate=win_rate, game_results=[])

        # Should not raise
        disabled_logger.log_baseline_evaluation({"random": result})

    def test_log_reflection_stats_noop(self, disabled_logger):
        """log_reflection_stats should be a no-op when disabled."""
        disabled_logger.log_reflection_stats(total_entries=100)

    def test_log_imitation_training_noop(self, disabled_logger):
        """log_imitation_training should be a no-op when disabled."""
        disabled_logger.log_imitation_training(epoch=1, loss=0.5, policy_loss=0.3, value_loss=0.2)

    def test_define_baseline_charts_noop(self, disabled_logger):
        """define_baseline_charts should be a no-op when disabled."""
        disabled_logger.define_baseline_charts()
