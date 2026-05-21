"""
Tests for alpha_zero/monitoring/mlflow_logger.py.

Tests the MLflowLogger class including the new log_evaluation_result() method,
context manager lifecycle, parameter/metric logging, and graceful degradation.
"""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from alpha_zero.monitoring.mlflow_logger import MLflowLogger, _truncate, MLFLOW_MAX_PARAM_VALUE_LEN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_win_rate_result(
    wins: int = 4,
    losses: int = 14,
    draws: int = 2,
    total_games: int = 20,
) -> Any:
    """Create a mock WinRateResult matching the real EvaluationResult structure."""
    wr = MagicMock()
    wr.wins = wins
    wr.losses = losses
    wr.draws = draws
    wr.total_games = total_games
    wr.win_rate = wins / total_games * 100.0 if total_games > 0 else 0.0
    wr.draw_rate = draws / total_games * 100.0 if total_games > 0 else 0.0
    wr.loss_rate = losses / total_games * 100.0 if total_games > 0 else 0.0
    return wr


def _make_eval_result(
    wins: int = 4,
    losses: int = 14,
    draws: int = 2,
    total_games: int = 20,
    avg_game_length: float = 18.5,
    avg_cities_won: float = 1.2,
    avg_cities_lost: float = 0.1,
    avg_units_won: float = 2.0,
    avg_units_lost: float = 0.5,
    total_evaluation_time: float = 55.3,
    value_correlation: float = 0.207,
) -> Any:
    """Create a mock EvaluationResult with realistic values."""
    result = MagicMock()
    result.win_rate = _make_win_rate_result(wins, losses, draws, total_games)
    result.avg_game_length = avg_game_length
    result.avg_cities_won = avg_cities_won
    result.avg_cities_lost = avg_cities_lost
    result.avg_units_won = avg_units_won
    result.avg_units_lost = avg_units_lost
    result.total_evaluation_time = total_evaluation_time
    result.value_correlation = value_correlation
    result.summary = MagicMock(return_value="SUMMARY STRING (not a dict)")
    return result


@pytest.fixture()
def active_logger() -> MLflowLogger:
    """Return an enabled MLflowLogger with a mocked mlflow module."""
    ml = MLflowLogger(
        tracking_uri="mlruns",
        experiment_name="test-exp",
        run_name="test-run",
        enabled=True,
    )
    mock_mlflow = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id-123"
    mock_mlflow.start_run.return_value = mock_run
    ml._mlflow = mock_mlflow
    ml._run = mock_run
    ml.enabled = True
    return ml


@pytest.fixture()
def disabled_logger() -> MLflowLogger:
    """Return a disabled MLflowLogger (no-op)."""
    return MLflowLogger(enabled=False)


# ---------------------------------------------------------------------------
# TestTruncate
# ---------------------------------------------------------------------------


class TestTruncate:
    """Tests for _truncate()."""

    def test_short_string_unchanged(self) -> None:
        """Strings shorter than max_len should be returned unchanged."""
        assert _truncate("hello", max_len=10) == "hello"

    def test_exactly_max_len_unchanged(self) -> None:
        """String exactly at max_len should be returned unchanged."""
        s = "x" * 10
        assert _truncate(s, max_len=10) == s

    def test_long_string_truncated(self) -> None:
        """String longer than max_len should be truncated with ellipsis."""
        s = "x" * 20
        result = _truncate(s, max_len=10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_default_max_len(self) -> None:
        """Default max_len should be MLFLOW_MAX_PARAM_VALUE_LEN."""
        s = "a" * (MLFLOW_MAX_PARAM_VALUE_LEN + 10)
        result = _truncate(s)
        assert len(result) == MLFLOW_MAX_PARAM_VALUE_LEN


# ---------------------------------------------------------------------------
# TestMLflowLoggerLifecycle
# ---------------------------------------------------------------------------


class TestMLflowLoggerLifecycle:
    """Tests for MLflowLogger lifecycle: start_run, end_run, context manager."""

    def test_disabled_logger_start_run_noop(self, disabled_logger: MLflowLogger) -> None:
        """Disabled logger should be a no-op on start_run."""
        disabled_logger.start_run()
        assert disabled_logger._mlflow is None

    def test_context_manager_calls_start_and_end(self) -> None:
        """Context manager should call start_run and end_run."""
        ml = MLflowLogger(enabled=True)
        with patch("mlflow.set_tracking_uri"), \
             patch("mlflow.set_experiment"), \
             patch("mlflow.start_run") as mock_start, \
             patch("mlflow.end_run") as mock_end:
            mock_run = MagicMock()
            mock_run.info.run_id = "ctx-run-id"
            mock_start.return_value = mock_run
            with ml:
                pass
            mock_end.assert_called_once()

    def test_context_manager_sets_failed_on_exception(self) -> None:
        """__exit__ should call end_run(status='FAILED') when exception raised."""
        ml = MLflowLogger(enabled=True)
        mock_mlflow = MagicMock()
        # Directly set _mlflow and _run to bypass start_run() which would override our mock
        ml._mlflow = mock_mlflow
        ml._run = MagicMock()
        ml._run.info.run_id = "err-run"
        # Call __exit__ directly with exception type to simulate a raised exception
        ml.__exit__(ValueError, ValueError("test error"), None)
        mock_mlflow.end_run.assert_called_once_with(status="FAILED")

    def test_context_manager_sets_finished_on_success(self) -> None:
        """__exit__ should call end_run(status='FINISHED') when no exception."""
        ml = MLflowLogger(enabled=True)
        mock_mlflow = MagicMock()
        # Directly set _mlflow and _run to bypass start_run() which would override our mock
        ml._mlflow = mock_mlflow
        ml._run = MagicMock()
        # Call __exit__ with all-None args (no exception)
        ml.__exit__(None, None, None)
        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    def test_run_id_property(self, active_logger: MLflowLogger) -> None:
        """run_id property should return the active run's ID."""
        assert active_logger.run_id == "test-run-id-123"

    def test_run_id_none_when_no_run(self) -> None:
        """run_id should return None when no run is active."""
        ml = MLflowLogger()
        assert ml.run_id is None


# ---------------------------------------------------------------------------
# TestMLflowLoggerLogging
# ---------------------------------------------------------------------------


class TestMLflowLoggerLogging:
    """Tests for log_params, log_metrics, set_tag, log_artifact."""

    def test_log_params_stringifies_values(self, active_logger: MLflowLogger) -> None:
        """log_params should stringify all values before calling mlflow."""
        active_logger.log_params({"lr": 0.001, "batch": 256})
        active_logger._mlflow.log_params.assert_called_once()
        call_kwargs = active_logger._mlflow.log_params.call_args[0][0]
        assert call_kwargs["lr"] == "0.001"
        assert call_kwargs["batch"] == "256"

    def test_log_params_truncates_long_values(self, active_logger: MLflowLogger) -> None:
        """Long parameter values should be truncated."""
        long_val = "x" * 300
        active_logger.log_params({"long": long_val})
        call_kwargs = active_logger._mlflow.log_params.call_args[0][0]
        assert len(call_kwargs["long"]) <= MLFLOW_MAX_PARAM_VALUE_LEN

    def test_log_params_noop_when_disabled(self, disabled_logger: MLflowLogger) -> None:
        """Disabled logger should not call mlflow.log_params."""
        disabled_logger.log_params({"lr": 0.01})
        # No exception, no mlflow call

    def test_log_metrics_passes_values(self, active_logger: MLflowLogger) -> None:
        """log_metrics should forward the dict and step to mlflow."""
        active_logger.log_metrics({"loss": 1.23, "acc": 0.87}, step=5)
        active_logger._mlflow.log_metrics.assert_called_once_with(
            {"loss": 1.23, "acc": 0.87}, step=5
        )

    def test_log_metric_single(self, active_logger: MLflowLogger) -> None:
        """log_metric should call log_metrics with a single-key dict."""
        active_logger.log_metric("val_loss", 0.5, step=10)
        active_logger._mlflow.log_metrics.assert_called_once_with({"val_loss": 0.5}, step=10)

    def test_set_tag_called(self, active_logger: MLflowLogger) -> None:
        """set_tag should call mlflow.set_tag."""
        active_logger.set_tag("env", "test")
        active_logger._mlflow.set_tag.assert_called_once_with("env", "test")

    def test_log_artifact_warns_on_missing_file(
        self, active_logger: MLflowLogger, caplog: pytest.LogCaptureFixture
    ) -> None:
        """log_artifact should warn when file doesn't exist."""
        import logging

        with caplog.at_level(logging.WARNING):
            active_logger.log_artifact("/nonexistent/path/file.txt")
        assert any("not found" in r.message for r in caplog.records)

    def test_from_config_creates_logger(self) -> None:
        """from_config() should create an MLflowLogger with config values."""
        config = MagicMock()
        config.tracking_uri = "sqlite:///test.db"
        config.experiment_name = "my-exp"
        config.run_name = "my-run"
        config.tags = {"env": "test"}
        config.description = "desc"
        config.enabled = True
        config.registry_uri = None
        ml = MLflowLogger.from_config(config)
        assert ml.tracking_uri == "sqlite:///test.db"
        assert ml.experiment_name == "my-exp"
        assert ml.enabled is True


# ---------------------------------------------------------------------------
# TestLogEvaluationResult
# ---------------------------------------------------------------------------


class TestLogEvaluationResult:
    """Tests for log_evaluation_result() — the new method."""

    def test_logs_all_14_metrics(self, active_logger: MLflowLogger) -> None:
        """Should log 14 metrics: 7 win-rate + 7 stats."""
        result = _make_eval_result()
        logged = active_logger.log_evaluation_result(result, prefix="baseline/random")
        assert len(logged) == 14
        active_logger._mlflow.log_metrics.assert_called_once()

    def test_win_rate_normalised_to_fraction(self, active_logger: MLflowLogger) -> None:
        """win_rate in logged dict should be 0-1, not 0-100."""
        result = _make_eval_result(wins=10, losses=10, draws=0, total_games=20)
        logged = active_logger.log_evaluation_result(result, prefix="x")
        assert 0.0 <= logged["x/win_rate"] <= 1.0
        assert abs(logged["x/win_rate"] - 0.5) < 1e-9

    def test_win_counts_are_integers_as_float(self, active_logger: MLflowLogger) -> None:
        """Win/loss/draw counts should be stored as floats (4.0 not 4)."""
        result = _make_eval_result(wins=4, losses=14, draws=2, total_games=20)
        logged = active_logger.log_evaluation_result(result, prefix="x")
        assert logged["x/wins"] == 4.0
        assert logged["x/losses"] == 14.0
        assert logged["x/draws"] == 2.0
        assert logged["x/total_games"] == 20.0

    def test_all_seven_game_stats_logged(self, active_logger: MLflowLogger) -> None:
        """All 7 game-stat metrics should be logged."""
        result = _make_eval_result()
        logged = active_logger.log_evaluation_result(result, prefix="x")
        stat_keys = {
            "x/avg_game_length",
            "x/avg_cities_won",
            "x/avg_cities_lost",
            "x/avg_units_won",
            "x/avg_units_lost",
            "x/total_evaluation_time",
            "x/value_correlation",
        }
        assert stat_keys.issubset(set(logged.keys()))

    def test_step_passed_to_log_metrics(self, active_logger: MLflowLogger) -> None:
        """step argument should be forwarded to mlflow.log_metrics."""
        result = _make_eval_result()
        active_logger.log_evaluation_result(result, prefix="x", step=50)
        call_args = active_logger._mlflow.log_metrics.call_args
        assert call_args[1].get("step") == 50 or call_args[0][1] == 50

    def test_disabled_logger_returns_empty(self, disabled_logger: MLflowLogger) -> None:
        """Disabled logger should return empty dict and not call mlflow."""
        result = _make_eval_result()
        logged = disabled_logger.log_evaluation_result(result, prefix="x")
        assert logged == {}

    def test_no_win_rate_attr_logs_warning(
        self, active_logger: MLflowLogger, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should warn when result has no win_rate attribute."""
        import logging

        result = MagicMock(spec=["avg_game_length"])
        result.avg_game_length = 10.0
        with caplog.at_level(logging.WARNING):
            active_logger.log_evaluation_result(result, prefix="x")
        assert any("win_rate" in r.message for r in caplog.records)

    def test_default_prefix_is_baseline(self, active_logger: MLflowLogger) -> None:
        """Default prefix should be 'baseline'."""
        result = _make_eval_result()
        logged = active_logger.log_evaluation_result(result)
        for key in logged:
            assert key.startswith("baseline/"), f"Bad key: {key}"

    def test_custom_prefix_applied(self, active_logger: MLflowLogger) -> None:
        """Custom prefix should be applied to all metric keys."""
        result = _make_eval_result()
        logged = active_logger.log_evaluation_result(result, prefix="eval/v091/heuristic")
        for key in logged:
            assert key.startswith("eval/v091/heuristic/"), f"Bad key: {key}"

    def test_all_zero_result(self, active_logger: MLflowLogger) -> None:
        """Zero wins/losses/draws should log zeros without errors."""
        result = _make_eval_result(wins=0, losses=0, draws=0, total_games=0)
        # Make win_rate handle zero total_games
        result.win_rate.win_rate = 0.0
        result.win_rate.draw_rate = 0.0
        result.win_rate.loss_rate = 0.0
        logged = active_logger.log_evaluation_result(result, prefix="x")
        assert logged["x/wins"] == 0.0
        assert logged["x/win_rate"] == 0.0

    def test_returns_dict_passed_to_mlflow(self, active_logger: MLflowLogger) -> None:
        """The returned dict should match what was passed to mlflow.log_metrics."""
        result = _make_eval_result()
        logged = active_logger.log_evaluation_result(result, prefix="x")
        mlflow_call_kwargs = active_logger._mlflow.log_metrics.call_args[0][0]
        assert logged == mlflow_call_kwargs
