"""Tests for :mod:`alpha_zero.monitoring.mlflow_logger`.

Combines the test suites from PR #78 (config-object lifecycle / nested runs
/ ``log_iteration`` / ``log_elo_update``) and PR #79 (``log_evaluation_result``
14-metric breakdown / ``_truncate`` / context-manager lifecycle).

All MLflow API calls are mocked at the module level; one test forces
``MLFLOW_AVAILABLE = False`` to cover the optional-import branch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.monitoring.config import MLflowConfig
from alpha_zero.monitoring.mlflow_logger import (
    MLFLOW_MAX_PARAM_VALUE_LEN,
    MLflowLogger,
    _truncate,
    get_default_tracking_uri,
)

# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _enabled_config(**overrides: Any) -> MLflowConfig:
    """Build a fully-enabled :class:`MLflowConfig` suitable for unit tests."""
    base: dict[str, Any] = {
        "enabled": True,
        "tracking_uri": "file:///tmp/mlruns_unit",
        "experiment_name": "unit_test",
        "run_name": "run_1",
        "tags": {"a": "b"},
    }
    base.update(overrides)
    return MLflowConfig(**base)


@pytest.fixture()
def mock_mlflow() -> Any:
    """Patch every MLflow integration symbol in the logger module."""
    with (
        patch("alpha_zero.monitoring.mlflow_logger.mlflow") as m,
        patch("alpha_zero.monitoring.mlflow_logger.MlflowClient") as client_cls,
        patch("alpha_zero.monitoring.mlflow_logger.MLFLOW_AVAILABLE", True),
    ):
        run = MagicMock()
        run.info.run_id = "run-1234"
        m.start_run.return_value = run

        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        client_cls.return_value = client
        yield {"mlflow": m, "client_cls": client_cls, "client": client, "run": run}


def _make_win_rate_result(
    wins: int = 4,
    losses: int = 14,
    draws: int = 2,
    total_games: int = 20,
) -> Any:
    """Mock a ``WinRateResult`` matching the real :class:`EvaluationResult` shape."""
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
    """Mock an :class:`EvaluationResult` with the full 14-metric surface."""
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
def active_logger(mock_mlflow: Any) -> MLflowLogger:
    """Return a fully-initialised logger using the kwargs constructor.

    Exercises the PR #79 API surface (``MLflowLogger(tracking_uri=...,
    enabled=True)``) while still routing through ``init()`` so the mocked
    ``mlflow`` and ``MlflowClient`` symbols are wired in.
    """
    ml = MLflowLogger(
        tracking_uri="file:///tmp/mlruns_unit",
        experiment_name="test-exp",
        run_name="test-run",
        enabled=True,
    )
    ml.init()
    return ml


@pytest.fixture()
def disabled_logger() -> MLflowLogger:
    """Return a disabled logger that should be a no-op everywhere."""
    return MLflowLogger(enabled=False)


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    """Behaviour of the 250-character parameter-value truncator."""

    def test_short_string_unchanged(self) -> None:
        assert _truncate("hello", max_len=10) == "hello"

    def test_exactly_max_len_unchanged(self) -> None:
        s = "x" * 10
        assert _truncate(s, max_len=10) == s

    def test_long_string_truncated(self) -> None:
        s = "x" * 20
        result = _truncate(s, max_len=10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_default_max_len(self) -> None:
        s = "a" * (MLFLOW_MAX_PARAM_VALUE_LEN + 10)
        result = _truncate(s)
        assert len(result) == MLFLOW_MAX_PARAM_VALUE_LEN


# ---------------------------------------------------------------------------
# Disabled / unavailable paths
# ---------------------------------------------------------------------------


def test_logger_disabled_when_config_disabled() -> None:
    logger = MLflowLogger(MLflowConfig(enabled=False))
    assert logger.is_enabled is False
    assert logger.is_initialized is False
    assert logger.run is None
    assert logger.init() is None
    # Every public method must be a no-op.
    logger.log_metrics({"x": 1.0}, step=0)
    logger.log_params({"k": "v"})
    logger.log_iteration(0, {"a": 1.0}, {"b": 2.0}, {})
    logger.log_evaluation({}, 0)
    logger.log_elo_update([], MagicMock(), 0)
    logger.log_artifact("/tmp/x")
    logger.log_model_checkpoint("/tmp/x", 0)
    logger.log_text("hi", "f.txt")
    logger.set_tags({"a": "b"})
    logger.finish()  # safe before init


def test_logger_unavailable_when_mlflow_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "alpha_zero.monitoring.mlflow_logger.MLFLOW_AVAILABLE",
        False,
    )
    logger = MLflowLogger(_enabled_config())
    assert logger.is_enabled is False
    assert logger.init() is None
    assert logger.is_initialized is False


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_accepts_config_object() -> None:
    cfg = _enabled_config()
    logger = MLflowLogger(cfg)
    assert logger.config is cfg


def test_constructor_accepts_kwargs() -> None:
    logger = MLflowLogger(
        tracking_uri="file:///tmp/x",
        experiment_name="kw-exp",
        run_name="kw-run",
        enabled=True,
        tags={"k": "v"},
    )
    assert logger.config.tracking_uri == "file:///tmp/x"
    assert logger.config.experiment_name == "kw-exp"
    assert logger.config.enabled is True
    assert logger.config.tags == {"k": "v"}


def test_constructor_rejects_config_plus_kwargs() -> None:
    with pytest.raises(TypeError):
        MLflowLogger(_enabled_config(), tracking_uri="file:///x")


def test_enabled_alias_property() -> None:
    logger = MLflowLogger(MLflowConfig(enabled=False))
    assert logger.enabled is False
    logger.enabled = True  # backwards-compat setter
    assert logger.config.enabled is True


# ---------------------------------------------------------------------------
# init / lifecycle
# ---------------------------------------------------------------------------


def test_init_opens_run_and_logs_params(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init(config_dict={"lr": 0.001, "batch_size": 64})
    assert logger.is_initialized is True
    assert logger.run is not None
    mock_mlflow["mlflow"].set_tracking_uri.assert_called_once_with("file:///tmp/mlruns_unit")
    mock_mlflow["client"].create_experiment.assert_called_once()
    mock_mlflow["mlflow"].set_experiment.assert_called_once_with("unit_test")
    mock_mlflow["mlflow"].start_run.assert_called_once()
    mock_mlflow["mlflow"].log_params.assert_called_once()


def test_init_skips_create_experiment_when_existing(mock_mlflow: Any) -> None:
    mock_mlflow["client"].get_experiment_by_name.return_value = MagicMock()
    logger = MLflowLogger(_enabled_config())
    logger.init()
    mock_mlflow["client"].create_experiment.assert_not_called()


def test_init_idempotent(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.init()  # second call returns the cached run
    assert mock_mlflow["mlflow"].start_run.call_count == 1


def test_init_swallows_exception(mock_mlflow: Any) -> None:
    mock_mlflow["mlflow"].start_run.side_effect = RuntimeError("server down")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    assert logger.init() is None
    assert logger.is_initialized is False
    assert logger.config.enabled is False


def test_init_raises_when_fail_silently_false(mock_mlflow: Any) -> None:
    mock_mlflow["mlflow"].start_run.side_effect = RuntimeError("server down")
    logger = MLflowLogger(_enabled_config(fail_silently=False))
    with pytest.raises(RuntimeError):
        logger.init()


def test_init_warns_when_artifact_location_dropped_on_client_failure(
    mock_mlflow: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """If MlflowClient setup fails AND artifact_location was set, warn loudly.

    The fallback path uses ``mlflow.set_experiment(name)`` which cannot honor
    ``artifact_location``. A silent debug log would mask a real misconfig.
    """
    mock_mlflow["client_cls"].side_effect = RuntimeError("backend unreachable")
    logger = MLflowLogger(
        _enabled_config(
            fail_silently=True,
            artifact_location="gs://bucket/civ6-mlflow",
        )
    )
    with caplog.at_level(logging.WARNING):
        logger.init()
    assert any(
        "artifact_location" in r.message and "NOT be honored" in r.message for r in caplog.records
    )


def test_init_does_not_warn_when_artifact_location_unset(
    mock_mlflow: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """If artifact_location is unset, the client fallback is debug-only."""
    mock_mlflow["client_cls"].side_effect = RuntimeError("backend unreachable")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    with caplog.at_level(logging.WARNING):
        logger.init()
    assert not any("artifact_location" in r.message for r in caplog.records)


def test_run_id_property(active_logger: MLflowLogger) -> None:
    assert active_logger.run_id == "run-1234"


def test_run_id_none_when_no_run() -> None:
    assert MLflowLogger().run_id is None


def test_context_manager_calls_init_and_finish(mock_mlflow: Any) -> None:
    """``with logger:`` opens the run on enter and closes it on exit."""
    ml = MLflowLogger(_enabled_config())
    with ml:
        assert ml.is_initialized is True
    mock_mlflow["mlflow"].end_run.assert_called_once_with(status="FINISHED")


def test_context_manager_sets_failed_on_exception(mock_mlflow: Any) -> None:
    ml = MLflowLogger(_enabled_config())
    with pytest.raises(ValueError):
        with ml:
            raise ValueError("boom")
    mock_mlflow["mlflow"].end_run.assert_called_once_with(status="FAILED")


# ---------------------------------------------------------------------------
# log_metrics / log_metric / step_context
# ---------------------------------------------------------------------------


def test_log_metrics_success(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.log_metrics({"x": 1.5, "y": 2.0}, step=1)
    mock_mlflow["mlflow"].log_metrics.assert_called_once_with(
        {"x": 1.5, "y": 2.0},
        step=1,
    )


def test_log_metrics_drops_non_numeric(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.log_metrics({"x": 1.0, "bad": "nope", "none": None})
    args, _ = mock_mlflow["mlflow"].log_metrics.call_args
    assert args[0] == {"x": 1.0}


def test_log_metrics_drops_nan(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.log_metrics({"x": float("nan"), "y": 1.0})
    args, _ = mock_mlflow["mlflow"].log_metrics.call_args
    assert "x" not in args[0]
    assert args[0] == {"y": 1.0}


def test_log_metrics_swallows_exception(mock_mlflow: Any) -> None:
    mock_mlflow["mlflow"].log_metrics.side_effect = RuntimeError("net err")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    logger.log_metrics({"x": 1.0})  # must not raise


def test_log_metrics_raises_when_fail_silently_false(mock_mlflow: Any) -> None:
    mock_mlflow["mlflow"].log_metrics.side_effect = RuntimeError("net err")
    logger = MLflowLogger(_enabled_config(fail_silently=False))
    logger.init()
    with pytest.raises(RuntimeError):
        logger.log_metrics({"x": 1.0})


def test_log_metric_single(active_logger: MLflowLogger, mock_mlflow: Any) -> None:
    active_logger.log_metric("val_loss", 0.5, step=10)
    mock_mlflow["mlflow"].log_metrics.assert_called_with({"val_loss": 0.5}, step=10)


def test_step_context_attaches_implicit_step(active_logger: MLflowLogger, mock_mlflow: Any) -> None:
    with active_logger.step_context(step=42):
        active_logger.log_metric("loss", 1.0)
    mock_mlflow["mlflow"].log_metrics.assert_called_with({"loss": 1.0}, step=42)


# ---------------------------------------------------------------------------
# log_params
# ---------------------------------------------------------------------------


def test_log_params_stringifies(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    mock_mlflow["mlflow"].log_params.reset_mock()
    logger.log_params({"lr": 0.01, "name": "foo", "complex": [1, 2, 3]})
    args, _ = mock_mlflow["mlflow"].log_params.call_args
    assert args[0]["lr"] == "0.01"
    assert args[0]["name"] == "foo"
    assert "1" in args[0]["complex"]


def test_log_params_truncates_long_values(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    long_val = "x" * 300
    logger.log_params({"long": long_val})
    args, _ = mock_mlflow["mlflow"].log_params.call_args
    assert len(args[0]["long"]) <= MLFLOW_MAX_PARAM_VALUE_LEN


def test_log_params_swallows(mock_mlflow: Any) -> None:
    mock_mlflow["mlflow"].log_params.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    logger.log_params({"a": "b"})


# ---------------------------------------------------------------------------
# log_iteration
# ---------------------------------------------------------------------------


def test_log_iteration_emits_parent_and_nested_runs(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config(nested_runs=True))
    logger.init()
    logger.log_iteration(
        iteration=3,
        train_metrics={"train/loss": 1.5},
        game_metrics={"game/win_rate": 0.6},
        scheduler_metrics={"train/learning_rate": 0.001},
    )
    keys_logged = [kwargs["key"] for _, kwargs in mock_mlflow["client"].log_metric.call_args_list]
    assert "train/loss" in keys_logged
    assert "game/win_rate" in keys_logged
    assert "train/learning_rate" in keys_logged
    # Parent + nested means start_run is called at least twice.
    assert mock_mlflow["mlflow"].start_run.call_count >= 2


def test_log_iteration_skips_nested_when_disabled(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config(nested_runs=False))
    logger.init()
    mock_mlflow["mlflow"].start_run.reset_mock()
    logger.log_iteration(
        iteration=1,
        train_metrics={"train/loss": 1.0},
        game_metrics={"game/win_rate": 0.4},
    )
    mock_mlflow["mlflow"].start_run.assert_not_called()


def test_log_iteration_no_op_when_no_metrics(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config(nested_runs=False))
    logger.init()
    mock_mlflow["client"].log_metric.reset_mock()
    logger.log_iteration(iteration=0, train_metrics={}, game_metrics={})
    mock_mlflow["client"].log_metric.assert_not_called()


def test_log_iteration_swallows_nested_error(mock_mlflow: Any) -> None:
    def start_run_side_effect(*_: Any, **kwargs: Any) -> Any:
        if not kwargs.get("nested"):
            return mock_mlflow["run"]
        raise RuntimeError("nested broke")

    mock_mlflow["mlflow"].start_run.side_effect = start_run_side_effect
    logger = MLflowLogger(_enabled_config(nested_runs=True, fail_silently=True))
    logger.init()
    logger.log_iteration(0, {"train/loss": 1.0}, {}, {})  # no raise


# ---------------------------------------------------------------------------
# log_evaluation (dict of EvaluationResult)
# ---------------------------------------------------------------------------


def _eval_result_with_winrate(
    win_rate: float = 0.7,
    value_corr: float = 0.42,
    avg_game_length: float = 42,
) -> Any:
    wr = MagicMock()
    wr.win_rate = win_rate * 100  # WinRateResult stores percentage
    wr.loss_rate = (1 - win_rate) * 100
    wr.draw_rate = 0.0
    res = MagicMock()
    res.win_rate = wr
    res.value_correlation = value_corr
    res.avg_game_length = avg_game_length
    return res


def test_log_evaluation_emits_per_opponent_metrics(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    mock_mlflow["client"].log_metric.reset_mock()
    results = {
        "random": _eval_result_with_winrate(win_rate=0.8, value_corr=0.5),
        "heuristic": _eval_result_with_winrate(win_rate=0.55, value_corr=0.3),
    }
    logger.log_evaluation(results, iteration=5)
    keys = [kwargs["key"] for _, kwargs in mock_mlflow["client"].log_metric.call_args_list]
    assert "eval/random/win_rate" in keys
    assert "eval/random/value_correlation" in keys
    assert "eval/heuristic/win_rate" in keys


def test_log_evaluation_empty_results(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    mock_mlflow["client"].log_metric.reset_mock()
    logger.log_evaluation({}, iteration=0)
    mock_mlflow["client"].log_metric.assert_not_called()


def test_log_evaluation_handles_plain_float_win_rate(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    mock_mlflow["client"].log_metric.reset_mock()
    res = MagicMock()
    res.win_rate = 0.5  # plain float, not WinRateResult
    res.value_correlation = 0.0
    res.avg_game_length = 30
    logger.log_evaluation({"opp": res}, iteration=1)
    keys = [kwargs["key"] for _, kwargs in mock_mlflow["client"].log_metric.call_args_list]
    assert "eval/opp/win_rate" in keys


# ---------------------------------------------------------------------------
# log_evaluation_result (PR #79: 14-metric breakdown)
# ---------------------------------------------------------------------------


class TestLogEvaluationResult:
    """``log_evaluation_result`` normalises rates to 0-1 and logs 14 metrics."""

    def test_logs_all_14_metrics(self, active_logger: MLflowLogger, mock_mlflow: Any) -> None:
        result = _make_eval_result()
        logged = active_logger.log_evaluation_result(result, prefix="baseline/random")
        assert len(logged) == 14
        mock_mlflow["mlflow"].log_metrics.assert_called()

    def test_win_rate_normalised_to_fraction(self, active_logger: MLflowLogger) -> None:
        result = _make_eval_result(wins=10, losses=10, draws=0, total_games=20)
        logged = active_logger.log_evaluation_result(result, prefix="x")
        assert 0.0 <= logged["x/win_rate"] <= 1.0
        assert abs(logged["x/win_rate"] - 0.5) < 1e-9

    def test_win_counts_are_integers_as_float(self, active_logger: MLflowLogger) -> None:
        result = _make_eval_result(wins=4, losses=14, draws=2, total_games=20)
        logged = active_logger.log_evaluation_result(result, prefix="x")
        assert logged["x/wins"] == 4.0
        assert logged["x/losses"] == 14.0
        assert logged["x/draws"] == 2.0
        assert logged["x/total_games"] == 20.0

    def test_all_seven_game_stats_logged(self, active_logger: MLflowLogger) -> None:
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

    def test_step_passed_to_log_metrics(
        self, active_logger: MLflowLogger, mock_mlflow: Any
    ) -> None:
        result = _make_eval_result()
        active_logger.log_evaluation_result(result, prefix="x", step=50)
        call_args = mock_mlflow["mlflow"].log_metrics.call_args
        assert call_args.kwargs.get("step") == 50

    def test_disabled_logger_returns_empty(self, disabled_logger: MLflowLogger) -> None:
        result = _make_eval_result()
        logged = disabled_logger.log_evaluation_result(result, prefix="x")
        assert logged == {}

    def test_no_win_rate_attr_logs_warning(
        self,
        active_logger: MLflowLogger,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        result = MagicMock(spec=["avg_game_length"])
        result.avg_game_length = 10.0
        with caplog.at_level(logging.WARNING):
            active_logger.log_evaluation_result(result, prefix="x")
        assert any("win_rate" in r.message for r in caplog.records)

    def test_default_prefix_is_baseline(self, active_logger: MLflowLogger) -> None:
        result = _make_eval_result()
        logged = active_logger.log_evaluation_result(result)
        for key in logged:
            assert key.startswith("baseline/"), f"Bad key: {key}"

    def test_custom_prefix_applied(self, active_logger: MLflowLogger) -> None:
        result = _make_eval_result()
        logged = active_logger.log_evaluation_result(result, prefix="eval/v091/heuristic")
        for key in logged:
            assert key.startswith("eval/v091/heuristic/"), f"Bad key: {key}"

    def test_all_zero_result(self, active_logger: MLflowLogger) -> None:
        result = _make_eval_result(wins=0, losses=0, draws=0, total_games=0)
        result.win_rate.win_rate = 0.0
        result.win_rate.draw_rate = 0.0
        result.win_rate.loss_rate = 0.0
        logged = active_logger.log_evaluation_result(result, prefix="x")
        assert logged["x/wins"] == 0.0
        assert logged["x/win_rate"] == 0.0

    def test_returns_dict_passed_to_mlflow(
        self, active_logger: MLflowLogger, mock_mlflow: Any
    ) -> None:
        result = _make_eval_result()
        logged = active_logger.log_evaluation_result(result, prefix="x")
        mlflow_call = mock_mlflow["mlflow"].log_metrics.call_args
        assert logged == mlflow_call.args[0]


# ---------------------------------------------------------------------------
# log_elo_update
# ---------------------------------------------------------------------------


class _FakeLeaderboard:
    def __init__(self, ratings: dict[str, float]) -> None:
        self._ratings = ratings

    def get_top(self, n: int = 10) -> list[tuple[str, float, int]]:
        items = sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)
        return [(pid, rating, i + 1) for i, (pid, rating) in enumerate(items[:n])]


def test_log_elo_update(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    mock_mlflow["client"].log_metric.reset_mock()
    update = MagicMock()
    update.player_id = "current"
    update.rating_after = 1520.0
    update.rating_change = 20.0
    leaderboard = _FakeLeaderboard({"current": 1520.0, "previous_best": 1480.0})
    logger.log_elo_update([update], leaderboard, iteration=3)
    keys = [kwargs["key"] for _, kwargs in mock_mlflow["client"].log_metric.call_args_list]
    assert "elo/current/rating" in keys
    assert "elo/current/rating_change" in keys
    assert "elo/spread" in keys
    assert "elo/max" in keys


# ---------------------------------------------------------------------------
# log_artifact / log_model_checkpoint / log_text / set_tags
# ---------------------------------------------------------------------------


def test_log_artifact(tmp_path: Path, mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    f = tmp_path / "x.txt"
    f.write_text("hello")
    logger.log_artifact(f)
    mock_mlflow["mlflow"].log_artifact.assert_called_once_with(
        str(f),
        artifact_path=None,
    )


def test_log_artifact_warns_on_missing_file(
    active_logger: MLflowLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        active_logger.log_artifact("/nonexistent/path/file.txt")
    assert any("not found" in r.message for r in caplog.records)


def test_log_artifact_skipped_when_disabled_flag(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config(log_artifacts=False))
    logger.init()
    mock_mlflow["mlflow"].log_artifact.reset_mock()
    logger.log_artifact("/tmp/x")
    mock_mlflow["mlflow"].log_artifact.assert_not_called()


def test_log_model_checkpoint_missing_file_skips(tmp_path: Path, mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    mock_mlflow["mlflow"].log_artifact.reset_mock()
    logger.log_model_checkpoint(tmp_path / "missing.ckpt", iteration=5)
    mock_mlflow["mlflow"].log_artifact.assert_not_called()


def test_log_model_checkpoint(tmp_path: Path, mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    ckpt = tmp_path / "iter.ckpt"
    ckpt.write_text("weights")
    logger.log_model_checkpoint(ckpt, iteration=7, metrics={"win_rate": 0.6})
    _, kwargs = mock_mlflow["mlflow"].log_artifact.call_args
    assert kwargs["artifact_path"] == "checkpoints/iter_000007"


def test_log_model_checkpoint_skipped_when_log_models_false(
    tmp_path: Path,
    mock_mlflow: Any,
) -> None:
    logger = MLflowLogger(_enabled_config(log_models=False))
    logger.init()
    ckpt = tmp_path / "x.ckpt"
    ckpt.write_text("x")
    mock_mlflow["mlflow"].log_artifact.reset_mock()
    logger.log_model_checkpoint(ckpt, iteration=1)
    mock_mlflow["mlflow"].log_artifact.assert_not_called()


def test_log_text(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.log_text("hi", "summary.txt")
    mock_mlflow["mlflow"].log_text.assert_called_once_with("hi", "summary.txt")


def test_set_tag(active_logger: MLflowLogger, mock_mlflow: Any) -> None:
    active_logger.set_tag("env", "test")
    mock_mlflow["mlflow"].set_tag.assert_called_once_with("env", "test")


def test_set_tags(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.set_tags({"trial": "alpha", "seed": "42"})
    mock_mlflow["mlflow"].set_tags.assert_called_once()


# ---------------------------------------------------------------------------
# start_run + finish
# ---------------------------------------------------------------------------


def test_start_run_disabled() -> None:
    logger = MLflowLogger(MLflowConfig(enabled=False))
    assert logger.start_run("x") is None


def test_start_run_returns_handle(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    assert logger.start_run("nested", nested=True) is not None


def test_start_run_refuses_before_init(mock_mlflow: Any, caplog: pytest.LogCaptureFixture) -> None:
    """``start_run(run_name=...)`` without ``init()`` must NOT open an orphan run."""
    logger = MLflowLogger(_enabled_config())
    mock_mlflow["mlflow"].start_run.reset_mock()
    with caplog.at_level(logging.WARNING):
        result = logger.start_run("nested", nested=True)
    assert result is None
    # The PR #78 patched ``mlflow.start_run`` is the same symbol the guard
    # would otherwise call - it must not be invoked when no parent is open.
    mock_mlflow["mlflow"].start_run.assert_not_called()
    assert any("orphan" in r.message for r in caplog.records)


def test_start_run_swallows(mock_mlflow: Any) -> None:
    """After ``init()``, a backend error on nested ``start_run`` is swallowed."""
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    # Now make the nested call raise; init's own start_run already succeeded.
    mock_mlflow["mlflow"].start_run.side_effect = RuntimeError("boom")
    assert logger.start_run("nested", nested=True) is None


def test_finish_idempotent(mock_mlflow: Any) -> None:
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.finish()
    logger.finish()  # second call is a no-op
    mock_mlflow["mlflow"].end_run.assert_called_once()


def test_finish_swallows_exception(mock_mlflow: Any) -> None:
    mock_mlflow["mlflow"].end_run.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    logger.finish()  # no raise


def test_finish_raises_when_fail_silently_false(mock_mlflow: Any) -> None:
    mock_mlflow["mlflow"].end_run.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=False))
    logger.init()
    with pytest.raises(RuntimeError):
        logger.finish()


def test_end_run_alias_calls_finish(mock_mlflow: Any) -> None:
    """``end_run`` is the PR #79 alias for ``finish``."""
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.end_run("FINISHED")
    mock_mlflow["mlflow"].end_run.assert_called_once_with(status="FINISHED")


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def test_from_env_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_MLFLOW_ENABLED", "false")
    logger = MLflowLogger.from_env()
    assert logger.config.enabled is False


def test_from_config_creates_logger() -> None:
    cfg = _enabled_config()
    logger = MLflowLogger.from_config(cfg)
    assert logger.config is cfg


def test_get_default_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_MLFLOW_TRACKING_URI", "file:///x")
    assert get_default_tracking_uri() == "file:///x"
    monkeypatch.delenv("CIV6AZ_MLFLOW_TRACKING_URI")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///std")
    assert get_default_tracking_uri() == "file:///std"


# ---------------------------------------------------------------------------
# Parent-run client.log_metric error handling
# ---------------------------------------------------------------------------


def test_parent_run_log_metric_swallows(mock_mlflow: Any) -> None:
    mock_mlflow["client"].log_metric.side_effect = RuntimeError("connection lost")
    logger = MLflowLogger(_enabled_config(nested_runs=False, fail_silently=True))
    logger.init()
    logger.log_iteration(0, {"train/loss": 1.0}, {}, {})  # must not raise


def test_parent_run_log_metric_raises_when_fail_silently_false(
    mock_mlflow: Any,
) -> None:
    mock_mlflow["client"].log_metric.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(nested_runs=False, fail_silently=False))
    logger.init()
    with pytest.raises(RuntimeError):
        logger.log_iteration(0, {"train/loss": 1.0}, {}, {})


def test_log_artifact_swallows(tmp_path: Path, mock_mlflow: Any) -> None:
    mock_mlflow["mlflow"].log_artifact.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    f = tmp_path / "f.txt"
    f.write_text("hi")
    logger.log_artifact(f)


def test_log_text_swallows(mock_mlflow: Any) -> None:
    mock_mlflow["mlflow"].log_text.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    logger.log_text("hi", "f.txt")


def test_set_tags_swallows(mock_mlflow: Any) -> None:
    mock_mlflow["mlflow"].set_tags.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    logger.set_tags({"a": "b"})


def test_log_model_checkpoint_swallows(tmp_path: Path, mock_mlflow: Any) -> None:
    mock_mlflow["mlflow"].log_artifact.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    ckpt = tmp_path / "x.ckpt"
    ckpt.write_text("x")
    logger.log_model_checkpoint(ckpt, iteration=1, metrics={"win": 0.5})
