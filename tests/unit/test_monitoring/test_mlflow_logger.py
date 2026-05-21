"""Tests for ``alpha_zero.monitoring.mlflow_logger``.

The MLflow client is mocked everywhere. One test forces
``MLFLOW_AVAILABLE = False`` to cover the optional-import branch.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.monitoring.config import MLflowConfig
from alpha_zero.monitoring.mlflow_logger import (
    MLflowLogger,
    get_default_tracking_uri,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _enabled_config(**overrides):
    base = {
        "enabled": True,
        "tracking_uri": "file:///tmp/mlruns_unit",
        "experiment_name": "unit_test",
        "run_name": "run_1",
        "tags": {"a": "b"},
    }
    base.update(overrides)
    return MLflowConfig(**base)


@pytest.fixture
def mock_mlflow():
    """Patch the ``mlflow`` symbol in mlflow_logger to a MagicMock."""
    with (
        patch("alpha_zero.monitoring.mlflow_logger.mlflow") as m,
        patch("alpha_zero.monitoring.mlflow_logger.MlflowClient") as client_cls,
        patch("alpha_zero.monitoring.mlflow_logger.MLFLOW_AVAILABLE", True),
    ):
        # Configure ``start_run`` to return an object with .info.run_id
        run = MagicMock()
        run.info.run_id = "run-1234"
        m.start_run.return_value = run
        # Make nested start_run usable as a context manager.
        nested_ctx = MagicMock()
        nested_ctx.__enter__ = MagicMock(return_value=run)
        nested_ctx.__exit__ = MagicMock(return_value=False)
        # When start_run is called WITH ``nested=True`` we want it to behave
        # like a context manager. The same MagicMock works because MagicMock
        # auto-provides __enter__/__exit__.
        client = MagicMock()
        client.get_experiment_by_name.return_value = None
        client_cls.return_value = client
        yield {"mlflow": m, "client_cls": client_cls, "client": client, "run": run}


# ---------------------------------------------------------------------
# Disabled / unavailable paths
# ---------------------------------------------------------------------


def test_logger_disabled_when_config_disabled():
    logger = MLflowLogger(MLflowConfig(enabled=False))
    assert logger.is_enabled is False
    assert logger.is_initialized is False
    assert logger.run is None
    assert logger.init() is None
    # All public methods must be no-ops:
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


def test_logger_unavailable_when_mlflow_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "alpha_zero.monitoring.mlflow_logger.MLFLOW_AVAILABLE",
        False,
    )
    logger = MLflowLogger(_enabled_config())
    assert logger.is_enabled is False
    assert logger.init() is None
    assert logger.is_initialized is False


# ---------------------------------------------------------------------
# init / lifecycle
# ---------------------------------------------------------------------


def test_init_opens_run_and_logs_params(mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init(config_dict={"lr": 0.001, "batch_size": 64})
    assert logger.is_initialized is True
    assert logger.run is not None
    mock_mlflow["mlflow"].set_tracking_uri.assert_called_once_with("file:///tmp/mlruns_unit")
    mock_mlflow["client"].create_experiment.assert_called_once()
    mock_mlflow["mlflow"].set_experiment.assert_called_once_with("unit_test")
    mock_mlflow["mlflow"].start_run.assert_called_once()
    mock_mlflow["mlflow"].log_params.assert_called_once()


def test_init_skips_create_experiment_when_existing(mock_mlflow):
    mock_mlflow["client"].get_experiment_by_name.return_value = MagicMock()
    logger = MLflowLogger(_enabled_config())
    logger.init()
    mock_mlflow["client"].create_experiment.assert_not_called()


def test_init_idempotent(mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.init()  # second call returns the cached run
    assert mock_mlflow["mlflow"].start_run.call_count == 1


def test_init_swallows_exception(mock_mlflow):
    mock_mlflow["mlflow"].start_run.side_effect = RuntimeError("server down")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    assert logger.init() is None
    assert logger.is_initialized is False
    # And the config gets disabled so further calls are no-ops.
    assert logger.config.enabled is False


def test_init_raises_when_fail_silently_false(mock_mlflow):
    mock_mlflow["mlflow"].start_run.side_effect = RuntimeError("server down")
    logger = MLflowLogger(_enabled_config(fail_silently=False))
    with pytest.raises(RuntimeError):
        logger.init()


# ---------------------------------------------------------------------
# log_metrics
# ---------------------------------------------------------------------


def test_log_metrics_success(mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.log_metrics({"x": 1.5, "y": 2.0}, step=1)
    mock_mlflow["mlflow"].log_metrics.assert_called_once_with(
        {"x": 1.5, "y": 2.0},
        step=1,
    )


def test_log_metrics_drops_non_numeric(mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.log_metrics({"x": 1.0, "bad": "nope", "none": None})
    args, kwargs = mock_mlflow["mlflow"].log_metrics.call_args
    assert args[0] == {"x": 1.0}


def test_log_metrics_drops_nan(mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.log_metrics({"x": float("nan"), "y": 1.0})
    args, kwargs = mock_mlflow["mlflow"].log_metrics.call_args
    assert "x" not in args[0]
    assert args[0] == {"y": 1.0}


def test_log_metrics_swallows_exception(mock_mlflow):
    mock_mlflow["mlflow"].log_metrics.side_effect = RuntimeError("net err")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    logger.log_metrics({"x": 1.0})  # must not raise


def test_log_metrics_raises_when_fail_silently_false(mock_mlflow):
    mock_mlflow["mlflow"].log_metrics.side_effect = RuntimeError("net err")
    logger = MLflowLogger(_enabled_config(fail_silently=False))
    logger.init()
    with pytest.raises(RuntimeError):
        logger.log_metrics({"x": 1.0})


# ---------------------------------------------------------------------
# log_params
# ---------------------------------------------------------------------


def test_log_params_stringifies(mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    mock_mlflow["mlflow"].log_params.reset_mock()
    logger.log_params({"lr": 0.01, "name": "foo", "complex": [1, 2, 3]})
    args, kwargs = mock_mlflow["mlflow"].log_params.call_args
    assert args[0]["lr"] == "0.01"
    assert args[0]["name"] == "foo"
    assert "1" in args[0]["complex"]


def test_log_params_swallows(mock_mlflow):
    mock_mlflow["mlflow"].log_params.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    logger.log_params({"a": "b"})


# ---------------------------------------------------------------------
# log_iteration
# ---------------------------------------------------------------------


def test_log_iteration_emits_parent_and_nested_runs(mock_mlflow):
    logger = MLflowLogger(_enabled_config(nested_runs=True))
    logger.init()
    logger.log_iteration(
        iteration=3,
        train_metrics={"train/loss": 1.5},
        game_metrics={"game/win_rate": 0.6},
        scheduler_metrics={"train/learning_rate": 0.001},
    )
    # Parent run path uses client.log_metric (one call per key).
    keys_logged = [kwargs["key"] for _, kwargs in mock_mlflow["client"].log_metric.call_args_list]
    assert "train/loss" in keys_logged
    assert "game/win_rate" in keys_logged
    assert "train/learning_rate" in keys_logged
    # Nested run: a SECOND start_run call (parent + nested).
    assert mock_mlflow["mlflow"].start_run.call_count >= 2


def test_log_iteration_skips_nested_when_disabled(mock_mlflow):
    logger = MLflowLogger(_enabled_config(nested_runs=False))
    logger.init()
    mock_mlflow["mlflow"].start_run.reset_mock()
    logger.log_iteration(
        iteration=1,
        train_metrics={"train/loss": 1.0},
        game_metrics={"game/win_rate": 0.4},
    )
    mock_mlflow["mlflow"].start_run.assert_not_called()


def test_log_iteration_no_op_when_no_metrics(mock_mlflow):
    logger = MLflowLogger(_enabled_config(nested_runs=False))
    logger.init()
    mock_mlflow["client"].log_metric.reset_mock()
    logger.log_iteration(iteration=0, train_metrics={}, game_metrics={})
    mock_mlflow["client"].log_metric.assert_not_called()


def test_log_iteration_swallows_nested_error(mock_mlflow):
    # First start_run is the parent (init), then subsequent calls raise.
    def start_run_side_effect(*args, **kwargs):
        if not kwargs.get("nested"):
            return mock_mlflow["run"]
        raise RuntimeError("nested broke")

    mock_mlflow["mlflow"].start_run.side_effect = start_run_side_effect
    logger = MLflowLogger(_enabled_config(nested_runs=True, fail_silently=True))
    logger.init()
    logger.log_iteration(0, {"train/loss": 1.0}, {}, {})  # no raise


# ---------------------------------------------------------------------
# log_evaluation
# ---------------------------------------------------------------------


def _make_eval_result(win_rate=0.7, value_corr=0.42, avg_game_length=42):
    wr = MagicMock()
    wr.win_rate = win_rate * 100  # WinRateResult stores percentage
    wr.loss_rate = (1 - win_rate) * 100
    wr.draw_rate = 0.0
    res = MagicMock()
    res.win_rate = wr
    res.value_correlation = value_corr
    res.avg_game_length = avg_game_length
    return res


def test_log_evaluation_emits_per_opponent_metrics(mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    mock_mlflow["client"].log_metric.reset_mock()
    results = {
        "random": _make_eval_result(win_rate=0.8, value_corr=0.5),
        "heuristic": _make_eval_result(win_rate=0.55, value_corr=0.3),
    }
    logger.log_evaluation(results, iteration=5)
    keys = [kwargs["key"] for _, kwargs in mock_mlflow["client"].log_metric.call_args_list]
    assert "eval/random/win_rate" in keys
    assert "eval/random/value_correlation" in keys
    assert "eval/heuristic/win_rate" in keys


def test_log_evaluation_empty_results(mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    mock_mlflow["client"].log_metric.reset_mock()
    logger.log_evaluation({}, iteration=0)
    mock_mlflow["client"].log_metric.assert_not_called()


def test_log_evaluation_handles_plain_float_win_rate(mock_mlflow):
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


# ---------------------------------------------------------------------
# log_elo_update
# ---------------------------------------------------------------------


class _FakeLeaderboard:
    def __init__(self, ratings):
        self._ratings = ratings

    def get_top(self, n=10):
        items = sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)
        return [(pid, rating, i + 1) for i, (pid, rating) in enumerate(items[:n])]


def test_log_elo_update(mock_mlflow):
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


# ---------------------------------------------------------------------
# log_artifact / log_model_checkpoint / log_text / set_tags
# ---------------------------------------------------------------------


def test_log_artifact(tmp_path: Path, mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    f = tmp_path / "x.txt"
    f.write_text("hello")
    logger.log_artifact(f)
    mock_mlflow["mlflow"].log_artifact.assert_called_once_with(
        str(f),
        artifact_path=None,
    )


def test_log_artifact_skipped_when_disabled_flag(mock_mlflow):
    logger = MLflowLogger(_enabled_config(log_artifacts=False))
    logger.init()
    mock_mlflow["mlflow"].log_artifact.reset_mock()
    logger.log_artifact("/tmp/x")
    mock_mlflow["mlflow"].log_artifact.assert_not_called()


def test_log_model_checkpoint_missing_file_skips(tmp_path: Path, mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    mock_mlflow["mlflow"].log_artifact.reset_mock()
    logger.log_model_checkpoint(tmp_path / "missing.ckpt", iteration=5)
    mock_mlflow["mlflow"].log_artifact.assert_not_called()


def test_log_model_checkpoint(tmp_path: Path, mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    ckpt = tmp_path / "iter.ckpt"
    ckpt.write_text("weights")
    logger.log_model_checkpoint(ckpt, iteration=7, metrics={"win_rate": 0.6})
    args, kwargs = mock_mlflow["mlflow"].log_artifact.call_args
    assert kwargs["artifact_path"] == "checkpoints/iter_000007"


def test_log_model_checkpoint_skipped_when_log_models_false(
    tmp_path: Path,
    mock_mlflow,
):
    logger = MLflowLogger(_enabled_config(log_models=False))
    logger.init()
    ckpt = tmp_path / "x.ckpt"
    ckpt.write_text("x")
    mock_mlflow["mlflow"].log_artifact.reset_mock()
    logger.log_model_checkpoint(ckpt, iteration=1)
    mock_mlflow["mlflow"].log_artifact.assert_not_called()


def test_log_text(mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.log_text("hi", "summary.txt")
    mock_mlflow["mlflow"].log_text.assert_called_once_with("hi", "summary.txt")


def test_set_tags(mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.set_tags({"trial": "alpha", "seed": "42"})
    mock_mlflow["mlflow"].set_tags.assert_called_once()


# ---------------------------------------------------------------------
# start_run + finish
# ---------------------------------------------------------------------


def test_start_run_disabled():
    logger = MLflowLogger(MLflowConfig(enabled=False))
    assert logger.start_run("x") is None


def test_start_run_returns_handle(mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    assert logger.start_run("nested", nested=True) is not None


def test_start_run_swallows(mock_mlflow):
    mock_mlflow["mlflow"].start_run.side_effect = RuntimeError("boom")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    # init failed → still safe to call start_run; returns None.
    assert logger.start_run("x") is None


def test_finish_idempotent(mock_mlflow):
    logger = MLflowLogger(_enabled_config())
    logger.init()
    logger.finish()
    logger.finish()  # second call is a no-op
    mock_mlflow["mlflow"].end_run.assert_called_once()


def test_finish_swallows_exception(mock_mlflow):
    mock_mlflow["mlflow"].end_run.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    logger.finish()  # no raise


def test_finish_raises_when_fail_silently_false(mock_mlflow):
    mock_mlflow["mlflow"].end_run.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=False))
    logger.init()
    with pytest.raises(RuntimeError):
        logger.finish()


# ---------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------


def test_from_env_construction(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CIV6AZ_MLFLOW_ENABLED", "false")
    logger = MLflowLogger.from_env()
    assert logger.config.enabled is False


def test_get_default_tracking_uri(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CIV6AZ_MLFLOW_TRACKING_URI", "file:///x")
    assert get_default_tracking_uri() == "file:///x"
    monkeypatch.delenv("CIV6AZ_MLFLOW_TRACKING_URI")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///std")
    assert get_default_tracking_uri() == "file:///std"


# ---------------------------------------------------------------------
# Parent-run client.log_metric error handling
# ---------------------------------------------------------------------


def test_parent_run_log_metric_swallows(mock_mlflow):
    mock_mlflow["client"].log_metric.side_effect = RuntimeError("connection lost")
    logger = MLflowLogger(_enabled_config(nested_runs=False, fail_silently=True))
    logger.init()
    # Must not raise even though the client raises on every metric.
    logger.log_iteration(0, {"train/loss": 1.0}, {}, {})


def test_parent_run_log_metric_raises_when_fail_silently_false(mock_mlflow):
    mock_mlflow["client"].log_metric.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(nested_runs=False, fail_silently=False))
    logger.init()
    with pytest.raises(RuntimeError):
        logger.log_iteration(0, {"train/loss": 1.0}, {}, {})


def test_log_artifact_swallows(tmp_path: Path, mock_mlflow):
    mock_mlflow["mlflow"].log_artifact.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    f = tmp_path / "f.txt"
    f.write_text("hi")
    logger.log_artifact(f)  # must not raise


def test_log_text_swallows(mock_mlflow):
    mock_mlflow["mlflow"].log_text.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    logger.log_text("hi", "f.txt")


def test_set_tags_swallows(mock_mlflow):
    mock_mlflow["mlflow"].set_tags.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    logger.set_tags({"a": "b"})


def test_log_model_checkpoint_swallows(tmp_path: Path, mock_mlflow):
    mock_mlflow["mlflow"].log_artifact.side_effect = RuntimeError("x")
    logger = MLflowLogger(_enabled_config(fail_silently=True))
    logger.init()
    ckpt = tmp_path / "x.ckpt"
    ckpt.write_text("x")
    logger.log_model_checkpoint(ckpt, iteration=1, metrics={"win": 0.5})
