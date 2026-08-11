"""End-to-end integration test for ``MLflowLogger`` against a real file backend.

Skipped when ``mlflow`` is not installed. Verifies that metrics, params, and
artifacts flow through the logger and land in a queryable form via
``MlflowClient``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow")
from mlflow.tracking import MlflowClient  # noqa: E402

from alpha_zero.monitoring import MLflowConfig, MLflowLogger  # noqa: E402

pytestmark = pytest.mark.integration


def _make_config(tmp_path: Path) -> MLflowConfig:
    return MLflowConfig(
        enabled=True,
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        experiment_name="civ6_e2e_test",
        run_name="e2e_run",
        tags={"suite": "integration"},
        nested_runs=False,  # keep test simple
    )


def test_logger_records_metrics_to_file_backend(tmp_path: Path):
    config = _make_config(tmp_path)
    logger = MLflowLogger(config)
    logger.init(config_dict={"lr": 0.001, "batch_size": 32})

    logger.log_metrics({"train/loss": 1.5}, step=0)
    logger.log_metrics({"train/loss": 1.2, "game/win_rate": 0.55}, step=1)
    logger.log_metrics({"train/loss": 0.9, "game/win_rate": 0.62}, step=2)
    logger.finish()

    client = MlflowClient(tracking_uri=config.tracking_uri)
    exp = client.get_experiment_by_name(config.experiment_name)
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) >= 1
    parent_run = runs[0]
    history = client.get_metric_history(parent_run.info.run_id, "train/loss")
    values = [m.value for m in history]
    assert 1.5 in values
    assert 1.2 in values
    assert 0.9 in values


def test_logger_records_evaluation_metrics(tmp_path: Path):
    config = _make_config(tmp_path)
    logger = MLflowLogger(config)
    logger.init()

    # Build a duck-typed EvaluationResult that the logger understands.
    eval_result = SimpleNamespace(
        win_rate=SimpleNamespace(win_rate=72.5, loss_rate=27.5, draw_rate=0.0),
        value_correlation=0.42,
        avg_game_length=85,
    )
    logger.log_evaluation({"random": eval_result, "heuristic": eval_result}, iteration=5)
    logger.finish()

    client = MlflowClient(tracking_uri=config.tracking_uri)
    exp = client.get_experiment_by_name(config.experiment_name)
    runs = client.search_runs([exp.experiment_id])
    run = runs[0]
    metric_keys = set(run.data.metrics.keys())
    assert "eval/random/win_rate" in metric_keys
    assert "eval/random/value_correlation" in metric_keys
    assert "eval/heuristic/win_rate" in metric_keys


def test_logger_records_elo_spread(tmp_path: Path):
    from alpha_zero.tournament.elo import RatingLeaderboard

    config = _make_config(tmp_path)
    logger = MLflowLogger(config)
    logger.init()

    leaderboard = RatingLeaderboard()
    leaderboard.update("current", 1520.0)
    leaderboard.update("previous_best", 1480.0)
    update = SimpleNamespace(
        player_id="current",
        rating_before=1500.0,
        rating_after=1520.0,
        rating_change=20.0,
    )
    logger.log_elo_update([update], leaderboard, iteration=1)
    logger.finish()

    client = MlflowClient(tracking_uri=config.tracking_uri)
    exp = client.get_experiment_by_name(config.experiment_name)
    runs = client.search_runs([exp.experiment_id])
    metrics = runs[0].data.metrics
    assert metrics["elo/spread"] == 40.0
    assert metrics["elo/current/rating"] == 1520.0


def test_logger_records_artifact(tmp_path: Path):
    config = _make_config(tmp_path)
    logger = MLflowLogger(config)
    logger.init()
    artifact_file = tmp_path / "report.txt"
    artifact_file.write_text("hello civ6")
    logger.log_artifact(artifact_file)
    logger.finish()

    client = MlflowClient(tracking_uri=config.tracking_uri)
    exp = client.get_experiment_by_name(config.experiment_name)
    runs = client.search_runs([exp.experiment_id])
    artifacts = client.list_artifacts(runs[0].info.run_id)
    assert any(a.path == "report.txt" for a in artifacts)


def test_logger_nested_runs(tmp_path: Path):
    config = _make_config(tmp_path)
    config.nested_runs = True
    logger = MLflowLogger(config)
    logger.init()
    logger.log_iteration(
        iteration=0,
        train_metrics={"train/loss": 1.0},
        game_metrics={"game/win_rate": 0.4},
    )
    logger.log_iteration(
        iteration=1,
        train_metrics={"train/loss": 0.5},
        game_metrics={"game/win_rate": 0.6},
    )
    logger.finish()

    client = MlflowClient(tracking_uri=config.tracking_uri)
    exp = client.get_experiment_by_name(config.experiment_name)
    runs = client.search_runs([exp.experiment_id])
    # Parent + 2 nested children = 3 runs total.
    assert len(runs) == 3
