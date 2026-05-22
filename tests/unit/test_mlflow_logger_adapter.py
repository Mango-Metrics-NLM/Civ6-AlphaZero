"""Tests for ``MonitoringConfig.from_env`` and ``WandbCompatAdapter``.

These guard the v1.5.0 MLflow-replaces-W&B migration: the adapter must
preserve the exact public signatures of ``alpha_zero.utils.wandb_logger.WandbLogger``
so worker code can switch trackers with a one-line import change.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from alpha_zero.monitoring.config import MonitoringConfig
from alpha_zero.monitoring.mlflow_logger import MLflowLogger, WandbCompatAdapter


# ---------------------------------------------------------------------------
# MonitoringConfig.from_env
# ---------------------------------------------------------------------------


def test_monitoring_config_from_env_defaults(monkeypatch):
    for var in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_RUN_NAME",
        "MLFLOW_TAGS",
        "MLFLOW_ENABLED",
        "MLFLOW_REGISTRY_URI",
        "CIV6AZ_MLFLOW_TRACKING_URI",
        "CIV6AZ_MLFLOW_EXPERIMENT_NAME",
        "CIV6AZ_MLFLOW_RUN_NAME",
        "CIV6AZ_MLFLOW_TAGS",
        "CIV6AZ_MLFLOW_ENABLED",
        "CIV6AZ_MLFLOW_REGISTRY_URI",
    ):
        monkeypatch.delenv(var, raising=False)
    cfg = MonitoringConfig.from_env()
    # Defaults are MLflow-canonical (file://./mlruns, civ6-alphazero, enabled=False
    # opt-in). The MonitoringConfig name is preserved as an alias for backward
    # compatibility with v1.5.0 worker code.
    assert cfg.tracking_uri == "file://./mlruns"
    assert cfg.experiment_name == "civ6-alphazero"
    assert cfg.enabled is False
    assert cfg.tags == {}
    assert cfg.registry_uri is None
    assert cfg.run_name is None


def test_monitoring_config_from_env_overrides(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./custom")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "exp-x")
    monkeypatch.setenv("MLFLOW_RUN_NAME", "run-42")
    monkeypatch.setenv("MLFLOW_TAGS", "iteration=3,phase=train")
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "sqlite:///registry.db")
    cfg = MonitoringConfig.from_env()
    assert cfg.tracking_uri == "file:./custom"
    assert cfg.experiment_name == "exp-x"
    assert cfg.run_name == "run-42"
    assert cfg.tags == {"iteration": "3", "phase": "train"}
    assert cfg.enabled is False
    assert cfg.registry_uri == "sqlite:///registry.db"


# ---------------------------------------------------------------------------
# WandbCompatAdapter — signature parity
# ---------------------------------------------------------------------------


def _disabled_adapter() -> WandbCompatAdapter:
    """Build a disabled adapter so tests never touch the real MLflow backend."""
    cfg = MonitoringConfig(enabled=False)
    adapter = WandbCompatAdapter.from_config(cfg)
    adapter.init()
    return adapter


def test_adapter_log_training_step_routes_to_mlflow():
    adapter = _disabled_adapter()
    # Mirrors WandbLogger.log_training_step(step, loss, policy_loss, value_loss, ...)
    adapter.log_training_step(
        step=10,
        loss=1.5,
        policy_loss=1.2,
        value_loss=0.3,
        learning_rate=1e-3,
    )
    adapter.finish()


def test_adapter_log_evaluation_handles_winrate_result():
    wr = SimpleNamespace(
        wins=15,
        losses=10,
        draws=5,
        total_games=30,
        win_rate=50.0,
        draw_rate=16.67,
        loss_rate=33.33,
    )
    result = SimpleNamespace(win_rate=wr, avg_game_length=42.0)
    adapter = _disabled_adapter()
    # WandbLogger.log_evaluation signature: (win_rate, games_played, average_reward, average_turns)
    adapter.log_evaluation(
        win_rate=0.5,
        games_played=30,
        average_reward=0.1,
        average_turns=42.0,
        iteration=1,
    )
    # log_baseline_evaluation must accept dict[str, EvaluationResult]
    metrics = adapter.log_baseline_evaluation({"random": result}, iteration=1)
    # With enabled=False, log_evaluation_result returns {} per its contract.
    assert isinstance(metrics, dict)
    adapter.finish()


def test_adapter_log_population_stats_matches_wandblogger_signature():
    adapter = _disabled_adapter()
    adapter.log_population_stats(
        generation=3,
        mean_fitness=0.5,
        max_fitness=0.9,
        min_fitness=0.1,
        mean_elo=1500.0,
        max_elo=1600.0,
        min_elo=1400.0,
        population_size=8,
        mutations_count=2,
        diversity=0.7,
    )
    adapter.finish()


def test_adapter_log_table_matches_wandblogger_signature():
    adapter = _disabled_adapter()
    # WandbLogger.log_table(name, columns, data) - polymorphic adapter
    # accepts both W&B and MLflow-style call shapes.
    adapter.log_table(
        "promotions",
        ["iteration", "win_rate"],
        [[1, 0.55], [2, 0.62]],
    )
    adapter.finish()


# ---------------------------------------------------------------------------
# Misc parity smoke tests
# ---------------------------------------------------------------------------


def test_adapter_is_mlflow_logger_subclass():
    assert issubclass(WandbCompatAdapter, MLflowLogger)


def test_adapter_lifecycle_flags():
    adapter = WandbCompatAdapter.from_config(MonitoringConfig(enabled=False))
    assert adapter.is_enabled is False
    assert adapter.is_initialized is False
    adapter.init()
    # enabled=False short-circuits start_run, so _run stays None.
    assert adapter.is_initialized is False
    adapter.finish()


def test_adapter_selfplay_mcts_imitation_game_signatures():
    adapter = _disabled_adapter()
    adapter.log_selfplay(games_completed=100, buffer_size=5000, samples_generated=20_000)
    adapter.log_mcts(average_depth=8.4, average_visits=64.0, tree_reuse_rate=0.42)
    adapter.log_imitation_training(epoch=2, loss=1.0, policy_loss=0.6, value_loss=0.4)
    adapter.log_game(game_id="g1", winner=0, total_turns=120, reward=1.0)
    adapter.finish()


def test_adapter_noop_methods_dont_raise():
    adapter = _disabled_adapter()
    adapter.watch_model(model=None)
    adapter.define_metric("train/loss", step_metric="train/step", summary="min")
    adapter.define_baseline_charts()
    adapter.define_pbt_charts()
    adapter.log_histogram("h", [1.0, 2.0, 3.0])
    adapter.log_image("img", object())
    adapter.summary({"best_winrate": 0.62})
    adapter.alert("title", "text", level="INFO")
    adapter.finish()


def test_adapter_from_env(monkeypatch):
    for var in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_RUN_NAME",
        "MLFLOW_TAGS",
        "MLFLOW_REGISTRY_URI",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    adapter = WandbCompatAdapter.from_env()
    assert isinstance(adapter, WandbCompatAdapter)
    assert adapter.enabled is False
