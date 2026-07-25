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


def test_adapter_log_table_rejects_non_rectangular_data():
    """A short row used to be silently dropped by ``zip(..., strict=False)``,
    producing column lists of mismatched lengths and a corrupted CSV. The
    adapter now raises ``ValueError`` so malformed input surfaces immediately.
    """
    adapter = _disabled_adapter()
    with pytest.raises(ValueError, match="non-rectangular"):
        adapter.log_table(
            "broken",
            ["a", "b", "c"],
            [[1, 2, 3], [4, 5]],  # second row missing a value
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


# ---------------------------------------------------------------------------
# WandbCompatAdapter.log() — numpy / nested-dict coercion contract
# (ports the _clean_data guarantees of the retired WandbLogger)
# ---------------------------------------------------------------------------


def _adapter_with_metric_spy() -> tuple[WandbCompatAdapter, list[tuple[dict, int | None]]]:
    """Disabled adapter whose log_metrics calls are recorded."""
    adapter = _disabled_adapter()
    calls: list[tuple[dict, int | None]] = []

    def _spy(metrics, step=None):
        calls.append((dict(metrics), step))

    adapter.log_metrics = _spy  # type: ignore[method-assign]
    return adapter, calls


def test_adapter_log_coerces_numpy_scalars():
    np = pytest.importorskip("numpy")
    adapter, calls = _adapter_with_metric_spy()
    adapter.log(
        {
            "int64": np.int64(7),
            "float32": np.float32(0.5),
            "bool_": np.bool_(True),
            "plain_bool": False,
        },
        step=3,
    )
    assert calls == [({"int64": 7.0, "float32": 0.5, "bool_": 1.0, "plain_bool": 0.0}, 3)]


def test_adapter_log_coerces_zero_d_and_size_one_arrays():
    np = pytest.importorskip("numpy")
    adapter, calls = _adapter_with_metric_spy()
    adapter.log({"zero_d": np.array(2.5), "size_one": np.array([4])})
    assert calls[0][0] == {"zero_d": 2.5, "size_one": 4.0}


def test_adapter_log_drops_oversized_arrays_and_strings():
    np = pytest.importorskip("numpy")
    adapter, calls = _adapter_with_metric_spy()
    adapter.log({"big": np.ones(8), "label": "hello", "ok": 1})
    assert calls == [({"ok": 1.0}, None)]


def test_adapter_log_flattens_one_level_of_nested_dicts():
    adapter, calls = _adapter_with_metric_spy()
    adapter.log({"train": {"loss": 1.5, "note": "skip-me"}, "top": 2})
    assert calls == [({"train/loss": 1.5, "top": 2.0}, None)]


def test_adapter_log_empty_after_filtering_makes_no_call():
    adapter, calls = _adapter_with_metric_spy()
    adapter.log({"label": "only-strings"})
    assert calls == []


# ---------------------------------------------------------------------------
# log_reflection_stats — previously covered only by the retired W&B suite
# ---------------------------------------------------------------------------


def test_adapter_log_reflection_stats_full_payload():
    adapter, calls = _adapter_with_metric_spy()
    adapter.log_reflection_stats(
        total_entries=12,
        unique_patterns=4,
        recent_additions=2,
        avg_critique_length=87.5,
        iteration=9,
    )
    assert calls == [
        (
            {
                "reflection/total_entries": 12.0,
                "reflection/unique_patterns": 4.0,
                "reflection/recent_additions": 2.0,
                "reflection/avg_critique_length": 87.5,
            },
            9,
        )
    ]


def test_adapter_log_reflection_stats_optional_fields_omitted():
    adapter, calls = _adapter_with_metric_spy()
    adapter.log_reflection_stats(total_entries=3)
    assert calls == [({"reflection/total_entries": 3.0}, None)]


# ---------------------------------------------------------------------------
# run_name property — hardened accessor for run-group propagation
# ---------------------------------------------------------------------------


def test_adapter_run_name_none_before_init():
    adapter = _disabled_adapter()
    assert adapter.run_name is None


def test_adapter_run_name_reads_info_run_name():
    adapter = _disabled_adapter()
    adapter._run = SimpleNamespace(info=SimpleNamespace(run_name="iter-5", run_id="abc123"))
    assert adapter.run_name == "iter-5"


def test_adapter_run_name_falls_back_to_run_id():
    adapter = _disabled_adapter()
    adapter._run = SimpleNamespace(info=SimpleNamespace(run_name=None, run_id="abc123"))
    assert adapter.run_name == "abc123"


# ---------------------------------------------------------------------------
# alert() / watch_model() fidelity — visible even when MLflow is disabled
# ---------------------------------------------------------------------------


def test_adapter_alert_emits_warning_log(caplog):
    adapter = _disabled_adapter()
    with caplog.at_level("WARNING", logger="alpha_zero.monitoring.mlflow_logger"):
        adapter.alert("BudgetExceeded", "cost above cap", level="ERROR")
    assert any(
        "BudgetExceeded" in rec.message and "cost above cap" in rec.message
        for rec in caplog.records
    )


def test_adapter_watch_model_sets_request_tag():
    adapter = _disabled_adapter()
    tags: list[tuple[str, str]] = []
    adapter.set_tag = lambda k, v: tags.append((k, str(v)))  # type: ignore[method-assign]
    adapter.watch_model(object(), log="gradients", log_freq=100)
    assert ("watch_model_requested", "true") in tags


# ---------------------------------------------------------------------------
# log_baseline_evaluation — per-opponent metric fan-out (ported from the
# retired test_wandb_baseline_metrics suite)
# ---------------------------------------------------------------------------


def test_adapter_log_baseline_evaluation_per_opponent_prefixes():
    adapter = _disabled_adapter()
    seen: list[tuple[object, str, int | None]] = []

    def _spy(result, prefix="eval", step=None):
        seen.append((result, prefix, step))
        return {f"{prefix}/win_rate": 0.5}

    adapter.log_evaluation_result = _spy  # type: ignore[method-assign]
    combined = adapter.log_baseline_evaluation(
        {"random": object(), "heuristic": object()}, iteration=7
    )
    assert [s[1] for s in seen] == ["eval/random", "eval/heuristic"]
    assert all(s[2] == 7 for s in seen)
    assert set(combined) == {"eval/random/win_rate", "eval/heuristic/win_rate"}
