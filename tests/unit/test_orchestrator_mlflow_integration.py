"""Integration tests for MLflow wiring inside ``TrainingOrchestrator``.

These tests exercise the public surface added by the MLflow integration
without spinning up real training. The MLflow client is mocked; the focus
is on the ``_run_periodic_full_evaluation`` and ``_log_iteration`` glue.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from alpha_zero.orchestrator.config import OrchestratorConfig
from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator


def _make_orchestrator() -> TrainingOrchestrator:
    config = OrchestratorConfig()
    config.mlflow.enabled = True
    config.training.full_eval_interval = 1
    config.training.evaluation_games = 2
    return TrainingOrchestrator(config)


def _attach_fake_state(orch: TrainingOrchestrator, iteration: int = 1) -> None:
    """Wire enough fake state on the orchestrator for unit tests."""
    orch.state.iteration = iteration
    orch.state.best_win_rate = 0.5
    orch._mlflow_logger = MagicMock()
    orch._mlflow_logger.is_initialized = True
    orch._network = MagicMock()


def test_run_periodic_full_evaluation_no_op_without_pipeline():
    orch = _make_orchestrator()
    _attach_fake_state(orch)
    orch._eval_pipeline = None
    orch._elo_system = None
    orch._run_periodic_full_evaluation(win_rate=0.6)
    # MLflow must not be touched when prerequisites are missing.
    orch._mlflow_logger.log_evaluation.assert_not_called()
    orch._mlflow_logger.log_elo_update.assert_not_called()


def test_run_periodic_full_evaluation_off_cadence():
    orch = _make_orchestrator()
    orch.config.training.full_eval_interval = 5
    _attach_fake_state(orch, iteration=3)
    orch._eval_pipeline = MagicMock()
    orch._elo_system = MagicMock()
    orch._run_periodic_full_evaluation(win_rate=0.6)
    orch._eval_pipeline.evaluate_vs_random.assert_not_called()


def test_run_periodic_full_evaluation_logs_metrics_on_cadence():
    orch = _make_orchestrator()
    _attach_fake_state(orch, iteration=2)
    orch._eval_pipeline = MagicMock()
    fake_random = SimpleNamespace(
        win_rate=SimpleNamespace(win_rate=70.0, loss_rate=30.0, draw_rate=0.0),
        value_correlation=0.4,
        avg_game_length=42,
    )
    fake_heuristic = SimpleNamespace(
        win_rate=SimpleNamespace(win_rate=55.0, loss_rate=45.0, draw_rate=0.0),
        value_correlation=0.3,
        avg_game_length=50,
    )
    orch._eval_pipeline.evaluate_vs_random.return_value = fake_random
    orch._eval_pipeline.evaluate_vs_heuristic.return_value = fake_heuristic

    elo_system = MagicMock()
    update1 = SimpleNamespace(
        player_id="current",
        rating_before=1500.0,
        rating_after=1520.0,
        rating_change=20.0,
    )
    update2 = SimpleNamespace(
        player_id="previous_best",
        rating_before=1500.0,
        rating_after=1480.0,
        rating_change=-20.0,
    )
    elo_system.process_match.return_value = (update1, update2)
    orch._elo_system = elo_system
    orch._elo_ratings = {"current": 1500.0, "previous_best": 1500.0}

    orch._run_periodic_full_evaluation(win_rate=0.7)  # higher than best (0.5)

    orch._mlflow_logger.log_evaluation.assert_called_once()
    payload = orch._mlflow_logger.log_evaluation.call_args[0][0]
    assert set(payload.keys()) == {"random", "heuristic"}

    # Score should reflect win.
    pm_call = elo_system.process_match.call_args.kwargs
    assert pm_call["score1"] == 1.0
    assert pm_call["score2"] == 0.0

    # ELO ratings should have been updated.
    assert orch._elo_ratings["current"] == 1520.0
    orch._mlflow_logger.log_elo_update.assert_called_once()


def test_run_periodic_full_evaluation_handles_pipeline_exception():
    orch = _make_orchestrator()
    _attach_fake_state(orch, iteration=1)
    orch._eval_pipeline = MagicMock()
    orch._eval_pipeline.evaluate_vs_random.side_effect = RuntimeError("boom")
    orch._elo_system = MagicMock()
    # Must not raise; must not call MLflow.
    orch._run_periodic_full_evaluation(win_rate=0.6)
    orch._mlflow_logger.log_evaluation.assert_not_called()


def test_run_periodic_full_evaluation_draw_path():
    orch = _make_orchestrator()
    _attach_fake_state(orch, iteration=1)
    orch.state.best_win_rate = 0.5
    orch._eval_pipeline = MagicMock()
    orch._eval_pipeline.evaluate_vs_random.return_value = SimpleNamespace(
        win_rate=SimpleNamespace(win_rate=0.0, loss_rate=0.0, draw_rate=0.0),
        value_correlation=0.0,
        avg_game_length=0,
    )
    orch._eval_pipeline.evaluate_vs_heuristic.return_value = SimpleNamespace(
        win_rate=SimpleNamespace(win_rate=0.0, loss_rate=0.0, draw_rate=0.0),
        value_correlation=0.0,
        avg_game_length=0,
    )
    elo_system = MagicMock()
    elo_system.process_match.return_value = (
        SimpleNamespace(
            player_id="current",
            rating_before=1500.0,
            rating_after=1500.0,
            rating_change=0.0,
        ),
        SimpleNamespace(
            player_id="previous_best",
            rating_before=1500.0,
            rating_after=1500.0,
            rating_change=0.0,
        ),
    )
    orch._elo_system = elo_system
    orch._elo_ratings = {"current": 1500.0, "previous_best": 1500.0}

    orch._run_periodic_full_evaluation(win_rate=0.5)  # equal to best
    pm_call = elo_system.process_match.call_args.kwargs
    assert pm_call["score1"] == 0.5
    assert pm_call["score2"] == 0.5


def test_build_elo_leaderboard_snapshot():
    orch = _make_orchestrator()
    orch._elo_ratings = {"a": 1500.0, "b": 1600.0}
    leaderboard = orch._build_elo_leaderboard()
    assert leaderboard.get_rating("a") == 1500.0
    assert leaderboard.get_rating("b") == 1600.0


def test_config_field_present():
    """Smoke test: OrchestratorConfig has the new fields."""
    config = OrchestratorConfig()
    assert hasattr(config, "mlflow")
    assert config.training.full_eval_interval >= 1
