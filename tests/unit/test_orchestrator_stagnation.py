"""Tests for orchestrator training-stagnation detection.

Two layers, both isolation-first (no networks/buffers/MCTS built):
  * ``StagnationConfig`` parsing + ``CIV6AZ_STAGNATION_*`` env overrides.
  * ``TrainingOrchestrator._check_training_stagnation`` wiring, exercised on an
    instance created via ``object.__new__`` with fake detector + MLflow logger.
"""

from __future__ import annotations

from types import SimpleNamespace

from alpha_zero.orchestrator.config import OrchestratorConfig, StagnationConfig
from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator


class TestStagnationConfig:
    def test_defaults_are_off(self) -> None:
        cfg = StagnationConfig()
        assert cfg.enabled is False
        assert cfg.auto_stop is False
        assert cfg.use_elo_plateau is True
        assert cfg.use_convergence is False  # loss-trap guard

    def test_from_dict(self) -> None:
        cfg = StagnationConfig.from_dict({"enabled": True, "auto_stop": True})
        assert cfg.enabled is True
        assert cfg.auto_stop is True

    def test_present_on_orchestrator_config(self) -> None:
        assert isinstance(OrchestratorConfig().stagnation, StagnationConfig)

    def test_from_dict_nested_in_orchestrator_config(self) -> None:
        config = OrchestratorConfig._from_dict({"stagnation": {"enabled": True}})
        assert config.stagnation.enabled is True

    def test_env_override_reaches_stagnation(self, monkeypatch) -> None:
        monkeypatch.setenv("CIV6AZ_STAGNATION_ENABLED", "true")
        monkeypatch.setenv("CIV6AZ_STAGNATION_AUTO_STOP", "true")
        config = OrchestratorConfig().apply_env_overrides()
        assert config.stagnation.enabled is True
        assert config.stagnation.auto_stop is True


class _FakeDetector:
    """Stand-in for EloPlateauDetector capturing fed values."""

    def __init__(self, plateau: bool) -> None:
        self._plateau = plateau
        self.records: list[dict] = []

    def record_evaluation(self, **kwargs) -> None:
        self.records.append(kwargs)

    def is_plateau(self) -> bool:
        return self._plateau

    def get_statistics(self) -> dict:
        return {"iterations_since_improvement": 5}


class _FakeMlflow:
    def __init__(self) -> None:
        self.metrics: list[dict] = []
        self.tags: dict[str, str] = {}

    def log_metrics(self, data, step=None) -> None:
        self.metrics.append(data)

    def set_tag(self, key, value) -> None:
        self.tags[key] = value


def _make_orchestrator(
    detector, *, auto_stop: bool, alert: bool = True, elo: bool = False
) -> TrainingOrchestrator:
    """Build a minimal orchestrator without running heavy __init__."""
    orch = object.__new__(TrainingOrchestrator)
    orch._stagnation_detector = detector
    orch._stagnation_stopped = False
    orch._shutdown_requested = False
    orch._mlflow_logger = _FakeMlflow()
    orch._elo_system = object() if elo else None
    orch._elo_ratings = {"current": 1550.0} if elo else {}
    orch.config = SimpleNamespace(
        stagnation=StagnationConfig(enabled=True, auto_stop=auto_stop, alert_to_mlflow=alert),
        training=SimpleNamespace(evaluation_games=4),
    )
    orch.state = SimpleNamespace(iteration=7)
    return orch


class TestCheckTrainingStagnation:
    def test_no_detector_is_noop(self) -> None:
        orch = _make_orchestrator(None, auto_stop=True)
        orch._stagnation_detector = None
        orch._check_training_stagnation(0.5)
        assert orch._shutdown_requested is False

    def test_plateau_alerts_without_stopping(self) -> None:
        detector = _FakeDetector(plateau=True)
        orch = _make_orchestrator(detector, auto_stop=False)
        orch._check_training_stagnation(0.5)
        assert orch._shutdown_requested is False  # alert-only
        assert "alert/WARNING/stagnation" in orch._mlflow_logger.tags
        assert orch._mlflow_logger.metrics  # stagnation/* logged

    def test_plateau_auto_stop_requests_shutdown(self) -> None:
        detector = _FakeDetector(plateau=True)
        orch = _make_orchestrator(detector, auto_stop=True)
        orch._check_training_stagnation(0.5)
        assert orch._shutdown_requested is True
        assert orch._stagnation_stopped is True

    def test_no_plateau_does_not_stop(self) -> None:
        detector = _FakeDetector(plateau=False)
        orch = _make_orchestrator(detector, auto_stop=True)
        orch._check_training_stagnation(0.6)
        assert orch._shutdown_requested is False
        assert "alert/WARNING/stagnation" not in orch._mlflow_logger.tags

    def test_uses_real_elo_when_available(self) -> None:
        detector = _FakeDetector(plateau=False)
        orch = _make_orchestrator(detector, auto_stop=False, elo=True)
        orch._check_training_stagnation(0.5)
        assert detector.records[0]["elo_rating"] == 1550.0

    def test_falls_back_to_winrate_scaled_rating(self) -> None:
        detector = _FakeDetector(plateau=False)
        orch = _make_orchestrator(detector, auto_stop=False, elo=False)
        orch._check_training_stagnation(0.5)
        # win_rate * 1000 keeps the point-based ELO threshold meaningful.
        assert detector.records[0]["elo_rating"] == 500.0
