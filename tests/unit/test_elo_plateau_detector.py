"""Unit tests for elo_plateau_detector module."""

from __future__ import annotations

import os

from alpha_zero.training.elo_plateau_detector import (
    EloPlateauDetector,
    EloPlateauDetectorConfig,
    EloSnapshot,
    EloStatus,
    get_elo_detector,
)


class TestEloStatus:
    def test_status_values(self) -> None:
        assert EloStatus.IMPROVING.value == "improving"
        assert EloStatus.STABLE.value == "stable"
        assert EloStatus.PLATEAU.value == "plateau"
        assert EloStatus.REGRESSING.value == "regressing"


class TestEloSnapshot:
    def test_creation(self) -> None:
        snapshot = EloSnapshot(
            iteration=100,
            elo_rating=1500.0,
            win_rate=0.55,
            games_played=200,
        )
        assert snapshot.iteration == 100
        assert snapshot.elo_rating == 1500.0

    def test_to_dict(self) -> None:
        snapshot = EloSnapshot(
            iteration=50,
            elo_rating=1450.0,
            win_rate=0.52,
            games_played=100,
        )
        data = snapshot.to_dict()
        assert data["iteration"] == 50
        assert data["elo_rating"] == 1450.0

    def test_from_dict(self) -> None:
        data = {
            "iteration": 75,
            "elo_rating": 1600.0,
            "win_rate": 0.60,
            "games_played": 150,
        }
        snapshot = EloSnapshot.from_dict(data)
        assert snapshot.iteration == 75
        assert snapshot.elo_rating == 1600.0


class TestEloPlateauDetectorConfig:
    def test_default_values(self) -> None:
        config = EloPlateauDetectorConfig()
        assert config.plateau_threshold_iterations == 5
        assert config.improvement_threshold_elo == 10.0
        assert config.window_size == 10
        assert config.alert_on_plateau is True

    def test_from_env(self) -> None:
        os.environ["CIV6AZ_ELO_PLATEAU_THRESHOLD"] = "3"
        os.environ["CIV6AZ_ELO_IMPROVEMENT_THRESHOLD"] = "15.0"
        try:
            config = EloPlateauDetectorConfig.from_env()
            assert config.plateau_threshold_iterations == 3
            assert config.improvement_threshold_elo == 15.0
        finally:
            del os.environ["CIV6AZ_ELO_PLATEAU_THRESHOLD"]
            del os.environ["CIV6AZ_ELO_IMPROVEMENT_THRESHOLD"]


class TestEloPlateauDetector:
    def test_init(self) -> None:
        config = EloPlateauDetectorConfig()
        detector = EloPlateauDetector(config)
        assert detector.config == config

    def test_record_evaluation(self) -> None:
        config = EloPlateauDetectorConfig()
        detector = EloPlateauDetector(config)
        detector.record_evaluation(
            iteration=1,
            elo_rating=1500.0,
            win_rate=0.50,
            games_played=100,
        )
        assert detector._best_elo == 1500.0

    def test_is_plateau_initial(self) -> None:
        config = EloPlateauDetectorConfig(min_iterations=3)
        detector = EloPlateauDetector(config)
        # Not enough data yet
        detector.record_evaluation(1, 1500.0, 0.50, 100)
        assert detector.is_plateau() is False

    def test_is_plateau_with_improvement(self) -> None:
        config = EloPlateauDetectorConfig(
            plateau_threshold_iterations=3,
            improvement_threshold_elo=10.0,
            min_iterations=2,
        )
        detector = EloPlateauDetector(config)
        # Record improving Elo
        detector.record_evaluation(1, 1500.0, 0.50, 100)
        detector.record_evaluation(2, 1520.0, 0.52, 100)
        detector.record_evaluation(3, 1540.0, 0.54, 100)
        assert detector.is_plateau() is False

    def test_is_plateau_detected(self) -> None:
        config = EloPlateauDetectorConfig(
            plateau_threshold_iterations=3,
            improvement_threshold_elo=10.0,
            min_iterations=2,
        )
        detector = EloPlateauDetector(config)
        # Record stagnant Elo
        detector.record_evaluation(1, 1500.0, 0.50, 100)
        detector.record_evaluation(2, 1502.0, 0.50, 100)  # +2 < threshold
        detector.record_evaluation(3, 1501.0, 0.50, 100)
        detector.record_evaluation(4, 1503.0, 0.50, 100)
        detector.record_evaluation(5, 1502.0, 0.50, 100)
        assert detector.is_plateau() is True

    def test_get_status(self) -> None:
        config = EloPlateauDetectorConfig(min_iterations=2)
        detector = EloPlateauDetector(config)
        detector.record_evaluation(1, 1500.0, 0.50, 100)
        detector.record_evaluation(2, 1520.0, 0.52, 100)
        status = detector.get_status()
        assert status == EloStatus.IMPROVING

    def test_get_trend(self) -> None:
        config = EloPlateauDetectorConfig(window_size=5)
        detector = EloPlateauDetector(config)
        # Record steadily increasing Elo
        for i in range(10):
            detector.record_evaluation(
                iteration=i + 1,
                elo_rating=1500.0 + i * 10,
                win_rate=0.50,
                games_played=100,
            )
        trend = detector.get_trend()
        assert trend is not None
        assert trend > 0  # Positive trend

    def test_get_recent_elo(self) -> None:
        config = EloPlateauDetectorConfig()
        detector = EloPlateauDetector(config)
        for i in range(5):
            detector.record_evaluation(i + 1, 1500.0 + i * 10, 0.50, 100)
        recent = detector.get_recent_elo(count=3)
        assert len(recent) == 3

    def test_get_statistics(self) -> None:
        config = EloPlateauDetectorConfig()
        detector = EloPlateauDetector(config)
        detector.record_evaluation(1, 1500.0, 0.50, 100)
        stats = detector.get_statistics()
        assert "best_elo" in stats
        assert "current_status" in stats
        assert "is_plateau" in stats

    def test_register_plateau_callback(self) -> None:
        config = EloPlateauDetectorConfig(
            plateau_threshold_iterations=2,
            improvement_threshold_elo=10.0,
            min_iterations=2,
        )
        detector = EloPlateauDetector(config)
        callback_called = []

        def callback(snapshot):
            callback_called.append(snapshot)

        detector.register_plateau_callback(callback)
        # Trigger plateau
        detector.record_evaluation(1, 1500.0, 0.50, 100)
        detector.record_evaluation(2, 1501.0, 0.50, 100)
        detector.record_evaluation(3, 1502.0, 0.50, 100)
        detector.record_evaluation(4, 1501.0, 0.50, 100)
        assert len(callback_called) >= 1

    def test_reset(self) -> None:
        config = EloPlateauDetectorConfig()
        detector = EloPlateauDetector(config)
        detector.record_evaluation(1, 1500.0, 0.50, 100)
        detector.reset()
        assert detector._best_elo is None
        assert len(detector._history) == 0


class TestGetEloDetector:
    def test_returns_detector(self) -> None:
        detector = get_elo_detector()
        assert isinstance(detector, EloPlateauDetector)
