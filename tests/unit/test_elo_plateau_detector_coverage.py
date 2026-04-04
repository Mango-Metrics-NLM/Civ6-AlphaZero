"""Comprehensive tests for alpha_zero.training.elo_plateau_detector.

Covers EloPlateauDetectorConfig, EloSnapshot, EloPlateauDetector,
and the get_elo_detector convenience function.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from alpha_zero.training.elo_plateau_detector import (
    EloPlateauDetector,
    EloPlateauDetectorConfig,
    EloSnapshot,
    EloStatus,
    get_elo_detector,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def detector_config():
    return EloPlateauDetectorConfig(
        plateau_threshold_iterations=5,
        improvement_threshold_elo=10.0,
        regression_threshold_elo=50.0,
        window_size=10,
        min_iterations=3,
        alert_on_plateau=True,
    )


@pytest.fixture()
def detector(detector_config):
    return EloPlateauDetector(detector_config)


# ---------------------------------------------------------------------------
# EloSnapshot
# ---------------------------------------------------------------------------


class TestEloSnapshot:
    def test_to_dict(self):
        snap = EloSnapshot(
            iteration=5,
            elo_rating=1500.0,
            win_rate=0.55,
            games_played=100,
            timestamp=1000.0,
            opponent_type="self",
        )
        d = snap.to_dict()
        assert d["iteration"] == 5
        assert d["elo_rating"] == 1500.0
        assert d["opponent_type"] == "self"

    def test_from_dict(self):
        d = {
            "iteration": 10,
            "elo_rating": 1600.0,
            "win_rate": 0.6,
            "games_played": 200,
            "timestamp": 2000.0,
            "opponent_type": "previous",
        }
        snap = EloSnapshot.from_dict(d)
        assert snap.iteration == 10
        assert snap.opponent_type == "previous"

    def test_from_dict_defaults(self):
        d = {
            "iteration": 1,
            "elo_rating": 1500.0,
            "win_rate": 0.5,
            "games_played": 50,
        }
        snap = EloSnapshot.from_dict(d)
        assert snap.opponent_type == "self"
        assert snap.timestamp > 0


# ---------------------------------------------------------------------------
# EloPlateauDetectorConfig
# ---------------------------------------------------------------------------


class TestEloPlateauDetectorConfig:
    def test_defaults(self):
        config = EloPlateauDetectorConfig()
        assert config.plateau_threshold_iterations >= 1
        assert config.window_size >= 2

    def test_validation_plateau_threshold(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="plateau_threshold"):
            EloPlateauDetectorConfig(plateau_threshold_iterations=0)

    def test_validation_improvement_threshold(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="improvement_threshold"):
            EloPlateauDetectorConfig(improvement_threshold_elo=0)

    def test_validation_regression_threshold(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="regression_threshold"):
            EloPlateauDetectorConfig(regression_threshold_elo=-1)

    def test_validation_window_size(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="window_size"):
            EloPlateauDetectorConfig(window_size=1)

    def test_validation_min_iterations(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="min_iterations"):
            EloPlateauDetectorConfig(min_iterations=0)

    def test_from_env(self):
        env = {"CIV6AZ_ELO_PLATEAU_THRESHOLD": "10"}
        with patch.dict("os.environ", env, clear=False):
            config = EloPlateauDetectorConfig.from_env()
            assert config.plateau_threshold_iterations == 10

    def test_from_yaml(self):
        data = {"plateau_threshold_iterations": 8, "window_size": 15}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(data, f)
            path = Path(f.name)

        config = EloPlateauDetectorConfig.from_yaml(path)
        assert config.plateau_threshold_iterations == 8
        assert config.window_size == 15
        path.unlink()

    def test_from_yaml_missing_file(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            EloPlateauDetectorConfig.from_yaml("/nonexistent.yaml")

    def test_from_yaml_invalid_yaml(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{{bad yaml::")
            path = Path(f.name)

        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            EloPlateauDetectorConfig.from_yaml(path)
        path.unlink()

    def test_from_dict(self):
        config = EloPlateauDetectorConfig.from_dict(
            {"plateau_threshold_iterations": 7, "alert_on_plateau": False}
        )
        assert config.plateau_threshold_iterations == 7
        assert config.alert_on_plateau is False

    def test_to_dict_roundtrip(self, detector_config):
        d = detector_config.to_dict()
        config2 = EloPlateauDetectorConfig.from_dict(d)
        assert config2.to_dict() == d


# ---------------------------------------------------------------------------
# EloPlateauDetector
# ---------------------------------------------------------------------------


class TestEloPlateauDetector:
    def test_initial_state(self, detector):
        assert detector._best_elo is None
        assert detector._total_evaluations == 0

    def test_record_first_evaluation(self, detector):
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        assert detector._best_elo == 1500.0
        assert detector._total_evaluations == 1

    def test_record_improvement(self, detector):
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        detector.record_evaluation(iteration=1, elo_rating=1520.0, win_rate=0.55, games_played=100)
        assert detector._best_elo == 1520.0
        assert detector._iterations_since_improvement == 0

    def test_record_no_improvement(self, detector):
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        detector.record_evaluation(iteration=1, elo_rating=1505.0, win_rate=0.5, games_played=100)
        assert detector._best_elo == 1500.0  # 5 < 10 threshold
        assert detector._iterations_since_improvement == 1

    def test_improvement_callback(self, detector):
        results = []
        detector.register_improvement_callback(lambda snap: results.append(snap.elo_rating))
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        assert results == [1500.0]  # First record counts as improvement

    def test_improvement_callback_error_handled(self, detector):
        def bad_cb(snap):
            raise RuntimeError("boom")

        detector.register_improvement_callback(bad_cb)
        # Should not raise
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)

    def test_plateau_detection(self, detector):
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        for i in range(1, 10):
            detector.record_evaluation(
                iteration=i, elo_rating=1500.0, win_rate=0.5, games_played=100
            )
        assert detector.is_plateau() is True

    def test_plateau_not_enough_iterations(self, detector):
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        assert detector.is_plateau() is False

    def test_plateau_callback(self, detector):
        plateau_events = []
        detector.register_plateau_callback(lambda snap: plateau_events.append(snap.elo_rating))
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        for i in range(1, 10):
            detector.record_evaluation(
                iteration=i, elo_rating=1500.0, win_rate=0.5, games_played=100
            )
        assert len(plateau_events) >= 1

    def test_plateau_callback_error_handled(self, detector):
        def bad_cb(snap):
            raise RuntimeError("boom")

        detector.register_plateau_callback(bad_cb)
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        for i in range(1, 10):
            detector.record_evaluation(
                iteration=i, elo_rating=1500.0, win_rate=0.5, games_played=100
            )

    def test_plateau_alert_sent(self):
        config = EloPlateauDetectorConfig(
            plateau_threshold_iterations=3,
            min_iterations=2,
            alert_on_plateau=True,
        )
        alert_mgr = MagicMock()
        det = EloPlateauDetector(config, alert_manager=alert_mgr)

        det.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        for i in range(1, 6):
            det.record_evaluation(iteration=i, elo_rating=1500.0, win_rate=0.5, games_played=100)
        alert_mgr.elo_plateau.assert_called()

    def test_plateau_alert_sent_only_once(self):
        config = EloPlateauDetectorConfig(
            plateau_threshold_iterations=3,
            min_iterations=2,
            alert_on_plateau=True,
        )
        alert_mgr = MagicMock()
        det = EloPlateauDetector(config, alert_manager=alert_mgr)

        det.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        for i in range(1, 10):
            det.record_evaluation(iteration=i, elo_rating=1500.0, win_rate=0.5, games_played=100)
        # Alert should be sent once (flag prevents re-sending)
        assert alert_mgr.elo_plateau.call_count == 1

    def test_plateau_alert_resets_on_improvement(self):
        config = EloPlateauDetectorConfig(
            plateau_threshold_iterations=3,
            min_iterations=2,
            alert_on_plateau=True,
        )
        alert_mgr = MagicMock()
        det = EloPlateauDetector(config, alert_manager=alert_mgr)

        det.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        for i in range(1, 6):
            det.record_evaluation(iteration=i, elo_rating=1500.0, win_rate=0.5, games_played=100)
        # Now improve
        det.record_evaluation(iteration=6, elo_rating=1520.0, win_rate=0.6, games_played=100)
        assert det._alert_sent_for_current_plateau is False

    def test_no_alert_when_disabled(self):
        config = EloPlateauDetectorConfig(
            plateau_threshold_iterations=3,
            min_iterations=2,
            alert_on_plateau=False,
        )
        alert_mgr = MagicMock()
        det = EloPlateauDetector(config, alert_manager=alert_mgr)

        det.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        for i in range(1, 6):
            det.record_evaluation(iteration=i, elo_rating=1500.0, win_rate=0.5, games_played=100)
        alert_mgr.elo_plateau.assert_not_called()

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    def test_get_status_stable_not_enough_data(self, detector):
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        assert detector.get_status() == EloStatus.STABLE

    def test_get_status_improving(self, detector):
        for i in range(5):
            detector.record_evaluation(
                iteration=i, elo_rating=1500.0 + i * 15, win_rate=0.5, games_played=100
            )
        assert detector.get_status() == EloStatus.IMPROVING

    def test_get_status_plateau(self, detector):
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        for i in range(1, 10):
            detector.record_evaluation(
                iteration=i, elo_rating=1500.0, win_rate=0.5, games_played=100
            )
        assert detector.get_status() == EloStatus.PLATEAU

    def test_get_status_regressing(self):
        config = EloPlateauDetectorConfig(
            plateau_threshold_iterations=100,
            regression_threshold_elo=30.0,
            window_size=10,
            min_iterations=3,
        )
        det = EloPlateauDetector(config)
        # Start high then drop significantly
        det.record_evaluation(iteration=0, elo_rating=1600.0, win_rate=0.6, games_played=100)
        for i in range(1, 5):
            det.record_evaluation(iteration=i, elo_rating=1600.0, win_rate=0.6, games_played=100)
        for i in range(5, 12):
            det.record_evaluation(iteration=i, elo_rating=1540.0, win_rate=0.45, games_played=100)
        status = det.get_status()
        assert status in (EloStatus.REGRESSING, EloStatus.STABLE)

    def test_get_status_stable_with_minor_changes(self, detector):
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        for i in range(1, 5):
            detector.record_evaluation(
                iteration=i, elo_rating=1502.0, win_rate=0.5, games_played=100
            )
        assert detector.get_status() == EloStatus.STABLE

    # -----------------------------------------------------------------------
    # Trend
    # -----------------------------------------------------------------------

    def test_get_trend_not_enough_data(self, detector):
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        assert detector.get_trend() is None

    def test_get_trend_increasing(self):
        config = EloPlateauDetectorConfig(window_size=5, min_iterations=1)
        det = EloPlateauDetector(config)
        for i in range(10):
            det.record_evaluation(
                iteration=i, elo_rating=1500.0 + i * 20, win_rate=0.5, games_played=100
            )
        trend = det.get_trend()
        assert trend is not None
        assert trend > 0

    def test_get_trend_decreasing(self):
        config = EloPlateauDetectorConfig(window_size=5, min_iterations=1)
        det = EloPlateauDetector(config)
        for i in range(10):
            det.record_evaluation(
                iteration=i, elo_rating=1700.0 - i * 20, win_rate=0.5, games_played=100
            )
        trend = det.get_trend()
        assert trend is not None
        assert trend < 0

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def test_get_recent_elo(self, detector):
        for i in range(5):
            detector.record_evaluation(
                iteration=i, elo_rating=1500.0 + i * 10, win_rate=0.5, games_played=100
            )
        recent = detector.get_recent_elo(count=3)
        assert len(recent) == 3
        assert recent[-1] == 1540.0

    def test_get_recent_elo_fewer_than_count(self, detector):
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        recent = detector.get_recent_elo(count=5)
        assert len(recent) == 1

    def test_get_statistics(self, detector):
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        detector.record_evaluation(iteration=1, elo_rating=1520.0, win_rate=0.55, games_played=100)
        stats = detector.get_statistics()
        assert stats["total_evaluations"] == 2
        assert stats["best_elo"] == 1520.0
        assert stats["current_elo"] == 1520.0
        assert "trend" in stats
        assert "is_plateau" in stats

    def test_get_statistics_empty(self, detector):
        stats = detector.get_statistics()
        assert stats["current_elo"] is None
        assert stats["recent_mean_elo"] is None
        assert stats["recent_std_elo"] is None

    def test_reset(self, detector):
        detector.record_evaluation(iteration=0, elo_rating=1500.0, win_rate=0.5, games_played=100)
        detector.reset()
        assert detector._best_elo is None
        assert len(detector._history) == 0
        assert detector._plateau_count == 0
        assert detector._alert_sent_for_current_plateau is False

    def test_opponent_type_passed(self, detector):
        detector.record_evaluation(
            iteration=0,
            elo_rating=1500.0,
            win_rate=0.5,
            games_played=100,
            opponent_type="random",
        )
        assert detector._history[-1].opponent_type == "random"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestGetEloDetector:
    def test_returns_detector(self):
        det = get_elo_detector()
        assert isinstance(det, EloPlateauDetector)

    def test_with_alert_manager(self):
        mgr = MagicMock()
        det = get_elo_detector(alert_manager=mgr)
        assert det._alert_manager is mgr
