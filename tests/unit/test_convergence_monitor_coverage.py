"""Comprehensive tests for alpha_zero.training.convergence_monitor.

Covers ConvergenceMonitorConfig, MetricSnapshot, Recommendation,
and ConvergenceMonitor including edge cases, callbacks, and YAML loading.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from alpha_zero.training.convergence_monitor import (
    ConvergenceMonitor,
    ConvergenceMonitorConfig,
    ConvergenceStatus,
    MetricSnapshot,
    Recommendation,
    RecommendationType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config():
    return ConvergenceMonitorConfig(
        window_size=10,
        stall_threshold_iterations=20,
        improvement_threshold=0.01,
        divergence_threshold=0.1,
        min_win_rate=0.5,
        target_win_rate=0.7,
        lr_reduction_factor=0.5,
        exploration_boost_factor=1.5,
        enable_pbt_recommendations=True,
        enable_curriculum_recommendations=True,
    )


@pytest.fixture()
def monitor(default_config):
    return ConvergenceMonitor(default_config)


# ---------------------------------------------------------------------------
# ConvergenceMonitorConfig
# ---------------------------------------------------------------------------


class TestConvergenceMonitorConfig:
    def test_defaults(self):
        config = ConvergenceMonitorConfig()
        assert config.window_size >= 5
        assert config.stall_threshold_iterations >= 10

    def test_validation_window_size_too_small(self):
        with pytest.raises(ValueError, match="window_size"):
            ConvergenceMonitorConfig(window_size=4)

    def test_validation_stall_threshold_too_small(self):
        with pytest.raises(ValueError, match="stall_threshold_iterations"):
            ConvergenceMonitorConfig(stall_threshold_iterations=9)

    def test_validation_improvement_threshold_zero(self):
        with pytest.raises(ValueError, match="improvement_threshold"):
            ConvergenceMonitorConfig(improvement_threshold=0)

    def test_validation_improvement_threshold_negative(self):
        with pytest.raises(ValueError, match="improvement_threshold"):
            ConvergenceMonitorConfig(improvement_threshold=-0.1)

    def test_validation_divergence_threshold_nonpositive(self):
        with pytest.raises(ValueError, match="divergence_threshold"):
            ConvergenceMonitorConfig(divergence_threshold=0)

    def test_validation_min_win_rate_out_of_range(self):
        with pytest.raises(ValueError, match="min_win_rate"):
            ConvergenceMonitorConfig(min_win_rate=-0.1)
        with pytest.raises(ValueError, match="min_win_rate"):
            ConvergenceMonitorConfig(min_win_rate=1.1)

    def test_validation_target_win_rate_out_of_range(self):
        with pytest.raises(ValueError, match="target_win_rate"):
            ConvergenceMonitorConfig(target_win_rate=-0.1)
        with pytest.raises(ValueError, match="target_win_rate"):
            ConvergenceMonitorConfig(target_win_rate=1.1)

    def test_validation_lr_reduction_factor_bounds(self):
        with pytest.raises(ValueError, match="lr_reduction_factor"):
            ConvergenceMonitorConfig(lr_reduction_factor=0)
        with pytest.raises(ValueError, match="lr_reduction_factor"):
            ConvergenceMonitorConfig(lr_reduction_factor=1)

    def test_validation_exploration_boost_factor(self):
        with pytest.raises(ValueError, match="exploration_boost_factor"):
            ConvergenceMonitorConfig(exploration_boost_factor=1.0)

    def test_from_env(self):
        env = {
            "CIV6AZ_CONVERGENCE_WINDOW": "20",
            "CIV6AZ_CONVERGENCE_STALL_THRESHOLD": "50",
        }
        with patch.dict("os.environ", env, clear=False):
            config = ConvergenceMonitorConfig.from_env()
            assert config.window_size == 20
            assert config.stall_threshold_iterations == 50

    def test_from_yaml(self):
        data = {
            "convergence_monitor": {
                "window_size": 15,
                "stall_threshold_iterations": 30,
                "improvement_threshold": 0.02,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(data, f)
            path = Path(f.name)

        config = ConvergenceMonitorConfig.from_yaml(path)
        assert config.window_size == 15
        assert config.stall_threshold_iterations == 30
        assert config.improvement_threshold == 0.02
        path.unlink()

    def test_from_yaml_flat(self):
        data = {"window_size": 25, "stall_threshold_iterations": 40}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(data, f)
            path = Path(f.name)

        config = ConvergenceMonitorConfig.from_yaml(path)
        assert config.window_size == 25
        path.unlink()

    def test_from_yaml_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ConvergenceMonitorConfig.from_yaml(Path("/nonexistent/config.yaml"))

    def test_from_dict(self):
        d = {"window_size": 30, "enable_pbt_recommendations": False}
        config = ConvergenceMonitorConfig.from_dict(d)
        assert config.window_size == 30
        assert config.enable_pbt_recommendations is False

    def test_to_dict_roundtrip(self, default_config):
        d = default_config.to_dict()
        config2 = ConvergenceMonitorConfig.from_dict(d)
        assert config2.to_dict() == d


# ---------------------------------------------------------------------------
# MetricSnapshot & Recommendation
# ---------------------------------------------------------------------------


class TestMetricSnapshot:
    def test_to_dict(self):
        snap = MetricSnapshot(
            iteration=100,
            timestamp=1234567890.0,
            policy_loss=0.5,
            value_loss=0.3,
            total_loss=0.8,
            win_rate=0.6,
        )
        d = snap.to_dict()
        assert d["iteration"] == 100
        assert d["policy_loss"] == 0.5
        assert d["extra_metrics"] == {}

    def test_defaults(self):
        snap = MetricSnapshot(iteration=1)
        assert snap.policy_loss is None
        assert snap.extra_metrics == {}


class TestRecommendation:
    def test_to_dict(self):
        rec = Recommendation(
            type=RecommendationType.REDUCE_LR,
            priority=5,
            reason="test",
            suggested_value=0.001,
            confidence=0.8,
        )
        d = rec.to_dict()
        assert d["type"] == "reduce_lr"
        assert d["priority"] == 5
        assert d["suggested_value"] == 0.001


# ---------------------------------------------------------------------------
# ConvergenceMonitor
# ---------------------------------------------------------------------------


class TestConvergenceMonitor:
    def test_initial_status_stable(self, monitor):
        """Not enough data returns STABLE."""
        assert monitor.get_status() == ConvergenceStatus.STABLE

    def test_record_metrics_first_improvement(self, monitor):
        monitor.record_metrics(iteration=1, total_loss=1.0)
        assert monitor._best_loss == 1.0
        assert monitor._best_iteration == 1

    def test_record_metrics_win_rate_improvement(self, monitor):
        monitor.record_metrics(iteration=1, win_rate=0.5)
        monitor.record_metrics(iteration=2, win_rate=0.52)
        assert monitor._best_win_rate == 0.52

    def test_record_metrics_no_improvement(self, monitor):
        monitor.record_metrics(iteration=1, total_loss=1.0)
        monitor.record_metrics(iteration=2, total_loss=1.0)
        assert monitor._iterations_since_improvement == 1

    def test_improvement_callback_fires(self, monitor):
        results = []
        monitor.register_improvement_callback(lambda snap: results.append(snap.iteration))
        monitor.record_metrics(iteration=1, total_loss=1.0)
        assert results == [1]

    def test_improvement_callback_error_handled(self, monitor):
        def bad_callback(snap):
            raise RuntimeError("boom")

        monitor.register_improvement_callback(bad_callback)
        # Should not raise
        monitor.record_metrics(iteration=1, total_loss=1.0)

    def test_stall_detection(self, default_config):
        config = ConvergenceMonitorConfig(
            window_size=5,
            stall_threshold_iterations=10,
        )
        mon = ConvergenceMonitor(config)
        # Record initial improvement then stall
        mon.record_metrics(iteration=0, total_loss=1.0, win_rate=0.5)
        for i in range(1, 20):
            mon.record_metrics(iteration=i, total_loss=1.0, win_rate=0.5)

        assert mon.is_stalled() is True
        assert mon._stalls_detected >= 1

    def test_stall_callback_fires(self):
        config = ConvergenceMonitorConfig(window_size=5, stall_threshold_iterations=10)
        mon = ConvergenceMonitor(config)
        stall_events = []
        mon.register_stall_callback(lambda status: stall_events.append(status))

        mon.record_metrics(iteration=0, total_loss=1.0)
        for i in range(1, 20):
            mon.record_metrics(iteration=i, total_loss=1.0)
        mon.is_stalled()
        assert len(stall_events) >= 1
        assert stall_events[0] == ConvergenceStatus.STALLED

    def test_stall_callback_error_handled(self):
        config = ConvergenceMonitorConfig(window_size=5, stall_threshold_iterations=10)
        mon = ConvergenceMonitor(config)
        mon.register_stall_callback(lambda s: (_ for _ in ()).throw(RuntimeError("boom")))

        mon.record_metrics(iteration=0, total_loss=1.0)
        for i in range(1, 20):
            mon.record_metrics(iteration=i, total_loss=1.0)
        # Should not raise
        mon.is_stalled()

    def test_diverging_status(self):
        config = ConvergenceMonitorConfig(
            window_size=20,
            stall_threshold_iterations=100,
            divergence_threshold=0.1,
        )
        mon = ConvergenceMonitor(config)
        # First half of window at 1.0, second half at 1.5 => diverging
        for i in range(10):
            mon.record_metrics(iteration=i, total_loss=1.0)
        for i in range(10, 25):
            mon.record_metrics(iteration=i, total_loss=1.5)

        assert mon.is_diverging() is True

    def test_improving_status_via_loss(self):
        config = ConvergenceMonitorConfig(window_size=10, stall_threshold_iterations=100)
        mon = ConvergenceMonitor(config)
        # Steadily decreasing losses
        for i in range(15):
            mon.record_metrics(iteration=i, total_loss=1.0 - i * 0.05)
        status = mon.get_status()
        assert status == ConvergenceStatus.IMPROVING

    def test_improving_status_via_win_rate(self):
        config = ConvergenceMonitorConfig(
            window_size=10,
            stall_threshold_iterations=100,
            improvement_threshold=0.01,
        )
        mon = ConvergenceMonitor(config)
        for i in range(15):
            mon.record_metrics(iteration=i, win_rate=0.4 + i * 0.02)
        status = mon.get_status()
        assert status == ConvergenceStatus.IMPROVING

    def test_slowing_status(self):
        config = ConvergenceMonitorConfig(
            window_size=10,
            stall_threshold_iterations=20,
            improvement_threshold=0.01,
        )
        mon = ConvergenceMonitor(config)
        # Initial improvement, then flat (but not stalled enough)
        mon.record_metrics(iteration=0, total_loss=2.0)
        for i in range(1, 20):
            # Flat — no improvement after iteration 0
            mon.record_metrics(iteration=i, total_loss=2.0)
        status = mon.get_status()
        assert status == ConvergenceStatus.SLOWING

    def test_is_stalled_returns_false_when_not_stalled(self, monitor):
        monitor.record_metrics(iteration=1, total_loss=1.0)
        assert monitor.is_stalled() is False

    def test_is_diverging_returns_false_when_not(self, monitor):
        monitor.record_metrics(iteration=1, total_loss=1.0)
        assert monitor.is_diverging() is False

    # -----------------------------------------------------------------------
    # Recommendations
    # -----------------------------------------------------------------------

    def test_recommendations_improving(self):
        config = ConvergenceMonitorConfig(window_size=10, stall_threshold_iterations=100)
        mon = ConvergenceMonitor(config)
        for i in range(15):
            mon.record_metrics(iteration=i, total_loss=1.0 - i * 0.05)
        recs = mon.get_recommendations()
        assert any(r.type == RecommendationType.CONTINUE for r in recs)

    def test_recommendations_stalled_with_lr(self):
        config = ConvergenceMonitorConfig(window_size=5, stall_threshold_iterations=10)
        mon = ConvergenceMonitor(config)
        mon.record_metrics(iteration=0, total_loss=1.0, learning_rate=0.01)
        for i in range(1, 20):
            mon.record_metrics(iteration=i, total_loss=1.0, learning_rate=0.01)
        recs = mon.get_recommendations()
        types = {r.type for r in recs}
        assert RecommendationType.REDUCE_LR in types

    def test_recommendations_stalled_with_exploration(self):
        config = ConvergenceMonitorConfig(window_size=5, stall_threshold_iterations=10)
        mon = ConvergenceMonitor(config)
        mon.record_metrics(iteration=0, total_loss=1.0, exploration_rate=0.5)
        for i in range(1, 20):
            mon.record_metrics(iteration=i, total_loss=1.0, exploration_rate=0.5)
        recs = mon.get_recommendations()
        types = {r.type for r in recs}
        assert RecommendationType.INCREASE_EXPLORATION in types

    def test_recommendations_stalled_pbt(self):
        config = ConvergenceMonitorConfig(
            window_size=5,
            stall_threshold_iterations=10,
            enable_pbt_recommendations=True,
        )
        mon = ConvergenceMonitor(config)
        mon.record_metrics(iteration=0, total_loss=1.0)
        for i in range(1, 20):
            mon.record_metrics(iteration=i, total_loss=1.0)
        recs = mon.get_recommendations()
        types = {r.type for r in recs}
        assert RecommendationType.PBT_MUTATE in types

    def test_recommendations_stalled_pbt_disabled(self):
        config = ConvergenceMonitorConfig(
            window_size=5,
            stall_threshold_iterations=10,
            enable_pbt_recommendations=False,
        )
        mon = ConvergenceMonitor(config)
        mon.record_metrics(iteration=0, total_loss=1.0)
        for i in range(1, 20):
            mon.record_metrics(iteration=i, total_loss=1.0)
        recs = mon.get_recommendations()
        types = {r.type for r in recs}
        assert RecommendationType.PBT_MUTATE not in types

    def test_recommendations_slowing_gives_increase_simulations(self):
        config = ConvergenceMonitorConfig(
            window_size=10,
            stall_threshold_iterations=20,
        )
        mon = ConvergenceMonitor(config)
        mon.record_metrics(iteration=0, total_loss=2.0, learning_rate=0.01)
        for i in range(1, 20):
            mon.record_metrics(iteration=i, total_loss=2.0, learning_rate=0.01)
        recs = mon.get_recommendations()
        types = {r.type for r in recs}
        assert RecommendationType.INCREASE_SIMULATIONS in types

    def test_recommendations_curriculum_advance(self):
        config = ConvergenceMonitorConfig(
            window_size=5,
            stall_threshold_iterations=100,
            target_win_rate=0.7,
            enable_curriculum_recommendations=True,
        )
        mon = ConvergenceMonitor(config)
        # Flat loss so status is STABLE, but high win_rate => CURRICULUM_ADVANCE
        mon.record_metrics(iteration=0, total_loss=1.0, win_rate=0.8)
        for i in range(1, 10):
            mon.record_metrics(iteration=i, total_loss=1.0, win_rate=0.8)
        recs = mon.get_recommendations()
        types = {r.type for r in recs}
        assert RecommendationType.CURRICULUM_ADVANCE in types

    def test_recommendations_curriculum_retreat(self):
        config = ConvergenceMonitorConfig(
            window_size=5,
            stall_threshold_iterations=10,
            min_win_rate=0.5,
            enable_curriculum_recommendations=True,
        )
        mon = ConvergenceMonitor(config)
        mon.record_metrics(iteration=0, total_loss=1.0, win_rate=0.3)
        for i in range(1, 20):
            mon.record_metrics(iteration=i, total_loss=1.0, win_rate=0.3)
        recs = mon.get_recommendations()
        types = {r.type for r in recs}
        assert RecommendationType.CURRICULUM_RETREAT in types

    def test_recommendations_curriculum_disabled(self):
        config = ConvergenceMonitorConfig(
            window_size=5,
            stall_threshold_iterations=100,
            enable_curriculum_recommendations=False,
        )
        mon = ConvergenceMonitor(config)
        for i in range(10):
            mon.record_metrics(iteration=i, total_loss=1.0, win_rate=0.9)
        recs = mon.get_recommendations()
        types = {r.type for r in recs}
        assert RecommendationType.CURRICULUM_ADVANCE not in types

    def test_recommendations_diverging(self):
        config = ConvergenceMonitorConfig(
            window_size=20,
            stall_threshold_iterations=100,
            divergence_threshold=0.1,
        )
        mon = ConvergenceMonitor(config)
        for i in range(10):
            mon.record_metrics(iteration=i, total_loss=1.0, learning_rate=0.01)
        for i in range(10, 25):
            mon.record_metrics(iteration=i, total_loss=1.5, learning_rate=0.01)
        recs = mon.get_recommendations()
        types = {r.type for r in recs}
        assert RecommendationType.REDUCE_LR in types

    def test_recommendations_early_stop(self):
        config = ConvergenceMonitorConfig(
            window_size=5,
            stall_threshold_iterations=10,
        )
        mon = ConvergenceMonitor(config)
        mon.record_metrics(iteration=0, total_loss=1.0)
        # iterations_since_improvement > stall_threshold * 2
        for i in range(1, 30):
            mon.record_metrics(iteration=i, total_loss=1.0)
        recs = mon.get_recommendations()
        types = {r.type for r in recs}
        assert RecommendationType.EARLY_STOP in types

    def test_recommendations_sorted_by_priority(self):
        config = ConvergenceMonitorConfig(
            window_size=5,
            stall_threshold_iterations=10,
            enable_pbt_recommendations=True,
        )
        mon = ConvergenceMonitor(config)
        mon.record_metrics(iteration=0, total_loss=1.0, learning_rate=0.01, exploration_rate=0.5)
        for i in range(1, 20):
            mon.record_metrics(
                iteration=i, total_loss=1.0, learning_rate=0.01, exploration_rate=0.5
            )
        recs = mon.get_recommendations()
        priorities = [r.priority for r in recs]
        assert priorities == sorted(priorities, reverse=True)

    # -----------------------------------------------------------------------
    # Improvement rate
    # -----------------------------------------------------------------------

    def test_get_improvement_rate_not_enough_data(self, monitor):
        assert monitor.get_improvement_rate() is None

    def test_get_improvement_rate_not_enough_losses(self):
        config = ConvergenceMonitorConfig(window_size=5)
        mon = ConvergenceMonitor(config)
        for i in range(10):
            mon.record_metrics(iteration=i, win_rate=0.5)
        # No total_loss recorded
        result = mon.get_improvement_rate()
        assert result is None

    def test_get_improvement_rate_zero_first_avg(self):
        config = ConvergenceMonitorConfig(window_size=10)
        mon = ConvergenceMonitor(config)
        for i in range(15):
            mon.record_metrics(iteration=i, total_loss=0.0)
        result = mon.get_improvement_rate()
        assert result is None

    def test_get_improvement_rate_improving(self):
        config = ConvergenceMonitorConfig(window_size=10)
        mon = ConvergenceMonitor(config)
        for i in range(15):
            mon.record_metrics(iteration=i, total_loss=1.0 - i * 0.03)
        rate = mon.get_improvement_rate()
        assert rate is not None
        assert rate < 0  # Loss decreasing = negative rate

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def test_get_best_metrics(self, monitor):
        monitor.record_metrics(iteration=1, total_loss=0.5, win_rate=0.6)
        best = monitor.get_best_metrics()
        assert best["best_loss"] == 0.5
        assert best["best_win_rate"] == 0.6
        assert best["best_iteration"] == 1

    def test_get_statistics(self, monitor):
        monitor.record_metrics(iteration=1, total_loss=0.5)
        stats = monitor.get_statistics()
        assert stats["history_size"] == 1
        assert "current_status" in stats

    def test_reset(self, monitor):
        monitor.record_metrics(iteration=1, total_loss=0.5)
        monitor.reset()
        assert monitor._best_loss is None
        assert monitor._best_win_rate is None
        assert len(monitor._history) == 0

    def test_repr(self, monitor):
        r = repr(monitor)
        assert "ConvergenceMonitor" in r
        assert "status=" in r

    def test_no_config_uses_default(self):
        mon = ConvergenceMonitor()
        assert mon.config is not None
        assert mon.config.window_size >= 5

    def test_total_recommendations_tracked(self):
        config = ConvergenceMonitorConfig(window_size=5, stall_threshold_iterations=10)
        mon = ConvergenceMonitor(config)
        mon.record_metrics(iteration=0, total_loss=1.0)
        for i in range(1, 20):
            mon.record_metrics(iteration=i, total_loss=1.0)
        recs = mon.get_recommendations()
        assert mon._total_recommendations == len(recs)

    def test_diverging_with_no_learning_rate(self):
        config = ConvergenceMonitorConfig(
            window_size=20,
            stall_threshold_iterations=100,
            divergence_threshold=0.1,
        )
        mon = ConvergenceMonitor(config)
        for i in range(10):
            mon.record_metrics(iteration=i, total_loss=1.0)
        for i in range(10, 25):
            mon.record_metrics(iteration=i, total_loss=1.5)
        recs = mon.get_recommendations()
        # Should still get REDUCE_LR with fallback 0.001 * 0.1
        lr_recs = [r for r in recs if r.type == RecommendationType.REDUCE_LR]
        assert len(lr_recs) >= 1
        assert lr_recs[0].suggested_value is not None
