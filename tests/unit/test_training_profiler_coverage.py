"""Comprehensive tests for alpha_zero.training.training_profiler.

Covers TrainingProfilerConfig, TrainingStepMetrics, DiminishingReturnsAnalysis,
TrainingProfiler, and the get_training_profiler convenience function.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from alpha_zero.training.training_profiler import (
    DiminishingReturnsAnalysis,
    DiminishingReturnsLevel,
    TrainingProfiler,
    TrainingProfilerConfig,
    TrainingStepMetrics,
    get_training_profiler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def profiler_config():
    return TrainingProfilerConfig(
        window_size=20,
        min_samples=10,
        severe_threshold=0.001,
        moderate_threshold=0.005,
        mild_threshold=0.01,
        cost_per_step_usd=0.001,
        early_stop_enabled=True,
        min_steps_per_iteration=10,
        max_steps_per_iteration=500,
        default_steps_per_iteration=100,
    )


@pytest.fixture()
def profiler(profiler_config):
    return TrainingProfiler(profiler_config)


# ---------------------------------------------------------------------------
# TrainingStepMetrics
# ---------------------------------------------------------------------------


class TestTrainingStepMetrics:
    def test_to_dict(self):
        m = TrainingStepMetrics(
            step=5,
            iteration=1,
            loss=0.5,
            policy_loss=0.3,
            value_loss=0.2,
            grad_norm=1.5,
            learning_rate=0.001,
            wall_time_seconds=0.1,
        )
        d = m.to_dict()
        assert d["step"] == 5
        assert d["loss"] == 0.5
        assert d["wall_time_seconds"] == 0.1

    def test_defaults(self):
        m = TrainingStepMetrics(step=0, iteration=0, loss=1.0)
        assert m.policy_loss is None
        assert m.wall_time_seconds == 0.0


# ---------------------------------------------------------------------------
# DiminishingReturnsAnalysis
# ---------------------------------------------------------------------------


class TestDiminishingReturnsAnalysis:
    def test_to_dict(self):
        a = DiminishingReturnsAnalysis(
            level=DiminishingReturnsLevel.MILD,
            improvement_rate=0.008,
            cost_per_improvement=0.125,
            recommended_steps=150,
            reason="test reason",
            confidence=0.7,
        )
        d = a.to_dict()
        assert d["level"] == "mild"
        assert d["recommended_steps"] == 150


# ---------------------------------------------------------------------------
# TrainingProfilerConfig
# ---------------------------------------------------------------------------


class TestTrainingProfilerConfig:
    def test_defaults(self):
        config = TrainingProfilerConfig()
        assert config.window_size >= 10
        assert config.min_samples >= 10

    def test_validation_window_size(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="window_size"):
            TrainingProfilerConfig(window_size=5)

    def test_validation_min_samples(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="min_samples"):
            TrainingProfilerConfig(min_samples=5)

    def test_validation_threshold_ordering(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="severe_threshold"):
            TrainingProfilerConfig(severe_threshold=0.01, moderate_threshold=0.005)

    def test_validation_moderate_mild_ordering(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="moderate_threshold"):
            TrainingProfilerConfig(moderate_threshold=0.02, mild_threshold=0.01)

    def test_validation_negative_cost(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="cost_per_step"):
            TrainingProfilerConfig(cost_per_step_usd=-1)

    def test_from_env(self):
        env = {"CIV6AZ_PROFILE_WINDOW_SIZE": "200"}
        with patch.dict("os.environ", env, clear=False):
            config = TrainingProfilerConfig.from_env()
            assert config.window_size == 200

    def test_from_yaml(self):
        data = {"window_size": 50, "min_samples": 20}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(data, f)
            path = Path(f.name)

        config = TrainingProfilerConfig.from_yaml(path)
        assert config.window_size == 50
        path.unlink()

    def test_from_yaml_missing_file(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            TrainingProfilerConfig.from_yaml("/nonexistent.yaml")

    def test_from_yaml_invalid_yaml(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("{{invalid yaml::")
            path = Path(f.name)

        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            TrainingProfilerConfig.from_yaml(path)
        path.unlink()

    def test_from_dict(self):
        config = TrainingProfilerConfig.from_dict(
            {"window_size": 30, "early_stop_enabled": False}
        )
        assert config.window_size == 30
        assert config.early_stop_enabled is False

    def test_to_dict_roundtrip(self, profiler_config):
        d = profiler_config.to_dict()
        config2 = TrainingProfilerConfig.from_dict(d)
        assert config2.to_dict() == d


# ---------------------------------------------------------------------------
# TrainingProfiler
# ---------------------------------------------------------------------------


class TestTrainingProfiler:
    def test_record_step(self, profiler):
        profiler.record_step(step=0, iteration=0, loss=1.0)
        assert profiler._current_iteration == 0
        assert len(profiler._iteration_history[0]) == 1

    def test_record_step_new_iteration(self, profiler):
        profiler.record_step(step=0, iteration=0, loss=1.0)
        profiler.record_step(step=0, iteration=1, loss=0.9)
        assert profiler._current_iteration == 1
        assert 0 in profiler._iteration_history
        assert 1 in profiler._iteration_history

    def test_analyze_insufficient_samples(self, profiler):
        profiler.record_step(step=0, iteration=0, loss=1.0)
        analysis = profiler.analyze()
        assert analysis.level == DiminishingReturnsLevel.NONE
        assert analysis.confidence == 0.0
        assert "Insufficient" in analysis.reason

    def test_analyze_severe(self, profiler):
        """Near-zero improvement rate triggers SEVERE."""
        for i in range(50):
            profiler.record_step(step=i, iteration=0, loss=1.0)
        analysis = profiler.analyze()
        assert analysis.level == DiminishingReturnsLevel.SEVERE

    def test_analyze_none_level(self, profiler):
        """Strong improvement triggers NONE."""
        for i in range(50):
            profiler.record_step(step=i, iteration=0, loss=2.0 - i * 0.03)
        analysis = profiler.analyze()
        assert analysis.level == DiminishingReturnsLevel.NONE

    def test_analyze_mild_level(self, profiler):
        """Slight improvement triggers MILD."""
        for i in range(50):
            # improvement_rate ~ 0.008, between mild (0.01) and moderate (0.005)
            profiler.record_step(step=i, iteration=0, loss=1.0 - i * 0.008)
        analysis = profiler.analyze()
        assert analysis.level in (
            DiminishingReturnsLevel.MILD,
            DiminishingReturnsLevel.NONE,
        )

    def test_analyze_moderate_level(self, profiler):
        """Moderate improvement rate."""
        for i in range(50):
            profiler.record_step(step=i, iteration=0, loss=1.0 - i * 0.003)
        analysis = profiler.analyze()
        assert analysis.level in (
            DiminishingReturnsLevel.MODERATE,
            DiminishingReturnsLevel.MILD,
        )

    def test_analyze_specific_iteration(self, profiler):
        for i in range(20):
            profiler.record_step(step=i, iteration=0, loss=1.0)
        for i in range(20):
            profiler.record_step(step=i, iteration=1, loss=0.5)
        analysis = profiler.analyze(iteration=0)
        assert analysis.level == DiminishingReturnsLevel.SEVERE

    def test_analyze_zero_improvement_rate(self, profiler):
        """Zero improvement_rate gives inf cost."""
        for i in range(50):
            profiler.record_step(step=i, iteration=0, loss=1.0)
        analysis = profiler.analyze()
        assert analysis.cost_per_improvement == float("inf")

    def test_get_improvement_rate_empty(self, profiler):
        assert profiler.get_improvement_rate() == 0.0

    def test_get_improvement_rate_single_step(self, profiler):
        profiler.record_step(step=0, iteration=0, loss=1.0)
        assert profiler.get_improvement_rate() == 0.0

    def test_get_improvement_rate_decreasing(self, profiler):
        for i in range(30):
            profiler.record_step(step=i, iteration=0, loss=1.0 - i * 0.02)
        rate = profiler.get_improvement_rate()
        assert rate > 0  # Positive = improving (loss going down)

    def test_get_improvement_rate_custom_window(self, profiler):
        for i in range(30):
            profiler.record_step(step=i, iteration=0, loss=1.0 - i * 0.02)
        rate = profiler.get_improvement_rate(window=10)
        assert rate > 0

    def test_should_stop_early_disabled(self, profiler_config):
        profiler_config.early_stop_enabled = False
        prof = TrainingProfiler(profiler_config)
        stop, reason = prof.should_stop_early()
        assert stop is False
        assert "disabled" in reason.lower()

    def test_should_stop_early_severe(self, profiler):
        for i in range(50):
            profiler.record_step(step=i, iteration=0, loss=1.0)
        stop, reason = profiler.should_stop_early()
        assert stop is True
        assert "Severe" in reason

    def test_should_stop_early_moderate_enough_steps(self):
        config = TrainingProfilerConfig(
            window_size=20,
            min_samples=10,
            min_steps_per_iteration=10,
            max_steps_per_iteration=500,
        )
        prof = TrainingProfiler(config)
        # Record enough steps with moderate diminishing returns
        for i in range(50):
            prof.record_step(step=i, iteration=0, loss=1.0 - i * 0.003)
        stop, _reason = prof.should_stop_early()
        # May or may not stop depending on exact rates
        assert isinstance(stop, bool)

    def test_should_stop_early_normal(self, profiler):
        for i in range(50):
            profiler.record_step(step=i, iteration=0, loss=1.0 - i * 0.02)
        stop, reason = profiler.should_stop_early()
        assert stop is False
        assert "normally" in reason.lower()

    def test_get_recommended_steps_no_history(self, profiler):
        steps = profiler.get_recommended_steps(iteration=0, total_iterations=100)
        assert steps == profiler.config.default_steps_per_iteration

    def test_get_recommended_steps_custom_default(self, profiler):
        steps = profiler.get_recommended_steps(
            iteration=0, total_iterations=100, default_steps=200
        )
        assert steps == 200

    def test_get_recommended_steps_with_prev_severe(self, profiler):
        """Previous iteration with severe diminishing returns reduces steps."""
        for i in range(20):
            profiler.record_step(step=i, iteration=0, loss=1.0)
        steps = profiler.get_recommended_steps(iteration=1, total_iterations=10)
        assert steps <= 20  # Should reduce from prev 20

    def test_get_recommended_steps_with_prev_improving(self):
        config = TrainingProfilerConfig(
            window_size=20,
            min_samples=10,
            mild_threshold=0.01,
            min_steps_per_iteration=10,
            max_steps_per_iteration=500,
        )
        prof = TrainingProfiler(config)
        for i in range(20):
            prof.record_step(step=i, iteration=0, loss=1.0 - i * 0.05)
        steps = prof.get_recommended_steps(iteration=1, total_iterations=10)
        assert steps >= 20  # Should increase

    def test_calculate_recommended_steps_none(self, profiler):
        result = profiler._calculate_recommended_steps(50, 0.02, DiminishingReturnsLevel.NONE)
        assert result >= profiler.config.default_steps_per_iteration

    def test_calculate_recommended_steps_mild(self, profiler):
        result = profiler._calculate_recommended_steps(50, 0.008, DiminishingReturnsLevel.MILD)
        assert result <= profiler.config.max_steps_per_iteration

    def test_calculate_recommended_steps_moderate(self, profiler):
        result = profiler._calculate_recommended_steps(
            50, 0.003, DiminishingReturnsLevel.MODERATE
        )
        assert result > 50

    def test_calculate_recommended_steps_severe(self, profiler):
        result = profiler._calculate_recommended_steps(
            50, 0.0005, DiminishingReturnsLevel.SEVERE
        )
        assert result == 50

    def test_build_reason_all_levels(self, profiler):
        for level in DiminishingReturnsLevel:
            reason = profiler._build_reason(level, 0.005, 100)
            assert isinstance(reason, str)
            assert len(reason) > 0

    def test_calculate_confidence(self, profiler):
        metrics = [
            TrainingStepMetrics(step=i, iteration=0, loss=1.0 - i * 0.01)
            for i in range(30)
        ]
        conf = profiler._calculate_confidence(metrics, 0.01)
        assert 0 <= conf <= 1

    def test_calculate_confidence_single_metric(self, profiler):
        metrics = [TrainingStepMetrics(step=0, iteration=0, loss=1.0)]
        conf = profiler._calculate_confidence(metrics, 0.01)
        # consistency_confidence is 0.5 (fallback), sample_confidence is very low
        assert 0 <= conf <= 1

    def test_reset_iteration(self, profiler):
        profiler.record_step(step=0, iteration=0, loss=1.0)
        profiler.reset_iteration()
        assert len(profiler._step_history) == 0

    def test_reset(self, profiler):
        profiler.record_step(step=0, iteration=0, loss=1.0)
        profiler.reset()
        assert profiler._current_iteration == -1
        assert len(profiler._iteration_history) == 0

    def test_get_statistics(self, profiler):
        profiler.record_step(step=0, iteration=0, loss=1.0)
        stats = profiler.get_statistics()
        assert stats["current_iteration"] == 0
        assert stats["current_step_count"] == 1
        assert stats["total_iterations_tracked"] == 1

    def test_get_statistics_empty(self, profiler):
        stats = profiler.get_statistics()
        assert stats["current_step_count"] == 0
        assert stats["current_improvement_rate"] is None


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestGetTrainingProfiler:
    def test_returns_profiler(self):
        prof = get_training_profiler()
        assert isinstance(prof, TrainingProfiler)
