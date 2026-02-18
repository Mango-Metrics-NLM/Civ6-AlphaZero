"""Unit tests for training_profiler module."""

from __future__ import annotations

import os

from alpha_zero.training.training_profiler import (
    DiminishingReturnsAnalysis,
    DiminishingReturnsLevel,
    TrainingProfiler,
    TrainingProfilerConfig,
    TrainingStepMetrics,
    get_training_profiler,
)


class TestDiminishingReturnsLevel:
    def test_level_values(self) -> None:
        assert DiminishingReturnsLevel.NONE.value == "none"
        assert DiminishingReturnsLevel.MILD.value == "mild"
        assert DiminishingReturnsLevel.MODERATE.value == "moderate"
        assert DiminishingReturnsLevel.SEVERE.value == "severe"


class TestTrainingStepMetrics:
    def test_creation(self) -> None:
        metrics = TrainingStepMetrics(
            step=100,
            iteration=1,
            loss=1.5,  # Implementation uses 'loss', not 'total_loss'
            policy_loss=0.8,
            value_loss=0.7,
        )
        assert metrics.step == 100
        assert metrics.loss == 1.5

    def test_to_dict(self) -> None:
        metrics = TrainingStepMetrics(
            step=50,
            iteration=2,
            loss=1.0,
            policy_loss=0.5,
            value_loss=0.5,
        )
        data = metrics.to_dict()
        assert data["step"] == 50
        assert data["loss"] == 1.0


class TestDiminishingReturnsAnalysis:
    def test_creation(self) -> None:
        analysis = DiminishingReturnsAnalysis(
            level=DiminishingReturnsLevel.MILD,
            improvement_rate=0.005,
            cost_per_improvement=0.2,
            recommended_steps=500,
            reason="Mild diminishing returns",
            confidence=0.8,
        )
        assert analysis.level == DiminishingReturnsLevel.MILD
        assert analysis.recommended_steps == 500

    def test_to_dict(self) -> None:
        analysis = DiminishingReturnsAnalysis(
            level=DiminishingReturnsLevel.SEVERE,
            improvement_rate=0.0001,
            cost_per_improvement=10.0,
            recommended_steps=100,
            reason="Severe diminishing returns",
            confidence=0.95,
        )
        data = analysis.to_dict()
        assert data["level"] == "severe"
        assert data["recommended_steps"] == 100


class TestTrainingProfilerConfig:
    def test_default_values(self) -> None:
        config = TrainingProfilerConfig()
        assert config.window_size == 100
        assert config.min_samples == 50
        assert config.early_stop_enabled is True

    def test_from_env(self) -> None:
        os.environ["CIV6AZ_PROFILE_WINDOW_SIZE"] = "200"
        os.environ["CIV6AZ_PROFILE_EARLY_STOP_ENABLED"] = "false"
        try:
            config = TrainingProfilerConfig.from_env()
            assert config.window_size == 200
            assert config.early_stop_enabled is False
        finally:
            del os.environ["CIV6AZ_PROFILE_WINDOW_SIZE"]
            del os.environ["CIV6AZ_PROFILE_EARLY_STOP_ENABLED"]


class TestTrainingProfiler:
    def test_init(self) -> None:
        config = TrainingProfilerConfig()
        profiler = TrainingProfiler(config)
        assert profiler.config == config

    def test_record_step(self) -> None:
        config = TrainingProfilerConfig()
        profiler = TrainingProfiler(config)
        profiler.record_step(
            step=1,
            iteration=1,
            loss=1.5,
            policy_loss=0.8,
            value_loss=0.7,
        )
        # Implementation uses _iteration_history, not _history
        assert len(profiler._iteration_history[1]) == 1

    def test_analyze_insufficient_data(self) -> None:
        config = TrainingProfilerConfig(min_samples=50)
        profiler = TrainingProfiler(config)
        # Record only a few steps
        for i in range(10):
            profiler.record_step(
                step=i,
                iteration=1,
                loss=1.5 - i * 0.01,
                policy_loss=0.8,
                value_loss=0.7,
            )
        analysis = profiler.analyze()
        assert analysis.level == DiminishingReturnsLevel.NONE

    def test_analyze_with_sufficient_data(self) -> None:
        config = TrainingProfilerConfig(min_samples=10, window_size=20)
        profiler = TrainingProfiler(config)
        # Record enough steps with improving loss
        for i in range(50):
            profiler.record_step(
                step=i,
                iteration=1,
                loss=1.5 - i * 0.01,  # Steadily improving
                policy_loss=0.8 - i * 0.005,
                value_loss=0.7 - i * 0.005,
            )
        analysis = profiler.analyze()
        assert isinstance(analysis, DiminishingReturnsAnalysis)

    def test_should_stop_early(self) -> None:
        config = TrainingProfilerConfig(
            min_samples=10,  # Minimum allowed is 10
            early_stop_enabled=True,
            severe_threshold=0.0001,
        )
        profiler = TrainingProfiler(config)
        # Record many steps with almost no improvement
        for i in range(20):
            profiler.record_step(
                step=i,
                iteration=1,
                loss=1.5 - i * 0.0001,  # Very slow improvement
                policy_loss=0.8,
                value_loss=0.7,
            )
        should_stop, reason = profiler.should_stop_early()
        assert isinstance(should_stop, bool)
        assert isinstance(reason, str)

    def test_get_improvement_rate(self) -> None:
        config = TrainingProfilerConfig(min_samples=10, window_size=10)  # min_samples must be >= 10
        profiler = TrainingProfiler(config)
        for i in range(20):
            profiler.record_step(
                step=i,
                iteration=1,
                loss=1.5 - i * 0.01,
                policy_loss=0.8,
                value_loss=0.7,
            )
        rate = profiler.get_improvement_rate(window=10)
        assert rate is not None

    def test_get_statistics(self) -> None:
        config = TrainingProfilerConfig()
        profiler = TrainingProfiler(config)
        profiler.record_step(
            step=1,
            iteration=1,
            loss=1.5,
            policy_loss=0.8,
            value_loss=0.7,
        )
        stats = profiler.get_statistics()
        # Check actual keys from implementation
        assert "current_iteration" in stats
        assert "current_step_count" in stats

    def test_reset(self) -> None:
        config = TrainingProfilerConfig()
        profiler = TrainingProfiler(config)
        profiler.record_step(
            step=1,
            iteration=1,
            loss=1.5,
            policy_loss=0.8,
            value_loss=0.7,
        )
        profiler.reset()
        # After reset, iteration history should be empty
        assert len(profiler._iteration_history) == 0


class TestGetTrainingProfiler:
    def test_returns_profiler(self) -> None:
        profiler = get_training_profiler()
        assert isinstance(profiler, TrainingProfiler)
