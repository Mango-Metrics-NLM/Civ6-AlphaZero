"""Unit tests for enhanced training configuration module."""

from __future__ import annotations

import pytest

from alpha_zero.utils.training_config_enhanced import (
    EnhancedTrainingConfig,
    ParameterSchedule,
    ResourceConfig,
    ResourceManager,
    SchedulePoint,
    ScheduleType,
    create_training_config,
    get_fast_training_config,
    get_production_training_config,
)


class TestSchedulePoint:
    """Tests for SchedulePoint."""

    def test_creation(self):
        """Test schedule point creation."""
        point = SchedulePoint(iteration=1000, value=0.001)

        assert point.iteration == 1000
        assert point.value == 0.001

    def test_to_dict(self):
        """Test to dictionary conversion."""
        point = SchedulePoint(iteration=500, value=0.5)

        data = point.to_dict()

        assert data["iteration"] == 500
        assert data["value"] == 0.5

    def test_from_dict(self):
        """Test from dictionary creation."""
        data = {"iteration": 2000, "value": 0.002}

        point = SchedulePoint.from_dict(data)

        assert point.iteration == 2000
        assert point.value == 0.002


class TestParameterSchedule:
    """Tests for ParameterSchedule."""

    def test_constant_schedule(self):
        """Test constant schedule."""
        schedule = ParameterSchedule(
            schedule_type=ScheduleType.CONSTANT,
            initial_value=0.001,
        )

        assert schedule.get_value(0) == 0.001
        assert schedule.get_value(1000) == 0.001
        assert schedule.get_value(100000) == 0.001

    def test_step_schedule(self):
        """Test step schedule."""
        schedule = ParameterSchedule(
            schedule_type=ScheduleType.STEP,
            initial_value=0.01,
            points=[
                SchedulePoint(iteration=1000, value=0.001),
                SchedulePoint(iteration=5000, value=0.0001),
            ],
        )

        assert schedule.get_value(0) == 0.01
        assert schedule.get_value(500) == 0.01
        assert schedule.get_value(1000) == 0.001
        assert schedule.get_value(3000) == 0.001
        assert schedule.get_value(5000) == 0.0001
        assert schedule.get_value(10000) == 0.0001

    def test_linear_schedule(self):
        """Test linear decay schedule."""
        schedule = ParameterSchedule(
            schedule_type=ScheduleType.LINEAR,
            initial_value=1.0,
            final_value=0.0,
            decay_steps=100,
        )

        assert schedule.get_value(0) == pytest.approx(1.0)
        assert schedule.get_value(50) == pytest.approx(0.5)
        assert schedule.get_value(100) == pytest.approx(0.0)
        assert schedule.get_value(200) == pytest.approx(0.0)  # Stays at final

    def test_exponential_schedule(self):
        """Test exponential decay schedule."""
        schedule = ParameterSchedule(
            schedule_type=ScheduleType.EXPONENTIAL,
            initial_value=1.0,
            decay_steps=100,
            decay_rate=0.1,
        )

        assert schedule.get_value(0) == pytest.approx(1.0)
        assert schedule.get_value(100) == pytest.approx(0.1)
        assert schedule.get_value(200) == pytest.approx(0.01)

    def test_cosine_schedule(self):
        """Test cosine annealing schedule."""
        schedule = ParameterSchedule(
            schedule_type=ScheduleType.COSINE,
            initial_value=1.0,
            final_value=0.0,
            decay_steps=100,
        )

        # At start: should be initial
        assert schedule.get_value(0) == pytest.approx(1.0)
        # At end: should be final
        assert schedule.get_value(100) == pytest.approx(0.0)
        # At middle: should be around 0.5
        assert 0.4 < schedule.get_value(50) < 0.6

    def test_warmup_cosine_schedule(self):
        """Test warmup + cosine schedule."""
        schedule = ParameterSchedule(
            schedule_type=ScheduleType.WARMUP_COSINE,
            initial_value=1.0,
            final_value=0.0,
            decay_steps=100,
            warmup_steps=20,
        )

        # During warmup: should be less than initial
        assert schedule.get_value(0) == pytest.approx(0.0)
        assert 0 < schedule.get_value(10) < 1.0
        # After warmup: should start cosine from initial
        assert schedule.get_value(20) == pytest.approx(1.0)
        # At end
        assert schedule.get_value(100) == pytest.approx(0.0)

    def test_to_dict(self):
        """Test schedule to dictionary."""
        schedule = ParameterSchedule(
            schedule_type=ScheduleType.LINEAR,
            initial_value=0.001,
            final_value=0.0001,
        )

        data = schedule.to_dict()

        assert data["schedule_type"] == "linear"
        assert data["initial_value"] == 0.001

    def test_from_dict(self):
        """Test schedule from dictionary."""
        data = {
            "schedule_type": "step",
            "initial_value": 0.01,
            "points": [
                {"iteration": 100, "value": 0.001},
            ],
        }

        schedule = ParameterSchedule.from_dict(data)

        assert schedule.schedule_type == ScheduleType.STEP
        assert len(schedule.points) == 1


class TestResourceConfig:
    """Tests for ResourceConfig."""

    def test_default_config(self):
        """Test default resource config."""
        config = ResourceConfig()

        assert config.max_memory_gb == 8.0
        assert config.gradient_checkpointing is False
        assert config.mixed_precision is False
        assert config.num_workers == 4

    def test_to_dict(self):
        """Test to dictionary."""
        config = ResourceConfig(max_memory_gb=16.0, mixed_precision=True)

        data = config.to_dict()

        assert data["max_memory_gb"] == 16.0
        assert data["mixed_precision"] is True

    def test_from_dict(self):
        """Test from dictionary."""
        data = {
            "max_memory_gb": 32.0,
            "gradient_checkpointing": True,
            "num_workers": 8,
        }

        config = ResourceConfig.from_dict(data)

        assert config.max_memory_gb == 32.0
        assert config.gradient_checkpointing is True
        assert config.num_workers == 8


class TestEnhancedTrainingConfig:
    """Tests for EnhancedTrainingConfig."""

    def test_default_config(self):
        """Test default training config."""
        config = EnhancedTrainingConfig()

        assert config.batch_size == 256
        assert config.learning_rate == 0.001
        assert config.num_iterations == 100000
        assert config.gradient_accumulation_steps == 1

    def test_get_batch_size_no_schedule(self):
        """Test batch size without schedule."""
        config = EnhancedTrainingConfig(batch_size=128)

        assert config.get_batch_size(0) == 128
        assert config.get_batch_size(1000) == 128

    def test_get_batch_size_with_schedule(self):
        """Test batch size with schedule."""
        schedule = ParameterSchedule(
            schedule_type=ScheduleType.STEP,
            initial_value=64,
            points=[SchedulePoint(iteration=1000, value=128)],
        )
        config = EnhancedTrainingConfig(
            batch_size=256,
            batch_size_schedule=schedule,
        )

        assert config.get_batch_size(0) == 64
        assert config.get_batch_size(1000) == 128

    def test_get_learning_rate_no_schedule(self):
        """Test learning rate without schedule."""
        config = EnhancedTrainingConfig(learning_rate=0.002)

        assert config.get_learning_rate(0) == 0.002
        assert config.get_learning_rate(50000) == 0.002

    def test_get_learning_rate_with_schedule(self):
        """Test learning rate with schedule."""
        schedule = ParameterSchedule(
            schedule_type=ScheduleType.LINEAR,
            initial_value=0.001,
            final_value=0.0001,
            decay_steps=10000,
        )
        config = EnhancedTrainingConfig(
            learning_rate=0.001,
            learning_rate_schedule=schedule,
        )

        assert config.get_learning_rate(0) == pytest.approx(0.001)
        assert config.get_learning_rate(10000) == pytest.approx(0.0001)

    def test_get_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = EnhancedTrainingConfig(
            batch_size=128,
            gradient_accumulation_steps=4,
        )

        assert config.get_effective_batch_size(0) == 512

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = EnhancedTrainingConfig()

        errors = config.validate()

        assert len(errors) == 0

    def test_validate_invalid_batch_size(self):
        """Test validation catches invalid batch size."""
        config = EnhancedTrainingConfig(batch_size=0)

        errors = config.validate()

        assert any("batch_size" in e for e in errors)

    def test_validate_invalid_learning_rate(self):
        """Test validation catches invalid learning rate."""
        config = EnhancedTrainingConfig(learning_rate=-0.001)

        errors = config.validate()

        assert any("learning_rate" in e for e in errors)

    def test_validate_invalid_buffer_sizes(self):
        """Test validation catches buffer size inconsistency."""
        config = EnhancedTrainingConfig(
            replay_buffer_size=100,
            min_replay_size=200,
        )

        errors = config.validate()

        assert any("replay_buffer" in e.lower() or "min_replay" in e.lower() for e in errors)

    def test_to_dict(self):
        """Test to dictionary."""
        config = EnhancedTrainingConfig(
            batch_size=512,
            learning_rate=0.002,
        )

        data = config.to_dict()

        assert data["batch_size"] == 512
        assert data["learning_rate"] == 0.002
        assert "resources" in data

    def test_from_dict(self):
        """Test from dictionary."""
        data = {
            "batch_size": 1024,
            "learning_rate": 0.0005,
            "num_iterations": 50000,
            "gradient_accumulation_steps": 2,
        }

        config = EnhancedTrainingConfig.from_dict(data)

        assert config.batch_size == 1024
        assert config.learning_rate == 0.0005
        assert config.num_iterations == 50000


class TestResourceManager:
    """Tests for ResourceManager."""

    @pytest.fixture
    def manager(self):
        """Create resource manager."""
        config = ResourceConfig(max_memory_gb=8.0)
        return ResourceManager(config)

    def test_estimate_memory_usage(self, manager):
        """Test memory usage estimation."""
        usage = manager.estimate_memory_usage(
            batch_size=64,
            spatial_size=(13, 16, 16),
            vector_size=64,
            model_params=1000000,
        )

        assert usage > 0
        assert isinstance(usage, float)

    def test_recommend_batch_size(self, manager):
        """Test batch size recommendation."""
        batch_size, accumulation = manager.recommend_batch_size(
            spatial_size=(13, 16, 16),
            vector_size=64,
            model_params=1000000,
            target_effective_batch=256,
        )

        assert batch_size > 0
        assert accumulation >= 1
        assert batch_size * accumulation <= 256 * 2  # Allow some tolerance

    def test_track_memory_usage(self, manager):
        """Test memory tracking."""
        manager.track_memory_usage(2.5)
        manager.track_memory_usage(3.0)
        manager.track_memory_usage(2.8)

        stats = manager.get_memory_stats()

        assert stats["current"] == 2.8
        assert stats["max"] == 3.0
        assert 2.5 <= stats["mean"] <= 3.0


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_training_config_basic(self):
        """Test basic config creation."""
        config = create_training_config(
            batch_size=128,
            learning_rate=0.002,
            num_iterations=1000,
            use_schedules=False,
        )

        assert config.batch_size == 128
        assert config.learning_rate == 0.002
        assert config.batch_size_schedule is None
        assert config.learning_rate_schedule is None

    def test_create_training_config_with_schedules(self):
        """Test config creation with schedules."""
        config = create_training_config(
            batch_size=256,
            learning_rate=0.001,
            num_iterations=10000,
            use_schedules=True,
        )

        assert config.learning_rate_schedule is not None
        assert config.batch_size_schedule is not None

        # Verify schedules work
        # Note: warmup starts at 0, so check after warmup period
        warmup_steps = config.learning_rate_schedule.warmup_steps
        lr_after_warmup = config.get_learning_rate(warmup_steps)
        lr_end = config.get_learning_rate(10000)
        assert lr_after_warmup > lr_end

    def test_create_training_config_invalid(self):
        """Test config creation with invalid params."""
        with pytest.raises(ValueError):
            create_training_config(
                batch_size=0,
                use_schedules=False,
            )

    def test_get_fast_training_config(self):
        """Test fast training config."""
        config = get_fast_training_config()

        assert config.batch_size == 64
        assert config.num_iterations == 1000
        assert config.games_per_iteration == 20

    def test_get_production_training_config(self):
        """Test production training config."""
        config = get_production_training_config()

        assert config.batch_size >= 256
        assert config.num_iterations >= 100000
        assert config.learning_rate_schedule is not None


class TestScheduleType:
    """Tests for ScheduleType enum."""

    def test_all_types_exist(self):
        """Test all schedule types exist."""
        assert hasattr(ScheduleType, "CONSTANT")
        assert hasattr(ScheduleType, "STEP")
        assert hasattr(ScheduleType, "LINEAR")
        assert hasattr(ScheduleType, "EXPONENTIAL")
        assert hasattr(ScheduleType, "COSINE")
        assert hasattr(ScheduleType, "WARMUP_COSINE")

    def test_type_values(self):
        """Test schedule type string values."""
        assert ScheduleType.CONSTANT.value == "constant"
        assert ScheduleType.STEP.value == "step"
        assert ScheduleType.LINEAR.value == "linear"
