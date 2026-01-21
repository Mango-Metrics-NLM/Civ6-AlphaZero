"""Unit tests for the HyperparameterScheduler module.

Tests all schedule types and scheduler functionality including:
- Schedule value computation (constant, linear, cosine, step, warmup_cosine, exponential)
- Scheduler state management
- Schedule preview functionality
- Edge cases and boundary conditions
"""

from __future__ import annotations

import math

import pytest

from alpha_zero.orchestrator.config import ScheduleConfig
from alpha_zero.orchestrator.scheduler import HyperparameterScheduler, SchedulerState


class TestSchedulerState:
    """Tests for SchedulerState dataclass."""

    def test_default_values(self):
        """Test default state values."""
        state = SchedulerState()
        assert state.current_step == 0
        assert state.learning_rate == 0.001
        assert state.batch_size == 64
        assert state.temperature == 1.0
        assert state.exploration == 0.25

    def test_custom_values(self):
        """Test custom state values."""
        state = SchedulerState(
            current_step=100,
            learning_rate=0.01,
            batch_size=128,
            temperature=0.5,
            exploration=0.1,
        )
        assert state.current_step == 100
        assert state.learning_rate == 0.01
        assert state.batch_size == 128
        assert state.temperature == 0.5
        assert state.exploration == 0.1


class TestHyperparameterSchedulerInit:
    """Tests for HyperparameterScheduler initialization."""

    def test_default_initialization(self):
        """Test scheduler with default values."""
        scheduler = HyperparameterScheduler()
        assert scheduler.current_step == 0
        assert scheduler.get_learning_rate() == 0.001
        assert scheduler.get_batch_size() == 64
        assert scheduler.get_temperature() == 1.0
        assert scheduler.get_exploration() == 0.25

    def test_custom_schedules(self):
        """Test scheduler with custom schedule configs."""
        lr_schedule = ScheduleConfig(
            schedule_type="constant",
            initial_value=0.01,
        )
        batch_schedule = ScheduleConfig(
            schedule_type="constant",
            initial_value=128,
        )

        scheduler = HyperparameterScheduler(
            lr_schedule=lr_schedule,
            batch_size_schedule=batch_schedule,
        )

        assert scheduler.get_learning_rate() == 0.01
        assert scheduler.get_batch_size() == 128

    def test_mixed_schedule_types(self):
        """Test scheduler with different schedule types for each parameter."""
        scheduler = HyperparameterScheduler(
            lr_schedule=ScheduleConfig(
                schedule_type="linear",
                initial_value=0.01,
                final_value=0.001,
                decay_steps=1000,
            ),
            batch_size_schedule=ScheduleConfig(
                schedule_type="step",
                initial_value=64,
                step_points=[
                    {"iteration": 500, "value": 128},
                    {"iteration": 1000, "value": 256},
                ],
            ),
            temperature_schedule=ScheduleConfig(
                schedule_type="cosine",
                initial_value=1.0,
                final_value=0.1,
                decay_steps=1000,
            ),
        )

        assert scheduler.lr_schedule.schedule_type == "linear"
        assert scheduler.batch_size_schedule.schedule_type == "step"
        assert scheduler.temperature_schedule.schedule_type == "cosine"


class TestSchedulerStepManagement:
    """Tests for step management methods."""

    def test_step_advances(self):
        """Test step() advances current step."""
        scheduler = HyperparameterScheduler()
        assert scheduler.current_step == 0

        scheduler.step()
        assert scheduler.current_step == 1

        scheduler.step(10)
        assert scheduler.current_step == 11

    def test_set_step(self):
        """Test set_step() sets step directly."""
        scheduler = HyperparameterScheduler()
        scheduler.set_step(500)
        assert scheduler.current_step == 500

        scheduler.set_step(0)
        assert scheduler.current_step == 0

    def test_reset(self):
        """Test reset() returns to initial state."""
        scheduler = HyperparameterScheduler()
        scheduler.step(100)
        assert scheduler.current_step == 100

        scheduler.reset()
        assert scheduler.current_step == 0


class TestConstantSchedule:
    """Tests for constant schedule type."""

    def test_constant_remains_fixed(self):
        """Test constant schedule returns same value at all steps."""
        scheduler = HyperparameterScheduler(
            lr_schedule=ScheduleConfig(
                schedule_type="constant",
                initial_value=0.005,
            )
        )

        assert scheduler.get_learning_rate(0) == 0.005
        assert scheduler.get_learning_rate(100) == 0.005
        assert scheduler.get_learning_rate(10000) == 0.005


class TestLinearSchedule:
    """Tests for linear schedule type."""

    def test_linear_interpolation(self):
        """Test linear schedule interpolates correctly."""
        scheduler = HyperparameterScheduler(
            lr_schedule=ScheduleConfig(
                schedule_type="linear",
                initial_value=1.0,
                final_value=0.0,
                decay_steps=100,
            )
        )

        assert scheduler.get_learning_rate(0) == 1.0
        assert scheduler.get_learning_rate(50) == pytest.approx(0.5)
        assert scheduler.get_learning_rate(100) == 0.0

    def test_linear_beyond_decay_steps(self):
        """Test linear schedule stays at final value after decay_steps."""
        scheduler = HyperparameterScheduler(
            lr_schedule=ScheduleConfig(
                schedule_type="linear",
                initial_value=1.0,
                final_value=0.1,
                decay_steps=100,
            )
        )

        # Use pytest.approx for floating point comparison
        assert scheduler.get_learning_rate(150) == pytest.approx(0.1, rel=1e-6)
        assert scheduler.get_learning_rate(1000) == pytest.approx(0.1, rel=1e-6)


class TestCosineSchedule:
    """Tests for cosine annealing schedule type."""

    def test_cosine_annealing(self):
        """Test cosine schedule follows cosine curve."""
        scheduler = HyperparameterScheduler(
            temperature_schedule=ScheduleConfig(
                schedule_type="cosine",
                initial_value=1.0,
                final_value=0.0,
                decay_steps=100,
            )
        )

        # At step 0, should be initial value
        assert scheduler.get_temperature(0) == 1.0

        # At step 50 (halfway), should be approximately 0.5
        # cosine(pi/2) = 0, so value = 0.5 * (1 + 0) = 0.5
        expected_mid = 0.0 + (1.0 - 0.0) * 0.5 * (1 + math.cos(math.pi * 0.5))
        assert scheduler.get_temperature(50) == pytest.approx(expected_mid, rel=0.01)

        # At step 100, should be final value
        assert scheduler.get_temperature(100) == pytest.approx(0.0, abs=1e-9)

    def test_cosine_beyond_decay_steps(self):
        """Test cosine schedule stays at final value after decay_steps."""
        scheduler = HyperparameterScheduler(
            temperature_schedule=ScheduleConfig(
                schedule_type="cosine",
                initial_value=1.0,
                final_value=0.1,
                decay_steps=100,
            )
        )

        assert scheduler.get_temperature(150) == pytest.approx(0.1, abs=1e-6)


class TestStepSchedule:
    """Tests for step schedule type."""

    def test_step_schedule_transitions(self):
        """Test step schedule changes at specified points."""
        scheduler = HyperparameterScheduler(
            batch_size_schedule=ScheduleConfig(
                schedule_type="step",
                initial_value=64,
                step_points=[
                    {"iteration": 100, "value": 128},
                    {"iteration": 200, "value": 256},
                    {"iteration": 300, "value": 512},
                ],
            )
        )

        # Before first step point
        assert scheduler.get_batch_size(0) == 64
        assert scheduler.get_batch_size(99) == 64

        # After first step point
        assert scheduler.get_batch_size(100) == 128
        assert scheduler.get_batch_size(199) == 128

        # After second step point
        assert scheduler.get_batch_size(200) == 256

        # After third step point
        assert scheduler.get_batch_size(300) == 512
        assert scheduler.get_batch_size(1000) == 512


class TestWarmupCosineSchedule:
    """Tests for warmup_cosine schedule type."""

    def test_warmup_phase(self):
        """Test warmup_cosine has linear warmup phase."""
        scheduler = HyperparameterScheduler(
            lr_schedule=ScheduleConfig(
                schedule_type="warmup_cosine",
                initial_value=0.001,
                final_value=0.0001,
                decay_steps=1000,
                warmup_steps=100,
            )
        )

        # At step 0, should be 0 (start of warmup)
        assert scheduler.get_learning_rate(0) == 0.0

        # At step 50 (halfway through warmup), should be half of initial
        assert scheduler.get_learning_rate(50) == pytest.approx(0.0005, rel=0.01)

        # At step 100 (end of warmup), should be initial value
        assert scheduler.get_learning_rate(100) == pytest.approx(0.001, rel=0.01)

    def test_cosine_phase_after_warmup(self):
        """Test warmup_cosine follows cosine after warmup."""
        scheduler = HyperparameterScheduler(
            lr_schedule=ScheduleConfig(
                schedule_type="warmup_cosine",
                initial_value=0.001,
                final_value=0.0001,
                decay_steps=1000,
                warmup_steps=100,
            )
        )

        # After warmup, should follow cosine schedule
        # At decay_steps, should be at final value
        assert scheduler.get_learning_rate(1000) == pytest.approx(0.0001, rel=0.01)


class TestExponentialSchedule:
    """Tests for exponential decay schedule type."""

    def test_exponential_decay(self):
        """Test exponential schedule decays correctly."""
        scheduler = HyperparameterScheduler(
            exploration_schedule=ScheduleConfig(
                schedule_type="exponential",
                initial_value=1.0,
                final_value=0.01,
                decay_steps=100,
            )
        )

        # At step 0, should be initial value
        assert scheduler.get_exploration(0) == 1.0

        # Should decay exponentially
        mid_value = scheduler.get_exploration(50)
        assert 0.01 < mid_value < 1.0

        # At decay_steps, should be near final value
        assert scheduler.get_exploration(100) == pytest.approx(0.01, rel=0.1)

    def test_exponential_beyond_decay_steps(self):
        """Test exponential schedule continues to decay beyond decay_steps."""
        scheduler = HyperparameterScheduler(
            exploration_schedule=ScheduleConfig(
                schedule_type="exponential",
                initial_value=1.0,
                final_value=0.01,
                decay_steps=100,
            )
        )

        # Exponential decay continues beyond decay_steps (no clamping)
        # At step 100, value should be ~0.01
        # At step 200, value should be ~0.01^2 = 0.0001
        value_at_100 = scheduler.get_exploration(100)
        value_at_200 = scheduler.get_exploration(200)

        assert value_at_100 == pytest.approx(0.01, rel=0.1)
        # Beyond decay_steps, it continues to decay
        assert value_at_200 < value_at_100


class TestGetAllValues:
    """Tests for get_all_values method."""

    def test_returns_all_parameters(self):
        """Test get_all_values returns all hyperparameters."""
        scheduler = HyperparameterScheduler()
        values = scheduler.get_all_values()

        assert "learning_rate" in values
        assert "batch_size" in values
        assert "temperature" in values
        assert "exploration" in values
        assert "step" in values

    def test_values_match_individual_getters(self):
        """Test get_all_values matches individual getter methods."""
        scheduler = HyperparameterScheduler()
        scheduler.set_step(50)

        values = scheduler.get_all_values()

        assert values["learning_rate"] == scheduler.get_learning_rate()
        assert values["batch_size"] == scheduler.get_batch_size()
        assert values["temperature"] == scheduler.get_temperature()
        assert values["exploration"] == scheduler.get_exploration()

    def test_values_at_specific_step(self):
        """Test get_all_values at specific step."""
        scheduler = HyperparameterScheduler()
        values = scheduler.get_all_values(step=100)

        assert values["step"] == 100


class TestGetState:
    """Tests for get_state method."""

    def test_returns_scheduler_state(self):
        """Test get_state returns SchedulerState object."""
        scheduler = HyperparameterScheduler()
        state = scheduler.get_state()

        assert isinstance(state, SchedulerState)

    def test_state_reflects_current_values(self):
        """Test state reflects current scheduler values."""
        scheduler = HyperparameterScheduler()
        scheduler.set_step(100)
        state = scheduler.get_state()

        assert state.current_step == 100
        assert state.learning_rate == scheduler.get_learning_rate()
        assert state.batch_size == scheduler.get_batch_size()


class TestGetScheduleInfo:
    """Tests for get_schedule_info method."""

    def test_returns_schedule_details(self):
        """Test get_schedule_info returns configuration details."""
        scheduler = HyperparameterScheduler(
            lr_schedule=ScheduleConfig(
                schedule_type="warmup_cosine",
                initial_value=0.001,
                final_value=0.0001,
                decay_steps=1000,
                warmup_steps=100,
            )
        )

        info = scheduler.get_schedule_info()

        assert "learning_rate" in info
        assert info["learning_rate"]["type"] == "warmup_cosine"
        assert info["learning_rate"]["initial"] == 0.001
        assert info["learning_rate"]["final"] == 0.0001
        assert info["learning_rate"]["decay_steps"] == 1000
        assert info["learning_rate"]["warmup_steps"] == 100


class TestPreviewSchedule:
    """Tests for preview_schedule method."""

    def test_preview_with_default_points(self):
        """Test preview_schedule generates points automatically."""
        scheduler = HyperparameterScheduler(
            lr_schedule=ScheduleConfig(
                schedule_type="linear",
                initial_value=1.0,
                final_value=0.0,
                decay_steps=100,
            )
        )

        preview = scheduler.preview_schedule("learning_rate", num_points=5)

        assert len(preview) == 5
        assert all(isinstance(p, tuple) and len(p) == 2 for p in preview)

    def test_preview_with_specific_steps(self):
        """Test preview_schedule with specified steps."""
        scheduler = HyperparameterScheduler(
            lr_schedule=ScheduleConfig(
                schedule_type="constant",
                initial_value=0.001,
            )
        )

        steps = [0, 50, 100, 200]
        preview = scheduler.preview_schedule("learning_rate", steps=steps)

        assert len(preview) == 4
        assert [p[0] for p in preview] == steps

    def test_preview_invalid_schedule_raises(self):
        """Test preview_schedule raises on invalid schedule name."""
        scheduler = HyperparameterScheduler()

        with pytest.raises(ValueError, match="Unknown schedule"):
            scheduler.preview_schedule("invalid_schedule")


class TestSchedulerRepr:
    """Tests for __repr__ method."""

    def test_repr_contains_info(self):
        """Test __repr__ contains useful information."""
        scheduler = HyperparameterScheduler()
        repr_str = repr(scheduler)

        assert "HyperparameterScheduler" in repr_str
        assert "step=0" in repr_str
        assert "lr=" in repr_str
        assert "batch_size=" in repr_str


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_decay_steps(self):
        """Test schedules with zero decay_steps."""
        scheduler = HyperparameterScheduler(
            lr_schedule=ScheduleConfig(
                schedule_type="linear",
                initial_value=0.001,
                final_value=0.0001,
                decay_steps=0,
            )
        )

        # With decay_steps=0, denominator becomes 1 (max(0, 1))
        # At step 0: progress = min(0/1, 1.0) = 0, returns initial_value
        assert scheduler.get_learning_rate(0) == 0.001
        # At step 1+: progress = min(1/1, 1.0) = 1, returns final_value
        assert scheduler.get_learning_rate(1) == pytest.approx(0.0001, rel=1e-6)

    def test_negative_step(self):
        """Test behavior with negative step."""
        scheduler = HyperparameterScheduler(
            lr_schedule=ScheduleConfig(
                schedule_type="linear",
                initial_value=1.0,
                final_value=0.0,
                decay_steps=100,
            )
        )

        # Should clamp to initial value
        value = scheduler.get_learning_rate(-10)
        assert value >= 0  # Should not be negative

    def test_batch_size_always_integer(self):
        """Test batch size is always an integer."""
        scheduler = HyperparameterScheduler(
            batch_size_schedule=ScheduleConfig(
                schedule_type="linear",
                initial_value=64,
                final_value=128,
                decay_steps=100,
            )
        )

        for step in [0, 25, 50, 75, 100]:
            batch_size = scheduler.get_batch_size(step)
            assert isinstance(batch_size, int)

    def test_uses_current_step_when_none(self):
        """Test getters use current_step when step is None."""
        scheduler = HyperparameterScheduler(
            lr_schedule=ScheduleConfig(
                schedule_type="linear",
                initial_value=1.0,
                final_value=0.0,
                decay_steps=100,
            )
        )

        scheduler.set_step(50)

        # Calling without step argument should use current_step
        assert scheduler.get_learning_rate() == scheduler.get_learning_rate(50)
