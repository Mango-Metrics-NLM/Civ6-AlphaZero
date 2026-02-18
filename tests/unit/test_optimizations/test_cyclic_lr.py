"""Unit tests for cyclic learning rate scheduler."""

import pytest

from alpha_zero.optimizations.cyclic_lr import (
    CycleInfo,
    CyclicLRConfig,
    CyclicLRScheduler,
    CyclicScheduleType,
)


class TestCyclicLRConfig:
    """Tests for CyclicLRConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CyclicLRConfig()
        assert config.schedule_type == "cosine_annealing"
        assert config.base_lr == 1e-5
        assert config.max_lr == 1e-3
        assert config.step_size_up == 5000

    def test_validation_base_lr(self):
        """Test validation of base_lr."""
        with pytest.raises(ValueError, match="base_lr must be > 0"):
            CyclicLRConfig(base_lr=0)
        with pytest.raises(ValueError, match="base_lr must be > 0"):
            CyclicLRConfig(base_lr=-0.001)

    def test_validation_max_lr(self):
        """Test validation of max_lr."""
        with pytest.raises(ValueError, match="max_lr must be > 0"):
            CyclicLRConfig(max_lr=0)
        with pytest.raises(ValueError, match="max_lr.*must be >= base_lr"):
            CyclicLRConfig(base_lr=0.01, max_lr=0.001)

    def test_validation_step_size(self):
        """Test validation of step_size_up."""
        with pytest.raises(ValueError, match="step_size_up must be >= 1"):
            CyclicLRConfig(step_size_up=0)

    def test_validation_warmup_steps(self):
        """Test validation of warmup_steps."""
        with pytest.raises(ValueError, match="warmup_steps must be >= 0"):
            CyclicLRConfig(warmup_steps=-10)

    def test_validation_t_mult(self):
        """Test validation of t_mult."""
        with pytest.raises(ValueError, match="t_mult must be >= 1.0"):
            CyclicLRConfig(t_mult=0.5)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "schedule_type": "triangular",
            "base_lr": 1e-6,
            "max_lr": 1e-2,
            "step_size_up": 2000,
        }
        config = CyclicLRConfig.from_dict(data)
        assert config.schedule_type == "triangular"
        assert config.base_lr == 1e-6
        assert config.max_lr == 1e-2
        assert config.step_size_up == 2000

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = CyclicLRConfig(schedule_type="warm_restarts")
        data = config.to_dict()
        assert data["schedule_type"] == "warm_restarts"
        assert "base_lr" in data
        assert "max_lr" in data


class TestCyclicScheduleType:
    """Tests for CyclicScheduleType enum."""

    def test_all_types_exist(self):
        """Test that all schedule types are defined."""
        assert CyclicScheduleType.TRIANGULAR.value == "triangular"
        assert CyclicScheduleType.TRIANGULAR2.value == "triangular2"
        assert CyclicScheduleType.EXP_RANGE.value == "exp_range"
        assert CyclicScheduleType.COSINE_ANNEALING.value == "cosine_annealing"
        assert CyclicScheduleType.WARM_RESTARTS.value == "warm_restarts"


class TestCyclicLRScheduler:
    """Tests for CyclicLRScheduler."""

    def test_initialization(self):
        """Test scheduler initialization."""
        config = CyclicLRConfig()
        scheduler = CyclicLRScheduler(config)
        assert scheduler.config is config

    def test_warmup_phase(self):
        """Test learning rate during warmup."""
        config = CyclicLRConfig(
            warmup_steps=1000,
            base_lr=1e-5,
            max_lr=1e-3,
        )
        scheduler = CyclicLRScheduler(config)

        # At step 0, LR should be 0 (or very small)
        lr_0 = scheduler.get_lr(0)
        assert lr_0 < config.max_lr

        # At middle of warmup
        lr_500 = scheduler.get_lr(500)
        assert lr_0 < lr_500 < config.max_lr

        # At end of warmup, should be close to max_lr
        lr_1000 = scheduler.get_lr(999)
        assert abs(lr_1000 - config.max_lr) < config.max_lr * 0.01

    def test_triangular_schedule(self):
        """Test triangular schedule oscillates correctly."""
        config = CyclicLRConfig(
            schedule_type="triangular",
            base_lr=1e-5,
            max_lr=1e-3,
            step_size_up=100,
            step_size_down=100,
            warmup_steps=0,
        )
        scheduler = CyclicLRScheduler(config)

        # At step 0 (cycle start), should be at base_lr
        lr_0 = scheduler.get_lr(0)
        assert abs(lr_0 - config.base_lr) < 1e-10

        # At step 100 (peak), should be at max_lr
        lr_100 = scheduler.get_lr(100)
        assert abs(lr_100 - config.max_lr) < 1e-10

        # At step 200 (cycle end), should be back at base_lr
        lr_200 = scheduler.get_lr(200)
        assert abs(lr_200 - config.base_lr) < 1e-10

    def test_triangular2_schedule(self):
        """Test triangular2 halves amplitude each cycle."""
        config = CyclicLRConfig(
            schedule_type="triangular2",
            base_lr=1e-10,  # Very small but positive
            max_lr=1.0,
            step_size_up=100,
            step_size_down=100,
            warmup_steps=0,
        )
        scheduler = CyclicLRScheduler(config)

        # Peak of first cycle
        peak1 = scheduler.get_lr(100)

        # Peak of second cycle (should be approximately half amplitude)
        peak2 = scheduler.get_lr(300)

        # The amplitude halves each cycle
        amplitude1 = peak1 - config.base_lr
        amplitude2 = peak2 - config.base_lr
        assert abs(amplitude2 - amplitude1 / 2) < 0.01

    def test_exp_range_schedule(self):
        """Test exp_range schedule decays max_lr."""
        config = CyclicLRConfig(
            schedule_type="exp_range",
            base_lr=1e-10,
            max_lr=1.0,
            step_size_up=100,
            step_size_down=100,
            gamma=0.9,
            warmup_steps=0,
        )
        scheduler = CyclicLRScheduler(config)

        # Peak of first cycle
        peak1 = scheduler.get_lr(100)
        assert abs(peak1 - config.max_lr) < 0.01  # First peak should be max_lr

        # Peak of second cycle (max_lr should be decayed by gamma)
        peak2 = scheduler.get_lr(300)

        # For exp_range, max_lr decays by gamma each cycle
        expected_peak2 = config.base_lr + (config.max_lr * config.gamma - config.base_lr)
        assert abs(peak2 - expected_peak2) < 0.01

    def test_cosine_annealing_schedule(self):
        """Test cosine annealing schedule."""
        config = CyclicLRConfig(
            schedule_type="cosine_annealing",
            base_lr=1e-5,
            max_lr=1.0,
            step_size_up=100,
            warmup_steps=0,
        )
        scheduler = CyclicLRScheduler(config)

        # At cycle start (step 0), should be at max_lr
        lr_0 = scheduler.get_lr(0)
        assert abs(lr_0 - config.max_lr) < 1e-6

        # At cycle midpoint, should be between max and base
        cycle_length = config.step_size_up * 2
        lr_mid = scheduler.get_lr(cycle_length // 2)
        assert config.base_lr < lr_mid < config.max_lr

    def test_warm_restarts_schedule(self):
        """Test SGDR warm restarts schedule."""
        config = CyclicLRConfig(
            schedule_type="warm_restarts",
            base_lr=1e-5,
            max_lr=1.0,
            step_size_up=100,
            t_mult=2.0,
            warmup_steps=0,
        )
        scheduler = CyclicLRScheduler(config)

        # At step 0, should be at max
        lr_0 = scheduler.get_lr(0)
        assert abs(lr_0 - config.max_lr) < 1e-6

        # At step 100 (restart), should be back at max
        lr_100 = scheduler.get_lr(100)
        assert abs(lr_100 - config.max_lr) < 1e-6

        # Second cycle is 200 steps, so step 300 is restart
        lr_300 = scheduler.get_lr(300)
        assert abs(lr_300 - config.max_lr) < 1e-6

    def test_lr_bounds(self):
        """Test LR stays within configured bounds."""
        for schedule in [
            "triangular",
            "triangular2",
            "exp_range",
            "cosine_annealing",
            "warm_restarts",
        ]:
            config = CyclicLRConfig(
                schedule_type=schedule,
                base_lr=1e-5,
                max_lr=1e-3,
                step_size_up=100,
                warmup_steps=0,
            )
            scheduler = CyclicLRScheduler(config)

            for step in range(0, 1000, 10):
                lr = scheduler.get_lr(step)
                assert config.base_lr <= lr <= config.max_lr + 1e-10

    def test_get_momentum(self):
        """Test momentum cycling."""
        config = CyclicLRConfig(
            schedule_type="triangular",
            base_lr=1e-5,
            max_lr=1e-3,
            step_size_up=100,
            warmup_steps=0,
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
        )
        scheduler = CyclicLRScheduler(config)

        # Momentum cycles inversely to LR
        # At high LR (step 100), momentum should be low
        mom_100 = scheduler.get_momentum(100)
        assert mom_100 is not None
        assert abs(mom_100 - config.base_momentum) < 0.01

        # At low LR (step 0), momentum should be high
        mom_0 = scheduler.get_momentum(0)
        assert mom_0 is not None
        assert abs(mom_0 - config.max_momentum) < 0.01

    def test_get_momentum_disabled(self):
        """Test momentum returns None when disabled."""
        config = CyclicLRConfig(cycle_momentum=False)
        scheduler = CyclicLRScheduler(config)
        assert scheduler.get_momentum(0) is None

    def test_get_cycle_info(self):
        """Test getting cycle information."""
        config = CyclicLRConfig(
            schedule_type="triangular",
            step_size_up=100,
            step_size_down=100,
            warmup_steps=50,
        )
        scheduler = CyclicLRScheduler(config)

        # During warmup
        info = scheduler.get_cycle_info(25)
        assert isinstance(info, CycleInfo)
        assert info.in_warmup is True
        assert info.phase == "warmup"

        # During ascending phase
        info = scheduler.get_cycle_info(100)  # 50 warmup + 50 ascending
        assert info.in_warmup is False
        assert info.phase == "ascending"

        # During descending phase
        info = scheduler.get_cycle_info(200)  # 50 warmup + 100 up + 50 down
        assert info.phase == "descending"

    def test_preview_schedule(self):
        """Test schedule preview."""
        config = CyclicLRConfig(step_size_up=100, warmup_steps=0)
        scheduler = CyclicLRScheduler(config)

        samples = scheduler.preview_schedule(num_steps=500, sample_interval=50)

        assert len(samples) == 10  # 500 / 50
        assert all(isinstance(s, tuple) and len(s) == 2 for s in samples)
        assert all(s[0] >= 0 and s[1] > 0 for s in samples)

    def test_repr(self):
        """Test string representation."""
        config = CyclicLRConfig(schedule_type="warm_restarts")
        scheduler = CyclicLRScheduler(config)
        repr_str = repr(scheduler)
        assert "warm_restarts" in repr_str
        assert "CyclicLRScheduler" in repr_str


class TestCyclicLRIntegration:
    """Integration tests for cyclic LR."""

    def test_full_training_simulation(self):
        """Test simulating a full training run."""
        config = CyclicLRConfig(
            schedule_type="cosine_annealing",
            base_lr=1e-5,
            max_lr=1e-3,
            step_size_up=1000,
            warmup_steps=500,
        )
        scheduler = CyclicLRScheduler(config)

        lrs = []
        for step in range(10000):
            lr = scheduler.get_lr(step)
            lrs.append(lr)

        # Check we got reasonable LR trajectory
        assert len(lrs) == 10000
        # During warmup (step 0), LR starts at 0 and ramps up
        # After warmup, LR should be in [base_lr, max_lr]
        assert all(0 <= lr <= config.max_lr for lr in lrs)
        # After warmup, check bounds
        assert all(config.base_lr <= lr <= config.max_lr for lr in lrs[500:])

        # Check warmup happened
        assert lrs[0] < lrs[250] < lrs[499]

    def test_all_schedules_work(self):
        """Test all schedule types run without error."""
        for schedule in CyclicScheduleType:
            config = CyclicLRConfig(
                schedule_type=schedule.value,
                warmup_steps=100,
            )
            scheduler = CyclicLRScheduler(config)

            # Should not raise
            for step in range(0, 1000, 100):
                lr = scheduler.get_lr(step)
                # During warmup, LR can be 0 at step 0
                assert lr >= 0
