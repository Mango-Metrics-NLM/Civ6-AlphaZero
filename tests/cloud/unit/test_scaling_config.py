"""Unit tests for scaling_config module.

Tests budget tier configurations and environment variable support.
"""

from __future__ import annotations

import math
import os

import pytest

from alpha_zero.cloud.gcp.scaling_config import (
    BudgetTier,
    CycleTierConfig,
    EvaluationTierConfig,
    GPUSpec,
    ScalingTierConfig,
    SelfPlayTierConfig,
    TrainingTierConfig,
    get_scaling_config,
)


class TestGPUSpec:
    """Tests for GPUSpec dataclass."""

    def test_default_values(self) -> None:
        """Test GPUSpec with T4 GPU."""
        gpu = GPUSpec(accelerator_type="NVIDIA_TESLA_T4")
        assert gpu.accelerator_type == "NVIDIA_TESLA_T4"
        assert gpu.accelerator_count == 1
        assert gpu.use_spot is False

    def test_custom_values(self) -> None:
        """Test GPUSpec with custom values."""
        gpu = GPUSpec(
            accelerator_type="NVIDIA_L4",
            accelerator_count=2,
            use_spot=True,
        )
        assert gpu.accelerator_type == "NVIDIA_L4"
        assert gpu.accelerator_count == 2
        assert gpu.use_spot is True

    def test_to_dict(self) -> None:
        """Test GPUSpec conversion to dictionary."""
        gpu = GPUSpec(accelerator_type="NVIDIA_A100", accelerator_count=4)
        data = gpu.to_dict()
        assert data["accelerator_type"] == "NVIDIA_A100"
        assert data["accelerator_count"] == 4

    def test_from_dict(self) -> None:
        """Test GPUSpec creation from dictionary."""
        data = {"accelerator_type": "NVIDIA_L4", "accelerator_count": 2}
        gpu = GPUSpec.from_dict(data)
        assert gpu.accelerator_type == "NVIDIA_L4"
        assert gpu.accelerator_count == 2


class TestBudgetTier:
    """Tests for BudgetTier enum."""

    def test_budget_tier_values(self) -> None:
        """Test BudgetTier enum values."""
        assert BudgetTier.BUDGET.value == "budget"
        assert BudgetTier.STANDARD.value == "standard"
        assert BudgetTier.BURST.value == "burst"

    def test_budget_tier_from_string(self) -> None:
        """Test BudgetTier creation from string."""
        assert BudgetTier("budget") == BudgetTier.BUDGET
        assert BudgetTier("standard") == BudgetTier.STANDARD
        assert BudgetTier("burst") == BudgetTier.BURST


class TestSelfPlayTierConfig:
    """Tests for SelfPlayTierConfig dataclass."""

    def test_creation(self) -> None:
        """Test SelfPlayTierConfig creation."""
        config = SelfPlayTierConfig(
            worker_count=5,
            gpu_spec=GPUSpec("NVIDIA_TESLA_T4"),
            games_per_cycle=500,
        )
        assert config.worker_count == 5
        assert config.games_per_cycle == 500
        assert isinstance(config.gpu_spec, GPUSpec)

    def test_to_dict(self) -> None:
        """Test SelfPlayTierConfig conversion to dictionary."""
        config = SelfPlayTierConfig(
            worker_count=10,
            gpu_spec=GPUSpec("NVIDIA_L4"),
            games_per_cycle=1000,
        )
        data = config.to_dict()
        assert data["worker_count"] == 10
        assert data["games_per_cycle"] == 1000
        assert "gpu_spec" in data


class TestScalingTierConfig:
    """Tests for ScalingTierConfig dataclass."""

    def test_budget_preset(self) -> None:
        """Test Budget tier preset values."""
        config = ScalingTierConfig.budget_preset()
        assert config.tier == BudgetTier.BUDGET
        assert config.monthly_budget_usd == 650.0
        assert config.self_play.worker_count == 3
        assert config.self_play.games_per_cycle == 300
        assert config.training.training_steps_per_cycle == 500
        assert config.cycle.frequency_hours == 8.0

    def test_standard_preset(self) -> None:
        """Test Standard tier preset values."""
        config = ScalingTierConfig.standard_preset()
        assert config.tier == BudgetTier.STANDARD
        assert config.monthly_budget_usd == 1200.0
        assert config.self_play.worker_count == 8
        assert config.self_play.games_per_cycle == 800

    def test_burst_preset(self) -> None:
        """Test Burst tier preset values."""
        config = ScalingTierConfig.burst_preset()
        assert config.tier == BudgetTier.BURST
        assert math.isinf(config.monthly_budget_usd)
        assert config.self_play.worker_count == 15

    def test_from_env_budget_tier(self) -> None:
        """Test loading Budget tier from environment."""
        os.environ["CIV6AZ_SCALING_TIER"] = "budget"
        try:
            config = ScalingTierConfig.from_env()
            assert config.tier == BudgetTier.BUDGET
        finally:
            del os.environ["CIV6AZ_SCALING_TIER"]

    def test_to_dict_and_from_dict(self) -> None:
        """Test round-trip serialization."""
        original = ScalingTierConfig.standard_preset()
        data = original.to_dict()
        restored = ScalingTierConfig.from_dict(data)
        assert restored.tier == original.tier
        assert restored.monthly_budget_usd == original.monthly_budget_usd

    def test_get_scaling_config(self) -> None:
        """Test the get_scaling_config convenience function."""
        os.environ["CIV6AZ_SCALING_TIER"] = "budget"
        try:
            config = get_scaling_config()
            assert config.tier == BudgetTier.BUDGET
        finally:
            del os.environ["CIV6AZ_SCALING_TIER"]
