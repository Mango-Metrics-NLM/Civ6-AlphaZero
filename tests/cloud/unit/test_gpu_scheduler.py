"""
Unit tests for DynamicGPUScheduler component.

Tests cover:
- Configuration validation
- Training phase detection
- GPU selection logic
- T4 → V100 upgrade strategy
- Cost-aware selection
- Spot instance handling
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError
from alpha_zero.cloud.gcp.gpu_scheduler import (
    DynamicGPUScheduler,
    GPUSchedulerConfig,
    GPUSelection,
    TrainingPhase,
)


class TestGPUSchedulerConfig:
    """Tests for GPUSchedulerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = GPUSchedulerConfig()

        assert config.default_gpu == "NVIDIA_TESLA_T4"
        assert config.final_phase_gpu == "NVIDIA_TESLA_V100"
        assert config.final_phase_threshold == 0.9
        assert config.prefer_spot is True
        assert config.upgrade_enabled is True

    def test_config_from_env(self) -> None:
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "CIV6AZ_GPU_DEFAULT": "NVIDIA_L4",
                "CIV6AZ_GPU_FINAL_PHASE": "NVIDIA_TESLA_A100",
                "CIV6AZ_GPU_FINAL_PHASE_THRESHOLD": "0.85",
                "CIV6AZ_GPU_PREFER_SPOT": "false",
                "CIV6AZ_GPU_UPGRADE_ENABLED": "true",
            },
        ):
            config = GPUSchedulerConfig.from_env()

            assert config.default_gpu == "NVIDIA_L4"
            assert config.final_phase_gpu == "NVIDIA_TESLA_A100"
            assert config.final_phase_threshold == 0.85
            assert config.prefer_spot is False
            assert config.upgrade_enabled is True

    def test_config_from_yaml(self) -> None:
        """Test configuration from YAML file."""
        yaml_content = {
            "gpu_scheduler": {
                "default_gpu": "NVIDIA_L4",
                "final_phase_gpu": "NVIDIA_TESLA_A100",
                "final_phase_threshold": 0.8,
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            f.flush()

            config = GPUSchedulerConfig.from_yaml(f.name)

            assert config.default_gpu == "NVIDIA_L4"
            assert config.final_phase_gpu == "NVIDIA_TESLA_A100"
            assert config.final_phase_threshold == 0.8

        os.unlink(f.name)

    def test_config_from_yaml_missing_file(self) -> None:
        """Test error when YAML file doesn't exist."""
        with pytest.raises(ConfigurationError, match="not found"):
            GPUSchedulerConfig.from_yaml("/nonexistent/path.yaml")

    def test_config_from_dict(self) -> None:
        """Test configuration from dictionary."""
        data = {
            "default_gpu": "NVIDIA_L4",
            "final_phase_threshold": 0.85,
        }
        config = GPUSchedulerConfig.from_dict(data)

        assert config.default_gpu == "NVIDIA_L4"
        assert config.final_phase_threshold == 0.85

    def test_config_to_dict(self) -> None:
        """Test configuration serialization to dictionary."""
        config = GPUSchedulerConfig(
            default_gpu="NVIDIA_TESLA_T4",
            final_phase_gpu="NVIDIA_TESLA_V100",
        )
        data = config.to_dict()

        assert data["default_gpu"] == "NVIDIA_TESLA_T4"
        assert data["final_phase_gpu"] == "NVIDIA_TESLA_V100"
        assert "phase_gpu_mapping" in data
        assert "gpu_machine_types" in data

    def test_phase_threshold_validation_early(self) -> None:
        """Test validation of early_phase_end threshold."""
        with pytest.raises(ConfigurationError, match="between 0 and 1"):
            GPUSchedulerConfig(early_phase_end=1.5)

    def test_phase_threshold_validation_middle(self) -> None:
        """Test validation of middle_phase_end ordering."""
        with pytest.raises(ConfigurationError, match="between early_phase_end"):
            GPUSchedulerConfig(early_phase_end=0.6, middle_phase_end=0.5)

    def test_phase_threshold_validation_late(self) -> None:
        """Test validation of late_phase_end ordering."""
        with pytest.raises(ConfigurationError, match="between middle_phase_end"):
            GPUSchedulerConfig(middle_phase_end=0.9, late_phase_end=0.85)

    def test_final_phase_threshold_validation(self) -> None:
        """Test validation of final_phase_threshold."""
        with pytest.raises(ConfigurationError, match="between 0 and 1"):
            GPUSchedulerConfig(final_phase_threshold=1.5)

    def test_custom_phase_gpu_mapping(self) -> None:
        """Test custom GPU mapping for phases."""
        mapping = {
            "early": "NVIDIA_TESLA_T4",
            "middle": "NVIDIA_L4",
            "late": "NVIDIA_TESLA_V100",
            "final": "NVIDIA_TESLA_A100",
        }
        config = GPUSchedulerConfig(phase_gpu_mapping=mapping)

        assert config.phase_gpu_mapping["early"] == "NVIDIA_TESLA_T4"
        assert config.phase_gpu_mapping["final"] == "NVIDIA_TESLA_A100"

    def test_custom_gpu_machine_types(self) -> None:
        """Test custom machine type mapping."""
        machine_types = {"NVIDIA_TESLA_T4": "n1-standard-16"}
        config = GPUSchedulerConfig(gpu_machine_types=machine_types)

        assert config.gpu_machine_types["NVIDIA_TESLA_T4"] == "n1-standard-16"


class TestGPUSelection:
    """Tests for GPUSelection dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        selection = GPUSelection(
            gpu_type="NVIDIA_TESLA_T4",
            machine_type="n1-standard-8",
            use_spot=True,
            reason="Test reason",
            estimated_hourly_cost=0.35,
        )
        data = selection.to_dict()

        assert data["gpu_type"] == "NVIDIA_TESLA_T4"
        assert data["machine_type"] == "n1-standard-8"
        assert data["use_spot"] is True
        assert data["reason"] == "Test reason"
        assert data["estimated_hourly_cost"] == 0.35


class TestTrainingPhaseDetection:
    """Tests for training phase detection."""

    @pytest.fixture
    def config(self) -> GPUSchedulerConfig:
        """Create test configuration with standard phase thresholds."""
        return GPUSchedulerConfig(
            early_phase_end=0.5,
            middle_phase_end=0.8,
            late_phase_end=0.9,
        )

    @pytest.fixture
    def scheduler(self, config: GPUSchedulerConfig) -> DynamicGPUScheduler:
        """Create test scheduler."""
        return DynamicGPUScheduler(config)

    def test_early_phase(self, scheduler: DynamicGPUScheduler) -> None:
        """Test detection of early phase (0-50%)."""
        assert scheduler.get_training_phase(0, 1000) == TrainingPhase.EARLY
        assert scheduler.get_training_phase(250, 1000) == TrainingPhase.EARLY
        assert scheduler.get_training_phase(499, 1000) == TrainingPhase.EARLY

    def test_middle_phase(self, scheduler: DynamicGPUScheduler) -> None:
        """Test detection of middle phase (50-80%)."""
        assert scheduler.get_training_phase(500, 1000) == TrainingPhase.MIDDLE
        assert scheduler.get_training_phase(650, 1000) == TrainingPhase.MIDDLE
        assert scheduler.get_training_phase(799, 1000) == TrainingPhase.MIDDLE

    def test_late_phase(self, scheduler: DynamicGPUScheduler) -> None:
        """Test detection of late phase (80-90%)."""
        assert scheduler.get_training_phase(800, 1000) == TrainingPhase.LATE
        assert scheduler.get_training_phase(850, 1000) == TrainingPhase.LATE
        assert scheduler.get_training_phase(899, 1000) == TrainingPhase.LATE

    def test_final_phase(self, scheduler: DynamicGPUScheduler) -> None:
        """Test detection of final phase (90-100%)."""
        assert scheduler.get_training_phase(900, 1000) == TrainingPhase.FINAL
        assert scheduler.get_training_phase(950, 1000) == TrainingPhase.FINAL
        assert scheduler.get_training_phase(1000, 1000) == TrainingPhase.FINAL

    def test_zero_total_iterations(self, scheduler: DynamicGPUScheduler) -> None:
        """Test handling of zero total iterations."""
        assert scheduler.get_training_phase(0, 0) == TrainingPhase.EARLY

    def test_negative_total_iterations(self, scheduler: DynamicGPUScheduler) -> None:
        """Test handling of negative total iterations."""
        assert scheduler.get_training_phase(0, -100) == TrainingPhase.EARLY


class TestShouldUpgrade:
    """Tests for GPU upgrade decision logic."""

    @pytest.fixture
    def config(self) -> GPUSchedulerConfig:
        """Create test configuration."""
        return GPUSchedulerConfig(
            final_phase_threshold=0.9,
            upgrade_enabled=True,
        )

    @pytest.fixture
    def scheduler(self, config: GPUSchedulerConfig) -> DynamicGPUScheduler:
        """Create test scheduler."""
        return DynamicGPUScheduler(config)

    def test_should_upgrade_true(self, scheduler: DynamicGPUScheduler) -> None:
        """Test upgrade returns True at 90% threshold."""
        assert scheduler.should_upgrade(900, 1000) is True
        assert scheduler.should_upgrade(950, 1000) is True
        assert scheduler.should_upgrade(1000, 1000) is True

    def test_should_upgrade_false_before_threshold(
        self, scheduler: DynamicGPUScheduler
    ) -> None:
        """Test upgrade returns False before threshold."""
        assert scheduler.should_upgrade(0, 1000) is False
        assert scheduler.should_upgrade(500, 1000) is False
        assert scheduler.should_upgrade(899, 1000) is False

    def test_should_upgrade_disabled(self) -> None:
        """Test upgrade returns False when disabled."""
        config = GPUSchedulerConfig(upgrade_enabled=False)
        scheduler = DynamicGPUScheduler(config)

        assert scheduler.should_upgrade(950, 1000) is False

    def test_should_upgrade_zero_total(
        self, scheduler: DynamicGPUScheduler
    ) -> None:
        """Test upgrade returns False with zero total iterations."""
        assert scheduler.should_upgrade(0, 0) is False


class TestGPUSelectionLogic:
    """Tests for GPU selection logic."""

    @pytest.fixture
    def config(self) -> GPUSchedulerConfig:
        """Create test configuration."""
        return GPUSchedulerConfig(
            default_gpu="NVIDIA_TESLA_T4",
            final_phase_gpu="NVIDIA_TESLA_V100",
            final_phase_threshold=0.9,
            spot_for_selfplay=True,
            spot_for_training=False,
        )

    @pytest.fixture
    def scheduler(self, config: GPUSchedulerConfig) -> DynamicGPUScheduler:
        """Create test scheduler."""
        return DynamicGPUScheduler(config)

    def test_select_gpu_early_phase(
        self, scheduler: DynamicGPUScheduler
    ) -> None:
        """Test GPU selection in early phase."""
        selection = scheduler.select_gpu(
            job_type="training",
            current_iteration=100,
            total_iterations=1000,
        )

        assert selection.gpu_type == "NVIDIA_TESLA_T4"
        assert TrainingPhase.EARLY.value in selection.reason.lower()

    def test_select_gpu_final_phase_upgrade(
        self, scheduler: DynamicGPUScheduler
    ) -> None:
        """Test GPU upgrade in final phase."""
        selection = scheduler.select_gpu(
            job_type="training",
            current_iteration=950,
            total_iterations=1000,
        )

        assert selection.gpu_type == "NVIDIA_TESLA_V100"
        assert "final" in selection.reason.lower()

    def test_select_gpu_selfplay_spot(
        self, scheduler: DynamicGPUScheduler
    ) -> None:
        """Test spot instance selection for self-play."""
        selection = scheduler.select_gpu(
            job_type="selfplay",
            current_iteration=100,
            total_iterations=1000,
        )

        assert selection.use_spot is True

    def test_select_gpu_training_no_spot(
        self, scheduler: DynamicGPUScheduler
    ) -> None:
        """Test no spot for training jobs."""
        selection = scheduler.select_gpu(
            job_type="training",
            current_iteration=100,
            total_iterations=1000,
        )

        assert selection.use_spot is False

    def test_select_gpu_final_phase_no_spot(
        self, scheduler: DynamicGPUScheduler
    ) -> None:
        """Test spot disabled for final training phase."""
        # Enable spot for training to test override
        config = GPUSchedulerConfig(
            spot_for_training=True,
            final_phase_threshold=0.9,
        )
        scheduler = DynamicGPUScheduler(config)

        selection = scheduler.select_gpu(
            job_type="training",
            current_iteration=950,
            total_iterations=1000,
        )

        assert selection.use_spot is False
        assert "spot disabled" in selection.reason.lower()

    def test_select_gpu_machine_type(
        self, scheduler: DynamicGPUScheduler
    ) -> None:
        """Test correct machine type is selected."""
        selection = scheduler.select_gpu(
            job_type="training",
            current_iteration=100,
            total_iterations=1000,
        )

        # T4 should use n1-standard-8
        assert selection.machine_type == "n1-standard-8"

    def test_select_gpu_cost_estimate(
        self, scheduler: DynamicGPUScheduler
    ) -> None:
        """Test cost estimate is included."""
        selection = scheduler.select_gpu(
            job_type="training",
            current_iteration=100,
            total_iterations=1000,
        )

        # T4 non-spot cost
        assert selection.estimated_hourly_cost == pytest.approx(0.35, rel=0.01)


class TestBudgetAwareSelection:
    """Tests for budget-aware GPU selection."""

    @pytest.fixture
    def config(self) -> GPUSchedulerConfig:
        """Create test configuration with V100 as default for early phase."""
        return GPUSchedulerConfig(
            default_gpu="NVIDIA_TESLA_V100",
            # Override phase mapping to use V100 for early phase
            phase_gpu_mapping={
                "early": "NVIDIA_TESLA_V100",
                "middle": "NVIDIA_TESLA_V100",
                "late": "NVIDIA_TESLA_V100",
                "final": "NVIDIA_TESLA_A100",
            },
            gpu_costs={
                "NVIDIA_TESLA_T4": 0.35,
                "NVIDIA_TESLA_T4_SPOT": 0.11,
                "NVIDIA_TESLA_V100": 2.48,
                "NVIDIA_TESLA_V100_SPOT": 0.74,
            },
        )

    @pytest.fixture
    def scheduler(self, config: GPUSchedulerConfig) -> DynamicGPUScheduler:
        """Create test scheduler."""
        return DynamicGPUScheduler(config)

    def test_downgrade_when_budget_low(
        self, scheduler: DynamicGPUScheduler
    ) -> None:
        """Test GPU downgrade when budget is low."""
        selection = scheduler.select_gpu(
            job_type="training",
            current_iteration=100,
            total_iterations=1000,
            budget_remaining=0.50,  # Less than V100 cost
        )

        assert selection.gpu_type == "NVIDIA_TESLA_T4"
        assert "budget" in selection.reason.lower()

    def test_no_downgrade_when_budget_ok(
        self, scheduler: DynamicGPUScheduler
    ) -> None:
        """Test no downgrade when budget is sufficient."""
        selection = scheduler.select_gpu(
            job_type="training",
            current_iteration=100,
            total_iterations=1000,
            budget_remaining=10.0,  # Plenty of budget
        )

        # Should keep configured V100 (from phase mapping)
        assert selection.gpu_type == "NVIDIA_TESLA_V100"


class TestCostMonitorIntegration:
    """Tests for CostMonitor integration."""

    @pytest.fixture
    def config(self) -> GPUSchedulerConfig:
        """Create test configuration with V100 as phase GPU."""
        return GPUSchedulerConfig(
            default_gpu="NVIDIA_TESLA_V100",
            # Override phase mapping to use V100 for early phase
            phase_gpu_mapping={
                "early": "NVIDIA_TESLA_V100",
                "middle": "NVIDIA_TESLA_V100",
                "late": "NVIDIA_TESLA_V100",
                "final": "NVIDIA_TESLA_A100",
            },
        )

    def test_critical_budget_forces_cheapest(self, config: GPUSchedulerConfig) -> None:
        """Test that critical budget alert forces cheapest GPU."""
        # Create mock cost monitor
        mock_cost_monitor = MagicMock()
        mock_status = MagicMock()

        # Import AlertLevel for the mock
        from alpha_zero.cloud.gcp.cost_monitor import AlertLevel

        mock_status.alert_level = AlertLevel.CRITICAL
        mock_cost_monitor.get_cost_status.return_value = mock_status

        scheduler = DynamicGPUScheduler(config, cost_monitor=mock_cost_monitor)

        selection = scheduler.select_gpu(
            job_type="training",
            current_iteration=100,
            total_iterations=1000,
        )

        assert selection.gpu_type == "NVIDIA_TESLA_T4"
        assert selection.use_spot is True
        assert "cheapest" in selection.reason.lower()

    def test_normal_budget_no_override(self, config: GPUSchedulerConfig) -> None:
        """Test that normal budget doesn't override selection."""
        mock_cost_monitor = MagicMock()
        mock_status = MagicMock()

        from alpha_zero.cloud.gcp.cost_monitor import AlertLevel

        mock_status.alert_level = AlertLevel.NORMAL
        mock_cost_monitor.get_cost_status.return_value = mock_status

        scheduler = DynamicGPUScheduler(config, cost_monitor=mock_cost_monitor)

        selection = scheduler.select_gpu(
            job_type="training",
            current_iteration=100,
            total_iterations=1000,
        )

        # Should keep configured V100 (from phase mapping)
        assert selection.gpu_type == "NVIDIA_TESLA_V100"


class TestHelperMethods:
    """Tests for helper methods."""

    @pytest.fixture
    def scheduler(self) -> DynamicGPUScheduler:
        """Create test scheduler."""
        return DynamicGPUScheduler()

    def test_get_gpu_for_phase(self, scheduler: DynamicGPUScheduler) -> None:
        """Test getting GPU for specific phase."""
        assert scheduler.get_gpu_for_phase(TrainingPhase.EARLY) == "NVIDIA_TESLA_T4"
        assert scheduler.get_gpu_for_phase(TrainingPhase.FINAL) == "NVIDIA_TESLA_V100"

    def test_get_statistics(self, scheduler: DynamicGPUScheduler) -> None:
        """Test getting scheduler statistics."""
        stats = scheduler.get_statistics()

        assert "default_gpu" in stats
        assert "final_phase_gpu" in stats
        assert "upgrade_enabled" in stats
        assert "phase_gpu_mapping" in stats

    def test_repr(self, scheduler: DynamicGPUScheduler) -> None:
        """Test string representation."""
        repr_str = repr(scheduler)

        assert "DynamicGPUScheduler" in repr_str
        assert "default=" in repr_str
        assert "final=" in repr_str

    def test_find_cheaper_gpu(self, scheduler: DynamicGPUScheduler) -> None:
        """Test finding cheaper GPU option."""
        result = scheduler._find_cheaper_gpu(max_cost=1.0, use_spot=False)

        assert result is not None
        gpu_type, cost = result
        assert cost < 1.0
        assert gpu_type in ["NVIDIA_TESLA_T4", "NVIDIA_L4"]

    def test_get_cheapest_gpu(self, scheduler: DynamicGPUScheduler) -> None:
        """Test getting cheapest GPU."""
        result = scheduler._get_cheapest_gpu(use_spot=True)

        assert result is not None
        gpu_type, cost = result
        # T4 Spot should be cheapest
        assert gpu_type == "NVIDIA_TESLA_T4"
        assert cost == pytest.approx(0.11, rel=0.01)
