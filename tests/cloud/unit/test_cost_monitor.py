"""
Unit tests for CostMonitor component.

Tests cover:
- Configuration validation
- Cost tracking and aggregation
- Budget threshold alerting
- Cost estimation
- Monitoring lifecycle
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from alpha_zero.cloud.gcp.cost_monitor import (
    AlertLevel,
    CostBreakdown,
    CostMonitor,
    CostMonitorConfig,
    CostStatus,
)
from alpha_zero.cloud.gcp.exceptions import BudgetExceededError, ConfigurationError


class TestCostMonitorConfig:
    """Tests for CostMonitorConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CostMonitorConfig()

        assert config.daily_budget_usd == 50.0
        assert config.total_budget_usd == 1000.0
        assert config.alert_threshold == 0.8
        assert config.critical_threshold == 0.95
        assert config.check_interval_seconds == 300
        assert config.billing_api_enabled is False

    def test_config_from_env(self) -> None:
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "CIV6AZ_COST_BUDGET_DAILY": "100.0",
                "CIV6AZ_COST_BUDGET_TOTAL": "2000.0",
                "CIV6AZ_COST_ALERT_THRESHOLD": "0.7",
                "CIV6AZ_COST_CRITICAL_THRESHOLD": "0.9",
                "CIV6AZ_COST_CHECK_INTERVAL_S": "600",
                "CIV6AZ_COST_BILLING_API_ENABLED": "true",
            },
        ):
            config = CostMonitorConfig.from_env()

            assert config.daily_budget_usd == 100.0
            assert config.total_budget_usd == 2000.0
            assert config.alert_threshold == 0.7
            assert config.critical_threshold == 0.9
            assert config.check_interval_seconds == 600
            assert config.billing_api_enabled is True

    def test_config_from_yaml(self) -> None:
        """Test configuration from YAML file."""
        yaml_content = {
            "cost_monitor": {
                "daily_budget_usd": 75.0,
                "total_budget_usd": 1500.0,
                "alert_threshold": 0.6,
                "critical_threshold": 0.85,
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()

            config = CostMonitorConfig.from_yaml(f.name)

            assert config.daily_budget_usd == 75.0
            assert config.total_budget_usd == 1500.0
            assert config.alert_threshold == 0.6
            assert config.critical_threshold == 0.85

        os.unlink(f.name)

    def test_config_from_yaml_missing_file(self) -> None:
        """Test error when YAML file doesn't exist."""
        with pytest.raises(ConfigurationError, match="not found"):
            CostMonitorConfig.from_yaml("/nonexistent/path.yaml")

    def test_config_from_dict(self) -> None:
        """Test configuration from dictionary."""
        data = {
            "daily_budget_usd": 80.0,
            "total_budget_usd": 1600.0,
        }
        config = CostMonitorConfig.from_dict(data)

        assert config.daily_budget_usd == 80.0
        assert config.total_budget_usd == 1600.0

    def test_config_to_dict(self) -> None:
        """Test configuration serialization to dictionary."""
        config = CostMonitorConfig(
            daily_budget_usd=50.0,
            total_budget_usd=1000.0,
        )
        data = config.to_dict()

        assert data["daily_budget_usd"] == 50.0
        assert data["total_budget_usd"] == 1000.0
        assert "gpu_prices" in data

    def test_budget_validation_negative(self) -> None:
        """Test that negative budget raises error."""
        with pytest.raises(ConfigurationError, match="positive"):
            CostMonitorConfig(daily_budget_usd=-10.0)

    def test_budget_validation_zero(self) -> None:
        """Test that zero budget raises error."""
        with pytest.raises(ConfigurationError, match="positive"):
            CostMonitorConfig(total_budget_usd=0.0)

    def test_threshold_validation_out_of_range(self) -> None:
        """Test that threshold outside 0-1 raises error."""
        with pytest.raises(ConfigurationError, match="between 0 and 1"):
            CostMonitorConfig(alert_threshold=1.5)

    def test_threshold_validation_order(self) -> None:
        """Test that alert >= critical raises error."""
        with pytest.raises(ConfigurationError, match="less than critical"):
            CostMonitorConfig(alert_threshold=0.9, critical_threshold=0.8)

    def test_gpu_prices_customization(self) -> None:
        """Test custom GPU prices."""
        custom_prices = {"NVIDIA_TESLA_T4": 0.50}
        config = CostMonitorConfig(gpu_prices=custom_prices)

        assert config.gpu_prices["NVIDIA_TESLA_T4"] == 0.50


class TestCostBreakdown:
    """Tests for CostBreakdown dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        breakdown = CostBreakdown(
            total_today_usd=10.0,
            total_project_usd=100.0,
            by_gpu_type={"NVIDIA_TESLA_T4": 50.0},
            by_job_type={"selfplay": 75.0},
            timestamp="2024-01-01T00:00:00",
        )
        data = breakdown.to_dict()

        assert data["total_today_usd"] == 10.0
        assert data["total_project_usd"] == 100.0
        assert data["by_gpu_type"]["NVIDIA_TESLA_T4"] == 50.0
        assert data["by_job_type"]["selfplay"] == 75.0


class TestCostStatus:
    """Tests for CostStatus dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        breakdown = CostBreakdown(
            total_today_usd=10.0,
            total_project_usd=100.0,
            by_gpu_type={},
            by_job_type={},
            timestamp="2024-01-01T00:00:00",
        )
        status = CostStatus(
            breakdown=breakdown,
            alert_level=AlertLevel.WARNING,
            daily_remaining_usd=40.0,
            total_remaining_usd=900.0,
            estimated_daily_rate_usd=50.0,
        )
        data = status.to_dict()

        assert data["alert_level"] == "warning"
        assert data["daily_remaining_usd"] == 40.0
        assert "breakdown" in data


class TestCostMonitor:
    """Tests for CostMonitor class."""

    @pytest.fixture
    def config(self) -> CostMonitorConfig:
        """Create test configuration."""
        return CostMonitorConfig(
            daily_budget_usd=100.0,
            total_budget_usd=500.0,
            alert_threshold=0.8,
            critical_threshold=0.95,
        )

    @pytest.fixture
    def monitor(self, config: CostMonitorConfig) -> CostMonitor:
        """Create test cost monitor."""
        return CostMonitor(config)

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        monitor = CostMonitor()
        assert monitor.config.daily_budget_usd == 50.0

    def test_init_custom_config(self, config: CostMonitorConfig) -> None:
        """Test initialization with custom config."""
        monitor = CostMonitor(config)
        assert monitor.config.daily_budget_usd == 100.0

    def test_record_cost(self, monitor: CostMonitor) -> None:
        """Test recording costs."""
        monitor.record_cost(10.0, gpu_type="NVIDIA_TESLA_T4", job_type="selfplay")

        breakdown = monitor.get_current_costs()
        assert breakdown.total_today_usd == 10.0
        assert breakdown.total_project_usd == 10.0
        assert breakdown.by_gpu_type["NVIDIA_TESLA_T4"] == 10.0
        assert breakdown.by_job_type["selfplay"] == 10.0

    def test_record_multiple_costs(self, monitor: CostMonitor) -> None:
        """Test recording multiple costs accumulates correctly."""
        monitor.record_cost(10.0, gpu_type="NVIDIA_TESLA_T4")
        monitor.record_cost(20.0, gpu_type="NVIDIA_TESLA_T4")
        monitor.record_cost(15.0, gpu_type="NVIDIA_L4")

        breakdown = monitor.get_current_costs()
        assert breakdown.total_today_usd == 45.0
        assert breakdown.by_gpu_type["NVIDIA_TESLA_T4"] == 30.0
        assert breakdown.by_gpu_type["NVIDIA_L4"] == 15.0

    def test_estimate_job_cost(self, monitor: CostMonitor) -> None:
        """Test job cost estimation."""
        # T4 is $0.35/hr by default
        cost = monitor.estimate_job_cost(
            gpu_type="NVIDIA_TESLA_T4",
            duration_hours=2.0,
            gpu_count=1,
            use_spot=False,
        )
        assert cost == pytest.approx(0.70, rel=0.01)

    def test_estimate_job_cost_spot(self, monitor: CostMonitor) -> None:
        """Test job cost estimation with spot instances."""
        # T4 Spot is $0.11/hr by default
        cost = monitor.estimate_job_cost(
            gpu_type="NVIDIA_TESLA_T4",
            duration_hours=2.0,
            gpu_count=1,
            use_spot=True,
        )
        assert cost == pytest.approx(0.22, rel=0.01)

    def test_estimate_job_cost_multiple_gpus(self, monitor: CostMonitor) -> None:
        """Test job cost estimation with multiple GPUs."""
        cost = monitor.estimate_job_cost(
            gpu_type="NVIDIA_TESLA_T4",
            duration_hours=1.0,
            gpu_count=4,
            use_spot=False,
        )
        assert cost == pytest.approx(1.40, rel=0.01)

    def test_get_cost_status(self, monitor: CostMonitor) -> None:
        """Test getting cost status."""
        monitor.record_cost(50.0)

        status = monitor.get_cost_status()
        assert status.breakdown.total_today_usd == 50.0
        assert status.daily_remaining_usd == 50.0  # 100 - 50
        assert status.total_remaining_usd == 450.0  # 500 - 50

    def test_can_afford_true(self, monitor: CostMonitor) -> None:
        """Test can_afford returns True when within budget."""
        monitor.record_cost(50.0)
        assert monitor.can_afford(40.0) is True

    def test_can_afford_false_daily(self, monitor: CostMonitor) -> None:
        """Test can_afford returns False when exceeds daily budget."""
        monitor.record_cost(90.0)
        assert monitor.can_afford(20.0) is False

    def test_can_afford_false_total(self, monitor: CostMonitor) -> None:
        """Test can_afford returns False when exceeds total budget."""
        monitor.record_cost(480.0)
        assert monitor.can_afford(30.0) is False


class TestAlertLevels:
    """Tests for alert level calculation."""

    @pytest.fixture
    def config(self) -> CostMonitorConfig:
        """Create test configuration."""
        return CostMonitorConfig(
            daily_budget_usd=100.0,
            total_budget_usd=1000.0,
            alert_threshold=0.8,
            critical_threshold=0.95,
        )

    @pytest.fixture
    def monitor(self, config: CostMonitorConfig) -> CostMonitor:
        """Create test cost monitor."""
        return CostMonitor(config)

    def test_alert_level_normal(self, monitor: CostMonitor) -> None:
        """Test normal alert level when under threshold."""
        monitor.record_cost(50.0)  # 50% of daily
        status = monitor.get_cost_status()
        assert status.alert_level == AlertLevel.NORMAL

    def test_alert_level_warning(self, monitor: CostMonitor) -> None:
        """Test warning alert level at 80% threshold."""
        monitor.record_cost(85.0)  # 85% of daily
        status = monitor.get_cost_status()
        assert status.alert_level == AlertLevel.WARNING

    def test_alert_level_critical(self, monitor: CostMonitor) -> None:
        """Test critical alert level at 95% threshold."""
        monitor.record_cost(97.0)  # 97% of daily
        status = monitor.get_cost_status()
        assert status.alert_level == AlertLevel.CRITICAL

    def test_alert_level_exceeded(self, monitor: CostMonitor) -> None:
        """Test budget exceeded alert level."""
        monitor.record_cost(105.0)  # Over 100%
        status = monitor.get_cost_status()
        assert status.alert_level == AlertLevel.BUDGET_EXCEEDED

    def test_alert_level_total_budget(self, monitor: CostMonitor) -> None:
        """Test alert level based on total budget."""
        monitor.record_cost(850.0)  # 85% of total (1000)
        status = monitor.get_cost_status()
        # Daily is over 100%, so should be BUDGET_EXCEEDED
        assert status.alert_level == AlertLevel.BUDGET_EXCEEDED


class TestBudgetEnforcement:
    """Tests for budget enforcement."""

    @pytest.fixture
    def config(self) -> CostMonitorConfig:
        """Create test configuration."""
        return CostMonitorConfig(
            daily_budget_usd=100.0,
            total_budget_usd=500.0,
        )

    @pytest.fixture
    def monitor(self, config: CostMonitorConfig) -> CostMonitor:
        """Create test cost monitor."""
        return CostMonitor(config)

    def test_check_budget_ok(self, monitor: CostMonitor) -> None:
        """Test check_budget passes when under budget."""
        monitor.record_cost(50.0)
        monitor.check_budget()  # Should not raise

    def test_check_budget_daily_exceeded(self, monitor: CostMonitor) -> None:
        """Test check_budget raises when daily budget exceeded."""
        monitor.record_cost(105.0)

        with pytest.raises(BudgetExceededError) as exc_info:
            monitor.check_budget()

        assert exc_info.value.budget_type == "daily"
        assert exc_info.value.budget_limit == 100.0
        assert exc_info.value.current_cost == 105.0

    def test_check_budget_total_exceeded(self, monitor: CostMonitor) -> None:
        """Test check_budget raises when total budget exceeded."""
        # Use a config where daily is high enough not to trigger first
        config = CostMonitorConfig(
            daily_budget_usd=1000.0,
            total_budget_usd=500.0,
        )
        monitor = CostMonitor(config)
        monitor.record_cost(550.0)

        with pytest.raises(BudgetExceededError) as exc_info:
            monitor.check_budget()

        assert exc_info.value.budget_type == "total"
        assert exc_info.value.budget_limit == 500.0


class TestAlertCallbacks:
    """Tests for alert callback functionality."""

    @pytest.fixture
    def config(self) -> CostMonitorConfig:
        """Create test configuration."""
        return CostMonitorConfig(
            daily_budget_usd=100.0,
            total_budget_usd=1000.0,
            alert_threshold=0.8,
            critical_threshold=0.95,
        )

    @pytest.fixture
    def monitor(self, config: CostMonitorConfig) -> CostMonitor:
        """Create test cost monitor."""
        return CostMonitor(config)

    def test_callback_on_alert_change(self, monitor: CostMonitor) -> None:
        """Test callback is called when alert level changes."""
        callback = MagicMock()
        monitor.register_alert_callback(callback)

        # Record cost to trigger warning (85%)
        monitor.record_cost(85.0)

        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == AlertLevel.WARNING
        assert isinstance(args[1], CostStatus)

    def test_callback_not_called_same_level(self, monitor: CostMonitor) -> None:
        """Test callback is not called when alert level unchanged."""
        callback = MagicMock()

        # First record to establish baseline
        monitor.record_cost(50.0)

        # Register callback after first cost
        monitor.register_alert_callback(callback)

        # Another normal cost - should not trigger callback
        monitor.record_cost(10.0)

        callback.assert_not_called()

    def test_multiple_callbacks(self, monitor: CostMonitor) -> None:
        """Test multiple callbacks are all called."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        monitor.register_alert_callback(callback1)
        monitor.register_alert_callback(callback2)

        monitor.record_cost(85.0)

        callback1.assert_called_once()
        callback2.assert_called_once()


class TestMonitoringLifecycle:
    """Tests for monitoring start/stop lifecycle."""

    @pytest.fixture
    def config(self) -> CostMonitorConfig:
        """Create test configuration with short interval."""
        return CostMonitorConfig(
            daily_budget_usd=100.0,
            total_budget_usd=1000.0,
            check_interval_seconds=1,  # Short for testing
        )

    @pytest.fixture
    def monitor(self, config: CostMonitorConfig) -> CostMonitor:
        """Create test cost monitor."""
        return CostMonitor(config)

    def test_start_monitoring(self, monitor: CostMonitor) -> None:
        """Test starting background monitoring."""
        monitor.start_monitoring()
        assert monitor._monitoring is True
        assert monitor._monitor_thread is not None
        assert monitor._monitor_thread.is_alive()

        monitor.stop_monitoring()

    def test_stop_monitoring(self, monitor: CostMonitor) -> None:
        """Test stopping background monitoring."""
        monitor.start_monitoring()
        monitor.stop_monitoring()

        assert monitor._monitoring is False

    def test_double_start_warning(self, monitor: CostMonitor) -> None:
        """Test warning when starting already running monitor."""
        monitor.start_monitoring()
        monitor.start_monitoring()  # Should log warning, not error
        monitor.stop_monitoring()

    def test_stop_without_start(self, monitor: CostMonitor) -> None:
        """Test stopping without starting is safe."""
        monitor.stop_monitoring()  # Should not raise


class TestStatisticsAndReset:
    """Tests for statistics and reset functionality."""

    @pytest.fixture
    def config(self) -> CostMonitorConfig:
        """Create test configuration."""
        return CostMonitorConfig(
            daily_budget_usd=100.0,
            total_budget_usd=500.0,
        )

    @pytest.fixture
    def monitor(self, config: CostMonitorConfig) -> CostMonitor:
        """Create test cost monitor."""
        return CostMonitor(config)

    def test_get_statistics(self, monitor: CostMonitor) -> None:
        """Test getting statistics."""
        monitor.record_cost(25.0, gpu_type="NVIDIA_TESLA_T4", job_type="selfplay")

        stats = monitor.get_statistics()

        assert stats["daily_cost_usd"] == 25.0
        assert stats["total_cost_usd"] == 25.0
        assert stats["daily_budget_usd"] == 100.0
        assert stats["total_budget_usd"] == 500.0
        assert stats["daily_usage_pct"] == 25.0
        assert stats["total_usage_pct"] == 5.0
        assert "costs_by_gpu" in stats
        assert "costs_by_job" in stats

    def test_reset(self, monitor: CostMonitor) -> None:
        """Test resetting cost tracking."""
        monitor.record_cost(50.0, gpu_type="NVIDIA_TESLA_T4")

        monitor.reset()

        breakdown = monitor.get_current_costs()
        assert breakdown.total_today_usd == 0.0
        assert breakdown.total_project_usd == 0.0
        assert len(breakdown.by_gpu_type) == 0
        assert len(breakdown.by_job_type) == 0

    def test_repr(self, monitor: CostMonitor) -> None:
        """Test string representation."""
        monitor.record_cost(25.0)
        repr_str = repr(monitor)

        assert "CostMonitor" in repr_str
        assert "daily_budget=$100.00" in repr_str
        assert "current_daily=$25.00" in repr_str
