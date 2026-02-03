"""
Unit tests for IdleInstanceManager component.

Tests cover:
- Configuration validation
- Idle detection logic
- Grace period handling
- Job registration and tracking
- Auto-stop functionality
"""

from __future__ import annotations

import os
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest
import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError
from alpha_zero.cloud.gcp.idle_manager import (
    IdleInstanceConfig,
    IdleInstanceManager,
    IdlePolicy,
    IdleStatus,
)


class TestIdleInstanceConfig:
    """Tests for IdleInstanceConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = IdleInstanceConfig()

        assert config.idle_threshold_minutes == 15
        assert config.gpu_utilization_threshold == 0.1
        assert config.check_interval_seconds == 60
        assert config.auto_stop_enabled is True
        assert config.grace_period_seconds == 300

    def test_config_from_env(self) -> None:
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "CIV6AZ_IDLE_THRESHOLD_MINUTES": "30",
                "CIV6AZ_IDLE_GPU_UTILIZATION_THRESHOLD": "0.05",
                "CIV6AZ_IDLE_CHECK_INTERVAL_S": "120",
                "CIV6AZ_IDLE_AUTO_STOP_ENABLED": "false",
                "CIV6AZ_IDLE_GRACE_PERIOD_S": "600",
            },
        ):
            config = IdleInstanceConfig.from_env()

            assert config.idle_threshold_minutes == 30
            assert config.gpu_utilization_threshold == 0.05
            assert config.check_interval_seconds == 120
            assert config.auto_stop_enabled is False
            assert config.grace_period_seconds == 600

    def test_config_from_yaml(self) -> None:
        """Test configuration from YAML file."""
        yaml_content = {
            "idle_manager": {
                "idle_threshold_minutes": 20,
                "grace_period_seconds": 180,
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            f.flush()

            config = IdleInstanceConfig.from_yaml(f.name)

            assert config.idle_threshold_minutes == 20
            assert config.grace_period_seconds == 180

        os.unlink(f.name)

    def test_config_from_yaml_missing_file(self) -> None:
        """Test error when YAML file doesn't exist."""
        with pytest.raises(ConfigurationError, match="not found"):
            IdleInstanceConfig.from_yaml("/nonexistent/path.yaml")

    def test_config_from_dict(self) -> None:
        """Test configuration from dictionary."""
        data = {
            "idle_threshold_minutes": 25,
            "auto_stop_enabled": False,
        }
        config = IdleInstanceConfig.from_dict(data)

        assert config.idle_threshold_minutes == 25
        assert config.auto_stop_enabled is False

    def test_config_to_dict(self) -> None:
        """Test configuration serialization to dictionary."""
        config = IdleInstanceConfig(
            idle_threshold_minutes=15,
            grace_period_seconds=300,
        )
        data = config.to_dict()

        assert data["idle_threshold_minutes"] == 15
        assert data["grace_period_seconds"] == 300
        assert "policies" in data

    def test_idle_threshold_validation(self) -> None:
        """Test validation of idle threshold."""
        with pytest.raises(ConfigurationError, match="at least 1"):
            IdleInstanceConfig(idle_threshold_minutes=0)

    def test_gpu_threshold_validation(self) -> None:
        """Test validation of GPU utilization threshold."""
        with pytest.raises(ConfigurationError, match="between 0 and 1"):
            IdleInstanceConfig(gpu_utilization_threshold=1.5)

    def test_check_interval_validation(self) -> None:
        """Test validation of check interval."""
        with pytest.raises(ConfigurationError, match="at least 1"):
            IdleInstanceConfig(check_interval_seconds=0)

    def test_grace_period_validation(self) -> None:
        """Test validation of grace period."""
        with pytest.raises(ConfigurationError, match="non-negative"):
            IdleInstanceConfig(grace_period_seconds=-1)

    def test_default_policies_created(self) -> None:
        """Test default policies are created."""
        config = IdleInstanceConfig()

        assert "selfplay" in config.policies
        assert "training" in config.policies
        assert "evaluation" in config.policies

    def test_get_policy_existing(self) -> None:
        """Test getting existing policy."""
        config = IdleInstanceConfig()
        policy = config.get_policy("training")

        assert policy.job_type == "training"
        assert policy.checkpoint_before_stop is True

    def test_get_policy_unknown(self) -> None:
        """Test getting policy for unknown job type."""
        config = IdleInstanceConfig()
        policy = config.get_policy("unknown_type")

        assert policy.job_type == "unknown_type"
        assert policy.idle_threshold_minutes == config.idle_threshold_minutes


class TestIdlePolicy:
    """Tests for IdlePolicy dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        policy = IdlePolicy(
            job_type="training",
            auto_stop=True,
            idle_threshold_minutes=10,
            checkpoint_before_stop=True,
        )
        data = policy.to_dict()

        assert data["job_type"] == "training"
        assert data["auto_stop"] is True
        assert data["idle_threshold_minutes"] == 10

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "job_type": "selfplay",
            "auto_stop": False,
            "idle_threshold_minutes": 20,
            "checkpoint_before_stop": False,
            "notify_before_stop": True,
            "min_instances": 2,
        }
        policy = IdlePolicy.from_dict(data)

        assert policy.job_type == "selfplay"
        assert policy.auto_stop is False
        assert policy.min_instances == 2


class TestIdleStatus:
    """Tests for IdleStatus dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        status = IdleStatus(
            job_name="test-job",
            job_type="training",
            is_idle=True,
            idle_duration_seconds=600.0,
            last_activity_time="2024-01-01T00:00:00",
            gpu_utilization=0.05,
            should_stop=False,
            reason="Idle for 10.0 minutes",
        )
        data = status.to_dict()

        assert data["job_name"] == "test-job"
        assert data["is_idle"] is True
        assert data["idle_duration_seconds"] == 600.0
        assert data["gpu_utilization"] == 0.05


class TestIdleInstanceManager:
    """Tests for IdleInstanceManager class."""

    @pytest.fixture
    def config(self) -> IdleInstanceConfig:
        """Create test configuration."""
        return IdleInstanceConfig(
            idle_threshold_minutes=1,  # 1 minute for faster testing
            grace_period_seconds=5,  # 5 seconds
            check_interval_seconds=1,
        )

    @pytest.fixture
    def manager(self, config: IdleInstanceConfig) -> IdleInstanceManager:
        """Create test manager."""
        return IdleInstanceManager(config)

    def test_init_default_config(self) -> None:
        """Test initialization with default config."""
        manager = IdleInstanceManager()
        assert manager.config.idle_threshold_minutes == 15

    def test_init_custom_config(self, config: IdleInstanceConfig) -> None:
        """Test initialization with custom config."""
        manager = IdleInstanceManager(config)
        assert manager.config.idle_threshold_minutes == 1


class TestJobRegistration:
    """Tests for job registration and tracking."""

    @pytest.fixture
    def config(self) -> IdleInstanceConfig:
        """Create test configuration."""
        return IdleInstanceConfig(idle_threshold_minutes=1)

    @pytest.fixture
    def manager(self, config: IdleInstanceConfig) -> IdleInstanceManager:
        """Create test manager."""
        return IdleInstanceManager(config)

    def test_register_job(self, manager: IdleInstanceManager) -> None:
        """Test registering a job."""
        manager.register_job("job-123", job_type="training")

        assert "job-123" in manager._jobs
        assert manager._jobs["job-123"]["job_type"] == "training"

    def test_register_job_with_metadata(self, manager: IdleInstanceManager) -> None:
        """Test registering job with metadata."""
        metadata = {"iteration": 100, "worker_id": 5}
        manager.register_job("job-123", job_type="selfplay", metadata=metadata)

        assert manager._jobs["job-123"]["metadata"]["iteration"] == 100

    def test_unregister_job(self, manager: IdleInstanceManager) -> None:
        """Test unregistering a job."""
        manager.register_job("job-123", job_type="training")
        manager.unregister_job("job-123")

        assert "job-123" not in manager._jobs

    def test_unregister_nonexistent_job(self, manager: IdleInstanceManager) -> None:
        """Test unregistering a non-existent job doesn't raise."""
        manager.unregister_job("nonexistent")  # Should not raise

    def test_mark_active_existing_job(self, manager: IdleInstanceManager) -> None:
        """Test marking existing job as active."""
        manager.register_job("job-123", job_type="training")
        old_activity = manager._jobs["job-123"]["last_activity"]

        time.sleep(0.01)  # Small delay
        manager.mark_active("job-123")

        assert manager._jobs["job-123"]["last_activity"] > old_activity

    def test_mark_active_with_gpu_util(self, manager: IdleInstanceManager) -> None:
        """Test marking active with GPU utilization."""
        manager.register_job("job-123", job_type="training")
        manager.mark_active("job-123", gpu_utilization=0.85)

        assert manager._jobs["job-123"]["gpu_utilization"] == 0.85

    def test_mark_active_new_job_with_type(self, manager: IdleInstanceManager) -> None:
        """Test marking active registers new job if type provided."""
        manager.mark_active("new-job", job_type="selfplay")

        assert "new-job" in manager._jobs
        assert manager._jobs["new-job"]["job_type"] == "selfplay"

    def test_mark_active_unknown_job_no_type(
        self, manager: IdleInstanceManager
    ) -> None:
        """Test marking unknown job without type logs warning."""
        manager.mark_active("unknown-job")  # Should log warning
        assert "unknown-job" not in manager._jobs


class TestIdleDetection:
    """Tests for idle detection logic."""

    @pytest.fixture
    def config(self) -> IdleInstanceConfig:
        """Create test configuration with short threshold."""
        return IdleInstanceConfig(
            idle_threshold_minutes=1,  # 60 seconds
            gpu_utilization_threshold=0.1,
            grace_period_seconds=30,
            auto_stop_enabled=True,
        )

    @pytest.fixture
    def manager(self, config: IdleInstanceConfig) -> IdleInstanceManager:
        """Create test manager."""
        return IdleInstanceManager(config)

    def test_check_active_job_not_idle(self, manager: IdleInstanceManager) -> None:
        """Test recently active job is not idle."""
        manager.register_job("job-123", job_type="training")
        status = manager.check_job_idle("job-123")

        assert status.is_idle is False
        assert status.should_stop is False
        assert "Active" in status.reason

    def test_check_idle_by_time(self, manager: IdleInstanceManager) -> None:
        """Test job becomes idle after threshold."""
        manager.register_job("job-123", job_type="training")

        # Manually set old last_activity time
        manager._jobs["job-123"]["last_activity"] = time.time() - 120  # 2 min ago

        status = manager.check_job_idle("job-123")

        assert status.is_idle is True
        assert "Idle for" in status.reason

    def test_check_idle_by_gpu_utilization(
        self, manager: IdleInstanceManager
    ) -> None:
        """Test job becomes idle with low GPU utilization."""
        manager.register_job("job-123", job_type="training")
        manager.mark_active("job-123", gpu_utilization=0.05)  # Below threshold

        status = manager.check_job_idle("job-123")

        assert status.is_idle is True
        assert "GPU util" in status.reason

    def test_check_unknown_job(self, manager: IdleInstanceManager) -> None:
        """Test checking unknown job returns not found status."""
        status = manager.check_job_idle("unknown")

        assert status.is_idle is False
        assert "not found" in status.reason.lower()

    def test_get_all_idle_jobs(self, manager: IdleInstanceManager) -> None:
        """Test getting all idle jobs."""
        manager.register_job("job-1", job_type="training")
        manager.register_job("job-2", job_type="selfplay")
        manager.register_job("job-3", job_type="evaluation")

        # Make job-1 and job-3 idle
        manager._jobs["job-1"]["last_activity"] = time.time() - 120
        manager._jobs["job-3"]["last_activity"] = time.time() - 120

        idle_jobs = manager.get_all_idle_jobs()

        assert len(idle_jobs) == 2
        job_names = [s.job_name for s in idle_jobs]
        assert "job-1" in job_names
        assert "job-3" in job_names


class TestGracePeriod:
    """Tests for grace period handling."""

    @pytest.fixture
    def config(self) -> IdleInstanceConfig:
        """Create test configuration with short grace period."""
        return IdleInstanceConfig(
            idle_threshold_minutes=1,  # 60 seconds
            grace_period_seconds=2,  # Short for testing
            auto_stop_enabled=True,
        )

    @pytest.fixture
    def manager(self, config: IdleInstanceConfig) -> IdleInstanceManager:
        """Create test manager."""
        return IdleInstanceManager(config)

    def test_grace_period_starts_on_idle(self, manager: IdleInstanceManager) -> None:
        """Test grace period starts when job becomes idle."""
        manager.register_job("job-123", job_type="training")
        manager._jobs["job-123"]["last_activity"] = time.time() - 120

        status = manager.check_job_idle("job-123")

        assert status.is_idle is True
        assert status.should_stop is False
        assert "grace period started" in status.reason.lower()
        assert "job-123" in manager._grace_period_starts

    def test_grace_period_resets_on_activity(
        self, manager: IdleInstanceManager
    ) -> None:
        """Test activity clears grace period."""
        manager.register_job("job-123", job_type="training")
        manager._jobs["job-123"]["last_activity"] = time.time() - 120

        # Start grace period
        manager.check_job_idle("job-123")
        assert "job-123" in manager._grace_period_starts

        # Mark active
        manager.mark_active("job-123")
        assert "job-123" not in manager._grace_period_starts

    def test_should_stop_after_grace_period(
        self, manager: IdleInstanceManager
    ) -> None:
        """Test should_stop is True after grace period expires."""
        manager.register_job("job-123", job_type="training")
        manager._jobs["job-123"]["last_activity"] = time.time() - 120

        # First check starts grace period
        manager.check_job_idle("job-123")

        # Set grace period start to be expired
        manager._grace_period_starts["job-123"] = time.time() - 10

        # Check again
        status = manager.check_job_idle("job-123")

        assert status.is_idle is True
        assert status.should_stop is True
        assert "grace period expired" in status.reason.lower()


class TestAutoStop:
    """Tests for auto-stop functionality."""

    @pytest.fixture
    def config(self) -> IdleInstanceConfig:
        """Create test configuration."""
        return IdleInstanceConfig(
            idle_threshold_minutes=1,
            grace_period_seconds=0,  # Immediate stop
            auto_stop_enabled=True,
        )

    @pytest.fixture
    def manager(self, config: IdleInstanceConfig) -> IdleInstanceManager:
        """Create test manager."""
        return IdleInstanceManager(config)

    def test_stop_idle_jobs_dry_run(self, manager: IdleInstanceManager) -> None:
        """Test dry run doesn't actually stop jobs."""
        manager.register_job("job-123", job_type="training")
        manager._jobs["job-123"]["last_activity"] = time.time() - 120
        manager._grace_period_starts["job-123"] = time.time() - 10

        stopped = manager.stop_idle_jobs(dry_run=True)

        assert "job-123" in stopped
        assert "job-123" in manager._jobs  # Still exists

    def test_stop_idle_jobs_actual(self, manager: IdleInstanceManager) -> None:
        """Test actual stop removes jobs."""
        manager.register_job("job-123", job_type="selfplay")
        manager._jobs["job-123"]["last_activity"] = time.time() - 120
        manager._grace_period_starts["job-123"] = time.time() - 10

        stopped = manager.stop_idle_jobs()

        assert "job-123" in stopped
        assert "job-123" not in manager._jobs
        assert "job-123" in manager._stopped_jobs

    def test_auto_stop_disabled(self) -> None:
        """Test auto-stop respects disabled config."""
        config = IdleInstanceConfig(
            idle_threshold_minutes=1,
            auto_stop_enabled=False,
        )
        manager = IdleInstanceManager(config)

        manager.register_job("job-123", job_type="training")
        manager._jobs["job-123"]["last_activity"] = time.time() - 120

        status = manager.check_job_idle("job-123")

        # Idle but should_stop is False because auto_stop disabled
        assert status.is_idle is True
        assert status.should_stop is False


class TestCallbacks:
    """Tests for callback functionality."""

    @pytest.fixture
    def config(self) -> IdleInstanceConfig:
        """Create test configuration."""
        return IdleInstanceConfig(
            idle_threshold_minutes=1,
            grace_period_seconds=0,
            auto_stop_enabled=True,
        )

    @pytest.fixture
    def manager(self, config: IdleInstanceConfig) -> IdleInstanceManager:
        """Create test manager."""
        return IdleInstanceManager(config)

    def test_stop_callback_called(self, manager: IdleInstanceManager) -> None:
        """Test stop callback is called when job is stopped."""
        callback = MagicMock()
        manager.register_stop_callback(callback)

        manager.register_job("job-123", job_type="selfplay")
        manager._jobs["job-123"]["last_activity"] = time.time() - 120
        manager._grace_period_starts["job-123"] = time.time() - 10

        stopped = manager.stop_idle_jobs()

        assert "job-123" in stopped
        callback.assert_called_once()
        args = callback.call_args[0]
        assert args[0] == "job-123"
        assert isinstance(args[1], IdleStatus)

    def test_warning_callback_not_called_for_active(
        self, manager: IdleInstanceManager
    ) -> None:
        """Test warning callback not called for active jobs."""
        callback = MagicMock()
        manager.register_warning_callback(callback)

        manager.register_job("job-123", job_type="training")
        # Job is active, no callback should be triggered during monitoring

        callback.assert_not_called()


class TestMonitoringLifecycle:
    """Tests for monitoring start/stop lifecycle."""

    @pytest.fixture
    def config(self) -> IdleInstanceConfig:
        """Create test configuration with short interval."""
        return IdleInstanceConfig(
            idle_threshold_minutes=1,
            check_interval_seconds=1,
            auto_stop_enabled=False,  # Disable auto-stop for lifecycle tests
        )

    @pytest.fixture
    def manager(self, config: IdleInstanceConfig) -> IdleInstanceManager:
        """Create test manager."""
        return IdleInstanceManager(config)

    def test_start_monitoring(self, manager: IdleInstanceManager) -> None:
        """Test starting background monitoring."""
        manager.start_monitoring()

        assert manager._monitoring is True
        assert manager._monitor_thread is not None
        assert manager._monitor_thread.is_alive()

        manager.stop_monitoring()

    def test_stop_monitoring(self, manager: IdleInstanceManager) -> None:
        """Test stopping background monitoring."""
        manager.start_monitoring()
        manager.stop_monitoring()

        assert manager._monitoring is False

    def test_double_start_warning(self, manager: IdleInstanceManager) -> None:
        """Test warning when starting already running monitor."""
        manager.start_monitoring()
        manager.start_monitoring()  # Should log warning

        manager.stop_monitoring()

    def test_stop_without_start(self, manager: IdleInstanceManager) -> None:
        """Test stopping without starting is safe."""
        manager.stop_monitoring()  # Should not raise


class TestStatisticsAndReset:
    """Tests for statistics and reset functionality."""

    @pytest.fixture
    def config(self) -> IdleInstanceConfig:
        """Create test configuration."""
        return IdleInstanceConfig()

    @pytest.fixture
    def manager(self, config: IdleInstanceConfig) -> IdleInstanceManager:
        """Create test manager."""
        return IdleInstanceManager(config)

    def test_get_statistics(self, manager: IdleInstanceManager) -> None:
        """Test getting statistics."""
        manager.register_job("job-1", job_type="training")
        manager.register_job("job-2", job_type="selfplay")

        stats = manager.get_statistics()

        assert stats["registered_jobs"] == 2
        assert "auto_stop_enabled" in stats
        assert "config" in stats

    def test_reset(self, manager: IdleInstanceManager) -> None:
        """Test resetting manager state."""
        manager.register_job("job-123", job_type="training")
        manager._grace_period_starts["job-123"] = time.time()

        manager.reset()

        assert len(manager._jobs) == 0
        assert len(manager._grace_period_starts) == 0

    def test_repr(self, manager: IdleInstanceManager) -> None:
        """Test string representation."""
        manager.register_job("job-123", job_type="training")
        repr_str = repr(manager)

        assert "IdleInstanceManager" in repr_str
        assert "threshold=" in repr_str
        assert "jobs=1" in repr_str
