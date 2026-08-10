"""
Unit tests for PreemptionHandler component.

Tests cover:
- Configuration validation
- Checkpoint creation and timing
- Signal handling
- Resume state management
- Retry logic with exponential backoff
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError, PreemptionError
from alpha_zero.cloud.gcp.preemption_handler import (
    PreemptionHandler,
    PreemptionHandlerConfig,
    PreemptionState,
)


class SimpleModel(torch.nn.Module):
    """Simple model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestPreemptionHandlerConfig:
    """Tests for PreemptionHandlerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PreemptionHandlerConfig()

        assert config.checkpoint_interval_seconds == 300  # 5 minutes
        assert config.min_checkpoint_interval_seconds == 60
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 60
        assert config.backoff_multiplier == 2.0
        assert config.handle_sigterm is True
        assert config.checkpoint_on_signal is True

    def test_config_from_env(self) -> None:
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "CIV6AZ_CHECKPOINT_INTERVAL_S": "600",
                "CIV6AZ_CHECKPOINT_MIN_INTERVAL_S": "120",
                "CIV6AZ_PREEMPTION_MAX_RETRIES": "10",
                "CIV6AZ_PREEMPTION_RETRY_DELAY_S": "30",
                "CIV6AZ_PREEMPTION_BACKOFF_MULTIPLIER": "1.5",
                "CIV6AZ_PREEMPTION_HANDLE_SIGTERM": "false",
            },
        ):
            config = PreemptionHandlerConfig.from_env()

            assert config.checkpoint_interval_seconds == 600
            assert config.min_checkpoint_interval_seconds == 120
            assert config.max_retries == 10
            assert config.retry_delay_seconds == 30
            assert config.backoff_multiplier == 1.5
            assert config.handle_sigterm is False

    def test_config_from_yaml(self) -> None:
        """Test configuration from YAML file."""
        yaml_content = {
            "preemption_handler": {
                "checkpoint_interval_seconds": 180,
                "max_retries": 3,
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            f.flush()

            config = PreemptionHandlerConfig.from_yaml(f.name)

            assert config.checkpoint_interval_seconds == 180
            assert config.max_retries == 3

        os.unlink(f.name)

    def test_config_from_yaml_missing_file(self) -> None:
        """Test error when YAML file doesn't exist."""
        with pytest.raises(ConfigurationError, match="not found"):
            PreemptionHandlerConfig.from_yaml("/nonexistent/path.yaml")

    def test_config_from_dict(self) -> None:
        """Test configuration from dictionary."""
        data = {
            "checkpoint_interval_seconds": 120,
            "max_retries": 7,
        }
        config = PreemptionHandlerConfig.from_dict(data)

        assert config.checkpoint_interval_seconds == 120
        assert config.max_retries == 7

    def test_config_to_dict(self) -> None:
        """Test configuration serialization to dictionary."""
        config = PreemptionHandlerConfig(
            checkpoint_interval_seconds=300,
            max_retries=5,
        )
        data = config.to_dict()

        assert data["checkpoint_interval_seconds"] == 300
        assert data["max_retries"] == 5
        assert "handle_sigterm" in data

    def test_checkpoint_interval_validation(self) -> None:
        """Test validation of checkpoint interval."""
        with pytest.raises(ConfigurationError, match="at least 1"):
            PreemptionHandlerConfig(checkpoint_interval_seconds=0)

    def test_min_interval_validation(self) -> None:
        """Test validation of minimum interval."""
        with pytest.raises(ConfigurationError, match="at least 1"):
            PreemptionHandlerConfig(min_checkpoint_interval_seconds=0)

    def test_max_retries_validation(self) -> None:
        """Test validation of max retries."""
        with pytest.raises(ConfigurationError, match="non-negative"):
            PreemptionHandlerConfig(max_retries=-1)

    def test_backoff_multiplier_validation(self) -> None:
        """Test validation of backoff multiplier."""
        with pytest.raises(ConfigurationError, match=r"at least 1\.0"):
            PreemptionHandlerConfig(backoff_multiplier=0.5)


class TestPreemptionState:
    """Tests for PreemptionState dataclass."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        state = PreemptionState(
            job_name="test-job",
            job_type="training",
            iteration=100,
            checkpoint_path="/path/to/checkpoint.pt",
            preemption_time="2024-01-01T00:00:00",
            retry_count=2,
            metadata={"key": "value"},
        )
        data = state.to_dict()

        assert data["job_name"] == "test-job"
        assert data["job_type"] == "training"
        assert data["iteration"] == 100
        assert data["retry_count"] == 2
        assert data["metadata"]["key"] == "value"

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "job_name": "test-job",
            "job_type": "selfplay",
            "iteration": 50,
            "checkpoint_path": "/path/checkpoint.pt",
            "preemption_time": "2024-01-01T12:00:00",
            "retry_count": 1,
            "metadata": {},
        }
        state = PreemptionState.from_dict(data)

        assert state.job_name == "test-job"
        assert state.job_type == "selfplay"
        assert state.iteration == 50
        assert state.retry_count == 1


class TestPreemptionHandler:
    """Tests for PreemptionHandler class."""

    @pytest.fixture
    def config(self) -> PreemptionHandlerConfig:
        """Create test configuration."""
        return PreemptionHandlerConfig(
            checkpoint_interval_seconds=60,
            min_checkpoint_interval_seconds=5,
            max_retries=3,
        )

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    @pytest.fixture
    def handler(self, config: PreemptionHandlerConfig, temp_dir: Path) -> PreemptionHandler:
        """Create test handler."""
        return PreemptionHandler(config, checkpoint_dir=temp_dir)

    @pytest.fixture
    def model(self) -> SimpleModel:
        """Create test model."""
        return SimpleModel()

    def test_init_default_config(self, temp_dir: Path) -> None:
        """Test initialization with default config."""
        handler = PreemptionHandler(checkpoint_dir=temp_dir)
        assert handler.config.checkpoint_interval_seconds == 300

    def test_init_custom_config(self, config: PreemptionHandlerConfig, temp_dir: Path) -> None:
        """Test initialization with custom config."""
        handler = PreemptionHandler(config, checkpoint_dir=temp_dir)
        assert handler.config.checkpoint_interval_seconds == 60

    def test_set_job_info(self, handler: PreemptionHandler) -> None:
        """Test setting job information."""
        handler.set_job_info("test-job-123", "training")
        assert handler._job_name == "test-job-123"
        assert handler._job_type == "training"


class TestCheckpointCreation:
    """Tests for checkpoint creation."""

    @pytest.fixture
    def config(self) -> PreemptionHandlerConfig:
        """Create test configuration."""
        return PreemptionHandlerConfig(
            checkpoint_interval_seconds=60,
            min_checkpoint_interval_seconds=1,
        )

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    @pytest.fixture
    def handler(self, config: PreemptionHandlerConfig, temp_dir: Path) -> PreemptionHandler:
        """Create test handler."""
        return PreemptionHandler(config, checkpoint_dir=temp_dir)

    @pytest.fixture
    def model(self) -> SimpleModel:
        """Create test model."""
        return SimpleModel()

    def test_checkpoint_now(self, handler: PreemptionHandler, model: SimpleModel) -> None:
        """Test creating checkpoint."""
        path = handler.checkpoint_now(model, iteration=100, reason="test")

        assert Path(path).exists()
        assert handler._checkpoint_count == 1
        assert handler._last_checkpoint_time > 0

    def test_checkpoint_with_optimizer(
        self, handler: PreemptionHandler, model: SimpleModel
    ) -> None:
        """Test checkpoint includes optimizer state."""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        path = handler.checkpoint_now(model, optimizer=optimizer, iteration=100)

        checkpoint = torch.load(path, weights_only=False)
        assert "optimizer_state_dict" in checkpoint

    def test_checkpoint_with_extra_data(
        self, handler: PreemptionHandler, model: SimpleModel
    ) -> None:
        """Test checkpoint includes extra data."""
        extra = {"win_rate": 0.75, "total_games": 1000}
        path = handler.checkpoint_now(model, iteration=100, extra=extra)

        checkpoint = torch.load(path, weights_only=False)
        assert checkpoint["extra"]["win_rate"] == 0.75
        assert checkpoint["extra"]["total_games"] == 1000

    def test_checkpoint_content(self, handler: PreemptionHandler, model: SimpleModel) -> None:
        """Test checkpoint contains expected data."""
        path = handler.checkpoint_now(model, iteration=150, reason="periodic")

        checkpoint = torch.load(path, weights_only=False)
        assert "model_state_dict" in checkpoint
        assert checkpoint["iteration"] == 150
        assert checkpoint["reason"] == "periodic"
        assert "checkpoint_time" in checkpoint

    def test_should_checkpoint_timing(self, handler: PreemptionHandler) -> None:
        """Test checkpoint timing logic."""
        # Initially should checkpoint (no previous checkpoint)
        assert handler.should_checkpoint() is True

        # Update last checkpoint time
        handler._last_checkpoint_time = handler._last_checkpoint_time or 1.0

        # Should not checkpoint if interval not elapsed
        import time

        handler._last_checkpoint_time = time.time()
        assert handler.should_checkpoint() is False

    def test_update_iteration(self, handler: PreemptionHandler) -> None:
        """Test updating iteration tracking."""
        handler.update_iteration(500)
        assert handler._current_iteration == 500


class TestRetryLogic:
    """Tests for retry and backoff logic."""

    @pytest.fixture
    def config(self) -> PreemptionHandlerConfig:
        """Create test configuration."""
        return PreemptionHandlerConfig(
            max_retries=3,
            retry_delay_seconds=10,
            backoff_multiplier=2.0,
            max_backoff_seconds=120,
        )

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    @pytest.fixture
    def handler(self, config: PreemptionHandlerConfig, temp_dir: Path) -> PreemptionHandler:
        """Create test handler."""
        return PreemptionHandler(config, checkpoint_dir=temp_dir)

    def test_should_retry_under_limit(self, handler: PreemptionHandler) -> None:
        """Test retry allowed when under limit."""
        assert handler.should_retry("test-job") is True

    def test_should_retry_at_limit(self, handler: PreemptionHandler) -> None:
        """Test retry not allowed at limit."""
        handler._retry_counts["test-job"] = 3
        assert handler.should_retry("test-job") is False

    def test_increment_retry_count(self, handler: PreemptionHandler) -> None:
        """Test incrementing retry count."""
        assert handler.increment_retry_count("test-job") == 1
        assert handler.increment_retry_count("test-job") == 2
        assert handler.increment_retry_count("test-job") == 3

    def test_get_retry_delay_first_retry(self, handler: PreemptionHandler) -> None:
        """Test retry delay for first retry."""
        delay = handler.get_retry_delay("new-job")
        assert delay == 10.0  # Base delay

    def test_get_retry_delay_exponential_backoff(self, handler: PreemptionHandler) -> None:
        """Test exponential backoff for subsequent retries."""
        handler._retry_counts["test-job"] = 1
        delay = handler.get_retry_delay("test-job")
        assert delay == 20.0  # 10 * 2^1

        handler._retry_counts["test-job"] = 2
        delay = handler.get_retry_delay("test-job")
        assert delay == 40.0  # 10 * 2^2

    def test_get_retry_delay_max_cap(self, handler: PreemptionHandler) -> None:
        """Test retry delay capped at max."""
        handler._retry_counts["test-job"] = 10  # Would be 10 * 2^10 = 10240
        delay = handler.get_retry_delay("test-job")
        assert delay == 120.0  # Capped at max_backoff_seconds


class TestPreemptionStateManagement:
    """Tests for preemption state persistence."""

    @pytest.fixture
    def config(self) -> PreemptionHandlerConfig:
        """Create test configuration."""
        return PreemptionHandlerConfig()

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    @pytest.fixture
    def handler(self, config: PreemptionHandlerConfig, temp_dir: Path) -> PreemptionHandler:
        """Create test handler."""
        return PreemptionHandler(config, checkpoint_dir=temp_dir)

    def test_save_preemption_state(self, handler: PreemptionHandler) -> None:
        """Test saving preemption state."""
        handler.set_job_info("test-job", "training")
        handler._current_iteration = 200
        handler._retry_counts["test-job"] = 1

        handler.save_preemption_state(
            checkpoint_path="/path/to/checkpoint.pt",
            metadata={"extra": "data"},
        )

        # Verify file was created
        state_path = handler.checkpoint_dir / "preemption_state_test-job.yaml"
        assert state_path.exists()

    def test_get_resume_state(self, handler: PreemptionHandler) -> None:
        """Test retrieving preemption state."""
        handler.set_job_info("test-job", "training")
        handler._current_iteration = 200

        handler.save_preemption_state(
            checkpoint_path="/path/to/checkpoint.pt",
        )

        state = handler.get_resume_state("test-job")

        assert state is not None
        assert state.job_name == "test-job"
        assert state.job_type == "training"
        assert state.iteration == 200

    def test_get_resume_state_not_found(self, handler: PreemptionHandler) -> None:
        """Test getting non-existent resume state."""
        state = handler.get_resume_state("nonexistent-job")
        assert state is None

    def test_clear_resume_state(self, handler: PreemptionHandler) -> None:
        """Test clearing preemption state."""
        handler.set_job_info("test-job", "training")
        handler.save_preemption_state(checkpoint_path="/path/checkpoint.pt")
        handler._retry_counts["test-job"] = 2

        handler.clear_resume_state("test-job")

        state_path = handler.checkpoint_dir / "preemption_state_test-job.yaml"
        assert not state_path.exists()
        assert "test-job" not in handler._retry_counts


class TestSignalHandling:
    """Tests for signal handling."""

    @pytest.fixture
    def config(self) -> PreemptionHandlerConfig:
        """Create test configuration."""
        return PreemptionHandlerConfig(
            handle_sigterm=True,
            checkpoint_on_signal=True,
        )

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    @pytest.fixture
    def handler(self, config: PreemptionHandlerConfig, temp_dir: Path) -> PreemptionHandler:
        """Create test handler."""
        return PreemptionHandler(config, checkpoint_dir=temp_dir)

    def test_install_signal_handlers(self, handler: PreemptionHandler) -> None:
        """Test installing signal handlers doesn't raise."""
        # May log warnings on Windows/non-main thread but shouldn't raise
        handler.install_signal_handlers()
        handler.uninstall_signal_handlers()

    def test_signal_handling_disabled(self, temp_dir: Path) -> None:
        """Test signal handling can be disabled."""
        config = PreemptionHandlerConfig(handle_sigterm=False)
        handler = PreemptionHandler(config, checkpoint_dir=temp_dir)

        # Should not raise and should log message
        handler.install_signal_handlers()

    def test_preemption_detected_flag(self, handler: PreemptionHandler) -> None:
        """Test preemption detection flag."""
        assert handler.preemption_detected is False

        handler._preemption_detected = True
        assert handler.preemption_detected is True


class TestTimerManagement:
    """Tests for checkpoint timer management."""

    @pytest.fixture
    def config(self) -> PreemptionHandlerConfig:
        """Create test configuration with short interval."""
        return PreemptionHandlerConfig(
            checkpoint_interval_seconds=1,
            min_checkpoint_interval_seconds=1,
        )

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    @pytest.fixture
    def handler(self, config: PreemptionHandlerConfig, temp_dir: Path) -> PreemptionHandler:
        """Create test handler."""
        return PreemptionHandler(config, checkpoint_dir=temp_dir)

    @pytest.fixture
    def model(self) -> SimpleModel:
        """Create test model."""
        return SimpleModel()

    def test_start_checkpoint_timer(self, handler: PreemptionHandler, model: SimpleModel) -> None:
        """Test starting checkpoint timer."""
        handler.start_checkpoint_timer(model)

        assert handler._timer_running is True
        assert handler._timer_thread is not None
        assert handler._timer_thread.is_alive()

        handler.stop_checkpoint_timer()

    def test_stop_checkpoint_timer(self, handler: PreemptionHandler, model: SimpleModel) -> None:
        """Test stopping checkpoint timer."""
        handler.start_checkpoint_timer(model)
        handler.stop_checkpoint_timer()

        assert handler._timer_running is False

    def test_double_start_warning(self, handler: PreemptionHandler, model: SimpleModel) -> None:
        """Test warning on double start."""
        handler.start_checkpoint_timer(model)
        handler.start_checkpoint_timer(model)  # Should log warning

        handler.stop_checkpoint_timer()

    def test_stop_without_start(self, handler: PreemptionHandler) -> None:
        """Test stopping without starting is safe."""
        handler.stop_checkpoint_timer()  # Should not raise


class TestHandlePreemption:
    """Tests for preemption handling."""

    @pytest.fixture
    def config(self) -> PreemptionHandlerConfig:
        """Create test configuration."""
        return PreemptionHandlerConfig(
            min_checkpoint_interval_seconds=1,
        )

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    @pytest.fixture
    def handler(self, config: PreemptionHandlerConfig, temp_dir: Path) -> PreemptionHandler:
        """Create test handler."""
        return PreemptionHandler(config, checkpoint_dir=temp_dir)

    @pytest.fixture
    def model(self) -> SimpleModel:
        """Create test model."""
        return SimpleModel()

    def test_handle_preemption_with_model(
        self, handler: PreemptionHandler, model: SimpleModel
    ) -> None:
        """Test handling preemption creates checkpoint."""
        handler.set_job_info("test-job", "training")
        handler._model = model
        handler._current_iteration = 100

        error = PreemptionError("Spot VM preempted", job_id="test-job")
        state = handler.handle_preemption(error)

        assert state is not None
        assert state.job_name == "test-job"
        assert state.iteration == 100

    def test_handle_preemption_no_model(self, handler: PreemptionHandler) -> None:
        """Test handling preemption without model."""
        handler.set_job_info("test-job", "training")

        state = handler.handle_preemption()
        assert state is None


class TestStatisticsAndCleanup:
    """Tests for statistics and cleanup."""

    @pytest.fixture
    def config(self) -> PreemptionHandlerConfig:
        """Create test configuration."""
        return PreemptionHandlerConfig()

    @pytest.fixture
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create temporary checkpoint directory."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        return checkpoint_dir

    @pytest.fixture
    def handler(self, config: PreemptionHandlerConfig, temp_dir: Path) -> PreemptionHandler:
        """Create test handler."""
        return PreemptionHandler(config, checkpoint_dir=temp_dir)

    def test_get_statistics(self, handler: PreemptionHandler) -> None:
        """Test getting statistics."""
        handler.set_job_info("test-job", "training")
        handler._checkpoint_count = 5
        handler._current_iteration = 100

        stats = handler.get_statistics()

        assert stats["checkpoint_count"] == 5
        assert stats["job_name"] == "test-job"
        assert stats["current_iteration"] == 100
        assert "config" in stats

    def test_cleanup(self, handler: PreemptionHandler) -> None:
        """Test cleanup releases resources."""
        handler.install_signal_handlers()
        handler.cleanup()

        # Should not raise on double cleanup
        handler.cleanup()

    def test_repr(self, handler: PreemptionHandler) -> None:
        """Test string representation."""
        repr_str = repr(handler)

        assert "PreemptionHandler" in repr_str
        assert "interval=" in repr_str
        assert "max_retries=" in repr_str
