"""Unit tests for cloud components.

Tests for:
- CloudConfig and MachineProfile
- PreemptionHandler
- SelfPlayConfig
- TrainerConfig
- EvaluatorConfig
- JobConfig
"""

from __future__ import annotations

import os
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.config import (
    MACHINE_PROFILES,
    CloudConfig,
    MachineProfile,
    get_machine_profile,
)
from alpha_zero.cloud.preemption import (
    PreemptionHandler,
    register_preemption_handler,
)


class TestMachineProfile:
    """Tests for MachineProfile dataclass."""

    def test_default_profile(self) -> None:
        """Test creating a default machine profile."""
        profile = MachineProfile(
            name="test",
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            spot=True,
        )
        assert profile.name == "test"
        assert profile.machine_type == "n1-standard-8"
        assert profile.accelerator_type == "NVIDIA_TESLA_T4"
        assert profile.accelerator_count == 1
        assert profile.spot is True

    def test_to_vertex_spec(self) -> None:
        """Test converting profile to Vertex AI spec."""
        profile = MachineProfile(
            name="test",
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            spot=True,
        )
        spec = profile.to_vertex_spec()

        assert spec["machine_spec"]["machine_type"] == "n1-standard-8"
        assert spec["machine_spec"]["accelerator_type"] == "NVIDIA_TESLA_T4"
        assert spec["machine_spec"]["accelerator_count"] == 1
        assert spec["replica_count"] == 1

    def test_to_vertex_spec_no_gpu(self) -> None:
        """Test Vertex spec without GPU."""
        profile = MachineProfile(
            name="cpu-only",
            machine_type="n1-standard-4",
            accelerator_type=None,
            accelerator_count=0,
            spot=False,
        )
        spec = profile.to_vertex_spec()

        assert spec["machine_spec"]["machine_type"] == "n1-standard-4"
        assert "accelerator_type" not in spec["machine_spec"]
        assert "accelerator_count" not in spec["machine_spec"]


class TestMachineProfiles:
    """Tests for predefined machine profiles."""

    def test_profiles_exist(self) -> None:
        """Test that all expected profiles exist."""
        expected_profiles = [
            "selfplay_t4_spot",
            "selfplay_l4_spot",
            "train_l4_single",
            "train_l4_ondemand",
            "train_a100_burst",
            "eval_l4_spot",
        ]
        for name in expected_profiles:
            assert name in MACHINE_PROFILES

    def test_get_machine_profile(self) -> None:
        """Test getting a profile by name."""
        profile = get_machine_profile("selfplay_t4_spot")
        assert profile.name == "selfplay_t4_spot"
        assert profile.spot is True

    def test_get_nonexistent_profile(self) -> None:
        """Test getting a non-existent profile raises KeyError."""
        with pytest.raises(KeyError):
            get_machine_profile("nonexistent")

    def test_spot_profiles_have_spot_rates(self) -> None:
        """Test that spot profiles have spot rates."""
        for name, profile in MACHINE_PROFILES.items():
            if profile.spot:
                assert profile.hourly_rate_spot > 0, f"{name} missing spot rate"


class TestCloudConfig:
    """Tests for CloudConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        # Clear env vars that might affect config
        with patch.dict(os.environ, {}, clear=True):
            config = CloudConfig()
            assert config.project_id == ""
            assert config.region == ""  # Empty by default
            assert config.default_bucket == ""

    def test_env_override(self) -> None:
        """Test creating config with environment overrides."""
        with patch.dict(os.environ, {
            "CLOUD_PROJECT_ID": "test-project",
            "CLOUD_REGION": "us-west1",
            "CLOUD_DEFAULT_BUCKET": "test-bucket",
        }, clear=True):
            config = CloudConfig.from_env()
            assert config.project_id == "test-project"
            assert config.region == "us-west1"
            assert config.default_bucket == "test-bucket"


class TestPreemptionHandler:
    """Tests for PreemptionHandler."""

    def test_default_handler(self) -> None:
        """Test default preemption handler."""
        handler = PreemptionHandler()
        assert handler.grace_period == 25.0
        assert handler.is_preempted is False
        assert handler.preemption_time is None

    def test_custom_grace_period(self) -> None:
        """Test custom grace period."""
        handler = PreemptionHandler(grace_period=10.0)
        assert handler.grace_period == 10.0

    def test_add_cleanup_callback(self) -> None:
        """Test adding cleanup callbacks."""
        handler = PreemptionHandler()
        callback = MagicMock()
        handler.add_cleanup_callback(callback)
        assert callback in handler.cleanup_callbacks

    def test_register_and_unregister(self) -> None:
        """Test signal handler registration."""
        handler = PreemptionHandler()

        # Register
        handler.register()
        # Unregister should restore original handlers
        handler.unregister()

        # No assertions - just verifying no exceptions

    def test_register_preemption_handler_function(self) -> None:
        """Test the module-level register function."""
        callbacks: list[Callable[[], None]] = [MagicMock(), MagicMock()]
        handler = register_preemption_handler(
            cleanup_callbacks=callbacks,
            grace_period=15.0,
        )

        assert handler.grace_period == 15.0
        assert len(handler.cleanup_callbacks) == 2

        # Clean up
        handler.unregister()


class TestSelfPlayConfig:
    """Tests for SelfPlayConfig."""

    def test_from_env(self) -> None:
        """Test creating config from environment."""
        from alpha_zero.cloud.selfplay_worker import SelfPlayConfig

        with patch.dict(os.environ, {
            "CIV6_NUM_GAMES": "50",
            "CIV6_ITERATION": "10",
            "CIV6_WORKER_ID": "3",
            "CIV6_NUM_SIMULATIONS": "400",
            "CIV6_DEVICE": "cpu",
            "CIV6_BOARD_SIZE": "16",
        }):
            config = SelfPlayConfig.from_env()
            assert config.num_games == 50
            assert config.iteration == 10
            assert config.worker_id == 3
            assert config.num_simulations == 400
            assert config.device == "cpu"
            assert config.board_size == 16

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from alpha_zero.cloud.selfplay_worker import SelfPlayConfig

        config = SelfPlayConfig()
        assert config.num_games == 100
        assert config.iteration == 0
        assert config.worker_id == 0
        assert config.num_simulations == 800


class TestTrainerConfig:
    """Tests for TrainerConfig."""

    def test_from_env(self) -> None:
        """Test creating config from environment."""
        from alpha_zero.cloud.trainer import TrainerConfig

        with patch.dict(os.environ, {
            "CIV6_TRAINING_STEPS": "500",
            "CIV6_ITERATION": "5",
            "CIV6_BATCH_SIZE": "128",
            "CIV6_LEARNING_RATE": "0.0001",
            "CIV6_WEIGHT_DECAY": "1e-5",
            "CIV6_USE_AMP": "false",
        }):
            config = TrainerConfig.from_env()
            assert config.training_steps == 500
            assert config.iteration == 5
            assert config.batch_size == 128
            assert config.learning_rate == 0.0001
            assert config.weight_decay == 1e-5
            assert config.use_amp is False

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from alpha_zero.cloud.trainer import TrainerConfig

        config = TrainerConfig()
        assert config.training_steps == 1000
        assert config.batch_size == 256
        assert config.use_amp is True


class TestEvaluatorConfig:
    """Tests for EvaluatorConfig."""

    def test_from_env(self) -> None:
        """Test creating config from environment."""
        from alpha_zero.cloud.evaluator import EvaluatorConfig

        with patch.dict(os.environ, {
            "CIV6_EVAL_GAMES": "200",
            "CIV6_ITERATION": "15",
            "CIV6_PROMOTION_THRESHOLD": "0.6",
            "CIV6_NUM_SIMULATIONS": "150",
        }):
            config = EvaluatorConfig.from_env()
            assert config.num_games == 200
            assert config.iteration == 15
            assert config.promotion_threshold == 0.6
            assert config.num_simulations == 150

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from alpha_zero.cloud.evaluator import EvaluatorConfig

        config = EvaluatorConfig()
        assert config.num_games == 400
        assert config.promotion_threshold == 0.55
        assert config.early_stop_threshold == 200


class TestJobConfig:
    """Tests for JobConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        from infra.vertex_launcher import JobConfig

        config = JobConfig(
            display_name="test-job",
            container_uri="gcr.io/test/image:latest",
        )
        assert config.display_name == "test-job"
        assert config.container_uri == "gcr.io/test/image:latest"
        assert config.machine_type == "n1-standard-8"
        assert config.accelerator_count == 0
        assert config.spot is False

    def test_to_worker_pool_spec(self) -> None:
        """Test converting to worker pool spec."""
        from infra.vertex_launcher import JobConfig

        config = JobConfig(
            display_name="gpu-job",
            container_uri="gcr.io/test/image:latest",
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            spot=True,
            args=["--arg1", "value1"],
            env_vars={"KEY": "VALUE"},
        )
        spec = config.to_worker_pool_spec()

        assert spec["machine_spec"]["machine_type"] == "n1-standard-8"
        assert spec["machine_spec"]["accelerator_type"] == "NVIDIA_TESLA_T4"
        assert spec["machine_spec"]["accelerator_count"] == 1
        assert spec["container_spec"]["image_uri"] == "gcr.io/test/image:latest"
        assert spec["container_spec"]["args"] == ["--arg1", "value1"]

    def test_to_worker_pool_spec_no_gpu(self) -> None:
        """Test worker pool spec without GPU."""
        from infra.vertex_launcher import JobConfig

        config = JobConfig(
            display_name="cpu-job",
            container_uri="gcr.io/test/image:latest",
        )
        spec = config.to_worker_pool_spec()

        assert spec["machine_spec"]["machine_type"] == "n1-standard-8"
        assert "accelerator_type" not in spec["machine_spec"]
        assert "accelerator_count" not in spec["machine_spec"]
