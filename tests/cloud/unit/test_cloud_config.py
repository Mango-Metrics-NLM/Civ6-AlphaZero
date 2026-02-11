"""Unit tests for cloud configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_zero.cloud.config import (
    CloudConfig,
    CloudEnvironment,
    CloudLoggingConfig,
    CloudMonitoringConfig,
    CloudProvider,
    GCSConfig,
    PreemptionConfig,
    TrainingConfig,
    VertexConfig,
)


class TestGCSConfig:
    """Tests for GCSConfig."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = GCSConfig()
        assert config.bucket_name == ""
        assert config.checkpoint_prefix == "checkpoints"
        assert config.enable_compression is True
        assert config.max_checkpoints == 10

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "bucket_name": "my-bucket",
            "checkpoint_prefix": "ckpts",
            "enable_compression": False,
            "max_checkpoints": 5,
        }
        config = GCSConfig.from_dict(data)
        assert config.bucket_name == "my-bucket"
        assert config.checkpoint_prefix == "ckpts"
        assert config.enable_compression is False
        assert config.max_checkpoints == 5

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creation from environment variables."""
        monkeypatch.setenv("CIV6AZ_GCS_BUCKET", "env-bucket")
        monkeypatch.setenv("CIV6AZ_GCS_CHECKPOINT_PREFIX", "env-ckpts")
        monkeypatch.setenv("CIV6AZ_GCS_ENABLE_COMPRESSION", "false")

        config = GCSConfig.from_env()
        assert config.bucket_name == "env-bucket"
        assert config.checkpoint_prefix == "env-ckpts"
        assert config.enable_compression is False


class TestVertexConfig:
    """Tests for VertexConfig."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = VertexConfig()
        assert config.machine_type == "n1-standard-8"
        assert config.accelerator_type == "NVIDIA_TESLA_T4"
        assert config.accelerator_count == 1
        assert config.boot_disk_size_gb == 100

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "machine_type": "n1-standard-16",
            "accelerator_type": "NVIDIA_TESLA_V100",
            "accelerator_count": 2,
        }
        config = VertexConfig.from_dict(data)
        assert config.machine_type == "n1-standard-16"
        assert config.accelerator_type == "NVIDIA_TESLA_V100"
        assert config.accelerator_count == 2

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creation from environment variables."""
        monkeypatch.setenv("CIV6AZ_VERTEX_MACHINE_TYPE", "n1-standard-32")
        monkeypatch.setenv("CIV6AZ_VERTEX_ACCELERATOR_COUNT", "4")

        config = VertexConfig.from_env()
        assert config.machine_type == "n1-standard-32"
        assert config.accelerator_count == 4


class TestPreemptionConfig:
    """Tests for PreemptionConfig."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = PreemptionConfig()
        assert config.enabled is True
        assert config.check_interval_seconds == 5.0
        assert config.checkpoint_on_preemption is True
        assert config.max_retry_count == 3

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "enabled": False,
            "check_interval_seconds": 10.0,
            "max_retry_count": 5,
        }
        config = PreemptionConfig.from_dict(data)
        assert config.enabled is False
        assert config.check_interval_seconds == 10.0
        assert config.max_retry_count == 5


class TestCloudLoggingConfig:
    """Tests for CloudLoggingConfig."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = CloudLoggingConfig()
        assert config.enabled is True
        assert config.log_name == "civ6-alphazero-training"
        assert config.structured is True

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "enabled": False,
            "log_name": "custom-log",
            "buffer_size": 50,
        }
        config = CloudLoggingConfig.from_dict(data)
        assert config.enabled is False
        assert config.log_name == "custom-log"
        assert config.buffer_size == 50


class TestCloudMonitoringConfig:
    """Tests for CloudMonitoringConfig."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = CloudMonitoringConfig()
        assert config.enabled is True
        assert "loss" in config.export_metrics
        assert "learning_rate" in config.export_metrics

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "enabled": False,
            "export_metrics": ["custom_metric"],
        }
        config = CloudMonitoringConfig.from_dict(data)
        assert config.enabled is False
        assert config.export_metrics == ["custom_metric"]


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = TrainingConfig()
        assert config.batch_size == 256
        assert config.learning_rate == 0.001
        assert config.games_per_iteration == 100

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "batch_size": 512,
            "learning_rate": 0.0001,
            "num_iterations": 500,
        }
        config = TrainingConfig.from_dict(data)
        assert config.batch_size == 512
        assert config.learning_rate == 0.0001
        assert config.num_iterations == 500


class TestCloudConfig:
    """Tests for main CloudConfig."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = CloudConfig()
        assert config.provider == CloudProvider.GCP.value
        assert config.environment == CloudEnvironment.DEVELOPMENT.value
        assert config.region == "us-central1"
        assert isinstance(config.gcs, GCSConfig)
        assert isinstance(config.vertex, VertexConfig)

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "cloud": {
                "provider": "gcp",
                "project_id": "my-project",
                "region": "europe-west1",
                "gcs": {
                    "bucket_name": "my-bucket",
                },
                "vertex": {
                    "machine_type": "n1-standard-16",
                },
            },
            "training": {
                "batch_size": 512,
            },
        }
        config = CloudConfig.from_dict(data)
        assert config.project_id == "my-project"
        assert config.region == "europe-west1"
        assert config.gcs.bucket_name == "my-bucket"
        assert config.vertex.machine_type == "n1-standard-16"
        assert config.training.batch_size == 512

    def test_from_yaml(self, tmp_path: Path) -> None:
        """Test loading from YAML file."""
        yaml_content = """
cloud:
  provider: gcp
  project_id: test-project
  region: us-west1
  gcs:
    bucket_name: test-bucket
training:
  batch_size: 128
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = CloudConfig.from_yaml(config_file)
        assert config.project_id == "test-project"
        assert config.region == "us-west1"
        assert config.gcs.bucket_name == "test-bucket"
        assert config.training.batch_size == 128

    def test_from_yaml_file_not_found(self) -> None:
        """Test error when YAML file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            CloudConfig.from_yaml("nonexistent.yaml")

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test creation from environment variables."""
        monkeypatch.setenv("CIV6AZ_GCP_PROJECT_ID", "env-project")
        monkeypatch.setenv("CIV6AZ_GCP_REGION", "asia-east1")
        monkeypatch.setenv("CIV6AZ_GCS_BUCKET", "env-bucket")

        config = CloudConfig.from_env()
        assert config.project_id == "env-project"
        assert config.region == "asia-east1"
        assert config.gcs.bucket_name == "env-bucket"

    def test_apply_env_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that env vars override YAML values."""
        yaml_content = """
cloud:
  project_id: yaml-project
  region: us-central1
  gcs:
    bucket_name: yaml-bucket
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        monkeypatch.setenv("CIV6AZ_GCP_PROJECT_ID", "env-project")

        config = CloudConfig.from_yaml(config_file)
        assert config.project_id == "env-project"  # Overridden
        assert config.gcs.bucket_name == "yaml-bucket"  # Not overridden

    def test_validate_valid_config(self) -> None:
        """Test validation of valid config."""
        config = CloudConfig(
            project_id="my-project",
            gcs=GCSConfig(bucket_name="my-bucket"),
            training=TrainingConfig(batch_size=256, learning_rate=0.001),
        )
        issues = config.validate(raise_on_error=False)
        assert len(issues) == 0
        assert config.is_valid()

    def test_validate_missing_project(self) -> None:
        """Test validation catches missing project_id."""
        config = CloudConfig(
            gcs=GCSConfig(bucket_name="my-bucket"),
        )
        issues = config.validate(raise_on_error=False)
        assert any("project_id" in issue for issue in issues)
        assert not config.is_valid()

    def test_validate_missing_bucket(self) -> None:
        """Test validation catches missing bucket."""
        config = CloudConfig(project_id="my-project")
        issues = config.validate(raise_on_error=False)
        assert any("bucket_name" in issue for issue in issues)

    def test_validate_invalid_batch_size(self) -> None:
        """Test validation catches invalid batch size."""
        config = CloudConfig(
            project_id="my-project",
            gcs=GCSConfig(bucket_name="my-bucket"),
            training=TrainingConfig(batch_size=0),
        )
        issues = config.validate(raise_on_error=False)
        assert any("batch_size" in issue for issue in issues)

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        config = CloudConfig(
            project_id="my-project",
            region="us-central1",
            gcs=GCSConfig(bucket_name="my-bucket"),
        )
        data = config.to_dict()
        assert data["cloud"]["project_id"] == "my-project"
        assert data["cloud"]["region"] == "us-central1"
        assert data["cloud"]["gcs"]["bucket_name"] == "my-bucket"

    def test_to_yaml(self, tmp_path: Path) -> None:
        """Test saving to YAML file."""
        config = CloudConfig(
            project_id="my-project",
            gcs=GCSConfig(bucket_name="my-bucket"),
        )
        config_file = tmp_path / "output.yaml"
        config.to_yaml(config_file)

        assert config_file.exists()

        # Reload and verify
        loaded = CloudConfig.from_yaml(config_file)
        assert loaded.project_id == "my-project"
        assert loaded.gcs.bucket_name == "my-bucket"
