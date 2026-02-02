"""Unit tests for cloud configuration module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

from alpha_zero.cloud.config import CloudConfig
from alpha_zero.cloud.exceptions import CloudConfigError


class TestCloudConfigDefaults:
    """Tests for CloudConfig default values."""

    def test_default_initialization(self) -> None:
        """Test default values are set correctly."""
        config = CloudConfig()
        
        assert config.project_id == ""
        assert config.region == "us-central1"
        assert config.bucket_name == ""
        assert config.checkpoint_prefix == "checkpoints/"
        assert config.enable_cloud_logging is True
        assert config.enable_cloud_monitoring is True
        assert config.use_spot is True
        assert config.accelerator_count == 1

    def test_custom_initialization(self) -> None:
        """Test initialization with custom values."""
        config = CloudConfig(
            project_id="my-project",
            region="europe-west1",
            bucket_name="my-bucket",
            use_spot=False,
            accelerator_count=4,
        )
        
        assert config.project_id == "my-project"
        assert config.region == "europe-west1"
        assert config.bucket_name == "my-bucket"
        assert config.use_spot is False
        assert config.accelerator_count == 4

    def test_prefix_normalization(self) -> None:
        """Test that prefixes are normalized with trailing slashes."""
        config = CloudConfig(
            checkpoint_prefix="checkpoints",  # Missing trailing slash
            artifact_prefix="artifacts",  # Missing trailing slash
        )
        
        assert config.checkpoint_prefix == "checkpoints/"
        assert config.artifact_prefix == "artifacts/"


class TestCloudConfigFromEnv:
    """Tests for CloudConfig.from_env()."""

    def test_from_env_basic(self) -> None:
        """Test loading configuration from environment variables."""
        env_vars = {
            "GCP_PROJECT_ID": "test-project",
            "GCP_REGION": "us-west1",
            "GCS_BUCKET_NAME": "test-bucket",
        }
        
        with mock.patch.dict(os.environ, env_vars, clear=False):
            config = CloudConfig.from_env()
            
        assert config.project_id == "test-project"
        assert config.region == "us-west1"
        assert config.bucket_name == "test-bucket"

    def test_from_env_with_overrides(self) -> None:
        """Test loading with CIV6AZ_ prefixed overrides."""
        env_vars = {
            "GCP_PROJECT_ID": "test-project",
            "CIV6AZ_ENABLE_CLOUD_LOGGING": "false",
            "CIV6AZ_USE_SPOT": "no",
            "CIV6AZ_ACCELERATOR_COUNT": "8",
            "CIV6AZ_PREEMPTION_CHECK_INTERVAL": "10.5",
        }
        
        with mock.patch.dict(os.environ, env_vars, clear=False):
            config = CloudConfig.from_env()
            
        assert config.project_id == "test-project"
        assert config.enable_cloud_logging is False
        assert config.use_spot is False
        assert config.accelerator_count == 8
        assert config.preemption_check_interval == 10.5

    def test_from_env_missing_vars(self) -> None:
        """Test that missing env vars result in defaults."""
        with mock.patch.dict(os.environ, {}, clear=True):
            config = CloudConfig.from_env()
            
        assert config.project_id == ""
        assert config.region == "us-central1"


class TestCloudConfigFromYaml:
    """Tests for CloudConfig.from_yaml()."""

    def test_from_yaml_basic(self) -> None:
        """Test loading configuration from YAML file."""
        yaml_content = """
        project_id: yaml-project
        region: asia-east1
        bucket_name: yaml-bucket
        use_spot: false
        accelerator_count: 2
        """
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            yaml_path.write_text(yaml_content)
            
            config = CloudConfig.from_yaml(str(yaml_path))
            
            assert config.project_id == "yaml-project"
            assert config.region == "asia-east1"
            assert config.bucket_name == "yaml-bucket"
            assert config.use_spot is False
            assert config.accelerator_count == 2

    def test_from_yaml_nested_structure(self) -> None:
        """Test loading from YAML with nested cloud/gcp structure."""
        yaml_content = """
        cloud:
          gcp:
            project_id: nested-project
            region: us-central1
        """
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "nested.yaml"
            yaml_path.write_text(yaml_content)
            
            config = CloudConfig.from_yaml(str(yaml_path))
            assert config.project_id == "nested-project"

    def test_from_yaml_file_not_found(self) -> None:
        """Test error when YAML file doesn't exist."""
        with pytest.raises(CloudConfigError, match="not found"):
            CloudConfig.from_yaml("/nonexistent/path.yaml")

    def test_from_yaml_invalid_yaml(self) -> None:
        """Test error when YAML is malformed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "invalid.yaml"
            yaml_path.write_text("invalid: yaml: [broken")
            
            with pytest.raises(CloudConfigError, match="Failed to parse"):
                CloudConfig.from_yaml(str(yaml_path))


class TestCloudConfigValidation:
    """Tests for CloudConfig.validate()."""

    def test_validate_success(self) -> None:
        """Test successful validation."""
        config = CloudConfig(
            project_id="valid-project",
            accelerator_count=4,
            replica_count=2,
            use_spot=True,
            max_run_duration_seconds=3600,
        )
        
        # Should not raise
        config.validate()

    def test_validate_missing_project(self) -> None:
        """Test validation fails when project_id is missing."""
        config = CloudConfig()
        
        with pytest.raises(CloudConfigError, match="project_id is required"):
            config.validate(require_project=True)

    def test_validate_optional_project(self) -> None:
        """Test validation passes when project_id is optional."""
        config = CloudConfig()
        
        # Should not raise
        config.validate(require_project=False)

    def test_validate_invalid_run_duration(self) -> None:
        """Test validation fails with too short run duration for spot."""
        config = CloudConfig(
            project_id="test",
            use_spot=True,
            max_run_duration_seconds=30,  # Too short
        )
        
        with pytest.raises(CloudConfigError, match="at least 60"):
            config.validate()

    def test_validate_negative_accelerator_count(self) -> None:
        """Test validation fails with negative accelerator count."""
        config = CloudConfig(
            project_id="test",
            accelerator_count=-1,
        )
        
        with pytest.raises(CloudConfigError, match="non-negative"):
            config.validate()

    def test_validate_zero_replica_count(self) -> None:
        """Test validation fails with zero replica count."""
        config = CloudConfig(
            project_id="test",
            replica_count=0,
        )
        
        with pytest.raises(CloudConfigError, match="at least 1"):
            config.validate()


class TestCloudConfigHelpers:
    """Tests for CloudConfig helper methods."""

    def test_get_checkpoint_uri(self) -> None:
        """Test checkpoint URI generation."""
        config = CloudConfig(
            bucket_name="my-bucket",
            checkpoint_prefix="checkpoints/",
        )
        
        uri = config.get_checkpoint_uri("model_v1.pt")
        assert uri == "gs://my-bucket/checkpoints/model_v1.pt"

    def test_get_checkpoint_uri_no_bucket(self) -> None:
        """Test checkpoint URI fails without bucket."""
        config = CloudConfig()
        
        with pytest.raises(CloudConfigError, match="bucket_name is required"):
            config.get_checkpoint_uri("model.pt")

    def test_get_artifact_uri(self) -> None:
        """Test artifact URI generation."""
        config = CloudConfig(
            bucket_name="my-bucket",
            artifact_prefix="artifacts/",
        )
        
        uri = config.get_artifact_uri("logs.json")
        assert uri == "gs://my-bucket/artifacts/logs.json"

    def test_to_dict(self) -> None:
        """Test configuration serialization to dict."""
        config = CloudConfig(
            project_id="test-project",
            region="us-west1",
            bucket_name="test-bucket",
        )
        
        data = config.to_dict()
        
        assert data["project_id"] == "test-project"
        assert data["region"] == "us-west1"
        assert data["bucket_name"] == "test-bucket"
        assert "checkpoint_prefix" in data
        assert "custom_labels" in data
