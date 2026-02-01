"""Unit tests for cloud configuration.

This module tests the cloud configuration system, including:
- Environment variable loading
- Serialization/deserialization
- Validation
- Path building
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from alpha_zero.cloud.config import (
    CloudConfig,
    CloudEnvironment,
    CloudProvider,
)


class TestCloudEnvironment:
    """Tests for CloudEnvironment enum."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("dev", CloudEnvironment.DEVELOPMENT),
            ("development", CloudEnvironment.DEVELOPMENT),
            ("DEVELOPMENT", CloudEnvironment.DEVELOPMENT),
            ("staging", CloudEnvironment.STAGING),
            ("STAGING", CloudEnvironment.STAGING),
            ("prod", CloudEnvironment.PRODUCTION),
            ("production", CloudEnvironment.PRODUCTION),
            ("test", CloudEnvironment.TEST),
        ],
    )
    def test_from_string(self, value: str, expected: CloudEnvironment) -> None:
        """Test environment parsing from string."""
        assert CloudEnvironment.from_string(value) == expected

    def test_from_string_invalid(self) -> None:
        """Test invalid environment string raises error."""
        with pytest.raises(ValueError, match="Unknown environment"):
            CloudEnvironment.from_string("invalid")


class TestCloudConfig:
    """Tests for CloudConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CloudConfig()
        assert config.provider == CloudProvider.LOCAL
        assert config.environment == CloudEnvironment.DEVELOPMENT
        assert config.project_id == ""
        assert config.region == ""

    def test_from_env_basic(self) -> None:
        """Test loading from environment variables."""
        env = {
            "CLOUD_PROVIDER": "gcp",
            "CLOUD_ENVIRONMENT": "prod",
            "CLOUD_PROJECT_ID": "my-project",
            "CLOUD_REGION": "us-central1",
            "CLOUD_DEFAULT_BUCKET": "my-bucket",
        }
        with patch.dict(os.environ, env, clear=False):
            config = CloudConfig.from_env()

        assert config.provider == CloudProvider.GCP
        assert config.environment == CloudEnvironment.PRODUCTION
        assert config.project_id == "my-project"
        assert config.region == "us-central1"
        assert config.default_bucket == "my-bucket"

    def test_from_env_with_labels(self) -> None:
        """Test loading labels from environment."""
        env = {
            "CLOUD_LABELS": "team=ml,env=dev,project=alphazero",
        }
        with patch.dict(os.environ, env, clear=False):
            config = CloudConfig.from_env()

        assert config.labels == {
            "team": "ml",
            "env": "dev",
            "project": "alphazero",
        }

    def test_from_env_with_tags(self) -> None:
        """Test loading tags from environment."""
        env = {
            "CLOUD_TAGS": "ml,training,gpu",
        }
        with patch.dict(os.environ, env, clear=False):
            config = CloudConfig.from_env()

        assert config.tags == ["ml", "training", "gpu"]

    def test_from_env_invalid_provider(self) -> None:
        """Test fallback for invalid provider."""
        env = {"CLOUD_PROVIDER": "invalid"}
        with patch.dict(os.environ, env, clear=False):
            config = CloudConfig.from_env()

        assert config.provider == CloudProvider.LOCAL

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        config = CloudConfig(
            provider=CloudProvider.GCP,
            environment=CloudEnvironment.STAGING,
            project_id="test-project",
            region="europe-west1",
        )
        data = config.to_dict()

        assert data["provider"] == "gcp"
        assert data["environment"] == "staging"
        assert data["project_id"] == "test-project"
        assert data["region"] == "europe-west1"

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "provider": "gcp",
            "environment": "prod",
            "project_id": "test-project",
            "region": "us-east1",
        }
        config = CloudConfig.from_dict(data)

        assert config.provider == CloudProvider.GCP
        assert config.environment == CloudEnvironment.PRODUCTION
        assert config.project_id == "test-project"
        assert config.region == "us-east1"

    def test_round_trip_serialization(self) -> None:
        """Test serialization round-trip."""
        original = CloudConfig(
            provider=CloudProvider.GCP,
            environment=CloudEnvironment.PRODUCTION,
            project_id="my-project",
            region="us-central1",
            labels={"team": "ml"},
            tags=["gpu", "training"],
        )

        data = original.to_dict()
        restored = CloudConfig.from_dict(data)

        assert restored.provider == original.provider
        assert restored.environment == original.environment
        assert restored.project_id == original.project_id
        assert restored.region == original.region
        assert restored.labels == original.labels
        assert restored.tags == original.tags

    def test_validate_cloud_provider_requires_project_id(self) -> None:
        """Test validation requires project_id for cloud providers."""
        config = CloudConfig(
            provider=CloudProvider.GCP,
            project_id="",
            region="",
        )
        errors = config.validate()

        assert "project_id is required" in errors[0]
        assert "region is required" in errors[1]
        assert not config.is_valid()

    def test_validate_local_provider_no_requirements(self) -> None:
        """Test validation passes for local provider without project_id."""
        config = CloudConfig(
            provider=CloudProvider.LOCAL,
            project_id="",
            region="",
        )
        errors = config.validate()

        assert len(errors) == 0
        assert config.is_valid()

    def test_get_storage_path_gcp(self) -> None:
        """Test storage path building for GCP."""
        config = CloudConfig(
            provider=CloudProvider.GCP,
            default_bucket="my-bucket",
        )
        path = config.get_storage_path("models", "v1", "weights.pt")

        assert path == "gs://my-bucket/models/v1/weights.pt"

    def test_get_checkpoint_path(self) -> None:
        """Test checkpoint path building."""
        config = CloudConfig(
            provider=CloudProvider.GCP,
            default_bucket="my-bucket",
            checkpoint_prefix="checkpoints",
        )
        path = config.get_checkpoint_path("epoch_10.pt")

        assert path == "gs://my-bucket/checkpoints/epoch_10.pt"

    def test_get_artifact_path(self) -> None:
        """Test artifact path building."""
        config = CloudConfig(
            provider=CloudProvider.GCP,
            default_bucket="my-bucket",
            artifact_prefix="artifacts",
        )
        path = config.get_artifact_path("model_final.pt")

        assert path == "gs://my-bucket/artifacts/model_final.pt"
