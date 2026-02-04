"""Unit tests for lifecycle module.

Tests GCS lifecycle management and retention policies.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.lifecycle import (
    CleanupResult,
    GCSLifecycleManager,
    LifecycleConfig,
    LifecycleRule,
    RetentionPolicyConfig,
    StorageClass,
    get_lifecycle_manager,
)


class TestStorageClass:
    """Tests for StorageClass enum."""

    def test_storage_class_values(self) -> None:
        """Test StorageClass enum values."""
        assert StorageClass.STANDARD.value == "STANDARD"
        assert StorageClass.NEARLINE.value == "NEARLINE"
        assert StorageClass.COLDLINE.value == "COLDLINE"
        assert StorageClass.ARCHIVE.value == "ARCHIVE"


class TestRetentionPolicyConfig:
    """Tests for RetentionPolicyConfig dataclass."""

    def test_default_values(self) -> None:
        """Test RetentionPolicyConfig default values."""
        config = RetentionPolicyConfig(prefix="test/")
        assert config.prefix == "test/"
        assert config.keep_recent_iterations == 20
        assert config.archive_storage_class == StorageClass.NEARLINE

    def test_to_dict(self) -> None:
        """Test RetentionPolicyConfig conversion to dictionary."""
        config = RetentionPolicyConfig(
            prefix="trajectories/",
            keep_recent_iterations=10,
        )
        data = config.to_dict()
        assert data["prefix"] == "trajectories/"
        assert data["keep_recent_iterations"] == 10

    def test_from_dict(self) -> None:
        """Test RetentionPolicyConfig creation from dictionary."""
        data = {
            "prefix": "logs/",
            "keep_recent_iterations": 50,
            "archive_storage_class": "COLDLINE",
        }
        config = RetentionPolicyConfig.from_dict(data)
        assert config.prefix == "logs/"
        assert config.archive_storage_class == StorageClass.COLDLINE


class TestLifecycleRule:
    """Tests for LifecycleRule dataclass."""

    def test_set_storage_class_rule(self) -> None:
        """Test SetStorageClass lifecycle rule."""
        rule = LifecycleRule(
            action_type="SetStorageClass",
            storage_class=StorageClass.NEARLINE,
            age_days=30,
            matches_prefix=["trajectories/"],
        )
        gcs_rule = rule.to_gcs_rule()
        assert gcs_rule["action"]["type"] == "SetStorageClass"
        assert gcs_rule["action"]["storageClass"] == "NEARLINE"

    def test_delete_rule(self) -> None:
        """Test Delete lifecycle rule."""
        rule = LifecycleRule(
            action_type="Delete",
            age_days=365,
            matches_prefix=["logs/"],
        )
        gcs_rule = rule.to_gcs_rule()
        assert gcs_rule["action"]["type"] == "Delete"


class TestCleanupResult:
    """Tests for CleanupResult dataclass."""

    def test_default_values(self) -> None:
        """Test CleanupResult default values."""
        result = CleanupResult()
        assert result.archived_count == 0
        assert result.deleted_count == 0

    def test_merge(self) -> None:
        """Test merging two CleanupResults."""
        result1 = CleanupResult(archived_count=5, bytes_archived=1000)
        result2 = CleanupResult(archived_count=3, bytes_archived=500)
        merged = result1.merge(result2)
        assert merged.archived_count == 8
        assert merged.bytes_archived == 1500


class TestLifecycleConfig:
    """Tests for LifecycleConfig dataclass."""

    def test_default_values(self) -> None:
        """Test LifecycleConfig default values."""
        config = LifecycleConfig()
        assert config.enabled is True
        assert config.dry_run is False

    def test_get_all_policies(self) -> None:
        """Test get_all_policies returns all retention policies."""
        config = LifecycleConfig()
        policies = config.get_all_policies()
        assert len(policies) == 5


class TestGCSLifecycleManager:
    """Tests for GCSLifecycleManager class."""

    def test_init(self) -> None:
        """Test GCSLifecycleManager initialization."""
        config = LifecycleConfig(enabled=True, dry_run=True)
        manager = GCSLifecycleManager("test-bucket", config)
        assert manager._bucket == "test-bucket"

    def test_bucket_prefix_removal(self) -> None:
        """Test that gs:// prefix is removed from bucket name."""
        manager = GCSLifecycleManager("gs://test-bucket")
        assert manager._bucket == "test-bucket"

    def test_should_run_cleanup(self) -> None:
        """Test should_run_cleanup."""
        config = LifecycleConfig(enabled=True, iteration_cleanup_interval=10)
        manager = GCSLifecycleManager("test-bucket", config)
        assert manager.should_run_cleanup(10) is True
        assert manager.should_run_cleanup(15) is False

    def test_generate_lifecycle_rules(self) -> None:
        """Test lifecycle rule generation."""
        config = LifecycleConfig()
        manager = GCSLifecycleManager("test-bucket", config)
        rules = manager.generate_lifecycle_rules()
        assert len(rules) > 0
