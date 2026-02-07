"""
Unit tests for GCS storage classes.

Uses mock backend to test without real GCS connection.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from alpha_zero.cloud.gcp.config import GCPConfig, GCSPathConfig
from alpha_zero.cloud.gcp.exceptions import GCSStorageError
from alpha_zero.cloud.gcp.storage import (
    GCSCheckpointStorage,
    GCSModelStorage,
    GCSTrajectoryStorage,
    ModelMetadata,
)

# =============================================================================
# Mock Backend
# =============================================================================


class MockStorageBackend:
    """In-memory storage backend for testing."""

    def __init__(self) -> None:
        self._storage: dict[str, bytes] = {}

    def upload_blob(self, bucket: str, blob_path: str, data: bytes) -> str:
        key = f"{bucket}/{blob_path}"
        self._storage[key] = data
        return f"gs://{key}"

    def download_blob(self, bucket: str, blob_path: str) -> bytes:
        key = f"{bucket}/{blob_path}"
        if key not in self._storage:
            raise GCSStorageError(
                f"Blob not found: {blob_path}",
                bucket=bucket,
                blob_path=blob_path,
                operation="download",
            )
        return self._storage[key]

    def blob_exists(self, bucket: str, blob_path: str) -> bool:
        key = f"{bucket}/{blob_path}"
        return key in self._storage

    def list_blobs(self, bucket: str, prefix: str) -> list[str]:
        results = []
        prefix_key = f"{bucket}/{prefix}"
        for key in self._storage:
            if key.startswith(prefix_key):
                # Return just the blob path (without bucket)
                results.append(key.split("/", 1)[1])
        return results

    def delete_blob(self, bucket: str, blob_path: str) -> None:
        key = f"{bucket}/{blob_path}"
        if key in self._storage:
            del self._storage[key]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_backend() -> MockStorageBackend:
    """Create mock storage backend."""
    return MockStorageBackend()


@pytest.fixture
def gcp_config() -> GCPConfig:
    """Create test GCP config."""
    return GCPConfig(
        project_id="test-project",
        region="us-central1",
        bucket="test-bucket",
        paths=GCSPathConfig(),
    )


@pytest.fixture
def simple_model() -> nn.Module:
    """Create simple model for testing."""
    return nn.Linear(10, 5)


# =============================================================================
# Model Metadata Tests
# =============================================================================


class TestModelMetadata:
    """Tests for ModelMetadata class."""

    def test_to_dict_excludes_none(self) -> None:
        """Test that to_dict excludes None values."""
        meta = ModelMetadata(version=1, iteration=10)
        d = meta.to_dict()
        assert d["version"] == 1
        assert d["iteration"] == 10
        assert "timestamp" in d

    def test_from_dict_round_trip(self) -> None:
        """Test round-trip serialization."""
        meta = ModelMetadata(version=2, iteration=20, elo=1500.0)
        d = meta.to_dict()
        restored = ModelMetadata.from_dict(d)
        assert restored.version == 2
        assert restored.iteration == 20
        assert restored.elo == 1500.0


# =============================================================================
# GCSModelStorage Tests
# =============================================================================


class TestGCSModelStorage:
    """Tests for model storage operations."""

    def test_save_model(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
        simple_model: nn.Module,
    ) -> None:
        """Test saving model to storage."""
        storage = GCSModelStorage(gcp_config, backend=mock_backend)

        uri = storage.save_model(
            simple_model,
            version=1,
            metadata={"iteration": 100, "elo": 1200.0},
        )

        assert "v001/model.pt" in uri
        assert mock_backend.blob_exists("test-bucket", "models/civ6az/v001/model.pt")
        assert mock_backend.blob_exists("test-bucket", "models/civ6az/v001/metadata.json")
        assert mock_backend.blob_exists("test-bucket", "models/civ6az/latest.json")

    def test_load_model(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
        simple_model: nn.Module,
    ) -> None:
        """Test loading model from storage."""
        storage = GCSModelStorage(gcp_config, backend=mock_backend)

        # Save first
        storage.save_model(simple_model, version=1, metadata={"iteration": 50})

        # Load back
        state_dict, metadata = storage.load_model(version=1)

        assert "weight" in state_dict
        assert metadata.version == 1
        assert metadata.iteration == 50

    def test_load_latest_model(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
        simple_model: nn.Module,
    ) -> None:
        """Test loading latest model version."""
        storage = GCSModelStorage(gcp_config, backend=mock_backend)

        # Save multiple versions
        storage.save_model(simple_model, version=1)
        storage.save_model(simple_model, version=2)
        storage.save_model(simple_model, version=3)

        # Load latest (should be v3)
        _state_dict, metadata = storage.load_model()
        assert metadata.version == 3

    def test_get_latest_version(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
        simple_model: nn.Module,
    ) -> None:
        """Test getting latest version number."""
        storage = GCSModelStorage(gcp_config, backend=mock_backend)

        storage.save_model(simple_model, version=5)
        assert storage.get_latest_version() == 5

    def test_list_versions(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
        simple_model: nn.Module,
    ) -> None:
        """Test listing available versions."""
        storage = GCSModelStorage(gcp_config, backend=mock_backend)

        storage.save_model(simple_model, version=1)
        storage.save_model(simple_model, version=3)
        storage.save_model(simple_model, version=5)

        versions = storage.list_versions()
        assert versions == [1, 3, 5]

    def test_load_nonexistent_raises(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
    ) -> None:
        """Test that loading nonexistent model raises error."""
        storage = GCSModelStorage(gcp_config, backend=mock_backend)

        with pytest.raises(GCSStorageError):
            storage.load_model(version=999)


# =============================================================================
# GCSTrajectoryStorage Tests
# =============================================================================


class TestGCSTrajectoryStorage:
    """Tests for trajectory storage operations."""

    def test_save_trajectories_compressed(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
    ) -> None:
        """Test saving compressed trajectories."""
        storage = GCSTrajectoryStorage(gcp_config, backend=mock_backend, compress=True)

        trajectories = [{"state": [1, 2, 3], "action": 1, "reward": 0.5} for _ in range(10)]

        uri = storage.save_trajectories(
            trajectories,
            iteration=1,
            worker_id=0,
            model_version=1,
        )

        assert ".xz" in uri
        assert mock_backend.blob_exists(
            "test-bucket", "trajectories/civ6az/iter_001/shard_worker_0.pt.xz"
        )

    def test_save_trajectories_uncompressed(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
    ) -> None:
        """Test saving uncompressed trajectories."""
        storage = GCSTrajectoryStorage(gcp_config, backend=mock_backend, compress=False)

        trajectories = [{"state": [1, 2, 3], "action": 1} for _ in range(5)]

        uri = storage.save_trajectories(
            trajectories,
            iteration=2,
            worker_id=1,
            model_version=1,
        )

        assert ".xz" not in uri
        assert mock_backend.blob_exists(
            "test-bucket", "trajectories/civ6az/iter_002/shard_worker_1.pt"
        )

    def test_load_trajectories(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
    ) -> None:
        """Test loading trajectories."""
        storage = GCSTrajectoryStorage(gcp_config, backend=mock_backend, compress=True)

        original = [{"game_id": i, "moves": [1, 2, 3]} for i in range(20)]

        storage.save_trajectories(original, iteration=1, worker_id=0, model_version=1)

        loaded = storage.load_trajectories(iteration=1)
        assert len(loaded) == 20
        assert loaded[0]["game_id"] == 0

    def test_load_trajectories_max_examples(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
    ) -> None:
        """Test loading with max_examples limit."""
        storage = GCSTrajectoryStorage(gcp_config, backend=mock_backend, compress=True)

        original = [{"id": i} for i in range(100)]
        storage.save_trajectories(original, iteration=1, worker_id=0, model_version=1)

        loaded = storage.load_trajectories(iteration=1, max_examples=10)
        assert len(loaded) == 10

    def test_manifest_updated(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
    ) -> None:
        """Test that manifest is updated on save."""
        storage = GCSTrajectoryStorage(gcp_config, backend=mock_backend)

        storage.save_trajectories([{"a": 1}] * 10, iteration=1, worker_id=0, model_version=1)
        storage.save_trajectories([{"b": 2}] * 20, iteration=1, worker_id=1, model_version=1)

        manifest_data = mock_backend.download_blob(
            "test-bucket", "trajectories/civ6az/iter_001/manifest.json"
        )
        manifest = json.loads(manifest_data.decode())

        assert manifest["total_examples"] == 30
        assert "0" in manifest["shards"]
        assert "1" in manifest["shards"]

    def test_get_available_iterations(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
    ) -> None:
        """Test listing available iterations."""
        storage = GCSTrajectoryStorage(gcp_config, backend=mock_backend)

        storage.save_trajectories([{}], iteration=1, worker_id=0, model_version=1)
        storage.save_trajectories([{}], iteration=5, worker_id=0, model_version=2)
        storage.save_trajectories([{}], iteration=10, worker_id=0, model_version=3)

        iterations = storage.get_available_iterations()
        assert iterations == [1, 5, 10]


# =============================================================================
# GCSCheckpointStorage Tests
# =============================================================================


class TestGCSCheckpointStorage:
    """Tests for checkpoint storage operations."""

    def test_save_checkpoint(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
        simple_model: nn.Module,
    ) -> None:
        """Test saving checkpoint."""
        storage = GCSCheckpointStorage(gcp_config, backend=mock_backend)
        optimizer = torch.optim.Adam(simple_model.parameters())

        uri = storage.save_checkpoint(
            simple_model,
            optimizer,
            step=1000,
            extra={"loss": 0.5},
        )

        assert "training_step_1000.pt" in uri
        assert mock_backend.blob_exists("test-bucket", "checkpoints/civ6az/training_step_1000.pt")
        assert mock_backend.blob_exists("test-bucket", "checkpoints/civ6az/latest.pt")

    def test_load_checkpoint(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
        simple_model: nn.Module,
    ) -> None:
        """Test loading checkpoint."""
        storage = GCSCheckpointStorage(gcp_config, backend=mock_backend)
        optimizer = torch.optim.Adam(simple_model.parameters())

        storage.save_checkpoint(simple_model, optimizer, step=500)

        checkpoint = storage.load_checkpoint(step=500)

        assert checkpoint is not None
        assert checkpoint["step"] == 500
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_load_latest_checkpoint(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
        simple_model: nn.Module,
    ) -> None:
        """Test loading latest checkpoint."""
        storage = GCSCheckpointStorage(gcp_config, backend=mock_backend)
        optimizer = torch.optim.Adam(simple_model.parameters())

        storage.save_checkpoint(simple_model, optimizer, step=100)
        storage.save_checkpoint(simple_model, optimizer, step=200)

        checkpoint = storage.load_checkpoint()

        assert checkpoint is not None
        assert checkpoint["step"] == 200

    def test_get_latest_step(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
        simple_model: nn.Module,
    ) -> None:
        """Test getting latest step number."""
        storage = GCSCheckpointStorage(gcp_config, backend=mock_backend)
        optimizer = torch.optim.Adam(simple_model.parameters())

        storage.save_checkpoint(simple_model, optimizer, step=1500)

        assert storage.get_latest_step() == 1500

    def test_cleanup_old_checkpoints(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
        simple_model: nn.Module,
    ) -> None:
        """Test checkpoint cleanup."""
        storage = GCSCheckpointStorage(gcp_config, backend=mock_backend)
        optimizer = torch.optim.Adam(simple_model.parameters())

        # Save 10 checkpoints
        for step in range(100, 1100, 100):
            storage.save_checkpoint(simple_model, optimizer, step=step)

        # Cleanup, keeping last 3
        deleted = storage.cleanup_old_checkpoints(keep_last=3)

        assert deleted == 7
        # Only steps 800, 900, 1000 should remain
        assert mock_backend.blob_exists("test-bucket", "checkpoints/civ6az/training_step_1000.pt")
        assert mock_backend.blob_exists("test-bucket", "checkpoints/civ6az/training_step_900.pt")
        assert mock_backend.blob_exists("test-bucket", "checkpoints/civ6az/training_step_800.pt")
        assert not mock_backend.blob_exists("test-bucket", "checkpoints/civ6az/training_step_100.pt")

    def test_load_nonexistent_returns_none(
        self,
        mock_backend: MockStorageBackend,
        gcp_config: GCPConfig,
    ) -> None:
        """Test that loading nonexistent checkpoint returns None."""
        storage = GCSCheckpointStorage(gcp_config, backend=mock_backend)

        checkpoint = storage.load_checkpoint(step=9999)
        assert checkpoint is None


# =============================================================================
# Config Tests
# =============================================================================


class TestGCPConfig:
    """Tests for GCP configuration."""

    def test_from_env_with_missing_vars(self) -> None:
        """Test that missing env vars raise ConfigurationError."""
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with patch.dict("os.environ", {}, clear=True):
            config = GCPConfig()
            with pytest.raises(ConfigurationError) as exc_info:
                config.validate()
            assert "CIV6AZ_VERTEX_PROJECT" in str(exc_info.value)

    def test_bucket_normalization(self) -> None:
        """Test that gs:// prefix is stripped from bucket."""
        config = GCPConfig(bucket="gs://my-bucket")
        assert config.bucket == "my-bucket"

    def test_with_tier_budget(self) -> None:
        """Test budget tier preset."""
        config = GCPConfig.with_tier("budget")
        assert config.self_play.accelerator_type == "NVIDIA_TESLA_T4"
        assert config.training.accelerator_type == "NVIDIA_TESLA_T4"
        assert config.self_play.spot is True

    def test_with_tier_ondemand(self) -> None:
        """Test on-demand tier preset."""
        config = GCPConfig.with_tier("ondemand")
        assert config.self_play.accelerator_type == "NVIDIA_TESLA_A100"
        assert config.training.accelerator_type == "NVIDIA_TESLA_A100"
        assert config.self_play.spot is False

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        config = GCPConfig(
            project_id="my-project",
            bucket="my-bucket",
        )
        d = config.to_dict()
        assert d["project_id"] == "my-project"
        assert d["bucket"] == "my-bucket"
        assert "self_play" in d
        assert "training" in d
