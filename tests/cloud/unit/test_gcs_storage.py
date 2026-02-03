"""Unit tests for GCS storage module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from alpha_zero.cloud.gcs_storage import (
    CheckpointMetadata,
    GCSCheckpointManager,
    GCSModelStorage,
)


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        meta = CheckpointMetadata(
            key="checkpoints/checkpoint_00000100.pt",
            size=1024,
            created="2024-01-01T00:00:00",
            training_step=1000,
            model_version=5,
            iteration=100,
            metrics={"loss": 0.5},
            custom={"note": "test"},
        )
        data = meta.to_dict()
        assert data["key"] == "checkpoints/checkpoint_00000100.pt"
        assert data["size"] == 1024
        assert data["metrics"]["loss"] == 0.5

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "key": "checkpoints/checkpoint_00000050.pt",
            "size": 2048,
            "iteration": 50,
            "training_step": 500,
        }
        meta = CheckpointMetadata.from_dict(data)
        assert meta.key == "checkpoints/checkpoint_00000050.pt"
        assert meta.size == 2048
        assert meta.iteration == 50


class TestGCSCheckpointManager:
    """Tests for GCSCheckpointManager."""

    @pytest.fixture
    def manager(self, tmp_path: Path) -> GCSCheckpointManager:
        """Create checkpoint manager with mock client."""
        return GCSCheckpointManager(
            bucket_name="test-bucket",
            prefix="checkpoints",
            local_cache_dir=str(tmp_path / "cache"),
            enable_compression=False,
            force_mock=True,
        )

    @pytest.fixture
    def checkpoint_file(self, tmp_path: Path) -> Path:
        """Create a test checkpoint file."""
        checkpoint = {
            "model_state_dict": {"layer.weight": torch.randn(10, 10)},
            "optimizer_state_dict": {},
            "step": 100,
        }
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def test_upload_checkpoint(
        self, manager: GCSCheckpointManager, checkpoint_file: Path
    ) -> None:
        """Test uploading a checkpoint."""
        uri = manager.upload_checkpoint(
            local_path=checkpoint_file,
            iteration=100,
            training_step=1000,
            model_version=5,
            metrics={"loss": 0.5},
        )

        assert uri == "gs://test-bucket/checkpoints/checkpoint_00000100.pt"
        assert manager.exists(100)

    def test_upload_checkpoint_file_not_found(
        self, manager: GCSCheckpointManager
    ) -> None:
        """Test error when checkpoint file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            manager.upload_checkpoint(
                local_path="nonexistent.pt",
                iteration=100,
            )

    def test_download_checkpoint(
        self,
        manager: GCSCheckpointManager,
        checkpoint_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test downloading a checkpoint."""
        # Upload first
        manager.upload_checkpoint(checkpoint_file, iteration=100)

        # Download
        download_path = tmp_path / "downloaded.pt"
        result = manager.download_checkpoint(100, local_path=download_path)

        assert result == download_path
        assert download_path.exists()

        # Verify content
        loaded = torch.load(download_path, weights_only=False)
        assert "model_state_dict" in loaded

    def test_download_checkpoint_not_found(
        self, manager: GCSCheckpointManager
    ) -> None:
        """Test error when checkpoint doesn't exist."""
        with pytest.raises(FileNotFoundError):
            manager.download_checkpoint(999)

    def test_download_latest(
        self,
        manager: GCSCheckpointManager,
        checkpoint_file: Path,
    ) -> None:
        """Test downloading latest checkpoint."""
        # Upload multiple checkpoints
        manager.upload_checkpoint(checkpoint_file, iteration=50)
        manager.upload_checkpoint(checkpoint_file, iteration=100)
        manager.upload_checkpoint(checkpoint_file, iteration=75)

        # Download latest
        result = manager.download_latest()

        assert result is not None
        assert "checkpoint_00000100" in str(result)

    def test_download_latest_empty(self, manager: GCSCheckpointManager) -> None:
        """Test downloading latest when no checkpoints exist."""
        result = manager.download_latest()
        assert result is None

    def test_list_checkpoints(
        self,
        manager: GCSCheckpointManager,
        checkpoint_file: Path,
    ) -> None:
        """Test listing checkpoints."""
        # Upload multiple checkpoints
        manager.upload_checkpoint(checkpoint_file, iteration=50, training_step=500)
        manager.upload_checkpoint(checkpoint_file, iteration=100, training_step=1000)

        checkpoints = manager.list_checkpoints()

        assert len(checkpoints) == 2
        assert checkpoints[0].iteration == 50
        assert checkpoints[1].iteration == 100

    def test_exists(
        self,
        manager: GCSCheckpointManager,
        checkpoint_file: Path,
    ) -> None:
        """Test checking if checkpoint exists."""
        assert not manager.exists(100)

        manager.upload_checkpoint(checkpoint_file, iteration=100)

        assert manager.exists(100)
        assert not manager.exists(50)

    def test_delete_checkpoint(
        self,
        manager: GCSCheckpointManager,
        checkpoint_file: Path,
    ) -> None:
        """Test deleting a checkpoint."""
        manager.upload_checkpoint(checkpoint_file, iteration=100)
        assert manager.exists(100)

        result = manager.delete_checkpoint(100)

        assert result is True
        assert not manager.exists(100)

    def test_delete_checkpoint_not_found(
        self, manager: GCSCheckpointManager
    ) -> None:
        """Test deleting non-existent checkpoint."""
        result = manager.delete_checkpoint(999)
        assert result is False

    def test_get_latest_iteration(
        self,
        manager: GCSCheckpointManager,
        checkpoint_file: Path,
    ) -> None:
        """Test getting latest iteration number."""
        assert manager.get_latest_iteration() is None

        manager.upload_checkpoint(checkpoint_file, iteration=50)
        manager.upload_checkpoint(checkpoint_file, iteration=100)

        assert manager.get_latest_iteration() == 100

    def test_cleanup_old_checkpoints(
        self,
        tmp_path: Path,
        checkpoint_file: Path,
    ) -> None:
        """Test automatic cleanup of old checkpoints."""
        manager = GCSCheckpointManager(
            bucket_name="test-bucket",
            prefix="checkpoints",
            local_cache_dir=str(tmp_path / "cache"),
            max_checkpoints=3,
            force_mock=True,
        )

        # Upload 5 checkpoints
        for i in range(5):
            manager.upload_checkpoint(checkpoint_file, iteration=i * 10)

        # Only 3 should remain
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3

        # Should be the latest 3
        iterations = [c.iteration for c in checkpoints]
        assert 40 in iterations
        assert 30 in iterations
        assert 20 in iterations

    def test_compression(
        self, tmp_path: Path, checkpoint_file: Path
    ) -> None:
        """Test checkpoint compression."""
        manager = GCSCheckpointManager(
            bucket_name="test-bucket",
            prefix="checkpoints",
            local_cache_dir=str(tmp_path / "cache"),
            enable_compression=True,
            force_mock=True,
        )

        # Upload with compression
        manager.upload_checkpoint(checkpoint_file, iteration=100)

        # Download and verify
        download_path = tmp_path / "downloaded.pt"
        manager.download_checkpoint(100, local_path=download_path)

        loaded = torch.load(download_path, weights_only=False)
        assert "model_state_dict" in loaded


class TestGCSModelStorage:
    """Tests for GCSModelStorage."""

    @pytest.fixture
    def storage(self) -> GCSModelStorage:
        """Create model storage with mock client."""
        return GCSModelStorage(
            bucket_name="test-bucket",
            prefix="models",
            force_mock=True,
        )

    @pytest.fixture
    def simple_model(self) -> torch.nn.Module:
        """Create a simple test model."""
        return torch.nn.Linear(10, 5)

    def test_save_model(
        self,
        storage: GCSModelStorage,
        simple_model: torch.nn.Module,
    ) -> None:
        """Test saving a model."""
        uri = storage.save_model(
            model=simple_model,
            version=1,
            metrics={"accuracy": 0.95},
        )

        assert uri == "gs://test-bucket/models/model_v000001.pt"

    def test_load_model(
        self,
        storage: GCSModelStorage,
        simple_model: torch.nn.Module,
    ) -> None:
        """Test loading a model."""
        # Save first
        storage.save_model(simple_model, version=1, metrics={"accuracy": 0.95})

        # Create new model and load weights
        new_model = torch.nn.Linear(10, 5)
        model_data, metrics = storage.load_model(new_model, version=1)

        assert "model_state_dict" in model_data
        assert metrics["accuracy"] == 0.95

    def test_load_model_not_found(
        self,
        storage: GCSModelStorage,
        simple_model: torch.nn.Module,
    ) -> None:
        """Test error when model version doesn't exist."""
        with pytest.raises(FileNotFoundError):
            storage.load_model(simple_model, version=999)

    def test_load_latest_model(
        self,
        storage: GCSModelStorage,
        simple_model: torch.nn.Module,
    ) -> None:
        """Test loading latest model version."""
        # Save multiple versions
        storage.save_model(simple_model, version=1)
        storage.save_model(simple_model, version=2, metrics={"accuracy": 0.98})

        # Load latest
        new_model = torch.nn.Linear(10, 5)
        model_data, metrics = storage.load_model(new_model, version=None)

        assert model_data["version"] == 2
        assert metrics["accuracy"] == 0.98

    def test_get_latest_version(
        self,
        storage: GCSModelStorage,
        simple_model: torch.nn.Module,
    ) -> None:
        """Test getting latest version number."""
        assert storage.get_latest_version() is None

        storage.save_model(simple_model, version=1)
        storage.save_model(simple_model, version=5)
        storage.save_model(simple_model, version=3)

        assert storage.get_latest_version() == 5

    def test_list_versions(
        self,
        storage: GCSModelStorage,
        simple_model: torch.nn.Module,
    ) -> None:
        """Test listing all versions."""
        storage.save_model(simple_model, version=1)
        storage.save_model(simple_model, version=5)
        storage.save_model(simple_model, version=3)

        versions = storage.list_versions()

        assert versions == [1, 3, 5]
