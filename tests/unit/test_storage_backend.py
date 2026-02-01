"""Unit tests for storage backend implementations.

Tests the abstract interface and LocalStorageBackend implementation.
GCS backend tests require the google-cloud-storage package and are
skipped if not available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

# Import shared_storage types for trajectory testing
from alpha_zero.distributed.shared_storage import GameTrajectory, TrainingExample
from alpha_zero.storage import (
    EvaluationResult,
    LocalStorageBackend,
    ModelMetadata,
    StorageConfig,
    create_storage_backend,
)


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """Create a temporary storage directory."""
    return tmp_path / "storage"


@pytest.fixture
def local_config(temp_storage_dir: Path) -> StorageConfig:
    """Create a local storage config."""
    return StorageConfig(
        backend_type="local",
        base_path=str(temp_storage_dir),
        compress_weights=False,  # Faster for tests
        compress_trajectories=False,
    )


@pytest.fixture
def local_backend(local_config: StorageConfig) -> LocalStorageBackend:
    """Create a local storage backend."""
    return LocalStorageBackend(local_config)


@pytest.fixture
def sample_state_dict() -> dict[str, torch.Tensor]:
    """Create a sample model state dict."""
    return {
        "layer1.weight": torch.randn(64, 32),
        "layer1.bias": torch.randn(64),
        "layer2.weight": torch.randn(32, 64),
        "layer2.bias": torch.randn(32),
    }


@pytest.fixture
def sample_trajectory() -> GameTrajectory:
    """Create a sample game trajectory."""
    examples = [
        TrainingExample(
            spatial=np.random.randn(13, 8, 8).astype(np.float32),
            vector=np.random.randn(64).astype(np.float32),
            action_mask=np.ones(356, dtype=np.float32),
            policy=np.random.rand(356).astype(np.float32),
            value=np.random.uniform(-1, 1),
        )
        for _ in range(10)
    ]
    return GameTrajectory(
        examples=examples,
        game_id=42,
        actor_id=0,
        game_length=50,
        winner=0,
    )


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = StorageConfig()
        assert config.backend_type == "local"
        assert config.base_path == "./storage"
        assert config.compress_weights is True

    def test_for_local(self) -> None:
        """Test local config factory."""
        config = StorageConfig.for_local("/tmp/test")
        assert config.backend_type == "local"
        assert config.base_path == "/tmp/test"

    def test_for_gcs(self) -> None:
        """Test GCS config factory."""
        config = StorageConfig.for_gcs("my-bucket", "my-prefix")
        assert config.backend_type == "gcs"
        assert config.gcs.bucket == "my-bucket"
        assert config.gcs.prefix == "my-prefix"

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        data = {
            "backend_type": "local",
            "base_path": "/custom/path",
            "compress_weights": False,
            "models_subdir": "custom_models",
        }
        config = StorageConfig.from_dict(data)
        assert config.backend_type == "local"
        assert config.base_path == "/custom/path"
        assert config.compress_weights is False
        assert config.models_subdir == "custom_models"

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = StorageConfig.for_local("/tmp/test")
        data = config.to_dict()
        assert data["backend_type"] == "local"
        assert data["base_path"] == "/tmp/test"

    def test_env_override(self) -> None:
        """Test environment variable override."""
        with patch.dict(os.environ, {"CIV6_STORAGE_BACKEND": "local"}):
            config = StorageConfig.from_env()
            assert config.backend_type == "local"

    def test_get_paths(self, local_config: StorageConfig) -> None:
        """Test path helper methods."""
        models_path = local_config.get_models_path()
        assert "models" in models_path

        games_path = local_config.get_games_path()
        assert "games" in games_path


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""

    def test_creation(self) -> None:
        """Test metadata creation."""
        metadata = ModelMetadata(
            iteration=100,
            version=5,
            elo_rating=1200.0,
            win_rate=0.65,
        )
        assert metadata.iteration == 100
        assert metadata.version == 5
        assert metadata.elo_rating == 1200.0

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        metadata = ModelMetadata(iteration=100, version=5)
        json_str = metadata.to_json()
        data = json.loads(json_str)
        assert data["iteration"] == 100
        assert data["version"] == 5

    def test_from_json(self) -> None:
        """Test JSON deserialization."""
        json_str = '{"iteration": 50, "version": 3}'
        metadata = ModelMetadata.from_json(json_str)
        assert metadata.iteration == 50
        assert metadata.version == 3


class TestLocalStorageBackend:
    """Tests for LocalStorageBackend."""

    def test_initialization(self, local_backend: LocalStorageBackend) -> None:
        """Test backend initialization creates directories."""
        assert local_backend.base_path.exists()
        assert (local_backend.base_path / "models").exists()
        assert (local_backend.base_path / "games").exists()
        assert (local_backend.base_path / "checkpoints").exists()

    def test_save_and_load_model(
        self,
        local_backend: LocalStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Test saving and loading model weights."""
        # Save model
        path = local_backend.save_model(
            state_dict=sample_state_dict,
            iteration=10,
        )
        assert Path(path).exists()

        # Load model
        loaded_dict, metadata = local_backend.load_model(iteration=10)

        # Verify weights match
        for key in sample_state_dict:
            assert key in loaded_dict
            assert torch.allclose(sample_state_dict[key], loaded_dict[key])

        # Verify metadata
        assert metadata.iteration == 10

    def test_save_with_metadata(
        self,
        local_backend: LocalStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Test saving model with custom metadata."""
        metadata = ModelMetadata(
            iteration=20,
            version=5,
            elo_rating=1500.0,
            win_rate=0.75,
        )

        local_backend.save_model(
            state_dict=sample_state_dict,
            iteration=20,
            metadata=metadata,
        )

        _, loaded_metadata = local_backend.load_model(iteration=20)
        assert loaded_metadata.version == 5
        assert loaded_metadata.elo_rating == 1500.0

    def test_load_nonexistent_model(self, local_backend: LocalStorageBackend) -> None:
        """Test loading non-existent model raises error."""
        with pytest.raises(FileNotFoundError):
            local_backend.load_model(iteration=999)

    def test_champion_operations(
        self,
        local_backend: LocalStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Test champion model save/load operations."""
        # Initially no champion
        assert not local_backend.champion_exists()
        assert local_backend.get_champion_version() == 0

        # Save champion
        metadata = ModelMetadata(iteration=50, version=10, win_rate=0.8)
        local_backend.save_champion(sample_state_dict, metadata)

        # Verify champion exists
        assert local_backend.champion_exists()
        assert local_backend.get_champion_version() == 10

        # Load champion
        loaded_dict, loaded_metadata = local_backend.load_champion()
        assert loaded_metadata.version == 10
        for key in sample_state_dict:
            assert torch.allclose(sample_state_dict[key], loaded_dict[key])

    def test_trajectory_operations(
        self,
        local_backend: LocalStorageBackend,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Test trajectory save/load operations."""
        # Save trajectory
        path = local_backend.save_trajectory(
            trajectory=sample_trajectory,
            iteration=5,
            worker_id=0,
            game_id="game_001",
        )
        assert Path(path).exists()

        # List trajectories
        metadata_list = local_backend.list_trajectories(iteration=5)
        assert len(metadata_list) == 1
        assert metadata_list[0].game_id == "game_001"
        assert metadata_list[0].worker_id == 0

        # Load trajectory
        loaded_trajectory = local_backend.load_trajectory(
            iteration=5,
            worker_id=0,
            game_id="game_001",
        )
        assert len(loaded_trajectory.examples) == len(sample_trajectory.examples)

    def test_trajectory_filtering(
        self,
        local_backend: LocalStorageBackend,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Test trajectory listing with filters."""
        # Save multiple trajectories across iterations and workers
        game_counter = 0
        for iteration in [1, 2, 2]:
            for worker_id in [0, 1]:
                local_backend.save_trajectory(
                    trajectory=sample_trajectory,
                    iteration=iteration,
                    worker_id=worker_id,
                    game_id=f"game_{game_counter}",
                )
                game_counter += 1

        # Filter by iteration
        iter1_trajectories = local_backend.list_trajectories(iteration=1)
        assert len(iter1_trajectories) == 2  # 2 workers in iteration 1

        iter2_trajectories = local_backend.list_trajectories(iteration=2)
        assert len(iter2_trajectories) == 4  # 2 workers x 2 saves at iteration 2

        # Filter by worker
        worker0_trajectories = local_backend.list_trajectories(
            iteration=2, worker_id=0
        )
        assert len(worker0_trajectories) == 2

    def test_stream_trajectories(
        self,
        local_backend: LocalStorageBackend,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Test streaming trajectories."""
        # Save multiple trajectories
        for i in range(5):
            local_backend.save_trajectory(
                trajectory=sample_trajectory,
                iteration=1,
                worker_id=0,
                game_id=f"game_{i}",
            )

        # Stream all trajectories
        streamed = list(local_backend.stream_trajectories(iterations=[1]))
        assert len(streamed) == 5

    def test_checkpoint_operations(
        self,
        local_backend: LocalStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Test checkpoint save/load operations."""
        checkpoint_data = {
            "model_state_dict": sample_state_dict,
            "optimizer_state_dict": {"param_groups": []},
            "training_step": 1000,
        }

        # Save checkpoint
        path = local_backend.save_checkpoint(
            checkpoint_data=checkpoint_data,
            iteration=100,
        )
        assert Path(path).exists()

        # Load checkpoint
        loaded_data, metadata = local_backend.load_checkpoint(iteration=100)
        assert loaded_data["training_step"] == 1000
        assert "model_state_dict" in loaded_data

        # List checkpoints
        checkpoints = local_backend.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0].iteration == 100

        # Load latest
        latest_data, latest_metadata = local_backend.load_latest_checkpoint()
        assert latest_metadata.iteration == 100

    def test_evaluation_operations(self, local_backend: LocalStorageBackend) -> None:
        """Test evaluation result save/load operations."""
        result = EvaluationResult(
            iteration=50,
            candidate_version=10,
            champion_version=9,
            num_games=400,
            candidate_wins=230,
            champion_wins=150,
            draws=20,
            win_rate=0.575,
            promoted=True,
        )

        # Save evaluation
        path = local_backend.save_evaluation(result)
        assert Path(path).exists()

        # Load evaluation
        loaded_result = local_backend.load_evaluation(iteration=50)
        assert loaded_result.win_rate == 0.575
        assert loaded_result.promoted is True

        # List evaluations
        evaluations = local_backend.list_evaluations()
        assert len(evaluations) == 1

    def test_config_operations(self, local_backend: LocalStorageBackend) -> None:
        """Test config save/load operations."""
        config = {
            "learning_rate": 0.001,
            "batch_size": 256,
            "num_simulations": 800,
        }

        # Save config
        path = local_backend.save_config(config, iteration=25)
        assert Path(path).exists()

        # Load config
        loaded_config = local_backend.load_config(iteration=25)
        assert loaded_config["learning_rate"] == 0.001

    def test_utility_operations(
        self,
        local_backend: LocalStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Test utility operations (exists, delete, list)."""
        # Save a model
        path = local_backend.save_model(sample_state_dict, iteration=1)

        # Check exists
        assert local_backend.exists(path)
        assert not local_backend.exists("/nonexistent/path")

        # List files (use *.pt* to match both compressed and uncompressed)
        files = local_backend.list_files(
            str(local_backend.base_path / "models"), "*.pt*"
        )
        assert len(files) > 0

        # Delete
        assert local_backend.delete(path)
        assert not local_backend.exists(path)

    def test_context_manager(self, local_config: StorageConfig) -> None:
        """Test using backend as context manager."""
        with LocalStorageBackend(local_config) as backend:
            assert backend.base_path.exists()


class TestStorageFactory:
    """Tests for create_storage_backend factory."""

    def test_create_local_backend(self, temp_storage_dir: Path) -> None:
        """Test creating local backend."""
        config = StorageConfig.for_local(str(temp_storage_dir))
        backend = create_storage_backend(config)
        assert isinstance(backend, LocalStorageBackend)

    def test_create_with_explicit_type(self, temp_storage_dir: Path) -> None:
        """Test creating backend with explicit type."""
        config = StorageConfig(base_path=str(temp_storage_dir))
        backend = create_storage_backend(config, backend_type="local")
        assert isinstance(backend, LocalStorageBackend)

    def test_invalid_backend_type(self) -> None:
        """Test invalid backend type raises error."""
        config = StorageConfig()
        with pytest.raises(ValueError, match="Invalid backend type"):
            create_storage_backend(config, backend_type="invalid")

    def test_gcs_missing_bucket(self) -> None:
        """Test GCS backend without bucket raises error."""
        config = StorageConfig(backend_type="gcs")
        config.gcs.bucket = ""  # Explicitly empty
        # This should either raise ImportError (no gcs package) or ValueError (no bucket)
        with pytest.raises((ImportError, ValueError)):
            create_storage_backend(config)


class TestCompression:
    """Tests for compression functionality."""

    def test_compressed_weights(self, temp_storage_dir: Path) -> None:
        """Test saving/loading with compressed weights."""
        config = StorageConfig(
            backend_type="local",
            base_path=str(temp_storage_dir),
            compress_weights=True,
        )
        backend = LocalStorageBackend(config)

        state_dict = {"weight": torch.randn(100, 100)}
        backend.save_model(state_dict, iteration=1)

        loaded_dict, _ = backend.load_model(iteration=1)
        assert torch.allclose(state_dict["weight"], loaded_dict["weight"])

    def test_compressed_trajectories(
        self,
        temp_storage_dir: Path,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Test saving/loading with compressed trajectories."""
        config = StorageConfig(
            backend_type="local",
            base_path=str(temp_storage_dir),
            compress_trajectories=True,
        )
        backend = LocalStorageBackend(config)

        backend.save_trajectory(
            trajectory=sample_trajectory,
            iteration=1,
            worker_id=0,
            game_id="game_1",
        )

        loaded = backend.load_trajectory(
            iteration=1,
            worker_id=0,
            game_id="game_1",
        )
        assert len(loaded.examples) == len(sample_trajectory.examples)


class TestVertexAIIntegration:
    """Tests for Vertex AI environment variable integration."""

    def test_aip_checkpoint_dir_local(self) -> None:
        """Test AIP_CHECKPOINT_DIR with local path."""
        with patch.dict(os.environ, {"AIP_CHECKPOINT_DIR": "/tmp/vertex_checkpoints"}):
            config = StorageConfig()
            # Should use local backend since path doesn't start with gs://
            assert config.backend_type == "local"

    def test_aip_checkpoint_dir_gcs(self) -> None:
        """Test AIP_CHECKPOINT_DIR with GCS path."""
        with patch.dict(
            os.environ, {"AIP_CHECKPOINT_DIR": "gs://my-bucket/checkpoints/job123"}
        ):
            config = StorageConfig()
            # Should detect GCS and parse the path
            assert config.backend_type == "gcs"
            assert config.gcs.bucket == "my-bucket"
            assert "checkpoints/job123" in config.gcs.prefix
