"""Unit tests for SpacetimeDB storage backend."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from alpha_zero.spacetimedb.config import SpacetimeDBConfig
from alpha_zero.storage.config import StorageConfig
from alpha_zero.storage.models import (
    EvaluationResult,
    ModelMetadata,
)
from alpha_zero.storage.spacetimedb_backend import SpacetimeDBStorageBackend
from tests.mocks.spacetimedb_mocks import MockSpacetimeDBClient


@pytest.fixture
def stdb_config() -> SpacetimeDBConfig:
    """Create SpacetimeDB config for testing."""
    return SpacetimeDBConfig(
        enabled=True,
        host="http://localhost:3000",
        module_address="test-module",
        connection_timeout=1.0,
        max_weight_blob_mb=100,
    )


@pytest.fixture
def storage_config(stdb_config: SpacetimeDBConfig) -> StorageConfig:
    """Create storage config with SpacetimeDB settings."""
    return StorageConfig(
        backend_type="spacetimedb",
        spacetimedb=stdb_config,
        compress_weights=False,  # Disable for faster tests
    )


@pytest.fixture
def mock_sdk() -> MockSpacetimeDBClient:
    """Create mock SDK client."""
    return MockSpacetimeDBClient()


@pytest.fixture
def backend(
    storage_config: StorageConfig,
    mock_sdk: MockSpacetimeDBClient,
) -> SpacetimeDBStorageBackend:
    """Create SpacetimeDB storage backend connected with mock."""
    backend = SpacetimeDBStorageBackend(storage_config)
    backend.connect(mock_client=mock_sdk)
    return backend


@pytest.fixture
def sample_state_dict() -> dict[str, torch.Tensor]:
    """Create a small model state dict for testing."""
    return {
        "layer1.weight": torch.randn(4, 4),
        "layer1.bias": torch.randn(4),
        "layer2.weight": torch.randn(2, 4),
        "layer2.bias": torch.randn(2),
    }


class TestModelOperations:
    """Test model save/load operations."""

    def test_save_model(
        self,
        backend: SpacetimeDBStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        path = backend.save_model(sample_state_dict, iteration=1)
        assert path.startswith("spacetimedb://models/")
        assert "1" in path

    def test_save_model_increments_version(
        self,
        backend: SpacetimeDBStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        backend.save_model(sample_state_dict, iteration=1)
        backend.save_model(sample_state_dict, iteration=2)
        # Client should have received two publish calls
        assert backend.client._connection.sdk_client.call_counts["publish_weights"] == 2

    def test_save_model_with_metadata(
        self,
        backend: SpacetimeDBStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        metadata = ModelMetadata(
            iteration=5,
            version=5,
            elo_rating=1200.0,
            win_rate=0.55,
        )
        path = backend.save_model(sample_state_dict, iteration=5, metadata=metadata)
        assert "5" in path

    def test_load_model(
        self,
        backend: SpacetimeDBStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        backend.save_model(sample_state_dict, iteration=1)
        loaded_dict, _metadata = backend.load_model(iteration=1)

        for key in sample_state_dict:
            assert key in loaded_dict
            assert torch.allclose(sample_state_dict[key], loaded_dict[key])

    def test_load_model_not_found(
        self,
        backend: SpacetimeDBStorageBackend,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="No model found"):
            backend.load_model(iteration=999)

    def test_load_model_by_version(
        self,
        backend: SpacetimeDBStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        backend.save_model(sample_state_dict, iteration=1)
        loaded_dict, _metadata = backend.load_model_by_version(version=1)

        for key in sample_state_dict:
            assert key in loaded_dict


class TestChampionOperations:
    """Test champion model operations."""

    def test_save_champion(
        self,
        backend: SpacetimeDBStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        metadata = ModelMetadata(iteration=1, version=1)
        path = backend.save_champion(sample_state_dict, metadata)
        assert "champion" in path

    def test_load_champion(
        self,
        backend: SpacetimeDBStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        metadata = ModelMetadata(iteration=1, version=1)
        backend.save_champion(sample_state_dict, metadata)

        loaded_dict, _loaded_meta = backend.load_champion()
        for key in sample_state_dict:
            assert key in loaded_dict

    def test_load_champion_not_found(
        self,
        backend: SpacetimeDBStorageBackend,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="No champion"):
            backend.load_champion()

    def test_champion_exists(
        self,
        backend: SpacetimeDBStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        assert not backend.champion_exists()
        metadata = ModelMetadata(iteration=1, version=1)
        backend.save_champion(sample_state_dict, metadata)
        assert backend.champion_exists()

    def test_get_champion_version(
        self,
        backend: SpacetimeDBStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        assert backend.get_champion_version() == 0
        metadata = ModelMetadata(iteration=1, version=3)
        backend.save_champion(sample_state_dict, metadata)
        assert backend.get_champion_version() == 3


class TestTrajectoryOperations:
    """Test trajectory save/list/load operations."""

    @pytest.fixture
    def mock_trajectory(self) -> Any:
        """Create a mock trajectory object."""
        from unittest.mock import MagicMock

        traj = MagicMock()
        traj.to_bytes.return_value = b"trajectory_data"
        traj.examples = [1, 2, 3]
        traj.game_length = 10
        return traj

    def test_save_trajectory(
        self,
        backend: SpacetimeDBStorageBackend,
        mock_trajectory: Any,
    ) -> None:
        path = backend.save_trajectory(
            mock_trajectory,
            iteration=1,
            worker_id=0,
            game_id="g1",
        )
        assert "trajectories" in path

    def test_list_trajectories(
        self,
        backend: SpacetimeDBStorageBackend,
        mock_trajectory: Any,
    ) -> None:
        backend.save_trajectory(mock_trajectory, iteration=1, worker_id=0, game_id="g1")
        backend.save_trajectory(mock_trajectory, iteration=1, worker_id=1, game_id="g2")

        all_trajs = backend.list_trajectories()
        assert len(all_trajs) == 2

    def test_list_trajectories_by_iteration(
        self,
        backend: SpacetimeDBStorageBackend,
        mock_trajectory: Any,
    ) -> None:
        backend.save_trajectory(mock_trajectory, iteration=1, worker_id=0, game_id="g1")
        backend.save_trajectory(mock_trajectory, iteration=2, worker_id=0, game_id="g2")

        iter1 = backend.list_trajectories(iteration=1)
        assert len(iter1) == 1
        assert iter1[0].iteration == 1

    def test_list_trajectories_by_worker(
        self,
        backend: SpacetimeDBStorageBackend,
        mock_trajectory: Any,
    ) -> None:
        backend.save_trajectory(mock_trajectory, iteration=1, worker_id=0, game_id="g1")
        backend.save_trajectory(mock_trajectory, iteration=1, worker_id=1, game_id="g2")

        worker0 = backend.list_trajectories(worker_id=0)
        assert len(worker0) == 1
        assert worker0[0].worker_id == 0

    def test_count_trajectories(
        self,
        backend: SpacetimeDBStorageBackend,
        mock_trajectory: Any,
    ) -> None:
        assert backend.count_trajectories() == 0
        backend.save_trajectory(mock_trajectory, iteration=1, worker_id=0, game_id="g1")
        assert backend.count_trajectories() == 1
        assert backend.count_trajectories(iteration=1) == 1
        assert backend.count_trajectories(iteration=2) == 0


class TestCheckpointOperations:
    """Test checkpoint save/load operations."""

    @pytest.fixture
    def checkpoint_data(self, sample_state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Create sample checkpoint data."""
        return {
            "model_state_dict": sample_state_dict,
            "iteration": 10,
            "step": 1000,
        }

    def test_save_checkpoint(
        self,
        backend: SpacetimeDBStorageBackend,
        checkpoint_data: dict[str, Any],
    ) -> None:
        path = backend.save_checkpoint(checkpoint_data, iteration=10)
        assert "checkpoints" in path

    def test_load_checkpoint(
        self,
        backend: SpacetimeDBStorageBackend,
        checkpoint_data: dict[str, Any],
    ) -> None:
        backend.save_checkpoint(checkpoint_data, iteration=10)
        loaded_data, _metadata = backend.load_checkpoint(iteration=10)
        assert loaded_data["iteration"] == 10

    def test_load_checkpoint_not_found(
        self,
        backend: SpacetimeDBStorageBackend,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            backend.load_checkpoint(iteration=999)

    def test_load_latest_checkpoint(
        self,
        backend: SpacetimeDBStorageBackend,
        checkpoint_data: dict[str, Any],
    ) -> None:
        backend.save_checkpoint(checkpoint_data, iteration=5)
        checkpoint_data["iteration"] = 10
        backend.save_checkpoint(checkpoint_data, iteration=10)

        result = backend.load_latest_checkpoint()
        assert result is not None
        loaded_data, _metadata = result
        assert loaded_data["iteration"] == 10

    def test_load_latest_checkpoint_none(
        self,
        backend: SpacetimeDBStorageBackend,
    ) -> None:
        result = backend.load_latest_checkpoint()
        assert result is None

    def test_list_checkpoints(
        self,
        backend: SpacetimeDBStorageBackend,
        checkpoint_data: dict[str, Any],
    ) -> None:
        backend.save_checkpoint(checkpoint_data, iteration=5)
        backend.save_checkpoint(checkpoint_data, iteration=10)

        checkpoints = backend.list_checkpoints()
        assert len(checkpoints) == 2
        assert checkpoints[0].iteration == 5
        assert checkpoints[1].iteration == 10


class TestEvaluationOperations:
    """Test evaluation save/load operations."""

    @pytest.fixture
    def eval_result(self) -> EvaluationResult:
        """Create sample evaluation result."""
        return EvaluationResult(
            iteration=5,
            candidate_version=5,
            champion_version=3,
            num_games=100,
            candidate_wins=55,
            champion_wins=40,
            draws=5,
            win_rate=0.55,
            promoted=True,
        )

    def test_save_evaluation(
        self,
        backend: SpacetimeDBStorageBackend,
        eval_result: EvaluationResult,
    ) -> None:
        path = backend.save_evaluation(eval_result)
        assert "evaluations" in path

    def test_load_evaluation(
        self,
        backend: SpacetimeDBStorageBackend,
        eval_result: EvaluationResult,
    ) -> None:
        backend.save_evaluation(eval_result)
        loaded = backend.load_evaluation(iteration=5)
        assert loaded.win_rate == 0.55
        assert loaded.promoted is True

    def test_load_evaluation_not_found(
        self,
        backend: SpacetimeDBStorageBackend,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            backend.load_evaluation(iteration=999)

    def test_list_evaluations(
        self,
        backend: SpacetimeDBStorageBackend,
    ) -> None:
        for i in range(3):
            result = EvaluationResult(
                iteration=i,
                candidate_version=i,
                champion_version=0,
                num_games=50,
                candidate_wins=25,
                champion_wins=20,
                draws=5,
                win_rate=0.5,
                promoted=False,
            )
            backend.save_evaluation(result)

        evals = backend.list_evaluations()
        assert len(evals) == 3
        assert evals[0].iteration == 0
        assert evals[2].iteration == 2


class TestConfigOperations:
    """Test config save/load operations."""

    def test_save_config(
        self,
        backend: SpacetimeDBStorageBackend,
    ) -> None:
        config = {"learning_rate": 0.001, "batch_size": 256}
        path = backend.save_config(config, iteration=1)
        assert "configs" in path

    def test_load_config(
        self,
        backend: SpacetimeDBStorageBackend,
    ) -> None:
        config = {"learning_rate": 0.001, "batch_size": 256}
        backend.save_config(config, iteration=1)
        loaded = backend.load_config(iteration=1)
        assert loaded["learning_rate"] == 0.001
        assert loaded["batch_size"] == 256

    def test_load_config_not_found(
        self,
        backend: SpacetimeDBStorageBackend,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            backend.load_config(iteration=999)


class TestUtilityOperations:
    """Test utility methods."""

    def test_exists_model(
        self,
        backend: SpacetimeDBStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        assert not backend.exists("spacetimedb://models/1")
        backend.save_model(sample_state_dict, iteration=1)
        assert backend.exists("spacetimedb://models/1")

    def test_exists_unknown_path(
        self,
        backend: SpacetimeDBStorageBackend,
    ) -> None:
        assert not backend.exists("unknown://path")

    def test_delete_returns_false(
        self,
        backend: SpacetimeDBStorageBackend,
    ) -> None:
        assert backend.delete("spacetimedb://models/1") is False

    def test_list_files_models(
        self,
        backend: SpacetimeDBStorageBackend,
        sample_state_dict: dict[str, torch.Tensor],
    ) -> None:
        backend.save_model(sample_state_dict, iteration=1)
        backend.save_model(sample_state_dict, iteration=2)
        files = backend.list_files("models")
        assert len(files) == 2

    def test_close(
        self,
        backend: SpacetimeDBStorageBackend,
    ) -> None:
        backend.close()
        assert not backend.client.is_connected

    def test_context_manager(
        self,
        storage_config: StorageConfig,
        mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        with SpacetimeDBStorageBackend(storage_config) as backend:
            backend.connect(mock_client=mock_sdk)
            assert backend.client.is_connected
        assert not backend.client.is_connected


class TestStorageConfigIntegration:
    """Test StorageConfig integration with spacetimedb."""

    def test_for_spacetimedb(self) -> None:
        config = StorageConfig.for_spacetimedb(
            module_address="test-module",
            host="http://localhost:3000",
        )
        assert config.backend_type == "spacetimedb"
        assert config.spacetimedb is not None
        assert config.spacetimedb.module_address == "test-module"
        assert config.spacetimedb.enabled is True

    def test_from_dict_with_spacetimedb(self) -> None:
        data = {
            "backend_type": "spacetimedb",
            "spacetimedb": {
                "enabled": True,
                "host": "http://localhost:3000",
                "module_address": "my-module",
            },
        }
        config = StorageConfig.from_dict(data)
        assert config.backend_type == "spacetimedb"
        assert config.spacetimedb is not None
        assert config.spacetimedb.module_address == "my-module"

    def test_to_dict_with_spacetimedb(self) -> None:
        config = StorageConfig.for_spacetimedb("my-module")
        data = config.to_dict()
        assert "spacetimedb" in data
        assert data["spacetimedb"]["module_address"] == "my-module"

    def test_to_dict_without_spacetimedb(self) -> None:
        config = StorageConfig.for_local()
        data = config.to_dict()
        assert "spacetimedb" not in data


class TestFactoryIntegration:
    """Test factory creates SpacetimeDB backend."""

    def test_factory_creates_backend(self, stdb_config: SpacetimeDBConfig) -> None:
        from alpha_zero.storage.factory import create_storage_backend

        config = StorageConfig(
            backend_type="spacetimedb",
            spacetimedb=stdb_config,
        )
        backend = create_storage_backend(config)
        assert isinstance(backend, SpacetimeDBStorageBackend)

    def test_factory_with_explicit_type(self) -> None:
        from alpha_zero.storage.factory import create_storage_backend

        config = StorageConfig(
            spacetimedb=SpacetimeDBConfig(
                enabled=True,
                module_address="test",
            ),
        )
        backend = create_storage_backend(config, backend_type="spacetimedb")
        assert isinstance(backend, SpacetimeDBStorageBackend)
