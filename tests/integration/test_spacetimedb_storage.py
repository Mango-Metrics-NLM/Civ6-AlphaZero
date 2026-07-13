"""Integration tests for SpacetimeDB storage.

Tests end-to-end workflows using the mock client, verifying that
all components work together correctly.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from alpha_zero.spacetimedb.client import SpacetimeDBClient
from alpha_zero.spacetimedb.config import SpacetimeDBConfig
from alpha_zero.spacetimedb.metrics_sink import SpacetimeDBMetricsSink
from alpha_zero.spacetimedb.serialization import (
    serialize_state_dict,
)
from alpha_zero.spacetimedb.weight_sync import SpacetimeDBWeightSynchronizer
from alpha_zero.storage.config import StorageConfig
from alpha_zero.storage.models import EvaluationResult, ModelMetadata
from alpha_zero.storage.spacetimedb_backend import SpacetimeDBStorageBackend
from tests.mocks.spacetimedb_mocks import MockSpacetimeDBClient


class SimpleModel(nn.Module):
    """Simple model for integration testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).mean(dim=(2, 3)))


@pytest.fixture
def config() -> SpacetimeDBConfig:
    """Create test config."""
    return SpacetimeDBConfig(
        enabled=True,
        host="http://localhost:3000",
        module_address="test-module",
        connection_timeout=1.0,
        max_weight_blob_mb=100,
    )


@pytest.fixture
def mock_sdk() -> MockSpacetimeDBClient:
    """Create mock SDK."""
    return MockSpacetimeDBClient()


@pytest.fixture
def storage_config(config: SpacetimeDBConfig) -> StorageConfig:
    """Create storage config."""
    return StorageConfig(
        backend_type="spacetimedb",
        spacetimedb=config,
        compress_weights=False,
    )


@pytest.fixture
def backend(
    storage_config: StorageConfig, mock_sdk: MockSpacetimeDBClient,
) -> SpacetimeDBStorageBackend:
    """Create connected storage backend."""
    backend = SpacetimeDBStorageBackend(storage_config)
    backend.connect(mock_client=mock_sdk)
    return backend


@pytest.fixture
def client(config: SpacetimeDBConfig, mock_sdk: MockSpacetimeDBClient) -> SpacetimeDBClient:
    """Create connected client."""
    c = SpacetimeDBClient(config)
    c.connect_with_mock(mock_sdk)
    return c


class TestModelRoundTrip:
    """Test model save → load round-trip through SpacetimeDB."""

    def test_save_and_load_model(
        self, backend: SpacetimeDBStorageBackend,
    ) -> None:
        """Save model weights and load them back, verifying equality."""
        model = SimpleModel()
        state_dict = model.state_dict()

        # Save
        path = backend.save_model(state_dict, iteration=1)
        assert path is not None

        # Load
        loaded_dict, _metadata = backend.load_model(iteration=1)

        # Verify
        for key in state_dict:
            assert key in loaded_dict
            assert torch.allclose(state_dict[key], loaded_dict[key])

    def test_champion_round_trip(
        self, backend: SpacetimeDBStorageBackend,
    ) -> None:
        """Save as champion and load champion, verifying equality."""
        model = SimpleModel()
        state_dict = model.state_dict()
        metadata = ModelMetadata(iteration=1, version=1, win_rate=0.6)

        backend.save_champion(state_dict, metadata)
        loaded_dict, _loaded_meta = backend.load_champion()

        for key in state_dict:
            assert torch.allclose(state_dict[key], loaded_dict[key])


class TestWeightSyncWorkflow:
    """Test the weight publish → subscribe → sync workflow."""

    def test_publish_and_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        """Publish weights through client and sync to a model via subscription."""
        model = SimpleModel()
        original_weights = model.conv.weight.clone()

        sync = SpacetimeDBWeightSynchronizer(
            client, update_interval=0.01, compress=False,
        )

        # Publish through client (fires events that sync subscribes to)
        new_state = model.state_dict()
        new_state["conv.weight"] = torch.randn_like(new_state["conv.weight"])
        weights_bytes = serialize_state_dict(new_state, compress=False)

        client.publish_weights_sync(
            iteration=1, version=1,
            weights_bytes=weights_bytes,
            metadata={"version": 1},
        )

        # Sync should detect the update via subscription
        updated, version = sync.update_if_available(model)
        assert updated is True
        assert version == 1
        assert not torch.equal(model.conv.weight, original_weights)

    def test_multiple_version_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        """Multiple version publishes, sync always gets latest."""
        model = SimpleModel()
        sync = SpacetimeDBWeightSynchronizer(
            client, update_interval=0.01, compress=False,
        )

        for v in range(1, 4):
            state = model.state_dict()
            state["fc.weight"] = torch.randn_like(state["fc.weight"]) * v
            bytes_data = serialize_state_dict(state, compress=False)
            client.publish_weights_sync(
                iteration=v, version=v,
                weights_bytes=bytes_data,
                metadata={"version": v},
            )

        updated, version = sync.update_if_available(model)
        assert updated is True
        # Should have synced to latest
        assert version == 3


class TestMetricsWorkflow:
    """Test metrics logging through the full pipeline."""

    def test_metrics_pipeline(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        """Log metrics through sink and verify they reach the mock."""
        sink = SpacetimeDBMetricsSink(client, batch_size=5)

        # Simulate training loop
        for step in range(10):
            sink.log_metrics(
                step=step,
                metrics={
                    "total_loss": 1.0 / (step + 1),
                    "policy_loss": 0.6 / (step + 1),
                    "value_loss": 0.4 / (step + 1),
                },
            )

        sink.flush()

        assert sink.total_metrics_logged == 10
        assert mock_sdk.call_counts["log_metrics"] == 10

    def test_protocol_compliance(
        self, client: SpacetimeDBClient,
    ) -> None:
        """Verify CloudMonitoringProtocol compliance."""
        from alpha_zero.cloud.protocols import CloudMonitoringProtocol

        sink = SpacetimeDBMetricsSink(client)
        assert isinstance(sink, CloudMonitoringProtocol)


class TestFullTrainingIteration:
    """Test a complete training iteration workflow."""

    def test_full_iteration(
        self, backend: SpacetimeDBStorageBackend, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        """Simulate a complete training iteration.

        1. Save initial model
        2. Submit trajectories from workers
        3. Save checkpoint
        4. Log metrics
        5. Save evaluation
        6. Promote champion
        """
        model = SimpleModel()
        state_dict = model.state_dict()

        # 1. Save initial model
        path = backend.save_model(state_dict, iteration=0)
        assert path is not None

        # 2. Simulate trajectories (use mock directly for trajectory data)
        from unittest.mock import MagicMock

        for i in range(3):
            traj = MagicMock()
            traj.to_bytes.return_value = f"traj_{i}".encode()
            traj.examples = list(range(10))
            traj.game_length = 20
            backend.save_trajectory(
                traj, iteration=1, worker_id=0, game_id=f"game_{i}",
            )

        assert backend.count_trajectories(iteration=1) == 3

        # 3. Save checkpoint
        checkpoint = {"model_state_dict": state_dict, "iteration": 1}
        backend.save_checkpoint(checkpoint, iteration=1)

        # 4. Log metrics via client
        backend.client.log_metrics_sync(
            step=100, metrics={"total_loss": 0.3, "policy_loss": 0.2},
        )

        # 5. Save evaluation
        eval_result = EvaluationResult(
            iteration=1,
            candidate_version=1,
            champion_version=0,
            num_games=100,
            candidate_wins=55,
            champion_wins=40,
            draws=5,
            win_rate=0.55,
            promoted=True,
        )
        backend.save_evaluation(eval_result)

        # 6. Promote champion
        backend.save_champion(
            state_dict,
            ModelMetadata(iteration=1, version=1, win_rate=0.55),
        )

        # Verify all operations
        assert backend.champion_exists()
        assert backend.get_champion_version() == 1
        assert len(backend.list_evaluations()) == 1
        assert len(backend.list_checkpoints()) == 1
        assert mock_sdk.call_counts["publish_weights"] >= 2  # initial + champion
        assert mock_sdk.call_counts["log_metrics"] == 1


class TestFactoryCreation:
    """Test creating SpacetimeDB backend through the factory."""

    def test_factory_creates_backend(self) -> None:
        from alpha_zero.storage.factory import create_storage_backend

        config = StorageConfig.for_spacetimedb(
            module_address="test-module",
            host="http://localhost:3000",
        )
        backend = create_storage_backend(config)
        assert isinstance(backend, SpacetimeDBStorageBackend)

    def test_factory_with_explicit_type(self) -> None:
        from alpha_zero.storage.factory import create_storage_backend

        config = StorageConfig(
            spacetimedb=SpacetimeDBConfig(enabled=True, module_address="test"),
        )
        backend = create_storage_backend(config, backend_type="spacetimedb")
        assert isinstance(backend, SpacetimeDBStorageBackend)
