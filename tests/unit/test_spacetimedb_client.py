"""Unit tests for SpacetimeDB client wrapper."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from alpha_zero.spacetimedb.client import SpacetimeDBClient
from alpha_zero.spacetimedb.config import SpacetimeDBConfig
from alpha_zero.spacetimedb.events import (
    CONNECTION_STATE_CHANGED,
    METRICS_UPDATED,
    MODEL_VERSION_CHANGED,
    TRAJECTORY_ADDED,
    TRAINING_STATE_CHANGED,
    WORKER_STATUS_CHANGED,
)
from alpha_zero.spacetimedb.exceptions import (
    SpacetimeDBConnectionError,
    SpacetimeDBReducerError,
    SpacetimeDBSerializationError,
)
from tests.mocks.spacetimedb_mocks import MockSpacetimeDBClient


@pytest.fixture
def config() -> SpacetimeDBConfig:
    """Create test config."""
    return SpacetimeDBConfig(
        enabled=True,
        host="http://localhost:3000",
        module_address="test-module",
        connection_timeout=1.0,
        subscription_timeout=1.0,
        max_weight_blob_mb=10,
    )


@pytest.fixture
def mock_sdk() -> MockSpacetimeDBClient:
    """Create mock SDK client."""
    return MockSpacetimeDBClient()


@pytest.fixture
def client(config: SpacetimeDBConfig, mock_sdk: MockSpacetimeDBClient) -> SpacetimeDBClient:
    """Create SpacetimeDBClient connected with mock SDK."""
    c = SpacetimeDBClient(config)
    c.connect_with_mock(mock_sdk)
    return c


class TestClientProperties:
    """Test client property accessors."""

    def test_is_connected_false_by_default(self, config: SpacetimeDBConfig) -> None:
        client = SpacetimeDBClient(config)
        assert not client.is_connected

    def test_is_connected_after_mock_connect(self, client: SpacetimeDBClient) -> None:
        assert client.is_connected

    def test_is_healthy_matches_connected(self, client: SpacetimeDBClient) -> None:
        assert client.is_healthy == client.is_connected

    def test_current_version_default(self, client: SpacetimeDBClient) -> None:
        assert client.current_version == 0

    def test_champion_version_default(self, client: SpacetimeDBClient) -> None:
        assert client.champion_version == 0

    def test_events_accessible(self, client: SpacetimeDBClient) -> None:
        assert client.events is not None

    def test_config_accessible(self, client: SpacetimeDBClient, config: SpacetimeDBConfig) -> None:
        assert client.config is config


class TestConnectionLifecycle:
    """Test connection lifecycle methods."""

    def test_connect_with_mock(
        self, config: SpacetimeDBConfig, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        client = SpacetimeDBClient(config)
        assert not client.is_connected
        result = client.connect_with_mock(mock_sdk)
        assert result is True
        assert client.is_connected

    def test_disconnect_sync(self, client: SpacetimeDBClient) -> None:
        assert client.is_connected
        client.disconnect_sync()
        assert not client.is_connected

    def test_disconnect_clears_events(self, client: SpacetimeDBClient) -> None:
        callback = MagicMock()
        client.subscribe_model_versions(callback)
        assert client.events.has_handlers(MODEL_VERSION_CHANGED)
        client.disconnect_sync()
        assert not client.events.has_handlers(MODEL_VERSION_CHANGED)


class TestWeightOperations:
    """Test weight publish/get operations."""

    def test_publish_weights_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        result = client.publish_weights_sync(
            iteration=1,
            version=1,
            weights_bytes=b"fake_weights",
            metadata={"loss": 0.5},
        )
        assert result is True
        assert mock_sdk.call_counts["publish_weights"] == 1
        assert client.current_version == 1

    def test_publish_weights_champion(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        result = client.publish_weights_sync(
            iteration=1,
            version=1,
            weights_bytes=b"weights",
            metadata={},
            is_champion=True,
        )
        assert result is True
        assert client.champion_version == 1

    def test_publish_weights_fires_event(
        self, client: SpacetimeDBClient,
    ) -> None:
        versions_received: list[int] = []
        client.subscribe_model_versions(lambda v: versions_received.append(v))

        client.publish_weights_sync(
            iteration=1, version=5, weights_bytes=b"w", metadata={},
        )
        assert versions_received == [5]

    def test_publish_weights_size_limit(
        self, client: SpacetimeDBClient,
    ) -> None:
        # Config has max_weight_blob_mb=10
        large_blob = b"x" * (11 * 1024 * 1024)  # 11 MB
        with pytest.raises(SpacetimeDBSerializationError, match="exceeds"):
            client.publish_weights_sync(
                iteration=1, version=1,
                weights_bytes=large_blob, metadata={},
            )

    def test_publish_weights_not_connected(
        self, config: SpacetimeDBConfig,
    ) -> None:
        client = SpacetimeDBClient(config)
        with pytest.raises(SpacetimeDBConnectionError, match="Not connected"):
            client.publish_weights_sync(
                iteration=1, version=1,
                weights_bytes=b"w", metadata={},
            )

    def test_get_latest_weights_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        # Publish first
        mock_sdk.publish_weights_sync(
            iteration=1, version=1,
            weights_bytes=b"latest_weights", metadata={"iter": 1},
        )
        result = client.get_latest_weights_sync()
        assert result is not None
        assert result[0] == b"latest_weights"

    def test_get_latest_weights_empty(
        self, client: SpacetimeDBClient,
    ) -> None:
        result = client.get_latest_weights_sync()
        assert result is None

    def test_get_champion_weights_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        mock_sdk.publish_weights_sync(
            iteration=1, version=1,
            weights_bytes=b"champ", metadata={},
            is_champion=True,
        )
        result = client.get_champion_weights_sync()
        assert result is not None
        assert result[0] == b"champ"

    def test_get_weights_by_version_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        mock_sdk.publish_weights_sync(
            iteration=1, version=3,
            weights_bytes=b"v3_weights", metadata={},
        )
        result = client.get_weights_by_version_sync(3)
        assert result is not None
        assert result[0] == b"v3_weights"

    def test_get_weights_by_version_not_found(
        self, client: SpacetimeDBClient,
    ) -> None:
        result = client.get_weights_by_version_sync(999)
        assert result is None

    def test_version_tracks_max(
        self, client: SpacetimeDBClient,
    ) -> None:
        client.publish_weights_sync(
            iteration=1, version=3, weights_bytes=b"w", metadata={},
        )
        client.publish_weights_sync(
            iteration=2, version=1, weights_bytes=b"w", metadata={},
        )
        # Should track the max version
        assert client.current_version == 3


class TestTrajectoryOperations:
    """Test trajectory submit operations."""

    def test_submit_trajectory_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        result = client.submit_trajectory_sync(
            iteration=1,
            worker_id=0,
            game_id="game_001",
            trajectory_bytes=b"trajectory_data",
            num_examples=10,
        )
        assert result is True
        assert mock_sdk.call_counts["submit_trajectory"] == 1

    def test_submit_trajectory_fires_event(
        self, client: SpacetimeDBClient,
    ) -> None:
        trajectories_received: list[str] = []
        client.subscribe_trajectories(lambda tid: trajectories_received.append(tid))

        client.submit_trajectory_sync(
            iteration=1, worker_id=0, game_id="g1",
            trajectory_bytes=b"data", num_examples=5,
        )
        assert len(trajectories_received) == 1
        assert "1_0_g1" in trajectories_received[0]

    def test_submit_trajectory_not_connected(
        self, config: SpacetimeDBConfig,
    ) -> None:
        client = SpacetimeDBClient(config)
        with pytest.raises(SpacetimeDBConnectionError, match="Not connected"):
            client.submit_trajectory_sync(
                iteration=1, worker_id=0, game_id="g1",
                trajectory_bytes=b"data", num_examples=5,
            )


class TestMetricsOperations:
    """Test metrics logging operations."""

    def test_log_metrics_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        result = client.log_metrics_sync(
            step=100,
            metrics={"total_loss": 0.5, "policy_loss": 0.3, "value_loss": 0.2},
        )
        assert result is True
        assert mock_sdk.call_counts["log_metrics"] == 1

    def test_log_metrics_fires_event(
        self, client: SpacetimeDBClient,
    ) -> None:
        metrics_received: list[tuple[int, dict[str, float]]] = []
        client.subscribe_metrics(
            lambda step, metrics: metrics_received.append((step, metrics)),
        )

        client.log_metrics_sync(step=50, metrics={"loss": 0.1})
        assert len(metrics_received) == 1
        assert metrics_received[0][0] == 50


class TestEvaluationOperations:
    """Test evaluation save operations."""

    def test_save_evaluation_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        result = client.save_evaluation_sync({
            "iteration": 5,
            "win_rate": 0.55,
            "games_played": 100,
            "promoted": True,
        })
        assert result is True
        assert mock_sdk.call_counts["save_evaluation"] == 1


class TestCheckpointOperations:
    """Test checkpoint save operations."""

    def test_save_checkpoint_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        result = client.save_checkpoint_sync(
            iteration=10,
            data=b"checkpoint_data",
            metadata={"model_version": 5},
        )
        assert result is True
        assert mock_sdk.call_counts["save_checkpoint"] == 1


class TestConfigOperations:
    """Test config save operations."""

    def test_save_config_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        result = client.save_config_sync(
            iteration=1,
            config_json='{"learning_rate": 0.001}',
        )
        assert result is True
        assert mock_sdk.call_counts["save_config"] == 1


class TestWorkerOperations:
    """Test worker registration and heartbeat."""

    def test_register_worker_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        result = client.register_worker_sync("selfplay", "worker_0")
        assert result is True
        assert mock_sdk.call_counts["register_worker"] == 1

    def test_register_worker_fires_event(
        self, client: SpacetimeDBClient,
    ) -> None:
        statuses: list[tuple[str, str]] = []
        client.subscribe_worker_status(
            lambda wid, status: statuses.append((wid, status)),
        )

        client.register_worker_sync("selfplay", "w1")
        assert statuses == [("w1", "running")]

    def test_heartbeat_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        client.register_worker_sync("selfplay", "w1")
        result = client.heartbeat_sync("w1", games_completed=5)
        assert result is True
        assert mock_sdk.call_counts["heartbeat"] == 1

    def test_heartbeat_not_connected(
        self, config: SpacetimeDBConfig,
    ) -> None:
        client = SpacetimeDBClient(config)
        # Heartbeat should not raise, just return False
        with pytest.raises(SpacetimeDBConnectionError):
            client.heartbeat_sync("w1", games_completed=0)


class TestTrainingState:
    """Test training state operations."""

    def test_request_stop_sync(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        result = client.request_stop_sync()
        assert result is True
        assert mock_sdk.call_counts["request_stop"] == 1

    def test_request_stop_fires_event(
        self, client: SpacetimeDBClient,
    ) -> None:
        states_received: list[dict[str, Any]] = []
        client.subscribe_training_state(
            lambda state: states_received.append(state),
        )

        client.request_stop_sync()
        assert len(states_received) == 1
        assert states_received[0]["should_stop"] is True


class TestSubscriptions:
    """Test subscription registration."""

    def test_subscribe_model_versions(self, client: SpacetimeDBClient) -> None:
        callback = MagicMock()
        client.subscribe_model_versions(callback)
        assert client.events.has_handlers(MODEL_VERSION_CHANGED)

    def test_subscribe_trajectories(self, client: SpacetimeDBClient) -> None:
        callback = MagicMock()
        client.subscribe_trajectories(callback)
        assert client.events.has_handlers(TRAJECTORY_ADDED)

    def test_subscribe_metrics(self, client: SpacetimeDBClient) -> None:
        callback = MagicMock()
        client.subscribe_metrics(callback)
        assert client.events.has_handlers(METRICS_UPDATED)

    def test_subscribe_training_state(self, client: SpacetimeDBClient) -> None:
        callback = MagicMock()
        client.subscribe_training_state(callback)
        assert client.events.has_handlers(TRAINING_STATE_CHANGED)

    def test_subscribe_worker_status(self, client: SpacetimeDBClient) -> None:
        callback = MagicMock()
        client.subscribe_worker_status(callback)
        assert client.events.has_handlers(WORKER_STATUS_CHANGED)

    def test_subscribe_connection_state(self, client: SpacetimeDBClient) -> None:
        callback = MagicMock()
        client.subscribe_connection_state(callback)
        assert client.events.has_handlers(CONNECTION_STATE_CHANGED)

    def test_multiple_subscribers(self, client: SpacetimeDBClient) -> None:
        cb1 = MagicMock()
        cb2 = MagicMock()
        client.subscribe_model_versions(cb1)
        client.subscribe_model_versions(cb2)

        client.publish_weights_sync(
            iteration=1, version=1, weights_bytes=b"w", metadata={},
        )
        cb1.assert_called_once_with(1)
        cb2.assert_called_once_with(1)


class TestFailureSimulation:
    """Test behavior when mock simulates failures."""

    def test_publish_weights_failure(self, config: SpacetimeDBConfig) -> None:
        mock = MockSpacetimeDBClient(simulate_failures=True)
        client = SpacetimeDBClient(config)
        client.connect_with_mock(mock)

        result = client.publish_weights_sync(
            iteration=1, version=1,
            weights_bytes=b"w", metadata={},
        )
        assert result is False

    def test_submit_trajectory_failure(self, config: SpacetimeDBConfig) -> None:
        mock = MockSpacetimeDBClient(simulate_failures=True)
        client = SpacetimeDBClient(config)
        client.connect_with_mock(mock)

        result = client.submit_trajectory_sync(
            iteration=1, worker_id=0, game_id="g1",
            trajectory_bytes=b"data", num_examples=5,
        )
        assert result is False

    def test_log_metrics_failure(self, config: SpacetimeDBConfig) -> None:
        mock = MockSpacetimeDBClient(simulate_failures=True)
        client = SpacetimeDBClient(config)
        client.connect_with_mock(mock)

        result = client.log_metrics_sync(step=1, metrics={"loss": 0.1})
        assert result is False


class TestAsyncOperations:
    """Test async versions of operations."""

    @pytest.mark.asyncio
    async def test_publish_weights_async(
        self, client: SpacetimeDBClient,
    ) -> None:
        result = await client.publish_weights(
            iteration=1, version=1,
            weights_bytes=b"w", metadata={},
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_submit_trajectory_async(
        self, client: SpacetimeDBClient,
    ) -> None:
        result = await client.submit_trajectory(
            iteration=1, worker_id=0, game_id="g1",
            trajectory_bytes=b"data", num_examples=5,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_log_metrics_async(
        self, client: SpacetimeDBClient,
    ) -> None:
        result = await client.log_metrics(step=1, metrics={"loss": 0.1})
        assert result is True

    @pytest.mark.asyncio
    async def test_disconnect_async(
        self, client: SpacetimeDBClient,
    ) -> None:
        assert client.is_connected
        await client.disconnect()
        assert not client.is_connected


class TestEndToEndWorkflow:
    """Test complete workflows."""

    def test_full_training_cycle(
        self, client: SpacetimeDBClient, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        """Simulate a complete training iteration cycle."""
        # Register worker
        client.register_worker_sync("selfplay", "worker_0")

        # Submit trajectories
        for i in range(3):
            client.submit_trajectory_sync(
                iteration=1,
                worker_id=0,
                game_id=f"game_{i}",
                trajectory_bytes=f"traj_{i}".encode(),
                num_examples=10,
            )

        # Publish weights
        client.publish_weights_sync(
            iteration=1, version=1,
            weights_bytes=b"new_weights", metadata={"loss": 0.3},
        )

        # Log metrics
        client.log_metrics_sync(step=100, metrics={"total_loss": 0.3})

        # Save evaluation
        client.save_evaluation_sync({
            "iteration": 1, "win_rate": 0.55,
            "games_played": 50, "promoted": True,
        })

        # Heartbeat
        client.heartbeat_sync("worker_0", games_completed=3)

        # Verify all operations tracked
        assert mock_sdk.call_counts["register_worker"] == 1
        assert mock_sdk.call_counts["submit_trajectory"] == 3
        assert mock_sdk.call_counts["publish_weights"] == 1
        assert mock_sdk.call_counts["log_metrics"] == 1
        assert mock_sdk.call_counts["save_evaluation"] == 1
        assert mock_sdk.call_counts["heartbeat"] == 1

    def test_version_subscription_workflow(
        self, client: SpacetimeDBClient,
    ) -> None:
        """Test that version subscriptions work through full workflow."""
        versions: list[int] = []
        client.subscribe_model_versions(lambda v: versions.append(v))

        for v in range(1, 4):
            client.publish_weights_sync(
                iteration=v, version=v,
                weights_bytes=b"w", metadata={},
            )

        assert versions == [1, 2, 3]
        assert client.current_version == 3
