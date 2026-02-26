"""Unit tests for SpacetimeDB weight synchronizer."""

from __future__ import annotations

import torch
import torch.nn as nn

import pytest

from alpha_zero.spacetimedb.client import SpacetimeDBClient
from alpha_zero.spacetimedb.config import SpacetimeDBConfig
from alpha_zero.spacetimedb.serialization import serialize_state_dict
from alpha_zero.spacetimedb.weight_sync import SpacetimeDBWeightSynchronizer
from tests.mocks.spacetimedb_mocks import MockSpacetimeDBClient


class SimpleNet(nn.Module):
    """Simple network for testing weight sync."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@pytest.fixture
def config() -> SpacetimeDBConfig:
    """Create test config."""
    return SpacetimeDBConfig(
        enabled=True,
        host="http://localhost:3000",
        module_address="test",
        connection_timeout=1.0,
        max_weight_blob_mb=10,
    )


@pytest.fixture
def mock_sdk() -> MockSpacetimeDBClient:
    """Create mock SDK client."""
    return MockSpacetimeDBClient()


@pytest.fixture
def client(config: SpacetimeDBConfig, mock_sdk: MockSpacetimeDBClient) -> SpacetimeDBClient:
    """Create connected client."""
    c = SpacetimeDBClient(config)
    c.connect_with_mock(mock_sdk)
    return c


@pytest.fixture
def model() -> SimpleNet:
    """Create simple network."""
    return SimpleNet()


@pytest.fixture
def sync(client: SpacetimeDBClient) -> SpacetimeDBWeightSynchronizer:
    """Create weight synchronizer."""
    return SpacetimeDBWeightSynchronizer(
        client, update_interval=0.01, compress=False,
    )


class TestSynchronizerInit:
    """Test synchronizer initialization."""

    def test_initial_version(self, sync: SpacetimeDBWeightSynchronizer) -> None:
        assert sync.current_version == 0

    def test_initial_latest_version(self, sync: SpacetimeDBWeightSynchronizer) -> None:
        assert sync.latest_available_version == 0

    def test_get_current_version(self, sync: SpacetimeDBWeightSynchronizer) -> None:
        assert sync.get_current_version() == 0


class TestSubscriptionNotification:
    """Test push-based version notifications."""

    def test_version_notification(
        self, sync: SpacetimeDBWeightSynchronizer, client: SpacetimeDBClient,
    ) -> None:
        """Publishing weights triggers version notification via subscription."""
        client.publish_weights_sync(
            iteration=1, version=1, weights_bytes=b"w", metadata={},
        )
        assert sync.latest_available_version == 1
        assert sync.should_update()

    def test_multiple_versions(
        self, sync: SpacetimeDBWeightSynchronizer, client: SpacetimeDBClient,
    ) -> None:
        """Multiple publishes update to highest version."""
        for v in range(1, 4):
            client.publish_weights_sync(
                iteration=v, version=v, weights_bytes=b"w", metadata={},
            )
        assert sync.latest_available_version == 3

    def test_older_version_ignored(
        self, sync: SpacetimeDBWeightSynchronizer,
    ) -> None:
        """Older version notifications don't regress."""
        sync._on_version_changed(5)
        sync._on_version_changed(3)  # Should be ignored
        assert sync.latest_available_version == 5


class TestShouldUpdate:
    """Test should_update logic."""

    def test_no_update_initially(
        self, sync: SpacetimeDBWeightSynchronizer,
    ) -> None:
        assert not sync.should_update()

    def test_update_after_notification(
        self, sync: SpacetimeDBWeightSynchronizer,
    ) -> None:
        sync._on_version_changed(1)
        assert sync.should_update()

    def test_polling_fallback(
        self, sync: SpacetimeDBWeightSynchronizer, client: SpacetimeDBClient,
        mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        """When server version changes without subscription, polling detects it."""
        # Simulate a version published by another process on the server
        mock_sdk._current_version = 5
        # Wait for interval to pass
        sync._last_check_time = 0.0
        assert sync.should_update()


class TestUpdateModel:
    """Test model weight updating."""

    def test_update_model_with_weights(
        self, sync: SpacetimeDBWeightSynchronizer,
        client: SpacetimeDBClient,
        mock_sdk: MockSpacetimeDBClient,
        model: SimpleNet,
    ) -> None:
        """Update model with published weights."""
        # Save the current model weights
        original_weights = model.linear.weight.clone()

        # Publish different weights
        new_state = model.state_dict()
        new_state["linear.weight"] = torch.randn_like(new_state["linear.weight"])
        weights_bytes = serialize_state_dict(new_state, compress=False)

        mock_sdk.publish_weights_sync(
            iteration=1, version=1,
            weights_bytes=weights_bytes,
            metadata={"version": 1},
        )

        # Update
        updated = sync.update_model(model)
        assert updated is True
        assert sync.current_version == 1
        assert not torch.equal(model.linear.weight, original_weights)

    def test_update_model_no_weights(
        self, sync: SpacetimeDBWeightSynchronizer, model: SimpleNet,
    ) -> None:
        """Update returns False when no weights available."""
        updated = sync.update_model(model)
        assert updated is False

    def test_update_clears_pending(
        self, sync: SpacetimeDBWeightSynchronizer,
        client: SpacetimeDBClient,
        mock_sdk: MockSpacetimeDBClient,
        model: SimpleNet,
    ) -> None:
        """After update, pending flag is cleared."""
        state = model.state_dict()
        weights_bytes = serialize_state_dict(state, compress=False)
        mock_sdk.publish_weights_sync(
            iteration=1, version=1,
            weights_bytes=weights_bytes,
            metadata={"version": 1},
        )

        sync._on_version_changed(1)
        assert sync.should_update()

        sync.update_model(model)
        assert not sync._pending_update


class TestUpdateIfAvailable:
    """Test the combined check-and-update method."""

    def test_no_update_when_current(
        self, sync: SpacetimeDBWeightSynchronizer, model: SimpleNet,
    ) -> None:
        updated, version = sync.update_if_available(model)
        assert updated is False
        assert version == 0

    def test_update_when_available(
        self, sync: SpacetimeDBWeightSynchronizer,
        client: SpacetimeDBClient,
        mock_sdk: MockSpacetimeDBClient,
        model: SimpleNet,
    ) -> None:
        state = model.state_dict()
        weights_bytes = serialize_state_dict(state, compress=False)
        mock_sdk.publish_weights_sync(
            iteration=1, version=1,
            weights_bytes=weights_bytes,
            metadata={"version": 1},
        )

        # Trigger notification
        sync._on_version_changed(1)

        updated, version = sync.update_if_available(model)
        assert updated is True
        assert version == 1

    def test_sequential_updates(
        self, sync: SpacetimeDBWeightSynchronizer,
        client: SpacetimeDBClient,
        mock_sdk: MockSpacetimeDBClient,
        model: SimpleNet,
    ) -> None:
        """Multiple sequential updates work correctly."""
        for v in range(1, 4):
            state = model.state_dict()
            state["linear.weight"] = torch.randn_like(state["linear.weight"])
            weights_bytes = serialize_state_dict(state, compress=False)
            mock_sdk.publish_weights_sync(
                iteration=v, version=v,
                weights_bytes=weights_bytes,
                metadata={"version": v},
            )

        # Process latest
        sync._on_version_changed(3)
        updated, version = sync.update_if_available(model)
        assert updated is True
        assert version == 3
