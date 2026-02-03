"""Unit tests for parameter server module."""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn

from alpha_zero.distributed.parameter_server import (
    DistributedParameterServer,
    MockParameterServer,
    ParameterServer,
    ParameterServerConfig,
    WeightUpdateMode,
    create_parameter_server,
)


class SimpleNetwork(nn.Module):
    """Simple network for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestWeightUpdateMode:
    """Tests for WeightUpdateMode enum."""

    def test_mode_values(self) -> None:
        """Test mode enum values."""
        assert WeightUpdateMode.REPLACE.value == "replace"
        assert WeightUpdateMode.GRADIENT.value == "gradient"
        assert WeightUpdateMode.DELTA.value == "delta"


class TestParameterServerConfig:
    """Tests for ParameterServerConfig."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = ParameterServerConfig()

        assert config.update_mode == WeightUpdateMode.REPLACE.value
        assert config.compress_weights is True
        assert config.max_version_history == 5


class TestParameterServer:
    """Tests for ParameterServer."""

    @pytest.fixture
    def network(self) -> SimpleNetwork:
        """Create test network."""
        return SimpleNetwork()

    @pytest.fixture
    def server(self) -> ParameterServer:
        """Create parameter server."""
        return ParameterServer()

    @pytest.fixture
    def server_with_network(self, network: SimpleNetwork) -> ParameterServer:
        """Create parameter server initialized with network."""
        return ParameterServer(network=network)

    def test_init_empty(self, server: ParameterServer) -> None:
        """Test initialization without network."""
        assert server.get_version() == 0
        assert server.get_weights() is None

    def test_init_with_network(
        self,
        server_with_network: ParameterServer,
        network: SimpleNetwork,
    ) -> None:
        """Test initialization with network."""
        assert server_with_network.get_version() == 1
        weights = server_with_network.get_weights()
        assert weights is not None
        assert "fc1.weight" in weights

    def test_set_weights(
        self,
        server: ParameterServer,
        network: SimpleNetwork,
    ) -> None:
        """Test setting weights."""
        version = server.set_weights(network.state_dict())

        assert version == 1
        assert server.get_version() == 1

        weights = server.get_weights()
        assert weights is not None
        assert "fc1.weight" in weights

    def test_set_weights_increment_version(
        self,
        server: ParameterServer,
        network: SimpleNetwork,
    ) -> None:
        """Test version incrementing on weight updates."""
        server.set_weights(network.state_dict())
        assert server.get_version() == 1

        server.set_weights(network.state_dict())
        assert server.get_version() == 2

    def test_set_weights_no_increment(
        self,
        server: ParameterServer,
        network: SimpleNetwork,
    ) -> None:
        """Test setting weights without incrementing version."""
        server.set_weights(network.state_dict())
        assert server.get_version() == 1

        server.set_weights(network.state_dict(), increment_version=False)
        assert server.get_version() == 1

    def test_get_weights_with_version(
        self,
        server_with_network: ParameterServer,
    ) -> None:
        """Test atomic get of weights and version."""
        weights, version = server_with_network.get_weights_with_version()

        assert weights is not None
        assert version == 1

    def test_get_serialized_weights(
        self,
        server_with_network: ParameterServer,
    ) -> None:
        """Test getting serialized weights."""
        data, version = server_with_network.get_serialized_weights()

        assert data is not None
        assert isinstance(data, bytes)
        assert version == 1

    def test_deserialize_weights(
        self,
        server_with_network: ParameterServer,
    ) -> None:
        """Test weight deserialization."""
        data, _ = server_with_network.get_serialized_weights()
        assert data is not None

        weights = server_with_network.deserialize_weights(data)

        assert "fc1.weight" in weights

    def test_get_weights_at_version(
        self,
        server: ParameterServer,
        network: SimpleNetwork,
    ) -> None:
        """Test getting weights at specific version."""
        # Store multiple versions
        server.set_weights(network.state_dict())  # v1
        server.set_weights(network.state_dict())  # v2
        server.set_weights(network.state_dict())  # v3

        # Get specific version
        weights = server.get_weights_at_version(2)
        assert weights is not None

        # Non-existent version
        weights = server.get_weights_at_version(10)
        assert weights is None

    def test_update_model(
        self,
        server_with_network: ParameterServer,
    ) -> None:
        """Test updating a model with server weights."""
        new_network = SimpleNetwork()

        # Update model
        updated, version = server_with_network.update_model(new_network)

        assert updated is True
        assert version == 1

    def test_update_model_min_version(
        self,
        server_with_network: ParameterServer,
    ) -> None:
        """Test update model with minimum version check."""
        new_network = SimpleNetwork()

        # Don't update if version <= min_version
        updated, version = server_with_network.update_model(
            new_network,
            min_version=1,
        )

        assert updated is False

    def test_apply_gradients(
        self,
        server_with_network: ParameterServer,
    ) -> None:
        """Test applying gradients to weights."""
        # Get current weights
        weights = server_with_network.get_weights()
        assert weights is not None

        # Create mock gradients
        gradients = {k: torch.ones_like(v) for k, v in weights.items()}

        # Apply gradients
        new_version = server_with_network.apply_gradients(
            gradients,
            learning_rate=0.01,
        )

        assert new_version == 2

    def test_apply_delta(
        self,
        server_with_network: ParameterServer,
    ) -> None:
        """Test applying weight delta."""
        initial_version = server_with_network.get_version()
        weights = server_with_network.get_weights()
        assert weights is not None

        # Create delta
        delta = {k: torch.ones_like(v) * 0.01 for k, v in weights.items()}

        # Apply delta with correct source version
        new_version = server_with_network.apply_delta(delta, initial_version)

        assert new_version == 2

    def test_apply_delta_version_mismatch(
        self,
        server_with_network: ParameterServer,
    ) -> None:
        """Test delta rejection on version mismatch."""
        weights = server_with_network.get_weights()
        assert weights is not None

        delta = {k: torch.ones_like(v) for k, v in weights.items()}

        # Apply with wrong source version
        new_version = server_with_network.apply_delta(delta, source_version=0)

        assert new_version is None

    def test_get_statistics(
        self,
        server_with_network: ParameterServer,
    ) -> None:
        """Test getting server statistics."""
        stats = server_with_network.get_statistics()

        assert stats["version"] == 1
        assert stats["total_updates"] == 1
        assert "compression_ratio" in stats

    def test_reset(
        self,
        server_with_network: ParameterServer,
    ) -> None:
        """Test server reset."""
        assert server_with_network.get_version() == 1

        server_with_network.reset()

        assert server_with_network.get_version() == 0
        assert server_with_network.get_weights() is None


class TestMockParameterServer:
    """Tests for MockParameterServer."""

    @pytest.fixture
    def mock_server(self) -> MockParameterServer:
        """Create mock parameter server."""
        return MockParameterServer()

    def test_simulate_latency(
        self,
        mock_server: MockParameterServer,
    ) -> None:
        """Test simulated network latency."""
        mock_server.simulate_latency(0.01)

        start = time.time()
        mock_server.set_weights({"test": torch.tensor([1.0])})
        elapsed = time.time() - start

        assert elapsed >= 0.01

    def test_fail_next_update(
        self,
        mock_server: MockParameterServer,
    ) -> None:
        """Test simulated update failure."""
        mock_server.fail_next_update()

        with pytest.raises(RuntimeError, match="Simulated update failure"):
            mock_server.set_weights({"test": torch.tensor([1.0])})

    def test_fail_next_fetch(
        self,
        mock_server: MockParameterServer,
    ) -> None:
        """Test simulated fetch failure."""
        mock_server.set_weights({"test": torch.tensor([1.0])})
        mock_server.fail_next_fetch()

        with pytest.raises(RuntimeError, match="Simulated fetch failure"):
            mock_server.get_weights()

    def test_update_history(
        self,
        mock_server: MockParameterServer,
    ) -> None:
        """Test update history tracking."""
        mock_server.set_weights({"a": torch.tensor([1.0])})
        mock_server.set_weights({"b": torch.tensor([2.0])})

        history = mock_server.get_update_history()

        assert len(history) == 2
        assert history[0]["version"] == 1
        assert history[1]["version"] == 2

    def test_clear_history(
        self,
        mock_server: MockParameterServer,
    ) -> None:
        """Test clearing update history."""
        mock_server.set_weights({"test": torch.tensor([1.0])})
        mock_server.clear_history()

        assert len(mock_server.get_update_history()) == 0


class TestDistributedParameterServer:
    """Tests for DistributedParameterServer."""

    @pytest.fixture
    def dist_server(self) -> DistributedParameterServer:
        """Create distributed parameter server."""
        return DistributedParameterServer(node_id="ps-test")

    def test_register_client(
        self,
        dist_server: DistributedParameterServer,
    ) -> None:
        """Test client registration."""
        result = dist_server.register_client("client-1", "learner")

        assert result is True
        assert dist_server.get_client_count() == 1
        assert dist_server.get_client_count("learner") == 1

    def test_unregister_client(
        self,
        dist_server: DistributedParameterServer,
    ) -> None:
        """Test client unregistration."""
        dist_server.register_client("client-1", "learner")
        result = dist_server.unregister_client("client-1")

        assert result is True
        assert dist_server.get_client_count() == 0

    def test_unregister_nonexistent(
        self,
        dist_server: DistributedParameterServer,
    ) -> None:
        """Test unregistering non-existent client."""
        result = dist_server.unregister_client("nonexistent")

        assert result is False

    def test_heartbeat(
        self,
        dist_server: DistributedParameterServer,
    ) -> None:
        """Test client heartbeat."""
        dist_server.register_client("client-1", "actor")

        result = dist_server.heartbeat("client-1")

        assert result is True

    def test_heartbeat_unregistered(
        self,
        dist_server: DistributedParameterServer,
    ) -> None:
        """Test heartbeat from unregistered client."""
        result = dist_server.heartbeat("unknown")

        assert result is False

    def test_get_stale_clients(
        self,
        dist_server: DistributedParameterServer,
    ) -> None:
        """Test stale client detection."""
        dist_server.register_client("client-1", "actor")

        # Small delay to ensure heartbeat is old enough
        time.sleep(0.02)

        # With a very small timeout, client should be stale
        stale = dist_server.get_stale_clients(timeout_seconds=0.01)

        assert "client-1" in stale

    def test_get_cluster_info(
        self,
        dist_server: DistributedParameterServer,
    ) -> None:
        """Test getting cluster information."""
        dist_server.register_client("learner-1", "learner")
        dist_server.register_client("actor-1", "actor")
        dist_server.register_client("actor-2", "actor")

        info = dist_server.get_cluster_info()

        assert info["node_id"] == "ps-test"
        assert info["clients"]["total"] == 3
        assert info["clients"]["learners"] == 1
        assert info["clients"]["actors"] == 2

    def test_broadcast_weights(
        self,
        dist_server: DistributedParameterServer,
    ) -> None:
        """Test weight broadcasting."""
        dist_server.register_client("client-1", "actor")
        dist_server.register_client("client-2", "actor")

        results = dist_server.broadcast_weights()

        assert "client-1" in results
        assert "client-2" in results


class TestCreateParameterServer:
    """Tests for factory function."""

    def test_create_default(self) -> None:
        """Test creating default parameter server."""
        server = create_parameter_server()

        assert isinstance(server, ParameterServer)

    def test_create_mock(self) -> None:
        """Test creating mock parameter server."""
        server = create_parameter_server(force_mock=True)

        assert isinstance(server, MockParameterServer)

    def test_create_distributed(self) -> None:
        """Test creating distributed parameter server."""
        server = create_parameter_server(distributed=True, node_id="test-node")

        assert isinstance(server, DistributedParameterServer)

    def test_create_with_network(self) -> None:
        """Test creating with network initialization."""
        network = SimpleNetwork()
        server = create_parameter_server(network=network)

        assert server.get_version() == 1
        assert server.get_weights() is not None

    def test_create_with_config(self) -> None:
        """Test creating with custom config."""
        config = ParameterServerConfig(
            compress_weights=False,
            max_version_history=10,
        )

        server = create_parameter_server(config=config)

        assert server.config.compress_weights is False
        assert server.config.max_version_history == 10
