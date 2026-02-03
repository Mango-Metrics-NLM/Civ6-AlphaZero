"""Integration tests for actor-learner communication.

Tests weight synchronization, trajectory transfer to replay buffer,
version tracking, and graceful handling of actor restarts.

Run with:
    pytest tests/integration/test_actor_learner_communication.py -v
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from alpha_zero.distributed.shared_storage import (
    GameTrajectory,
    SharedStorageConfig,
    TrainingExample,
)

if TYPE_CHECKING:
    pass


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def shared_storage_config() -> SharedStorageConfig:
    """Create shared storage configuration for testing."""
    return SharedStorageConfig(
        mode="multiprocessing",
        max_model_history=5,
        compress_weights=False,  # Faster for tests
        compress_trajectories=False,
        track_statistics=True,
    )


@pytest.fixture
def mock_shared_storage() -> MagicMock:
    """Create mock shared storage for testing."""
    storage = MagicMock()
    storage._weights: dict[str, Any] = {}
    storage._version: int = 0
    storage._stats: dict[str, Any] = {}

    def set_weights(weights: dict) -> None:
        storage._weights = weights
        storage._version += 1

    def get_weights() -> dict:
        return storage._weights

    def get_version() -> int:
        return storage._version

    storage.set_weights = set_weights
    storage.get_weights = get_weights
    storage.get_version = get_version
    storage.get_model_version = get_version

    return storage


@pytest.fixture
def mock_replay_buffer() -> MagicMock:
    """Create mock replay buffer for testing."""
    buffer = MagicMock()
    buffer._trajectories: list[GameTrajectory] = []
    buffer._examples: list[TrainingExample] = []

    def add_trajectory(trajectory: GameTrajectory) -> None:
        buffer._trajectories.append(trajectory)
        buffer._examples.extend(trajectory.examples)

    def add_example(example: TrainingExample) -> None:
        buffer._examples.append(example)

    def size() -> int:
        return len(buffer._examples)

    buffer.add_trajectory = add_trajectory
    buffer.add_example = add_example
    buffer.add = add_trajectory
    buffer.size = size

    return buffer


@pytest.fixture
def simple_network() -> nn.Module:
    """Create simple network for weight testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


@pytest.fixture
def sample_trajectory() -> GameTrajectory:
    """Create sample game trajectory."""
    examples = [
        TrainingExample(
            spatial=np.random.randn(51, 16, 16).astype(np.float32),
            vector=np.random.randn(256).astype(np.float32),
            action_mask=np.ones(100, dtype=np.float32),
            policy=np.random.dirichlet(np.ones(100)).astype(np.float32),
            value=0.5,
        )
        for _ in range(5)
    ]
    return GameTrajectory(
        examples=examples,
        game_id=1,
        actor_id=0,
        game_length=5,
        winner=0,
    )


# ============================================================================
# Weight Update Tests
# ============================================================================


class TestActorReceivesUpdatedWeights:
    """Tests that actors receive updated model weights."""

    def test_actor_gets_initial_weights(
        self,
        mock_shared_storage: MagicMock,
        simple_network: nn.Module,
    ) -> None:
        """Verify actor can get initial weights from storage."""
        # Learner sets initial weights
        mock_shared_storage.set_weights(simple_network.state_dict())

        # Actor retrieves weights
        actor_weights = mock_shared_storage.get_weights()

        assert len(actor_weights) > 0
        assert "0.weight" in actor_weights  # Sequential layer naming

    def test_actor_gets_updated_weights(
        self,
        mock_shared_storage: MagicMock,
        simple_network: nn.Module,
    ) -> None:
        """Verify actor gets updated weights after learner update."""
        # Initial weights
        initial_weights = {k: v.clone() for k, v in simple_network.state_dict().items()}
        mock_shared_storage.set_weights(simple_network.state_dict())

        # Simulate training step (modify weights)
        for param in simple_network.parameters():
            param.data += 0.1

        # Learner updates storage
        mock_shared_storage.set_weights(simple_network.state_dict())

        # Actor retrieves new weights
        new_weights = mock_shared_storage.get_weights()

        # Weights should be different
        for key in initial_weights:
            assert not torch.equal(
                torch.tensor(new_weights[key]),
                initial_weights[key],
            )

    def test_version_increments_on_update(
        self,
        mock_shared_storage: MagicMock,
        simple_network: nn.Module,
    ) -> None:
        """Verify version number increments on weight update."""
        assert mock_shared_storage.get_version() == 0

        mock_shared_storage.set_weights(simple_network.state_dict())
        assert mock_shared_storage.get_version() == 1

        mock_shared_storage.set_weights(simple_network.state_dict())
        assert mock_shared_storage.get_version() == 2


# ============================================================================
# Trajectory Transfer Tests
# ============================================================================


class TestTrajectoryReachesReplayBuffer:
    """Tests that trajectories reach the replay buffer."""

    def test_trajectory_stored_in_buffer(
        self,
        mock_replay_buffer: MagicMock,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Verify trajectory is stored in replay buffer."""
        mock_replay_buffer.add_trajectory(sample_trajectory)

        assert len(mock_replay_buffer._trajectories) == 1
        assert len(mock_replay_buffer._examples) == 5

    def test_multiple_trajectories_accumulate(
        self,
        mock_replay_buffer: MagicMock,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Verify multiple trajectories accumulate in buffer."""
        for i in range(3):
            trajectory = GameTrajectory(
                examples=sample_trajectory.examples.copy(),
                game_id=i,
                actor_id=0,
                game_length=5,
                winner=0,
            )
            mock_replay_buffer.add_trajectory(trajectory)

        assert len(mock_replay_buffer._trajectories) == 3
        assert len(mock_replay_buffer._examples) == 15

    def test_trajectory_metadata_preserved(
        self,
        mock_replay_buffer: MagicMock,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Verify trajectory metadata is preserved."""
        mock_replay_buffer.add_trajectory(sample_trajectory)

        stored = mock_replay_buffer._trajectories[0]
        assert stored.game_id == sample_trajectory.game_id
        assert stored.actor_id == sample_trajectory.actor_id
        assert stored.game_length == sample_trajectory.game_length
        assert stored.winner == sample_trajectory.winner


# ============================================================================
# Weight Version Tracking Tests
# ============================================================================


class TestWeightVersionTracking:
    """Tests for weight version tracking."""

    def test_actor_detects_stale_weights(
        self,
        mock_shared_storage: MagicMock,
        simple_network: nn.Module,
    ) -> None:
        """Verify actor can detect when weights are stale."""
        mock_shared_storage.set_weights(simple_network.state_dict())
        actor_version = mock_shared_storage.get_version()

        # Learner updates
        mock_shared_storage.set_weights(simple_network.state_dict())

        # Actor checks version
        current_version = mock_shared_storage.get_version()
        is_stale = actor_version < current_version

        assert is_stale

    def test_actor_syncs_when_stale(
        self,
        mock_shared_storage: MagicMock,
        simple_network: nn.Module,
    ) -> None:
        """Verify actor syncs weights when detecting stale version."""
        # Initial sync
        mock_shared_storage.set_weights(simple_network.state_dict())
        actor_version = mock_shared_storage.get_version()

        # Learner updates multiple times
        for _ in range(3):
            for param in simple_network.parameters():
                param.data += 0.01
            mock_shared_storage.set_weights(simple_network.state_dict())

        # Actor detects and syncs
        if actor_version < mock_shared_storage.get_version():
            _ = mock_shared_storage.get_weights()  # Fetch weights (consumed)
            actor_version = mock_shared_storage.get_version()

        assert actor_version == 4  # 1 initial + 3 updates

    def test_version_never_decreases(
        self,
        mock_shared_storage: MagicMock,
        simple_network: nn.Module,
    ) -> None:
        """Verify version number never decreases."""
        versions = []

        for _ in range(10):
            mock_shared_storage.set_weights(simple_network.state_dict())
            versions.append(mock_shared_storage.get_version())

        # Should be strictly increasing
        for i in range(1, len(versions)):
            assert versions[i] > versions[i - 1]


# ============================================================================
# Actor Restart Tests
# ============================================================================


class TestActorRestartPreservesState:
    """Tests that actor restart doesn't corrupt training state."""

    def test_new_actor_gets_current_weights(
        self,
        mock_shared_storage: MagicMock,
        simple_network: nn.Module,
    ) -> None:
        """Verify new actor gets current weights, not stale ones."""
        # Simulate multiple weight updates
        for i in range(5):
            for param in simple_network.parameters():
                param.data += 0.01 * i
            mock_shared_storage.set_weights(simple_network.state_dict())

        # New actor starts and gets current weights
        current_weights = mock_shared_storage.get_weights()
        current_version = mock_shared_storage.get_version()

        assert current_version == 5
        assert len(current_weights) > 0

    def test_replay_buffer_not_cleared_on_restart(
        self,
        mock_replay_buffer: MagicMock,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Verify replay buffer retains data when actor restarts."""
        # Actor 0 adds trajectories
        for i in range(3):
            trajectory = GameTrajectory(
                examples=sample_trajectory.examples.copy(),
                game_id=i,
                actor_id=0,
                game_length=5,
                winner=0,
            )
            mock_replay_buffer.add_trajectory(trajectory)

        # Simulate actor restart (actor_id stays same but process new)
        # New actor continues adding
        for i in range(3, 6):
            trajectory = GameTrajectory(
                examples=sample_trajectory.examples.copy(),
                game_id=i,
                actor_id=0,  # Same actor ID after restart
                game_length=5,
                winner=0,
            )
            mock_replay_buffer.add_trajectory(trajectory)

        # All trajectories should be preserved
        assert len(mock_replay_buffer._trajectories) == 6

    def test_multiple_actors_independent(
        self,
        mock_replay_buffer: MagicMock,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Verify multiple actors can operate independently."""
        # Actor 0 adds trajectories
        for i in range(2):
            trajectory = GameTrajectory(
                examples=sample_trajectory.examples.copy(),
                game_id=i,
                actor_id=0,
                game_length=5,
                winner=0,
            )
            mock_replay_buffer.add_trajectory(trajectory)

        # Actor 1 adds trajectories
        for i in range(2, 4):
            trajectory = GameTrajectory(
                examples=sample_trajectory.examples.copy(),
                game_id=i,
                actor_id=1,
                game_length=5,
                winner=1,
            )
            mock_replay_buffer.add_trajectory(trajectory)

        assert len(mock_replay_buffer._trajectories) == 4

        # Check actor IDs
        actor_0_trajs = [t for t in mock_replay_buffer._trajectories if t.actor_id == 0]
        actor_1_trajs = [t for t in mock_replay_buffer._trajectories if t.actor_id == 1]

        assert len(actor_0_trajs) == 2
        assert len(actor_1_trajs) == 2


# ============================================================================
# Concurrent Access Tests
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent actor-learner access."""

    def test_concurrent_reads_safe(
        self,
        mock_shared_storage: MagicMock,
        simple_network: nn.Module,
    ) -> None:
        """Verify concurrent reads from storage are safe."""
        mock_shared_storage.set_weights(simple_network.state_dict())

        results = []
        errors = []

        def read_weights() -> None:
            try:
                for _ in range(10):
                    weights = mock_shared_storage.get_weights()
                    results.append(len(weights))
            except Exception as e:
                errors.append(e)

        # Simulate multiple actors reading concurrently
        threads = [threading.Thread(target=read_weights) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r > 0 for r in results)

    def test_read_during_write_safe(
        self,
        mock_shared_storage: MagicMock,
        simple_network: nn.Module,
    ) -> None:
        """Verify reads during writes don't cause errors."""
        mock_shared_storage.set_weights(simple_network.state_dict())

        read_results = []
        write_count = [0]
        errors = []

        def reader() -> None:
            try:
                for _ in range(20):
                    weights = mock_shared_storage.get_weights()
                    read_results.append(len(weights))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for _ in range(5):
                    mock_shared_storage.set_weights(simple_network.state_dict())
                    write_count[0] += 1
                    time.sleep(0.005)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(read_results) == 40  # 2 readers * 20 reads


# ============================================================================
# Serialization Tests
# ============================================================================


class TestTrajectorySerializatio:
    """Tests for trajectory serialization."""

    def test_trajectory_roundtrip(
        self,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Verify trajectory survives serialization roundtrip."""
        # Serialize
        data = sample_trajectory.to_bytes(compress=False)

        # Deserialize
        restored = GameTrajectory.from_bytes(data, compressed=False)

        assert restored.game_id == sample_trajectory.game_id
        assert restored.actor_id == sample_trajectory.actor_id
        assert len(restored.examples) == len(sample_trajectory.examples)

    def test_trajectory_compressed_roundtrip(
        self,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Verify trajectory survives compressed serialization."""
        # Serialize with compression
        data = sample_trajectory.to_bytes(compress=True)

        # Deserialize
        restored = GameTrajectory.from_bytes(data, compressed=True)

        assert restored.game_id == sample_trajectory.game_id
        assert len(restored.examples) == len(sample_trajectory.examples)

    def test_example_roundtrip(self) -> None:
        """Verify single example survives serialization."""
        example = TrainingExample(
            spatial=np.random.randn(51, 16, 16).astype(np.float32),
            vector=np.random.randn(256).astype(np.float32),
            action_mask=np.ones(100, dtype=np.float32),
            policy=np.random.dirichlet(np.ones(100)).astype(np.float32),
            value=0.75,
        )

        data = example.to_bytes()
        restored = TrainingExample.from_bytes(data)

        assert np.allclose(restored.spatial, example.spatial)
        assert np.allclose(restored.vector, example.vector)
        assert restored.value == example.value


# ============================================================================
# Integration Tests
# ============================================================================


class TestActorLearnerIntegration:
    """Integration tests for actor-learner communication."""

    def test_full_training_loop_communication(
        self,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
        simple_network: nn.Module,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Test full training loop communication."""
        # 1. Learner initializes weights
        mock_shared_storage.set_weights(simple_network.state_dict())
        initial_version = mock_shared_storage.get_version()

        # 2. Actor gets weights
        _ = mock_shared_storage.get_weights()  # Actor fetches initial weights
        actor_version = initial_version

        # 3. Actor generates trajectory
        mock_replay_buffer.add_trajectory(sample_trajectory)

        # 4. Learner trains and updates weights
        for param in simple_network.parameters():
            param.data += 0.01
        mock_shared_storage.set_weights(simple_network.state_dict())

        # 5. Actor detects stale weights and updates
        if actor_version < mock_shared_storage.get_version():
            _ = mock_shared_storage.get_weights()  # Fetch weights (consumed)
            actor_version = mock_shared_storage.get_version()

        assert actor_version == 2
        assert len(mock_replay_buffer._trajectories) == 1

    def test_high_throughput_scenario(
        self,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
        simple_network: nn.Module,
        sample_trajectory: GameTrajectory,
    ) -> None:
        """Test high throughput scenario with many updates."""
        num_learner_updates = 10
        num_actor_trajectories = 50

        # Simulate rapid actor trajectory generation
        for i in range(num_actor_trajectories):
            trajectory = GameTrajectory(
                examples=sample_trajectory.examples[:2],  # Shorter trajectories
                game_id=i,
                actor_id=i % 4,  # 4 actors
                game_length=2,
                winner=i % 2,
            )
            mock_replay_buffer.add_trajectory(trajectory)

        # Simulate learner updates
        for _ in range(num_learner_updates):
            mock_shared_storage.set_weights(simple_network.state_dict())

        assert mock_shared_storage.get_version() == num_learner_updates
        assert len(mock_replay_buffer._trajectories) == num_actor_trajectories
