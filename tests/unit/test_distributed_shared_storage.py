"""Tests for distributed shared storage components.

Tests cover:
- TrainingExample serialization/deserialization
- GameTrajectory serialization/deserialization
- SharedStorageConfig defaults and validation
- SharedStorage model weight management
- ReplayBufferConfig defaults
- DistributedReplayBuffer submission and sampling
- WeightSynchronizer update logic
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    GameTrajectory,
    ReplayBufferConfig,
    SharedStorage,
    SharedStorageConfig,
    TrainingExample,
    WeightSynchronizer,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_spatial() -> np.ndarray:
    """Sample spatial observation (13 channels, 16x16)."""
    return np.random.rand(13, 16, 16).astype(np.float32)


@pytest.fixture
def sample_vector() -> np.ndarray:
    """Sample vector observation."""
    return np.random.rand(64).astype(np.float32)


@pytest.fixture
def sample_action_mask() -> np.ndarray:
    """Sample action mask (381 actions for MicroCiv)."""
    mask = np.zeros(381, dtype=np.float32)
    mask[:50] = 1.0  # First 50 actions are valid
    return mask


@pytest.fixture
def sample_policy() -> np.ndarray:
    """Sample policy distribution."""
    policy = np.random.rand(381).astype(np.float32)
    policy = policy / policy.sum()  # Normalize
    return policy


@pytest.fixture
def training_example(
    sample_spatial: np.ndarray,
    sample_vector: np.ndarray,
    sample_action_mask: np.ndarray,
    sample_policy: np.ndarray,
) -> TrainingExample:
    """Create a sample training example."""
    return TrainingExample(
        spatial=sample_spatial,
        vector=sample_vector,
        action_mask=sample_action_mask,
        policy=sample_policy,
        value=0.5,
    )


@pytest.fixture
def game_trajectory(training_example: TrainingExample) -> GameTrajectory:
    """Create a sample game trajectory with multiple examples."""
    examples = [training_example for _ in range(10)]
    return GameTrajectory(
        examples=examples,
        game_id=1,
        actor_id=0,
        game_length=10,
        winner=1,
    )


@pytest.fixture
def simple_network() -> torch.nn.Module:
    """Simple network for weight tests."""
    return torch.nn.Linear(10, 5)


# =============================================================================
# TrainingExample Tests
# =============================================================================


class TestTrainingExample:
    """Tests for TrainingExample dataclass."""

    def test_creation(self, training_example: TrainingExample) -> None:
        """Test TrainingExample can be created with valid data."""
        assert training_example.spatial.shape == (13, 16, 16)
        assert training_example.vector.shape == (64,)
        assert training_example.action_mask.shape == (381,)
        assert training_example.policy.shape == (381,)
        assert training_example.value == 0.5

    def test_to_bytes_uncompressed(self, training_example: TrainingExample) -> None:
        """Test serialization to bytes without compression."""
        data = training_example.to_bytes(compress=False)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_to_bytes_compressed(self, training_example: TrainingExample) -> None:
        """Test serialization to bytes with compression."""
        data_compressed = training_example.to_bytes(compress=True)

        assert isinstance(data_compressed, bytes)
        # Compressed should typically be smaller for larger data
        # But for small random data, might not always be true
        assert len(data_compressed) > 0

    def test_from_bytes_uncompressed(self, training_example: TrainingExample) -> None:
        """Test deserialization from bytes without compression."""
        data = training_example.to_bytes(compress=False)
        restored = TrainingExample.from_bytes(data, compressed=False)

        np.testing.assert_array_almost_equal(restored.spatial, training_example.spatial)
        np.testing.assert_array_almost_equal(restored.vector, training_example.vector)
        np.testing.assert_array_almost_equal(restored.action_mask, training_example.action_mask)
        np.testing.assert_array_almost_equal(restored.policy, training_example.policy)
        assert restored.value == training_example.value

    def test_from_bytes_compressed(self, training_example: TrainingExample) -> None:
        """Test deserialization from bytes with compression."""
        data = training_example.to_bytes(compress=True)
        restored = TrainingExample.from_bytes(data, compressed=True)

        np.testing.assert_array_almost_equal(restored.spatial, training_example.spatial)
        np.testing.assert_array_almost_equal(restored.vector, training_example.vector)
        assert restored.value == training_example.value

    def test_roundtrip_preserves_dtypes(self, training_example: TrainingExample) -> None:
        """Test that serialization preserves numpy dtypes."""
        data = training_example.to_bytes(compress=False)
        restored = TrainingExample.from_bytes(data, compressed=False)

        assert restored.spatial.dtype == training_example.spatial.dtype
        assert restored.vector.dtype == training_example.vector.dtype


# =============================================================================
# GameTrajectory Tests
# =============================================================================


class TestGameTrajectory:
    """Tests for GameTrajectory dataclass."""

    def test_creation(self, game_trajectory: GameTrajectory) -> None:
        """Test GameTrajectory can be created."""
        assert len(game_trajectory.examples) == 10
        assert game_trajectory.game_id == 1
        assert game_trajectory.actor_id == 0
        assert game_trajectory.game_length == 10
        assert game_trajectory.winner == 1

    def test_creation_empty(self) -> None:
        """Test GameTrajectory with no examples."""
        trajectory = GameTrajectory()
        assert len(trajectory.examples) == 0
        assert trajectory.game_id == 0
        assert trajectory.winner is None

    def test_to_bytes_uncompressed(self, game_trajectory: GameTrajectory) -> None:
        """Test serialization to bytes."""
        data = game_trajectory.to_bytes(compress=False)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_to_bytes_compressed(self, game_trajectory: GameTrajectory) -> None:
        """Test serialization with compression."""
        data = game_trajectory.to_bytes(compress=True)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_from_bytes_roundtrip(self, game_trajectory: GameTrajectory) -> None:
        """Test roundtrip serialization."""
        data = game_trajectory.to_bytes(compress=False)
        restored = GameTrajectory.from_bytes(data, compressed=False)

        assert len(restored.examples) == len(game_trajectory.examples)
        assert restored.game_id == game_trajectory.game_id
        assert restored.actor_id == game_trajectory.actor_id
        assert restored.game_length == game_trajectory.game_length
        assert restored.winner == game_trajectory.winner

    def test_from_bytes_compressed_roundtrip(self, game_trajectory: GameTrajectory) -> None:
        """Test compressed roundtrip serialization."""
        data = game_trajectory.to_bytes(compress=True)
        restored = GameTrajectory.from_bytes(data, compressed=True)

        assert len(restored.examples) == len(game_trajectory.examples)
        assert restored.game_id == game_trajectory.game_id


# =============================================================================
# SharedStorageConfig Tests
# =============================================================================


class TestSharedStorageConfig:
    """Tests for SharedStorageConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SharedStorageConfig()
        assert config.mode == "multiprocessing"
        assert config.max_model_history == 10
        assert config.compress_weights is True
        assert config.compress_trajectories is False
        assert config.track_statistics is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = SharedStorageConfig(
            mode="distributed",
            max_model_history=5,
            compress_weights=False,
            track_statistics=False,
        )
        assert config.mode == "distributed"
        assert config.max_model_history == 5
        assert config.compress_weights is False
        assert config.track_statistics is False


# =============================================================================
# SharedStorage Tests
# =============================================================================


class TestSharedStorage:
    """Tests for SharedStorage class."""

    def test_initialization(self) -> None:
        """Test SharedStorage initializes correctly."""
        storage = SharedStorage()
        try:
            assert storage.get_model_version() == 0
            assert storage.get_training_step() == 0
            assert storage.get_games_played() == 0
            assert storage.get_total_examples() == 0
            assert storage.is_training() is False
            assert storage.should_stop() is False
        finally:
            storage.shutdown()

    def test_initialization_with_config(self) -> None:
        """Test SharedStorage with custom config."""
        config = SharedStorageConfig(compress_weights=False, track_statistics=False)
        storage = SharedStorage(config=config)
        try:
            assert storage.config.compress_weights is False
            assert storage.config.track_statistics is False
        finally:
            storage.shutdown()

    def test_set_get_model_weights(self, simple_network: torch.nn.Module) -> None:
        """Test storing and retrieving model weights."""
        storage = SharedStorage()
        try:
            # Store weights
            state_dict = simple_network.state_dict()
            version = storage.set_model_weights(state_dict)
            assert version == 1

            # Retrieve weights
            retrieved = storage.get_model_weights()
            assert retrieved is not None

            # Verify weights match
            for key in state_dict:
                torch.testing.assert_close(retrieved[key], state_dict[key])
        finally:
            storage.shutdown()

    def test_model_version_increment(self, simple_network: torch.nn.Module) -> None:
        """Test model version increments on weight update."""
        storage = SharedStorage()
        try:
            state_dict = simple_network.state_dict()

            v1 = storage.set_model_weights(state_dict)
            assert v1 == 1

            v2 = storage.set_model_weights(state_dict)
            assert v2 == 2

            v3 = storage.set_model_weights(state_dict, increment_version=False)
            assert v3 == 2  # No increment
        finally:
            storage.shutdown()

    def test_get_model_weights_returns_none_initially(self) -> None:
        """Test get_model_weights returns None when no weights stored."""
        storage = SharedStorage()
        try:
            assert storage.get_model_weights() is None
        finally:
            storage.shutdown()

    def test_increment_games_played(self) -> None:
        """Test incrementing games played counter."""
        storage = SharedStorage()
        try:
            assert storage.get_games_played() == 0

            result = storage.increment_games_played()
            assert result == 1
            assert storage.get_games_played() == 1

            result = storage.increment_games_played(5)
            assert result == 6
            assert storage.get_games_played() == 6
        finally:
            storage.shutdown()

    def test_increment_examples(self) -> None:
        """Test incrementing examples counter."""
        storage = SharedStorage()
        try:
            assert storage.get_total_examples() == 0

            result = storage.increment_examples(100)
            assert result == 100

            result = storage.increment_examples(50)
            assert result == 150
        finally:
            storage.shutdown()

    def test_training_step(self) -> None:
        """Test setting and getting training step."""
        storage = SharedStorage()
        try:
            assert storage.get_training_step() == 0

            storage.set_training_step(100)
            assert storage.get_training_step() == 100

            storage.set_training_step(500)
            assert storage.get_training_step() == 500
        finally:
            storage.shutdown()

    def test_training_flag(self) -> None:
        """Test setting and checking training flag."""
        storage = SharedStorage()
        try:
            assert storage.is_training() is False

            storage.set_training(True)
            assert storage.is_training() is True

            storage.set_training(False)
            assert storage.is_training() is False
        finally:
            storage.shutdown()

    def test_request_stop(self) -> None:
        """Test requesting stop."""
        storage = SharedStorage()
        try:
            assert storage.should_stop() is False

            storage.request_stop()
            assert storage.should_stop() is True
        finally:
            storage.shutdown()

    def test_add_loss_entry(self) -> None:
        """Test adding loss history entries."""
        storage = SharedStorage()
        try:
            entry = {"total_loss": 1.5, "policy_loss": 0.8, "value_loss": 0.7}
            storage.add_loss_entry(entry)

            stats = storage.get_statistics()
            assert len(stats["loss_history"]) == 1
            assert stats["loss_history"][0] == entry
        finally:
            storage.shutdown()

    def test_add_eval_entry(self) -> None:
        """Test adding evaluation history entries."""
        storage = SharedStorage()
        try:
            entry = {"win_rate": 0.65, "games": 20}
            storage.add_eval_entry(entry)

            stats = storage.get_statistics()
            assert len(stats["eval_history"]) == 1
            assert stats["eval_history"][0] == entry
        finally:
            storage.shutdown()

    def test_statistics_disabled(self) -> None:
        """Test that statistics are not tracked when disabled."""
        config = SharedStorageConfig(track_statistics=False)
        storage = SharedStorage(config=config)
        try:
            storage.add_loss_entry({"loss": 1.0})
            storage.add_eval_entry({"win_rate": 0.5})

            stats = storage.get_statistics()
            assert len(stats["loss_history"]) == 0
            assert len(stats["eval_history"]) == 0
        finally:
            storage.shutdown()

    def test_get_statistics(self, simple_network: torch.nn.Module) -> None:
        """Test get_statistics returns comprehensive data."""
        storage = SharedStorage()
        try:
            storage.set_model_weights(simple_network.state_dict())
            storage.set_training_step(50)
            storage.increment_games_played(10)
            storage.increment_examples(1000)
            storage.set_training(True)

            stats = storage.get_statistics()

            assert stats["model_version"] == 1
            assert stats["training_step"] == 50
            assert stats["games_played"] == 10
            assert stats["total_examples"] == 1000
            assert stats["is_training"] is True
        finally:
            storage.shutdown()


# =============================================================================
# ReplayBufferConfig Tests
# =============================================================================


class TestReplayBufferConfig:
    """Tests for ReplayBufferConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ReplayBufferConfig()
        assert config.max_size == 500_000
        assert config.min_size == 10_000
        assert config.use_prioritized is False
        assert config.priority_alpha == 0.6
        assert config.priority_beta == 0.4
        assert config.compress is False
        assert config.trajectory_queue_size == 1000
        assert config.sample_batch_size == 256

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ReplayBufferConfig(
            max_size=100_000,
            min_size=1_000,
            compress=True,
            sample_batch_size=128,
        )
        assert config.max_size == 100_000
        assert config.min_size == 1_000
        assert config.compress is True
        assert config.sample_batch_size == 128


# =============================================================================
# DistributedReplayBuffer Tests
# =============================================================================


class TestDistributedReplayBuffer:
    """Tests for DistributedReplayBuffer class."""

    def test_initialization(self) -> None:
        """Test DistributedReplayBuffer initializes correctly."""
        buffer = DistributedReplayBuffer()
        try:
            assert buffer.size() == 0
            assert buffer.is_ready() is False

            stats = buffer.get_statistics()
            assert stats["total_trajectories"] == 0
            assert stats["total_examples"] == 0
            assert stats["buffer_size"] == 0
            assert stats["is_ready"] is False
        finally:
            buffer.shutdown()

    def test_initialization_with_config(self) -> None:
        """Test buffer with custom config."""
        config = ReplayBufferConfig(max_size=1000, min_size=100)
        buffer = DistributedReplayBuffer(config=config)
        try:
            assert buffer.config.max_size == 1000
            assert buffer.config.min_size == 100
        finally:
            buffer.shutdown()

    def test_submit_and_process_trajectory(self, game_trajectory: GameTrajectory) -> None:
        """Test submitting and processing a trajectory."""
        buffer = DistributedReplayBuffer()
        try:
            # Submit trajectory
            success = buffer.submit_trajectory(game_trajectory, timeout=1.0)
            assert success is True

            # Process it
            processed = buffer.process_pending_trajectories(max_count=10)
            assert processed == 1

            # Check buffer state
            assert buffer.size() == len(game_trajectory.examples)

            stats = buffer.get_statistics()
            assert stats["total_trajectories"] == 1
            assert stats["total_examples"] == len(game_trajectory.examples)
        finally:
            buffer.shutdown()

    def test_sample_from_buffer(self, training_example: TrainingExample) -> None:
        """Test sampling from the buffer."""
        config = ReplayBufferConfig(min_size=5)
        buffer = DistributedReplayBuffer(config=config)
        try:
            # Add examples directly (bypassing queue for simplicity)
            with buffer._buffer_lock:
                for _ in range(20):
                    buffer._buffer.append(training_example)
            buffer._stats["buffer_size"] = 20

            # Sample
            samples = buffer.sample(5)
            assert len(samples) == 5
            assert all(isinstance(s, TrainingExample) for s in samples)
        finally:
            buffer.shutdown()

    def test_sample_batch_returns_tensors(self, training_example: TrainingExample) -> None:
        """Test sample_batch returns properly formatted tensors."""
        config = ReplayBufferConfig(sample_batch_size=4, min_size=5)
        buffer = DistributedReplayBuffer(config=config)
        try:
            # Add examples directly
            with buffer._buffer_lock:
                for _ in range(20):
                    buffer._buffer.append(training_example)

            # Sample batch
            batch = buffer.sample_batch(batch_size=4)

            assert "spatial" in batch
            assert "vector" in batch
            assert "action_mask" in batch
            assert "policy" in batch
            assert "value" in batch

            assert batch["spatial"].shape == (4, 13, 16, 16)
            assert batch["vector"].shape == (4, 64)
            assert batch["action_mask"].shape == (4, 381)
            assert batch["policy"].shape == (4, 381)
            assert batch["value"].shape == (4,)
        finally:
            buffer.shutdown()

    def test_sample_empty_buffer(self) -> None:
        """Test sampling from empty buffer returns empty."""
        buffer = DistributedReplayBuffer()
        try:
            samples = buffer.sample(10)
            assert samples == []

            batch = buffer.sample_batch(10)
            assert batch == {}
        finally:
            buffer.shutdown()

    def test_is_ready(self, training_example: TrainingExample) -> None:
        """Test is_ready based on min_size."""
        config = ReplayBufferConfig(min_size=10)
        buffer = DistributedReplayBuffer(config=config)
        try:
            # Not ready with no examples
            assert buffer.is_ready() is False

            # Add 5 examples
            with buffer._buffer_lock:
                for _ in range(5):
                    buffer._buffer.append(training_example)
            assert buffer.is_ready() is False

            # Add 5 more examples (total 10)
            with buffer._buffer_lock:
                for _ in range(5):
                    buffer._buffer.append(training_example)
            assert buffer.is_ready() is True
        finally:
            buffer.shutdown()

    def test_buffer_max_size(self, training_example: TrainingExample) -> None:
        """Test buffer respects max_size limit."""
        config = ReplayBufferConfig(max_size=10, min_size=1)
        buffer = DistributedReplayBuffer(config=config)
        try:
            # Add 20 examples
            with buffer._buffer_lock:
                for _ in range(20):
                    buffer._buffer.append(training_example)

            # Should only keep max_size
            assert buffer.size() == 10
        finally:
            buffer.shutdown()

    def test_clear_buffer(self, training_example: TrainingExample) -> None:
        """Test clearing the buffer."""
        buffer = DistributedReplayBuffer()
        try:
            # Add examples
            with buffer._buffer_lock:
                for _ in range(10):
                    buffer._buffer.append(training_example)
            buffer._stats["buffer_size"] = 10

            assert buffer.size() == 10

            # Clear
            buffer.clear()
            assert buffer.size() == 0
        finally:
            buffer.shutdown()


# =============================================================================
# WeightSynchronizer Tests
# =============================================================================


class TestWeightSynchronizer:
    """Tests for WeightSynchronizer class."""

    def test_initialization(self) -> None:
        """Test WeightSynchronizer initializes correctly."""
        storage = SharedStorage()
        try:
            sync = WeightSynchronizer(storage, update_interval=1.0)

            assert sync.storage is storage
            assert sync.update_interval == 1.0
            assert sync.get_current_version() == 0
        finally:
            storage.shutdown()

    def test_should_update_false_initially(self) -> None:
        """Test should_update returns False when no weights stored."""
        storage = SharedStorage()
        try:
            sync = WeightSynchronizer(storage, update_interval=0.0)

            # No weights stored, but version is 0
            assert sync.should_update() is False
        finally:
            storage.shutdown()

    def test_should_update_after_new_weights(self, simple_network: torch.nn.Module) -> None:
        """Test should_update returns True after weights are stored."""
        storage = SharedStorage()
        try:
            sync = WeightSynchronizer(storage, update_interval=0.0)

            # Store weights
            storage.set_model_weights(simple_network.state_dict())

            # Should now return True
            assert sync.should_update() is True
        finally:
            storage.shutdown()

    def test_should_update_respects_interval(self, simple_network: torch.nn.Module) -> None:
        """Test should_update respects update_interval."""
        storage = SharedStorage()
        try:
            sync = WeightSynchronizer(storage, update_interval=10.0)

            # Store weights
            storage.set_model_weights(simple_network.state_dict())

            # First call should check
            result1 = sync.should_update()
            assert result1 is True

            # Update model to clear the flag
            sync.update_model(simple_network)

            # Store new weights
            storage.set_model_weights(simple_network.state_dict())

            # Second call within interval should return False
            result2 = sync.should_update()
            assert result2 is False
        finally:
            storage.shutdown()

    def test_update_model(self, simple_network: torch.nn.Module) -> None:
        """Test update_model loads weights into model."""
        storage = SharedStorage()
        try:
            sync = WeightSynchronizer(storage, update_interval=0.0)

            # Modify network
            original_weight = simple_network.weight.data.clone()
            simple_network.weight.data.fill_(999.0)

            # Store original weights
            storage.set_model_weights({"weight": original_weight, "bias": simple_network.bias.data})

            # Update model
            success = sync.update_model(simple_network)
            assert success is True
            assert sync.get_current_version() == 1

            # Verify weights were restored
            torch.testing.assert_close(simple_network.weight.data, original_weight)
        finally:
            storage.shutdown()

    def test_update_model_returns_false_no_weights(self, simple_network: torch.nn.Module) -> None:
        """Test update_model returns False when no weights available."""
        storage = SharedStorage()
        try:
            sync = WeightSynchronizer(storage, update_interval=0.0)

            success = sync.update_model(simple_network)
            assert success is False
        finally:
            storage.shutdown()

    def test_version_tracking(self, simple_network: torch.nn.Module) -> None:
        """Test version tracking after updates."""
        storage = SharedStorage()
        try:
            sync = WeightSynchronizer(storage, update_interval=0.0)

            assert sync.get_current_version() == 0

            # Store and update
            storage.set_model_weights(simple_network.state_dict())
            sync.update_model(simple_network)
            assert sync.get_current_version() == 1

            # Store and update again
            storage.set_model_weights(simple_network.state_dict())
            sync.update_model(simple_network)
            assert sync.get_current_version() == 2
        finally:
            storage.shutdown()
