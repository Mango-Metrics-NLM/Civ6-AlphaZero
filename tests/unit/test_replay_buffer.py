"""Unit tests for storage-backed replay buffer.

Tests both StorageReplayBuffer (for streaming from storage) and
InMemoryReplayBuffer (for local training).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from alpha_zero.distributed.shared_storage import GameTrajectory, TrainingExample
from alpha_zero.storage import LocalStorageBackend, StorageConfig
from alpha_zero.storage.replay_buffer import (
    InMemoryReplayBuffer,
    ReplayBufferConfig,
    StorageReplayBuffer,
)


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """Create a temporary storage directory."""
    return tmp_path / "storage"


@pytest.fixture
def local_backend(temp_storage_dir: Path) -> LocalStorageBackend:
    """Create a local storage backend."""
    config = StorageConfig(
        backend_type="local",
        base_path=str(temp_storage_dir),
        compress_weights=False,
        compress_trajectories=False,
    )
    return LocalStorageBackend(config)


@pytest.fixture
def sample_example() -> TrainingExample:
    """Create a sample training example."""
    return TrainingExample(
        spatial=np.random.randn(13, 8, 8).astype(np.float32),
        vector=np.random.randn(64).astype(np.float32),
        action_mask=np.ones(356, dtype=np.float32),
        policy=np.random.rand(356).astype(np.float32),
        value=float(np.random.uniform(-1, 1)),
    )


@pytest.fixture
def sample_trajectory(sample_example: TrainingExample) -> GameTrajectory:
    """Create a sample game trajectory."""
    examples = [
        TrainingExample(
            spatial=np.random.randn(13, 8, 8).astype(np.float32),
            vector=np.random.randn(64).astype(np.float32),
            action_mask=np.ones(356, dtype=np.float32),
            policy=np.random.rand(356).astype(np.float32),
            value=float(np.random.uniform(-1, 1)),
        )
        for _ in range(10)
    ]
    return GameTrajectory(
        examples=examples,
        game_id=1,
        actor_id=0,
        game_length=50,
        winner=0,
    )


def populate_storage(
    backend: LocalStorageBackend,
    num_iterations: int = 3,
    trajectories_per_iteration: int = 5,
    examples_per_trajectory: int = 10,
) -> int:
    """Populate storage with sample trajectories.

    Returns:
        Total number of examples added.
    """
    total_examples = 0

    for iteration in range(num_iterations):
        for traj_idx in range(trajectories_per_iteration):
            examples = [
                TrainingExample(
                    spatial=np.random.randn(13, 8, 8).astype(np.float32),
                    vector=np.random.randn(64).astype(np.float32),
                    action_mask=np.ones(356, dtype=np.float32),
                    policy=np.random.rand(356).astype(np.float32),
                    value=float(np.random.uniform(-1, 1)),
                )
                for _ in range(examples_per_trajectory)
            ]

            trajectory = GameTrajectory(
                examples=examples,
                game_id=traj_idx,
                actor_id=0,
                game_length=50,
                winner=0,
            )

            backend.save_trajectory(
                trajectory=trajectory,
                iteration=iteration,
                worker_id=0,
                game_id=f"game_{iteration}_{traj_idx}",
            )
            total_examples += examples_per_trajectory

    return total_examples


class TestReplayBufferConfig:
    """Tests for ReplayBufferConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ReplayBufferConfig()
        assert config.window_size == 10
        assert config.min_examples == 1000
        assert config.batch_size == 256
        assert config.shuffle_buffer_size == 10000

    def test_from_dict(self) -> None:
        """Test creating config from dictionary."""
        data = {
            "window_size": 5,
            "min_examples": 500,
            "batch_size": 128,
        }
        config = ReplayBufferConfig.from_dict(data)
        assert config.window_size == 5
        assert config.min_examples == 500
        assert config.batch_size == 128


class TestStorageReplayBuffer:
    """Tests for StorageReplayBuffer."""

    def test_initialization(self, local_backend: LocalStorageBackend) -> None:
        """Test buffer initialization."""
        config = ReplayBufferConfig(window_size=5, batch_size=32)
        buffer = StorageReplayBuffer(
            storage=local_backend,
            config=config,
            current_iteration=10,
        )
        assert buffer.current_iteration == 10
        assert buffer.config.batch_size == 32

    def test_set_current_iteration(self, local_backend: LocalStorageBackend) -> None:
        """Test updating current iteration."""
        buffer = StorageReplayBuffer(storage=local_backend, current_iteration=0)
        buffer.set_current_iteration(50)
        assert buffer.current_iteration == 50

    def test_count_available_examples(self, local_backend: LocalStorageBackend) -> None:
        """Test counting available examples."""
        total_examples = populate_storage(
            local_backend,
            num_iterations=3,
            trajectories_per_iteration=2,
            examples_per_trajectory=5,
        )

        config = ReplayBufferConfig(window_size=10)  # Include all
        buffer = StorageReplayBuffer(
            storage=local_backend,
            config=config,
            current_iteration=2,
        )

        count = buffer.count_available_examples()
        assert count == total_examples

    def test_is_ready(self, local_backend: LocalStorageBackend) -> None:
        """Test readiness check based on min_examples."""
        # Low threshold - should be ready
        config = ReplayBufferConfig(min_examples=10, window_size=10)
        buffer = StorageReplayBuffer(
            storage=local_backend,
            config=config,
            current_iteration=5,
        )

        # Populate with enough examples
        populate_storage(
            local_backend,
            num_iterations=2,
            trajectories_per_iteration=2,
            examples_per_trajectory=10,
        )

        assert buffer.is_ready()

        # High threshold - should not be ready
        config_high = ReplayBufferConfig(min_examples=10000, window_size=10)
        buffer_high = StorageReplayBuffer(
            storage=local_backend,
            config=config_high,
            current_iteration=5,
        )
        assert not buffer_high.is_ready()

    def test_iteration(self, local_backend: LocalStorageBackend) -> None:
        """Test iterating over batches."""
        populate_storage(
            local_backend,
            num_iterations=2,
            trajectories_per_iteration=3,
            examples_per_trajectory=20,
        )

        config = ReplayBufferConfig(
            window_size=10,
            batch_size=16,
            min_examples=10,
            shuffle_buffer_size=50,
        )
        buffer = StorageReplayBuffer(
            storage=local_backend,
            config=config,
            current_iteration=1,
        )

        batches = list(buffer)
        assert len(batches) > 0

        # Check batch structure
        batch = batches[0]
        assert "spatial" in batch
        assert "vector" in batch
        assert "action_mask" in batch
        assert "policy" in batch
        assert "value" in batch

        # Check tensor shapes
        assert batch["spatial"].ndim == 4  # (B, C, H, W)
        assert batch["vector"].ndim == 2  # (B, D)
        assert batch["value"].ndim == 1  # (B,)

    def test_window_filtering(self, local_backend: LocalStorageBackend) -> None:
        """Test that window_size correctly filters iterations."""
        populate_storage(
            local_backend,
            num_iterations=5,
            trajectories_per_iteration=2,
            examples_per_trajectory=10,
        )

        # Window of 2 from iteration 4 should include iterations 3 and 4
        config = ReplayBufferConfig(window_size=2, min_examples=1)
        buffer = StorageReplayBuffer(
            storage=local_backend,
            config=config,
            current_iteration=4,
        )

        iterations = buffer._get_iterations_to_include()
        assert iterations == [3, 4]

    def test_explicit_iterations(self, local_backend: LocalStorageBackend) -> None:
        """Test explicit iteration list overrides window."""
        populate_storage(
            local_backend,
            num_iterations=5,
            trajectories_per_iteration=2,
            examples_per_trajectory=10,
        )

        config = ReplayBufferConfig(
            window_size=100,  # Would include all
            iterations=[0, 2],  # But this overrides
        )
        buffer = StorageReplayBuffer(
            storage=local_backend,
            config=config,
            current_iteration=4,
        )

        iterations = buffer._get_iterations_to_include()
        assert iterations == [0, 2]

    def test_sample_batch(self, local_backend: LocalStorageBackend) -> None:
        """Test random batch sampling."""
        populate_storage(
            local_backend,
            num_iterations=2,
            trajectories_per_iteration=3,
            examples_per_trajectory=20,
        )

        config = ReplayBufferConfig(
            window_size=10,
            batch_size=16,
        )
        buffer = StorageReplayBuffer(
            storage=local_backend,
            config=config,
            current_iteration=1,
        )

        batch = buffer.sample_batch()
        assert len(batch) > 0
        assert batch["spatial"].shape[0] == 16

        # Test custom batch size
        batch_small = buffer.sample_batch(batch_size=8)
        assert batch_small["spatial"].shape[0] == 8

    def test_get_statistics(self, local_backend: LocalStorageBackend) -> None:
        """Test buffer statistics."""
        total_examples = populate_storage(
            local_backend,
            num_iterations=3,
            trajectories_per_iteration=2,
            examples_per_trajectory=10,
        )

        config = ReplayBufferConfig(window_size=10, min_examples=10)
        buffer = StorageReplayBuffer(
            storage=local_backend,
            config=config,
            current_iteration=2,
        )

        stats = buffer.get_statistics()
        assert stats["iterations_included"] == 3
        assert stats["total_examples"] == total_examples
        assert stats["is_ready"] is True

    def test_empty_storage(self, local_backend: LocalStorageBackend) -> None:
        """Test behavior with empty storage."""
        config = ReplayBufferConfig(min_examples=10)
        buffer = StorageReplayBuffer(
            storage=local_backend,
            config=config,
            current_iteration=0,
        )

        assert buffer.count_available_examples() == 0
        assert not buffer.is_ready()

        batches = list(buffer)
        assert len(batches) == 0


class TestInMemoryReplayBuffer:
    """Tests for InMemoryReplayBuffer."""

    def test_initialization(self) -> None:
        """Test buffer initialization."""
        config = ReplayBufferConfig(batch_size=64, min_examples=100)
        buffer = InMemoryReplayBuffer(config=config, max_size=10000)
        assert buffer.size() == 0
        assert not buffer.is_ready()

    def test_add_example(self, sample_example: TrainingExample) -> None:
        """Test adding individual examples."""
        buffer = InMemoryReplayBuffer(max_size=100)

        for _ in range(50):
            buffer.add_example(sample_example)

        assert buffer.size() == 50

    def test_add_trajectory(self, sample_trajectory: GameTrajectory) -> None:
        """Test adding entire trajectories."""
        buffer = InMemoryReplayBuffer(max_size=1000)

        buffer.add_trajectory(sample_trajectory)
        assert buffer.size() == len(sample_trajectory.examples)

    def test_max_size(self, sample_example: TrainingExample) -> None:
        """Test that buffer respects max size."""
        buffer = InMemoryReplayBuffer(max_size=50)

        for _ in range(100):
            buffer.add_example(sample_example)

        assert buffer.size() == 50

    def test_sample_batch(self, sample_example: TrainingExample) -> None:
        """Test batch sampling."""
        config = ReplayBufferConfig(batch_size=16)
        buffer = InMemoryReplayBuffer(config=config, max_size=1000)

        # Add enough examples
        for _ in range(100):
            buffer.add_example(sample_example)

        batch = buffer.sample_batch()
        assert batch["spatial"].shape[0] == 16
        assert batch["vector"].shape[0] == 16
        assert batch["value"].shape[0] == 16

    def test_sample_batch_small_buffer(self, sample_example: TrainingExample) -> None:
        """Test sampling when buffer has fewer examples than batch size."""
        config = ReplayBufferConfig(batch_size=100)
        buffer = InMemoryReplayBuffer(config=config, max_size=1000)

        # Add fewer examples than batch size
        for _ in range(20):
            buffer.add_example(sample_example)

        batch = buffer.sample_batch()
        assert batch["spatial"].shape[0] == 20  # Should return all available

    def test_is_ready(self, sample_example: TrainingExample) -> None:
        """Test readiness check."""
        config = ReplayBufferConfig(min_examples=50)
        buffer = InMemoryReplayBuffer(config=config, max_size=1000)

        # Not ready initially
        assert not buffer.is_ready()

        # Add some examples
        for _ in range(30):
            buffer.add_example(sample_example)
        assert not buffer.is_ready()

        # Add more to reach threshold
        for _ in range(30):
            buffer.add_example(sample_example)
        assert buffer.is_ready()

    def test_clear(self, sample_example: TrainingExample) -> None:
        """Test clearing the buffer."""
        buffer = InMemoryReplayBuffer(max_size=100)

        for _ in range(50):
            buffer.add_example(sample_example)
        assert buffer.size() == 50

        buffer.clear()
        assert buffer.size() == 0

    def test_get_statistics(self, sample_example: TrainingExample) -> None:
        """Test buffer statistics."""
        config = ReplayBufferConfig(min_examples=100)
        buffer = InMemoryReplayBuffer(config=config, max_size=500)

        for _ in range(75):
            buffer.add_example(sample_example)

        stats = buffer.get_statistics()
        assert stats["size"] == 75
        assert stats["max_size"] == 500
        assert stats["is_ready"] is False

    def test_batch_tensor_types(self, sample_example: TrainingExample) -> None:
        """Test that batch tensors have correct types."""
        buffer = InMemoryReplayBuffer(max_size=100)

        for _ in range(50):
            buffer.add_example(sample_example)

        batch = buffer.sample_batch(batch_size=10)

        assert batch["spatial"].dtype == torch.float32
        assert batch["vector"].dtype == torch.float32
        assert batch["action_mask"].dtype == torch.float32
        assert batch["policy"].dtype == torch.float32
        assert batch["value"].dtype == torch.float32
