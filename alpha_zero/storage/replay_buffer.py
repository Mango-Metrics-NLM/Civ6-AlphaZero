"""Storage-backed replay buffer for distributed training.

Provides a PyTorch IterableDataset that streams training examples
from storage (local or GCS), enabling memory-efficient training
on large trajectory datasets.

Usage:
    from alpha_zero.storage import create_storage_backend, StorageConfig
    from alpha_zero.storage.replay_buffer import StorageReplayBuffer

    storage = create_storage_backend(StorageConfig.for_local("./data"))
    buffer = StorageReplayBuffer(
        storage=storage,
        window_size=10,  # Last 10 iterations
        batch_size=256,
    )

    # Use with PyTorch DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(buffer, batch_size=None)  # batch_size handled by buffer

    for batch in loader:
        # Train on batch
        pass
"""

from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from alpha_zero.distributed.shared_storage import GameTrajectory, TrainingExample
    from alpha_zero.storage.backend import StorageBackend

logger = logging.getLogger(__name__)


@dataclass
class ReplayBufferConfig:
    """Configuration for storage-backed replay buffer.

    Attributes:
        window_size: Number of recent iterations to include (0 = all).
        min_examples: Minimum examples before yielding batches.
        batch_size: Number of examples per batch.
        prefetch_trajectories: Number of trajectories to prefetch.
        shuffle_buffer_size: Size of shuffle buffer (0 = no shuffling).
        cache_size: Maximum trajectories to cache in memory.
        iterations: Explicit list of iterations to include (overrides window).
    """

    window_size: int = 10
    min_examples: int = 1000
    batch_size: int = 256
    prefetch_trajectories: int = 10
    shuffle_buffer_size: int = 10000
    cache_size: int = 100
    iterations: list[int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayBufferConfig:
        """Create from dictionary."""
        return cls(
            window_size=data.get("window_size", 10),
            min_examples=data.get("min_examples", 1000),
            batch_size=data.get("batch_size", 256),
            prefetch_trajectories=data.get("prefetch_trajectories", 10),
            shuffle_buffer_size=data.get("shuffle_buffer_size", 10000),
            cache_size=data.get("cache_size", 100),
            iterations=data.get("iterations"),
        )


class StorageReplayBuffer(IterableDataset):
    """Storage-backed replay buffer as a PyTorch IterableDataset.

    This class provides a memory-efficient way to iterate over training
    examples stored in local filesystem or GCS. It supports:

    - Sliding window over recent iterations
    - Shuffled streaming without loading all data
    - Prefetching for reduced latency
    - Batching with collation

    The buffer yields dictionaries with batched tensors ready for training:
    {
        "spatial": torch.Tensor,    # (batch, C, H, W)
        "vector": torch.Tensor,     # (batch, D)
        "action_mask": torch.Tensor,# (batch, A)
        "policy": torch.Tensor,     # (batch, A)
        "value": torch.Tensor,      # (batch,)
    }

    Attributes:
        storage: Storage backend for loading trajectories.
        config: Buffer configuration.
    """

    def __init__(
        self,
        storage: StorageBackend,
        config: ReplayBufferConfig | None = None,
        current_iteration: int | None = None,
    ) -> None:
        """Initialize storage-backed replay buffer.

        Args:
            storage: Storage backend instance.
            config: Buffer configuration.
            current_iteration: Current training iteration (for window calculation).
        """
        self.storage = storage
        self.config = config or ReplayBufferConfig()
        self.current_iteration = current_iteration or 0

        # Cache for loaded trajectories
        self._trajectory_cache: deque[GameTrajectory] = deque(
            maxlen=self.config.cache_size
        )

        # Shuffle buffer for examples
        self._shuffle_buffer: list[TrainingExample] = []

        logger.info(
            f"StorageReplayBuffer initialized "
            f"(window={self.config.window_size}, batch={self.config.batch_size})"
        )

    def set_current_iteration(self, iteration: int) -> None:
        """Update the current iteration for window calculation.

        Args:
            iteration: New current iteration.
        """
        self.current_iteration = iteration

    def _get_iterations_to_include(self) -> list[int]:
        """Determine which iterations to include based on config."""
        if self.config.iterations is not None:
            return self.config.iterations

        if self.config.window_size <= 0:
            # Include all iterations
            all_metadata = self.storage.list_trajectories()
            return sorted({m.iteration for m in all_metadata})

        # Window from current iteration
        start_iter = max(0, self.current_iteration - self.config.window_size + 1)
        return list(range(start_iter, self.current_iteration + 1))

    def count_available_examples(self) -> int:
        """Count total available examples across all included iterations.

        Returns:
            Total number of training examples.
        """
        iterations = self._get_iterations_to_include()
        total = 0

        for iteration in iterations:
            metadata_list = self.storage.list_trajectories(iteration=iteration)
            total += sum(m.num_examples for m in metadata_list)

        return total

    def is_ready(self) -> bool:
        """Check if buffer has enough examples for training.

        Returns:
            True if minimum examples threshold is met.
        """
        return self.count_available_examples() >= self.config.min_examples

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Iterate over batches of training examples.

        Yields:
            Dictionaries with batched tensors for training.
        """
        iterations = self._get_iterations_to_include()
        logger.debug(f"Streaming from iterations: {iterations}")

        # Stream trajectories with shuffling
        trajectory_stream = self.storage.stream_trajectories(
            iterations=iterations,
            shuffle=True,
        )

        # Fill shuffle buffer and yield batches
        batch_examples: list[TrainingExample] = []

        for trajectory in trajectory_stream:
            # Add examples to shuffle buffer
            self._shuffle_buffer.extend(trajectory.examples)

            # Shuffle when buffer is full enough
            if len(self._shuffle_buffer) >= self.config.shuffle_buffer_size:
                random.shuffle(self._shuffle_buffer)

            # Yield batches from shuffle buffer
            while len(self._shuffle_buffer) >= self.config.batch_size:
                batch_examples = self._shuffle_buffer[: self.config.batch_size]
                self._shuffle_buffer = self._shuffle_buffer[self.config.batch_size :]

                yield self._collate_batch(batch_examples)

        # Handle remaining examples
        random.shuffle(self._shuffle_buffer)
        while len(self._shuffle_buffer) >= self.config.batch_size:
            batch_examples = self._shuffle_buffer[: self.config.batch_size]
            self._shuffle_buffer = self._shuffle_buffer[self.config.batch_size :]
            yield self._collate_batch(batch_examples)

        # Final partial batch if any remaining
        if self._shuffle_buffer:
            yield self._collate_batch(self._shuffle_buffer)
            self._shuffle_buffer = []

    def _collate_batch(
        self,
        examples: list[TrainingExample],
    ) -> dict[str, torch.Tensor]:
        """Collate a list of examples into a batch tensor dictionary.

        Args:
            examples: List of training examples.

        Returns:
            Dictionary with batched tensors.
        """
        if not examples:
            return {}

        return {
            "spatial": torch.stack(
                [torch.from_numpy(ex.spatial).float() for ex in examples]
            ),
            "vector": torch.stack(
                [torch.from_numpy(ex.vector).float() for ex in examples]
            ),
            "action_mask": torch.stack(
                [torch.from_numpy(ex.action_mask).float() for ex in examples]
            ),
            "policy": torch.stack(
                [torch.from_numpy(ex.policy).float() for ex in examples]
            ),
            "value": torch.tensor([ex.value for ex in examples]).float(),
        }

    def sample_batch(
        self,
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample a random batch from available trajectories.

        This method is useful for non-streaming access patterns.

        Args:
            batch_size: Batch size (defaults to config.batch_size).

        Returns:
            Dictionary with batched tensors.
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        iterations = self._get_iterations_to_include()

        # Collect examples from random trajectories
        all_examples: list[TrainingExample] = []
        metadata_list = []
        for iteration in iterations:
            metadata_list.extend(self.storage.list_trajectories(iteration=iteration))

        if not metadata_list:
            return {}

        # Random sampling with replacement until we have enough
        while len(all_examples) < batch_size:
            meta = random.choice(metadata_list)
            try:
                trajectory = self.storage.load_trajectory(
                    iteration=meta.iteration,
                    worker_id=meta.worker_id,
                    game_id=meta.game_id,
                )
                all_examples.extend(trajectory.examples)
            except Exception as e:
                logger.warning(f"Failed to load trajectory: {e}")

        # Sample batch
        if len(all_examples) >= batch_size:
            batch_indices = np.random.choice(
                len(all_examples), size=batch_size, replace=False
            )
            batch_examples = [all_examples[i] for i in batch_indices]
        else:
            batch_examples = all_examples

        return self._collate_batch(batch_examples)

    def get_statistics(self) -> dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer statistics.
        """
        iterations = self._get_iterations_to_include()
        total_trajectories = 0
        total_examples = 0

        for iteration in iterations:
            metadata_list = self.storage.list_trajectories(iteration=iteration)
            total_trajectories += len(metadata_list)
            total_examples += sum(m.num_examples for m in metadata_list)

        return {
            "iterations_included": len(iterations),
            "iteration_range": (min(iterations), max(iterations)) if iterations else (0, 0),
            "total_trajectories": total_trajectories,
            "total_examples": total_examples,
            "is_ready": total_examples >= self.config.min_examples,
            "shuffle_buffer_size": len(self._shuffle_buffer),
            "cache_size": len(self._trajectory_cache),
        }


class InMemoryReplayBuffer:
    """In-memory replay buffer for local training.

    This buffer keeps all examples in memory for fast access,
    suitable for single-machine training with moderate dataset sizes.

    Attributes:
        config: Buffer configuration.
    """

    def __init__(
        self,
        config: ReplayBufferConfig | None = None,
        max_size: int = 500_000,
    ) -> None:
        """Initialize in-memory replay buffer.

        Args:
            config: Buffer configuration.
            max_size: Maximum number of examples to store.
        """
        self.config = config or ReplayBufferConfig()
        self.max_size = max_size

        self._buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add_trajectory(self, trajectory: GameTrajectory) -> None:
        """Add a trajectory's examples to the buffer.

        Args:
            trajectory: Game trajectory to add.
        """
        for example in trajectory.examples:
            self._buffer.append(example)

    def add_example(self, example: TrainingExample) -> None:
        """Add a single example to the buffer.

        Args:
            example: Training example to add.
        """
        self._buffer.append(example)

    def sample_batch(
        self,
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample a random batch from the buffer.

        Args:
            batch_size: Batch size (defaults to config.batch_size).

        Returns:
            Dictionary with batched tensors.
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        if len(self._buffer) < batch_size:
            batch_size = len(self._buffer)

        if batch_size == 0:
            return {}

        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch_examples = [self._buffer[i] for i in indices]

        return {
            "spatial": torch.stack(
                [torch.from_numpy(ex.spatial).float() for ex in batch_examples]
            ),
            "vector": torch.stack(
                [torch.from_numpy(ex.vector).float() for ex in batch_examples]
            ),
            "action_mask": torch.stack(
                [torch.from_numpy(ex.action_mask).float() for ex in batch_examples]
            ),
            "policy": torch.stack(
                [torch.from_numpy(ex.policy).float() for ex in batch_examples]
            ),
            "value": torch.tensor([ex.value for ex in batch_examples]).float(),
        }

    def size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)

    def is_ready(self) -> bool:
        """Check if buffer has enough examples."""
        return len(self._buffer) >= self.config.min_examples

    def clear(self) -> None:
        """Clear all examples from buffer."""
        self._buffer.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get buffer statistics."""
        return {
            "size": len(self._buffer),
            "max_size": self.max_size,
            "is_ready": self.is_ready(),
        }


__all__ = [
    "ReplayBufferConfig",
    "StorageReplayBuffer",
    "InMemoryReplayBuffer",
]
