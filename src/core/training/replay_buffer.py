"""Replay buffer for storing and sampling training examples."""
from __future__ import annotations

__all__ = ["ReplayBuffer"]

import logging
import random
from collections import deque

import numpy as np
from numpy.typing import NDArray

from src.core.types import GameTrajectory, TrainingExample

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Circular replay buffer for training examples.

    Stores TrainingExample instances and supports uniform random sampling.
    Thread-safe for single-writer / single-reader patterns.

    Args:
        capacity: Maximum number of examples stored. When full, oldest removed.
        seed: Optional random seed for reproducible sampling.
    """

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._buffer: deque[TrainingExample] = deque(maxlen=capacity)
        self._rng = random.Random(seed)  # noqa: S311
        self._total_added = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self._capacity

    @property
    def total_added(self) -> int:
        """Total examples added (including evicted ones)."""
        return self._total_added

    def add(self, example: TrainingExample) -> None:
        """Add a single training example."""
        self._buffer.append(example)
        self._total_added += 1

    def add_trajectory(self, trajectory: GameTrajectory) -> None:
        """Add all examples from a game trajectory."""
        for example in trajectory.examples:
            self.add(example)
        logger.debug(
            "Added trajectory %s with %d examples (buffer size: %d/%d)",
            trajectory.episode_id[:8],
            len(trajectory.examples),
            self.size,
            self.capacity,
        )

    def sample(self, batch_size: int) -> list[TrainingExample]:
        """Sample a random batch of examples.

        Args:
            batch_size: Number of examples to sample.

        Returns:
            List of sampled TrainingExample instances.

        Raises:
            ValueError: If batch_size > buffer size.
        """
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {self.size}"
            )
        indices = self._rng.sample(range(self.size), batch_size)
        return [self._buffer[i] for i in indices]

    def sample_batch(self, batch_size: int) -> dict[str, NDArray[np.float32]]:
        """Sample and collate into numpy arrays ready for training.

        Returns dict with keys: spatial, vector, action_mask, policy, value
        """
        examples = self.sample(batch_size)
        return {
            "spatial": np.stack([e.spatial for e in examples]),
            "vector": np.stack([e.vector for e in examples]),
            "action_mask": np.stack([e.action_mask for e in examples]),
            "policy": np.stack([e.policy for e in examples]),
            "value": np.array([e.value for e in examples], dtype=np.float32),
        }

    def clear(self) -> None:
        """Remove all examples from the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, size={self.size})"
