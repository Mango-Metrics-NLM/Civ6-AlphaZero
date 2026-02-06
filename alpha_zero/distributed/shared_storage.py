"""Shared storage for distributed AlphaZero training.

Provides:
- SharedStorage: Thread/process-safe model weight sharing
- DistributedReplayBuffer: Process-safe replay buffer with multiprocessing queues
"""

from __future__ import annotations

import io
import logging
import lzma
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import Manager, Queue
from multiprocessing.managers import DictProxy, SyncManager

import numpy as np
import torch
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """Single training example from self-play."""

    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    policy: np.ndarray
    value: float

    def to_bytes(self, compress: bool = False) -> bytes:
        """Serialize to bytes.

        Args:
            compress: Whether to compress with lzma.

        Returns:
            Serialized bytes.
        """
        data = pickle.dumps(
            {
                "spatial": self.spatial,
                "vector": self.vector,
                "action_mask": self.action_mask,
                "policy": self.policy,
                "value": self.value,
            }
        )
        if compress:
            data = lzma.compress(data)
        return data

    @classmethod
    def from_bytes(cls, data: bytes, compressed: bool = False) -> TrainingExample:
        """Deserialize from bytes.

        Args:
            data: Serialized bytes.
            compressed: Whether data is compressed.

        Returns:
            TrainingExample instance.
        """
        if compressed:
            data = lzma.decompress(data)
        d = pickle.loads(data)
        return cls(
            spatial=d["spatial"],
            vector=d["vector"],
            action_mask=d["action_mask"],
            policy=d["policy"],
            value=d["value"],
        )


@dataclass
class GameTrajectory:
    """Collection of training examples from a single game."""

    examples: list[TrainingExample] = field(default_factory=list)
    game_id: int = 0
    actor_id: int = 0
    game_length: int = 0
    winner: int | None = None

    def to_bytes(self, compress: bool = False) -> bytes:
        """Serialize to bytes."""
        data = pickle.dumps(
            {
                "examples": [ex.to_bytes(compress=False) for ex in self.examples],
                "game_id": self.game_id,
                "actor_id": self.actor_id,
                "game_length": self.game_length,
                "winner": self.winner,
            }
        )
        if compress:
            data = lzma.compress(data)
        return data

    @classmethod
    def from_bytes(cls, data: bytes, compressed: bool = False) -> GameTrajectory:
        """Deserialize from bytes."""
        if compressed:
            data = lzma.decompress(data)
        d = pickle.loads(data)
        return cls(
            examples=[TrainingExample.from_bytes(ex) for ex in d["examples"]],
            game_id=d["game_id"],
            actor_id=d["actor_id"],
            game_length=d["game_length"],
            winner=d["winner"],
        )


@dataclass
class SharedStorageConfig:
    """Configuration for shared storage."""

    # Storage mode: 'multiprocessing' for local, 'distributed' for future remote
    mode: str = "multiprocessing"

    # Model update settings
    max_model_history: int = 10

    # Compression settings
    compress_weights: bool = True
    compress_trajectories: bool = False

    # Statistics tracking
    track_statistics: bool = True


class SharedStorage:
    """Process-safe shared storage for model weights and training statistics.

    This class provides thread and process-safe access to:
    - Current model weights (for actors to fetch)
    - Model version tracking
    - Training statistics

    Uses multiprocessing.Manager for cross-process synchronization.
    """

    def __init__(
        self,
        config: SharedStorageConfig | None = None,
        manager: SyncManager | None = None,
    ) -> None:
        """Initialize shared storage.

        Args:
            config: Storage configuration.
            manager: Multiprocessing manager. If None, creates a new one.
        """
        self.config = config or SharedStorageConfig()

        # Create or use existing manager
        if manager is not None:
            self._manager = manager
            self._owns_manager = False
        else:
            self._manager = Manager()
            self._owns_manager = True

        # Shared state using manager proxies
        self._state: DictProxy = self._manager.dict()
        self._lock = self._manager.Lock()

        # Initialize state
        self._state["model_version"] = 0
        self._state["model_weights"] = None
        self._state["training_step"] = 0
        self._state["games_played"] = 0
        self._state["total_examples"] = 0
        self._state["is_training"] = False
        self._state["should_stop"] = False

        # Statistics history
        self._state["loss_history"] = []
        self._state["eval_history"] = []

        logger.info("SharedStorage initialized")

    def shutdown(self) -> None:
        """Cleanup resources."""
        if self._owns_manager:
            try:
                self._manager.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down manager: {e}")

    def set_model_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        increment_version: bool = True,
    ) -> int:
        """Store new model weights.

        Args:
            state_dict: Model state dictionary.
            increment_version: Whether to increment version number.

        Returns:
            New model version.
        """
        with self._lock:
            # Serialize weights
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            weights_bytes = buffer.getvalue()

            if self.config.compress_weights:
                weights_bytes = lzma.compress(weights_bytes)

            self._state["model_weights"] = weights_bytes

            if increment_version:
                self._state["model_version"] = self._state["model_version"] + 1

            version = int(self._state["model_version"])
            logger.debug(f"Stored model weights v{version} ({len(weights_bytes)} bytes)")
            return version

    def get_model_weights(self) -> dict[str, torch.Tensor] | None:
        """Retrieve current model weights.

        Returns:
            Model state dictionary, or None if not set.
        """
        weights, _ = self.get_model_weights_with_version()
        return weights

    def get_model_weights_with_version(self) -> tuple[dict[str, torch.Tensor] | None, int]:
        """Retrieve model weights and version atomically.

        This method ensures that the returned weights match the returned version,
        preventing race conditions where version changes during weight retrieval.

        Returns:
            Tuple of (state_dict, version). state_dict is None if weights not set.
        """
        with self._lock:
            weights_bytes = self._state.get("model_weights")
            version = int(self._state["model_version"])

        if weights_bytes is None:
            return None, version

        # Decompress if needed (outside lock - we have consistent bytes+version)
        if self.config.compress_weights:
            weights_bytes = lzma.decompress(weights_bytes)

        # Deserialize
        buffer = io.BytesIO(weights_bytes)
        state_dict: dict[str, torch.Tensor] = torch.load(
            buffer,
            map_location="cpu",
            weights_only=True,
        )
        return state_dict, version

    def get_model_version(self) -> int:
        """Get current model version."""
        return int(self._state["model_version"])

    def increment_games_played(self, count: int = 1) -> int:
        """Increment games played counter.

        Args:
            count: Number of games to add.

        Returns:
            New total games played.
        """
        with self._lock:
            self._state["games_played"] = self._state["games_played"] + count
            return int(self._state["games_played"])

    def increment_examples(self, count: int) -> int:
        """Increment total examples counter.

        Args:
            count: Number of examples to add.

        Returns:
            New total examples.
        """
        with self._lock:
            self._state["total_examples"] = self._state["total_examples"] + count
            return int(self._state["total_examples"])

    def set_training_step(self, step: int) -> None:
        """Set current training step."""
        self._state["training_step"] = step

    def get_training_step(self) -> int:
        """Get current training step."""
        return int(self._state["training_step"])

    def get_games_played(self) -> int:
        """Get total games played."""
        return int(self._state["games_played"])

    def get_total_examples(self) -> int:
        """Get total training examples generated."""
        return int(self._state["total_examples"])

    def set_training(self, is_training: bool) -> None:
        """Set training flag."""
        self._state["is_training"] = is_training

    def is_training(self) -> bool:
        """Check if training is in progress."""
        return bool(self._state["is_training"])

    def request_stop(self) -> None:
        """Request all processes to stop."""
        self._state["should_stop"] = True
        logger.info("Stop requested")

    def should_stop(self) -> bool:
        """Check if stop has been requested."""
        return bool(self._state["should_stop"])

    def add_loss_entry(self, entry: dict) -> None:
        """Add a loss history entry.

        Args:
            entry: Dictionary with loss values.
        """
        if not self.config.track_statistics:
            return
        with self._lock:
            history = list(self._state["loss_history"])
            history.append(entry)
            # Keep last 1000 entries
            if len(history) > 1000:
                history = history[-1000:]
            self._state["loss_history"] = history

    def add_eval_entry(self, entry: dict) -> None:
        """Add an evaluation history entry.

        Args:
            entry: Dictionary with evaluation results.
        """
        if not self.config.track_statistics:
            return
        with self._lock:
            history = list(self._state["eval_history"])
            history.append(entry)
            if len(history) > 100:
                history = history[-100:]
            self._state["eval_history"] = history

    def get_statistics(self) -> dict:
        """Get current training statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "model_version": self._state["model_version"],
            "training_step": self._state["training_step"],
            "games_played": self._state["games_played"],
            "total_examples": self._state["total_examples"],
            "is_training": self._state["is_training"],
            "loss_history": list(self._state["loss_history"]),
            "eval_history": list(self._state["eval_history"]),
        }


@dataclass
class ReplayBufferConfig:
    """Configuration for distributed replay buffer."""

    # Buffer capacity
    max_size: int = 500_000

    # Minimum examples before training can start
    min_size: int = 10_000

    # Prioritized replay (future extension)
    use_prioritized: bool = False
    priority_alpha: float = 0.6
    priority_beta: float = 0.4

    # Compression
    compress: bool = False

    # Queue sizes
    trajectory_queue_size: int = 1000
    sample_batch_size: int = 256


class DistributedReplayBuffer:
    """Process-safe replay buffer using multiprocessing queues.

    Actors send game trajectories through a queue, and the learner
    samples batches from the accumulated examples.

    Supports:
    - Thread/process-safe insertion and sampling
    - Trajectory-based insertion (maintains game context)
    - Memory-efficient storage with optional compression
    """

    def __init__(
        self,
        config: ReplayBufferConfig | None = None,
        manager: SyncManager | None = None,
    ) -> None:
        """Initialize replay buffer.

        Args:
            config: Buffer configuration.
            manager: Multiprocessing manager.
        """
        self.config = config or ReplayBufferConfig()

        # Create or use existing manager
        if manager is not None:
            self._manager = manager
            self._owns_manager = False
        else:
            self._manager = Manager()
            self._owns_manager = True

        # Trajectory input queue (actors -> buffer)
        self._trajectory_queue: Queue = mp.Queue(
            maxsize=self.config.trajectory_queue_size,
        )

        # Internal buffer storage (managed by learner process)
        self._buffer: deque[TrainingExample] = deque(maxlen=self.config.max_size)
        self._buffer_lock = threading.Lock()

        # Statistics
        self._stats = self._manager.dict()
        self._stats["total_trajectories"] = 0
        self._stats["total_examples"] = 0
        self._stats["buffer_size"] = 0

        logger.info(
            f"DistributedReplayBuffer initialized (max_size={self.config.max_size})",
        )

    def shutdown(self) -> None:
        """Cleanup resources."""
        try:
            self._trajectory_queue.close()
            self._trajectory_queue.join_thread()
        except Exception as e:
            logger.warning(f"Error closing trajectory queue: {e}")

        if self._owns_manager:
            try:
                self._manager.shutdown()
            except Exception as e:
                logger.warning(f"Error shutting down manager: {e}")

    def submit_trajectory(
        self,
        trajectory: GameTrajectory,
        timeout: float | None = None,
    ) -> bool:
        """Submit a game trajectory to the buffer.

        This is called by actor processes.

        Args:
            trajectory: Game trajectory to add.
            timeout: Queue timeout in seconds.

        Returns:
            True if successfully submitted.
        """
        try:
            data: bytes | GameTrajectory
            if self.config.compress:
                data = trajectory.to_bytes(compress=True)
            else:
                data = trajectory

            self._trajectory_queue.put(data, timeout=timeout)
            return True
        except Exception as e:
            logger.warning(f"Failed to submit trajectory: {e}")
            return False

    def process_pending_trajectories(self, max_count: int = 100) -> int:
        """Process pending trajectories from the queue.

        This should be called by the learner process.

        Args:
            max_count: Maximum trajectories to process.

        Returns:
            Number of trajectories processed.
        """
        processed = 0

        while processed < max_count:
            try:
                data = self._trajectory_queue.get_nowait()

                # Decompress if needed
                if self.config.compress:
                    trajectory = GameTrajectory.from_bytes(data, compressed=True)
                else:
                    trajectory = data

                # Add examples to buffer and update stats atomically
                with self._buffer_lock:
                    for example in trajectory.examples:
                        self._buffer.append(example)

                    # Update stats inside lock to prevent race conditions
                    self._stats["total_trajectories"] = self._stats["total_trajectories"] + 1
                    self._stats["total_examples"] = self._stats["total_examples"] + len(
                        trajectory.examples
                    )
                    self._stats["buffer_size"] = len(self._buffer)

                processed += 1

            except Exception:
                # Queue empty or error
                break

        return processed

    def sample(self, batch_size: int) -> list[TrainingExample]:
        """Sample a batch of training examples.

        Args:
            batch_size: Number of examples to sample.

        Returns:
            List of training examples.
        """
        with self._buffer_lock:
            if len(self._buffer) < batch_size:
                batch_size = len(self._buffer)

            if batch_size == 0:
                return []

            indices = np.random.choice(
                len(self._buffer),
                size=batch_size,
                replace=False,
            )
            return [self._buffer[i] for i in indices]

    def sample_batch(
        self,
        batch_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample and collate a batch for training.

        Args:
            batch_size: Batch size (defaults to config).

        Returns:
            Dictionary with batched tensors.
        """
        if batch_size is None:
            batch_size = self.config.sample_batch_size

        examples = self.sample(batch_size)

        if not examples:
            return {}

        return {
            "spatial": torch.stack([torch.from_numpy(ex.spatial).float() for ex in examples]),
            "vector": torch.stack([torch.from_numpy(ex.vector).float() for ex in examples]),
            "action_mask": torch.stack(
                [torch.from_numpy(ex.action_mask).float() for ex in examples]
            ),
            "policy": torch.stack([torch.from_numpy(ex.policy).float() for ex in examples]),
            "value": torch.tensor([ex.value for ex in examples]).float(),
        }

    def size(self) -> int:
        """Get current buffer size."""
        with self._buffer_lock:
            return len(self._buffer)

    def is_ready(self) -> bool:
        """Check if buffer has enough examples for training."""
        return self.size() >= self.config.min_size

    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        return {
            "total_trajectories": self._stats["total_trajectories"],
            "total_examples": self._stats["total_examples"],
            "buffer_size": self._stats["buffer_size"],
            "is_ready": self.is_ready(),
        }

    def clear(self) -> None:
        """Clear the buffer."""
        with self._buffer_lock:
            self._buffer.clear()
        self._stats["buffer_size"] = 0
        logger.info("Replay buffer cleared")


class WeightSynchronizer:
    """Utility for synchronizing model weights between processes.

    Provides helper methods for actors to efficiently check for and
    load updated weights.
    """

    def __init__(
        self,
        shared_storage: SharedStorage,
        update_interval: float = 1.0,
    ) -> None:
        """Initialize weight synchronizer.

        Args:
            shared_storage: Shared storage instance.
            update_interval: Minimum seconds between weight checks.
        """
        self.storage = shared_storage
        self.update_interval = update_interval

        self._current_version = 0
        self._last_check_time = 0.0

    def should_update(self) -> bool:
        """Check if weights should be updated.

        Returns:
            True if new weights are available.
        """
        current_time = time.time()
        if current_time - self._last_check_time < self.update_interval:
            return False

        self._last_check_time = current_time
        latest_version = self.storage.get_model_version()
        return latest_version > self._current_version

    def update_model(self, model: torch.nn.Module) -> bool:
        """Update model with latest weights.

        Args:
            model: Model to update.

        Returns:
            True if model was updated.
        """
        # Use atomic method to get weights and version together
        weights, version = self.storage.get_model_weights_with_version()
        if weights is None:
            return False

        try:
            model.load_state_dict(weights)
            self._current_version = version
            logger.debug(f"Updated to model v{self._current_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return False

    def update_if_available(self, model: torch.nn.Module) -> tuple[bool, int]:
        """Atomically check for and apply weight updates.

        This method combines should_update() and update_model() into a single
        atomic operation to prevent race conditions where version changes
        between check and update.

        Args:
            model: Model to update.

        Returns:
            Tuple of (was_updated, current_version).
        """
        current_time = time.time()
        if current_time - self._last_check_time < self.update_interval:
            return False, self._current_version

        self._last_check_time = current_time

        # Atomic get of weights and version
        weights, latest_version = self.storage.get_model_weights_with_version()

        if weights is None or latest_version <= self._current_version:
            return False, self._current_version

        try:
            model.load_state_dict(weights)
            self._current_version = latest_version
            logger.debug(f"Updated to model v{self._current_version}")
            return True, self._current_version
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return False, self._current_version

    def get_current_version(self) -> int:
        """Get currently loaded model version."""
        return self._current_version
