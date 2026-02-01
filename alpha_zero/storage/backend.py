"""Abstract storage backend interface.

Defines the contract that all storage backends must implement,
enabling seamless switching between local and cloud storage.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator

import torch

if TYPE_CHECKING:
    from alpha_zero.distributed.shared_storage import GameTrajectory
    from alpha_zero.storage.config import StorageConfig
    from alpha_zero.storage.models import (
        CheckpointMetadata,
        EvaluationResult,
        ModelMetadata,
        TrajectoryMetadata,
    )

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    This interface defines all storage operations needed for:
    - Model checkpoint storage and retrieval
    - Game trajectory persistence
    - Evaluation results storage
    - Champion model management

    Implementations must be thread-safe for concurrent access.

    Example:
        class MyStorageBackend(StorageBackend):
            def save_model(self, state_dict, iteration, metadata=None):
                # Implementation
                pass
    """

    @abstractmethod
    def __init__(self, config: StorageConfig) -> None:
        """Initialize the storage backend.

        Args:
            config: Storage configuration.
        """
        ...

    # =========================================================================
    # Model Operations
    # =========================================================================

    @abstractmethod
    def save_model(
        self,
        state_dict: dict[str, torch.Tensor],
        iteration: int,
        metadata: ModelMetadata | None = None,
    ) -> str:
        """Save model weights to storage.

        Args:
            state_dict: Model state dictionary.
            iteration: Training iteration number.
            metadata: Optional model metadata.

        Returns:
            Path/URI where model was saved.
        """
        ...

    @abstractmethod
    def load_model(
        self,
        iteration: int,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """Load model weights from a specific iteration.

        Args:
            iteration: Training iteration to load.
            map_location: Device to load tensors to.

        Returns:
            Tuple of (state_dict, metadata).

        Raises:
            FileNotFoundError: If model doesn't exist.
        """
        ...

    @abstractmethod
    def load_model_by_version(
        self,
        version: int,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """Load model weights by version number.

        Args:
            version: Model version to load.
            map_location: Device to load tensors to.

        Returns:
            Tuple of (state_dict, metadata).

        Raises:
            FileNotFoundError: If version doesn't exist.
        """
        ...

    # =========================================================================
    # Champion Model Operations
    # =========================================================================

    @abstractmethod
    def save_champion(
        self,
        state_dict: dict[str, torch.Tensor],
        metadata: ModelMetadata,
    ) -> str:
        """Save model as the current champion.

        The champion is the best-performing model used for self-play.

        Args:
            state_dict: Model state dictionary.
            metadata: Model metadata including version and metrics.

        Returns:
            Path/URI where champion was saved.
        """
        ...

    @abstractmethod
    def load_champion(
        self,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """Load the current champion model.

        Args:
            map_location: Device to load tensors to.

        Returns:
            Tuple of (state_dict, metadata).

        Raises:
            FileNotFoundError: If no champion exists.
        """
        ...

    @abstractmethod
    def get_champion_version(self) -> int:
        """Get the current champion model version.

        Returns:
            Champion version number, or 0 if no champion exists.
        """
        ...

    @abstractmethod
    def champion_exists(self) -> bool:
        """Check if a champion model exists.

        Returns:
            True if champion exists.
        """
        ...

    # =========================================================================
    # Trajectory Operations
    # =========================================================================

    @abstractmethod
    def save_trajectory(
        self,
        trajectory: GameTrajectory,
        iteration: int,
        worker_id: int,
        game_id: str,
    ) -> str:
        """Save a game trajectory.

        Args:
            trajectory: Game trajectory with training examples.
            iteration: Training iteration.
            worker_id: ID of the worker that generated this trajectory.
            game_id: Unique game identifier.

        Returns:
            Path/URI where trajectory was saved.
        """
        ...

    @abstractmethod
    def list_trajectories(
        self,
        iteration: int | None = None,
        worker_id: int | None = None,
    ) -> list[TrajectoryMetadata]:
        """List available trajectories.

        Args:
            iteration: Filter by iteration (None for all).
            worker_id: Filter by worker ID (None for all).

        Returns:
            List of trajectory metadata.
        """
        ...

    @abstractmethod
    def load_trajectory(
        self,
        iteration: int,
        worker_id: int,
        game_id: str,
    ) -> GameTrajectory:
        """Load a specific trajectory.

        Args:
            iteration: Training iteration.
            worker_id: Worker ID.
            game_id: Game identifier.

        Returns:
            The game trajectory.

        Raises:
            FileNotFoundError: If trajectory doesn't exist.
        """
        ...

    @abstractmethod
    def stream_trajectories(
        self,
        iterations: list[int] | None = None,
        shuffle: bool = True,
    ) -> Iterator[GameTrajectory]:
        """Stream trajectories for training.

        This method enables efficient streaming of trajectories
        without loading all into memory.

        Args:
            iterations: List of iterations to include (None for all).
            shuffle: Whether to shuffle trajectories.

        Yields:
            Game trajectories one at a time.
        """
        ...

    @abstractmethod
    def count_trajectories(
        self,
        iteration: int | None = None,
    ) -> int:
        """Count available trajectories.

        Args:
            iteration: Filter by iteration (None for all).

        Returns:
            Number of trajectories.
        """
        ...

    # =========================================================================
    # Checkpoint Operations
    # =========================================================================

    @abstractmethod
    def save_checkpoint(
        self,
        checkpoint_data: dict[str, Any],
        iteration: int,
        metadata: CheckpointMetadata | None = None,
    ) -> str:
        """Save a full training checkpoint.

        Includes model, optimizer, scheduler, and training state.

        Args:
            checkpoint_data: Full checkpoint dictionary.
            iteration: Training iteration.
            metadata: Optional checkpoint metadata.

        Returns:
            Path/URI where checkpoint was saved.
        """
        ...

    @abstractmethod
    def load_checkpoint(
        self,
        iteration: int,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, Any], CheckpointMetadata]:
        """Load a specific checkpoint.

        Args:
            iteration: Training iteration to load.
            map_location: Device to load tensors to.

        Returns:
            Tuple of (checkpoint_data, metadata).

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
        """
        ...

    @abstractmethod
    def load_latest_checkpoint(
        self,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, Any], CheckpointMetadata] | None:
        """Load the most recent checkpoint.

        Args:
            map_location: Device to load tensors to.

        Returns:
            Tuple of (checkpoint_data, metadata), or None if no checkpoints.
        """
        ...

    @abstractmethod
    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List all available checkpoints.

        Returns:
            List of checkpoint metadata, sorted by iteration.
        """
        ...

    # =========================================================================
    # Evaluation Operations
    # =========================================================================

    @abstractmethod
    def save_evaluation(
        self,
        result: EvaluationResult,
    ) -> str:
        """Save evaluation results.

        Args:
            result: Evaluation result with metrics.

        Returns:
            Path/URI where result was saved.
        """
        ...

    @abstractmethod
    def load_evaluation(
        self,
        iteration: int,
    ) -> EvaluationResult:
        """Load evaluation results for an iteration.

        Args:
            iteration: Training iteration.

        Returns:
            Evaluation result.

        Raises:
            FileNotFoundError: If evaluation doesn't exist.
        """
        ...

    @abstractmethod
    def list_evaluations(self) -> list[EvaluationResult]:
        """List all evaluation results.

        Returns:
            List of evaluation results, sorted by iteration.
        """
        ...

    # =========================================================================
    # Config Operations
    # =========================================================================

    @abstractmethod
    def save_config(
        self,
        config: dict[str, Any],
        iteration: int,
    ) -> str:
        """Save training configuration.

        Args:
            config: Configuration dictionary.
            iteration: Training iteration.

        Returns:
            Path/URI where config was saved.
        """
        ...

    @abstractmethod
    def load_config(
        self,
        iteration: int,
    ) -> dict[str, Any]:
        """Load training configuration.

        Args:
            iteration: Training iteration.

        Returns:
            Configuration dictionary.

        Raises:
            FileNotFoundError: If config doesn't exist.
        """
        ...

    # =========================================================================
    # Utility Operations
    # =========================================================================

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if a path exists in storage.

        Args:
            path: Path to check.

        Returns:
            True if path exists.
        """
        ...

    @abstractmethod
    def delete(self, path: str) -> bool:
        """Delete a file from storage.

        Args:
            path: Path to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    def list_files(
        self,
        prefix: str,
        pattern: str | None = None,
    ) -> list[str]:
        """List files matching a prefix and optional pattern.

        Args:
            prefix: Path prefix to search.
            pattern: Optional glob pattern (e.g., "*.pt").

        Returns:
            List of matching file paths.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the storage backend and release resources."""
        ...

    def __enter__(self) -> StorageBackend:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


__all__ = ["StorageBackend"]
