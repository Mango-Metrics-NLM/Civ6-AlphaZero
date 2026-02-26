"""SpacetimeDB storage backend implementation.

Implements the StorageBackend ABC using SpacetimeDB for persistence,
enabling real-time distributed storage with subscription-based updates.

Follows the same patterns as LocalStorageBackend and GCSStorageBackend.
"""

from __future__ import annotations

import logging
import random
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from alpha_zero.spacetimedb.client import SpacetimeDBClient
from alpha_zero.spacetimedb.config import SpacetimeDBConfig
from alpha_zero.spacetimedb.exceptions import SpacetimeDBConnectionError
from alpha_zero.spacetimedb.serialization import (
    deserialize_checkpoint,
    deserialize_config,
    deserialize_state_dict,
    serialize_checkpoint,
    serialize_config,
    serialize_state_dict,
)
from alpha_zero.storage.backend import StorageBackend
from alpha_zero.storage.models import (
    CheckpointMetadata,
    EvaluationResult,
    ModelMetadata,
    TrajectoryMetadata,
)

if TYPE_CHECKING:
    from alpha_zero.distributed.shared_storage import GameTrajectory
    from alpha_zero.storage.config import StorageConfig

logger = logging.getLogger(__name__)


class SpacetimeDBStorageBackend(StorageBackend):
    """Storage backend using SpacetimeDB for persistence.

    Implements all StorageBackend abstract methods using SpacetimeDBClient
    for distributed real-time storage. Supports push-based notifications
    when data changes through SpacetimeDB subscriptions.

    Thread-safe with RLock (matching LocalStorageBackend pattern).

    Usage:
        from alpha_zero.storage import create_storage_backend, StorageConfig

        config = StorageConfig(backend_type="spacetimedb")
        storage = create_storage_backend(config)
        storage.save_model(state_dict, iteration=1)
    """

    def __init__(self, config: StorageConfig) -> None:
        """Initialize SpacetimeDB storage backend.

        Args:
            config: Storage configuration with spacetimedb settings.
        """
        self._config = config
        self._lock = threading.RLock()

        # Extract SpacetimeDB config
        spacetimedb_config = getattr(config, "spacetimedb", None)
        if spacetimedb_config is None:
            spacetimedb_config = SpacetimeDBConfig.from_env()

        self._stdb_config = spacetimedb_config
        self._client = SpacetimeDBClient(spacetimedb_config)

        # Version tracking
        self._model_version_counter: int = 0
        self._champion_version: int = 0

        # In-memory indices (populated from subscriptions or queries)
        self._model_metadata: dict[int, ModelMetadata] = {}
        self._trajectory_metadata: dict[str, TrajectoryMetadata] = {}
        self._checkpoint_metadata: dict[int, CheckpointMetadata] = {}
        self._evaluation_results: dict[int, EvaluationResult] = {}

        logger.info(
            "SpacetimeDB storage backend initialized (host=%s, module=%s)",
            spacetimedb_config.host,
            spacetimedb_config.module_address,
        )

    @property
    def client(self) -> SpacetimeDBClient:
        """Access the underlying SpacetimeDB client."""
        return self._client

    def connect(self, mock_client: Any = None) -> bool:
        """Connect to SpacetimeDB.

        Args:
            mock_client: Optional mock client for testing.

        Returns:
            True if connected successfully.
        """
        if mock_client is not None:
            return self._client.connect_with_mock(mock_client)
        return False  # Async connect not available in sync context

    # =========================================================================
    # Model Operations
    # =========================================================================

    def save_model(
        self,
        state_dict: dict[str, torch.Tensor],
        iteration: int,
        metadata: ModelMetadata | None = None,
    ) -> str:
        """Save model weights to SpacetimeDB."""
        with self._lock:
            self._model_version_counter += 1
            version = self._model_version_counter

        if metadata is None:
            metadata = ModelMetadata(
                iteration=iteration,
                version=version,
            )

        compress = self._config.compress_weights
        weights_bytes = serialize_state_dict(state_dict, compress=compress)
        metadata_dict = metadata.to_dict()

        success = self._client.publish_weights_sync(
            iteration=iteration,
            version=version,
            weights_bytes=weights_bytes,
            metadata=metadata_dict,
        )

        if success:
            with self._lock:
                self._model_metadata[iteration] = metadata
            logger.info("Saved model iteration=%d, version=%d", iteration, version)
            return f"spacetimedb://models/{iteration}"

        raise SpacetimeDBConnectionError(
            f"Failed to save model for iteration {iteration}"
        )

    def load_model(
        self,
        iteration: int,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """Load model weights from SpacetimeDB."""
        with self._lock:
            metadata = self._model_metadata.get(iteration)

        if metadata is None:
            raise FileNotFoundError(
                f"No model found for iteration {iteration}"
            )

        result = self._client.get_weights_by_version_sync(metadata.version)
        if result is None:
            raise FileNotFoundError(
                f"Model weights not found for version {metadata.version}"
            )

        weights_bytes, _meta_dict = result
        compress = self._config.compress_weights
        state_dict = deserialize_state_dict(
            weights_bytes, compressed=compress, map_location=map_location,
        )
        return state_dict, metadata

    def load_model_by_version(
        self,
        version: int,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """Load model weights by version number."""
        result = self._client.get_weights_by_version_sync(version)
        if result is None:
            raise FileNotFoundError(
                f"Model weights not found for version {version}"
            )

        weights_bytes, meta_dict = result
        compress = self._config.compress_weights
        state_dict = deserialize_state_dict(
            weights_bytes, compressed=compress, map_location=map_location,
        )

        metadata = ModelMetadata.from_dict(meta_dict) if meta_dict else ModelMetadata(
            iteration=0, version=version,
        )
        return state_dict, metadata

    # =========================================================================
    # Champion Model Operations
    # =========================================================================

    def save_champion(
        self,
        state_dict: dict[str, torch.Tensor],
        metadata: ModelMetadata,
    ) -> str:
        """Save model as the current champion."""
        compress = self._config.compress_weights
        weights_bytes = serialize_state_dict(state_dict, compress=compress)

        success = self._client.publish_weights_sync(
            iteration=metadata.iteration,
            version=metadata.version,
            weights_bytes=weights_bytes,
            metadata=metadata.to_dict(),
            is_champion=True,
        )

        if success:
            with self._lock:
                self._champion_version = metadata.version
                self._model_metadata[metadata.iteration] = metadata
            logger.info("Saved champion model version=%d", metadata.version)
            return f"spacetimedb://champion/{metadata.version}"

        raise SpacetimeDBConnectionError("Failed to save champion model")

    def load_champion(
        self,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """Load the current champion model."""
        result = self._client.get_champion_weights_sync()
        if result is None:
            raise FileNotFoundError("No champion model found")

        weights_bytes, meta_dict = result
        compress = self._config.compress_weights
        state_dict = deserialize_state_dict(
            weights_bytes, compressed=compress, map_location=map_location,
        )

        metadata = ModelMetadata.from_dict(meta_dict) if meta_dict else ModelMetadata(
            iteration=0, version=self._champion_version,
        )
        return state_dict, metadata

    def get_champion_version(self) -> int:
        """Get the current champion model version."""
        return self._client.champion_version or self._champion_version

    def champion_exists(self) -> bool:
        """Check if a champion model exists."""
        return self.get_champion_version() > 0

    # =========================================================================
    # Trajectory Operations
    # =========================================================================

    def save_trajectory(
        self,
        trajectory: GameTrajectory,
        iteration: int,
        worker_id: int,
        game_id: str,
    ) -> str:
        """Save a game trajectory to SpacetimeDB."""
        trajectory_bytes = trajectory.to_bytes(compress=False)
        num_examples = len(trajectory.examples) if hasattr(trajectory, "examples") else 0

        success = self._client.submit_trajectory_sync(
            iteration=iteration,
            worker_id=worker_id,
            game_id=game_id,
            trajectory_bytes=trajectory_bytes,
            num_examples=num_examples,
        )

        if success:
            key = f"{iteration}_{worker_id}_{game_id}"
            meta = TrajectoryMetadata(
                game_id=game_id,
                worker_id=worker_id,
                iteration=iteration,
                num_examples=num_examples,
                game_length=trajectory.game_length if hasattr(trajectory, "game_length") else 0,
            )
            with self._lock:
                self._trajectory_metadata[key] = meta
            return f"spacetimedb://trajectories/{key}"

        raise SpacetimeDBConnectionError(
            f"Failed to save trajectory for game {game_id}"
        )

    def list_trajectories(
        self,
        iteration: int | None = None,
        worker_id: int | None = None,
    ) -> list[TrajectoryMetadata]:
        """List available trajectories."""
        with self._lock:
            results = []
            for meta in self._trajectory_metadata.values():
                if iteration is not None and meta.iteration != iteration:
                    continue
                if worker_id is not None and meta.worker_id != worker_id:
                    continue
                results.append(meta)
            return results

    def load_trajectory(
        self,
        iteration: int,
        worker_id: int,
        game_id: str,
    ) -> GameTrajectory:
        """Load a specific trajectory from SpacetimeDB."""
        from alpha_zero.spacetimedb.serialization import deserialize_trajectory

        # Retrieve from mock/SDK client
        sdk = self._client._connection.sdk_client
        if sdk is not None and hasattr(sdk, "get_trajectory"):
            result = sdk.get_trajectory(iteration, worker_id, game_id)
            if result is not None:
                trajectory_bytes, _ = result
                return deserialize_trajectory(trajectory_bytes, compressed=False)

        raise FileNotFoundError(
            f"Trajectory not found: {iteration}_{worker_id}_{game_id}"
        )

    def stream_trajectories(
        self,
        iterations: list[int] | None = None,
        shuffle: bool = True,
    ) -> Iterator[GameTrajectory]:
        """Stream trajectories for training."""
        meta_list = self.list_trajectories()

        if iterations is not None:
            meta_list = [m for m in meta_list if m.iteration in iterations]

        if shuffle:
            random.shuffle(meta_list)

        for meta in meta_list:
            try:
                yield self.load_trajectory(
                    meta.iteration, meta.worker_id, meta.game_id,
                )
            except FileNotFoundError:
                logger.warning(
                    "Trajectory not found during streaming: %s",
                    meta.game_id,
                )

    def count_trajectories(
        self,
        iteration: int | None = None,
    ) -> int:
        """Count available trajectories."""
        return len(self.list_trajectories(iteration=iteration))

    # =========================================================================
    # Checkpoint Operations
    # =========================================================================

    def save_checkpoint(
        self,
        checkpoint_data: dict[str, Any],
        iteration: int,
        metadata: CheckpointMetadata | None = None,
    ) -> str:
        """Save a full training checkpoint."""
        compress = self._config.compress_weights
        data_bytes = serialize_checkpoint(checkpoint_data, compress=compress)

        if metadata is None:
            metadata = CheckpointMetadata(iteration=iteration)

        success = self._client.save_checkpoint_sync(
            iteration=iteration,
            data=data_bytes,
            metadata=metadata.to_dict(),
        )

        if success:
            with self._lock:
                self._checkpoint_metadata[iteration] = metadata
            logger.info("Saved checkpoint iteration=%d", iteration)
            return f"spacetimedb://checkpoints/{iteration}"

        raise SpacetimeDBConnectionError(
            f"Failed to save checkpoint for iteration {iteration}"
        )

    def load_checkpoint(
        self,
        iteration: int,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, Any], CheckpointMetadata]:
        """Load a specific checkpoint."""
        sdk = self._client._connection.sdk_client
        if sdk is not None and hasattr(sdk, "load_checkpoint"):
            result = sdk.load_checkpoint(iteration)
            if result is not None:
                data_bytes, _meta_dict = result
                compress = self._config.compress_weights
                checkpoint_data = deserialize_checkpoint(
                    data_bytes, compressed=compress, map_location=map_location,
                )
                with self._lock:
                    metadata = self._checkpoint_metadata.get(
                        iteration,
                        CheckpointMetadata(iteration=iteration),
                    )
                return checkpoint_data, metadata

        raise FileNotFoundError(
            f"Checkpoint not found for iteration {iteration}"
        )

    def load_latest_checkpoint(
        self,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, Any], CheckpointMetadata] | None:
        """Load the most recent checkpoint."""
        sdk = self._client._connection.sdk_client
        if sdk is not None and hasattr(sdk, "load_latest_checkpoint"):
            result = sdk.load_latest_checkpoint()
            if result is not None:
                iteration, data_bytes, _meta_dict = result
                compress = self._config.compress_weights
                checkpoint_data = deserialize_checkpoint(
                    data_bytes, compressed=compress, map_location=map_location,
                )
                with self._lock:
                    metadata = self._checkpoint_metadata.get(
                        iteration,
                        CheckpointMetadata(iteration=iteration),
                    )
                return checkpoint_data, metadata
        return None

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List all available checkpoints."""
        with self._lock:
            return sorted(
                self._checkpoint_metadata.values(),
                key=lambda m: m.iteration,
            )

    # =========================================================================
    # Evaluation Operations
    # =========================================================================

    def save_evaluation(
        self,
        result: EvaluationResult,
    ) -> str:
        """Save evaluation results."""
        result_dict = result.to_dict()
        success = self._client.save_evaluation_sync(result_dict)

        if success:
            with self._lock:
                self._evaluation_results[result.iteration] = result
            return f"spacetimedb://evaluations/{result.iteration}"

        raise SpacetimeDBConnectionError(
            f"Failed to save evaluation for iteration {result.iteration}"
        )

    def load_evaluation(
        self,
        iteration: int,
    ) -> EvaluationResult:
        """Load evaluation results for an iteration."""
        with self._lock:
            result = self._evaluation_results.get(iteration)

        if result is None:
            raise FileNotFoundError(
                f"Evaluation not found for iteration {iteration}"
            )
        return result

    def list_evaluations(self) -> list[EvaluationResult]:
        """List all evaluation results."""
        with self._lock:
            return sorted(
                self._evaluation_results.values(),
                key=lambda r: r.iteration,
            )

    # =========================================================================
    # Config Operations
    # =========================================================================

    def save_config(
        self,
        config: dict[str, Any],
        iteration: int,
    ) -> str:
        """Save training configuration."""
        config_json = serialize_config(config)
        success = self._client.save_config_sync(iteration, config_json)

        if success:
            return f"spacetimedb://configs/{iteration}"

        raise SpacetimeDBConnectionError(
            f"Failed to save config for iteration {iteration}"
        )

    def load_config(
        self,
        iteration: int,
    ) -> dict[str, Any]:
        """Load training configuration."""
        sdk = self._client._connection.sdk_client
        if sdk is not None and hasattr(sdk, "load_config"):
            result = sdk.load_config(iteration)
            if result is not None:
                return deserialize_config(result)

        raise FileNotFoundError(
            f"Config not found for iteration {iteration}"
        )

    # =========================================================================
    # Utility Operations
    # =========================================================================

    def exists(self, path: str) -> bool:
        """Check if a path exists in storage."""
        if path.startswith("spacetimedb://models/"):
            try:
                iteration = int(path.split("/")[-1])
                with self._lock:
                    return iteration in self._model_metadata
            except (ValueError, IndexError):
                return False
        if path.startswith("spacetimedb://trajectories/"):
            key = path.split("/")[-1]
            with self._lock:
                return key in self._trajectory_metadata
        if path.startswith("spacetimedb://checkpoints/"):
            try:
                iteration = int(path.split("/")[-1])
                with self._lock:
                    return iteration in self._checkpoint_metadata
            except (ValueError, IndexError):
                return False
        return False

    def delete(self, path: str) -> bool:
        """Delete a file from storage."""
        logger.warning(
            "SpacetimeDB storage backend does not support direct deletion: %s",
            path,
        )
        return False

    def list_files(
        self,
        prefix: str,
        pattern: str | None = None,
    ) -> list[str]:
        """List files matching a prefix."""
        results: list[str] = []
        if "models" in prefix:
            with self._lock:
                results = [
                    f"spacetimedb://models/{it}" for it in self._model_metadata
                ]
        elif "trajectories" in prefix:
            with self._lock:
                results = [
                    f"spacetimedb://trajectories/{k}"
                    for k in self._trajectory_metadata
                ]
        elif "checkpoints" in prefix:
            with self._lock:
                results = [
                    f"spacetimedb://checkpoints/{it}"
                    for it in self._checkpoint_metadata
                ]
        return results

    def close(self) -> None:
        """Close the storage backend and release resources."""
        self._client.disconnect_sync()
        logger.info("SpacetimeDB storage backend closed")


__all__ = ["SpacetimeDBStorageBackend"]
