"""Local filesystem storage backend.

Implements the StorageBackend interface for local filesystem storage,
suitable for development and single-machine training.
"""

from __future__ import annotations

import io
import json
import logging
import lzma
import pickle
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from alpha_zero.storage.backend import StorageBackend
from alpha_zero.storage.config import StorageConfig
from alpha_zero.storage.models import (
    CheckpointMetadata,
    EvaluationResult,
    ModelMetadata,
    TrajectoryMetadata,
)
from alpha_zero.utils.compat import UTC

if TYPE_CHECKING:
    from alpha_zero.distributed.shared_storage import GameTrajectory

logger = logging.getLogger(__name__)


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend.

    Implements storage operations using the local filesystem.
    Thread-safe for concurrent access within a single process.

    Directory structure:
        {base_path}/
        ├── models/
        │   ├── champion/
        │   │   ├── model.pt
        │   │   └── metadata.json
        │   └── iteration_{n}/
        │       ├── model.pt
        │       └── metadata.json
        ├── games/
        │   └── iteration_{n}/
        │       └── worker_{id}/
        │           └── {game_id}.pkl
        ├── checkpoints/
        │   └── iteration_{n}/
        │       └── checkpoint.pt
        ├── eval/
        │   └── iteration_{n}/
        │       └── results.json
        └── configs/
            └── iteration_{n}/
                └── config.yaml

    Attributes:
        config: Storage configuration.
        base_path: Base directory for all storage.
    """

    def __init__(self, config: StorageConfig) -> None:
        """Initialize local storage backend.

        Args:
            config: Storage configuration.
        """
        self.config = config
        self.base_path = Path(config.base_path)
        self._lock = threading.RLock()

        # Create directory structure
        self._ensure_directories()

        logger.info(f"LocalStorageBackend initialized at {self.base_path}")

    def _ensure_directories(self) -> None:
        """Create required directory structure."""
        directories = [
            self.base_path / self.config.models_subdir,
            self.base_path / self.config.models_subdir / self.config.champion_subdir,
            self.base_path / self.config.games_subdir,
            self.base_path / self.config.checkpoints_subdir,
            self.base_path / self.config.eval_subdir,
            self.base_path / self.config.configs_subdir,
            self.base_path / self.config.logs_subdir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(UTC).isoformat()

    def _serialize_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> bytes:
        """Serialize model weights to bytes."""
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        data = buffer.getvalue()
        if self.config.compress_weights:
            data = lzma.compress(data)
        return data

    def _deserialize_weights(
        self,
        data: bytes,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Deserialize model weights from bytes."""
        if self.config.compress_weights:
            data = lzma.decompress(data)
        buffer = io.BytesIO(data)
        return cast(
            dict[str, torch.Tensor],
            torch.load(buffer, map_location=map_location, weights_only=True),
        )

    # =========================================================================
    # Model Operations
    # =========================================================================

    def save_model(
        self,
        state_dict: dict[str, torch.Tensor],
        iteration: int,
        metadata: ModelMetadata | None = None,
    ) -> str:
        """Save model weights to storage."""
        with self._lock:
            model_dir = self.base_path / self.config.models_subdir / f"iteration_{iteration:08d}"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model weights
            model_path = model_dir / "model.pt"
            weights_bytes = self._serialize_weights(state_dict)
            model_path.write_bytes(weights_bytes)

            # Save metadata
            if metadata is None:
                metadata = ModelMetadata(
                    iteration=iteration,
                    version=iteration,
                    timestamp=self._get_timestamp(),
                )

            metadata_path = model_dir / "metadata.json"
            metadata_path.write_text(metadata.to_json())

            logger.debug(f"Saved model at iteration {iteration}: {model_path}")
            return str(model_path)

    def load_model(
        self,
        iteration: int,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """Load model weights from a specific iteration."""
        model_dir = self.base_path / self.config.models_subdir / f"iteration_{iteration:08d}"
        model_path = model_dir / "model.pt"
        metadata_path = model_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at iteration {iteration}")

        weights_bytes = model_path.read_bytes()
        state_dict = self._deserialize_weights(weights_bytes, map_location)

        if metadata_path.exists():
            metadata = ModelMetadata.from_json(metadata_path.read_text())
        else:
            metadata = ModelMetadata(iteration=iteration, version=iteration)

        logger.debug(f"Loaded model from iteration {iteration}")
        return state_dict, metadata

    def load_model_by_version(
        self,
        version: int,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """Load model weights by version number."""
        # In local storage, version typically equals iteration
        # Search through models to find matching version
        models_dir = self.base_path / self.config.models_subdir
        for model_dir in models_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == self.config.champion_subdir:
                continue

            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                metadata = ModelMetadata.from_json(metadata_path.read_text())
                if metadata.version == version:
                    model_path = model_dir / "model.pt"
                    weights_bytes = model_path.read_bytes()
                    state_dict = self._deserialize_weights(weights_bytes, map_location)
                    return state_dict, metadata

        raise FileNotFoundError(f"Model version {version} not found")

    # =========================================================================
    # Champion Model Operations
    # =========================================================================

    def save_champion(
        self,
        state_dict: dict[str, torch.Tensor],
        metadata: ModelMetadata,
    ) -> str:
        """Save model as the current champion."""
        with self._lock:
            champion_dir = self.base_path / self.config.models_subdir / self.config.champion_subdir
            champion_dir.mkdir(parents=True, exist_ok=True)

            # Save model weights
            model_path = champion_dir / "model.pt"
            weights_bytes = self._serialize_weights(state_dict)
            model_path.write_bytes(weights_bytes)

            # Update metadata timestamp
            metadata.timestamp = self._get_timestamp()

            # Save metadata
            metadata_path = champion_dir / "metadata.json"
            metadata_path.write_text(metadata.to_json())

            logger.info(f"Saved champion model v{metadata.version}")
            return str(model_path)

    def load_champion(
        self,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """Load the current champion model."""
        champion_dir = self.base_path / self.config.models_subdir / self.config.champion_subdir
        model_path = champion_dir / "model.pt"
        metadata_path = champion_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError("No champion model found")

        weights_bytes = model_path.read_bytes()
        state_dict = self._deserialize_weights(weights_bytes, map_location)

        if metadata_path.exists():
            metadata = ModelMetadata.from_json(metadata_path.read_text())
        else:
            metadata = ModelMetadata(iteration=0, version=0)

        logger.debug(f"Loaded champion model v{metadata.version}")
        return state_dict, metadata

    def get_champion_version(self) -> int:
        """Get the current champion model version."""
        champion_dir = self.base_path / self.config.models_subdir / self.config.champion_subdir
        metadata_path = champion_dir / "metadata.json"

        if not metadata_path.exists():
            return 0

        metadata = ModelMetadata.from_json(metadata_path.read_text())
        return metadata.version

    def champion_exists(self) -> bool:
        """Check if a champion model exists."""
        champion_dir = self.base_path / self.config.models_subdir / self.config.champion_subdir
        return (champion_dir / "model.pt").exists()

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
        """Save a game trajectory."""
        with self._lock:
            trajectory_dir = (
                self.base_path
                / self.config.games_subdir
                / f"iteration_{iteration:08d}"
                / f"worker_{worker_id:04d}"
            )
            trajectory_dir.mkdir(parents=True, exist_ok=True)

            trajectory_path = trajectory_dir / f"{game_id}.pkl"

            # Serialize trajectory
            if self.config.compress_trajectories:
                data = trajectory.to_bytes(compress=True)
            else:
                data = pickle.dumps(trajectory)

            trajectory_path.write_bytes(data)

            # Save metadata alongside
            metadata = TrajectoryMetadata(
                game_id=game_id,
                worker_id=worker_id,
                iteration=iteration,
                game_length=trajectory.game_length,
                winner=trajectory.winner,
                model_version=0,  # Would need to be passed in
                num_examples=len(trajectory.examples),
            )
            metadata_path = trajectory_dir / f"{game_id}.meta.json"
            metadata_path.write_text(metadata.to_json())

            logger.debug(f"Saved trajectory {game_id} (iter={iteration}, worker={worker_id})")
            return str(trajectory_path)

    def list_trajectories(
        self,
        iteration: int | None = None,
        worker_id: int | None = None,
    ) -> list[TrajectoryMetadata]:
        """List available trajectories."""
        games_dir = self.base_path / self.config.games_subdir
        metadata_list: list[TrajectoryMetadata] = []

        if not games_dir.exists():
            return metadata_list

        # Determine which iteration directories to scan
        if iteration is not None:
            iter_dirs = [games_dir / f"iteration_{iteration:08d}"]
        else:
            iter_dirs = [d for d in games_dir.iterdir() if d.is_dir()]

        for iter_dir in iter_dirs:
            if not iter_dir.exists():
                continue

            # Determine which worker directories to scan
            if worker_id is not None:
                worker_dirs = [iter_dir / f"worker_{worker_id:04d}"]
            else:
                worker_dirs = [d for d in iter_dir.iterdir() if d.is_dir()]

            for worker_dir in worker_dirs:
                if not worker_dir.exists():
                    continue

                for meta_file in worker_dir.glob("*.meta.json"):
                    try:
                        metadata = TrajectoryMetadata.from_json(meta_file.read_text())
                        metadata_list.append(metadata)
                    except Exception as e:
                        logger.warning(f"Failed to load metadata {meta_file}: {e}")

        return sorted(metadata_list, key=lambda x: (x.iteration, x.worker_id, x.game_id))

    def load_trajectory(
        self,
        iteration: int,
        worker_id: int,
        game_id: str,
    ) -> GameTrajectory:
        """Load a specific trajectory."""
        # Import here to avoid circular import
        from alpha_zero.distributed.shared_storage import GameTrajectory

        trajectory_path = (
            self.base_path
            / self.config.games_subdir
            / f"iteration_{iteration:08d}"
            / f"worker_{worker_id:04d}"
            / f"{game_id}.pkl"
        )

        if not trajectory_path.exists():
            raise FileNotFoundError(
                f"Trajectory not found: iter={iteration}, worker={worker_id}, game={game_id}"
            )

        data = trajectory_path.read_bytes()

        if self.config.compress_trajectories:
            trajectory = GameTrajectory.from_bytes(data, compressed=True)
        else:
            trajectory = pickle.loads(data)  # noqa: S301

        return trajectory

    def stream_trajectories(
        self,
        iterations: list[int] | None = None,
        shuffle: bool = True,
    ) -> Iterator[GameTrajectory]:
        """Stream trajectories for training."""
        metadata_list = []

        if iterations is None:
            metadata_list = self.list_trajectories()
        else:
            for iteration in iterations:
                metadata_list.extend(self.list_trajectories(iteration=iteration))

        if shuffle:
            np.random.shuffle(metadata_list)

        for meta in metadata_list:
            try:
                trajectory = self.load_trajectory(
                    iteration=meta.iteration,
                    worker_id=meta.worker_id,
                    game_id=meta.game_id,
                )
                yield trajectory
            except Exception as e:
                logger.warning(f"Failed to load trajectory {meta.game_id}: {e}")

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
        with self._lock:
            checkpoint_dir = (
                self.base_path / self.config.checkpoints_subdir / f"iteration_{iteration:08d}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Save checkpoint
            checkpoint_path = checkpoint_dir / "checkpoint.pt"
            torch.save(checkpoint_data, checkpoint_path)

            # Save metadata
            if metadata is None:
                metadata = CheckpointMetadata(
                    iteration=iteration,
                    timestamp=self._get_timestamp(),
                )

            metadata_path = checkpoint_dir / "metadata.json"
            metadata_path.write_text(metadata.to_json())

            logger.info(f"Saved checkpoint at iteration {iteration}")
            return str(checkpoint_path)

    def load_checkpoint(
        self,
        iteration: int,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, Any], CheckpointMetadata]:
        """Load a specific checkpoint."""
        checkpoint_dir = (
            self.base_path / self.config.checkpoints_subdir / f"iteration_{iteration:08d}"
        )
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        metadata_path = checkpoint_dir / "metadata.json"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at iteration {iteration}")

        checkpoint_data = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        if metadata_path.exists():
            metadata = CheckpointMetadata.from_json(metadata_path.read_text())
        else:
            metadata = CheckpointMetadata(iteration=iteration)

        logger.info(f"Loaded checkpoint from iteration {iteration}")
        return checkpoint_data, metadata

    def load_latest_checkpoint(
        self,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, Any], CheckpointMetadata] | None:
        """Load the most recent checkpoint."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda x: x.iteration)
        return self.load_checkpoint(latest.iteration, map_location)

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List all available checkpoints."""
        checkpoints_dir = self.base_path / self.config.checkpoints_subdir
        metadata_list: list[CheckpointMetadata] = []

        if not checkpoints_dir.exists():
            return metadata_list

        for checkpoint_dir in checkpoints_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue

            metadata_path = checkpoint_dir / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = CheckpointMetadata.from_json(metadata_path.read_text())
                    metadata_list.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to load metadata {metadata_path}: {e}")

        return sorted(metadata_list, key=lambda x: x.iteration)

    # =========================================================================
    # Evaluation Operations
    # =========================================================================

    def save_evaluation(
        self,
        result: EvaluationResult,
    ) -> str:
        """Save evaluation results."""
        with self._lock:
            eval_dir = (
                self.base_path / self.config.eval_subdir / f"iteration_{result.iteration:08d}"
            )
            eval_dir.mkdir(parents=True, exist_ok=True)

            result_path = eval_dir / "results.json"
            result_path.write_text(result.to_json())

            logger.info(
                f"Saved evaluation at iteration {result.iteration} "
                f"(win_rate={result.win_rate:.2%}, promoted={result.promoted})"
            )
            return str(result_path)

    def load_evaluation(
        self,
        iteration: int,
    ) -> EvaluationResult:
        """Load evaluation results for an iteration."""
        result_path = (
            self.base_path / self.config.eval_subdir / f"iteration_{iteration:08d}" / "results.json"
        )

        if not result_path.exists():
            raise FileNotFoundError(f"Evaluation not found at iteration {iteration}")

        return EvaluationResult.from_json(result_path.read_text())

    def list_evaluations(self) -> list[EvaluationResult]:
        """List all evaluation results."""
        eval_dir = self.base_path / self.config.eval_subdir
        results: list[EvaluationResult] = []

        if not eval_dir.exists():
            return results

        for iter_dir in eval_dir.iterdir():
            if not iter_dir.is_dir():
                continue

            result_path = iter_dir / "results.json"
            if result_path.exists():
                try:
                    result = EvaluationResult.from_json(result_path.read_text())
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to load evaluation {result_path}: {e}")

        return sorted(results, key=lambda x: x.iteration)

    # =========================================================================
    # Config Operations
    # =========================================================================

    def save_config(
        self,
        config: dict[str, Any],
        iteration: int,
    ) -> str:
        """Save training configuration."""
        with self._lock:
            config_dir = self.base_path / self.config.configs_subdir / f"iteration_{iteration:08d}"
            config_dir.mkdir(parents=True, exist_ok=True)

            config_path = config_dir / "config.json"
            config_path.write_text(json.dumps(config, indent=2))

            logger.debug(f"Saved config at iteration {iteration}")
            return str(config_path)

    def load_config(
        self,
        iteration: int,
    ) -> dict[str, Any]:
        """Load training configuration."""
        config_path = (
            self.base_path
            / self.config.configs_subdir
            / f"iteration_{iteration:08d}"
            / "config.json"
        )

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at iteration {iteration}")

        return cast(dict[str, Any], json.loads(config_path.read_text()))

    # =========================================================================
    # Utility Operations
    # =========================================================================

    def exists(self, path: str) -> bool:
        """Check if a path exists in storage."""
        return Path(path).exists()

    def delete(self, path: str) -> bool:
        """Delete a file from storage."""
        path_obj = Path(path)
        if not path_obj.exists():
            return False

        with self._lock:
            path_obj.unlink()
            logger.debug(f"Deleted: {path}")
            return True

    def list_files(
        self,
        prefix: str,
        pattern: str | None = None,
    ) -> list[str]:
        """List files matching a prefix and optional pattern.

        Uses recursive glob to search subdirectories.
        """
        prefix_path = Path(prefix)
        if not prefix_path.exists():
            return []

        if pattern:
            # Use rglob to search recursively
            # Prepend **/ to match in subdirectories
            if not pattern.startswith("**/"):
                pattern = f"**/{pattern}"
            files = list(prefix_path.glob(pattern))
        else:
            files = list(prefix_path.rglob("*"))

        return [str(f) for f in files if f.is_file()]

    def close(self) -> None:
        """Close the storage backend."""
        logger.debug("LocalStorageBackend closed")


__all__ = ["LocalStorageBackend"]
