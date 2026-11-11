"""Google Cloud Storage backend.

Implements the StorageBackend interface for GCS, enabling cloud-native
storage for distributed training on Vertex AI.

Requirements:
    pip install google-cloud-storage gcsfs

Environment Variables:
    GOOGLE_APPLICATION_CREDENTIALS: Path to service account key file
    GOOGLE_CLOUD_PROJECT: GCP project ID
"""

from __future__ import annotations

import io
import json
import logging
import lzma
import pickle
from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from alpha_zero.local.isolation import guard_gcs
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


# Optional imports with graceful degradation
try:
    from google.api_core import exceptions as gcs_exceptions
    from google.api_core import retry as gcs_retry
    from google.cloud import storage as gcs

    HAS_GCS = True
except ImportError:
    HAS_GCS = False
    gcs = None
    gcs_exceptions = None  # type: ignore[assignment]
    gcs_retry = None  # type: ignore[assignment]


class GCSStorageBackend(StorageBackend):
    """Google Cloud Storage backend.

    Implements storage operations using GCS for cloud-native distributed
    training on Vertex AI.

    Directory structure in bucket:
        {prefix}/
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
        bucket_name: GCS bucket name.
        prefix: Path prefix within bucket.
    """

    def __init__(self, config: StorageConfig) -> None:
        """Initialize GCS storage backend.

        Args:
            config: Storage configuration with GCS settings.

        Raises:
            ImportError: If google-cloud-storage is not installed.
            ValueError: If GCS bucket is not configured.
        """
        guard_gcs("GCSStorageBackend")
        if not HAS_GCS:
            raise ImportError(
                "google-cloud-storage is required for GCS backend. "
                "Install with: pip install google-cloud-storage"
            )

        self.config = config
        self.bucket_name = config.gcs.bucket
        self.prefix = config.gcs.prefix.strip("/")

        if not self.bucket_name:
            raise ValueError("GCS bucket name is required. Set CIV6_GCS_BUCKET env var.")

        # Initialize GCS client
        self._client = gcs.Client(project=config.gcs.project)
        self._bucket = self._client.bucket(self.bucket_name)

        # Retry configuration
        self._retry = gcs_retry.Retry(
            initial=config.gcs.retry_initial_delay,
            maximum=config.gcs.retry_max_delay,
            multiplier=2.0,
            deadline=config.gcs.timeout * config.gcs.retry_max_attempts,
        )

        logger.info(f"GCSStorageBackend initialized: gs://{self.bucket_name}/{self.prefix}")

    def _get_timestamp(self) -> str:
        """Get current ISO timestamp."""
        return datetime.now(UTC).isoformat()

    def _gcs_path(self, *parts: str) -> str:
        """Build a GCS path from parts."""
        return "/".join([self.prefix] + [p.strip("/") for p in parts if p])

    def _upload_bytes(
        self,
        data: bytes,
        blob_path: str,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload bytes to GCS with retry."""
        blob = self._bucket.blob(blob_path)
        blob.upload_from_string(data, content_type=content_type, retry=self._retry)

    def _upload_json(self, data: dict[str, Any], blob_path: str) -> None:
        """Upload JSON to GCS."""
        json_bytes = json.dumps(data, indent=2).encode("utf-8")
        self._upload_bytes(json_bytes, blob_path, content_type="application/json")

    def _download_bytes(self, blob_path: str) -> bytes:
        """Download bytes from GCS with retry."""
        blob = self._bucket.blob(blob_path)
        return cast(bytes, blob.download_as_bytes(retry=self._retry))

    def _download_json(self, blob_path: str) -> dict[str, Any]:
        """Download and parse JSON from GCS."""
        data = self._download_bytes(blob_path)
        return cast(dict[str, Any], json.loads(data.decode("utf-8")))

    def _blob_exists(self, blob_path: str) -> bool:
        """Check if a blob exists."""
        blob = self._bucket.blob(blob_path)
        return cast(bool, blob.exists())

    def _list_blobs(
        self,
        prefix: str,
        delimiter: str | None = None,
    ) -> list[str]:
        """List blobs matching a prefix."""
        full_prefix = self._gcs_path(prefix)
        blobs = self._client.list_blobs(
            self.bucket_name,
            prefix=full_prefix,
            delimiter=delimiter,
        )
        return [blob.name for blob in blobs]

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
        """Save model weights to GCS."""
        model_prefix = f"{self.config.models_subdir}/iteration_{iteration:08d}"
        model_path = self._gcs_path(model_prefix, "model.pt")

        # Save model weights
        weights_bytes = self._serialize_weights(state_dict)
        self._upload_bytes(weights_bytes, model_path)

        # Save metadata
        if metadata is None:
            metadata = ModelMetadata(
                iteration=iteration,
                version=iteration,
                timestamp=self._get_timestamp(),
            )

        metadata_path = self._gcs_path(model_prefix, "metadata.json")
        self._upload_json(metadata.to_dict(), metadata_path)

        gcs_uri = f"gs://{self.bucket_name}/{model_path}"
        logger.debug(f"Saved model at iteration {iteration}: {gcs_uri}")
        return gcs_uri

    def load_model(
        self,
        iteration: int,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """Load model weights from a specific iteration."""
        model_prefix = f"{self.config.models_subdir}/iteration_{iteration:08d}"
        model_path = self._gcs_path(model_prefix, "model.pt")
        metadata_path = self._gcs_path(model_prefix, "metadata.json")

        if not self._blob_exists(model_path):
            raise FileNotFoundError(f"Model not found at iteration {iteration}")

        weights_bytes = self._download_bytes(model_path)
        state_dict = self._deserialize_weights(weights_bytes, map_location)

        if self._blob_exists(metadata_path):
            metadata_dict = self._download_json(metadata_path)
            metadata = ModelMetadata.from_dict(metadata_dict)
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
        # List all model blobs
        blobs = self._list_blobs(self.config.models_subdir)

        # Find metadata files
        metadata_blobs = [b for b in blobs if b.endswith("metadata.json")]

        for metadata_blob in metadata_blobs:
            if self.config.champion_subdir in metadata_blob:
                continue

            try:
                metadata_dict = self._download_json(metadata_blob)
                metadata = ModelMetadata.from_dict(metadata_dict)
                if metadata.version == version:
                    # Load the corresponding model
                    model_blob = metadata_blob.replace("metadata.json", "model.pt")
                    weights_bytes = self._download_bytes(model_blob)
                    state_dict = self._deserialize_weights(weights_bytes, map_location)
                    return state_dict, metadata
            except Exception as e:
                logger.warning(f"Failed to check metadata {metadata_blob}: {e}")

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
        champion_prefix = f"{self.config.models_subdir}/{self.config.champion_subdir}"
        model_path = self._gcs_path(champion_prefix, "model.pt")

        # Save model weights
        weights_bytes = self._serialize_weights(state_dict)
        self._upload_bytes(weights_bytes, model_path)

        # Update metadata timestamp
        metadata.timestamp = self._get_timestamp()

        # Save metadata
        metadata_path = self._gcs_path(champion_prefix, "metadata.json")
        self._upload_json(metadata.to_dict(), metadata_path)

        gcs_uri = f"gs://{self.bucket_name}/{model_path}"
        logger.info(f"Saved champion model v{metadata.version}: {gcs_uri}")
        return gcs_uri

    def load_champion(
        self,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, torch.Tensor], ModelMetadata]:
        """Load the current champion model."""
        champion_prefix = f"{self.config.models_subdir}/{self.config.champion_subdir}"
        model_path = self._gcs_path(champion_prefix, "model.pt")
        metadata_path = self._gcs_path(champion_prefix, "metadata.json")

        if not self._blob_exists(model_path):
            raise FileNotFoundError("No champion model found")

        weights_bytes = self._download_bytes(model_path)
        state_dict = self._deserialize_weights(weights_bytes, map_location)

        if self._blob_exists(metadata_path):
            metadata_dict = self._download_json(metadata_path)
            metadata = ModelMetadata.from_dict(metadata_dict)
        else:
            metadata = ModelMetadata(iteration=0, version=0)

        logger.debug(f"Loaded champion model v{metadata.version}")
        return state_dict, metadata

    def get_champion_version(self) -> int:
        """Get the current champion model version."""
        champion_prefix = f"{self.config.models_subdir}/{self.config.champion_subdir}"
        metadata_path = self._gcs_path(champion_prefix, "metadata.json")

        if not self._blob_exists(metadata_path):
            return 0

        try:
            metadata_dict = self._download_json(metadata_path)
            return cast(int, metadata_dict.get("version", 0))
        except Exception:
            return 0

    def champion_exists(self) -> bool:
        """Check if a champion model exists."""
        champion_prefix = f"{self.config.models_subdir}/{self.config.champion_subdir}"
        model_path = self._gcs_path(champion_prefix, "model.pt")
        return self._blob_exists(model_path)

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
        """Save a game trajectory to GCS."""
        trajectory_prefix = (
            f"{self.config.games_subdir}/iteration_{iteration:08d}/worker_{worker_id:04d}"
        )
        trajectory_path = self._gcs_path(trajectory_prefix, f"{game_id}.pkl")

        # Serialize trajectory
        if self.config.compress_trajectories:
            data = trajectory.to_bytes(compress=True)
        else:
            data = pickle.dumps(trajectory)

        self._upload_bytes(data, trajectory_path)

        # Save metadata alongside
        metadata = TrajectoryMetadata(
            game_id=game_id,
            worker_id=worker_id,
            iteration=iteration,
            game_length=trajectory.game_length,
            winner=trajectory.winner,
            model_version=0,
            num_examples=len(trajectory.examples),
        )
        metadata_path = self._gcs_path(trajectory_prefix, f"{game_id}.meta.json")
        self._upload_json(metadata.to_dict(), metadata_path)

        gcs_uri = f"gs://{self.bucket_name}/{trajectory_path}"
        logger.debug(f"Saved trajectory {game_id} (iter={iteration}, worker={worker_id})")
        return gcs_uri

    def list_trajectories(
        self,
        iteration: int | None = None,
        worker_id: int | None = None,
    ) -> list[TrajectoryMetadata]:
        """List available trajectories in GCS."""
        metadata_list = []

        # Build prefix based on filters
        if iteration is not None and worker_id is not None:
            prefix = f"{self.config.games_subdir}/iteration_{iteration:08d}/worker_{worker_id:04d}"
        elif iteration is not None:
            prefix = f"{self.config.games_subdir}/iteration_{iteration:08d}"
        else:
            prefix = self.config.games_subdir

        # List all metadata files
        blobs = self._list_blobs(prefix)
        metadata_blobs = [b for b in blobs if b.endswith(".meta.json")]

        for blob_path in metadata_blobs:
            try:
                metadata_dict = self._download_json(blob_path)
                metadata = TrajectoryMetadata.from_dict(metadata_dict)

                # Apply worker filter if needed
                if worker_id is not None and metadata.worker_id != worker_id:
                    continue

                metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata {blob_path}: {e}")

        return sorted(metadata_list, key=lambda x: (x.iteration, x.worker_id, x.game_id))

    def load_trajectory(
        self,
        iteration: int,
        worker_id: int,
        game_id: str,
    ) -> GameTrajectory:
        """Load a specific trajectory from GCS."""
        from alpha_zero.distributed.shared_storage import GameTrajectory

        trajectory_prefix = (
            f"{self.config.games_subdir}/iteration_{iteration:08d}/worker_{worker_id:04d}"
        )
        trajectory_path = self._gcs_path(trajectory_prefix, f"{game_id}.pkl")

        if not self._blob_exists(trajectory_path):
            raise FileNotFoundError(
                f"Trajectory not found: iter={iteration}, worker={worker_id}, game={game_id}"
            )

        data = self._download_bytes(trajectory_path)

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
        """Stream trajectories from GCS."""
        metadata_list = []

        if iterations is None:
            metadata_list = self.list_trajectories()
        else:
            for iteration in iterations:
                metadata_list.extend(self.list_trajectories(iteration=iteration))

        if shuffle:
            np.random.shuffle(metadata_list)  # type: ignore[arg-type]

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
        """Count available trajectories in GCS."""
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
        """Save a full training checkpoint to GCS."""
        checkpoint_prefix = f"{self.config.checkpoints_subdir}/iteration_{iteration:08d}"
        checkpoint_path = self._gcs_path(checkpoint_prefix, "checkpoint.pt")

        # Serialize checkpoint
        buffer = io.BytesIO()
        torch.save(checkpoint_data, buffer)
        checkpoint_bytes = buffer.getvalue()

        self._upload_bytes(checkpoint_bytes, checkpoint_path)

        # Save metadata
        if metadata is None:
            metadata = CheckpointMetadata(
                iteration=iteration,
                timestamp=self._get_timestamp(),
            )

        metadata_path = self._gcs_path(checkpoint_prefix, "metadata.json")
        self._upload_json(metadata.to_dict(), metadata_path)

        gcs_uri = f"gs://{self.bucket_name}/{checkpoint_path}"
        logger.info(f"Saved checkpoint at iteration {iteration}: {gcs_uri}")
        return gcs_uri

    def load_checkpoint(
        self,
        iteration: int,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, Any], CheckpointMetadata]:
        """Load a specific checkpoint from GCS."""
        checkpoint_prefix = f"{self.config.checkpoints_subdir}/iteration_{iteration:08d}"
        checkpoint_path = self._gcs_path(checkpoint_prefix, "checkpoint.pt")
        metadata_path = self._gcs_path(checkpoint_prefix, "metadata.json")

        if not self._blob_exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at iteration {iteration}")

        checkpoint_bytes = self._download_bytes(checkpoint_path)
        buffer = io.BytesIO(checkpoint_bytes)
        checkpoint_data = torch.load(buffer, map_location=map_location, weights_only=False)

        if self._blob_exists(metadata_path):
            metadata_dict = self._download_json(metadata_path)
            metadata = CheckpointMetadata.from_dict(metadata_dict)
        else:
            metadata = CheckpointMetadata(iteration=iteration)

        logger.info(f"Loaded checkpoint from iteration {iteration}")
        return checkpoint_data, metadata

    def load_latest_checkpoint(
        self,
        map_location: str | torch.device = "cpu",
    ) -> tuple[dict[str, Any], CheckpointMetadata] | None:
        """Load the most recent checkpoint from GCS."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda x: x.iteration)
        return self.load_checkpoint(latest.iteration, map_location)

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List all available checkpoints in GCS."""
        metadata_list = []

        blobs = self._list_blobs(self.config.checkpoints_subdir)
        metadata_blobs = [b for b in blobs if b.endswith("metadata.json")]

        for blob_path in metadata_blobs:
            try:
                metadata_dict = self._download_json(blob_path)
                metadata = CheckpointMetadata.from_dict(metadata_dict)
                metadata_list.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata {blob_path}: {e}")

        return sorted(metadata_list, key=lambda x: x.iteration)

    # =========================================================================
    # Evaluation Operations
    # =========================================================================

    def save_evaluation(
        self,
        result: EvaluationResult,
    ) -> str:
        """Save evaluation results to GCS."""
        eval_prefix = f"{self.config.eval_subdir}/iteration_{result.iteration:08d}"
        result_path = self._gcs_path(eval_prefix, "results.json")

        self._upload_json(result.to_dict(), result_path)

        gcs_uri = f"gs://{self.bucket_name}/{result_path}"
        logger.info(
            f"Saved evaluation at iteration {result.iteration} "
            f"(win_rate={result.win_rate:.2%}, promoted={result.promoted})"
        )
        return gcs_uri

    def load_evaluation(
        self,
        iteration: int,
    ) -> EvaluationResult:
        """Load evaluation results for an iteration."""
        eval_prefix = f"{self.config.eval_subdir}/iteration_{iteration:08d}"
        result_path = self._gcs_path(eval_prefix, "results.json")

        if not self._blob_exists(result_path):
            raise FileNotFoundError(f"Evaluation not found at iteration {iteration}")

        result_dict = self._download_json(result_path)
        return EvaluationResult.from_dict(result_dict)

    def list_evaluations(self) -> list[EvaluationResult]:
        """List all evaluation results in GCS."""
        results = []

        blobs = self._list_blobs(self.config.eval_subdir)
        result_blobs = [b for b in blobs if b.endswith("results.json")]

        for blob_path in result_blobs:
            try:
                result_dict = self._download_json(blob_path)
                result = EvaluationResult.from_dict(result_dict)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load evaluation {blob_path}: {e}")

        return sorted(results, key=lambda x: x.iteration)

    # =========================================================================
    # Config Operations
    # =========================================================================

    def save_config(
        self,
        config: dict[str, Any],
        iteration: int,
    ) -> str:
        """Save training configuration to GCS."""
        config_prefix = f"{self.config.configs_subdir}/iteration_{iteration:08d}"
        config_path = self._gcs_path(config_prefix, "config.json")

        self._upload_json(config, config_path)

        gcs_uri = f"gs://{self.bucket_name}/{config_path}"
        logger.debug(f"Saved config at iteration {iteration}: {gcs_uri}")
        return gcs_uri

    def load_config(
        self,
        iteration: int,
    ) -> dict[str, Any]:
        """Load training configuration from GCS."""
        config_prefix = f"{self.config.configs_subdir}/iteration_{iteration:08d}"
        config_path = self._gcs_path(config_prefix, "config.json")

        if not self._blob_exists(config_path):
            raise FileNotFoundError(f"Config not found at iteration {iteration}")

        return self._download_json(config_path)

    # =========================================================================
    # Utility Operations
    # =========================================================================

    def exists(self, path: str) -> bool:
        """Check if a path exists in GCS."""
        # Handle gs:// URI format
        if path.startswith("gs://"):
            path = path.replace(f"gs://{self.bucket_name}/", "")
        return self._blob_exists(path)

    def delete(self, path: str) -> bool:
        """Delete a file from GCS."""
        # Handle gs:// URI format
        if path.startswith("gs://"):
            path = path.replace(f"gs://{self.bucket_name}/", "")

        if not self._blob_exists(path):
            return False

        blob = self._bucket.blob(path)
        blob.delete()
        logger.debug(f"Deleted: gs://{self.bucket_name}/{path}")
        return True

    def list_files(
        self,
        prefix: str,
        pattern: str | None = None,
    ) -> list[str]:
        """List files matching a prefix and optional pattern."""
        blobs = self._list_blobs(prefix)

        if pattern:
            import fnmatch

            blobs = [b for b in blobs if fnmatch.fnmatch(b.split("/")[-1], pattern)]

        return [f"gs://{self.bucket_name}/{b}" for b in blobs]

    def close(self) -> None:
        """Close the GCS client."""
        self._client.close()
        logger.debug("GCSStorageBackend closed")


__all__ = ["GCSStorageBackend"]
