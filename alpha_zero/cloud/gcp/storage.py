"""
GCS Storage Classes for Model, Trajectory, and Checkpoint Management.

Provides Protocol-based storage abstractions for:
- Model weights with version tracking
- Game trajectories in sharded format
- Training checkpoints with recovery support

All operations are designed for cloud-native patterns:
- Atomic writes with temp file upload
- Version metadata for consistency
- LZMA compression for trajectory data
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from alpha_zero.cloud.gcp.exceptions import GCSStorageError
from alpha_zero.utils.compat import UTC

if TYPE_CHECKING:
    from google.cloud import storage as gcs

    from alpha_zero.cloud.gcp.config import GCPConfig

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _torch_load_compat(
    buffer: io.BytesIO,
    *,
    map_location: str = "cpu",
    weights_only: bool = True,
) -> Any:
    """Load PyTorch checkpoint with backwards compatibility.

    Handles compatibility for PyTorch versions that don't support the
    `weights_only` parameter (added in PyTorch 1.13+, enforced in 2.0+).

    Args:
        buffer: BytesIO buffer containing the checkpoint.
        map_location: Device to load tensors to.
        weights_only: If True, only load tensor data (safer). If False,
            allows loading arbitrary Python objects (needed for checkpoints
            with optimizer state and custom objects).

    Returns:
        Loaded checkpoint data.

    Note:
        When weights_only=False, this can execute arbitrary code if the
        checkpoint contains malicious pickled objects. Only use with
        trusted data sources (e.g., checkpoints you generated yourself).
    """
    import torch

    try:
        # Try with weights_only parameter (PyTorch 1.13+)
        return torch.load(buffer, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Fallback for older PyTorch versions without weights_only
        buffer.seek(0)  # Reset buffer position after failed attempt
        return torch.load(buffer, map_location=map_location)


# =============================================================================
# Protocols for Type Safety
# =============================================================================


class StorageBackend(Protocol):
    """Protocol for storage backends (enables testing with mocks)."""

    def upload_blob(self, bucket: str, blob_path: str, data: bytes) -> str:
        """Upload data to storage."""
        ...

    def download_blob(self, bucket: str, blob_path: str) -> bytes:
        """Download data from storage."""
        ...

    def blob_exists(self, bucket: str, blob_path: str) -> bool:
        """Check if blob exists."""
        ...

    def list_blobs(self, bucket: str, prefix: str) -> list[str]:
        """List blobs with prefix."""
        ...

    def delete_blob(self, bucket: str, blob_path: str) -> None:
        """Delete a blob."""
        ...


class ModelLike(Protocol):
    """Protocol for objects that can be saved/loaded as models."""

    def state_dict(self) -> dict[str, Any]:
        """Get model state dictionary."""
        ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load model state dictionary."""
        ...


# =============================================================================
# Metadata Classes
# =============================================================================


@dataclass
class ModelMetadata:
    """Metadata for a stored model version.

    Attributes:
        version: Model version number.
        iteration: Training iteration when model was saved.
        elo: ELO rating if available.
        timestamp: ISO format timestamp.
        games_played: Total games used for training.
        extra: Additional custom metadata.
    """

    version: int
    iteration: int
    elo: float | None = None
    timestamp: str | None = None
    games_played: int = 0
    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrajectoryMetadata:
    """Metadata for trajectory shard.

    Attributes:
        iteration: Training iteration.
        worker_id: Worker that generated this shard.
        model_version: Model version used for generation.
        num_examples: Number of training examples.
        timestamp: ISO format timestamp.
    """

    iteration: int
    worker_id: int
    model_version: int
    num_examples: int
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC).isoformat()


# =============================================================================
# GCS Backend Implementation
# =============================================================================


class GCSBackend:
    """Real GCS backend using google-cloud-storage.

    This class wraps the GCS client and provides a simple interface
    for common operations. It's designed to be easily mockable for testing.
    """

    def __init__(self, project_id: str | None = None) -> None:
        """Initialize GCS backend.

        Args:
            project_id: GCP project ID. If None, uses default.
        """
        self._project_id = project_id
        self._client: gcs.Client | None = None

    @property
    def client(self) -> gcs.Client:
        """Lazy-initialize GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage as gcs_lib

                self._client = gcs_lib.Client(project=self._project_id)
            except ImportError as e:
                raise GCSStorageError(
                    "google-cloud-storage not installed. Install with: pip install google-cloud-storage",
                    operation="client_init",
                ) from e
            except Exception as e:
                raise GCSStorageError(
                    f"Failed to initialize GCS client: {e}",
                    operation="client_init",
                ) from e
        return self._client

    def upload_blob(self, bucket: str, blob_path: str, data: bytes) -> str:
        """Upload data to GCS.

        Args:
            bucket: Bucket name (without gs:// prefix).
            blob_path: Path within bucket.
            data: Bytes to upload.

        Returns:
            Full GCS URI of uploaded blob.

        Raises:
            GCSStorageError: If upload fails.
        """
        try:
            bucket_obj = self.client.bucket(bucket)
            blob = bucket_obj.blob(blob_path)
            blob.upload_from_string(data)
            uri = f"gs://{bucket}/{blob_path}"
            logger.debug("Uploaded %d bytes to %s", len(data), uri)
            return uri
        except Exception as e:
            raise GCSStorageError(
                f"Failed to upload to {blob_path}: {e}",
                bucket=bucket,
                blob_path=blob_path,
                operation="upload",
            ) from e

    def download_blob(self, bucket: str, blob_path: str) -> bytes:
        """Download data from GCS.

        Args:
            bucket: Bucket name.
            blob_path: Path within bucket.

        Returns:
            Downloaded bytes.

        Raises:
            GCSStorageError: If download fails or blob doesn't exist.
        """
        try:
            bucket_obj = self.client.bucket(bucket)
            blob = bucket_obj.blob(blob_path)
            data = blob.download_as_bytes()
            logger.debug("Downloaded %d bytes from gs://%s/%s", len(data), bucket, blob_path)
            return data
        except Exception as e:
            raise GCSStorageError(
                f"Failed to download {blob_path}: {e}",
                bucket=bucket,
                blob_path=blob_path,
                operation="download",
            ) from e

    def blob_exists(self, bucket: str, blob_path: str) -> bool:
        """Check if blob exists.

        Args:
            bucket: Bucket name.
            blob_path: Path within bucket.

        Returns:
            True if blob exists.
        """
        try:
            bucket_obj = self.client.bucket(bucket)
            blob = bucket_obj.blob(blob_path)
            return blob.exists()
        except Exception:
            return False

    def list_blobs(self, bucket: str, prefix: str) -> list[str]:
        """List blobs with prefix.

        Args:
            bucket: Bucket name.
            prefix: Path prefix to filter.

        Returns:
            List of blob paths.
        """
        try:
            bucket_obj = self.client.bucket(bucket)
            blobs = bucket_obj.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            raise GCSStorageError(
                f"Failed to list blobs with prefix {prefix}: {e}",
                bucket=bucket,
                blob_path=prefix,
                operation="list",
            ) from e

    def delete_blob(self, bucket: str, blob_path: str) -> None:
        """Delete a blob.

        Args:
            bucket: Bucket name.
            blob_path: Path within bucket.
        """
        try:
            bucket_obj = self.client.bucket(bucket)
            blob = bucket_obj.blob(blob_path)
            blob.delete()
            logger.debug("Deleted gs://%s/%s", bucket, blob_path)
        except Exception as e:
            raise GCSStorageError(
                f"Failed to delete {blob_path}: {e}",
                bucket=bucket,
                blob_path=blob_path,
                operation="delete",
            ) from e


# =============================================================================
# Storage Classes
# =============================================================================


class GCSModelStorage:
    """GCS storage for model weights with version tracking.

    Models are stored with the following structure:
        gs://{bucket}/models/civ6az/v001/model.pt
        gs://{bucket}/models/civ6az/v001/metadata.json
        gs://{bucket}/models/civ6az/latest.json  # Points to latest version

    Thread-safe for concurrent reads. Writes should be coordinated externally.
    """

    def __init__(
        self,
        config: GCPConfig,
        backend: StorageBackend | None = None,
    ) -> None:
        """Initialize model storage.

        Args:
            config: GCP configuration.
            backend: Storage backend (uses GCSBackend if None).
        """
        self.config = config
        self.backend = backend or GCSBackend(config.project_id)
        self._bucket = config.bucket
        self._paths = config.paths

    @property
    def bucket(self) -> str:
        """Get the GCS bucket name."""
        return self._bucket

    def save_model(
        self,
        model: ModelLike,
        version: int,
        metadata: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> str:
        """Save model weights to GCS.

        Args:
            model: Model with state_dict() method.
            version: Version number for this model.
            metadata: Optional metadata dict (elo, iteration, etc.).
            model_config: Optional model architecture config dict. If provided,
                saved alongside state_dict for architecture reconstruction.

        Returns:
            GCS URI of saved model.

        Raises:
            GCSStorageError: If save fails.
        """
        import torch

        # Serialize model to bytes with optional config
        buffer = io.BytesIO()
        if model_config is not None:
            # New format: dict with state_dict and model_config
            checkpoint = {
                "state_dict": model.state_dict(),
                "model_config": model_config,
            }
            torch.save(checkpoint, buffer)
        else:
            # Legacy format: just state_dict
            torch.save(model.state_dict(), buffer)
        model_bytes = buffer.getvalue()

        # Upload model
        model_path = self._paths.model_path(version)
        model_uri = self.backend.upload_blob(self._bucket, model_path, model_bytes)

        # Create and upload metadata
        meta = ModelMetadata(
            version=version,
            iteration=metadata.get("iteration", 0) if metadata else 0,
            elo=metadata.get("elo") if metadata else None,
            games_played=metadata.get("games_played", 0) if metadata else 0,
            extra=metadata,
        )
        meta_bytes = json.dumps(meta.to_dict(), indent=2).encode()
        meta_path = self._paths.metadata_path(version)
        self.backend.upload_blob(self._bucket, meta_path, meta_bytes)

        # Update latest pointer
        latest_data = {"version": version, "path": model_path}
        latest_bytes = json.dumps(latest_data).encode()
        latest_path = f"{self._paths.models_prefix}/latest.json"
        self.backend.upload_blob(self._bucket, latest_path, latest_bytes)

        logger.info("Saved model v%d to %s", version, model_uri)
        return model_uri

    def load_model(
        self,
        version: int | None = None,
    ) -> tuple[dict[str, Any], ModelMetadata, dict[str, Any] | None]:
        """Load model weights from GCS.

        Args:
            version: Version to load (None = latest).

        Returns:
            Tuple of (state_dict, metadata, model_config).
            model_config may be None for legacy models saved without config.

        Raises:
            GCSStorageError: If load fails.
        """
        if version is None:
            version = self.get_latest_version()

        # Download model
        model_path = self._paths.model_path(version)
        model_bytes = self.backend.download_blob(self._bucket, model_path)
        buffer = io.BytesIO(model_bytes)
        # Use weights_only=False to support both new format (dict) and legacy format
        checkpoint = _torch_load_compat(buffer, map_location="cpu", weights_only=False)

        # Handle both new format (dict with state_dict + model_config) and legacy format
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # New format
            state_dict = checkpoint["state_dict"]
            model_config = checkpoint.get("model_config")
        else:
            # Legacy format: checkpoint is the state_dict directly
            state_dict = checkpoint
            model_config = None

        # Download metadata
        meta_path = self._paths.metadata_path(version)
        try:
            meta_bytes = self.backend.download_blob(self._bucket, meta_path)
            meta_dict = json.loads(meta_bytes.decode())
            metadata = ModelMetadata.from_dict(meta_dict)
        except GCSStorageError:
            # Metadata might not exist for older models
            metadata = ModelMetadata(version=version, iteration=0)

        logger.info("Loaded model v%d", version)
        return state_dict, metadata, model_config

    def get_latest_version(self) -> int:
        """Get the latest model version number.

        Returns:
            Latest version number.

        Raises:
            GCSStorageError: If no models exist.
        """
        latest_path = f"{self._paths.models_prefix}/latest.json"
        try:
            data = self.backend.download_blob(self._bucket, latest_path)
            info = json.loads(data.decode())
            return info["version"]
        except GCSStorageError:
            # Scan for versions if latest.json doesn't exist
            blobs = self.backend.list_blobs(self._bucket, self._paths.models_prefix)
            versions = []
            for blob in blobs:
                if "/v" in blob and blob.endswith("model.pt"):
                    try:
                        # Extract version from path like "models/civ6az/v001/model.pt"
                        v_part = blob.split("/v")[1].split("/")[0]
                        versions.append(int(v_part))
                    except (IndexError, ValueError):
                        continue

            if not versions:
                raise GCSStorageError(
                    "No models found in storage",
                    bucket=self._bucket,
                    blob_path=self._paths.models_prefix,
                    operation="get_latest",
                ) from None
            return max(versions)

    def list_versions(self) -> list[int]:
        """List all available model versions.

        Returns:
            Sorted list of version numbers.
        """
        blobs = self.backend.list_blobs(self._bucket, self._paths.models_prefix)
        versions = set()
        for blob in blobs:
            if "/v" in blob and blob.endswith("model.pt"):
                try:
                    v_part = blob.split("/v")[1].split("/")[0]
                    versions.add(int(v_part))
                except (IndexError, ValueError):
                    continue
        return sorted(versions)


class GCSTrajectoryStorage:
    """GCS storage for game trajectories in sharded format.

    Trajectories are stored as:
        gs://{bucket}/trajectories/civ6az/iter_001/shard_worker_0.pt
        gs://{bucket}/trajectories/civ6az/iter_001/manifest.json

    Supports LZMA compression for efficient storage.
    """

    def __init__(
        self,
        config: GCPConfig,
        backend: StorageBackend | None = None,
        compress: bool = True,
    ) -> None:
        """Initialize trajectory storage.

        Args:
            config: GCP configuration.
            backend: Storage backend (uses GCSBackend if None).
            compress: Whether to compress trajectories with LZMA.
        """
        self.config = config
        self.backend = backend or GCSBackend(config.project_id)
        self._bucket = config.bucket
        self._paths = config.paths
        self._compress = compress

    @property
    def bucket(self) -> str:
        """Get the GCS bucket name."""
        return self._bucket

    def save_trajectories(
        self,
        trajectories: list[Any],
        iteration: int,
        worker_id: int,
        model_version: int,
    ) -> str:
        """Save trajectory shard to GCS.

        Args:
            trajectories: List of trajectory objects (must be picklable).
            iteration: Training iteration number.
            worker_id: Worker ID that generated these trajectories.
            model_version: Model version used for generation.

        Returns:
            GCS URI of saved trajectories.
        """
        import lzma
        import pickle

        # Serialize trajectories
        data = pickle.dumps(trajectories)
        if self._compress:
            data = lzma.compress(data)

        # Upload shard
        shard_path = self._paths.trajectory_path(iteration, worker_id)
        if self._compress:
            shard_path = shard_path.replace(".pt", ".pt.xz")
        uri = self.backend.upload_blob(self._bucket, shard_path, data)

        # Update manifest
        self._update_manifest(iteration, worker_id, model_version, len(trajectories))

        logger.info(
            "Saved %d trajectories for iter %d worker %d to %s",
            len(trajectories),
            iteration,
            worker_id,
            uri,
        )
        return uri

    def _update_manifest(
        self,
        iteration: int,
        worker_id: int,
        model_version: int,
        num_examples: int,
    ) -> None:
        """Update iteration manifest with new shard info."""
        manifest_path = f"{self._paths.trajectories_prefix}/iter_{iteration:03d}/manifest.json"

        # Try to load existing manifest
        try:
            data = self.backend.download_blob(self._bucket, manifest_path)
            manifest = json.loads(data.decode())
        except GCSStorageError:
            manifest = {"iteration": iteration, "model_version": model_version, "shards": {}}

        # Update shard info
        manifest["shards"][str(worker_id)] = {
            "num_examples": num_examples,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        manifest["total_examples"] = sum(s["num_examples"] for s in manifest["shards"].values())

        # Save manifest
        manifest_bytes = json.dumps(manifest, indent=2).encode()
        self.backend.upload_blob(self._bucket, manifest_path, manifest_bytes)

    def load_trajectories(
        self,
        iteration: int,
        max_examples: int | None = None,
    ) -> list[Any]:
        """Load trajectories from an iteration.

        Args:
            iteration: Training iteration to load.
            max_examples: Maximum examples to return (samples if needed).

        Returns:
            List of trajectory objects.
        """
        import lzma
        import pickle
        import random

        prefix = f"{self._paths.trajectories_prefix}/iter_{iteration:03d}/"
        blobs = self.backend.list_blobs(self._bucket, prefix)

        all_trajectories: list[Any] = []
        for blob_path in blobs:
            if blob_path.endswith(".pt") or blob_path.endswith(".pt.xz"):
                data = self.backend.download_blob(self._bucket, blob_path)
                if blob_path.endswith(".xz"):
                    data = lzma.decompress(data)
                # SECURITY NOTE: pickle.loads can execute arbitrary code.
                # This is safe here because trajectories are only loaded from
                # our own GCS bucket (controlled storage), written by our own
                # workers. Never use this to load untrusted/external data.
                trajectories = pickle.loads(data)  # noqa: S301
                all_trajectories.extend(trajectories)

        # Sample if needed
        if max_examples and len(all_trajectories) > max_examples:
            all_trajectories = random.sample(all_trajectories, max_examples)

        logger.info("Loaded %d trajectories from iteration %d", len(all_trajectories), iteration)
        return all_trajectories

    def get_available_iterations(self) -> list[int]:
        """Get list of available trajectory iterations.

        Returns:
            Sorted list of iteration numbers.
        """
        blobs = self.backend.list_blobs(self._bucket, self._paths.trajectories_prefix)
        iterations = set()
        for blob in blobs:
            if "/iter_" in blob:
                try:
                    iter_part = blob.split("/iter_")[1].split("/")[0]
                    iterations.add(int(iter_part))
                except (IndexError, ValueError):
                    continue
        return sorted(iterations)


class GCSCheckpointStorage:
    """GCS storage for training checkpoints.

    Checkpoints are stored as:
        gs://{bucket}/checkpoints/civ6az/training_step_1000.pt
        gs://{bucket}/checkpoints/civ6az/latest.pt

    Includes optimizer state and training metadata for recovery.
    """

    def __init__(
        self,
        config: GCPConfig,
        backend: StorageBackend | None = None,
    ) -> None:
        """Initialize checkpoint storage.

        Args:
            config: GCP configuration.
            backend: Storage backend.
        """
        self.config = config
        self.backend = backend or GCSBackend(config.project_id)
        self._bucket = config.bucket
        self._paths = config.paths

    @property
    def bucket(self) -> str:
        """Get the GCS bucket name."""
        return self._bucket

    def save_checkpoint(
        self,
        model: ModelLike,
        optimizer: Any,
        step: int,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Save training checkpoint.

        Args:
            model: Model to checkpoint.
            optimizer: Optimizer state.
            step: Training step number.
            extra: Additional state to save.

        Returns:
            GCS URI of checkpoint.
        """
        import torch

        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            if hasattr(optimizer, "state_dict")
            else None,
            "timestamp": datetime.now(UTC).isoformat(),
            **(extra or {}),
        }

        buffer = io.BytesIO()
        torch.save(checkpoint, buffer)
        data = buffer.getvalue()

        # Save numbered checkpoint
        ckpt_path = self._paths.checkpoint_path(step)
        uri = self.backend.upload_blob(self._bucket, ckpt_path, data)

        # Update latest pointer
        self.backend.upload_blob(self._bucket, self._paths.latest_checkpoint_path(), data)

        logger.info("Saved checkpoint at step %d to %s", step, uri)
        return uri

    def load_checkpoint(
        self,
        step: int | None = None,
    ) -> dict[str, Any] | None:
        """Load training checkpoint.

        Args:
            step: Specific step to load (None = latest).

        Returns:
            Checkpoint dict or None if not found.
        """
        if step is None:
            ckpt_path = self._paths.latest_checkpoint_path()
        else:
            ckpt_path = self._paths.checkpoint_path(step)

        try:
            data = self.backend.download_blob(self._bucket, ckpt_path)
            buffer = io.BytesIO(data)
            # weights_only=False needed for optimizer state and training metadata
            checkpoint = _torch_load_compat(buffer, map_location="cpu", weights_only=False)
            logger.info(
                "Loaded checkpoint from %s (step %d)", ckpt_path, checkpoint.get("step", -1)
            )
            return checkpoint
        except GCSStorageError:
            logger.warning("No checkpoint found at %s", ckpt_path)
            return None

    def get_latest_step(self) -> int | None:
        """Get the latest checkpoint step number.

        Returns:
            Latest step number or None if no checkpoints.
        """
        checkpoint = self.load_checkpoint()
        return checkpoint.get("step") if checkpoint else None

    def cleanup_old_checkpoints(self, keep_last: int = 5) -> int:
        """Delete old checkpoints, keeping only the most recent.

        Args:
            keep_last: Number of recent checkpoints to keep.

        Returns:
            Number of checkpoints deleted.
        """
        blobs = self.backend.list_blobs(self._bucket, self._paths.checkpoints_prefix)

        # Find numbered checkpoints
        checkpoints: list[tuple[int, str]] = []
        for blob in blobs:
            if "training_step_" in blob and blob.endswith(".pt"):
                try:
                    step = int(blob.split("training_step_")[1].replace(".pt", ""))
                    checkpoints.append((step, blob))
                except ValueError:
                    continue

        # Sort by step and delete old ones
        checkpoints.sort(reverse=True)
        deleted = 0
        for _step, blob_path in checkpoints[keep_last:]:
            self.backend.delete_blob(self._bucket, blob_path)
            deleted += 1
            logger.debug("Deleted old checkpoint: %s", blob_path)

        if deleted:
            logger.info("Cleaned up %d old checkpoints", deleted)
        return deleted
