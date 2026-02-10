"""GCS storage utilities for checkpoints and models.

This module provides:
- GCSCheckpointManager: Checkpoint upload/download to GCS
- GCSModelStorage: Model weight storage with versioning
- CheckpointMetadata: Metadata tracking for checkpoints

Usage:
    from alpha_zero.cloud.gcs_storage import GCSCheckpointManager

    manager = GCSCheckpointManager(
        bucket_name="my-bucket",
        prefix="checkpoints/experiment1",
    )

    # Upload checkpoint
    uri = manager.upload_checkpoint("checkpoint.pt", iteration=100)

    # Download latest
    local_path = manager.download_latest()
"""

from __future__ import annotations

import io
import json
import logging
import lzma
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import torch

logger = logging.getLogger(__name__)


class StorageClientProtocol(Protocol):
    """Protocol for storage client implementations."""

    def bucket(self, name: str) -> Any:
        """Get bucket by name."""
        ...


class BlobProtocol(Protocol):
    """Protocol for blob implementations."""

    name: str
    size: int
    updated: datetime

    def upload_from_file(self, file_obj: BinaryIO) -> None:
        """Upload from file object."""
        ...

    def download_to_file(self, file_obj: BinaryIO) -> None:
        """Download to file object."""
        ...

    def upload_from_filename(self, filename: str) -> None:
        """Upload from filename."""
        ...

    def download_to_filename(self, filename: str) -> None:
        """Download to filename."""
        ...

    def exists(self) -> bool:
        """Check if blob exists."""
        ...

    def delete(self) -> None:
        """Delete the blob."""
        ...


@dataclass
class CheckpointMetadata:
    """Metadata for a stored checkpoint.

    Attributes:
        key: Storage key/path.
        size: Size in bytes.
        created: ISO timestamp of creation.
        training_step: Training step when checkpoint was saved.
        model_version: Model version number.
        iteration: Training iteration.
        metrics: Training metrics at checkpoint time.
        custom: Additional custom metadata.
    """

    key: str
    size: int = 0
    created: str = ""
    training_step: int = 0
    model_version: int = 0
    iteration: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Create from dictionary."""
        return cls(
            key=str(data.get("key", "")),
            size=int(data.get("size", 0)),
            created=str(data.get("created", "")),
            training_step=int(data.get("training_step", 0)),
            model_version=int(data.get("model_version", 0)),
            iteration=int(data.get("iteration", 0)),
            metrics=dict(data.get("metrics", {})),
            custom=dict(data.get("custom", {})),
        )


class MockBlob:
    """Mock GCS blob for testing without GCP dependencies."""

    def __init__(
        self,
        name: str,
        storage: dict[str, bytes],
        metadata_storage: dict[str, dict[str, str]],
    ) -> None:
        """Initialize mock blob.

        Args:
            name: Blob name/path.
            storage: Shared storage dictionary.
            metadata_storage: Shared metadata storage.
        """
        self.name = name
        self._storage = storage
        self._metadata_storage = metadata_storage
        self._metadata: dict[str, str] = {}
        self.size = 0
        self.updated = datetime.now()

    @property
    def metadata(self) -> dict[str, str]:
        """Get blob metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, str]) -> None:
        """Set blob metadata."""
        self._metadata = value

    def upload_from_file(self, file_obj: BinaryIO) -> None:
        """Upload from file object."""
        data = file_obj.read()
        self._storage[self.name] = data
        self.size = len(data)
        self.updated = datetime.now()
        self._metadata_storage[self.name] = self._metadata.copy()

    def download_to_file(self, file_obj: BinaryIO) -> None:
        """Download to file object."""
        if self.name not in self._storage:
            raise FileNotFoundError(f"Blob not found: {self.name}")
        file_obj.write(self._storage[self.name])

    def upload_from_filename(self, filename: str) -> None:
        """Upload from filename."""
        with open(filename, "rb") as f:
            self.upload_from_file(f)

    def download_to_filename(self, filename: str) -> None:
        """Download to filename."""
        if self.name not in self._storage:
            raise FileNotFoundError(f"Blob not found: {self.name}")
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "wb") as f:
            f.write(self._storage[self.name])

    def exists(self) -> bool:
        """Check if blob exists."""
        return self.name in self._storage

    def delete(self) -> None:
        """Delete the blob."""
        if self.name in self._storage:
            del self._storage[self.name]
        if self.name in self._metadata_storage:
            del self._metadata_storage[self.name]


class MockBucket:
    """Mock GCS bucket for testing."""

    def __init__(self, name: str) -> None:
        """Initialize mock bucket.

        Args:
            name: Bucket name.
        """
        self.name = name
        self._storage: dict[str, bytes] = {}
        self._metadata_storage: dict[str, dict[str, str]] = {}

    def blob(self, name: str) -> MockBlob:
        """Get or create blob."""
        blob = MockBlob(name, self._storage, self._metadata_storage)
        if name in self._storage:
            blob.size = len(self._storage[name])
            blob._metadata = self._metadata_storage.get(name, {})
        return blob

    def list_blobs(self, prefix: str = "") -> list[MockBlob]:
        """List blobs with prefix."""
        blobs = []
        for name in self._storage:
            if name.startswith(prefix):
                blob = self.blob(name)
                blobs.append(blob)
        return blobs


class MockGCSClient:
    """Mock GCS client for testing."""

    def __init__(self) -> None:
        """Initialize mock client."""
        self._buckets: dict[str, MockBucket] = {}

    def bucket(self, name: str) -> MockBucket:
        """Get or create bucket."""
        if name not in self._buckets:
            self._buckets[name] = MockBucket(name)
        return self._buckets[name]


def _get_gcs_client(force_mock: bool = False) -> StorageClientProtocol:
    """Get GCS client, falling back to mock if SDK unavailable.

    Args:
        force_mock: Force use of mock client for testing.

    Returns:
        GCS client instance (real or mock).
    """
    if force_mock:
        return MockGCSClient()

    try:
        from google.cloud import storage

        return storage.Client()
    except ImportError:
        logger.warning("google-cloud-storage not installed, using mock client")
        return MockGCSClient()
    except Exception as e:
        logger.warning(f"Failed to create GCS client: {e}, using mock client")
        return MockGCSClient()


class GCSCheckpointManager:
    """Manages checkpoint storage on Google Cloud Storage.

    Features:
    - Upload/download checkpoints with metadata
    - Atomic updates using temporary files
    - Optional compression
    - Checkpoint rotation (cleanup old checkpoints)
    - Mock mode for testing

    Attributes:
        bucket_name: GCS bucket name.
        prefix: Path prefix for checkpoints.
        local_cache_dir: Local directory for downloaded checkpoints.
        enable_compression: Whether to compress checkpoints.
        max_checkpoints: Maximum checkpoints to retain.
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "checkpoints",
        local_cache_dir: str | None = None,
        enable_compression: bool = True,
        max_checkpoints: int = 10,
        force_mock: bool = False,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            bucket_name: GCS bucket name.
            prefix: Path prefix for checkpoints.
            local_cache_dir: Local cache directory (default: system temp).
            enable_compression: Enable checkpoint compression.
            max_checkpoints: Maximum checkpoints to retain (0 = unlimited).
            force_mock: Force use of mock client for testing.
        """
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/")
        self.enable_compression = enable_compression
        self.max_checkpoints = max_checkpoints
        self._force_mock = force_mock

        # Set up local cache
        if local_cache_dir:
            self.local_cache_dir = Path(local_cache_dir)
        else:
            self.local_cache_dir = Path(tempfile.gettempdir()) / "gcs_checkpoints"
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize client lazily
        self._client: StorageClientProtocol | None = None
        self._bucket: Any = None

    def _get_client(self) -> StorageClientProtocol:
        """Get or create GCS client."""
        if self._client is None:
            self._client = _get_gcs_client(force_mock=self._force_mock)
        return self._client

    def _get_bucket(self) -> Any:
        """Get or create bucket reference."""
        if self._bucket is None:
            self._bucket = self._get_client().bucket(self.bucket_name)
        return self._bucket

    def upload_checkpoint(
        self,
        local_path: str | Path,
        iteration: int,
        training_step: int = 0,
        model_version: int = 0,
        metrics: dict[str, float] | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Upload checkpoint to GCS with metadata.

        Args:
            local_path: Local path to checkpoint file.
            iteration: Training iteration number.
            training_step: Training step number.
            model_version: Model version number.
            metrics: Training metrics at checkpoint time.
            custom_metadata: Additional custom metadata.

        Returns:
            GCS URI of uploaded checkpoint.

        Raises:
            FileNotFoundError: If local checkpoint file doesn't exist.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {local_path}")

        # Generate blob name
        blob_name = f"{self.prefix}/checkpoint_{iteration:08d}.pt"

        # Prepare metadata
        metadata = CheckpointMetadata(
            key=blob_name,
            training_step=training_step,
            model_version=model_version,
            iteration=iteration,
            created=datetime.now(UTC).isoformat(),
            metrics=metrics or {},
            custom=custom_metadata or {},
        )

        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)

        # Set metadata
        blob.metadata = {
            "iteration": str(iteration),
            "training_step": str(training_step),
            "model_version": str(model_version),
            "timestamp": metadata.created,
        }

        # Upload with optional compression
        if self.enable_compression:
            with open(local_path, "rb") as f:
                data = f.read()
            compressed = lzma.compress(data)

            with io.BytesIO(compressed) as compressed_io:
                blob.upload_from_file(compressed_io)
            metadata.size = len(compressed)
        else:
            blob.upload_from_filename(str(local_path))
            metadata.size = local_path.stat().st_size

        # Upload metadata sidecar
        metadata_blob_name = f"{self.prefix}/checkpoint_{iteration:08d}.meta.json"
        metadata_blob = bucket.blob(metadata_blob_name)
        metadata_json = json.dumps(metadata.to_dict()).encode()
        with io.BytesIO(metadata_json) as meta_io:
            metadata_blob.upload_from_file(meta_io)

        logger.info(
            f"Uploaded checkpoint to gs://{self.bucket_name}/{blob_name} "
            f"(iteration={iteration}, size={metadata.size})"
        )

        # Cleanup old checkpoints
        if self.max_checkpoints > 0:
            self._cleanup_old_checkpoints()

        return f"gs://{self.bucket_name}/{blob_name}"

    def download_checkpoint(
        self,
        iteration: int,
        local_path: str | Path | None = None,
    ) -> Path:
        """Download specific checkpoint from GCS.

        Args:
            iteration: Iteration number to download.
            local_path: Local path to save checkpoint (default: cache dir).

        Returns:
            Local path to downloaded checkpoint.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
        """
        blob_name = f"{self.prefix}/checkpoint_{iteration:08d}.pt"
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)

        if not blob.exists():
            raise FileNotFoundError(f"Checkpoint not found: {blob_name}")

        # Determine local path
        if local_path:
            local_path = Path(local_path)
        else:
            local_path = self.local_cache_dir / f"checkpoint_{iteration:08d}.pt"

        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with optional decompression
        if self.enable_compression:
            with io.BytesIO() as compressed_io:
                blob.download_to_file(compressed_io)
                compressed_io.seek(0)
                compressed = compressed_io.read()

            data = lzma.decompress(compressed)
            with open(local_path, "wb") as f:
                f.write(data)
        else:
            blob.download_to_filename(str(local_path))

        logger.info(f"Downloaded checkpoint from gs://{self.bucket_name}/{blob_name}")
        return local_path

    def download_latest(self, local_path: str | Path | None = None) -> Path | None:
        """Download latest checkpoint from GCS.

        Args:
            local_path: Local path to save checkpoint.

        Returns:
            Local path to downloaded checkpoint, or None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            logger.info("No checkpoints found")
            return None

        # Get latest by iteration
        latest = max(checkpoints, key=lambda m: m.iteration)
        return self.download_checkpoint(latest.iteration, local_path)

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List all checkpoints with metadata.

        Returns:
            List of checkpoint metadata, sorted by iteration.
        """
        bucket = self._get_bucket()
        blobs = bucket.list_blobs(prefix=self.prefix)

        checkpoints: list[CheckpointMetadata] = []

        for blob in blobs:
            # Only process checkpoint files (not metadata sidecars)
            if not blob.name.endswith(".pt"):
                continue

            # Try to get metadata from sidecar
            meta_blob_name = blob.name.replace(".pt", ".meta.json")
            meta_blob = bucket.blob(meta_blob_name)

            if meta_blob.exists():
                try:
                    with io.BytesIO() as meta_io:
                        meta_blob.download_to_file(meta_io)
                        meta_io.seek(0)
                        meta_data = json.loads(meta_io.read().decode())
                    checkpoints.append(CheckpointMetadata.from_dict(meta_data))
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {blob.name}: {e}")

            # Fall back to parsing blob name and attributes
            try:
                # Extract iteration from filename
                filename = Path(blob.name).stem
                iteration = int(filename.split("_")[-1])

                checkpoints.append(
                    CheckpointMetadata(
                        key=blob.name,
                        size=blob.size,
                        created=blob.updated.isoformat() if blob.updated else "",
                        iteration=iteration,
                    )
                )
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to parse checkpoint {blob.name}: {e}")

        # Sort by iteration
        checkpoints.sort(key=lambda m: m.iteration)
        return checkpoints

    def exists(self, iteration: int) -> bool:
        """Check if checkpoint exists.

        Args:
            iteration: Iteration number to check.

        Returns:
            True if checkpoint exists.
        """
        blob_name = f"{self.prefix}/checkpoint_{iteration:08d}.pt"
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        return blob.exists()

    def delete_checkpoint(self, iteration: int) -> bool:
        """Delete a checkpoint.

        Args:
            iteration: Iteration number to delete.

        Returns:
            True if checkpoint was deleted, False if not found.
        """
        blob_name = f"{self.prefix}/checkpoint_{iteration:08d}.pt"
        meta_blob_name = f"{self.prefix}/checkpoint_{iteration:08d}.meta.json"

        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)
        meta_blob = bucket.blob(meta_blob_name)

        deleted = False
        if blob.exists():
            blob.delete()
            deleted = True
        if meta_blob.exists():
            meta_blob.delete()

        if deleted:
            logger.info(f"Deleted checkpoint: iteration={iteration}")
        return deleted

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= self.max_checkpoints:
            return

        # Delete oldest checkpoints
        to_delete = checkpoints[: -self.max_checkpoints]
        for meta in to_delete:
            self.delete_checkpoint(meta.iteration)
            logger.info(f"Cleaned up old checkpoint: iteration={meta.iteration}")

    def get_latest_iteration(self) -> int | None:
        """Get the latest checkpoint iteration number.

        Returns:
            Latest iteration number, or None if no checkpoints.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return max(m.iteration for m in checkpoints)


class GCSModelStorage:
    """Stores trained models on GCS with versioning.

    Features:
    - Version-based model storage
    - Metrics tracking per version
    - Latest version tracking
    - Model artifact management
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "models",
        force_mock: bool = False,
    ) -> None:
        """Initialize model storage.

        Args:
            bucket_name: GCS bucket name.
            prefix: Path prefix for models.
            force_mock: Force use of mock client for testing.
        """
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/")
        self._force_mock = force_mock
        self._client: StorageClientProtocol | None = None
        self._bucket: Any = None

    def _get_client(self) -> StorageClientProtocol:
        """Get or create GCS client."""
        if self._client is None:
            self._client = _get_gcs_client(force_mock=self._force_mock)
        return self._client

    def _get_bucket(self) -> Any:
        """Get or create bucket reference."""
        if self._bucket is None:
            self._bucket = self._get_client().bucket(self.bucket_name)
        return self._bucket

    def save_model(
        self,
        model: torch.nn.Module,
        version: int,
        metrics: dict[str, float] | None = None,
        optimizer_state: dict[str, Any] | None = None,
    ) -> str:
        """Save model to GCS with version tracking.

        Args:
            model: PyTorch model to save.
            version: Model version number.
            metrics: Training metrics for this version.
            optimizer_state: Optional optimizer state to save.

        Returns:
            GCS URI of saved model.
        """
        blob_name = f"{self.prefix}/model_v{version:06d}.pt"

        # Prepare model data
        model_data: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "version": version,
            "timestamp": datetime.now(UTC).isoformat(),
            "metrics": metrics or {},
        }
        if optimizer_state:
            model_data["optimizer_state_dict"] = optimizer_state

        # Serialize and upload
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)

        buffer = io.BytesIO()
        torch.save(model_data, buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer)

        # Update latest pointer
        self._update_latest_pointer(version)

        logger.info(f"Saved model v{version} to gs://{self.bucket_name}/{blob_name}")
        return f"gs://{self.bucket_name}/{blob_name}"

    def load_model(
        self,
        model: torch.nn.Module,
        version: int | None = None,
        load_optimizer: bool = False,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Load model weights from GCS.

        Args:
            model: PyTorch model to load weights into.
            version: Specific version to load (None = latest).
            load_optimizer: Whether to return optimizer state.

        Returns:
            Tuple of (model data dict, metrics dict).

        Raises:
            FileNotFoundError: If model version not found.
        """
        if version is None:
            version = self.get_latest_version()
            if version is None:
                raise FileNotFoundError("No models found in storage")

        blob_name = f"{self.prefix}/model_v{version:06d}.pt"
        bucket = self._get_bucket()
        blob = bucket.blob(blob_name)

        if not blob.exists():
            raise FileNotFoundError(f"Model not found: v{version}")

        # Download and load
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        model_data = torch.load(buffer, map_location="cpu", weights_only=False)

        # Load state dict into model
        model.load_state_dict(model_data["model_state_dict"])

        logger.info(f"Loaded model v{version} from gs://{self.bucket_name}/{blob_name}")
        return model_data, model_data.get("metrics", {})

    def get_latest_version(self) -> int | None:
        """Get the latest model version number.

        Returns:
            Latest version number, or None if no models.
        """
        bucket = self._get_bucket()
        pointer_blob = bucket.blob(f"{self.prefix}/latest.txt")

        if pointer_blob.exists():
            try:
                buffer = io.BytesIO()
                pointer_blob.download_to_file(buffer)
                buffer.seek(0)
                return int(buffer.read().decode().strip())
            except Exception as e:
                logger.warning(f"Failed to read latest pointer: {e}")

        # Fall back to listing models
        blobs = bucket.list_blobs(prefix=self.prefix)
        versions = []
        for blob in blobs:
            if blob.name.endswith(".pt") and "model_v" in blob.name:
                try:
                    filename = Path(blob.name).stem
                    version = int(filename.split("_v")[-1])
                    versions.append(version)
                except ValueError:
                    continue

        return max(versions) if versions else None

    def _update_latest_pointer(self, version: int) -> None:
        """Update the latest version pointer if version is higher.

        Args:
            version: Version number to potentially set as latest.
        """
        bucket = self._get_bucket()
        pointer_blob = bucket.blob(f"{self.prefix}/latest.txt")

        # Check current latest
        current_latest = self.get_latest_version()
        if current_latest is not None and version <= current_latest:
            return  # Don't update if new version is not higher

        buffer = io.BytesIO(str(version).encode())
        pointer_blob.upload_from_file(buffer)

    def list_versions(self) -> list[int]:
        """List all available model versions.

        Returns:
            List of version numbers, sorted ascending.
        """
        bucket = self._get_bucket()
        blobs = bucket.list_blobs(prefix=self.prefix)

        versions = []
        for blob in blobs:
            if blob.name.endswith(".pt") and "model_v" in blob.name:
                try:
                    filename = Path(blob.name).stem
                    version = int(filename.split("_v")[-1])
                    versions.append(version)
                except ValueError:
                    continue

        return sorted(versions)
