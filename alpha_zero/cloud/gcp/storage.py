"""Google Cloud Storage checkpoint management.

Provides GCSCheckpointManager for storing and retrieving training
checkpoints in Google Cloud Storage. Follows the force_mock pattern
for testing without requiring GCP SDK.
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_zero.cloud.config import CloudConfig
from alpha_zero.cloud.exceptions import CloudConfigError, CloudConnectionError, CloudStorageError
from alpha_zero.cloud.protocols import CloudStorageProtocol

logger = logging.getLogger(__name__)

# SDK availability check
GCS_AVAILABLE = False
try:
    from google.cloud import storage  # type: ignore[import-untyped]
    from google.cloud.exceptions import NotFound  # type: ignore[import-untyped]

    GCS_AVAILABLE = True
except ImportError:
    storage = None  # type: ignore[assignment]
    NotFound = Exception  # type: ignore[assignment, misc]


@dataclass
class CheckpointMetadata:
    """Metadata for a stored checkpoint.

    Attributes:
        key: Storage key/path.
        size: Size in bytes.
        created: Creation timestamp as ISO string.
        training_step: Training step from metadata.
        model_version: Model version from metadata.
        custom: Additional custom metadata.
    """

    key: str
    size: int
    created: str
    training_step: int | None = None
    model_version: int | None = None
    custom: dict[str, str] | None = None


class MockGCSClient:
    """Mock GCS client for testing without SDK.

    Stores blobs in memory for testing checkpoint operations.
    """

    def __init__(self) -> None:
        """Initialize mock client."""
        self._blobs: dict[str, tuple[bytes, dict[str, str]]] = {}
        self._bucket_name: str = "mock-bucket"

    def bucket(self, bucket_name: str) -> MockGCSClient:
        """Get bucket (returns self for chaining)."""
        self._bucket_name = bucket_name
        return self

    def blob(self, blob_name: str) -> MockBlob:
        """Get blob reference."""
        return MockBlob(self, blob_name)

    def list_blobs(
        self, prefix: str | None = None, max_results: int | None = None
    ) -> list[MockBlob]:
        """List blobs with optional prefix filter."""
        blobs = []
        for key in self._blobs:
            if prefix is None or key.startswith(prefix):
                blobs.append(MockBlob(self, key))
                if max_results and len(blobs) >= max_results:
                    break
        return blobs


class MockBlob:
    """Mock blob for testing."""

    def __init__(self, client: MockGCSClient, name: str) -> None:
        """Initialize mock blob."""
        self._client = client
        self.name = name
        self._metadata: dict[str, str] = {}

    @property
    def metadata(self) -> dict[str, str]:
        """Get blob metadata."""
        if self.name in self._client._blobs:
            return self._client._blobs[self.name][1]
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, str]) -> None:
        """Set blob metadata."""
        self._metadata = value

    @property
    def size(self) -> int:
        """Get blob size."""
        if self.name in self._client._blobs:
            return len(self._client._blobs[self.name][0])
        return 0

    @property
    def time_created(self) -> Any:
        """Get creation time (mock)."""

        class MockTime:
            def isoformat(self) -> str:
                return "2026-02-02T00:00:00Z"

        return MockTime()

    def upload_from_filename(self, filename: str) -> None:
        """Upload from file."""
        with open(filename, "rb") as f:
            data = f.read()
        self._client._blobs[self.name] = (data, self._metadata.copy())

    def upload_from_file(self, file_obj: io.IOBase) -> None:
        """Upload from file object."""
        data = file_obj.read()
        if isinstance(data, str):
            data = data.encode()
        self._client._blobs[self.name] = (data, self._metadata.copy())

    def download_to_filename(self, filename: str) -> None:
        """Download to file."""
        if self.name not in self._client._blobs:
            raise KeyError(f"Blob not found: {self.name}")
        data, _ = self._client._blobs[self.name]
        with open(filename, "wb") as f:
            f.write(data)

    def download_as_bytes(self) -> bytes:
        """Download as bytes."""
        if self.name not in self._client._blobs:
            raise KeyError(f"Blob not found: {self.name}")
        return self._client._blobs[self.name][0]

    def exists(self) -> bool:
        """Check if blob exists."""
        return self.name in self._client._blobs

    def delete(self) -> None:
        """Delete blob."""
        if self.name in self._client._blobs:
            del self._client._blobs[self.name]

    def reload(self) -> None:
        """Reload blob metadata (no-op for mock)."""
        pass


class GCSCheckpointManager(CloudStorageProtocol):
    """Google Cloud Storage checkpoint manager.

    Implements CloudStorageProtocol for storing checkpoints in GCS.
    Follows the force_mock pattern for testing without SDK.

    Example:
        >>> config = CloudConfig(project_id="my-project", bucket_name="my-bucket")
        >>> manager = GCSCheckpointManager(config)
        >>> uri = manager.upload_checkpoint("model.pt", "checkpoints/model_v1.pt")
        >>> manager.download_checkpoint("checkpoints/model_v1.pt", "local_model.pt")
    """

    def __init__(
        self,
        config: CloudConfig,
        force_mock: bool = False,
    ) -> None:
        """Initialize GCS checkpoint manager.

        Args:
            config: Cloud configuration with bucket settings.
            force_mock: Force use of mock client for testing.

        Raises:
            CloudConfigError: If bucket_name is not configured.
        """
        self.config = config
        self._use_mock = force_mock or not GCS_AVAILABLE

        if not config.bucket_name and not self._use_mock:
            raise CloudConfigError("bucket_name is required for GCS storage")

        if self._use_mock:
            logger.info("Using mock GCS client (force_mock=%s, SDK available=%s)", 
                       force_mock, GCS_AVAILABLE)
            self._client: Any = MockGCSClient()
            self._bucket = self._client.bucket(config.bucket_name or "mock-bucket")
        else:
            try:
                self._client = storage.Client(project=config.project_id or None)
                self._bucket = self._client.bucket(config.bucket_name)
            except Exception as e:
                raise CloudConnectionError(
                    f"Failed to connect to GCS: {e}",
                    service="storage",
                    cause=e,
                ) from e

    @property
    def is_mock(self) -> bool:
        """Check if using mock client."""
        return self._use_mock

    def upload_checkpoint(
        self,
        local_path: Path | str,
        remote_key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload a checkpoint to GCS.

        Args:
            local_path: Path to local checkpoint file.
            remote_key: Key/path in GCS bucket.
            metadata: Optional metadata to attach.

        Returns:
            GCS URI of uploaded checkpoint (gs://bucket/key).

        Raises:
            CloudStorageError: If upload fails.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise CloudStorageError(
                f"Local file not found: {local_path}",
                operation="upload",
                path=str(local_path),
            )

        # Normalize key (remove leading slashes, use forward slashes)
        remote_key = remote_key.lstrip("/").replace("\\", "/")

        try:
            blob = self._bucket.blob(remote_key)
            if metadata:
                blob.metadata = metadata
            blob.upload_from_filename(str(local_path))

            uri = f"gs://{self.config.bucket_name}/{remote_key}"
            logger.info("Uploaded checkpoint: %s -> %s", local_path, uri)
            return uri

        except Exception as e:
            raise CloudStorageError(
                f"Failed to upload checkpoint: {e}",
                operation="upload",
                path=remote_key,
                cause=e,
            ) from e

    def download_checkpoint(
        self,
        remote_key: str,
        local_path: Path | str,
    ) -> Path:
        """Download a checkpoint from GCS.

        Args:
            remote_key: Key/path in GCS bucket.
            local_path: Local path to save checkpoint.

        Returns:
            Path to downloaded file.

        Raises:
            CloudStorageError: If download fails.
        """
        local_path = Path(local_path)
        remote_key = remote_key.lstrip("/").replace("\\", "/")

        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            blob = self._bucket.blob(remote_key)
            blob.download_to_filename(str(local_path))

            logger.info("Downloaded checkpoint: gs://%s/%s -> %s",
                       self.config.bucket_name, remote_key, local_path)
            return local_path

        except (KeyError, NotFound) as e:
            raise CloudStorageError(
                f"Checkpoint not found: {remote_key}",
                operation="download",
                path=remote_key,
                cause=e if not isinstance(e, KeyError) else None,
            ) from e
        except Exception as e:
            raise CloudStorageError(
                f"Failed to download checkpoint: {e}",
                operation="download",
                path=remote_key,
                cause=e,
            ) from e

    def list_checkpoints(
        self,
        prefix: str | None = None,
        max_results: int = 100,
    ) -> list[str]:
        """List available checkpoints.

        Args:
            prefix: Optional prefix to filter checkpoints.
            max_results: Maximum number of results.

        Returns:
            List of checkpoint keys.
        """
        # Use config prefix if not specified
        if prefix is None:
            prefix = self.config.checkpoint_prefix

        try:
            blobs = self._bucket.list_blobs(prefix=prefix, max_results=max_results)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.warning("Failed to list checkpoints: %s", e)
            return []

    def get_latest_checkpoint(self, prefix: str | None = None) -> str | None:
        """Get the most recent checkpoint.

        Checkpoints are expected to have step numbers in their names,
        e.g., "checkpoint_step_1000.pt".

        Args:
            prefix: Optional prefix to filter checkpoints.

        Returns:
            Key of latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints(prefix=prefix)
        if not checkpoints:
            return None

        # Sort by step number if present in name
        def extract_step(name: str) -> int:
            match = re.search(r"step[_-]?(\d+)", name, re.IGNORECASE)
            if match:
                return int(match.group(1))
            # Fall back to natural sort
            return 0

        sorted_checkpoints = sorted(checkpoints, key=extract_step, reverse=True)
        return sorted_checkpoints[0] if sorted_checkpoints else None

    def delete_checkpoint(self, remote_key: str) -> bool:
        """Delete a checkpoint from GCS.

        Args:
            remote_key: Key/path in GCS bucket.

        Returns:
            True if deleted successfully.

        Raises:
            CloudStorageError: If deletion fails.
        """
        remote_key = remote_key.lstrip("/").replace("\\", "/")

        try:
            blob = self._bucket.blob(remote_key)
            blob.delete()
            logger.info("Deleted checkpoint: gs://%s/%s", self.config.bucket_name, remote_key)
            return True
        except (KeyError, NotFound):
            logger.warning("Checkpoint not found for deletion: %s", remote_key)
            return False
        except Exception as e:
            raise CloudStorageError(
                f"Failed to delete checkpoint: {e}",
                operation="delete",
                path=remote_key,
                cause=e,
            ) from e

    def get_checkpoint_metadata(self, remote_key: str) -> CheckpointMetadata | None:
        """Get metadata for a checkpoint.

        Args:
            remote_key: Key/path in GCS bucket.

        Returns:
            Checkpoint metadata, or None if not found.
        """
        remote_key = remote_key.lstrip("/").replace("\\", "/")

        try:
            blob = self._bucket.blob(remote_key)
            if not blob.exists():
                return None

            blob.reload()
            meta = blob.metadata or {}

            return CheckpointMetadata(
                key=remote_key,
                size=blob.size,
                created=blob.time_created.isoformat(),
                training_step=int(meta.get("training_step", 0)) or None,
                model_version=int(meta.get("model_version", 0)) or None,
                custom=meta,
            )
        except Exception as e:
            logger.warning("Failed to get checkpoint metadata: %s", e)
            return None
