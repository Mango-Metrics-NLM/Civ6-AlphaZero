"""Cloud Storage client for GCP.

This module provides a Cloud Storage client that implements the
CloudStorageProtocol for uploading, downloading, and managing
files in Google Cloud Storage.

The client supports:
- File upload/download with retry logic
- Directory upload/download
- Checkpoint and artifact management
- Graceful fallback when GCP SDK is not installed

Example:
    from alpha_zero.cloud.gcp import CloudStorageClient, GCPConfig

    config = GCPConfig.from_env()
    storage = CloudStorageClient(config)

    # Upload checkpoint
    uri = storage.upload_file("model.pt", "checkpoints/epoch_10.pt")

    # Download checkpoint
    storage.download_file("checkpoints/epoch_10.pt", "local_model.pt")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.protocols import CloudStorageProtocol

logger = logging.getLogger(__name__)

# Try to import GCP SDK
try:
    from google.cloud import storage

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None  # type: ignore


class CloudStorageClient:
    """Cloud Storage client implementing CloudStorageProtocol.

    This client provides methods for uploading, downloading, and managing
    files in Google Cloud Storage. All operations include retry logic
    and proper error handling.

    If the GCP SDK is not installed, the client will raise an error
    on initialization unless `allow_fallback=True`.
    """

    def __init__(
        self,
        config: GCPConfig,
        *,
        allow_fallback: bool = False,
        force_mock: bool = False,
        client: Any = None,
    ):
        """Initialize Cloud Storage client.

        Args:
            config: GCP configuration.
            allow_fallback: If True, allow local filesystem fallback when GCP SDK is unavailable.
            force_mock: If True, use mock/fallback mode even if SDK is available (for testing).
            client: Optional pre-configured storage client (for testing).
        """
        self.config = config
        self.allow_fallback = allow_fallback
        self._client = client
        # Use mock if forced, or if client is provided, or if SDK unavailable with fallback
        self._use_mock = force_mock or (not GCS_AVAILABLE and allow_fallback)

        if not GCS_AVAILABLE and not allow_fallback and not force_mock:
            raise ImportError(
                "google-cloud-storage is not installed. "
                "Install with: pip install google-cloud-storage"
            )

        if GCS_AVAILABLE and self._client is None and not self._use_mock:
            self._client = storage.Client(project=config.project_id)

        self._bucket_cache: dict[str, Any] = {}

    def _get_bucket(self, bucket_name: str | None = None) -> Any:
        """Get or create bucket reference."""
        bucket_name = bucket_name or self.config.storage.bucket

        if not bucket_name:
            raise ValueError("No bucket specified and no default bucket configured")

        if bucket_name not in self._bucket_cache:
            if self._use_mock or self._client is None:
                raise RuntimeError("GCS SDK not available or in mock mode")
            self._bucket_cache[bucket_name] = self._client.bucket(bucket_name)

        return self._bucket_cache[bucket_name]

    def _parse_gcs_uri(self, uri: str) -> tuple[str, str]:
        """Parse a gs:// URI into bucket and path.

        Args:
            uri: GCS URI (e.g., gs://bucket/path/to/file).

        Returns:
            Tuple of (bucket_name, blob_path).
        """
        if uri.startswith("gs://"):
            uri = uri[5:]
        parts = uri.split("/", 1)
        bucket = parts[0]
        path = parts[1] if len(parts) > 1 else ""
        return bucket, path

    def _retry_operation(
        self,
        operation: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute an operation with retry logic.

        Args:
            operation: Callable to execute.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the operation.
        """
        retry_count = self.config.storage.retry_count
        retry_delay = self.config.storage.retry_delay_seconds

        last_error: Exception | None = None
        for attempt in range(retry_count + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < retry_count:
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{retry_count + 1}): {e}"
                    )
                    time.sleep(retry_delay * (2**attempt))  # Exponential backoff

        raise last_error  # type: ignore

    def upload_file(
        self,
        local_path: str | Path,
        remote_path: str,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        bucket_name: str | None = None,
    ) -> str:
        """Upload a file to Cloud Storage.

        Args:
            local_path: Local file path.
            remote_path: Destination path (blob name or full gs:// URI).
            content_type: MIME type of the file.
            metadata: Custom metadata to attach.
            bucket_name: Override default bucket.

        Returns:
            Full gs:// URI of the uploaded file.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        # Parse remote path
        if remote_path.startswith("gs://"):
            bucket_name, remote_path = self._parse_gcs_uri(remote_path)

        if self._use_mock:
            # Fallback to local storage (for testing)
            fallback_path = Path("storage") / (bucket_name or self.config.storage.bucket or "default") / remote_path
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            fallback_path.write_bytes(local_path.read_bytes())
            logger.info(f"Fallback: Saved to {fallback_path}")
            return f"file://{fallback_path.absolute()}"

        bucket = self._get_bucket(bucket_name)
        blob = bucket.blob(remote_path)

        if metadata:
            blob.metadata = metadata

        def do_upload() -> None:
            blob.upload_from_filename(
                str(local_path),
                content_type=content_type,
            )

        self._retry_operation(do_upload)

        uri = f"gs://{bucket.name}/{remote_path}"
        logger.info(f"Uploaded {local_path} to {uri}")
        return uri

    def download_file(
        self,
        remote_path: str,
        local_path: str | Path,
        *,
        bucket_name: str | None = None,
    ) -> Path:
        """Download a file from Cloud Storage.

        Args:
            remote_path: Source path (blob name or full gs:// URI).
            local_path: Local destination path.
            bucket_name: Override default bucket.

        Returns:
            Path to the downloaded file.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Parse remote path
        if remote_path.startswith("gs://"):
            bucket_name, remote_path = self._parse_gcs_uri(remote_path)

        if self._use_mock:
            fallback_path = Path("storage") / (bucket_name or self.config.storage.bucket or "default") / remote_path
            if fallback_path.exists():
                local_path.write_bytes(fallback_path.read_bytes())
                return local_path
            raise FileNotFoundError(f"Fallback file not found: {fallback_path}")

        bucket = self._get_bucket(bucket_name)
        blob = bucket.blob(remote_path)

        def do_download() -> None:
            blob.download_to_filename(str(local_path))

        self._retry_operation(do_download)

        logger.info(f"Downloaded gs://{bucket.name}/{remote_path} to {local_path}")
        return local_path

    def exists(
        self,
        remote_path: str,
        *,
        bucket_name: str | None = None,
    ) -> bool:
        """Check if a file exists in Cloud Storage.

        Args:
            remote_path: Path to check (blob name or full gs:// URI).
            bucket_name: Override default bucket.

        Returns:
            True if file exists.
        """
        if remote_path.startswith("gs://"):
            bucket_name, remote_path = self._parse_gcs_uri(remote_path)

        if self._use_mock:
            fallback_path = Path("storage") / (bucket_name or self.config.storage.bucket or "default") / remote_path
            return fallback_path.exists()

        bucket = self._get_bucket(bucket_name)
        blob = bucket.blob(remote_path)
        return bool(blob.exists())

    def delete(
        self,
        remote_path: str,
        *,
        bucket_name: str | None = None,
    ) -> bool:
        """Delete a file from Cloud Storage.

        Args:
            remote_path: Path to delete (blob name or full gs:// URI).
            bucket_name: Override default bucket.

        Returns:
            True if deletion was successful.
        """
        if remote_path.startswith("gs://"):
            bucket_name, remote_path = self._parse_gcs_uri(remote_path)

        if self._use_mock:
            fallback_path = Path("storage") / (bucket_name or self.config.storage.bucket or "default") / remote_path
            if fallback_path.exists():
                fallback_path.unlink()
                return True
            return False

        bucket = self._get_bucket(bucket_name)
        blob = bucket.blob(remote_path)

        if not blob.exists():
            return False

        blob.delete()
        logger.info(f"Deleted gs://{bucket.name}/{remote_path}")
        return True

    def list_files(
        self,
        prefix: str,
        *,
        bucket_name: str | None = None,
        max_results: int = 1000,
    ) -> list[str]:
        """List files with a given prefix.

        Args:
            prefix: Path prefix to filter by.
            bucket_name: Override default bucket.
            max_results: Maximum number of files to return.

        Returns:
            List of file paths (without gs:// prefix).
        """
        if prefix.startswith("gs://"):
            bucket_name, prefix = self._parse_gcs_uri(prefix)

        if self._use_mock:
            fallback_dir = Path("storage") / (bucket_name or self.config.storage.bucket or "default") / prefix
            if fallback_dir.exists():
                return [
                    str(p.relative_to(fallback_dir.parent))
                    for p in fallback_dir.rglob("*")
                    if p.is_file()
                ][:max_results]
            return []

        bucket = self._get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix, max_results=max_results)
        return [blob.name for blob in blobs]

    def upload_directory(
        self,
        local_dir: str | Path,
        remote_prefix: str,
        *,
        bucket_name: str | None = None,
    ) -> list[str]:
        """Upload an entire directory to Cloud Storage.

        Args:
            local_dir: Local directory to upload.
            remote_prefix: Destination prefix in cloud storage.
            bucket_name: Override default bucket.

        Returns:
            List of uploaded file URIs.
        """
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {local_dir}")

        uploaded = []
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(local_dir)
                remote_path = f"{remote_prefix}/{relative}".replace("\\", "/")
                uri = self.upload_file(
                    file_path, remote_path, bucket_name=bucket_name
                )
                uploaded.append(uri)

        logger.info(f"Uploaded {len(uploaded)} files from {local_dir}")
        return uploaded

    def download_directory(
        self,
        remote_prefix: str,
        local_dir: str | Path,
        *,
        bucket_name: str | None = None,
    ) -> list[Path]:
        """Download all files with a prefix to a local directory.

        Args:
            remote_prefix: Source prefix in cloud storage.
            local_dir: Local destination directory.
            bucket_name: Override default bucket.

        Returns:
            List of downloaded file paths.
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        files = self.list_files(remote_prefix, bucket_name=bucket_name)
        downloaded = []

        for remote_path in files:
            relative = remote_path[len(remote_prefix):].lstrip("/")
            local_path = local_dir / relative
            self.download_file(remote_path, local_path, bucket_name=bucket_name)
            downloaded.append(local_path)

        logger.info(f"Downloaded {len(downloaded)} files to {local_dir}")
        return downloaded

    # Checkpoint-specific methods

    def save_checkpoint(
        self,
        local_path: str | Path,
        checkpoint_name: str,
    ) -> str:
        """Save a checkpoint to Cloud Storage.

        Args:
            local_path: Local checkpoint file.
            checkpoint_name: Name for the checkpoint.

        Returns:
            Full gs:// URI of the uploaded checkpoint.
        """
        remote_path = f"{self.config.storage.checkpoint_prefix}/{checkpoint_name}"
        return self.upload_file(local_path, remote_path)

    def load_checkpoint(
        self,
        checkpoint_name: str,
        local_path: str | Path,
    ) -> Path:
        """Load a checkpoint from Cloud Storage.

        Args:
            checkpoint_name: Name of the checkpoint.
            local_path: Local destination path.

        Returns:
            Path to the downloaded checkpoint.
        """
        remote_path = f"{self.config.storage.checkpoint_prefix}/{checkpoint_name}"
        return self.download_file(remote_path, local_path)

    def list_checkpoints(self, max_results: int = 100) -> list[str]:
        """List available checkpoints.

        Args:
            max_results: Maximum number of checkpoints to return.

        Returns:
            List of checkpoint names.
        """
        prefix = self.config.storage.checkpoint_prefix
        files = self.list_files(prefix, max_results=max_results)
        # Strip prefix
        return [f[len(prefix):].lstrip("/") for f in files]


# Protocol compliance check
def _check_protocol_compliance() -> None:
    """Verify CloudStorageClient implements CloudStorageProtocol."""
    client: CloudStorageProtocol = CloudStorageClient(  # type: ignore
        GCPConfig(), allow_fallback=True
    )
    assert isinstance(client, CloudStorageProtocol)
