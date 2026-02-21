"""Mock implementations for GCP cloud services.

This module provides mock implementations for:
- Google Cloud Storage (GCS)
- Vertex AI
- Cloud Logging
- Cloud Monitoring

These mocks enable testing cloud functionality without GCP dependencies.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class MockGCSBlob:
    """Mock GCS blob for testing."""

    def __init__(
        self,
        name: str,
        bucket: MockBucket,
    ) -> None:
        """Initialize mock blob.

        Args:
            name: Blob name/path.
            bucket: Parent bucket.
        """
        self.name = name
        self._bucket = bucket
        self._metadata: dict[str, str] = {}
        self.size = 0
        self.updated: datetime | None = None

    @property
    def metadata(self) -> dict[str, str]:
        """Get blob metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, str]) -> None:
        """Set blob metadata."""
        self._metadata = value

    def upload_from_file(self, file_obj: io.IOBase) -> None:
        """Upload from file object."""
        data = file_obj.read()
        if isinstance(data, str):
            data = data.encode()
        self._bucket._storage[self.name] = data
        self._bucket._metadata[self.name] = self._metadata.copy()
        self.size = len(data)
        self.updated = datetime.now()

    def download_to_file(self, file_obj: io.IOBase) -> None:
        """Download to file object."""
        if self.name not in self._bucket._storage:
            raise FileNotFoundError(f"Blob not found: {self.name}")
        data = self._bucket._storage[self.name]
        file_obj.write(data)

    def upload_from_filename(self, filename: str) -> None:
        """Upload from filename."""
        with open(filename, "rb") as f:
            self.upload_from_file(f)

    def download_to_filename(self, filename: str) -> None:
        """Download to filename."""
        if self.name not in self._bucket._storage:
            raise FileNotFoundError(f"Blob not found: {self.name}")
        with open(filename, "wb") as f:
            f.write(self._bucket._storage[self.name])

    def exists(self) -> bool:
        """Check if blob exists."""
        return self.name in self._bucket._storage

    def delete(self) -> None:
        """Delete the blob."""
        if self.name in self._bucket._storage:
            del self._bucket._storage[self.name]
        if self.name in self._bucket._metadata:
            del self._bucket._metadata[self.name]


class MockBucket:
    """Mock GCS bucket for testing."""

    def __init__(self, name: str) -> None:
        """Initialize mock bucket.

        Args:
            name: Bucket name.
        """
        self.name = name
        self._storage: dict[str, bytes] = {}
        self._metadata: dict[str, dict[str, str]] = {}

    def blob(self, name: str) -> MockGCSBlob:
        """Get or create blob."""
        blob = MockGCSBlob(name, self)
        if name in self._storage:
            blob.size = len(self._storage[name])
            blob._metadata = self._metadata.get(name, {})
        return blob

    def list_blobs(self, prefix: str = "") -> list[MockGCSBlob]:
        """List blobs with prefix."""
        blobs = []
        for name in self._storage:
            if name.startswith(prefix):
                blob = self.blob(name)
                blobs.append(blob)
        return blobs


class MockGCSClient:
    """Mock GCS client for testing."""

    def __init__(self, project: str | None = None) -> None:
        """Initialize mock client.

        Args:
            project: GCP project ID (ignored in mock).
        """
        self.project = project
        self._buckets: dict[str, MockBucket] = {}

    def bucket(self, name: str) -> MockBucket:
        """Get or create bucket."""
        if name not in self._buckets:
            self._buckets[name] = MockBucket(name)
        return self._buckets[name]


@dataclass
class MockVertexJob:
    """Mock Vertex AI custom job."""

    display_name: str
    resource_name: str
    state: str = "JOB_STATE_PENDING"
    create_time: datetime = field(default_factory=datetime.now)
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None

    def run(self, sync: bool = False) -> None:
        """Run the mock job."""
        self.state = "JOB_STATE_RUNNING"
        self.start_time = datetime.now()

        if sync:
            time.sleep(0.01)  # Minimal delay for sync mode
            self.state = "JOB_STATE_SUCCEEDED"
            self.end_time = datetime.now()

    def cancel(self) -> None:
        """Cancel the mock job."""
        self.state = "JOB_STATE_CANCELLED"
        self.end_time = datetime.now()

    @property
    def state_obj(self) -> Any:
        """Get state as object with name attribute."""
        return type("State", (), {"name": self.state})()


class MockAIPlatform:
    """Mock AI Platform SDK for testing."""

    _jobs: dict[str, MockVertexJob] = {}
    _initialized: bool = False
    _project: str = ""
    _location: str = ""
    _job_counter: int = 0

    @classmethod
    def reset(cls) -> None:
        """Reset mock state."""
        cls._jobs = {}
        cls._initialized = False
        cls._project = ""
        cls._location = ""
        cls._job_counter = 0

    @classmethod
    def init(
        cls,
        project: str = "",
        location: str = "us-central1",
        staging_bucket: str | None = None,
    ) -> None:
        """Initialize mock AI Platform.

        Args:
            project: GCP project ID.
            location: GCP region.
            staging_bucket: Optional staging bucket (ignored).
        """
        cls._initialized = True
        cls._project = project
        cls._location = location

    @classmethod
    def CustomJob(  # noqa: N802
        cls,
        display_name: str,
        worker_pool_specs: list[dict[str, Any]],
        **kwargs: Any,
    ) -> MockVertexJob:
        """Create mock custom job.

        Args:
            display_name: Job display name.
            worker_pool_specs: Worker pool specifications (ignored).
            **kwargs: Additional arguments (ignored).

        Returns:
            Mock custom job instance.
        """
        cls._job_counter += 1
        job = MockVertexJob(
            display_name=display_name,
            resource_name=f"projects/{cls._project}/locations/{cls._location}/customJobs/mock-{cls._job_counter}",
        )
        cls._jobs[job.resource_name] = job
        return job

    @classmethod
    def get_job(cls, resource_name: str) -> MockVertexJob | None:
        """Get job by resource name."""
        return cls._jobs.get(resource_name)


class MockCloudLoggingClient:
    """Mock Cloud Logging client for testing."""

    def __init__(self, project: str | None = None) -> None:
        """Initialize mock client."""
        self.project = project
        self._loggers: dict[str, MockCloudLogger] = {}

    def logger(self, name: str) -> MockCloudLogger:
        """Get or create logger."""
        if name not in self._loggers:
            self._loggers[name] = MockCloudLogger(name)
        return self._loggers[name]

    def setup_logging(self) -> None:
        """Set up Python logging (no-op for mock)."""
        pass


class MockCloudLogger:
    """Mock Cloud Logging logger."""

    def __init__(self, name: str) -> None:
        """Initialize mock logger."""
        self.name = name
        self.entries: list[dict[str, Any]] = []

    def log_struct(
        self,
        info: dict[str, Any],
        severity: str = "INFO",
    ) -> None:
        """Log structured data."""
        self.entries.append(
            {
                "type": "struct",
                "severity": severity,
                "payload": info,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_text(self, text: str, severity: str = "INFO") -> None:
        """Log text message."""
        self.entries.append(
            {
                "type": "text",
                "severity": severity,
                "message": text,
                "timestamp": datetime.now().isoformat(),
            }
        )


class MockStorageBackend:
    """In-memory storage backend for testing GCS operations.

    Stores blobs keyed by ``"{bucket}/{blob_path}"`` so multiple buckets
    are supported in a single instance.  All five methods match the
    interface expected by GCSModelStorage, GCSTrajectoryStorage, and
    GCSCheckpointStorage.
    """

    def __init__(self) -> None:
        """Initialize empty in-memory store."""
        self._storage: dict[str, bytes] = {}

    def upload_blob(self, bucket: str, blob_path: str, data: bytes) -> str:
        """Store *data* and return a ``gs://`` URI."""
        key = f"{bucket}/{blob_path}"
        self._storage[key] = data
        return f"gs://{key}"

    def download_blob(self, bucket: str, blob_path: str) -> bytes:
        """Return stored bytes or raise GCSStorageError if missing."""
        from alpha_zero.cloud.gcp.exceptions import GCSStorageError

        key = f"{bucket}/{blob_path}"
        if key not in self._storage:
            raise GCSStorageError(
                f"Blob not found: {blob_path}",
                bucket=bucket,
                blob_path=blob_path,
                operation="download",
            )
        return self._storage[key]

    def blob_exists(self, bucket: str, blob_path: str) -> bool:
        """Return True if the blob is present."""
        return f"{bucket}/{blob_path}" in self._storage

    def list_blobs(self, bucket: str, prefix: str) -> list[str]:
        """List blob paths (without bucket prefix) that match *prefix*."""
        results = []
        prefix_key = f"{bucket}/{prefix}"
        for key in self._storage:
            if key.startswith(prefix_key):
                # Strip the leading "bucket/" segment
                results.append(key.split("/", 1)[1])
        return results

    def delete_blob(self, bucket: str, blob_path: str) -> None:
        """Delete blob if present (no-op if missing)."""
        self._storage.pop(f"{bucket}/{blob_path}", None)


class MockCloudMonitoringClient:
    """Mock Cloud Monitoring client for testing."""

    def __init__(self, project: str | None = None) -> None:
        """Initialize mock client."""
        self.project = project
        self.time_series: list[dict[str, Any]] = []

    def create_time_series(self, request: dict[str, Any]) -> None:
        """Create time series (store for verification)."""
        self.time_series.append(request)
