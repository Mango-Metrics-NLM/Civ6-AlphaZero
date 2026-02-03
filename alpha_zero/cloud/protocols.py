"""Cloud provider protocol definitions.

This module defines protocols (interfaces) for cloud infrastructure, enabling
backwards-compatible, pluggable cloud operations across different providers.

Protocols defined:
- CloudStorageProtocol: Checkpoint and artifact storage
- CloudTrainerProtocol: Training job management
- CloudMonitoringProtocol: Metrics and logging
- PreemptionHandlerProtocol: Spot/preemptible instance handling

Using Protocol allows type checking while maintaining flexibility for
different implementations (GCP, AWS, Azure, or mock implementations for testing).

Example:
    from alpha_zero.cloud.protocols import CloudTrainerProtocol

    def submit_job(trainer: CloudTrainerProtocol) -> str:
        return trainer.submit_training_job(script_path="train.py")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class JobStatus(Enum):
    """Training job status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class TrainingJobInfo:
    """Information about a training job.

    This is a provider-agnostic representation of job status
    that can be used by any cloud provider.
    """

    job_id: str
    job_name: str
    status: JobStatus
    created_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error_message: str | None = None
    metrics: dict[str, Any] | None = None
    logs_uri: str | None = None
    output_uri: str | None = None

    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in (
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        )

    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.SUCCEEDED

    def duration_seconds(self) -> float | None:
        """Get job duration in seconds, if available."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None


@runtime_checkable
class CloudStorageProtocol(Protocol):
    """Protocol for cloud storage operations.

    Provides a unified interface for uploading, downloading,
    and managing files in cloud storage.
    """

    def upload_file(
        self,
        local_path: str | Path,
        remote_path: str,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload a file to cloud storage.

        Args:
            local_path: Local file path.
            remote_path: Destination path in cloud storage.
            content_type: MIME type of the file.
            metadata: Custom metadata to attach.

        Returns:
            Full URI of the uploaded file.
        """
        ...

    def download_file(
        self,
        remote_path: str,
        local_path: str | Path,
    ) -> Path:
        """Download a file from cloud storage.

        Args:
            remote_path: Source path in cloud storage.
            local_path: Local destination path.

        Returns:
            Path to the downloaded file.
        """
        ...

    def exists(self, remote_path: str) -> bool:
        """Check if a file exists in cloud storage.

        Args:
            remote_path: Path to check.

        Returns:
            True if file exists.
        """
        ...

    def delete(self, remote_path: str) -> bool:
        """Delete a file from cloud storage.

        Args:
            remote_path: Path to delete.

        Returns:
            True if deletion was successful.
        """
        ...

    def list_files(
        self,
        prefix: str,
        *,
        max_results: int = 1000,
    ) -> list[str]:
        """List files with a given prefix.

        Args:
            prefix: Path prefix to filter by.
            max_results: Maximum number of files to return.

        Returns:
            List of file paths.
        """
        ...

    def upload_directory(
        self,
        local_dir: str | Path,
        remote_prefix: str,
    ) -> list[str]:
        """Upload an entire directory to cloud storage.

        Args:
            local_dir: Local directory to upload.
            remote_prefix: Destination prefix in cloud storage.

        Returns:
            List of uploaded file URIs.
        """
        ...

    def download_directory(
        self,
        remote_prefix: str,
        local_dir: str | Path,
    ) -> list[Path]:
        """Download all files with a prefix to a local directory.

        Args:
            remote_prefix: Source prefix in cloud storage.
            local_dir: Local destination directory.

        Returns:
            List of downloaded file paths.
        """
        ...


@runtime_checkable
class CloudTrainerProtocol(Protocol):
    """Protocol for cloud-based training job management.

    Any cloud provider (GCP, AWS, Azure) should implement
    this protocol to enable pluggable training infrastructure.
    """

    def submit_training_job(
        self,
        script_path: str | Path,
        *,
        args: list[str] | None = None,
        machine_type: str | None = None,
        accelerator_type: str | None = None,
        accelerator_count: int | None = None,
        base_output_dir: str | None = None,
        job_name: str | None = None,
        environment_variables: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a training job to the cloud.

        Args:
            script_path: Path to the training script.
            args: Command-line arguments for the script.
            machine_type: VM machine type.
            accelerator_type: GPU/TPU type.
            accelerator_count: Number of accelerators.
            base_output_dir: Output directory for artifacts.
            job_name: Custom job name (auto-generated if not provided).
            environment_variables: Environment variables for the job.
            **kwargs: Provider-specific options.

        Returns:
            Job ID or name for tracking.
        """
        ...

    def get_job_status(self, job_id: str) -> TrainingJobInfo:
        """Get the current status of a training job.

        Args:
            job_id: The job ID returned from submit_training_job.

        Returns:
            TrainingJobInfo with current status.
        """
        ...

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job.

        Args:
            job_id: The job ID to cancel.

        Returns:
            True if cancellation was successful.
        """
        ...

    def list_jobs(
        self,
        *,
        status_filter: JobStatus | None = None,
        max_results: int = 100,
    ) -> list[TrainingJobInfo]:
        """List training jobs.

        Args:
            status_filter: Filter by job status.
            max_results: Maximum number of jobs to return.

        Returns:
            List of TrainingJobInfo objects.
        """
        ...


@runtime_checkable
class CloudMonitoringProtocol(Protocol):
    """Protocol for cloud monitoring and logging backends.

    Implementations must provide methods for:
    - Logging training metrics
    - Logging events with severity levels
    - Flushing pending logs
    """

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, float],
        labels: dict[str, str] | None = None,
    ) -> None:
        """Log training metrics.

        Args:
            step: Training step number.
            metrics: Dictionary of metric names to values.
            labels: Optional labels for filtering.
        """
        ...

    def log_event(
        self,
        severity: str,
        message: str,
        labels: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Log an event with severity level.

        Args:
            severity: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            message: Event message.
            labels: Optional labels for filtering.
            payload: Optional structured data.
        """
        ...

    def flush(self) -> None:
        """Flush any pending logs to the backend."""
        ...


@runtime_checkable
class PreemptionHandlerProtocol(Protocol):
    """Protocol for preemption detection and handling.

    Implementations must provide methods for:
    - Starting/stopping preemption monitoring
    - Checking preemption status
    - Registering cleanup callbacks
    """

    def start_monitoring(self) -> None:
        """Start monitoring for preemption signals."""
        ...

    def stop_monitoring(self) -> None:
        """Stop monitoring for preemption signals."""
        ...

    def is_preemption_pending(self) -> bool:
        """Check if preemption is pending.

        Returns:
            True if preemption notice has been received.
        """
        ...

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback for preemption.

        Args:
            callback: Function to call during graceful shutdown.
        """
        ...

    def get_time_remaining(self) -> float | None:
        """Get estimated time remaining before termination.

        Returns:
            Seconds remaining, or None if not preempted.
        """
        ...
