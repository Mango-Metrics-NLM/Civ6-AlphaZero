"""Cloud provider protocol definitions.

This module defines protocols (interfaces) that cloud providers must implement,
enabling backwards-compatible, pluggable cloud infrastructure.

Using Protocol allows type checking while maintaining flexibility for
different implementations (GCP, AWS, Azure, or mock implementations for testing).

Example:
    from alpha_zero.cloud.protocols import CloudTrainerProtocol

    def submit_job(trainer: CloudTrainerProtocol) -> str:
        return trainer.submit_training_job(script_path="train.py")
"""

from __future__ import annotations

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
