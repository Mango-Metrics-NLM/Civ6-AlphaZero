"""Vertex AI job launcher for cloud training.

This module provides:
- VertexJobLauncher: Submit and monitor Vertex AI training jobs
- JobInfo: Job status and metadata tracking

Usage:
    from alpha_zero.cloud.vertex_launcher import VertexJobLauncher

    launcher = VertexJobLauncher(
        project_id="my-project",
        region="us-central1",
    )

    job_id = launcher.submit_job(
        display_name="civ6-training-run-1",
        container_image="gcr.io/my-project/civ6-trainer:latest",
        args=["--config", "configs/cloud_training.yaml"],
    )

    status = launcher.get_job_status(job_id)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Protocol

from alpha_zero.cloud.config import CloudConfig

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Vertex AI job status."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CANCELLING = "cancelling"
    UNKNOWN = "unknown"


@dataclass
class JobInfo:
    """Information about a Vertex AI training job.

    Attributes:
        job_id: Unique job identifier (resource name).
        display_name: Human-readable job name.
        status: Current job status.
        created_at: Job creation timestamp.
        started_at: Job start timestamp (when execution began).
        ended_at: Job completion timestamp.
        error_message: Error message if job failed.
        metrics: Training metrics from the job.
        logs_uri: URI to job logs in Cloud Logging.
        output_uri: URI to job output artifacts.
    """

    job_id: str
    display_name: str
    status: JobStatus
    created_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error_message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    logs_uri: str = ""
    output_uri: str = ""

    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }

    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.SUCCEEDED

    def duration_seconds(self) -> float | None:
        """Get job duration in seconds.

        Returns:
            Duration in seconds, or None if not available.
        """
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None


class AIplatformProtocol(Protocol):
    """Protocol for AI Platform SDK."""

    def init(
        self,
        project: str,
        location: str,
        staging_bucket: str | None = None,
    ) -> None:
        """Initialize AI Platform."""
        ...


class CustomJobProtocol(Protocol):
    """Protocol for CustomJob class."""

    resource_name: str
    state: Any
    create_time: Any
    start_time: Any
    end_time: Any
    error: Any

    def run(self, sync: bool = False) -> None:
        """Run the job."""
        ...


class MockCustomJob:
    """Mock Vertex AI CustomJob for testing."""

    _job_counter: ClassVar[int] = 0

    def __init__(
        self,
        display_name: str,
        worker_pool_specs: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Initialize mock job.

        Args:
            display_name: Job display name.
            worker_pool_specs: Worker pool specifications.
            **kwargs: Additional arguments.
        """
        MockCustomJob._job_counter += 1
        self.display_name = display_name
        self.worker_pool_specs = worker_pool_specs
        self.resource_name = (
            f"projects/mock/locations/mock/customJobs/mock-{MockCustomJob._job_counter}"
        )
        self._state = "JOB_STATE_PENDING"
        self._create_time = datetime.now()
        self._start_time: datetime | None = None
        self._end_time: datetime | None = None
        self._error: Any = None

    @property
    def state(self) -> Any:
        """Get job state."""
        return type("State", (), {"name": self._state})()

    @property
    def create_time(self) -> datetime:
        """Get creation time."""
        return self._create_time

    @property
    def start_time(self) -> datetime | None:
        """Get start time."""
        return self._start_time

    @property
    def end_time(self) -> datetime | None:
        """Get end time."""
        return self._end_time

    @property
    def error(self) -> Any:
        """Get error if any."""
        return self._error

    def run(self, sync: bool = False) -> None:
        """Run the mock job.

        Args:
            sync: Whether to run synchronously.
        """
        self._state = "JOB_STATE_RUNNING"
        self._start_time = datetime.now()

        if sync:
            # Simulate completion
            time.sleep(0.1)
            self._state = "JOB_STATE_SUCCEEDED"
            self._end_time = datetime.now()


class MockAIPlatform:
    """Mock AI Platform SDK for testing."""

    _jobs: ClassVar[dict[str, MockCustomJob]] = {}
    _initialized: ClassVar[bool] = False

    @classmethod
    def init(
        cls,
        project: str,
        location: str,
        staging_bucket: str | None = None,
    ) -> None:
        """Initialize mock AI Platform.

        Args:
            project: GCP project ID.
            location: GCP region.
            staging_bucket: Optional staging bucket.
        """
        cls._initialized = True

    @classmethod
    def CustomJob(  # noqa: N802
        cls,
        display_name: str,
        worker_pool_specs: list[dict[str, Any]],
        **kwargs: Any,
    ) -> MockCustomJob:
        """Create mock custom job.

        Args:
            display_name: Job display name.
            worker_pool_specs: Worker pool specifications.
            **kwargs: Additional arguments.

        Returns:
            Mock custom job instance.
        """
        job = MockCustomJob(display_name, worker_pool_specs, **kwargs)
        cls._jobs[job.resource_name] = job
        return job

    @classmethod
    def get_job(cls, resource_name: str) -> MockCustomJob | None:
        """Get job by resource name.

        Args:
            resource_name: Job resource name.

        Returns:
            Mock job instance or None.
        """
        return cls._jobs.get(resource_name)


def _get_aiplatform(force_mock: bool = False) -> Any:
    """Get AI Platform SDK, falling back to mock if unavailable.

    Args:
        force_mock: Force use of mock SDK.

    Returns:
        AI Platform SDK module or mock.
    """
    if force_mock:
        return MockAIPlatform

    try:
        from google.cloud import aiplatform

        return aiplatform
    except ImportError:
        logger.warning("google-cloud-aiplatform not installed, using mock")
        return MockAIPlatform


class VertexJobLauncher:
    """Launches and monitors Vertex AI training jobs.

    Features:
    - Submit custom training jobs
    - Monitor job status
    - Wait for completion
    - List historical jobs
    - Cancel running jobs

    Attributes:
        project_id: GCP project ID.
        region: GCP region.
        staging_bucket: Optional GCS staging bucket.
    """

    def __init__(
        self,
        project_id: str,
        region: str = "us-central1",
        staging_bucket: str | None = None,
        force_mock: bool = False,
    ) -> None:
        """Initialize job launcher.

        Args:
            project_id: GCP project ID.
            region: GCP region.
            staging_bucket: Optional GCS bucket for staging.
            force_mock: Force use of mock SDK for testing.
        """
        self.project_id = project_id
        self.region = region
        self.staging_bucket = staging_bucket
        self._force_mock = force_mock

        # Initialize SDK
        self._aiplatform = _get_aiplatform(force_mock=force_mock)
        self._aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=staging_bucket,
        )

        # Track submitted jobs
        self._jobs: dict[str, Any] = {}

    def submit_job(
        self,
        display_name: str,
        container_image: str,
        command: list[str] | None = None,
        args: list[str] | None = None,
        machine_type: str = "n1-standard-8",
        accelerator_type: str = "NVIDIA_TESLA_T4",
        accelerator_count: int = 1,
        boot_disk_size_gb: int = 100,
        env_vars: dict[str, str] | None = None,
        service_account: str | None = None,
        network: str | None = None,
        base_output_dir: str | None = None,
        replica_count: int = 1,
        labels: dict[str, str] | None = None,
    ) -> str:
        """Submit a training job to Vertex AI.

        Args:
            display_name: Human-readable job name.
            container_image: Container image URI.
            command: Command to run in container.
            args: Arguments for the command.
            machine_type: GCE machine type.
            accelerator_type: GPU accelerator type.
            accelerator_count: Number of GPUs.
            boot_disk_size_gb: Boot disk size in GB.
            env_vars: Environment variables to set.
            service_account: Service account for the job.
            network: VPC network for the job.
            base_output_dir: Base output directory in GCS.
            replica_count: Number of worker replicas.
            labels: Custom labels for the job.

        Returns:
            Job resource name (job ID).
        """
        # Build worker pool spec
        machine_spec: dict[str, Any] = {
            "machine_type": machine_type,
        }

        if accelerator_count > 0 and accelerator_type:
            machine_spec["accelerator_type"] = accelerator_type
            machine_spec["accelerator_count"] = accelerator_count

        container_spec: dict[str, Any] = {
            "image_uri": container_image,
        }

        if command:
            container_spec["command"] = command
        if args:
            container_spec["args"] = args
        if env_vars:
            container_spec["env"] = [{"name": k, "value": v} for k, v in env_vars.items()]

        worker_pool_specs = [
            {
                "machine_spec": machine_spec,
                "replica_count": replica_count,
                "container_spec": container_spec,
                "disk_spec": {
                    "boot_disk_type": "pd-ssd",
                    "boot_disk_size_gb": boot_disk_size_gb,
                },
            }
        ]

        # Build job kwargs
        job_kwargs: dict[str, Any] = {
            "display_name": display_name,
            "worker_pool_specs": worker_pool_specs,
        }

        if base_output_dir:
            job_kwargs["base_output_dir"] = base_output_dir
        if labels:
            job_kwargs["labels"] = labels
        if service_account:
            job_kwargs["service_account"] = service_account
        if network:
            job_kwargs["network"] = network

        # Create and run job
        custom_job = self._aiplatform.CustomJob(**job_kwargs)
        custom_job.run(sync=False)

        job_id = custom_job.resource_name
        self._jobs[job_id] = custom_job

        logger.info(f"Submitted job: {display_name} (id={job_id})")
        return job_id  # type: ignore[no-any-return]

    def submit_from_config(
        self,
        display_name: str,
        container_image: str,
        config: CloudConfig,
        args: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> str:
        """Submit job using CloudConfig settings.

        Args:
            display_name: Human-readable job name.
            container_image: Container image URI.
            config: Cloud configuration.
            args: Additional command arguments.
            env_vars: Additional environment variables.

        Returns:
            Job resource name (job ID).
        """
        # Merge environment variables
        all_env_vars = {
            "CIV6AZ_GCP_PROJECT_ID": config.project_id,
            "CIV6AZ_GCP_REGION": config.region,
            "CIV6AZ_GCS_BUCKET": config.gcs.bucket_name,
            "CIV6AZ_TRAINING_BATCH_SIZE": str(config.training.batch_size),
            "CIV6AZ_TRAINING_LR": str(config.training.learning_rate),
        }
        if env_vars:
            all_env_vars.update(env_vars)

        return self.submit_job(
            display_name=display_name,
            container_image=container_image,
            args=args,
            machine_type=config.vertex.machine_type,
            accelerator_type=config.vertex.accelerator_type,
            accelerator_count=config.vertex.accelerator_count,
            boot_disk_size_gb=config.vertex.boot_disk_size_gb,
            env_vars=all_env_vars,
            service_account=config.vertex.service_account or None,
            network=config.vertex.network or None,
            labels=config.custom_labels,
        )

    def get_job_status(self, job_id: str) -> JobInfo:
        """Get current status of a job.

        Args:
            job_id: Job resource name.

        Returns:
            JobInfo with current status.

        Raises:
            KeyError: If job not found.
        """
        job = self._jobs.get(job_id)
        if job is None:
            # Try to fetch from API (for mock, this returns None)
            if hasattr(self._aiplatform, "get_job"):
                job = self._aiplatform.get_job(job_id)
            if job is None:
                raise KeyError(f"Job not found: {job_id}")

        # Map state to JobStatus
        state_name = job.state.name if hasattr(job.state, "name") else str(job.state)
        status_map = {
            "JOB_STATE_PENDING": JobStatus.PENDING,
            "JOB_STATE_QUEUED": JobStatus.QUEUED,
            "JOB_STATE_RUNNING": JobStatus.RUNNING,
            "JOB_STATE_SUCCEEDED": JobStatus.SUCCEEDED,
            "JOB_STATE_FAILED": JobStatus.FAILED,
            "JOB_STATE_CANCELLED": JobStatus.CANCELLED,
            "JOB_STATE_CANCELLING": JobStatus.CANCELLING,
        }
        status = status_map.get(state_name, JobStatus.UNKNOWN)

        # Extract error message if failed
        error_message = ""
        if job.error:
            error_message = str(job.error.message if hasattr(job.error, "message") else job.error)

        return JobInfo(
            job_id=job_id,
            display_name=job.display_name,
            status=status,
            created_at=job.create_time,
            started_at=job.start_time,
            ended_at=job.end_time,
            error_message=error_message,
        )

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 60.0,
        timeout: float | None = None,
        callback: Any | None = None,
    ) -> JobInfo:
        """Wait for job completion with polling.

        Args:
            job_id: Job resource name.
            poll_interval: Seconds between status checks.
            timeout: Maximum wait time in seconds (None = unlimited).
            callback: Optional callback function(JobInfo) called on each poll.

        Returns:
            Final JobInfo when job completes.

        Raises:
            TimeoutError: If timeout exceeded.
        """
        start_time = time.time()

        while True:
            info = self.get_job_status(job_id)

            if callback:
                callback(info)

            if info.is_terminal():
                return info

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")

            logger.info(f"Job {job_id} status: {info.status.value}")
            time.sleep(poll_interval)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job resource name.

        Returns:
            True if cancellation was requested.
        """
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning(f"Job not found for cancellation: {job_id}")
            return False

        try:
            if hasattr(job, "cancel"):
                job.cancel()
            logger.info(f"Cancelled job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

    def list_jobs(
        self,
        filter_str: str | None = None,
        limit: int = 100,
    ) -> list[JobInfo]:
        """List recent jobs.

        Args:
            filter_str: Optional filter string.
            limit: Maximum number of jobs to return.

        Returns:
            List of JobInfo for matching jobs.
        """
        # For mock, just return tracked jobs
        jobs = []
        for job_id in list(self._jobs.keys())[:limit]:
            try:
                jobs.append(self.get_job_status(job_id))
            except KeyError:
                continue
        return jobs


def create_launcher_from_config(
    config: CloudConfig,
    force_mock: bool = False,
) -> VertexJobLauncher:
    """Create VertexJobLauncher from CloudConfig.

    Args:
        config: Cloud configuration.
        force_mock: Force use of mock SDK.

    Returns:
        Configured VertexJobLauncher.
    """
    staging_bucket = f"gs://{config.gcs.bucket_name}" if config.gcs.bucket_name else None

    return VertexJobLauncher(
        project_id=config.project_id,
        region=config.region,
        staging_bucket=staging_bucket,
        force_mock=force_mock,
    )
