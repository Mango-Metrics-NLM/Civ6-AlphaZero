"""Vertex AI Custom Training job management.

Provides VertexAITrainer for submitting and managing training jobs
on Google Cloud Vertex AI. Follows the force_mock pattern for testing
without requiring GCP SDK.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from alpha_zero.cloud.config import CloudConfig
from alpha_zero.cloud.exceptions import CloudConfigError, CloudConnectionError

logger = logging.getLogger(__name__)

# SDK availability check
VERTEX_AVAILABLE = False
try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import CustomJob

    VERTEX_AVAILABLE = True
except ImportError:
    aiplatform = None
    CustomJob = None


class JobStatus(Enum):
    """Status of a Vertex AI training job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class VertexJobConfig:
    """Configuration for a Vertex AI training job.

    Attributes:
        display_name: Job display name.
        script_path: Path to training script.
        args: Command-line arguments for script.
        machine_type: GCE machine type.
        accelerator_type: GPU/TPU type.
        accelerator_count: Number of accelerators.
        replica_count: Number of worker replicas.
        boot_disk_size_gb: Boot disk size in GB.
        container_uri: Custom container image URI (optional).
        base_output_dir: GCS path for job outputs.
        service_account: Service account for the job.
        enable_web_access: Enable web access for debugging.
        environment_variables: Environment variables for workers.
        labels: Job labels for organization.
    """

    display_name: str
    script_path: str = ""
    args: list[str] = field(default_factory=list)
    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    replica_count: int = 1
    boot_disk_size_gb: int = 100
    container_uri: str = ""
    base_output_dir: str = ""
    service_account: str = ""
    enable_web_access: bool = False
    environment_variables: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_cloud_config(
        cls,
        cloud_config: CloudConfig,
        display_name: str,
        script_path: str = "",
        args: list[str] | None = None,
    ) -> VertexJobConfig:
        """Create job config from CloudConfig.

        Args:
            cloud_config: Cloud configuration.
            display_name: Job display name.
            script_path: Path to training script.
            args: Command-line arguments.

        Returns:
            VertexJobConfig instance.
        """
        base_output_dir = ""
        if cloud_config.bucket_name:
            base_output_dir = f"gs://{cloud_config.bucket_name}/{cloud_config.artifact_prefix}jobs"

        return cls(
            display_name=display_name,
            script_path=script_path,
            args=args or [],
            machine_type=cloud_config.machine_type,
            accelerator_type=cloud_config.accelerator_type,
            accelerator_count=cloud_config.accelerator_count,
            replica_count=cloud_config.replica_count,
            base_output_dir=base_output_dir,
            service_account=cloud_config.service_account,
            labels=cloud_config.custom_labels.copy(),
        )


class MockVertexJob:
    """Mock Vertex AI job for testing."""

    _job_counter = 0

    def __init__(self, config: VertexJobConfig) -> None:
        """Initialize mock job."""
        MockVertexJob._job_counter += 1
        base_path = "projects/mock/locations/us-central1/customJobs"
        self.resource_name = f"{base_path}/{MockVertexJob._job_counter}"
        self.display_name = config.display_name
        self._status = JobStatus.PENDING
        self._start_time = time.time()
        self._config = config

    @property
    def state(self) -> str:
        """Get job state (Vertex AI style)."""
        elapsed = time.time() - self._start_time
        if self._status == JobStatus.CANCELLED:
            return "JOB_STATE_CANCELLED"
        elif elapsed < 0.1:
            return "JOB_STATE_PENDING"
        elif elapsed < 0.5:
            return "JOB_STATE_RUNNING"
        else:
            return "JOB_STATE_SUCCEEDED"

    def cancel(self) -> None:
        """Cancel the job."""
        self._status = JobStatus.CANCELLED


class VertexAITrainer:
    """Vertex AI Custom Training job manager.

    Manages submission and monitoring of training jobs on Vertex AI.
    Follows the force_mock pattern for testing without SDK.

    Example:
        >>> config = CloudConfig(project_id="my-project", region="us-central1")
        >>> trainer = VertexAITrainer(config)
        >>> job_config = VertexJobConfig(display_name="training-v1", script_path="train.py")
        >>> job_id = trainer.submit_training_job(job_config)
        >>> status = trainer.get_job_status(job_id)
    """

    def __init__(
        self,
        config: CloudConfig,
        force_mock: bool = False,
    ) -> None:
        """Initialize Vertex AI trainer.

        Args:
            config: Cloud configuration.
            force_mock: Force use of mock for testing.

        Raises:
            CloudConfigError: If configuration is invalid.
        """
        self.config = config
        self._use_mock = force_mock or not VERTEX_AVAILABLE
        self._jobs: dict[str, Any] = {}

        if not self._use_mock:
            if not config.project_id:
                raise CloudConfigError("project_id is required for Vertex AI")

            try:
                aiplatform.init(
                    project=config.project_id,
                    location=config.region,
                )
            except Exception as e:
                raise CloudConnectionError(
                    f"Failed to initialize Vertex AI: {e}",
                    service="vertex",
                    cause=e,
                ) from e

        logger.info(
            "Vertex AI trainer initialized (mock=%s, SDK available=%s)",
            self._use_mock,
            VERTEX_AVAILABLE,
        )

    @property
    def is_mock(self) -> bool:
        """Check if using mock implementation."""
        return self._use_mock

    def _create_job_spec(
        self,
        display_name: str,
        container_uri: str,
        args: list[str] | None = None,
        job_config: VertexJobConfig | None = None,
    ) -> dict[str, Any]:
        """Create a Vertex AI job specification without submitting.

        This method builds the worker pool specification that would be used
        to submit a training job. Useful for validation and testing.

        Args:
            display_name: Job display name.
            container_uri: Container image URI.
            args: Command-line arguments for the container.
            job_config: Optional full job configuration. If provided,
                uses its settings; otherwise uses defaults.

        Returns:
            Dictionary containing the job specification with workerPoolSpecs.
        """
        if job_config is None:
            job_config = VertexJobConfig(display_name=display_name)

        # Build machine spec
        machine_spec: dict[str, Any] = {
            "machine_type": job_config.machine_type,
        }

        # Add accelerator if specified
        if job_config.accelerator_count > 0 and job_config.accelerator_type:
            machine_spec["accelerator_type"] = job_config.accelerator_type
            machine_spec["accelerator_count"] = job_config.accelerator_count

        # Build worker pool spec
        worker_pool_specs: list[dict[str, Any]] = [
            {
                "machine_spec": machine_spec,
                "replica_count": job_config.replica_count,
                "container_spec": {
                    "image_uri": container_uri
                    or "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest",
                    "command": ["python", job_config.script_path] if job_config.script_path else [],
                    "args": args or job_config.args,
                    "env": [
                        {"name": k, "value": v} for k, v in job_config.environment_variables.items()
                    ],
                },
            }
        ]

        return {
            "display_name": display_name,
            "workerPoolSpecs": worker_pool_specs,
            "base_output_dir": job_config.base_output_dir or None,
            "labels": job_config.labels,
        }

    def submit_training_job(
        self,
        job_config: VertexJobConfig,
    ) -> str:
        """Submit a custom training job.

        Args:
            job_config: Job configuration.

        Returns:
            Job resource name/ID.

        Raises:
            CloudConnectionError: If job submission fails.
        """
        if self._use_mock:
            job = MockVertexJob(job_config)
            self._jobs[job.resource_name] = job
            logger.info("Submitted mock training job: %s", job.resource_name)
            return job.resource_name

        try:
            # Create job spec using the shared method
            job_spec = self._create_job_spec(
                display_name=job_config.display_name,
                container_uri=job_config.container_uri,
                args=job_config.args,
                job_config=job_config,
            )

            vertex_job = CustomJob(
                display_name=job_spec["display_name"],
                worker_pool_specs=job_spec["workerPoolSpecs"],
                base_output_dir=job_spec["base_output_dir"],
                labels=job_spec["labels"],
            )

            vertex_job.run(
                service_account=job_config.service_account or None,
                enable_web_access=job_config.enable_web_access,
                sync=False,  # Non-blocking
            )

            self._jobs[vertex_job.resource_name] = vertex_job
            logger.info("Submitted Vertex AI job: %s", vertex_job.resource_name)
            return vertex_job.resource_name

        except Exception as e:
            raise CloudConnectionError(
                f"Failed to submit training job: {e}",
                service="vertex",
                cause=e,
            ) from e

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of a training job.

        Args:
            job_id: Job resource name.

        Returns:
            Current job status.
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.warning("Job not found: %s", job_id)
            return JobStatus.UNKNOWN

        try:
            state = job.state
            status_map = {
                "JOB_STATE_PENDING": JobStatus.PENDING,
                "JOB_STATE_QUEUED": JobStatus.PENDING,
                "JOB_STATE_RUNNING": JobStatus.RUNNING,
                "JOB_STATE_SUCCEEDED": JobStatus.SUCCEEDED,
                "JOB_STATE_FAILED": JobStatus.FAILED,
                "JOB_STATE_CANCELLED": JobStatus.CANCELLED,
                "JOB_STATE_CANCELLING": JobStatus.CANCELLED,
            }
            return status_map.get(state, JobStatus.UNKNOWN)
        except Exception as e:
            logger.warning("Failed to get job status: %s", e)
            return JobStatus.UNKNOWN

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job.

        Args:
            job_id: Job resource name.

        Returns:
            True if cancellation was initiated.
        """
        job = self._jobs.get(job_id)
        if not job:
            logger.warning("Job not found for cancellation: %s", job_id)
            return False

        try:
            job.cancel()
            logger.info("Cancelled job: %s", job_id)
            return True
        except Exception as e:
            logger.warning("Failed to cancel job: %s", e)
            return False

    def wait_for_completion(
        self,
        job_id: str,
        timeout: float = 3600,
        poll_interval: float = 30,
    ) -> JobStatus:
        """Wait for job to complete.

        Args:
            job_id: Job resource name.
            timeout: Maximum wait time in seconds.
            poll_interval: Time between status checks.

        Returns:
            Final job status.
        """
        start_time = time.time()
        terminal_states = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}

        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            if status in terminal_states:
                return status

            logger.debug("Job %s status: %s, waiting...", job_id, status.value)
            time.sleep(poll_interval)

        logger.warning("Job %s timed out after %s seconds", job_id, timeout)
        return self.get_job_status(job_id)

    def list_jobs(
        self,
        filter_display_name: str | None = None,
        max_results: int = 100,
    ) -> list[str]:
        """List training jobs.

        Args:
            filter_display_name: Optional display name filter.
            max_results: Maximum number of results.

        Returns:
            List of job resource names.
        """
        if self._use_mock:
            jobs = list(self._jobs.keys())[:max_results]
            if filter_display_name:
                jobs = [j for j in jobs if filter_display_name in self._jobs[j].display_name]
            return jobs

        try:
            filter_str = ""
            if filter_display_name:
                filter_str = f'display_name="{filter_display_name}"'

            jobs = aiplatform.CustomJob.list(filter=filter_str)
            return [getattr(job, "resource_name", str(job)) for job in jobs[:max_results]]
        except Exception as e:
            logger.warning("Failed to list jobs: %s", e)
            return []
