"""Vertex AI trainer for distributed training.

This module provides a Vertex AI trainer that implements the
CloudTrainerProtocol for submitting and managing training jobs
on Google Cloud Vertex AI.

Features:
- Custom training job submission
- Hyperparameter tuning support
- Multi-GPU and TPU training
- Job status monitoring
- Graceful fallback when GCP SDK is not installed

Example:
    from alpha_zero.cloud.gcp import VertexTrainer, GCPConfig

    config = GCPConfig.from_env()
    trainer = VertexTrainer(config)

    # Submit training job
    job_id = trainer.submit_training_job(
        script_path="train.py",
        args=["--epochs", "100"],
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
    )

    # Monitor job
    status = trainer.get_job_status(job_id)
    print(f"Job {job_id} status: {status.status}")
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.protocols import (
    CloudTrainerProtocol,
    JobStatus,
    TrainingJobInfo,
)

logger = logging.getLogger(__name__)

# Try to import Vertex AI SDK
try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import CustomJob

    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    aiplatform = None  # type: ignore
    CustomJob = None  # type: ignore


class VertexTrainer:
    """Vertex AI trainer implementing CloudTrainerProtocol.

    This trainer provides methods for submitting and managing
    training jobs on Google Cloud Vertex AI.
    """

    def __init__(
        self,
        config: GCPConfig,
        *,
        allow_fallback: bool = False,
        force_mock: bool = False,
    ) -> None:
        """Initialize Vertex AI trainer.

        Args:
            config: GCP configuration.
            allow_fallback: If True, allow mock mode when SDK is unavailable.
            force_mock: If True, use mock mode even if SDK is available (for testing).
        """
        self.config = config
        self.allow_fallback = allow_fallback
        self._initialized = False
        self._mock_jobs: dict[str, TrainingJobInfo] = {}
        # Use mock if forced or if SDK unavailable and fallback allowed
        self._use_mock = force_mock or (not VERTEX_AVAILABLE and allow_fallback)

        if not VERTEX_AVAILABLE and not allow_fallback and not force_mock:
            raise ImportError(
                "google-cloud-aiplatform is not installed. "
                "Install with: pip install google-cloud-aiplatform"
            )

    def _ensure_initialized(self) -> None:
        """Initialize Vertex AI SDK if not already done."""
        if self._initialized:
            return

        if self._use_mock:
            logger.warning("Using mock mode for Vertex AI")
            self._initialized = True
            return

        aiplatform.init(
            project=self.config.project_id,
            location=self.config.region,
            staging_bucket=f"gs://{self.config.bucket}",
        )
        self._initialized = True
        logger.info(
            f"Vertex AI initialized: project={self.config.project_id}, "
            f"location={self.config.region}"
        )

    def _generate_job_name(self, prefix: str = "alphazero") -> str:
        """Generate a unique job name."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        short_id = uuid.uuid4().hex[:8]
        return f"{prefix}-{timestamp}-{short_id}"

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
        container_uri: str | None = None,
        python_module: str | None = None,
        replica_count: int | None = None,
        boot_disk_size_gb: int | None = None,
        timeout_seconds: int | None = None,
        sync: bool = False,
        **kwargs: Any,
    ) -> str:
        """Submit a training job to Vertex AI.

        Args:
            script_path: Path to the training script.
            args: Command-line arguments for the script.
            machine_type: VM machine type (default from config).
            accelerator_type: GPU/TPU type (default from config).
            accelerator_count: Number of accelerators (default from config).
            base_output_dir: Output directory for artifacts.
            job_name: Custom job name (auto-generated if not provided).
            environment_variables: Environment variables for the job.
            container_uri: Training container image (default from config).
            python_module: Python module to run (alternative to script_path).
            replica_count: Number of worker replicas.
            boot_disk_size_gb: Boot disk size in GB.
            timeout_seconds: Job timeout in seconds.
            sync: If True, wait for job completion.
            **kwargs: Additional provider-specific options.

        Returns:
            Job ID or name for tracking.
        """
        self._ensure_initialized()

        # Apply defaults from config — use `is None` to preserve valid falsy values (e.g. 0)
        if machine_type is None:
            machine_type = self.config.training.machine_type
        if accelerator_type is None:
            accelerator_type = self.config.training.accelerator_type
        if accelerator_count is None:
            accelerator_count = self.config.training.accelerator_count
        if container_uri is None:
            container_uri = self.config.container_image_uri
        if replica_count is None:
            replica_count = 1
        if boot_disk_size_gb is None:
            boot_disk_size_gb = 100
        if timeout_seconds is None:
            timeout_seconds = 3600
        if job_name is None:
            job_name = self._generate_job_name()

        if base_output_dir is None:
            base_output_dir = f"gs://{self.config.bucket}/output/{job_name}"

        # Build args list
        script_args = list(args) if args else []

        if self._use_mock:
            # Mock mode
            job_info = TrainingJobInfo(
                job_id=job_name,
                job_name=job_name,
                status=JobStatus.PENDING,
                created_at=datetime.now(),
                output_uri=base_output_dir,
            )
            self._mock_jobs[job_name] = job_info
            logger.info(f"Mock job created: {job_name}")
            return job_name

        # Build worker pool spec
        worker_pool_specs = [
            {
                "machine_spec": {
                    "machine_type": machine_type,
                },
                "replica_count": replica_count,
                "disk_spec": {
                    "boot_disk_type": "pd-ssd",
                    "boot_disk_size_gb": boot_disk_size_gb,
                },
                "python_package_spec": {
                    "executor_image_uri": container_uri,
                    "python_module": python_module or "",
                    "args": script_args,
                },
            }
        ]

        # Add accelerator if specified
        if accelerator_type and accelerator_count > 0:
            machine_spec = worker_pool_specs[0]["machine_spec"]
            if isinstance(machine_spec, dict):
                machine_spec["accelerator_type"] = accelerator_type
                machine_spec["accelerator_count"] = accelerator_count

        # Use container run command if script_path is specified
        if script_path and not python_module:
            worker_pool_specs[0]["container_spec"] = {
                "image_uri": container_uri,
                "command": ["python", str(script_path)],
                "args": script_args,
                "env": [{"name": k, "value": v} for k, v in (environment_variables or {}).items()],
            }
            del worker_pool_specs[0]["python_package_spec"]

        # Create and run job
        job = CustomJob(
            display_name=job_name,
            worker_pool_specs=worker_pool_specs,
            base_output_dir=base_output_dir,
            labels=getattr(self.config, "custom_labels", {}),
        )

        logger.info(
            f"Submitting Vertex AI job: {job_name}, "
            f"machine={machine_type}, accelerator={accelerator_type}x{accelerator_count}"
        )

        job.run(
            sync=sync,
            timeout=timeout_seconds,
        )

        return job.resource_name

    def get_job_status(self, job_id: str) -> TrainingJobInfo:
        """Get the current status of a training job.

        Args:
            job_id: The job ID returned from submit_training_job.

        Returns:
            TrainingJobInfo with current status.
        """
        self._ensure_initialized()

        if self._use_mock:
            # Mock mode
            if job_id in self._mock_jobs:
                return self._mock_jobs[job_id]
            return TrainingJobInfo(
                job_id=job_id,
                job_name=job_id,
                status=JobStatus.UNKNOWN,
            )

        try:
            job = CustomJob.get(job_id)
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return TrainingJobInfo(
                job_id=job_id,
                job_name=job_id,
                status=JobStatus.UNKNOWN,
                error_message=str(e),
            )

        # Map Vertex AI state to our JobStatus
        state = job.state.name if job.state else "UNKNOWN"
        status_map = {
            "JOB_STATE_PENDING": JobStatus.PENDING,
            "JOB_STATE_QUEUED": JobStatus.PENDING,
            "JOB_STATE_RUNNING": JobStatus.RUNNING,
            "JOB_STATE_SUCCEEDED": JobStatus.SUCCEEDED,
            "JOB_STATE_FAILED": JobStatus.FAILED,
            "JOB_STATE_CANCELLED": JobStatus.CANCELLED,
            "JOB_STATE_CANCELLING": JobStatus.RUNNING,
        }
        status = status_map.get(state, JobStatus.UNKNOWN)

        return TrainingJobInfo(
            job_id=job.resource_name,
            job_name=job.display_name,
            status=status,
            created_at=job.create_time,
            started_at=job.start_time,
            ended_at=job.end_time,
            error_message=job.error.message if job.error else None,
            logs_uri=None,  # Would need to construct from job metadata
            output_uri=getattr(
                getattr(getattr(job, "job_spec", None), "base_output_directory", None),
                "output_uri_prefix",
                None,
            ),
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job.

        Args:
            job_id: The job ID to cancel.

        Returns:
            True if cancellation was successful.
        """
        self._ensure_initialized()

        if self._use_mock:
            # Mock mode
            if job_id in self._mock_jobs:
                self._mock_jobs[job_id].status = JobStatus.CANCELLED
                return True
            return False

        try:
            job = CustomJob.get(job_id)
            job.cancel()
            logger.info(f"Cancelled job: {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False

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
        self._ensure_initialized()

        if self._use_mock:
            # Mock mode
            jobs = list(self._mock_jobs.values())
            if status_filter:
                jobs = [j for j in jobs if j.status == status_filter]
            return jobs[:max_results]

        # Build filter string
        filter_str = ""
        if status_filter:
            state_map = {
                JobStatus.PENDING: "JOB_STATE_PENDING",
                JobStatus.RUNNING: "JOB_STATE_RUNNING",
                JobStatus.SUCCEEDED: "JOB_STATE_SUCCEEDED",
                JobStatus.FAILED: "JOB_STATE_FAILED",
                JobStatus.CANCELLED: "JOB_STATE_CANCELLED",
            }
            if status_filter in state_map:
                filter_str = f'state="{state_map[status_filter]}"'

        jobs = CustomJob.list(
            filter=filter_str,
        )

        results: list[TrainingJobInfo] = []
        for job in list(jobs)[:max_results]:
            job_resource_name = getattr(job, "resource_name", None)
            if job_resource_name:
                results.append(self.get_job_status(job_resource_name))

        return results

    def wait_for_job(
        self,
        job_id: str,
        *,
        poll_interval_seconds: float = 60.0,
        timeout_seconds: float | None = None,
    ) -> TrainingJobInfo:
        """Wait for a job to complete.

        Args:
            job_id: The job ID to wait for.
            poll_interval_seconds: Time between status checks.
            timeout_seconds: Maximum time to wait.

        Returns:
            Final TrainingJobInfo.
        """
        import time

        start_time = time.time()

        while True:
            status = self.get_job_status(job_id)

            if status.is_terminal():
                return status

            if timeout_seconds:
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    raise TimeoutError(f"Job {job_id} did not complete within {timeout_seconds}s")

            logger.info(f"Job {job_id} status: {status.status.value}, waiting...")
            time.sleep(poll_interval_seconds)


# Protocol compliance check
def _check_protocol_compliance() -> None:
    """Verify VertexTrainer implements CloudTrainerProtocol."""
    trainer: CloudTrainerProtocol = VertexTrainer(
        GCPConfig(), allow_fallback=True
    )
    assert isinstance(trainer, CloudTrainerProtocol)
