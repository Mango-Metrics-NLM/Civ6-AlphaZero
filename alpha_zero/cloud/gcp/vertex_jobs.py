"""
Vertex AI Job Wrappers for Self-Play and Training.

Provides job management abstractions for:
- Self-play workers on Spot VMs
- Training jobs with checkpoint recovery
- Preemption handling and job monitoring
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.exceptions import PreemptionError, QuotaExceededError, VertexJobError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    """Vertex AI job states."""

    PENDING = "JOB_STATE_PENDING"
    QUEUED = "JOB_STATE_QUEUED"
    RUNNING = "JOB_STATE_RUNNING"
    SUCCEEDED = "JOB_STATE_SUCCEEDED"
    FAILED = "JOB_STATE_FAILED"
    CANCELLING = "JOB_STATE_CANCELLING"
    CANCELLED = "JOB_STATE_CANCELLED"
    PAUSED = "JOB_STATE_PAUSED"
    EXPIRED = "JOB_STATE_EXPIRED"

    @classmethod
    def is_terminal(cls, state: str) -> bool:
        """Check if state is terminal (job finished)."""
        return state in {
            cls.SUCCEEDED.value,
            cls.FAILED.value,
            cls.CANCELLED.value,
            cls.EXPIRED.value,
        }

    @classmethod
    def is_running(cls, state: str) -> bool:
        """Check if job is currently running."""
        return state in {cls.PENDING.value, cls.QUEUED.value, cls.RUNNING.value}


@dataclass
class JobSpec:
    """Specification for a Vertex AI Custom Job.

    This is a platform-agnostic representation that can be converted
    to Vertex AI-specific formats.
    """

    display_name: str
    container_uri: str
    command: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    replica_count: int = 1
    spot: bool = False
    service_account: str | None = None
    timeout_seconds: int = 86400  # 24 hours

    def to_worker_pool_spec(self) -> dict[str, Any]:
        """Convert to Vertex AI worker pool spec format."""
        container_spec = {
            "image_uri": self.container_uri,
        }
        if self.command:
            container_spec["command"] = self.command
        if self.args:
            container_spec["args"] = self.args
        if self.env_vars:
            container_spec["env"] = [{"name": k, "value": v} for k, v in self.env_vars.items()]

        machine_spec: dict[str, Any] = {"machine_type": self.machine_type}
        if self.accelerator_type and self.accelerator_count > 0:
            machine_spec["accelerator_type"] = self.accelerator_type
            machine_spec["accelerator_count"] = self.accelerator_count

        return {
            "replica_count": self.replica_count,
            "machine_spec": machine_spec,
            "container_spec": container_spec,
        }


@dataclass
class JobResult:
    """Result of a completed job."""

    job_id: str
    job_name: str
    state: str
    success: bool
    start_time: str | None = None
    end_time: str | None = None
    error_message: str | None = None
    output_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class VertexAIClient:
    """Client wrapper for Vertex AI operations.

    Provides a testable abstraction over the Vertex AI SDK.
    """

    def __init__(self, project: str, region: str, service_account: str | None = None) -> None:
        """Initialize Vertex AI client.

        Args:
            project: GCP project ID.
            region: GCP region.
            service_account: Service account email (optional).
        """
        self._project = project
        self._region = region
        self._service_account = service_account
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazily initialize the Vertex AI SDK."""
        if not self._initialized:
            try:
                from google.cloud import aiplatform

                aiplatform.init(
                    project=self._project,
                    location=self._region,
                )
                self._initialized = True
            except ImportError as e:
                raise VertexJobError(
                    "google-cloud-aiplatform not installed. Install with: pip install google-cloud-aiplatform",
                ) from e
            except Exception as e:
                raise VertexJobError(
                    f"Failed to initialize Vertex AI: {e}",
                ) from e

    def submit_custom_job(
        self,
        spec: JobSpec,
        staging_bucket: str | None = None,
    ) -> str:
        """Submit a custom job to Vertex AI.

        Args:
            spec: Job specification.
            staging_bucket: GCS bucket for staging (optional).

        Returns:
            Job resource name.

        Raises:
            VertexJobError: If submission fails.
            QuotaExceededError: If quota is exceeded.
        """
        self._ensure_initialized()

        try:
            from google.cloud import aiplatform
            from google.protobuf import duration_pb2

            worker_pool_specs = [spec.to_worker_pool_spec()]

            job = aiplatform.CustomJob(
                display_name=spec.display_name,
                worker_pool_specs=worker_pool_specs,
                staging_bucket=staging_bucket,
            )

            # Build submit kwargs with spot/scheduling and timeout
            submit_kwargs: dict[str, Any] = {
                "service_account": self._service_account or spec.service_account,
                "enable_web_access": False,
            }

            # Configure spot VM scheduling if requested
            if spec.spot:
                # Vertex AI uses restart_job_on_worker_restart for spot/preemptible
                submit_kwargs["restart_job_on_worker_restart"] = True

            # Configure timeout if specified
            if spec.timeout_seconds > 0:
                submit_kwargs["timeout"] = duration_pb2.Duration(seconds=spec.timeout_seconds)

            job.submit(**submit_kwargs)

            logger.info("Submitted job: %s (resource: %s)", spec.display_name, job.resource_name)
            return job.resource_name

        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg:
                raise QuotaExceededError(
                    f"Quota exceeded when submitting job: {e}",
                ) from e
            raise VertexJobError(
                f"Failed to submit job {spec.display_name}: {e}",
            ) from e

    def get_job_state(self, job_name: str) -> tuple[str, dict[str, Any]]:
        """Get current state of a job.

        Args:
            job_name: Job resource name.

        Returns:
            Tuple of (state, metadata).
        """
        self._ensure_initialized()

        try:
            from google.cloud import aiplatform

            job = aiplatform.CustomJob.get(job_name)
            state = job.state.name if job.state else "UNKNOWN"

            metadata = {
                "display_name": job.display_name,
                "create_time": str(job.create_time) if job.create_time else None,
                "start_time": str(job.start_time) if job.start_time else None,
                "end_time": str(job.end_time) if job.end_time else None,
            }

            return state, metadata

        except Exception as e:
            raise VertexJobError(
                f"Failed to get job state: {e}",
                job_id=job_name,
            ) from e

    def cancel_job(self, job_name: str) -> None:
        """Cancel a running job.

        Args:
            job_name: Job resource name.
        """
        self._ensure_initialized()

        try:
            from google.cloud import aiplatform

            job = aiplatform.CustomJob.get(job_name)
            job.cancel()
            logger.info("Cancelled job: %s", job_name)

        except Exception as e:
            raise VertexJobError(
                f"Failed to cancel job: {e}",
                job_id=job_name,
            ) from e

    def wait_for_completion(
        self,
        job_name: str,
        timeout_seconds: int = 86400,
        poll_interval_seconds: int = 30,
    ) -> JobResult:
        """Wait for a job to complete.

        Args:
            job_name: Job resource name.
            timeout_seconds: Maximum wait time.
            poll_interval_seconds: Polling interval.

        Returns:
            JobResult with final state.

        Raises:
            VertexJobError: If job fails or times out.
            PreemptionError: If job was preempted.
        """
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise VertexJobError(
                    f"Job timed out after {timeout_seconds} seconds",
                    job_id=job_name,
                )

            state, metadata = self.get_job_state(job_name)

            if JobState.is_terminal(state):
                success = state == JobState.SUCCEEDED.value
                error_message = None

                if not success:
                    # Check for preemption in job metadata/error message
                    # The state string (e.g. JOB_STATE_FAILED) doesn't contain preemption info
                    error_info = metadata.get("error", {})
                    error_msg = ""
                    if isinstance(error_info, dict):
                        error_msg = error_info.get("message", "")
                    elif isinstance(error_info, str):
                        error_msg = error_info

                    # Also check status_message for preemption indicators
                    status_msg = metadata.get("status_message", "")
                    combined_msg = f"{error_msg} {status_msg}".lower()

                    if "preempt" in combined_msg or "spot" in combined_msg:
                        raise PreemptionError(
                            f"Job was preempted: {error_msg or status_msg}",
                            job_id=job_name,
                        )
                    error_message = f"Job ended with state: {state}. {error_msg}"

                return JobResult(
                    job_id=job_name,
                    job_name=metadata.get("display_name", ""),
                    state=state,
                    success=success,
                    start_time=metadata.get("start_time"),
                    end_time=metadata.get("end_time"),
                    error_message=error_message,
                    metadata=metadata,
                )

            logger.debug("Job %s in state %s, waiting...", job_name, state)
            time.sleep(poll_interval_seconds)


class VertexSelfPlayJob:
    """Manages self-play worker jobs on Vertex AI.

    Self-play workers are stateless and designed for Spot VMs:
    - Each worker fetches the latest model from GCS
    - Generates game trajectories using MCTS
    - Writes trajectories to GCS immediately
    - Can be preempted without losing significant work
    """

    def __init__(
        self,
        config: GCPConfig,
        client: VertexAIClient | None = None,
    ) -> None:
        """Initialize self-play job manager.

        Args:
            config: GCP configuration.
            client: Vertex AI client (creates one if None).
        """
        self.config = config
        self.client = client or VertexAIClient(
            project=config.project_id,
            region=config.region,
            service_account=config.service_account,
        )
        self._active_jobs: list[str] = []

    def launch(
        self,
        num_workers: int,
        model_version: int,
        iteration: int,
        container_uri: str | None = None,
    ) -> list[str]:
        """Launch self-play worker jobs.

        Args:
            num_workers: Number of parallel workers.
            model_version: Model version for self-play.
            iteration: Current training iteration.
            container_uri: Container image URI (uses config default if None).

        Returns:
            List of job resource names.

        Raises:
            VertexJobError: If launch fails.
        """
        container = container_uri or self.config.container_image_uri
        if not container:
            raise VertexJobError("No container image URI configured")

        job_names: list[str] = []
        worker_config = self.config.self_play

        for worker_id in range(num_workers):
            spec = JobSpec(
                display_name=f"civ6az-selfplay-iter{iteration:03d}-worker{worker_id:02d}",
                container_uri=container,
                args=[
                    "--mode=selfplay",
                    f"--worker_id={worker_id}",
                    f"--model_version={model_version}",
                    f"--iteration={iteration}",
                    f"--games={worker_config.games_per_worker}",
                ],
                env_vars={
                    "CIV6AZ_GCS_BUCKET": self.config.bucket,
                    "CIV6AZ_VERTEX_PROJECT": self.config.project_id,
                    "CIV6AZ_VERTEX_REGION": self.config.region,
                },
                machine_type=worker_config.machine_type,
                accelerator_type=worker_config.accelerator_type,
                accelerator_count=worker_config.accelerator_count,
                replica_count=1,
                spot=worker_config.spot,
                service_account=self.config.service_account,
            )

            job_name = self.client.submit_custom_job(
                spec,
                staging_bucket=f"gs://{self.config.bucket}",
            )
            job_names.append(job_name)
            self._active_jobs.append(job_name)

        logger.info("Launched %d self-play workers for iteration %d", num_workers, iteration)
        return job_names

    def wait_for_completion(
        self,
        job_names: list[str],
        timeout_seconds: int = 14400,  # 4 hours
    ) -> list[JobResult]:
        """Wait for all self-play jobs to complete.

        Args:
            job_names: List of job resource names.
            timeout_seconds: Maximum wait time.

        Returns:
            List of JobResults.
        """
        results: list[JobResult] = []
        start_time = time.time()

        for job_name in job_names:
            remaining_time = timeout_seconds - (time.time() - start_time)
            if remaining_time <= 0:
                raise VertexJobError("Overall timeout exceeded waiting for self-play jobs")

            try:
                result = self.client.wait_for_completion(
                    job_name,
                    timeout_seconds=int(remaining_time),
                )
                results.append(result)

                # Remove from active jobs
                if job_name in self._active_jobs:
                    self._active_jobs.remove(job_name)

            except PreemptionError as e:
                # Log preemption but continue - other workers may have succeeded
                logger.warning("Worker %s was preempted: %s", job_name, e)
                results.append(JobResult(
                    job_id=job_name,
                    job_name="",
                    state=JobState.FAILED.value,
                    success=False,
                    error_message="Preempted",
                ))

        successful = sum(1 for r in results if r.success)
        logger.info("Self-play complete: %d/%d workers succeeded", successful, len(results))
        return results

    def cancel_all(self, job_names: list[str] | None = None) -> None:
        """Cancel all active jobs.

        Args:
            job_names: Specific jobs to cancel (all active if None).
        """
        jobs_to_cancel = job_names or self._active_jobs.copy()

        for job_name in jobs_to_cancel:
            try:
                self.client.cancel_job(job_name)
            except VertexJobError as e:
                logger.warning("Failed to cancel job %s: %s", job_name, e)

        self._active_jobs.clear()


class VertexTrainingJob:
    """Manages training jobs on Vertex AI.

    Training jobs use reliable VMs (not Spot) and include:
    - Checkpoint recovery from GCS
    - Periodic checkpointing during training
    - Graceful preemption handling
    """

    def __init__(
        self,
        config: GCPConfig,
        client: VertexAIClient | None = None,
    ) -> None:
        """Initialize training job manager.

        Args:
            config: GCP configuration.
            client: Vertex AI client (creates one if None).
        """
        self.config = config
        self.client = client or VertexAIClient(
            project=config.project_id,
            region=config.region,
            service_account=config.service_account,
        )
        self._current_job: str | None = None

    def launch(
        self,
        iteration: int,
        trajectory_iterations: list[int],
        resume_from_checkpoint: str | None = None,
        container_uri: str | None = None,
    ) -> str:
        """Launch a training job.

        Args:
            iteration: Current training iteration.
            trajectory_iterations: Iterations to sample trajectories from.
            resume_from_checkpoint: GCS path to checkpoint (optional).
            container_uri: Container image URI (uses config default if None).

        Returns:
            Job resource name.

        Raises:
            VertexJobError: If launch fails.
        """
        container = container_uri or self.config.container_image_uri
        if not container:
            raise VertexJobError("No container image URI configured")

        training_config = self.config.training

        args = [
            "--mode=training",
            f"--iteration={iteration}",
            f"--trajectory_iterations={','.join(map(str, trajectory_iterations))}",
            f"--epochs={training_config.epochs_per_iteration}",
            f"--checkpoint_interval={training_config.checkpoint_interval}",
        ]

        if resume_from_checkpoint:
            args.append(f"--resume_from={resume_from_checkpoint}")

        spec = JobSpec(
            display_name=f"civ6az-training-iter{iteration:03d}",
            container_uri=container,
            args=args,
            env_vars={
                "CIV6AZ_GCS_BUCKET": self.config.bucket,
                "CIV6AZ_VERTEX_PROJECT": self.config.project_id,
                "CIV6AZ_VERTEX_REGION": self.config.region,
            },
            machine_type=training_config.machine_type,
            accelerator_type=training_config.accelerator_type,
            accelerator_count=training_config.accelerator_count,
            replica_count=1,
            spot=training_config.spot,  # Usually False for training
            service_account=self.config.service_account,
            timeout_seconds=86400,  # 24 hours
        )

        job_name = self.client.submit_custom_job(
            spec,
            staging_bucket=f"gs://{self.config.bucket}",
        )
        self._current_job = job_name

        logger.info("Launched training job for iteration %d", iteration)
        return job_name

    def wait_for_completion(
        self,
        job_name: str | None = None,
        timeout_seconds: int = 86400,  # 24 hours
    ) -> JobResult:
        """Wait for training job to complete.

        Args:
            job_name: Job to wait for (uses current if None).
            timeout_seconds: Maximum wait time.

        Returns:
            JobResult with final state.

        Raises:
            VertexJobError: If job fails or times out.
        """
        job = job_name or self._current_job
        if not job:
            raise VertexJobError("No training job to wait for")

        result = self.client.wait_for_completion(job, timeout_seconds)

        if result.success:
            logger.info("Training job completed successfully")
        else:
            logger.error("Training job failed: %s", result.error_message)

        self._current_job = None
        return result

    def get_output_model_version(self, job_name: str | None = None) -> int | None:
        """Get the model version produced by a training job.

        Args:
            job_name: Job to query (uses current if None).

        Returns:
            Model version or None if not available.
        """
        # This would typically parse job metadata or query GCS
        # For now, return None - the caller should check GCS directly
        return None

    def cancel(self) -> None:
        """Cancel the current training job."""
        if self._current_job:
            self.client.cancel_job(self._current_job)
            self._current_job = None
