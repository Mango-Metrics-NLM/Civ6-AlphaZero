"""
Vertex AI Job Wrappers for Self-Play and Training.

Provides job management abstractions for:
- Self-play workers on Spot VMs
- Training jobs with checkpoint recovery
- Preemption handling and job monitoring
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.exceptions import PreemptionError, QuotaExceededError, VertexJobError


def _wandb_env_vars() -> dict[str, str]:
    """Get W&B env vars for container passthrough, filtering out empty values.

    Vertex AI rejects env vars with empty string values, so we only
    include vars that have non-empty values set.

    The WANDB_API_KEY is resolved with the following precedence:
    1. ``WANDB_API_KEY`` env var (backwards compatible)
    2. GCP Secret Manager secret named by ``CIV6AZ_WANDB_SECRET_NAME``
       (default: ``wandb-api-key``).  Falls back silently if SM is
       unavailable.
    """
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        wandb_key = _resolve_wandb_key_from_secret_manager()

    candidates = {
        "WANDB_ENABLED": os.environ.get("WANDB_ENABLED", "false"),
        "WANDB_API_KEY": wandb_key,
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "civ6-alphazero"),
        "WANDB_ENTITY": os.environ.get("WANDB_ENTITY", ""),
        "WANDB_MODE": os.environ.get("WANDB_MODE", "online"),
        "WANDB_GROUP": os.environ.get("WANDB_GROUP", ""),
        "WANDB_RUN_NAME": os.environ.get("WANDB_RUN_NAME", ""),
    }
    return {k: v for k, v in candidates.items() if v}


def _resolve_wandb_key_from_secret_manager() -> str:
    """Attempt to read WANDB_API_KEY from GCP Secret Manager.

    Returns:
        The secret value, or empty string if unavailable.
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    secret_name = os.environ.get("CIV6AZ_WANDB_SECRET_NAME", "wandb-api-key")
    if not project:
        return ""
    try:
        from google.cloud import secretmanager  # type: ignore[import-untyped]

        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        key = response.payload.data.decode("UTF-8")
        logger.info("WANDB_API_KEY resolved from Secret Manager (%s)", secret_name)
        return key
    except Exception as e:
        logger.debug("Could not read WANDB_API_KEY from Secret Manager: %s", e)
        return ""


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
        container_spec: dict[str, Any] = {
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

            worker_pool_specs = [spec.to_worker_pool_spec()]

            job = aiplatform.CustomJob(
                display_name=spec.display_name,
                worker_pool_specs=worker_pool_specs,
                staging_bucket=staging_bucket,
            )

            # Build submit kwargs with spot/scheduling and timeout
            submit_kwargs: dict[str, Any] = {
                "enable_web_access": False,
            }

            # Only set service_account if one is provided
            service_account = self._service_account or spec.service_account
            if service_account:
                submit_kwargs["service_account"] = service_account

            # Configure spot VM scheduling if requested
            if spec.spot:
                # Vertex AI uses restart_job_on_worker_restart for spot/preemptible
                submit_kwargs["restart_job_on_worker_restart"] = True

            # Configure timeout if specified (SDK expects int seconds)
            if spec.timeout_seconds > 0:
                submit_kwargs["timeout"] = spec.timeout_seconds

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

            # Extract error/status information from the underlying GCA resource
            error_info: Any | None = None
            status_message: str | None = None
            gca_resource = getattr(job, "_gca_resource", None)
            if gca_resource is not None:
                gca_error = getattr(gca_resource, "error", None)
                if gca_error is not None:
                    error_info = {
                        "code": getattr(gca_error, "code", None),
                        "message": getattr(gca_error, "message", None),
                        "details": getattr(gca_error, "details", None),
                    }
                    status_message = getattr(gca_error, "message", None)

            metadata = {
                "display_name": job.display_name,
                "create_time": str(job.create_time) if job.create_time else None,
                "start_time": str(job.start_time) if job.start_time else None,
                "end_time": str(job.end_time) if job.end_time else None,
                "error": error_info,
                "status_message": status_message,
            }

            return state, metadata

        except Exception as e:
            raise VertexJobError(
                f"Failed to get job state: {e}",
                job_id=job_name,
            ) from e

    def submit_tuning_job(
        self,
        spec: JobSpec,
        metric_spec: dict[str, str],
        parameter_spec: dict[str, Any],
        max_trial_count: int,
        parallel_trial_count: int,
        staging_bucket: str | None = None,
    ) -> str:
        """Submit a specific HyperparameterTuningJob.

        Args:
            spec: Job specification for the trial job.
            metric_spec: Dictionary mapping metric IDs to 'maximize' or 'minimize'.
            parameter_spec: Dictionary defining the search space.
            max_trial_count: Total number of trials.
            parallel_trial_count: Number of parallel trials.
            staging_bucket: GCS bucket for staging.

        Returns:
            Job resource name.
        """
        self._ensure_initialized()

        try:
            from google.cloud import aiplatform

            worker_pool_specs = [spec.to_worker_pool_spec()]

            # Create the custom job spec for each trial
            custom_job = aiplatform.CustomJob(
                display_name=spec.display_name,
                worker_pool_specs=worker_pool_specs,
                staging_bucket=staging_bucket,
            )

            # Create the tuning job
            tuning_job = aiplatform.HyperparameterTuningJob(
                display_name=f"{spec.display_name}-tuning",
                custom_job=custom_job,
                metric_spec=metric_spec,
                parameter_spec=parameter_spec,
                max_trial_count=max_trial_count,
                parallel_trial_count=parallel_trial_count,
            )

            tuning_job.submit(service_account=spec.service_account)  # type: ignore

            logger.info(
                "Submitted tuning job: %s (resource: %s)",
                spec.display_name,
                tuning_job.resource_name,
            )
            return tuning_job.resource_name

        except Exception as e:
            raise VertexJobError(
                f"Failed to submit tuning job {spec.display_name}: {e}",
            ) from e

    def wait_for_completion(
        self,
        job_name: str,
        timeout_seconds: int = 86400,
        poll_interval_seconds: int = 10,
    ) -> JobResult:
        """Wait for a job to complete.

        Args:
            job_name: Job resource name.
            timeout_seconds: Maximum wait time.
            poll_interval_seconds: Polling interval.

        Returns:
            JobResult with final state.
        """
        start_time = time.time()
        while True:
            state, metadata = self.get_job_state(job_name)

            if JobState.is_terminal(state):
                success = state == JobState.SUCCEEDED.value
                return JobResult(
                    job_id=job_name,
                    job_name=metadata.get("display_name", job_name),
                    state=state,
                    success=success,
                    start_time=metadata.get("start_time"),
                    end_time=metadata.get("end_time"),
                    error_message=metadata.get("status_message"),
                    metadata=metadata,
                )

            if time.time() - start_time > timeout_seconds:
                raise VertexJobError(f"Job {job_name} timed out after {timeout_seconds} seconds")

            time.sleep(poll_interval_seconds)

    def cancel_job(self, job_name: str) -> None:
        """Cancel a running job.

        Args:
            job_name: Job resource name.
        """
        self._ensure_initialized()
        try:
            from google.cloud import aiplatform

            # Try to get as CustomJob first, then HyperparameterTuningJob
            try:
                job: Any = aiplatform.CustomJob.get(job_name)
            except Exception:
                job = aiplatform.HyperparameterTuningJob.get(job_name)

            job.cancel()
            logger.info("Cancelled job: %s", job_name)
        except Exception as e:
            raise VertexJobError(f"Failed to cancel job {job_name}: {e}") from e


class VertexTuningJob:
    """Manages hyperparameter tuning jobs on Vertex AI."""

    def __init__(
        self,
        config: GCPConfig,
        client: VertexAIClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or VertexAIClient(
            project=config.project_id,
            region=config.region,
            service_account=config.service_account,
        )

    def launch(
        self,
        iteration: int,
        parameter_spec: dict[str, Any],
        metric_spec: dict[str, str],
        max_trials: int = 20,
        parallel_trials: int = 4,
        container_uri: str | None = None,
        extra_args: list[str] | None = None,
    ) -> str:
        """Launch a hyperparameter tuning job.

        Args:
            iteration: Current iteration (for naming).
            parameter_spec: Vizier parameter search space.
            metric_spec: Metric to optimize (e.g. {'elo': 'maximize'}).
            max_trials: Total trials to run.
            parallel_trials: Trials to run in parallel.
            container_uri: Container image URI.
            extra_args: Additional args for the container.

        Returns:
            Job resource name.
        """
        container = container_uri or self.config.container_image_uri
        if not container:
            raise VertexJobError("No container image URI configured")

        # Base args - Vizier will append hyperparameters as CLI flags
        args = [
            "--mode=tuning_worker",
            f"--iteration={iteration}",
        ]
        if extra_args:
            args.extend(extra_args)

        # Create base job spec
        spec = JobSpec(
            display_name=f"civ6az-tuning-iter{iteration:03d}",
            container_uri=container,
            args=args,
            env_vars={
                "CIV6AZ_GCS_BUCKET": self.config.bucket,
                "CIV6AZ_VERTEX_PROJECT": self.config.project_id,
                "CIV6AZ_VERTEX_REGION": self.config.region,
                "CIV6AZ_ENCODER_TYPE": getattr(self.config.network, "encoder_type", "patch_embed"),
                **_wandb_env_vars(),
            },
            machine_type="n1-standard-4",  # Tuning trials usually smaller
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            replica_count=1,
            spot=False,  # Vizier can use spot, but it's simpler to use on-demand for reliability
            service_account=self.config.service_account,
        )

        job_name = self.client.submit_tuning_job(
            spec=spec,
            metric_spec=metric_spec,
            parameter_spec=parameter_spec,
            max_trial_count=max_trials,
            parallel_trial_count=parallel_trials,
            staging_bucket=f"gs://{self.config.bucket}",
        )

        return job_name

    def wait_for_completion(
        self,
        job_name: str,
        timeout_seconds: int = 86400,
    ) -> JobResult:
        """Wait for tuning job to complete.

        Args:
            job_name: Job resource name.
            timeout_seconds: Maximum wait time.

        Returns:
            JobResult with final state.
        """
        # Tuning jobs are long running, so we might want a longer poll interval
        return self.client.wait_for_completion(
            job_name,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=60,
        )

    def cancel(self, job_name: str) -> None:
        """Cancel a tuning job.

        Args:
            job_name: Job resource name.
        """
        self.client.cancel_job(job_name)


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
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Launch self-play worker jobs.

        Args:
            num_workers: Number of parallel workers.
            model_version: Model version for self-play.
            iteration: Current training iteration.
            container_uri: Container image URI (uses config default if None).
            extra_args: Additional CLI args to pass to the container.

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
            args = [
                f"--worker_id={worker_id}",
                f"--model_version={model_version}",
                f"--iteration={iteration}",
                f"--games={worker_config.games_per_worker}",
            ]
            if extra_args:
                args.extend(extra_args)

            spec = JobSpec(
                display_name=f"civ6az-selfplay-iter{iteration:03d}-worker{worker_id:02d}",
                container_uri=container,
                command=["python", "-m", "alpha_zero.cloud.gcp.selfplay_worker"],
                args=args,
                env_vars={
                    "CIV6AZ_GCS_BUCKET": self.config.bucket,
                    "CIV6AZ_VERTEX_PROJECT": self.config.project_id,
                    "CIV6AZ_VERTEX_REGION": self.config.region,
                    **_wandb_env_vars(),
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
                # Remove preempted job from active jobs
                if job_name in self._active_jobs:
                    self._active_jobs.remove(job_name)
                results.append(
                    JobResult(
                        job_id=job_name,
                        job_name=job_name,
                        state=JobState.FAILED.value,
                        success=False,
                        error_message="Preempted",
                    )
                )

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
        extra_args: list[str] | None = None,
    ) -> str:
        """Launch a training job.

        Args:
            iteration: Current training iteration.
            trajectory_iterations: Iterations to sample trajectories from.
            resume_from_checkpoint: GCS path to checkpoint (optional).
            container_uri: Container image URI (uses config default if None).
            extra_args: Additional CLI args to pass to the container.

        Returns:
            Job resource name.

        Raises:
            VertexJobError: If launch fails.
        """
        container = container_uri or self.config.container_image_uri
        if not container:
            raise VertexJobError("No container image URI configured")

        training_config = self.config.training

        # Multi-node DDP is not yet supported — only single-node torchrun.
        # TODO: Add multi-node rendezvous args (--nnodes, --rdzv_*) when needed.
        if training_config.replica_count > 1:
            raise VertexJobError(
                f"Multi-node training (replica_count={training_config.replica_count}) "
                "is not yet supported. Use replica_count=1 with multiple GPUs "
                "for single-node DDP via torchrun."
            )

        if training_config.accelerator_count > 1:
            # Use torchrun for DDP
            command = [
                "torchrun",
                f"--nproc_per_node={training_config.accelerator_count}",
                "-m",
                "alpha_zero.cloud.gcp.training_worker",
            ]
        else:
            command = ["python", "-m", "alpha_zero.cloud.gcp.training_worker"]

        # Base args for the worker
        worker_args = [
            f"--iteration={iteration}",
            f"--trajectory_iterations={','.join(map(str, trajectory_iterations))}",
            f"--epochs={training_config.epochs_per_iteration}",
            f"--checkpoint_interval={training_config.checkpoint_interval}",
        ]

        if resume_from_checkpoint:
            worker_args.append(f"--resume_from={resume_from_checkpoint}")

        if extra_args:
            worker_args.extend(extra_args)

        spec = JobSpec(
            display_name=f"civ6az-training-iter{iteration:03d}",
            container_uri=container,
            command=command,
            args=worker_args,
            env_vars={
                "CIV6AZ_GCS_BUCKET": self.config.bucket,
                "CIV6AZ_VERTEX_PROJECT": self.config.project_id,
                "CIV6AZ_VERTEX_REGION": self.config.region,
                "CIV6AZ_ENCODER_TYPE": getattr(self.config.network, "encoder_type", "patch_embed"),
                **_wandb_env_vars(),
            },
            machine_type=training_config.machine_type,
            accelerator_type=training_config.accelerator_type,
            accelerator_count=training_config.accelerator_count,
            replica_count=training_config.replica_count,
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

    def cancel(self) -> None:
        """Cancel the current training job."""
        if self._current_job:
            self.client.cancel_job(self._current_job)
            self._current_job = None


class VertexEvaluationJob:
    """Manages model evaluation jobs on Vertex AI.

    Runs arena matches between two model versions to measure improvement.
    Uses CPU-only instances since evaluation is not latency-sensitive.
    """

    def __init__(self, config: GCPConfig, client: VertexAIClient) -> None:
        self.config = config
        self.client = client
        self._current_job: str | None = None

    def launch(
        self,
        model_version_new: int,
        model_version_old: int,
        iteration: int,
        num_games: int = 20,
        extra_args: list[str] | None = None,
    ) -> str:
        """Launch an evaluation job.

        Args:
            model_version_new: New model version to evaluate.
            model_version_old: Baseline model version.
            iteration: Training iteration.
            num_games: Number of games to play.
            extra_args: Additional container arguments.

        Returns:
            Job resource name.
        """
        container = self.config.container_image_uri
        if not container:
            raise VertexJobError("container_image_uri not configured")

        args = [
            "--mode=evaluation",
            f"--model_version_new={model_version_new}",
            f"--model_version_old={model_version_old}",
            f"--iteration={iteration}",
            f"--games={num_games}",
        ]
        if extra_args:
            args.extend(extra_args)

        spec = JobSpec(
            display_name=f"civ6az-eval-iter{iteration:03d}",
            container_uri=container,
            command=["python", "-m", "alpha_zero.cloud.gcp.entrypoint"],
            args=args,
            env_vars={
                "CIV6AZ_GCS_BUCKET": self.config.bucket,
                "CIV6AZ_VERTEX_PROJECT": self.config.project_id,
                "CIV6AZ_VERTEX_REGION": self.config.region,
                "CIV6AZ_ENCODER_TYPE": getattr(self.config.network, "encoder_type", "patch_embed"),
                **_wandb_env_vars(),
            },
            # CPU-only: evaluation doesn't need GPU
            machine_type="n1-standard-4",
            accelerator_type="",
            accelerator_count=0,
            replica_count=1,
            spot=False,
            service_account=self.config.service_account,
            timeout_seconds=3600,  # 1 hour
        )

        job_name = self.client.submit_custom_job(
            spec,
            staging_bucket=f"gs://{self.config.bucket}",
        )
        self._current_job = job_name

        logger.info(
            "Launched evaluation job: v%d vs v%d (%d games)",
            model_version_new,
            model_version_old,
            num_games,
        )
        return job_name

    def wait_for_completion(
        self,
        job_name: str | None = None,
        timeout_seconds: int = 3600,
    ) -> JobResult:
        """Wait for evaluation job to complete."""
        job = job_name or self._current_job
        if not job:
            raise VertexJobError("No evaluation job to wait for")

        result = self.client.wait_for_completion(job, timeout_seconds)
        self._current_job = None
        return result

    def cancel(self) -> None:
        """Cancel the current evaluation job."""
        if self._current_job:
            self.client.cancel_job(self._current_job)
            self._current_job = None


class VertexTournamentJob:
    """Manages tournament jobs on Vertex AI.

    Runs a round-robin tournament across model versions using the standalone
    tournament_worker module. Unlike VertexEvaluationJob, supports configurable
    machine types and GPU accelerators.
    """

    def __init__(self, config: GCPConfig, client: VertexAIClient) -> None:
        self.config = config
        self.client = client
        self._current_job: str | None = None

    def launch(
        self,
        model_versions: list[int],
        tournament_games: int = 10,
        mcts_type: str = "gumbel",
        mcts_sims: int = 25,
        gumbel_num_actions: int = 16,
        board_size: int = 8,
        action_space_size: int = 376,
        max_turns: int = 30,
        machine_type: str = "n1-standard-8",
        accelerator_type: str = "NVIDIA_TESLA_T4",
        accelerator_count: int = 1,
        extra_args: list[str] | None = None,
    ) -> str:
        """Launch a tournament job.

        Args:
            model_versions: Model versions to include in the tournament.
            tournament_games: Games per matchup.
            mcts_type: MCTS algorithm variant.
            mcts_sims: MCTS simulations per move.
            gumbel_num_actions: Gumbel sampled actions.
            board_size: Board size.
            action_space_size: Action space size.
            max_turns: Max turns per game.
            machine_type: GCE machine type.
            accelerator_type: GPU type (empty string for CPU-only).
            accelerator_count: Number of GPUs (0 for CPU-only).
            extra_args: Additional container arguments.

        Returns:
            Job resource name.
        """
        container = self.config.container_image_uri
        if not container:
            raise VertexJobError("container_image_uri not configured")

        versions_str = ",".join(str(v) for v in model_versions)

        args = [
            "--mode=tournament",
            f"--model_versions={versions_str}",
            f"--tournament_games={tournament_games}",
            f"--mcts_type={mcts_type}",
            f"--tournament_mcts_sims={mcts_sims}",
            f"--gumbel_num_actions={gumbel_num_actions}",
            f"--board_size={board_size}",
            f"--action_space_size={action_space_size}",
            f"--max_turns={max_turns}",
        ]
        if extra_args:
            args.extend(extra_args)

        env_vars = {
            "CIV6AZ_GCS_BUCKET": self.config.bucket,
            "CIV6AZ_VERTEX_PROJECT": self.config.project_id,
            "CIV6AZ_VERTEX_REGION": self.config.region,
            "CIV6AZ_ENCODER_TYPE": getattr(self.config.network, "encoder_type", "patch_embed"),
        }
        env_vars.update(_wandb_env_vars())
        env_vars = {k: v for k, v in env_vars.items() if v}

        spec = JobSpec(
            display_name=f"civ6az-tournament-{len(model_versions)}models",
            container_uri=container,
            command=["python", "-m", "alpha_zero.cloud.gcp.entrypoint"],
            args=args,
            env_vars=env_vars,
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            replica_count=1,
            spot=False,
            service_account=self.config.service_account,
            timeout_seconds=86400,  # 24 hours
        )

        job_name = self.client.submit_custom_job(
            spec,
            staging_bucket=f"gs://{self.config.bucket}",
        )
        self._current_job = job_name

        num_matchups = len(model_versions) * (len(model_versions) - 1) // 2
        logger.info(
            "Launched tournament job: %d models, %d matchups, %d total games",
            len(model_versions),
            num_matchups,
            num_matchups * tournament_games,
        )
        return job_name

    def wait_for_completion(
        self,
        job_name: str | None = None,
        timeout_seconds: int = 86400,
    ) -> JobResult:
        """Wait for tournament job to complete."""
        job = job_name or self._current_job
        if not job:
            raise VertexJobError("No tournament job to wait for")

        result = self.client.wait_for_completion(job, timeout_seconds)
        self._current_job = None
        return result

    def cancel(self) -> None:
        """Cancel the current tournament job."""
        if self._current_job:
            self.client.cancel_job(self._current_job)
            self._current_job = None
