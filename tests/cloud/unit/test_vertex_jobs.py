"""
Unit tests for Vertex AI job wrappers.

Uses mock Vertex AI client to test without real GCP connection.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.exceptions import PreemptionError, QuotaExceededError, VertexJobError
from alpha_zero.cloud.gcp.vertex_jobs import (
    JobResult,
    JobSpec,
    JobState,
    VertexAIClient,
    VertexSelfPlayJob,
    VertexTrainingJob,
)

# =============================================================================
# Mock Vertex AI Client
# =============================================================================


class MockVertexAIClient(VertexAIClient):
    """Mock Vertex AI client for testing."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._job_counter = 0
        self._should_fail_submit = False
        self._should_quota_fail = False
        self._preempt_jobs: set[str] = set()

    def _ensure_initialized(self) -> None:
        """Override to do nothing."""
        pass

    def submit_custom_job(
        self,
        spec: JobSpec,
        staging_bucket: str | None = None,
    ) -> str:
        """Submit a mock job."""
        if self._should_quota_fail:
            raise QuotaExceededError("Mock quota exceeded")
        if self._should_fail_submit:
            raise VertexJobError("Mock submit failure")

        self._job_counter += 1
        job_name = f"projects/test/locations/us-central1/customJobs/{self._job_counter}"

        self._jobs[job_name] = {
            "display_name": spec.display_name,
            "state": JobState.RUNNING.value,
            "create_time": "2026-01-01T00:00:00Z",
            "start_time": "2026-01-01T00:01:00Z",
            "end_time": None,
        }

        return job_name

    def get_job_state(self, job_name: str) -> tuple[str, dict[str, Any]]:
        """Get mock job state."""
        if job_name not in self._jobs:
            raise VertexJobError(f"Job not found: {job_name}")

        job = self._jobs[job_name]
        return job["state"], job

    def cancel_job(self, job_name: str) -> None:
        """Cancel a mock job."""
        if job_name not in self._jobs:
            raise VertexJobError(f"Job not found: {job_name}")
        self._jobs[job_name]["state"] = JobState.CANCELLED.value

    def wait_for_completion(
        self,
        job_name: str,
        timeout_seconds: int = 86400,
        poll_interval_seconds: int = 30,
    ) -> JobResult:
        """Wait for mock job completion."""
        if job_name not in self._jobs:
            raise VertexJobError(f"Job not found: {job_name}")

        job = self._jobs[job_name]

        # Check if this job should be preempted
        if job_name in self._preempt_jobs:
            raise PreemptionError("Mock preemption", job_id=job_name)

        # Simulate successful completion for non-preempted jobs
        job["state"] = JobState.SUCCEEDED.value
        job["end_time"] = "2026-01-01T01:00:00Z"

        return JobResult(
            job_id=job_name,
            job_name=job["display_name"],
            state=job["state"],
            success=True,
            start_time=job["start_time"],
            end_time=job["end_time"],
            metadata=job,
        )

    def set_job_will_preempt(self, job_name: str) -> None:
        """Mark a job to be preempted."""
        self._preempt_jobs.add(job_name)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gcp_config() -> GCPConfig:
    """Create test GCP config."""
    return GCPConfig(
        project_id="test-project",
        region="us-central1",
        bucket="test-bucket",
        container_image_uri="gcr.io/test-project/civ6az:latest",
    )


@pytest.fixture
def mock_client() -> MockVertexAIClient:
    """Create mock Vertex AI client."""
    return MockVertexAIClient()


# =============================================================================
# JobSpec Tests
# =============================================================================


class TestJobSpec:
    """Tests for JobSpec class."""

    def test_to_worker_pool_spec_basic(self) -> None:
        """Test basic worker pool spec generation."""
        spec = JobSpec(
            display_name="test-job",
            container_uri="gcr.io/test/image:latest",
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
        )

        pool_spec = spec.to_worker_pool_spec()

        assert pool_spec["replica_count"] == 1
        assert pool_spec["machine_spec"]["machine_type"] == "n1-standard-8"
        assert pool_spec["machine_spec"]["accelerator_type"] == "NVIDIA_TESLA_T4"
        assert pool_spec["container_spec"]["image_uri"] == "gcr.io/test/image:latest"

    def test_to_worker_pool_spec_with_command_and_args(self) -> None:
        """Test worker pool spec with command and args."""
        spec = JobSpec(
            display_name="test-job",
            container_uri="gcr.io/test/image:latest",
            command=["python", "main.py"],
            args=["--mode=train", "--epochs=10"],
        )

        pool_spec = spec.to_worker_pool_spec()

        assert pool_spec["container_spec"]["command"] == ["python", "main.py"]
        assert pool_spec["container_spec"]["args"] == ["--mode=train", "--epochs=10"]

    def test_to_worker_pool_spec_with_env_vars(self) -> None:
        """Test worker pool spec with environment variables."""
        spec = JobSpec(
            display_name="test-job",
            container_uri="gcr.io/test/image:latest",
            env_vars={"KEY1": "value1", "KEY2": "value2"},
        )

        pool_spec = spec.to_worker_pool_spec()

        env_list = pool_spec["container_spec"]["env"]
        assert len(env_list) == 2
        assert {"name": "KEY1", "value": "value1"} in env_list


# =============================================================================
# JobState Tests
# =============================================================================


class TestJobState:
    """Tests for JobState enum."""

    def test_is_terminal(self) -> None:
        """Test terminal state detection."""
        assert JobState.is_terminal(JobState.SUCCEEDED.value)
        assert JobState.is_terminal(JobState.FAILED.value)
        assert JobState.is_terminal(JobState.CANCELLED.value)
        assert not JobState.is_terminal(JobState.RUNNING.value)
        assert not JobState.is_terminal(JobState.PENDING.value)

    def test_is_running(self) -> None:
        """Test running state detection."""
        assert JobState.is_running(JobState.RUNNING.value)
        assert JobState.is_running(JobState.PENDING.value)
        assert JobState.is_running(JobState.QUEUED.value)
        assert not JobState.is_running(JobState.SUCCEEDED.value)


# =============================================================================
# VertexSelfPlayJob Tests
# =============================================================================


class TestVertexSelfPlayJob:
    """Tests for self-play job management."""

    def test_launch_workers(
        self,
        gcp_config: GCPConfig,
        mock_client: MockVertexAIClient,
    ) -> None:
        """Test launching multiple self-play workers."""
        job = VertexSelfPlayJob(gcp_config, client=mock_client)

        job_names = job.launch(
            num_workers=3,
            model_version=1,
            iteration=10,
        )

        assert len(job_names) == 3
        assert all(name.startswith("projects/") for name in job_names)

    def test_wait_for_completion(
        self,
        gcp_config: GCPConfig,
        mock_client: MockVertexAIClient,
    ) -> None:
        """Test waiting for self-play jobs to complete."""
        job = VertexSelfPlayJob(gcp_config, client=mock_client)

        job_names = job.launch(num_workers=2, model_version=1, iteration=5)
        results = job.wait_for_completion(job_names)

        assert len(results) == 2
        assert all(r.success for r in results)

    def test_handles_preemption(
        self,
        gcp_config: GCPConfig,
        mock_client: MockVertexAIClient,
    ) -> None:
        """Test that preemption is handled gracefully."""
        job = VertexSelfPlayJob(gcp_config, client=mock_client)

        job_names = job.launch(num_workers=3, model_version=1, iteration=1)

        # Mark one job for preemption
        mock_client.set_job_will_preempt(job_names[1])

        results = job.wait_for_completion(job_names)

        # Should complete without raising, but one job failed
        assert len(results) == 3
        assert results[0].success
        assert not results[1].success
        assert results[2].success

    def test_cancel_all(
        self,
        gcp_config: GCPConfig,
        mock_client: MockVertexAIClient,
    ) -> None:
        """Test cancelling all active jobs."""
        job = VertexSelfPlayJob(gcp_config, client=mock_client)

        job_names = job.launch(num_workers=2, model_version=1, iteration=1)
        job.cancel_all(job_names)

        # Verify jobs are cancelled
        for job_name in job_names:
            state, _ = mock_client.get_job_state(job_name)
            assert state == JobState.CANCELLED.value

    def test_no_container_uri_raises(
        self,
        mock_client: MockVertexAIClient,
    ) -> None:
        """Test that missing container URI raises error."""
        config = GCPConfig(
            project_id="test",
            bucket="test-bucket",
            container_image_uri=None,
        )
        job = VertexSelfPlayJob(config, client=mock_client)

        with pytest.raises(VertexJobError, match="No container image"):
            job.launch(num_workers=1, model_version=1, iteration=1)


# =============================================================================
# VertexTrainingJob Tests
# =============================================================================


class TestVertexTrainingJob:
    """Tests for training job management."""

    def test_launch_training_job(
        self,
        gcp_config: GCPConfig,
        mock_client: MockVertexAIClient,
    ) -> None:
        """Test launching a training job."""
        job = VertexTrainingJob(gcp_config, client=mock_client)

        job_name = job.launch(
            iteration=10,
            trajectory_iterations=[8, 9, 10],
        )

        assert job_name.startswith("projects/")

    def test_launch_ddp_training_job(
        self,
        gcp_config: GCPConfig,
        mock_client: MockVertexAIClient,
    ) -> None:
        """Test launching a multi-GPU training job uses torchrun."""
        # Configure for multi-GPU
        gcp_config.training.accelerator_count = 2

        job = VertexTrainingJob(gcp_config, client=mock_client)

        with patch.object(
            mock_client, "submit_custom_job", wraps=mock_client.submit_custom_job
        ) as mock_submit:
            job.launch(
                iteration=10,
                trajectory_iterations=[10],
            )

            # Verify submit_custom_job was called
            mock_submit.assert_called_once()

            # Inspect the JobSpec passed to submit_custom_job
            call_args = mock_submit.call_args
            job_spec: JobSpec = call_args[0][0]

            # Verify command uses torchrun
            assert job_spec.command is not None
            assert job_spec.command[0] == "torchrun"
            assert "--nproc_per_node=2" in job_spec.command
            assert "alpha_zero.cloud.gcp.training_worker" in job_spec.command

            # Verify worker args are passed correctly
            assert "--iteration=10" in job_spec.args
            assert any("--trajectory_iterations=" in a for a in job_spec.args)

    def test_launch_with_resume(
        self,
        gcp_config: GCPConfig,
        mock_client: MockVertexAIClient,
    ) -> None:
        """Test launching with checkpoint resume."""
        job = VertexTrainingJob(gcp_config, client=mock_client)

        job_name = job.launch(
            iteration=10,
            trajectory_iterations=[8, 9, 10],
            resume_from_checkpoint="gs://bucket/checkpoints/step_1000.pt",
        )

        assert job_name.startswith("projects/")

    def test_wait_for_completion(
        self,
        gcp_config: GCPConfig,
        mock_client: MockVertexAIClient,
    ) -> None:
        """Test waiting for training job completion."""
        job = VertexTrainingJob(gcp_config, client=mock_client)

        job_name = job.launch(iteration=10, trajectory_iterations=[10])
        result = job.wait_for_completion(job_name)

        assert result.success
        assert result.state == JobState.SUCCEEDED.value

    def test_cancel(
        self,
        gcp_config: GCPConfig,
        mock_client: MockVertexAIClient,
    ) -> None:
        """Test cancelling training job."""
        job = VertexTrainingJob(gcp_config, client=mock_client)

        job_name = job.launch(iteration=10, trajectory_iterations=[10])
        job.cancel()

        state, _ = mock_client.get_job_state(job_name)
        assert state == JobState.CANCELLED.value

    def test_no_current_job_raises(
        self,
        gcp_config: GCPConfig,
        mock_client: MockVertexAIClient,
    ) -> None:
        """Test that waiting without job raises error."""
        job = VertexTrainingJob(gcp_config, client=mock_client)

        with pytest.raises(VertexJobError, match="No training job"):
            job.wait_for_completion()


# =============================================================================
# VertexAIClient Tests
# =============================================================================


class TestVertexAIClient:
    """Tests for Vertex AI client initialization."""

    def test_init_without_gcp_raises(self) -> None:
        """Test that missing google-cloud-aiplatform raises clear error."""
        client = VertexAIClient(
            project="test",
            region="us-central1",
        )

        # Mock ImportError for aiplatform
        with patch.dict("sys.modules", {"google.cloud.aiplatform": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(VertexJobError, match="google-cloud-aiplatform not installed"):
                    client._ensure_initialized()
