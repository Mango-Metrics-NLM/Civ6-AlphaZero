"""Unit tests for Vertex AI launcher module."""

from __future__ import annotations

import pytest

from alpha_zero.cloud.config import CloudConfig, GCSConfig, VertexConfig
from alpha_zero.cloud.vertex_launcher import (
    JobInfo,
    JobStatus,
    VertexJobLauncher,
    create_launcher_from_config,
)


class TestJobStatus:
    """Tests for JobStatus enum."""

    def test_status_values(self) -> None:
        """Test status enum values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.SUCCEEDED.value == "succeeded"
        assert JobStatus.FAILED.value == "failed"


class TestJobInfo:
    """Tests for JobInfo dataclass."""

    def test_is_terminal_succeeded(self) -> None:
        """Test terminal state detection for succeeded job."""
        info = JobInfo(
            job_id="test-job",
            display_name="Test",
            status=JobStatus.SUCCEEDED,
        )
        assert info.is_terminal()
        assert info.is_successful()

    def test_is_terminal_failed(self) -> None:
        """Test terminal state detection for failed job."""
        info = JobInfo(
            job_id="test-job",
            display_name="Test",
            status=JobStatus.FAILED,
        )
        assert info.is_terminal()
        assert not info.is_successful()

    def test_is_not_terminal_running(self) -> None:
        """Test non-terminal state for running job."""
        info = JobInfo(
            job_id="test-job",
            display_name="Test",
            status=JobStatus.RUNNING,
        )
        assert not info.is_terminal()
        assert not info.is_successful()

    def test_duration_seconds(self) -> None:
        """Test duration calculation."""
        from datetime import datetime, timedelta

        start = datetime.now()
        end = start + timedelta(hours=1, minutes=30)

        info = JobInfo(
            job_id="test-job",
            display_name="Test",
            status=JobStatus.SUCCEEDED,
            started_at=start,
            ended_at=end,
        )

        assert info.duration_seconds() == 5400  # 1.5 hours in seconds

    def test_duration_not_available(self) -> None:
        """Test duration when times not available."""
        info = JobInfo(
            job_id="test-job",
            display_name="Test",
            status=JobStatus.RUNNING,
        )
        assert info.duration_seconds() is None


class TestVertexJobLauncher:
    """Tests for VertexJobLauncher."""

    @pytest.fixture
    def launcher(self) -> VertexJobLauncher:
        """Create launcher with mock SDK."""
        # Reset mock state before each test
        from alpha_zero.cloud.vertex_launcher import MockAIPlatform

        MockAIPlatform._jobs = {}
        MockAIPlatform._initialized = False

        return VertexJobLauncher(
            project_id="test-project",
            region="us-central1",
            force_mock=True,
        )

    def test_init(self, launcher: VertexJobLauncher) -> None:
        """Test launcher initialization."""
        assert launcher.project_id == "test-project"
        assert launcher.region == "us-central1"

    def test_submit_job(self, launcher: VertexJobLauncher) -> None:
        """Test submitting a training job."""
        job_id = launcher.submit_job(
            display_name="test-training",
            container_image="gcr.io/test/image:latest",
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
        )

        assert job_id is not None
        assert "customJobs" in job_id

    def test_submit_job_with_env_vars(self, launcher: VertexJobLauncher) -> None:
        """Test submitting job with environment variables."""
        job_id = launcher.submit_job(
            display_name="test-training",
            container_image="gcr.io/test/image:latest",
            env_vars={
                "BATCH_SIZE": "256",
                "LEARNING_RATE": "0.001",
            },
        )

        assert job_id is not None

    def test_submit_job_with_args(self, launcher: VertexJobLauncher) -> None:
        """Test submitting job with command arguments."""
        job_id = launcher.submit_job(
            display_name="test-training",
            container_image="gcr.io/test/image:latest",
            command=["python", "-m", "train"],
            args=["--config", "config.yaml"],
        )

        assert job_id is not None

    def test_get_job_status(self, launcher: VertexJobLauncher) -> None:
        """Test getting job status."""
        job_id = launcher.submit_job(
            display_name="test-training",
            container_image="gcr.io/test/image:latest",
        )

        info = launcher.get_job_status(job_id)

        assert info.job_id == job_id
        assert info.display_name == "test-training"
        assert info.status == JobStatus.RUNNING  # Mock starts in RUNNING state

    def test_get_job_status_not_found(self, launcher: VertexJobLauncher) -> None:
        """Test getting status of non-existent job."""
        with pytest.raises(KeyError):
            launcher.get_job_status("nonexistent-job")

    def test_wait_for_completion(self, launcher: VertexJobLauncher) -> None:
        """Test waiting for job completion."""
        # Submit a job that will complete quickly in mock
        job_id = launcher.submit_job(
            display_name="test-training",
            container_image="gcr.io/test/image:latest",
        )

        # Get the mock job and set it to succeeded via internal state
        from alpha_zero.cloud.vertex_launcher import MockAIPlatform

        mock_job = MockAIPlatform.get_job(job_id)
        if mock_job:
            mock_job._state = "JOB_STATE_SUCCEEDED"
            from datetime import datetime

            mock_job._end_time = datetime.now()

        # Wait should return immediately
        info = launcher.wait_for_completion(job_id, poll_interval=0.01, timeout=1)

        assert info.is_terminal()
        assert info.status == JobStatus.SUCCEEDED

    def test_wait_for_completion_timeout(self, launcher: VertexJobLauncher) -> None:
        """Test timeout when waiting for completion."""
        job_id = launcher.submit_job(
            display_name="test-training",
            container_image="gcr.io/test/image:latest",
        )

        # Job stays in RUNNING state, should timeout
        with pytest.raises(TimeoutError):
            launcher.wait_for_completion(job_id, poll_interval=0.01, timeout=0.05)

    def test_cancel_job(self, launcher: VertexJobLauncher) -> None:
        """Test cancelling a job."""
        job_id = launcher.submit_job(
            display_name="test-training",
            container_image="gcr.io/test/image:latest",
        )

        result = launcher.cancel_job(job_id)
        assert result is True

    def test_cancel_job_not_found(self, launcher: VertexJobLauncher) -> None:
        """Test cancelling non-existent job."""
        result = launcher.cancel_job("nonexistent-job")
        assert result is False

    def test_list_jobs(self, launcher: VertexJobLauncher) -> None:
        """Test listing jobs."""
        # Submit multiple jobs
        launcher.submit_job(
            display_name="job-1",
            container_image="gcr.io/test/image:latest",
        )
        launcher.submit_job(
            display_name="job-2",
            container_image="gcr.io/test/image:latest",
        )

        jobs = launcher.list_jobs()

        assert len(jobs) >= 2

    def test_submit_from_config(self, launcher: VertexJobLauncher) -> None:
        """Test submitting job from CloudConfig."""
        config = CloudConfig(
            project_id="test-project",
            region="us-central1",
            gcs=GCSConfig(bucket_name="test-bucket"),
            vertex=VertexConfig(
                machine_type="n1-standard-16",
                accelerator_count=2,
            ),
        )

        job_id = launcher.submit_from_config(
            display_name="config-training",
            container_image="gcr.io/test/image:latest",
            config=config,
        )

        assert job_id is not None


class TestCreateLauncherFromConfig:
    """Tests for factory function."""

    def test_create_from_config(self) -> None:
        """Test creating launcher from config."""
        config = CloudConfig(
            project_id="my-project",
            region="europe-west1",
            gcs=GCSConfig(bucket_name="my-bucket"),
        )

        launcher = create_launcher_from_config(config, force_mock=True)

        assert launcher.project_id == "my-project"
        assert launcher.region == "europe-west1"
        assert launcher.staging_bucket == "gs://my-bucket"

    def test_create_without_bucket(self) -> None:
        """Test creating launcher without bucket."""
        config = CloudConfig(
            project_id="my-project",
            region="us-central1",
        )

        launcher = create_launcher_from_config(config, force_mock=True)

        assert launcher.staging_bucket is None
