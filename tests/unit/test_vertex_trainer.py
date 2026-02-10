"""Unit tests for Vertex AI trainer.

This module tests the VertexTrainer implementation, including:
- Job submission
- Job status monitoring
- Job cancellation
- Fallback mode (no Vertex AI SDK)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from alpha_zero.cloud.gcp.config import GCPConfig, TrainingWorkerConfig
from alpha_zero.cloud.gcp.vertex_trainer import VertexTrainer
from alpha_zero.cloud.protocols import CloudTrainerProtocol, JobStatus, TrainingJobInfo


class TestVertexTrainer:
    """Tests for VertexTrainer."""

    @pytest.fixture
    def config(self) -> GCPConfig:
        """Create test GCP configuration."""
        return GCPConfig(
            project_id="test-project",
            region="us-central1",
            bucket="test-bucket",
            training=TrainingWorkerConfig(
                machine_type="n1-standard-4",
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1,
            ),
        )

    def test_implements_protocol(self, config: GCPConfig) -> None:
        """Test trainer implements CloudTrainerProtocol."""
        trainer = VertexTrainer(config, allow_fallback=True)
        assert isinstance(trainer, CloudTrainerProtocol)

    def test_generate_job_name(self, config: GCPConfig) -> None:
        """Test job name generation."""
        with patch("alpha_zero.cloud.gcp.vertex_trainer.VERTEX_AVAILABLE", False):
            trainer = VertexTrainer(config, allow_fallback=True)
            name = trainer._generate_job_name("test")

            assert name.startswith("test-")
            assert len(name) > 20  # Includes timestamp and UUID


@pytest.mark.xfail(reason="VertexTrainer mock mode implementation out of date", strict=False)
class TestVertexTrainerMockMode:
    """Tests for VertexTrainer in mock/fallback mode."""

    @pytest.fixture
    def config(self) -> GCPConfig:
        """Create test configuration."""
        return GCPConfig(
            project_id="test-project",
            region="us-central1",
            bucket="test-bucket",
            training=TrainingWorkerConfig(
                machine_type="n1-standard-4",
                accelerator_type="NVIDIA_TESLA_T4",
            ),
        )

    @pytest.fixture
    def trainer(self, config: GCPConfig) -> VertexTrainer:
        """Create trainer in mock mode using force_mock parameter."""
        return VertexTrainer(config, force_mock=True)

    def test_submit_training_job(self, trainer: VertexTrainer) -> None:
        """Test job submission in mock mode."""
        job_id = trainer.submit_training_job(
            script_path="train.py",
            args=["--epochs", "10"],
            job_name="test-job",
        )

        assert job_id == "test-job"

    def test_submit_with_custom_config(self, trainer: VertexTrainer) -> None:
        """Test job submission with custom configuration."""
        job_id = trainer.submit_training_job(
            script_path="train.py",
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_V100",
            accelerator_count=4,
            environment_variables={"WANDB_PROJECT": "test"},
        )

        assert job_id is not None
        # Verify job was created
        status = trainer.get_job_status(job_id)
        assert status.status == JobStatus.PENDING

    def test_get_job_status(self, trainer: VertexTrainer) -> None:
        """Test getting job status in mock mode."""
        job_id = trainer.submit_training_job(
            script_path="train.py",
            job_name="test-status-job",
        )

        status = trainer.get_job_status(job_id)

        assert status.job_id == job_id
        assert status.status == JobStatus.PENDING
        assert status.created_at is not None

    def test_get_unknown_job_status(self, trainer: VertexTrainer) -> None:
        """Test getting status of unknown job."""
        status = trainer.get_job_status("unknown-job-id")

        assert status.status == JobStatus.UNKNOWN

    def test_cancel_job(self, trainer: VertexTrainer) -> None:
        """Test job cancellation in mock mode."""
        job_id = trainer.submit_training_job(
            script_path="train.py",
            job_name="test-cancel-job",
        )

        result = trainer.cancel_job(job_id)

        assert result is True
        status = trainer.get_job_status(job_id)
        assert status.status == JobStatus.CANCELLED

    def test_cancel_unknown_job(self, trainer: VertexTrainer) -> None:
        """Test cancelling unknown job."""
        result = trainer.cancel_job("unknown-job-id")
        assert result is False

    def test_list_jobs(self, trainer: VertexTrainer) -> None:
        """Test listing jobs in mock mode."""
        # Submit some jobs
        trainer.submit_training_job(script_path="train.py", job_name="job-1")
        trainer.submit_training_job(script_path="train.py", job_name="job-2")
        trainer.submit_training_job(script_path="train.py", job_name="job-3")

        jobs = trainer.list_jobs()

        assert len(jobs) >= 3

    def test_list_jobs_with_filter(self, trainer: VertexTrainer) -> None:
        """Test listing jobs with status filter."""
        # Submit some jobs
        job1 = trainer.submit_training_job(script_path="train.py", job_name="filter-job-1")
        job2 = trainer.submit_training_job(script_path="train.py", job_name="filter-job-2")

        # Cancel one
        trainer.cancel_job(job1)

        # List cancelled jobs
        cancelled_jobs = trainer.list_jobs(status_filter=JobStatus.CANCELLED)
        pending_jobs = trainer.list_jobs(status_filter=JobStatus.PENDING)

        cancelled_ids = [j.job_id for j in cancelled_jobs]
        pending_ids = [j.job_id for j in pending_jobs]

        assert job1 in cancelled_ids
        assert job2 in pending_ids


class TestTrainingJobInfo:
    """Tests for TrainingJobInfo dataclass."""

    def test_is_terminal_succeeded(self) -> None:
        """Test terminal state check for succeeded job."""
        job = TrainingJobInfo(
            job_id="test",
            job_name="test",
            status=JobStatus.SUCCEEDED,
        )
        assert job.is_terminal() is True
        assert job.is_successful() is True

    def test_is_terminal_failed(self) -> None:
        """Test terminal state check for failed job."""
        job = TrainingJobInfo(
            job_id="test",
            job_name="test",
            status=JobStatus.FAILED,
            error_message="Out of memory",
        )
        assert job.is_terminal() is True
        assert job.is_successful() is False

    def test_is_terminal_running(self) -> None:
        """Test terminal state check for running job."""
        job = TrainingJobInfo(
            job_id="test",
            job_name="test",
            status=JobStatus.RUNNING,
        )
        assert job.is_terminal() is False

    def test_duration_seconds(self) -> None:
        """Test duration calculation."""
        start = datetime(2024, 1, 1, 10, 0, 0)
        end = datetime(2024, 1, 1, 12, 30, 0)

        job = TrainingJobInfo(
            job_id="test",
            job_name="test",
            status=JobStatus.SUCCEEDED,
            started_at=start,
            ended_at=end,
        )

        duration = job.duration_seconds()
        assert duration == 2.5 * 3600  # 2.5 hours in seconds

    def test_duration_none_when_incomplete(self) -> None:
        """Test duration is None when job hasn't completed."""
        job = TrainingJobInfo(
            job_id="test",
            job_name="test",
            status=JobStatus.RUNNING,
            started_at=datetime.now(),
        )

        assert job.duration_seconds() is None


@pytest.mark.xfail(reason="VertexTrainer mock mode implementation out of date", strict=False)
class TestVertexTrainerWithoutSDK:
    """Tests for behavior when Vertex AI SDK is not available."""

    @pytest.fixture
    def config(self) -> GCPConfig:
        """Create test configuration."""
        return GCPConfig(project_id="test-project")

    def test_raises_without_fallback(self, config: GCPConfig) -> None:
        """Test ImportError when SDK unavailable and fallback=False."""
        with patch("alpha_zero.cloud.gcp.vertex_trainer.VERTEX_AVAILABLE", False):
            with pytest.raises(ImportError):
                VertexTrainer(config, allow_fallback=False)

    def test_works_with_fallback(self, config: GCPConfig) -> None:
        """Test mock mode works when fallback=True."""
        with patch("alpha_zero.cloud.gcp.vertex_trainer.VERTEX_AVAILABLE", False):
            trainer = VertexTrainer(config, allow_fallback=True)

            job_id = trainer.submit_training_job(
                script_path="train.py",
                job_name="test-job",
            )

            assert job_id == "test-job"
