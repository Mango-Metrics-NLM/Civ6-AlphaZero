"""Unit tests for Vertex AI trainer."""

from __future__ import annotations

import time
from unittest import mock

import pytest

from alpha_zero.cloud.config import CloudConfig
from alpha_zero.cloud.gcp.vertex import VertexAITrainer, VertexJobConfig, JobStatus


@pytest.fixture
def cloud_config() -> CloudConfig:
    """Create test cloud configuration."""
    return CloudConfig(
        project_id="test-project",
        region="us-central1",
        bucket_name="test-bucket",
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
    )


@pytest.fixture
def mock_trainer(cloud_config: CloudConfig) -> VertexAITrainer:
    """Create mock Vertex AI trainer."""
    return VertexAITrainer(cloud_config, force_mock=True)


class TestVertexJobConfig:
    """Tests for VertexJobConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = VertexJobConfig(display_name="test-job")
        
        assert config.display_name == "test-job"
        assert config.machine_type == "n1-standard-8"
        assert config.accelerator_type == "NVIDIA_TESLA_T4"
        assert config.accelerator_count == 1
        assert config.replica_count == 1

    def test_from_cloud_config(self, cloud_config: CloudConfig) -> None:
        """Test creating job config from CloudConfig."""
        job_config = VertexJobConfig.from_cloud_config(
            cloud_config,
            display_name="training-v1",
            script_path="train.py",
            args=["--epochs", "100"],
        )
        
        assert job_config.display_name == "training-v1"
        assert job_config.script_path == "train.py"
        assert job_config.args == ["--epochs", "100"]
        assert job_config.machine_type == cloud_config.machine_type
        assert job_config.accelerator_type == cloud_config.accelerator_type
        assert "jobs" in job_config.base_output_dir

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = VertexJobConfig(
            display_name="custom-job",
            machine_type="n1-highmem-16",
            accelerator_type="NVIDIA_TESLA_A100",
            accelerator_count=8,
            replica_count=4,
            boot_disk_size_gb=200,
            enable_web_access=True,
            environment_variables={"DEBUG": "1"},
            labels={"env": "test"},
        )
        
        assert config.machine_type == "n1-highmem-16"
        assert config.accelerator_type == "NVIDIA_TESLA_A100"
        assert config.accelerator_count == 8
        assert config.replica_count == 4
        assert config.boot_disk_size_gb == 200
        assert config.enable_web_access is True
        assert config.environment_variables["DEBUG"] == "1"


class TestVertexAITrainerInit:
    """Tests for VertexAITrainer initialization."""

    def test_mock_initialization(self, cloud_config: CloudConfig) -> None:
        """Test mock trainer initialization."""
        trainer = VertexAITrainer(cloud_config, force_mock=True)
        
        assert trainer.is_mock is True
        assert trainer.config == cloud_config

    def test_force_mock_overrides_sdk(self, cloud_config: CloudConfig) -> None:
        """Test that force_mock=True always uses mock."""
        trainer = VertexAITrainer(cloud_config, force_mock=True)
        assert trainer.is_mock is True


class TestVertexAIJobSubmission:
    """Tests for job submission."""

    def test_submit_training_job(self, mock_trainer: VertexAITrainer) -> None:
        """Test submitting a training job."""
        job_config = VertexJobConfig(
            display_name="test-training",
            script_path="train.py",
        )
        
        job_id = mock_trainer.submit_training_job(job_config)
        
        assert job_id is not None
        assert "customJobs" in job_id

    def test_submit_multiple_jobs(self, mock_trainer: VertexAITrainer) -> None:
        """Test submitting multiple jobs."""
        job_ids = []
        
        for i in range(3):
            job_config = VertexJobConfig(display_name=f"job-{i}")
            job_id = mock_trainer.submit_training_job(job_config)
            job_ids.append(job_id)
        
        # All job IDs should be unique
        assert len(set(job_ids)) == 3


class TestVertexAIJobStatus:
    """Tests for job status tracking."""

    def test_get_job_status_pending(self, mock_trainer: VertexAITrainer) -> None:
        """Test getting status of pending job."""
        job_config = VertexJobConfig(display_name="status-test")
        job_id = mock_trainer.submit_training_job(job_config)
        
        # Status should eventually progress (mock simulates this)
        status = mock_trainer.get_job_status(job_id)
        
        assert status in [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.SUCCEEDED]

    def test_get_job_status_unknown(self, mock_trainer: VertexAITrainer) -> None:
        """Test getting status of unknown job."""
        status = mock_trainer.get_job_status("nonexistent-job-id")
        assert status == JobStatus.UNKNOWN

    def test_job_progresses_to_succeeded(self, mock_trainer: VertexAITrainer) -> None:
        """Test that mock job eventually succeeds."""
        job_config = VertexJobConfig(display_name="progress-test")
        job_id = mock_trainer.submit_training_job(job_config)
        
        # Wait a bit for mock to progress
        time.sleep(0.6)
        
        status = mock_trainer.get_job_status(job_id)
        assert status == JobStatus.SUCCEEDED


class TestVertexAIJobControl:
    """Tests for job control operations."""

    def test_cancel_job(self, mock_trainer: VertexAITrainer) -> None:
        """Test cancelling a job."""
        job_config = VertexJobConfig(display_name="cancel-test")
        job_id = mock_trainer.submit_training_job(job_config)
        
        result = mock_trainer.cancel_job(job_id)
        assert result is True
        
        status = mock_trainer.get_job_status(job_id)
        assert status == JobStatus.CANCELLED

    def test_cancel_nonexistent_job(self, mock_trainer: VertexAITrainer) -> None:
        """Test cancelling nonexistent job."""
        result = mock_trainer.cancel_job("nonexistent-job")
        assert result is False

    def test_list_jobs(self, mock_trainer: VertexAITrainer) -> None:
        """Test listing submitted jobs."""
        # Submit some jobs
        for i in range(3):
            job_config = VertexJobConfig(display_name=f"list-test-{i}")
            mock_trainer.submit_training_job(job_config)
        
        jobs = mock_trainer.list_jobs()
        assert len(jobs) >= 3

    def test_list_jobs_with_filter(self, mock_trainer: VertexAITrainer) -> None:
        """Test listing jobs with name filter."""
        # Submit jobs with different names
        mock_trainer.submit_training_job(VertexJobConfig(display_name="alpha-job"))
        mock_trainer.submit_training_job(VertexJobConfig(display_name="beta-job"))
        
        alpha_jobs = mock_trainer.list_jobs(filter_display_name="alpha")
        
        # Should only find alpha jobs
        assert len(alpha_jobs) >= 1
        for job_id in alpha_jobs:
            assert "alpha" in mock_trainer._jobs[job_id].display_name.lower()


class TestVertexAIWaitForCompletion:
    """Tests for wait_for_completion."""

    def test_wait_for_completion_success(self, mock_trainer: VertexAITrainer) -> None:
        """Test waiting for job completion."""
        job_config = VertexJobConfig(display_name="wait-test")
        job_id = mock_trainer.submit_training_job(job_config)
        
        # Mock job should complete quickly
        status = mock_trainer.wait_for_completion(
            job_id,
            timeout=5.0,
            poll_interval=0.1,
        )
        
        assert status == JobStatus.SUCCEEDED

    def test_wait_for_cancelled_job(self, mock_trainer: VertexAITrainer) -> None:
        """Test waiting for cancelled job."""
        job_config = VertexJobConfig(display_name="cancel-wait-test")
        job_id = mock_trainer.submit_training_job(job_config)
        
        mock_trainer.cancel_job(job_id)
        
        status = mock_trainer.wait_for_completion(
            job_id,
            timeout=1.0,
            poll_interval=0.1,
        )
        
        assert status == JobStatus.CANCELLED
