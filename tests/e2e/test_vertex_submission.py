"""Dry-run tests for Vertex AI job submission.

Verifies that job specifications are correctly constructed and submitted
to the Vertex AI SDK, without actually triggering cloud resources.
"""

from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.config import GCPConfig, SelfPlayWorkerConfig, TrainingWorkerConfig
from alpha_zero.cloud.gcp.vertex_jobs import (
    VertexAIClient,
    VertexEvaluationJob,
    VertexSelfPlayJob,
    VertexTrainingJob,
)


@pytest.fixture
def gcp_config():
    """Create a standard GCP config for testing."""
    config = GCPConfig(
        project_id="test-project",
        bucket="test-bucket",
        region="us-central1",
        container_image_uri="gcr.io/test/image:latest",
    )
    # Configure sub-components
    config.self_play = SelfPlayWorkerConfig(
        machine_type="n2-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
    )
    config.training = TrainingWorkerConfig(
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_V100",
        accelerator_count=2,  # DDP trigger
    )
    return config


@pytest.fixture
def mock_aiplatform():
    """Mock the google.cloud.aiplatform module."""
    import sys

    mock = MagicMock()

    # Create a mock for google.cloud package
    mock_google = MagicMock()
    mock_cloud = MagicMock()
    mock_cloud.aiplatform = mock
    mock_google.cloud = mock_cloud

    # Patch sys.modules to intercept the import
    with patch.dict(
        sys.modules,
        {
            "google": mock_google,
            "google.cloud": mock_cloud,
            "google.cloud.aiplatform": mock,
        },
    ):
        # Setup CustomJob mock
        mock_job = MagicMock()
        mock_job.resource_name = "projects/123/locations/us-central1/customJobs/456"
        mock.CustomJob.return_value = mock_job
        yield mock


class TestVertexSubmission:
    """Verify Vertex AI job specs."""

    def test_self_play_submission(self, gcp_config, mock_aiplatform):
        """Verify self-play job construction."""
        client = VertexSelfPlayJob(config=gcp_config)

        # Launch
        job_names = client.launch(
            num_workers=2,
            model_version=10,
            iteration=5,
        )

        assert len(job_names) == 2
        assert mock_aiplatform.CustomJob.call_count == 2

        # Verify first worker spec
        call_args = mock_aiplatform.CustomJob.call_args_list[0]
        _, kwargs = call_args

        # Check display name
        assert kwargs["display_name"] == "civ6az-selfplay-iter005-worker00"

        # Check worker pool spec
        worker_specs = kwargs["worker_pool_specs"]
        worker_spec = worker_specs[0]
        assert worker_spec["container_spec"]["image_uri"] == "gcr.io/test/image:latest"

        # Check args
        cmd_args = worker_spec["container_spec"].get("args", [])
        assert "--mode=selfplay" in cmd_args
        assert "--worker_id=0" in cmd_args

        # Check machine spec
        assert worker_spec["machine_spec"]["machine_type"] == "n2-standard-4"
        assert worker_spec["machine_spec"]["accelerator_type"] == "NVIDIA_TESLA_T4"

    def test_training_submission_ddp(self, gcp_config, mock_aiplatform):
        """Verify training job construction with DDP (torchrun)."""
        client = VertexTrainingJob(config=gcp_config)

        client.launch(
            iteration=5,
            trajectory_iterations=[3, 4, 5],
        )

        assert mock_aiplatform.CustomJob.call_count == 1
        _, kwargs = mock_aiplatform.CustomJob.call_args

        worker_spec = kwargs["worker_pool_specs"][0]
        command = worker_spec["container_spec"].get("command", [])
        args = worker_spec["container_spec"].get("args", [])

        # Verify torchrun Usage since accelerator_count=2
        assert "torchrun" in command
        assert "--nproc_per_node=2" in command

        # Verify args
        assert "--mode=training" in args
        assert "--iteration=5" in args

        # Verify machine spec
        assert worker_spec["machine_spec"]["machine_type"] == "n1-standard-8"
        assert worker_spec["machine_spec"]["accelerator_count"] == 2

    def test_evaluation_submission(self, gcp_config, mock_aiplatform):
        """Verify evaluation job construction."""
        # Reuse the client from training/selfplay or create new one
        # Use a real client wrapper which uses the mocked SDK
        client_wrapper = VertexAIClient(gcp_config.project_id, gcp_config.region)
        eval_client = VertexEvaluationJob(config=gcp_config, client=client_wrapper)

        eval_client.launch(
            model_version_new=10,
            model_version_old=5,
            iteration=5,
        )

        _, kwargs = mock_aiplatform.CustomJob.call_args
        worker_spec = kwargs["worker_pool_specs"][0]

        assert worker_spec["machine_spec"]["machine_type"] == "n1-standard-4"
        assert "accelerator_count" not in worker_spec["machine_spec"]

        args = worker_spec["container_spec"].get("args", [])
        assert "--mode=evaluation" in args
