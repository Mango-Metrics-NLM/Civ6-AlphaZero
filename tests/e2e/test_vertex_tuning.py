from unittest.mock import MagicMock, patch
import sys
import pytest

# Mock google.cloud (namespace pkg) to avoid ImportError if not installed
if "google.cloud" not in sys.modules:
    sys.modules["google.cloud"] = MagicMock()

from alpha_zero.cloud.gcp.config import GCPConfig, GCSPathConfig
from alpha_zero.cloud.gcp.vertex_jobs import VertexTuningJob, VertexAIClient, JobSpec

@pytest.fixture
def gcp_config():
    config = GCPConfig()
    config.project_id = "test-project"
    config.bucket = "test-bucket"
    config.region = "us-central1"
    config.service_account = "test-sa@test-project.iam.gserviceaccount.com"
    config.container_image_uri = "gcr.io/test-project/test-image:latest"
    config.paths = GCSPathConfig(models_prefix="models/test", trajectories_prefix="traj/test", checkpoints_prefix="ckpt/test")
    return config

@pytest.fixture
def mock_aiplatform():
    with patch("alpha_zero.cloud.gcp.vertex_jobs.VertexAIClient._ensure_initialized") as mock_init:
        # Mock the import inside the method
        with patch.dict(sys.modules, {"google.cloud.aiplatform": MagicMock()}):
            yield sys.modules["google.cloud.aiplatform"]

def test_tuning_job_submission(gcp_config, mock_aiplatform):
    """Test submission of a hyperparameter tuning job."""
    client = VertexAIClient(gcp_config.project_id, gcp_config.region)
    job_manager = VertexTuningJob(gcp_config, client)
    
    # Define search space
    parameter_spec = {
        "learning_rate": {
            "doubleValueSpec": {"minValue": 0.0001, "maxValue": 0.01},
            "scaleType": "UNIT_LOG_SCALE"
        }
    }
    metric_spec = {"loss": "MINIMIZE"}
    
    # Submitting job
    job_name = job_manager.launch(
        iteration=1,
        parameter_spec=parameter_spec,
        metric_spec=metric_spec,
        max_trials=10,
        parallel_trials=2,
    )
    
    # Verification
    # 1. CustomJob creation
    assert mock_aiplatform.CustomJob.called
    custom_job_call = mock_aiplatform.CustomJob.call_args
    assert custom_job_call.kwargs["display_name"] == "civ6az-tuning-iter001"
    assert "worker_pool_specs" in custom_job_call.kwargs
    
    # 2. HyperparameterTuningJob creation
    assert mock_aiplatform.HyperparameterTuningJob.called
    tuning_job_call = mock_aiplatform.HyperparameterTuningJob.call_args
    assert tuning_job_call.kwargs["display_name"] == "civ6az-tuning-iter001-tuning"
    assert tuning_job_call.kwargs["max_trial_count"] == 10
    assert tuning_job_call.kwargs["parallel_trial_count"] == 2
    assert tuning_job_call.kwargs["metric_spec"] == metric_spec
    assert tuning_job_call.kwargs["parameter_spec"] == parameter_spec
    
    # 3. Submission
    mock_aiplatform.HyperparameterTuningJob.return_value.submit.assert_called_once()
    
def test_tuning_job_args(gcp_config, mock_aiplatform):
    """Test that arguments are correctly passed to the tuning job."""
    client = VertexAIClient(gcp_config.project_id, gcp_config.region)
    job_manager = VertexTuningJob(gcp_config, client)
    
    extra_args = ["--custom_flag=true"]
    
    job_manager.launch(
        iteration=1,
        parameter_spec={},
        metric_spec={},
        extra_args=extra_args
    )
    
    # Check CustomJob args
    custom_job_call = mock_aiplatform.CustomJob.call_args
    worker_spec = custom_job_call.kwargs["worker_pool_specs"][0]
    container_spec = worker_spec["container_spec"]
    
    # Args should include base args + extra args
    # Note: Vertex AI Vizier appends the hyperparameters as args automatically
    assert "--mode=tuning_worker" in container_spec["args"]
    assert "--iteration=1" in container_spec["args"]
    assert "--custom_flag=true" in container_spec["args"]
