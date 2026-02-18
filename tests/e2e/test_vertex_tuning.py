"""Vertex AI hyperparameter tuning job dry-run tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.config import GCPConfig, GCSPathConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_google_cloud(monkeypatch: pytest.MonkeyPatch):
    """Scope the google.cloud.aiplatform mock so it doesn't leak into other tests."""
    mock_aiplatform = MagicMock()
    mock_google = MagicMock()
    mock_google_cloud = MagicMock()
    mock_google_cloud.aiplatform = mock_aiplatform

    import sys

    monkeypatch.setitem(sys.modules, "google", mock_google)
    monkeypatch.setitem(sys.modules, "google.cloud", mock_google_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", mock_aiplatform)


@pytest.fixture
def gcp_config() -> GCPConfig:
    config = GCPConfig()
    config.project_id = "test-project"
    config.bucket = "test-bucket"
    config.region = "us-central1"
    config.service_account = "test-sa@test-project.iam.gserviceaccount.com"
    config.container_image_uri = "gcr.io/test-project/test-image:latest"
    config.paths = GCSPathConfig(
        models_prefix="models/test",
        trajectories_prefix="traj/test",
        checkpoints_prefix="ckpt/test",
    )
    return config


@pytest.fixture
def mock_aiplatform():  # type: ignore[misc]
    """Yield the mock that ``from google.cloud import aiplatform`` resolves to."""
    import sys

    mock_aip = sys.modules["google.cloud.aiplatform"]

    with patch("alpha_zero.cloud.gcp.vertex_jobs.VertexAIClient._ensure_initialized"):
        mock_aip.reset_mock()
        yield mock_aip


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_tuning_job_submission(gcp_config: GCPConfig, mock_aiplatform: MagicMock) -> None:
    """Test submission of a hyperparameter tuning job."""
    from alpha_zero.cloud.gcp.vertex_jobs import VertexAIClient, VertexTuningJob

    client = VertexAIClient(gcp_config.project_id, gcp_config.region)
    job_manager = VertexTuningJob(gcp_config, client)

    parameter_spec = {
        "learning_rate": {
            "doubleValueSpec": {"minValue": 0.0001, "maxValue": 0.01},
            "scaleType": "UNIT_LOG_SCALE",
        }
    }
    metric_spec = {"loss": "MINIMIZE"}

    job_manager.launch(
        iteration=1,
        parameter_spec=parameter_spec,
        metric_spec=metric_spec,
        max_trials=10,
        parallel_trials=2,
    )

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


def test_tuning_job_args(gcp_config: GCPConfig, mock_aiplatform: MagicMock) -> None:
    """Test that arguments are correctly passed to the tuning job."""
    from alpha_zero.cloud.gcp.vertex_jobs import VertexAIClient, VertexTuningJob

    client = VertexAIClient(gcp_config.project_id, gcp_config.region)
    job_manager = VertexTuningJob(gcp_config, client)

    extra_args = ["--custom_flag=true"]

    job_manager.launch(
        iteration=1,
        parameter_spec={},
        metric_spec={},
        extra_args=extra_args,
    )

    custom_job_call = mock_aiplatform.CustomJob.call_args
    worker_spec = custom_job_call.kwargs["worker_pool_specs"][0]
    container_spec = worker_spec["container_spec"]

    assert "--mode=tuning_worker" in container_spec["args"]
    assert "--iteration=1" in container_spec["args"]
    assert "--custom_flag=true" in container_spec["args"]
