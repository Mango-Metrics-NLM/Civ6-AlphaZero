"""End-to-end tests for GCP cloud integration.

These tests interact with real GCP resources and require:
1. Active GCP credentials (gcloud auth application-default login)
2. GCP_PROJECT_ID environment variable set
3. GCS_BUCKET_NAME environment variable set

Tests are skipped automatically when credentials are unavailable.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from alpha_zero.cloud.config import CloudConfig
from alpha_zero.cloud.gcp import (
    GCS_AVAILABLE,
    VERTEX_AVAILABLE,
    GCSCheckpointManager,
)

# Configure logging for E2E tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def e2e_cloud_config() -> CloudConfig:
    """Create cloud configuration from environment for E2E testing.

    This fixture relies on environment variables being set:
    - GCP_PROJECT_ID
    - GCS_BUCKET_NAME
    - GCP_REGION (optional, defaults to us-central1)
    """
    config = CloudConfig.from_env()
    config.validate()
    return config


@pytest.fixture
def test_artifact_path() -> str:
    """Generate unique path for test artifacts to avoid conflicts."""
    return f"e2e-tests/{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup_gcs_artifacts(
    e2e_cloud_config: CloudConfig,
    test_artifact_path: str,
) -> Generator[str, None, None]:
    """Fixture that yields path and cleans up after test.

    Yields:
        The test artifact path for use in tests.
    """
    yield test_artifact_path

    # Cleanup: Delete test artifacts from GCS
    if GCS_AVAILABLE:
        try:
            manager = GCSCheckpointManager(e2e_cloud_config)
            # List and delete all objects with the test prefix
            checkpoints = manager.list_checkpoints(prefix=test_artifact_path)
            for cp in checkpoints:
                try:
                    manager._client.bucket(e2e_cloud_config.bucket_name).blob(cp).delete()
                    logger.info(f"Cleaned up: {cp}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {cp}: {e}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


@pytest.mark.requires_gcp
class TestGCSIntegration:
    """E2E tests for Google Cloud Storage integration."""

    def test_gcs_client_initialization(self, e2e_cloud_config: CloudConfig) -> None:
        """Test that GCS client initializes with real credentials."""
        if not GCS_AVAILABLE:
            pytest.skip("google-cloud-storage not installed")

        manager = GCSCheckpointManager(e2e_cloud_config)
        assert manager.is_mock is False
        assert manager._client is not None
        logger.info(f"GCS client initialized for bucket: {e2e_cloud_config.bucket_name}")

    def test_gcs_upload_download_small_file(
        self,
        e2e_cloud_config: CloudConfig,
        cleanup_gcs_artifacts: str,
    ) -> None:
        """Test uploading and downloading a small file to GCS."""
        if not GCS_AVAILABLE:
            pytest.skip("google-cloud-storage not installed")

        manager = GCSCheckpointManager(e2e_cloud_config)
        test_path = cleanup_gcs_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create local test file
            local_file = Path(tmpdir) / "test_checkpoint.bin"
            test_content = b"test checkpoint content " * 100  # ~2.4KB
            local_file.write_bytes(test_content)

            # Upload
            remote_key = f"{test_path}/test_checkpoint.bin"
            manager.upload_checkpoint(str(local_file), remote_key)
            logger.info(f"Uploaded to: gs://{e2e_cloud_config.bucket_name}/{remote_key}")

            # Verify upload by listing
            checkpoints = manager.list_checkpoints(prefix=test_path)
            assert any(remote_key in cp for cp in checkpoints), f"Upload not found in {checkpoints}"

            # Download to different location
            download_file = Path(tmpdir) / "downloaded.bin"
            manager.download_checkpoint(remote_key, str(download_file))

            # Verify content
            downloaded_content = download_file.read_bytes()
            assert downloaded_content == test_content
            logger.info("Upload/download round-trip successful")

    def test_gcs_list_checkpoints(
        self,
        e2e_cloud_config: CloudConfig,
        cleanup_gcs_artifacts: str,
    ) -> None:
        """Test listing checkpoints in GCS bucket."""
        if not GCS_AVAILABLE:
            pytest.skip("google-cloud-storage not installed")

        manager = GCSCheckpointManager(e2e_cloud_config)
        test_path = cleanup_gcs_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            # Upload multiple files
            for i in range(3):
                local_file = Path(tmpdir) / f"checkpoint_{i}.bin"
                local_file.write_bytes(f"checkpoint {i}".encode())
                manager.upload_checkpoint(
                    str(local_file),
                    f"{test_path}/checkpoint_{i}.bin",
                )

            # List all checkpoints under test path
            checkpoints = manager.list_checkpoints(prefix=test_path)
            assert len(checkpoints) >= 3
            logger.info(f"Listed {len(checkpoints)} checkpoints")

    def test_gcs_get_latest_checkpoint(
        self,
        e2e_cloud_config: CloudConfig,
        cleanup_gcs_artifacts: str,
    ) -> None:
        """Test getting the latest checkpoint by timestamp."""
        if not GCS_AVAILABLE:
            pytest.skip("google-cloud-storage not installed")

        manager = GCSCheckpointManager(e2e_cloud_config)
        test_path = cleanup_gcs_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            # Upload files with some delay to ensure different timestamps
            import time

            for i in range(2):
                local_file = Path(tmpdir) / f"step_{i * 100}.bin"
                local_file.write_bytes(f"step {i * 100}".encode())
                manager.upload_checkpoint(
                    str(local_file),
                    f"{test_path}/step_{i * 100}.bin",
                )
                time.sleep(0.1)  # Small delay for timestamp ordering

            # Get latest checkpoint
            latest = manager.get_latest_checkpoint(prefix=test_path)
            assert latest is not None
            assert "step_100" in latest
            logger.info(f"Latest checkpoint: {latest}")


@pytest.mark.requires_gcp
class TestVertexAIIntegration:
    """E2E tests for Vertex AI integration."""

    def test_vertex_trainer_initialization(self, e2e_cloud_config: CloudConfig) -> None:
        """Test that Vertex AI trainer initializes with real credentials."""
        if not VERTEX_AVAILABLE:
            pytest.skip("google-cloud-aiplatform not installed")

        from alpha_zero.cloud.gcp import VertexAITrainer

        trainer = VertexAITrainer(e2e_cloud_config)
        assert trainer.is_mock is False
        logger.info(f"Vertex AI trainer initialized for project: {e2e_cloud_config.project_id}")

    def test_vertex_job_spec_validation(self, e2e_cloud_config: CloudConfig) -> None:
        """Test creating a valid Vertex AI job spec (without submission)."""
        if not VERTEX_AVAILABLE:
            pytest.skip("google-cloud-aiplatform not installed")

        from alpha_zero.cloud.gcp import VertexAITrainer

        trainer = VertexAITrainer(e2e_cloud_config)

        # Create job spec without submitting
        job_spec = trainer._create_job_spec(
            display_name="e2e-test-job-validation",
            container_uri="gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest",
            args=["--test", "true"],
        )

        # Validate spec structure
        assert job_spec is not None
        assert "workerPoolSpecs" in job_spec or hasattr(job_spec, "worker_pool_specs")
        logger.info("Job spec validation passed")


@pytest.mark.requires_gcp
class TestCloudLoggingIntegration:
    """E2E tests for Cloud Logging integration."""

    def test_cloud_logger_initialization(self, e2e_cloud_config: CloudConfig) -> None:
        """Test that Cloud Logger initializes with real credentials."""
        from alpha_zero.cloud.gcp import CloudLogger

        logger_instance = CloudLogger(
            e2e_cloud_config,
            log_name="e2e-test-log",
        )

        # Should not be in mock mode
        assert logger_instance.is_mock is False
        logger.info("Cloud Logger initialized successfully")

    def test_cloud_logger_write(self, e2e_cloud_config: CloudConfig) -> None:
        """Test writing a log entry to Cloud Logging."""
        from alpha_zero.cloud.gcp import CloudLogger

        cloud_logger = CloudLogger(
            e2e_cloud_config,
            log_name="e2e-test-log",
        )

        # Write test log entry
        cloud_logger.log_event(
            severity="INFO",
            message="E2E test log entry",
            payload={"test_id": uuid.uuid4().hex, "source": "e2e_test"},
        )

        # Flush to ensure write
        cloud_logger.flush()
        logger.info("Log entry written to Cloud Logging")


@pytest.mark.requires_gcp
class TestFullCloudWorkflow:
    """E2E tests for complete cloud training workflow."""

    def test_coordinator_real_gcs(
        self,
        e2e_cloud_config: CloudConfig,
        cleanup_gcs_artifacts: str,
    ) -> None:
        """Test CloudTrainingCoordinator with real GCS storage."""
        if not GCS_AVAILABLE:
            pytest.skip("google-cloud-storage not installed")

        from alpha_zero.distributed.cloud_coordinator import (
            CloudCoordinatorConfig,
            CloudTrainingCoordinator,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure coordinator with real GCS
            config = CloudCoordinatorConfig(
                cloud_config=e2e_cloud_config,
                checkpoint_dir=tmpdir,
                sync_interval=1.0,  # Fast sync for testing
                enable_cloud_checkpoints=True,
                enable_preemption_handling=False,  # Disable for E2E to avoid signal handling
            )

            coordinator = CloudTrainingCoordinator(config)
            coordinator.initialize(force_mock=False)

            try:
                # Log some metrics
                coordinator.log_training_metrics(
                    step=1,
                    metrics={"loss": 0.5, "e2e_test": 1.0},
                )

                logger.info("CloudTrainingCoordinator E2E test passed")
            finally:
                coordinator.shutdown()
