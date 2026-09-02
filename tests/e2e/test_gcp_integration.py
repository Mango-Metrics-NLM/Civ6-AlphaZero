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
import time
from pathlib import Path

import pytest

from alpha_zero.cloud.gcp import GCSCheckpointStorage
from alpha_zero.cloud.gcp.config import GCPConfig

# Configure logging for E2E tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SDK availability check
try:
    from google.cloud import storage as _gcs_sdk  # noqa: F401

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

# Fixtures e2e_cloud_config, test_artifact_path, and cleanup_gcs_artifacts
# are provided by tests/e2e/conftest.py

# Skip marker for SDK availability. The Vertex and Cloud Logging classes that
# used to follow imported names the deleted LEGACY cloud modules never exported;
# they were removed on 2026-09-02 with those modules.
requires_gcs = pytest.mark.skipif(not GCS_AVAILABLE, reason="google-cloud-storage not installed")


@pytest.mark.requires_gcp
@requires_gcs
class TestGCSIntegration:
    """E2E tests for Google Cloud Storage integration."""

    def test_gcs_client_initialization(self, e2e_cloud_config: GCPConfig) -> None:
        """Test that GCS client initializes with real credentials."""
        manager = GCSCheckpointStorage(e2e_cloud_config)
        # GCSCheckpointStorage doesn't have is_mock or _client attributes
        # Just verify it was created successfully
        assert manager is not None
        logger.info(f"GCS client initialized for bucket: {e2e_cloud_config.bucket_name}")  # type: ignore

    def test_gcs_upload_download_small_file(
        self,
        e2e_cloud_config: GCPConfig,
        cleanup_gcs_artifacts: str,
    ) -> None:
        """Test uploading and downloading a small file to GCS."""
        manager = GCSCheckpointStorage(e2e_cloud_config)
        test_path = cleanup_gcs_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create local test file
            local_file = Path(tmpdir) / "test_checkpoint.bin"
            test_content = b"test checkpoint content " * 100  # ~2.4KB
            local_file.write_bytes(test_content)

            # Upload
            remote_key = f"{test_path}/test_checkpoint.bin"
            manager.upload_checkpoint(str(local_file), remote_key)  # type: ignore
            logger.info(f"Uploaded to: gs://{e2e_cloud_config.bucket_name}/{remote_key}")  # type: ignore

            # Verify upload by listing
            checkpoints = manager.list_checkpoints(prefix=test_path)  # type: ignore
            assert any(remote_key in cp for cp in checkpoints), f"Upload not found in {checkpoints}"

            # Download to different location
            download_file = Path(tmpdir) / "downloaded.bin"
            manager.download_checkpoint(remote_key, str(download_file))  # type: ignore

            # Verify content
            downloaded_content = download_file.read_bytes()
            assert downloaded_content == test_content
            logger.info("Upload/download round-trip successful")

    def test_gcs_list_checkpoints(
        self,
        e2e_cloud_config: GCPConfig,
        cleanup_gcs_artifacts: str,
    ) -> None:
        """Test listing checkpoints in GCS bucket."""
        manager = GCSCheckpointStorage(e2e_cloud_config)
        test_path = cleanup_gcs_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            # Upload multiple files
            for i in range(3):
                local_file = Path(tmpdir) / f"checkpoint_{i}.bin"
                local_file.write_bytes(f"checkpoint {i}".encode())
                manager.upload_checkpoint(  # type: ignore
                    str(local_file),
                    f"{test_path}/checkpoint_{i}.bin",
                )

            # List all checkpoints under test path
            checkpoints = manager.list_checkpoints(prefix=test_path)  # type: ignore
            assert len(checkpoints) >= 3
            logger.info(f"Listed {len(checkpoints)} checkpoints")

    def test_gcs_get_latest_checkpoint(
        self,
        e2e_cloud_config: GCPConfig,
        cleanup_gcs_artifacts: str,
    ) -> None:
        """Test getting the latest checkpoint by timestamp."""
        manager = GCSCheckpointStorage(e2e_cloud_config)
        test_path = cleanup_gcs_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            # Upload files with some delay to ensure different timestamps
            for i in range(2):
                local_file = Path(tmpdir) / f"step_{i * 100}.bin"
                local_file.write_bytes(f"step {i * 100}".encode())
                manager.upload_checkpoint(  # type: ignore
                    str(local_file),
                    f"{test_path}/step_{i * 100}.bin",
                )
                time.sleep(0.1)  # Small delay for timestamp ordering

            # Get latest checkpoint
            latest = manager.get_latest_checkpoint(prefix=test_path)  # type: ignore
            assert latest is not None
            assert "step_100" in latest
            logger.info(f"Latest checkpoint: {latest}")
