"""Unit tests for GCP Cloud Storage client.

This module tests the CloudStorageClient implementation, including:
- File upload/download
- Directory operations
- Checkpoint management
- Error handling
- Fallback mode (no GCP SDK)
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from alpha_zero.cloud.gcp.config import GCPConfig, StorageConfig
from alpha_zero.cloud.gcp.storage import CloudStorageClient
from alpha_zero.cloud.protocols import CloudStorageProtocol


class TestCloudStorageClient:
    """Tests for CloudStorageClient in mock mode."""

    @pytest.fixture
    def config(self) -> GCPConfig:
        """Create test GCP configuration."""
        return GCPConfig(
            project_id="test-project",
            region="us-central1",
            storage=StorageConfig(
                bucket="test-bucket",
                checkpoint_prefix="checkpoints",
                artifact_prefix="artifacts",
            ),
        )

    @pytest.fixture
    def storage_client(self, config: GCPConfig) -> CloudStorageClient:
        """Create storage client in mock mode."""
        return CloudStorageClient(config, force_mock=True)

    def test_implements_protocol(self, config: GCPConfig) -> None:
        """Test client implements CloudStorageProtocol."""
        client = CloudStorageClient(config, force_mock=True)
        assert isinstance(client, CloudStorageProtocol)

    def test_parse_gcs_uri(self, storage_client: CloudStorageClient) -> None:
        """Test GCS URI parsing."""
        bucket, path = storage_client._parse_gcs_uri("gs://my-bucket/path/to/file.pt")
        assert bucket == "my-bucket"
        assert path == "path/to/file.pt"

    def test_parse_gcs_uri_no_prefix(self, storage_client: CloudStorageClient) -> None:
        """Test parsing without gs:// prefix."""
        bucket, path = storage_client._parse_gcs_uri("my-bucket/path/file.pt")
        assert bucket == "my-bucket"
        assert path == "path/file.pt"

    def test_parse_gcs_uri_bucket_only(self, storage_client: CloudStorageClient) -> None:
        """Test parsing URI with bucket only."""
        bucket, path = storage_client._parse_gcs_uri("gs://my-bucket")
        assert bucket == "my-bucket"
        assert path == ""

    def test_upload_file_not_found(self, storage_client: CloudStorageClient) -> None:
        """Test upload with non-existent file."""
        with pytest.raises(FileNotFoundError):
            storage_client.upload_file("/nonexistent/file.pt", "remote.pt")


class TestCloudStorageClientMockOperations:
    """Tests for mock mode operations."""

    @pytest.fixture
    def config(self) -> GCPConfig:
        """Create test configuration."""
        return GCPConfig(
            project_id="test-project",
            storage=StorageConfig(bucket="mock-test-bucket"),
        )

    @pytest.fixture
    def storage_client(self, config: GCPConfig) -> CloudStorageClient:
        """Create storage client in mock mode."""
        return CloudStorageClient(config, force_mock=True)

    @pytest.fixture(autouse=True)
    def cleanup_storage(self) -> Generator[None, None, None]:
        """Clean up storage directory after each test."""
        yield
        storage_dir = Path("storage")
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    def test_mock_upload_and_download(self, storage_client: CloudStorageClient) -> None:
        """Test upload and download in mock mode."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            f.write(b"test data")
            temp_path = Path(f.name)

        try:
            # Upload
            uri = storage_client.upload_file(temp_path, "models/test.pt")
            assert uri.startswith("file://")

            # Verify file was created
            fallback_path = Path("storage/mock-test-bucket/models/test.pt")
            assert fallback_path.exists()

            # Download
            with tempfile.TemporaryDirectory() as temp_dir:
                download_path = Path(temp_dir) / "downloaded.pt"
                result = storage_client.download_file("models/test.pt", download_path)
                assert result == download_path
                assert download_path.read_bytes() == b"test data"
        finally:
            temp_path.unlink()

    def test_mock_exists(self, storage_client: CloudStorageClient) -> None:
        """Test exists check in mock mode."""
        # File doesn't exist
        assert storage_client.exists("models/nonexistent.pt") is False

        # Create file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            f.write(b"test data")
            temp_path = Path(f.name)

        try:
            storage_client.upload_file(temp_path, "models/exists_test.pt")
            assert storage_client.exists("models/exists_test.pt") is True
        finally:
            temp_path.unlink()

    def test_mock_delete(self, storage_client: CloudStorageClient) -> None:
        """Test delete in mock mode."""
        # Can't delete non-existent file
        assert storage_client.delete("models/nonexistent.pt") is False

        # Create file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            f.write(b"test data")
            temp_path = Path(f.name)

        try:
            storage_client.upload_file(temp_path, "models/delete_test.pt")
            assert storage_client.exists("models/delete_test.pt") is True

            # Delete it
            assert storage_client.delete("models/delete_test.pt") is True
            assert storage_client.exists("models/delete_test.pt") is False
        finally:
            temp_path.unlink()


class TestCloudStorageClientFallback:
    """Tests for fallback mode (when SDK is not available)."""

    @pytest.fixture
    def config(self) -> GCPConfig:
        """Create test configuration."""
        return GCPConfig(
            project_id="test-project",
            storage=StorageConfig(bucket="fallback-bucket"),
        )

    @pytest.fixture(autouse=True)
    def cleanup_storage(self) -> Generator[None, None, None]:
        """Clean up storage directory after each test."""
        yield
        storage_dir = Path("storage")
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    def test_fallback_upload(self, config: GCPConfig) -> None:
        """Test fallback upload to local filesystem."""
        # force_mock simulates SDK not available + fallback allowed
        client = CloudStorageClient(config, force_mock=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            f.write(b"test data")
            temp_path = Path(f.name)

        try:
            uri = client.upload_file(temp_path, "models/test.pt")
            assert uri.startswith("file://")

            # Verify file was created
            fallback_path = Path("storage/fallback-bucket/models/test.pt")
            assert fallback_path.exists()
        finally:
            temp_path.unlink()

    def test_fallback_without_flag_raises(self, config: GCPConfig) -> None:
        """Test that creating client without SDK or fallback raises ImportError."""
        with patch("alpha_zero.cloud.gcp.storage.GCS_AVAILABLE", False):
            with pytest.raises(ImportError):
                CloudStorageClient(config, allow_fallback=False)


class TestCheckpointMethods:
    """Tests for checkpoint-specific methods."""

    @pytest.fixture
    def config(self) -> GCPConfig:
        """Create test configuration."""
        return GCPConfig(
            project_id="test-project",
            storage=StorageConfig(
                bucket="checkpoint-bucket",
                checkpoint_prefix="checkpoints",
            ),
        )

    @pytest.fixture
    def storage_client(self, config: GCPConfig) -> CloudStorageClient:
        """Create storage client in mock mode."""
        return CloudStorageClient(config, force_mock=True)

    @pytest.fixture(autouse=True)
    def cleanup_storage(self) -> Generator[None, None, None]:
        """Clean up storage directory after each test."""
        yield
        storage_dir = Path("storage")
        if storage_dir.exists():
            shutil.rmtree(storage_dir)

    def test_save_checkpoint(self, storage_client: CloudStorageClient) -> None:
        """Test saving a checkpoint."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            f.write(b"checkpoint data")
            temp_path = Path(f.name)

        try:
            uri = storage_client.save_checkpoint(temp_path, "epoch_10.pt")
            # Handle both forward and back slashes for cross-platform compatibility
            assert "epoch_10.pt" in uri
            assert "checkpoints" in uri.replace("\\", "/")

            # Verify file was created
            fallback_path = Path("storage/checkpoint-bucket/checkpoints/epoch_10.pt")
            assert fallback_path.exists()
        finally:
            temp_path.unlink()

    def test_load_checkpoint(self, storage_client: CloudStorageClient) -> None:
        """Test loading a checkpoint."""
        # First save a checkpoint
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            f.write(b"checkpoint data")
            temp_path = Path(f.name)

        try:
            storage_client.save_checkpoint(temp_path, "epoch_5.pt")

            # Now load it
            with tempfile.TemporaryDirectory() as temp_dir:
                load_path = Path(temp_dir) / "loaded.pt"
                result = storage_client.load_checkpoint("epoch_5.pt", load_path)
                assert result == load_path
                assert load_path.read_bytes() == b"checkpoint data"
        finally:
            temp_path.unlink()

    def test_list_checkpoints(self, storage_client: CloudStorageClient) -> None:
        """Test listing checkpoints."""
        # Save some checkpoints
        for epoch in [5, 10, 15]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
                f.write(f"checkpoint {epoch}".encode())
                temp_path = Path(f.name)

            try:
                storage_client.save_checkpoint(temp_path, f"epoch_{epoch}.pt")
            finally:
                temp_path.unlink()

        checkpoints = storage_client.list_checkpoints()

        assert len(checkpoints) == 3
        # Check that checkpoint names are in the list (order may vary)
        checkpoint_names = set(checkpoints)
        assert "epoch_5.pt" in checkpoint_names or len([c for c in checkpoints if "epoch_5" in c]) > 0
