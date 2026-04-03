"""Coverage tests for alpha_zero.storage.factory.

Covers:
- create_storage_backend with local, gcs, spacetimedb, and invalid backend types
- Vertex AI auto-detection via AIP_CHECKPOINT_DIR
- get_storage_backend_from_env convenience function
- ImportError handling for GCS and SpacetimeDB backends
- ValueError for missing GCS bucket
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.storage.config import StorageConfig
from alpha_zero.storage.factory import create_storage_backend, get_storage_backend_from_env
from alpha_zero.storage.local_backend import LocalStorageBackend


class TestCreateStorageBackendLocal:
    def test_default_creates_local(self, tmp_path):
        config = StorageConfig.for_local(str(tmp_path))
        backend = create_storage_backend(config)
        assert isinstance(backend, LocalStorageBackend)

    def test_explicit_local_type(self, tmp_path):
        config = StorageConfig.for_local(str(tmp_path))
        backend = create_storage_backend(config, backend_type="local")
        assert isinstance(backend, LocalStorageBackend)

    def test_no_config_uses_env(self, tmp_path):
        with patch.dict(
            os.environ,
            {
                "CIV6_STORAGE_BACKEND": "local",
                "CIV6_STORAGE_BASE_PATH": str(tmp_path),
            },
            clear=False,
        ):
            backend = create_storage_backend()
            assert isinstance(backend, LocalStorageBackend)


class TestCreateStorageBackendGCS:
    def test_gcs_import_error(self, tmp_path):
        config = StorageConfig.for_local(str(tmp_path))
        config.backend_type = "gcs"
        config.gcs.bucket = "test-bucket"
        with patch.dict("sys.modules", {"alpha_zero.storage.gcs_backend": None}):
            with pytest.raises(ImportError, match="google-cloud-storage"):
                create_storage_backend(config)

    def test_gcs_missing_bucket(self, tmp_path):
        config = StorageConfig.for_local(str(tmp_path))
        config.backend_type = "gcs"
        config.gcs.bucket = ""
        # Mock the import so it doesn't fail on ImportError
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"alpha_zero.storage.gcs_backend": mock_module}):
            with pytest.raises(ValueError, match="GCS bucket name is required"):
                create_storage_backend(config)


class TestCreateStorageBackendSpacetimeDB:
    def test_spacetimedb_import_error(self, tmp_path):
        config = StorageConfig.for_local(str(tmp_path))
        config.backend_type = "spacetimedb"
        with patch.dict("sys.modules", {"alpha_zero.storage.spacetimedb_backend": None}):
            with pytest.raises(ImportError, match="spacetimedb-sdk"):
                create_storage_backend(config)


class TestCreateStorageBackendInvalid:
    def test_invalid_backend_type(self, tmp_path):
        config = StorageConfig.for_local(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid backend type"):
            create_storage_backend(config, backend_type="redis")


class TestVertexAIAutoDetection:
    def test_aip_checkpoint_dir_switches_to_gcs(self, tmp_path):
        config = StorageConfig.for_local(str(tmp_path))
        config.gcs.bucket = "my-bucket"

        mock_gcs_module = MagicMock()
        mock_gcs_backend = MagicMock()
        mock_gcs_module.GCSStorageBackend.return_value = mock_gcs_backend

        with patch.dict(
            os.environ, {"AIP_CHECKPOINT_DIR": "gs://my-bucket/checkpoints"}, clear=False
        ):
            with patch.dict("sys.modules", {"alpha_zero.storage.gcs_backend": mock_gcs_module}):
                backend = create_storage_backend(config)

        assert backend is mock_gcs_backend

    def test_aip_non_gs_path_stays_local(self, tmp_path):
        config = StorageConfig.for_local(str(tmp_path))
        with patch.dict(os.environ, {"AIP_CHECKPOINT_DIR": "/local/path"}, clear=False):
            backend = create_storage_backend(config)
        assert isinstance(backend, LocalStorageBackend)


class TestGetStorageBackendFromEnv:
    def test_creates_local_from_env(self, tmp_path):
        with patch.dict(
            os.environ,
            {
                "CIV6_STORAGE_BACKEND": "local",
                "CIV6_STORAGE_BASE_PATH": str(tmp_path),
            },
            clear=False,
        ):
            backend = get_storage_backend_from_env()
            assert isinstance(backend, LocalStorageBackend)
