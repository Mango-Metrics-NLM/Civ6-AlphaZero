"""E2E tests for live GCS storage operations.

These tests are skipped by default — they upload to and delete from a REAL
bucket, so they require an explicit opt-in:
  - CIV6AZ_LIVE_GCS_TESTS=1
  - GCP credentials (Application Default Credentials or service account)
  - A valid GCS bucket (set via CIV6AZ_GCS_BUCKET env var)

Run with:  CIV6AZ_LIVE_GCS_TESTS=1 pytest tests/e2e/test_gcs_live.py -m "requires_env"

The opt-in is checked before any bucket value is read: a workstation that merely
has ``CIV6AZ_GCS_BUCKET`` and Application Default Credentials must never reach
the bucket through an ordinary suite run.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Generator
from typing import Any

import pytest

LIVE_OPT_IN_ENV = "CIV6AZ_LIVE_GCS_TESTS"

if os.environ.get(LIVE_OPT_IN_ENV) != "1":
    pytest.skip(
        f"live GCS tests need an explicit opt-in: set {LIVE_OPT_IN_ENV}=1",
        allow_module_level=True,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all from environment, nothing hardcoded
# ---------------------------------------------------------------------------

_BUCKET_ENV_VAR = "CIV6AZ_GCS_BUCKET"
_PROJECT_ENV_VAR = "CIV6AZ_VERTEX_PROJECT"

_bucket: str | None = os.environ.get(_BUCKET_ENV_VAR)
_project: str | None = os.environ.get(_PROJECT_ENV_VAR)

# Unique namespace so parallel runs don't collide
_TEST_PREFIX = f"e2e-test-{uuid.uuid4().hex[:8]}"


def _skip_if_no_gcp() -> None:
    if not _bucket:
        pytest.skip(f"{_BUCKET_ENV_VAR} not set — skipping live GCS tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gcs_backend() -> Generator[Any, None, None]:
    """Create a live GCSBackend for the test session."""
    _skip_if_no_gcp()

    from alpha_zero.cloud.gcp.storage import GCSBackend

    backend = GCSBackend(project_id=_project)
    yield backend

    # Cleanup: delete all blobs created during tests
    try:
        blobs = backend.list_blobs(_bucket, _TEST_PREFIX)  # type: ignore[arg-type]
        for blob_name in blobs:
            try:
                backend.delete_blob(_bucket, blob_name)  # type: ignore[arg-type]
                logger.info("Cleaned up: %s", blob_name)
            except Exception:
                logger.warning("Failed to clean up: %s", blob_name)
    except Exception:
        logger.warning("Failed to list blobs for cleanup")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_env
class TestGCSLive:
    """Live GCS upload/download roundtrip tests."""

    def test_upload_download_roundtrip(self, gcs_backend: Any) -> None:
        """Upload bytes → download → verify content matches."""
        assert _bucket is not None
        blob_path = f"{_TEST_PREFIX}/roundtrip.bin"
        payload = b"alphazero-test-data-" + uuid.uuid4().bytes

        gcs_backend.upload_blob(_bucket, blob_path, payload)
        downloaded = gcs_backend.download_blob(_bucket, blob_path)

        assert downloaded == payload, "Downloaded bytes should match uploaded"
        logger.info("Roundtrip OK: %d bytes", len(payload))

    def test_blob_exists(self, gcs_backend: Any) -> None:
        """blob_exists returns True for existing blobs, False otherwise."""
        assert _bucket is not None
        blob_path = f"{_TEST_PREFIX}/exists-check.bin"

        assert not gcs_backend.blob_exists(_bucket, blob_path)
        gcs_backend.upload_blob(_bucket, blob_path, b"data")
        assert gcs_backend.blob_exists(_bucket, blob_path)

    def test_list_blobs(self, gcs_backend: Any) -> None:
        """list_blobs finds all blobs under a prefix."""
        assert _bucket is not None
        sub_prefix = f"{_TEST_PREFIX}/list-test"
        for i in range(3):
            gcs_backend.upload_blob(_bucket, f"{sub_prefix}/item-{i}.bin", f"data-{i}".encode())

        blobs = gcs_backend.list_blobs(_bucket, sub_prefix)
        assert len(blobs) >= 3

    def test_delete_blob(self, gcs_backend: Any) -> None:
        """delete_blob removes a blob and it no longer exists."""
        assert _bucket is not None
        blob_path = f"{_TEST_PREFIX}/to-delete.bin"

        gcs_backend.upload_blob(_bucket, blob_path, b"temp")
        assert gcs_backend.blob_exists(_bucket, blob_path)

        gcs_backend.delete_blob(_bucket, blob_path)
        assert not gcs_backend.blob_exists(_bucket, blob_path)

    def test_json_roundtrip(self, gcs_backend: Any) -> None:
        """Upload and download a JSON-serialized dict."""
        assert _bucket is not None
        blob_path = f"{_TEST_PREFIX}/metadata.json"
        metadata = {
            "version": 1,
            "iteration": 42,
            "timestamp": time.time(),
            "config": {"lr": 0.001, "batch_size": 256},
        }

        payload = json.dumps(metadata).encode("utf-8")
        gcs_backend.upload_blob(_bucket, blob_path, payload)

        downloaded = gcs_backend.download_blob(_bucket, blob_path)
        result = json.loads(downloaded.decode("utf-8"))

        assert result == metadata

    def test_large_payload(self, gcs_backend: Any) -> None:
        """Upload a moderately large payload (~1 MB)."""
        assert _bucket is not None
        blob_path = f"{_TEST_PREFIX}/large-payload.bin"
        one_mb = 1024 * 1024
        payload = os.urandom(one_mb)

        gcs_backend.upload_blob(_bucket, blob_path, payload)
        downloaded = gcs_backend.download_blob(_bucket, blob_path)

        assert len(downloaded) == one_mb
        assert downloaded == payload

    def test_model_storage_roundtrip(self, gcs_backend: Any) -> None:
        """Test GCSModelStorage save/load cycle with a real backend."""
        _skip_if_no_gcp()
        assert _bucket is not None

        try:
            from unittest.mock import MagicMock

            import torch

            from alpha_zero.cloud.gcp.config import GCPConfig, GCSPathConfig
            from alpha_zero.cloud.gcp.storage import GCSModelStorage
        except ImportError:
            pytest.skip("torch not available")

        # Build a GCPConfig pointing at the test bucket/prefix
        config = GCPConfig(
            project_id=_project or "",
            bucket=_bucket,
            paths=GCSPathConfig(models_prefix=f"{_TEST_PREFIX}/models"),
        )

        storage = GCSModelStorage(config=config, backend=gcs_backend)

        # Create a small state dict and wrap it in a ModelLike mock
        state_dict = {"weight": torch.randn(4, 4), "bias": torch.zeros(4)}
        mock_model = MagicMock()
        mock_model.state_dict.return_value = state_dict

        storage.save_model(
            model=mock_model,
            version=1,
            metadata={"iteration": 1, "elo": 1200.0},
        )

        loaded_state, metadata, _ = storage.load_model(version=1)
        assert metadata.iteration == 1
        assert metadata.elo == pytest.approx(1200.0)
        assert torch.equal(loaded_state["weight"], state_dict["weight"])
