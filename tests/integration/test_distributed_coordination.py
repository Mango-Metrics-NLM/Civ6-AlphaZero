"""Integration tests for distributed training coordination.

Simulates multi-worker coordination, trajectory manifest merging,
and model version synchronization using mock backends.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_NUM_WORKERS = 4
_DEFAULT_TIMEOUT_SEC = 10.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage_backend() -> MagicMock:
    """Mock GCS storage backend for distributed tests."""
    backend = MagicMock()
    storage: dict[str, bytes] = {}

    def upload(bucket: str, path: str, data: bytes) -> None:
        storage[f"{bucket}/{path}"] = data

    def download(bucket: str, path: str) -> bytes:
        key = f"{bucket}/{path}"
        if key not in storage:
            raise FileNotFoundError(key)
        return storage[key]

    def exists(bucket: str, path: str) -> bool:
        return f"{bucket}/{path}" in storage

    def list_blobz(bucket: str, prefix: str) -> list[str]:
        full_prefix = f"{bucket}/{prefix}"
        return [k.split("/", 1)[1] for k in storage if k.startswith(full_prefix)]

    backend.upload_blob.side_effect = upload
    backend.download_blob.side_effect = download
    backend.blob_exists.side_effect = exists
    backend.list_blobs.side_effect = list_blobz
    backend._storage = storage  # Expose for assertions
    return backend


@pytest.fixture
def worker_configs() -> list[dict[str, Any]]:
    """Config dicts for a pool of simulated workers."""
    return [
        {
            "worker_id": i,
            "role": "selfplay" if i < 3 else "training",
            "gpu_type": "NVIDIA_TESLA_T4",
            "iteration": 0,
        }
        for i in range(_DEFAULT_NUM_WORKERS)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDistributedCoordination:
    """Multi-worker coordination protocol tests."""

    def test_worker_registration(
        self,
        mock_storage_backend: MagicMock,
        worker_configs: list[dict[str, Any]],
    ) -> None:
        """Workers register themselves via storage backend."""
        bucket = "test-bucket"
        for cfg in worker_configs:
            manifest = json.dumps(
                {
                    "worker_id": cfg["worker_id"],
                    "role": cfg["role"],
                    "status": "registered",
                    "timestamp": time.time(),
                }
            ).encode()
            mock_storage_backend.upload_blob(
                bucket,
                f"coordination/workers/{cfg['worker_id']}/manifest.json",
                manifest,
            )

        # Verify all registered
        blobs = mock_storage_backend.list_blobs(bucket, "coordination/workers/")
        assert len(blobs) == _DEFAULT_NUM_WORKERS

    def test_trajectory_manifest_merging(self, mock_storage_backend: MagicMock) -> None:
        """Multiple workers' trajectory shards merge into a single manifest."""
        bucket = "test-bucket"
        num_workers = 3
        examples_per_worker = 100

        # Each worker uploads a shard manifest
        shard_manifests: list[dict[str, Any]] = []
        for w in range(num_workers):
            shard = {
                "worker_id": w,
                "iteration": 1,
                "num_examples": examples_per_worker,
                "shard_path": f"trajectories/iter_1/shard_{w}.pt",
                "model_version": 1,
            }
            shard_manifests.append(shard)
            mock_storage_backend.upload_blob(
                bucket,
                f"trajectories/iter_1/manifest_{w}.json",
                json.dumps(shard).encode(),
            )

        # Merge manifests
        merged = {
            "iteration": 1,
            "total_examples": sum(s["num_examples"] for s in shard_manifests),
            "num_shards": len(shard_manifests),
            "shards": shard_manifests,
        }

        assert merged["total_examples"] == num_workers * examples_per_worker
        assert merged["num_shards"] == num_workers

    def test_model_version_synchronization(self, mock_storage_backend: MagicMock) -> None:
        """Workers converge on the latest model version."""
        bucket = "test-bucket"

        # Upload model version metadata
        for version in range(1, 4):
            metadata = json.dumps(
                {
                    "version": version,
                    "iteration": version * 10,
                    "elo": 1000 + version * 50,
                }
            ).encode()
            mock_storage_backend.upload_blob(
                bucket,
                f"models/v{version}/metadata.json",
                metadata,
            )

        # Workers should find the latest version
        blobs = mock_storage_backend.list_blobs(bucket, "models/")
        versions = []
        for blob in blobs:
            parts = blob.split("/")
            for part in parts:
                if part.startswith("v") and part[1:].isdigit():
                    versions.append(int(part[1:]))
        latest = max(versions)

        assert latest == 3
        logger.info("Latest model version: %d", latest)

    def test_concurrent_uploads(self, mock_storage_backend: MagicMock) -> None:
        """Multiple threads can upload simultaneously without data loss."""
        bucket = "test-bucket"
        num_threads = _DEFAULT_NUM_WORKERS
        results: list[bool] = []

        def worker_upload(worker_id: int) -> None:
            try:
                for i in range(10):
                    mock_storage_backend.upload_blob(
                        bucket,
                        f"concurrent/{worker_id}/item_{i}.bin",
                        f"data-{worker_id}-{i}".encode(),
                    )
                results.append(True)
            except Exception:
                results.append(False)

        threads = [threading.Thread(target=worker_upload, args=(w,)) for w in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=_DEFAULT_TIMEOUT_SEC)

        assert all(results), "All concurrent uploads should succeed"
        assert len(results) == num_threads

    def test_heartbeat_protocol(self, mock_storage_backend: MagicMock) -> None:
        """Workers send heartbeats and stale workers are detectable."""
        bucket = "test-bucket"
        now = time.time()
        stale_threshold_sec = 60.0

        heartbeats = [
            {"worker_id": 0, "timestamp": now, "status": "active"},  # Fresh
            {"worker_id": 1, "timestamp": now - 30, "status": "active"},  # OK
            {"worker_id": 2, "timestamp": now - 120, "status": "active"},  # Stale
            {"worker_id": 3, "timestamp": now - 300, "status": "active"},  # Very stale
        ]

        for hb in heartbeats:
            mock_storage_backend.upload_blob(
                bucket,
                f"heartbeats/worker_{hb['worker_id']}.json",
                json.dumps(hb).encode(),
            )

        # Detect stale workers
        stale_workers = [
            hb["worker_id"]
            for hb in heartbeats
            if now - cast(float, hb["timestamp"]) > stale_threshold_sec
        ]

        assert stale_workers == [2, 3]
        logger.info("Stale workers detected: %s", stale_workers)
