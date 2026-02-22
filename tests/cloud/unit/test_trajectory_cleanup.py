"""Unit tests for GCSTrajectoryStorage.delete_iterations / list_iteration_stats
and scripts/cleanup_trajectories.py CLI helpers."""

from __future__ import annotations

import argparse

import pytest

from alpha_zero.cloud.gcp.config import GCPConfig, GCSPathConfig
from alpha_zero.cloud.gcp.storage import GCSTrajectoryStorage
from tests.mocks.cloud_mocks import MockStorageBackend


class FailingDeleteBackend(MockStorageBackend):
    """Backend that raises on delete for a specific blob."""

    def __init__(self, fail_blob: str) -> None:
        super().__init__()
        self._fail_blob = fail_blob

    def delete_blob(self, bucket: str, blob_path: str) -> None:
        if blob_path == self._fail_blob:
            raise RuntimeError(f"Simulated deletion failure: {blob_path}")
        super().delete_blob(bucket, blob_path)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_backend() -> MockStorageBackend:
    return MockStorageBackend()


@pytest.fixture
def gcp_config() -> GCPConfig:
    return GCPConfig(
        project_id="test-project",
        region="us-central1",
        bucket="test-bucket",
        paths=GCSPathConfig(),
    )


@pytest.fixture
def populated_storage(
    mock_backend: MockStorageBackend, gcp_config: GCPConfig
) -> GCSTrajectoryStorage:
    """Storage with sample blobs in iter_028-030."""
    storage = GCSTrajectoryStorage(gcp_config, backend=mock_backend, compress=False)
    prefix = gcp_config.paths.trajectories_prefix

    # iter_028: 2 shards + manifest
    mock_backend.upload_blob("test-bucket", f"{prefix}/iter_028/shard_worker_0.pt", b"data0")
    mock_backend.upload_blob("test-bucket", f"{prefix}/iter_028/shard_worker_1.pt", b"data1")
    mock_backend.upload_blob("test-bucket", f"{prefix}/iter_028/manifest.json", b"{}")

    # iter_029: 1 shard
    mock_backend.upload_blob("test-bucket", f"{prefix}/iter_029/shard_worker_0.pt", b"data0")

    # iter_030: empty (no blobs uploaded)

    return storage


# =============================================================================
# list_iteration_stats tests
# =============================================================================


class TestListIterationStats:
    """Tests for GCSTrajectoryStorage.list_iteration_stats."""

    def test_with_data(self, populated_storage: GCSTrajectoryStorage) -> None:
        stats = populated_storage.list_iteration_stats([28, 29])
        assert len(stats) == 2

        iter28 = stats[0]
        assert iter28["iteration"] == 28
        assert iter28["blob_count"] == 3  # 2 shards + manifest

        iter29 = stats[1]
        assert iter29["iteration"] == 29
        assert iter29["blob_count"] == 1

    def test_empty_iteration(self, populated_storage: GCSTrajectoryStorage) -> None:
        stats = populated_storage.list_iteration_stats([30])
        assert len(stats) == 1
        assert stats[0]["blob_count"] == 0
        assert stats[0]["blob_paths"] == []

    def test_auto_discovers_all(self, populated_storage: GCSTrajectoryStorage) -> None:
        """None → all available iterations."""
        stats = populated_storage.list_iteration_stats(None)
        iterations = [s["iteration"] for s in stats]
        assert 28 in iterations
        assert 29 in iterations


# =============================================================================
# delete_iterations tests
# =============================================================================


class TestDeleteIterations:
    """Tests for GCSTrajectoryStorage.delete_iterations."""

    def test_dry_run_no_deletions(
        self,
        populated_storage: GCSTrajectoryStorage,
        mock_backend: MockStorageBackend,
    ) -> None:
        """Dry run should not actually delete anything."""
        result = populated_storage.delete_iterations([28, 29], dry_run=True)

        assert result["dry_run"] is True
        assert result["deleted_count"] == 4  # 3 (iter28) + 1 (iter29)
        assert len(result["deleted_blobs"]) == 4

        # Verify nothing was actually removed
        assert mock_backend.blob_exists(
            "test-bucket", "trajectories/civ6az/iter_028/shard_worker_0.pt"
        )
        assert mock_backend.blob_exists(
            "test-bucket", "trajectories/civ6az/iter_029/shard_worker_0.pt"
        )

    def test_real_delete(
        self,
        populated_storage: GCSTrajectoryStorage,
        mock_backend: MockStorageBackend,
    ) -> None:
        """Real delete removes all blobs."""
        result = populated_storage.delete_iterations([28], dry_run=False)

        assert result["deleted_count"] == 3
        assert result["dry_run"] is False
        assert len(result["errors"]) == 0

        # Blobs should be gone
        assert not mock_backend.blob_exists(
            "test-bucket", "trajectories/civ6az/iter_028/shard_worker_0.pt"
        )
        assert not mock_backend.blob_exists(
            "test-bucket", "trajectories/civ6az/iter_028/shard_worker_1.pt"
        )
        assert not mock_backend.blob_exists(
            "test-bucket", "trajectories/civ6az/iter_028/manifest.json"
        )

        # iter_029 should be untouched
        assert mock_backend.blob_exists(
            "test-bucket", "trajectories/civ6az/iter_029/shard_worker_0.pt"
        )

    def test_empty_iteration_skipped(self, populated_storage: GCSTrajectoryStorage) -> None:
        """Iterations with no blobs land in skipped_iterations."""
        result = populated_storage.delete_iterations([30, 31], dry_run=False)
        assert result["deleted_count"] == 0
        assert set(result["skipped_iterations"]) == {30, 31}

    def test_partial_error(self, gcp_config: GCPConfig) -> None:
        """Errors on individual blobs are collected, others proceed."""
        fail_blob = "trajectories/civ6az/iter_028/shard_worker_0.pt"
        backend = FailingDeleteBackend(fail_blob)
        prefix = gcp_config.paths.trajectories_prefix

        backend.upload_blob("test-bucket", f"{prefix}/iter_028/shard_worker_0.pt", b"x")
        backend.upload_blob("test-bucket", f"{prefix}/iter_028/shard_worker_1.pt", b"y")

        storage = GCSTrajectoryStorage(gcp_config, backend=backend, compress=False)
        result = storage.delete_iterations([28], dry_run=False)

        assert result["deleted_count"] == 1  # worker_1 succeeded
        assert len(result["errors"]) == 1  # worker_0 failed
        assert "shard_worker_0.pt" in result["errors"][0]

    def test_respects_agent_id(self, mock_backend: MockStorageBackend) -> None:
        """Agent ID scopes the trajectory prefix."""
        config = GCPConfig(
            project_id="test-project",
            region="us-central1",
            bucket="test-bucket",
            paths=GCSPathConfig(agent_id="ab_patch_embed"),
        )
        storage = GCSTrajectoryStorage(config, backend=mock_backend, compress=False)
        # Resolved prefix includes agent_id
        resolved_prefix = config.paths._resolve_prefix(config.paths.trajectories_prefix)

        # Upload under agent-scoped prefix (the resolved path)
        mock_backend.upload_blob(
            "test-bucket", f"{resolved_prefix}/iter_001/shard_worker_0.pt", b"data"
        )
        # Upload under default (unscoped) prefix — should NOT be touched
        mock_backend.upload_blob(
            "test-bucket", "trajectories/civ6az/iter_001/shard_worker_0.pt", b"data"
        )

        result = storage.delete_iterations([1], dry_run=False)
        assert result["deleted_count"] == 1

        # Default prefix should be untouched
        assert mock_backend.blob_exists(
            "test-bucket", "trajectories/civ6az/iter_001/shard_worker_0.pt"
        )


# =============================================================================
# parse_iteration_range tests
# =============================================================================


class TestParseIterationRange:
    """Tests for the CLI iteration parser."""

    def test_dash_range(self) -> None:
        from scripts.cleanup_trajectories import parse_iteration_range

        result = parse_iteration_range("28-33")
        assert result == [28, 29, 30, 31, 32, 33]

    def test_comma_list(self) -> None:
        from scripts.cleanup_trajectories import parse_iteration_range

        result = parse_iteration_range("28,30,33")
        assert result == [28, 30, 33]

    def test_mixed(self) -> None:
        from scripts.cleanup_trajectories import parse_iteration_range

        result = parse_iteration_range("1-3,10,20-22")
        assert result == [1, 2, 3, 10, 20, 21, 22]

    def test_deduplicates(self) -> None:
        from scripts.cleanup_trajectories import parse_iteration_range

        result = parse_iteration_range("1-3,2-4")
        assert result == [1, 2, 3, 4]

    def test_invalid_raises(self) -> None:
        from scripts.cleanup_trajectories import parse_iteration_range

        with pytest.raises(argparse.ArgumentTypeError):
            parse_iteration_range("abc")

    def test_reversed_range_raises(self) -> None:
        from scripts.cleanup_trajectories import parse_iteration_range

        with pytest.raises(argparse.ArgumentTypeError):
            parse_iteration_range("10-5")
