"""Tests for GCS data fetcher wrapping existing storage classes."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.analysis.data_model import (
    IterationData,
    ModelVersionData,
    RunData,
    RunIdentifier,
)
from alpha_zero.cloud.gcp.storage import ModelMetadata


class TestGCSRunFetcher:
    """Tests for GCSRunFetcher."""

    def _make_fetcher(self):
        """Create a GCSRunFetcher with mocked storage classes."""
        from alpha_zero.analysis.fetchers.gcs_fetcher import GCSRunFetcher

        config = MagicMock()
        config.project_id = "test-project"
        config.bucket = "test-bucket"
        config.paths = MagicMock()

        fetcher = GCSRunFetcher(config=config, backend=MagicMock())
        return fetcher

    def test_fetch_model_versions(self):
        """Mock list_versions returning [0, 1, 2] and load metadata for each."""
        fetcher = self._make_fetcher()

        # Mock list_versions
        fetcher._model_storage.list_versions = MagicMock(return_value=[0, 1, 2])

        # Mock _load_metadata to return ModelMetadata objects
        def mock_load_metadata(version):
            return ModelMetadata(
                version=version,
                iteration=version * 5,
                elo=1500.0 + version * 10,
                timestamp="2026-03-01T00:00:00",
                games_played=100 * (version + 1),
                extra={"source": "test"},
            )

        fetcher._load_metadata = mock_load_metadata

        results = fetcher.fetch_model_versions()

        assert len(results) == 3
        assert all(isinstance(r, ModelVersionData) for r in results)
        assert results[0].version == 0
        assert results[0].iteration == 0
        assert results[0].elo == 1500.0
        assert results[0].games_played == 100
        assert results[1].version == 1
        assert results[1].iteration == 5
        assert results[2].version == 2

    def test_version_range_filtering(self):
        """Versions [0, 1, 2, 3, 4], range (1, 3) -> only 1, 2, 3 returned."""
        fetcher = self._make_fetcher()

        fetcher._model_storage.list_versions = MagicMock(return_value=[0, 1, 2, 3, 4])
        fetcher._load_metadata = MagicMock(
            side_effect=lambda v: ModelMetadata(version=v, iteration=v)
        )

        results = fetcher.fetch_model_versions(version_range=(1, 3))

        assert len(results) == 3
        assert [r.version for r in results] == [1, 2, 3]

    def test_legacy_models_no_metadata(self):
        """load_metadata raises Exception for v0 -> graceful fallback."""
        fetcher = self._make_fetcher()

        fetcher._model_storage.list_versions = MagicMock(return_value=[0, 1])

        def mock_load_metadata(version):
            if version == 0:
                raise Exception("No metadata for legacy model v0")
            return ModelMetadata(version=version, iteration=version * 5)

        fetcher._load_metadata = mock_load_metadata

        results = fetcher.fetch_model_versions()

        assert len(results) == 2
        # v0 should have fallback values
        assert results[0].version == 0
        assert results[0].iteration == 0
        assert results[0].elo is None
        # v1 should have real metadata
        assert results[1].version == 1
        assert results[1].iteration == 5

    def test_fetch_trajectory_stats(self):
        """Mock list_iteration_stats returning stats dicts."""
        fetcher = self._make_fetcher()

        fetcher._traj_storage.list_iteration_stats = MagicMock(
            return_value=[
                {"iteration": 0, "blob_count": 4, "blob_paths": ["a", "b", "c", "d"]},
                {"iteration": 1, "blob_count": 3, "blob_paths": ["e", "f", "g"]},
                {"iteration": 2, "blob_count": 5, "blob_paths": ["h", "i", "j", "k", "l"]},
            ]
        )

        result = fetcher.fetch_trajectory_stats()

        assert len(result) == 3
        assert result[0]["blob_count"] == 4
        assert result[1]["iteration"] == 1
        assert result[2]["blob_count"] == 5

    def test_fetch_trajectory_stats_filtered(self):
        """Filter trajectory stats to specific iterations."""
        fetcher = self._make_fetcher()

        fetcher._traj_storage.list_iteration_stats = MagicMock(
            return_value=[
                {"iteration": 0, "blob_count": 4, "blob_paths": []},
                {"iteration": 1, "blob_count": 3, "blob_paths": []},
                {"iteration": 2, "blob_count": 5, "blob_paths": []},
            ]
        )

        result = fetcher.fetch_trajectory_stats(iterations=[0, 2])

        assert len(result) == 2
        assert result[0]["iteration"] == 0
        assert result[2]["iteration"] == 2
        assert 1 not in result

    def test_augment_run_data(self):
        """Create RunData with iterations, augment with GCS data."""
        fetcher = self._make_fetcher()

        # Mock model fetching
        fetcher._model_storage.list_versions = MagicMock(return_value=[10, 11, 12])
        fetcher._load_metadata = MagicMock(
            side_effect=lambda v: ModelMetadata(version=v, iteration=v - 10)
        )

        # Mock trajectory stats
        fetcher._traj_storage.list_iteration_stats = MagicMock(
            return_value=[
                {"iteration": 0, "blob_count": 4, "blob_paths": []},
                {"iteration": 1, "blob_count": 3, "blob_paths": []},
                {"iteration": 2, "blob_count": 5, "blob_paths": []},
            ]
        )

        run_data = RunData(
            identifier=RunIdentifier(wandb_run_id="test-run"),
            iterations=[
                IterationData(iteration=0),
                IterationData(iteration=1),
                IterationData(iteration=2),
            ],
            start_model_version=10,
            end_model_version=12,
        )

        result = fetcher.augment_run_data(run_data)

        # Models should be populated
        assert len(result.models) == 3
        assert result.models[0].version == 10
        assert result.models[2].version == 12

        # Trajectory stats should be merged
        for it in result.iterations:
            assert it.trajectory_count is not None

    def test_augment_run_data_preserves_existing_counts(self):
        """augment_run_data should NOT overwrite existing trajectory/game counts."""
        fetcher = self._make_fetcher()

        fetcher._model_storage.list_versions = MagicMock(return_value=[])
        fetcher._load_metadata = MagicMock(side_effect=Exception("no models"))
        fetcher._traj_storage.list_iteration_stats = MagicMock(
            return_value=[
                {"iteration": 0, "blob_count": 99, "blob_paths": []},
            ]
        )

        run_data = RunData(
            identifier=RunIdentifier(wandb_run_id="test-run"),
            iterations=[
                IterationData(iteration=0, trajectory_count=42, games_completed=20),
            ],
        )

        result = fetcher.augment_run_data(run_data)

        # Existing values should be preserved
        assert result.iterations[0].trajectory_count == 42
        assert result.iterations[0].games_completed == 20

    def test_gcs_unavailable_raises(self):
        """Patch GCS_AVAILABLE=False, verify ImportError."""
        with patch("alpha_zero.analysis.fetchers.gcs_fetcher.GCS_AVAILABLE", False):
            from alpha_zero.analysis.fetchers.gcs_fetcher import GCSRunFetcher

            with pytest.raises(ImportError, match="GCP dependencies"):
                GCSRunFetcher()

    def test_load_metadata_from_backend(self):
        """Test that _load_metadata reads JSON from the storage backend."""
        from alpha_zero.analysis.fetchers.gcs_fetcher import GCSRunFetcher

        config = MagicMock()
        config.project_id = "test-project"
        config.bucket = "test-bucket"
        config.paths = MagicMock()
        config.paths.metadata_path.return_value = "models/civ6az/v005/metadata.json"

        backend = MagicMock()
        meta_dict = {
            "version": 5,
            "iteration": 10,
            "elo": 1520.5,
            "timestamp": "2026-03-01T00:00:00",
            "games_played": 200,
            "extra": {"promoted": True},
        }
        backend.download_blob.return_value = json.dumps(meta_dict).encode()

        fetcher = GCSRunFetcher(config=config, backend=backend)
        metadata = fetcher._load_metadata(5)

        assert metadata.version == 5
        assert metadata.iteration == 10
        assert metadata.elo == 1520.5
        assert metadata.games_played == 200
        backend.download_blob.assert_called_once_with(
            "test-bucket", "models/civ6az/v005/metadata.json"
        )
