"""GCS data fetcher wrapping existing storage classes.

Provides a thin wrapper around GCSModelStorage and GCSTrajectoryStorage
for use by the training run analysis pipeline. Does NOT re-implement
any storage logic — delegates entirely to the existing classes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from alpha_zero.analysis.data_model import ModelVersionData, RunData

logger = logging.getLogger(__name__)

# Graceful imports for GCP dependencies
try:
    from alpha_zero.cloud.gcp.config import GCPConfig
    from alpha_zero.cloud.gcp.storage import (
        GCSModelStorage,
        GCSTrajectoryStorage,
        ModelMetadata,
    )

    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False


class GCSRunFetcher:
    """Fetches model and trajectory data from GCS, wrapping existing storage classes."""

    def __init__(
        self,
        config: Any = None,  # GCPConfig | None
        backend: Any = None,  # StorageBackend | None
    ) -> None:
        """Initialize GCS fetcher.

        Args:
            config: GCP configuration. If None, creates default from env vars.
            backend: Storage backend for GCS operations. If None, uses default.

        Raises:
            ImportError: If GCP dependencies are not available.
        """
        if not GCS_AVAILABLE:
            raise ImportError(
                "GCP dependencies not available. Install with: pip install google-cloud-storage"
            )
        if config is None:
            config = GCPConfig()
        self._config = config
        self._backend = backend
        self._model_storage = GCSModelStorage(config, backend=backend)
        self._traj_storage = GCSTrajectoryStorage(config, backend=backend)

    def _load_metadata(self, version: int) -> ModelMetadata:
        """Load model metadata from GCS without downloading full model weights.

        Reads the metadata.json file directly from the storage backend,
        avoiding the heavyweight load_model() call which downloads the
        entire model checkpoint.

        Args:
            version: Model version number.

        Returns:
            ModelMetadata for the requested version.

        Raises:
            Exception: If metadata cannot be loaded (missing or corrupt).
        """
        meta_path = self._config.paths.metadata_path(version)
        meta_bytes = self._model_storage.backend.download_blob(
            self._config.bucket, meta_path
        )
        meta_dict = json.loads(meta_bytes.decode())
        return ModelMetadata.from_dict(meta_dict)

    def fetch_model_versions(
        self,
        version_range: tuple[int, int] | None = None,
    ) -> list[ModelVersionData]:
        """Fetch metadata for model versions. Optionally filter to a range (inclusive).

        Args:
            version_range: Optional (lo, hi) inclusive range to filter versions.

        Returns:
            Sorted list of ModelVersionData for each version.
        """
        all_versions = self._model_storage.list_versions()

        if version_range is not None:
            lo, hi = version_range
            all_versions = [v for v in all_versions if lo <= v <= hi]

        results = []
        for version in sorted(all_versions):
            try:
                metadata = self._load_metadata(version)
                results.append(
                    ModelVersionData(
                        version=version,
                        iteration=metadata.iteration,
                        elo=metadata.elo,
                        timestamp=metadata.timestamp,
                        games_played=metadata.games_played,
                        model_config=metadata.extra,
                    )
                )
            except Exception as e:
                # Legacy models (v0-v29) may have missing/incomplete metadata
                logger.warning("Could not load metadata for v%d: %s", version, e)
                results.append(
                    ModelVersionData(
                        version=version,
                        iteration=0,
                    )
                )

        return results

    def fetch_trajectory_stats(
        self,
        iterations: list[int] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Fetch trajectory statistics per iteration.

        Args:
            iterations: Specific iterations to fetch. If None, fetches all.

        Returns:
            Dict mapping iteration number to its stats dict.
        """
        stats = self._traj_storage.list_iteration_stats()

        result: dict[int, dict[str, Any]] = {}
        for stat in stats:
            iter_num = stat.get("iteration", -1)
            if iterations is not None and iter_num not in iterations:
                continue
            result[iter_num] = stat

        return result

    def augment_run_data(self, run_data: RunData) -> RunData:
        """Merge GCS data into existing RunData.

        Adds model metadata and trajectory stats. Does not overwrite
        existing non-None values on IterationData.

        Args:
            run_data: Existing RunData to augment.

        Returns:
            The same RunData instance, mutated with GCS data.
        """
        # Fetch models for the run's version range
        version_range = None
        if run_data.start_model_version is not None and run_data.end_model_version is not None:
            version_range = (run_data.start_model_version, run_data.end_model_version)

        run_data.models = self.fetch_model_versions(version_range)

        # Fetch trajectory stats for each iteration
        iter_nums = [it.iteration for it in run_data.iterations]
        traj_stats = self.fetch_trajectory_stats(iter_nums)

        for iteration in run_data.iterations:
            stat = traj_stats.get(iteration.iteration)
            if stat is not None:
                if iteration.trajectory_count is None:
                    iteration.trajectory_count = stat.get(
                        "blob_count", stat.get("num_trajectories")
                    )
                if iteration.games_completed is None:
                    iteration.games_completed = stat.get(
                        "num_games", stat.get("games")
                    )

        return run_data
