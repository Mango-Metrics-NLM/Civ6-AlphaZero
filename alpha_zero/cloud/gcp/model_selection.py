"""Shared model version selection utilities for GCP cloud pipeline.

Provides reusable functions for filtering and subsampling model versions
used by both the orchestrator and standalone tournament worker.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def subsample_model_versions(
    versions: list[int],
    stride: int | None = 1,
    version_range: str | None = None,
) -> list[int]:
    """Apply stride and range filters to model versions.

    Filters a list of model versions by an optional range (e.g. ``"5:20"``)
    and then subsamples every *stride*-th entry.  The **first and last**
    versions are always preserved in the subsampled result.

    Args:
        versions: Unsorted or sorted list of model version numbers.
            Duplicates are removed before filtering.
        stride: Take every Nth version.  ``None`` or ``1`` means no
            subsampling.
        version_range: Optional ``"start:end"`` string (inclusive on both
            sides).  Versions outside this range are discarded *before*
            stride is applied.

    Returns:
        Sorted, deduplicated list of selected model versions.
        Returns ``[]`` when the input is empty or the range excludes all
        versions.

    Examples:
        >>> subsample_model_versions([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], stride=3)
        [0, 3, 6, 9]
        >>> subsample_model_versions(list(range(10)), stride=2, version_range="2:8")
        [2, 4, 6, 8]
    """
    versions = sorted(set(versions))

    # --- range filter --------------------------------------------------------
    if version_range:
        parts = version_range.split(":")
        if len(parts) == 2:
            try:
                range_start = int(parts[0])
                range_end = int(parts[1])
            except ValueError:
                logger.warning(
                    "Invalid version_range format %r — expected 'start:end' with integers",
                    version_range,
                )
            else:
                versions = [v for v in versions if range_start <= v <= range_end]

    if not versions:
        return []

    # --- stride --------------------------------------------------------------
    effective_stride = stride or 1
    if effective_stride > 1:
        first, last = versions[0], versions[-1]
        sampled = [v for i, v in enumerate(versions) if i % effective_stride == 0]
        # Always preserve first and last
        if first not in sampled:
            sampled.insert(0, first)
        if last not in sampled:
            sampled.append(last)
        return sorted(set(sampled))

    return versions
