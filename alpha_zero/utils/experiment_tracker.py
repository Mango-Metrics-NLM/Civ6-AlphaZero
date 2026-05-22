"""Canonical experiment tracker import surface.

All worker code imports tracking primitives from this module. Today
they route to MLflow via :class:`WandbCompatAdapter`; swapping backends
only requires editing this file.
"""
from __future__ import annotations

import logging
from typing import cast

from alpha_zero.monitoring.config import MonitoringConfig
from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter

logger = logging.getLogger(__name__)

_TRACKER: WandbCompatAdapter | None = None


def get_tracker() -> WandbCompatAdapter:
    """Return the process-wide tracker singleton (creates one if needed)."""
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = cast(
            WandbCompatAdapter,
            WandbCompatAdapter.from_config(MonitoringConfig.from_env()),
        )
    return _TRACKER


def init_tracker(config: MonitoringConfig | None = None) -> WandbCompatAdapter:
    """Initialise (or re-initialise) the singleton tracker."""
    global _TRACKER
    cfg = config or MonitoringConfig.from_env()
    _TRACKER = cast(WandbCompatAdapter, WandbCompatAdapter.from_config(cfg))
    _TRACKER.init()
    return _TRACKER


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Convenience: log metrics on the singleton."""
    get_tracker().log_metrics(metrics, step=step)


def reset_tracker_for_tests() -> None:
    """Clear the singleton (test helper only)."""
    global _TRACKER
    _TRACKER = None
