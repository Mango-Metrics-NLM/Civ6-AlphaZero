"""Deprecated: Use alpha_zero.utils.logging_config instead.

This module re-exports all symbols from logging_config for backwards compatibility.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "alpha_zero.utils.logging module is deprecated. Use alpha_zero.utils.logging_config instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical module
from alpha_zero.utils.logging_config import (  # noqa: E402
    CircularBuffer,
    ColoredFormatter,
    GameLogger,
    LogConfig,
    LogLevel,
    MetricsTracker,
    PerformanceProfiler,
    StructuredFormatter,
    TrainingLogger,
    TrainingMetricsAdapter,
    configure_from_yaml,
    get_logger,
    get_profiler,
    get_training_logger,
    set_level,
    setup_logging,
    timed_section,
    timer,
)

__all__ = [
    "CircularBuffer",
    "ColoredFormatter",
    "GameLogger",
    "LogConfig",
    "LogLevel",
    "MetricsTracker",
    "PerformanceProfiler",
    "StructuredFormatter",
    "TrainingLogger",
    "TrainingMetricsAdapter",
    "configure_from_yaml",
    "get_logger",
    "get_profiler",
    "get_training_logger",
    "set_level",
    "setup_logging",
    "timed_section",
    "timer",
]
