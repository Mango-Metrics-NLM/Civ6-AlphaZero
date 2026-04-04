"""SpacetimeDB integration for Civ6 AlphaZero.

Provides real-time state synchronization, storage backend, and metrics
streaming using SpacetimeDB's table subscriptions and reducer pattern.

The SpacetimeDB Python SDK (spacetimedb-sdk) is an optional dependency.
All functionality gracefully degrades when the SDK is not installed.

Usage:
    from alpha_zero.spacetimedb import SpacetimeDBConfig, HAS_SPACETIMEDB

    if HAS_SPACETIMEDB:
        from alpha_zero.spacetimedb import SpacetimeDBClient
        client = SpacetimeDBClient(SpacetimeDBConfig.for_local("my-module"))
"""

from __future__ import annotations

from typing import Any

# Optional SDK import — gracefully handle missing dependency
try:
    from spacetimedb_sdk.spacetimedb_async_client import (
        SpacetimeDBAsyncClient as _SpacetimeDBAsyncClient,
    )

    HAS_SPACETIMEDB = True
except ImportError:
    HAS_SPACETIMEDB = False
    _SpacetimeDBAsyncClient = None

# Always-available imports
from alpha_zero.spacetimedb.config import SpacetimeDBConfig
from alpha_zero.spacetimedb.exceptions import (
    SpacetimeDBConnectionError,
    SpacetimeDBError,
    SpacetimeDBReducerError,
    SpacetimeDBSerializationError,
    SpacetimeDBSubscriptionError,
    SpacetimeDBTimeoutError,
    SpacetimeDBUnavailableError,
)

__all__ = [
    "HAS_SPACETIMEDB",
    "SpacetimeDBConfig",
    "SpacetimeDBConnectionError",
    "SpacetimeDBError",
    "SpacetimeDBReducerError",
    "SpacetimeDBSerializationError",
    "SpacetimeDBSubscriptionError",
    "SpacetimeDBTimeoutError",
    "SpacetimeDBUnavailableError",
]


def __getattr__(name: str) -> Any:
    """Lazy import for classes that depend on other submodules.

    SpacetimeDBClient, WeightSynchronizer, and MetricsSink are always
    importable — they work with mock clients even without the SDK.
    The SDK is only required when establishing a real connection.
    """
    if name == "SpacetimeDBClient":
        from alpha_zero.spacetimedb.client import SpacetimeDBClient

        return SpacetimeDBClient

    if name == "SpacetimeDBWeightSynchronizer":
        from alpha_zero.spacetimedb.weight_sync import SpacetimeDBWeightSynchronizer

        return SpacetimeDBWeightSynchronizer

    if name == "SpacetimeDBMetricsSink":
        from alpha_zero.spacetimedb.metrics_sink import SpacetimeDBMetricsSink

        return SpacetimeDBMetricsSink

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
