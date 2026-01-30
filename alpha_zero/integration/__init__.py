"""Game integration layer for Civ6-AlphaZero.

This module provides robust connection management for game environments
with error recovery, retry logic, and latency compensation.
"""

from __future__ import annotations

from .game_connector import (
    CheckpointManager,
    ConnectionConfig,
    ConnectionMetrics,
    ConnectionStatus,
    ConnectionTimeoutError,
    ErrorSeverity,
    GameBackend,
    GameConnectionError,
    GameConnector,
    GameType,
    LatencyConfig,
    LatencyTracker,
    MicroCivBackend,
    RecoveryConfig,
    RetryConfig,
    TimeoutConfig,
    UncivBackend,
    create_game_connector,
    create_microciv_connector,
    create_unciv_connector,
)

__all__ = [
    # Main connector
    "GameConnector",
    "GameType",
    "ConnectionStatus",
    # Configuration
    "ConnectionConfig",
    "RetryConfig",
    "TimeoutConfig",
    "LatencyConfig",
    "RecoveryConfig",
    # Backends
    "GameBackend",
    "MicroCivBackend",
    "UncivBackend",
    # Monitoring
    "ConnectionMetrics",
    "LatencyTracker",
    "CheckpointManager",
    # Errors
    "GameConnectionError",
    "ConnectionTimeoutError",
    "ErrorSeverity",
    # Factory functions
    "create_game_connector",
    "create_microciv_connector",
    "create_unciv_connector",
]
