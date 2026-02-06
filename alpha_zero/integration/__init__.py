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
    "CheckpointManager",
    # Configuration
    "ConnectionConfig",
    # Monitoring
    "ConnectionMetrics",
    "ConnectionStatus",
    "ConnectionTimeoutError",
    "ErrorSeverity",
    # Backends
    "GameBackend",
    # Errors
    "GameConnectionError",
    # Main connector
    "GameConnector",
    "GameType",
    "LatencyConfig",
    "LatencyTracker",
    "MicroCivBackend",
    "RecoveryConfig",
    "RetryConfig",
    "TimeoutConfig",
    "UncivBackend",
    # Factory functions
    "create_game_connector",
    "create_microciv_connector",
    "create_unciv_connector",
]
