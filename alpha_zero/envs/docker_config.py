"""Docker connection configuration for CivRealm environments.

This module provides configuration for Docker-based CivRealm connections,
including retry logic, health checking, and timeout management.

All configuration values can be overridden via environment variables:
- CIVREALM_HOST: Server hostname (default: localhost)
- CIVREALM_PORT: Server port (default: 6300)
- CIVREALM_CLIENT_PORT: Client port (default: 6001)
- CIVREALM_TIMEOUT_S: Connection timeout in seconds (default: 30.0)
- CIVREALM_RETRY_COUNT: Number of connection retries (default: 3)
- CIVREALM_RETRY_DELAY_S: Delay between retries in seconds (default: 2.0)
- CIVREALM_HEALTHCHECK_INTERVAL_S: Health check interval (default: 5.0)

Example usage:
    >>> from alpha_zero.envs.docker_config import DockerConnectionConfig
    >>> config = DockerConnectionConfig()  # Uses environment variables or defaults
    >>> config = DockerConnectionConfig(host="192.168.1.100", port=6301)  # Override
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Status of Docker connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


def _get_env_int(name: str, default: int) -> int:
    """Get integer from environment variable with default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s: %r, using default %d", name, value, default)
        return default


def _get_env_float(name: str, default: float) -> float:
    """Get float from environment variable with default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s: %r, using default %f", name, value, default)
        return default


def _get_env_str(name: str, default: str) -> str:
    """Get string from environment variable with default."""
    return os.getenv(name, default)


@dataclass
class DockerConnectionConfig:
    """Configuration for Docker-based CivRealm connection.

    Supports environment variable overrides for all settings, enabling
    flexible deployment without code changes.

    Attributes:
        host: CivRealm server hostname.
        port: CivRealm server port.
        client_port: Client connection port.
        connection_timeout_s: Timeout for initial connection in seconds.
        retry_count: Number of connection retries on failure.
        retry_delay_s: Base delay between retries (exponential backoff applied).
        healthcheck_interval_s: Interval between health checks in seconds.
        max_backoff_s: Maximum backoff delay for exponential backoff.
        enable_healthcheck: Whether to enable periodic health checks.
    """

    # Connection endpoints - use factory defaults to read env vars at instantiation
    host: str = field(default_factory=lambda: _get_env_str("CIVREALM_HOST", "localhost"))
    port: int = field(default_factory=lambda: _get_env_int("CIVREALM_PORT", 6300))
    client_port: int = field(default_factory=lambda: _get_env_int("CIVREALM_CLIENT_PORT", 6001))

    # Timeout and retry settings
    connection_timeout_s: float = field(
        default_factory=lambda: _get_env_float("CIVREALM_TIMEOUT_S", 30.0)
    )
    retry_count: int = field(default_factory=lambda: _get_env_int("CIVREALM_RETRY_COUNT", 3))
    retry_delay_s: float = field(
        default_factory=lambda: _get_env_float("CIVREALM_RETRY_DELAY_S", 2.0)
    )

    # Health check settings
    healthcheck_interval_s: float = field(
        default_factory=lambda: _get_env_float("CIVREALM_HEALTHCHECK_INTERVAL_S", 5.0)
    )
    max_backoff_s: float = field(
        default_factory=lambda: _get_env_float("CIVREALM_MAX_BACKOFF_S", 60.0)
    )
    enable_healthcheck: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}")
        if self.client_port < 1 or self.client_port > 65535:
            raise ValueError(f"Invalid client_port: {self.client_port}")
        if self.connection_timeout_s <= 0:
            raise ValueError(f"Invalid connection_timeout_s: {self.connection_timeout_s}")
        if self.retry_count < 0:
            raise ValueError(f"Invalid retry_count: {self.retry_count}")
        if self.retry_delay_s < 0:
            raise ValueError(f"Invalid retry_delay_s: {self.retry_delay_s}")

    @property
    def server_address(self) -> str:
        """Get formatted server address."""
        return f"{self.host}:{self.port}"

    def get_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff.

        Args:
            attempt: Current retry attempt (0-indexed).

        Returns:
            Delay in seconds for this attempt.
        """
        delay = self.retry_delay_s * (2**attempt)
        return min(delay, self.max_backoff_s)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "host": self.host,
            "port": self.port,
            "client_port": self.client_port,
            "connection_timeout_s": self.connection_timeout_s,
            "retry_count": self.retry_count,
            "retry_delay_s": self.retry_delay_s,
            "healthcheck_interval_s": self.healthcheck_interval_s,
            "max_backoff_s": self.max_backoff_s,
            "enable_healthcheck": self.enable_healthcheck,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DockerConnectionConfig:
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            New DockerConnectionConfig instance.
        """
        return cls(
            host=data.get("host", "localhost"),
            port=data.get("port", 6300),
            client_port=data.get("client_port", 6001),
            connection_timeout_s=data.get("connection_timeout_s", 30.0),
            retry_count=data.get("retry_count", 3),
            retry_delay_s=data.get("retry_delay_s", 2.0),
            healthcheck_interval_s=data.get("healthcheck_interval_s", 5.0),
            max_backoff_s=data.get("max_backoff_s", 60.0),
            enable_healthcheck=data.get("enable_healthcheck", True),
        )


@dataclass
class ConnectionState:
    """Tracks the current state of a Docker connection.

    Attributes:
        status: Current connection status.
        last_connected_at: Timestamp of last successful connection.
        last_error: Last error message if any.
        retry_attempts: Number of retry attempts made.
        consecutive_failures: Number of consecutive connection failures.
    """

    status: ConnectionStatus = ConnectionStatus.DISCONNECTED
    last_connected_at: float | None = None
    last_error: str | None = None
    retry_attempts: int = 0
    consecutive_failures: int = 0

    def reset(self) -> None:
        """Reset connection state."""
        self.status = ConnectionStatus.DISCONNECTED
        self.last_error = None
        self.retry_attempts = 0
        self.consecutive_failures = 0

    def on_connected(self, timestamp: float) -> None:
        """Update state on successful connection.

        Args:
            timestamp: Time of connection.
        """
        self.status = ConnectionStatus.CONNECTED
        self.last_connected_at = timestamp
        self.last_error = None
        self.retry_attempts = 0
        self.consecutive_failures = 0

    def on_disconnected(self, error: str | None = None) -> None:
        """Update state on disconnection.

        Args:
            error: Error message if disconnection was due to error.
        """
        self.status = ConnectionStatus.DISCONNECTED
        if error:
            self.last_error = error
            self.consecutive_failures += 1

    def on_retry(self) -> None:
        """Update state when starting a retry attempt."""
        self.status = ConnectionStatus.RECONNECTING
        self.retry_attempts += 1

    def on_failure(self, error: str) -> None:
        """Update state on connection failure.

        Args:
            error: Error message describing the failure.
        """
        self.status = ConnectionStatus.FAILED
        self.last_error = error
        self.consecutive_failures += 1
