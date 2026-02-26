"""SpacetimeDB configuration.

Provides a dataclass-based configuration for SpacetimeDB connections,
following the pattern established by GCSConfig and StorageConfig.

Environment Variables:
    CIV6_SPACETIMEDB_ENABLED: Enable SpacetimeDB integration (default: false)
    CIV6_SPACETIMEDB_HOST: SpacetimeDB server host URL
    CIV6_SPACETIMEDB_MODULE: Module address/name
    CIV6_SPACETIMEDB_AUTH_TOKEN: Authentication token
    CIV6_SPACETIMEDB_SSL: Enable SSL for connections
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class SpacetimeDBConfig:
    """Configuration for SpacetimeDB connection.

    Attributes:
        enabled: Whether SpacetimeDB integration is active.
        host: SpacetimeDB server host URL.
        module_address: Module address or name to connect to.
        auth_token: Authentication token for the connection.
        ssl_enabled: Whether to use SSL for connections.
        reconnect_interval: Seconds between reconnect attempts.
        max_reconnect_attempts: Maximum number of reconnect attempts.
        connection_timeout: Timeout for initial connection in seconds.
        subscription_timeout: Timeout for subscription operations in seconds.
        compress_weights: Whether to compress model weights before storage.
        max_weight_blob_mb: Maximum weight blob size in MB (safety limit).
        subscribe_model_versions: Subscribe to model version changes.
        subscribe_training_state: Subscribe to training state changes.
        subscribe_metrics: Subscribe to training metrics updates.
    """

    enabled: bool = False
    host: str = "http://localhost:3000"
    module_address: str = ""
    auth_token: str | None = None
    ssl_enabled: bool = False

    # Connection resilience
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 10
    connection_timeout: float = 30.0
    subscription_timeout: float = 30.0

    # Weight transfer
    compress_weights: bool = True
    max_weight_blob_mb: int = 500

    # Subscription toggles
    subscribe_model_versions: bool = True
    subscribe_training_state: bool = True
    subscribe_metrics: bool = False

    def __post_init__(self) -> None:
        """Apply environment variable overrides after initialization."""
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply CIV6_SPACETIMEDB_* environment variable overrides."""
        if env_enabled := os.environ.get("CIV6_SPACETIMEDB_ENABLED"):
            self.enabled = env_enabled.lower() in ("true", "1", "yes")

        if env_host := os.environ.get("CIV6_SPACETIMEDB_HOST"):
            self.host = env_host

        if env_module := os.environ.get("CIV6_SPACETIMEDB_MODULE"):
            self.module_address = env_module

        if env_token := os.environ.get("CIV6_SPACETIMEDB_AUTH_TOKEN"):
            self.auth_token = env_token

        if env_ssl := os.environ.get("CIV6_SPACETIMEDB_SSL"):
            self.ssl_enabled = env_ssl.lower() in ("true", "1", "yes")

    @classmethod
    def from_env(cls) -> SpacetimeDBConfig:
        """Create configuration from environment variables only.

        Returns:
            SpacetimeDBConfig with values from environment.
        """
        config = cls()
        # __post_init__ already applies env overrides
        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpacetimeDBConfig:
        """Create configuration from a dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            SpacetimeDBConfig instance.
        """
        return cls(
            enabled=data.get("enabled", False),
            host=data.get("host", "http://localhost:3000"),
            module_address=data.get("module_address", ""),
            auth_token=data.get("auth_token"),
            ssl_enabled=data.get("ssl_enabled", False),
            reconnect_interval=data.get("reconnect_interval", 5.0),
            max_reconnect_attempts=data.get("max_reconnect_attempts", 10),
            connection_timeout=data.get("connection_timeout", 30.0),
            subscription_timeout=data.get("subscription_timeout", 30.0),
            compress_weights=data.get("compress_weights", True),
            max_weight_blob_mb=data.get("max_weight_blob_mb", 500),
            subscribe_model_versions=data.get("subscribe_model_versions", True),
            subscribe_training_state=data.get("subscribe_training_state", True),
            subscribe_metrics=data.get("subscribe_metrics", False),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Configuration as a dictionary (excludes auth_token for security).
        """
        return {
            "enabled": self.enabled,
            "host": self.host,
            "module_address": self.module_address,
            "ssl_enabled": self.ssl_enabled,
            "reconnect_interval": self.reconnect_interval,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "connection_timeout": self.connection_timeout,
            "subscription_timeout": self.subscription_timeout,
            "compress_weights": self.compress_weights,
            "max_weight_blob_mb": self.max_weight_blob_mb,
            "subscribe_model_versions": self.subscribe_model_versions,
            "subscribe_training_state": self.subscribe_training_state,
            "subscribe_metrics": self.subscribe_metrics,
        }

    @classmethod
    def for_local(
        cls,
        module_address: str,
        host: str = "http://localhost:3000",
    ) -> SpacetimeDBConfig:
        """Create configuration for a local SpacetimeDB instance.

        Args:
            module_address: Module address to connect to.
            host: Local server host URL.

        Returns:
            SpacetimeDBConfig configured for local development.
        """
        return cls(
            enabled=True,
            host=host,
            module_address=module_address,
        )


__all__ = ["SpacetimeDBConfig"]
