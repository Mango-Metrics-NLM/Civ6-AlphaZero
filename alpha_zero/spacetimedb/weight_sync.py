"""Push-based weight synchronizer using SpacetimeDB subscriptions.

Provides a drop-in replacement for the polling-based WeightSynchronizer
in distributed/shared_storage.py. Uses SpacetimeDB subscriptions for
immediate weight update notifications instead of polling.

Falls back to polling if subscription is unavailable.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from alpha_zero.spacetimedb.client import SpacetimeDBClient
from alpha_zero.spacetimedb.serialization import deserialize_state_dict

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SpacetimeDBWeightSynchronizer:
    """Push-based weight synchronizer using SpacetimeDB subscriptions.

    Subscribes to model version changes and immediately fetches new weights
    when an update is published. Falls back to polling if subscriptions
    are unavailable or disconnected.

    Provides the same API as WeightSynchronizer for drop-in replacement.

    Usage:
        client = SpacetimeDBClient(config)
        client.connect_with_mock(mock)  # or await client.connect()

        sync = SpacetimeDBWeightSynchronizer(client)

        # In actor loop:
        updated, version = sync.update_if_available(model)
        if updated:
            print(f"Model updated to version {version}")
    """

    def __init__(
        self,
        client: SpacetimeDBClient,
        update_interval: float = 1.0,
        compress: bool = True,
    ) -> None:
        """Initialize the weight synchronizer.

        Args:
            client: Connected SpacetimeDB client.
            update_interval: Minimum seconds between weight checks (for polling fallback).
            compress: Whether weights are compressed in storage.
        """
        self._client = client
        self._update_interval = update_interval
        self._compress = compress

        self._current_version: int = 0
        self._latest_available_version: int = 0
        self._last_check_time: float = 0.0
        self._pending_update: bool = False

        # Register subscription for push-based updates
        self._client.subscribe_model_versions(self._on_version_changed)
        logger.info(
            "SpacetimeDB weight synchronizer initialized (interval=%.1fs)",
            update_interval,
        )

    def _on_version_changed(self, version: int) -> None:
        """Callback when a new model version is available.

        Args:
            version: New model version number.
        """
        if version > self._latest_available_version:
            self._latest_available_version = version
            self._pending_update = True
            logger.info("New model version available: %d", version)

    @property
    def current_version(self) -> int:
        """Get currently loaded model version."""
        return self._current_version

    @property
    def latest_available_version(self) -> int:
        """Get the latest available version from subscriptions."""
        return self._latest_available_version

    def should_update(self) -> bool:
        """Check if weights should be updated.

        Uses subscription notifications if available, otherwise
        falls back to time-based polling.

        Returns:
            True if new weights are available.
        """
        # Subscription-based: immediate notification
        if self._pending_update:
            return True

        # Polling fallback: check time interval
        now = time.monotonic()
        if now - self._last_check_time < self._update_interval:
            return False

        self._last_check_time = now

        # Check client's version against our version
        client_version = self._client.current_version
        if client_version > self._current_version:
            self._latest_available_version = client_version
            self._pending_update = True
            return True

        return False

    def update_model(self, model: torch.nn.Module) -> bool:
        """Update model with latest weights from SpacetimeDB.

        Args:
            model: Model to update.

        Returns:
            True if model was updated.
        """
        try:
            result = self._client.get_latest_weights_sync()
            if result is None:
                logger.debug("No weights available to update")
                return False

            weights_bytes, metadata = result
            state_dict = deserialize_state_dict(
                weights_bytes,
                compressed=self._compress,
            )

            model.load_state_dict(state_dict)

            new_version = metadata.get("version", self._latest_available_version)
            if isinstance(new_version, int):
                self._current_version = new_version
            else:
                self._current_version = self._latest_available_version

            self._pending_update = False
            logger.info("Model updated to version %d", self._current_version)
            return True

        except Exception:
            logger.exception("Failed to update model weights")
            return False

    def update_if_available(
        self, model: torch.nn.Module,
    ) -> tuple[bool, int]:
        """Atomically check for and apply weight updates.

        Args:
            model: Model to update.

        Returns:
            Tuple of (was_updated, current_version).
        """
        if self.should_update():
            updated = self.update_model(model)
            return updated, self._current_version
        return False, self._current_version

    def get_current_version(self) -> int:
        """Get currently loaded model version.

        Returns:
            Current model version number.
        """
        return self._current_version


__all__ = ["SpacetimeDBWeightSynchronizer"]
