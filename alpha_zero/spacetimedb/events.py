"""Event handler registry for SpacetimeDB subscriptions.

Provides a typed callback system for SpacetimeDB table update events,
used by both the real client and mock implementations.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class EventRegistry:
    """Registry for SpacetimeDB subscription event handlers.

    Supports registering and firing typed callbacks for different
    table update events. Thread-safe for use across async boundaries.

    Usage:
        registry = EventRegistry()
        registry.on("model_version_changed", my_callback)
        registry.fire("model_version_changed", new_version=5)
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    def on(self, event: str, callback: Callable[..., Any]) -> None:
        """Register a callback for an event.

        Args:
            event: Event name (e.g., "model_version_changed").
            callback: Function to call when event fires.
        """
        self._handlers[event].append(callback)

    def off(self, event: str, callback: Callable[..., Any]) -> None:
        """Unregister a callback for an event.

        Args:
            event: Event name.
            callback: Function to remove.
        """
        handlers = self._handlers.get(event, [])
        if callback in handlers:
            handlers.remove(callback)

    def fire(self, event: str, *args: Any, **kwargs: Any) -> int:
        """Fire all callbacks for an event.

        Args:
            event: Event name.
            *args: Positional arguments passed to callbacks.
            **kwargs: Keyword arguments passed to callbacks.

        Returns:
            Number of callbacks that were called.
        """
        handlers = self._handlers.get(event, [])
        called = 0
        for handler in handlers:
            try:
                handler(*args, **kwargs)
                called += 1
            except Exception:
                logger.exception(
                    "Error in event handler for '%s'",
                    event,
                )
        return called

    def has_handlers(self, event: str) -> bool:
        """Check if any handlers are registered for an event.

        Args:
            event: Event name.

        Returns:
            True if at least one handler is registered.
        """
        return bool(self._handlers.get(event))

    def clear(self, event: str | None = None) -> None:
        """Clear handlers.

        Args:
            event: If specified, clear only handlers for this event.
                   If None, clear all handlers.
        """
        if event is not None:
            self._handlers.pop(event, None)
        else:
            self._handlers.clear()

    @property
    def registered_events(self) -> list[str]:
        """Get list of events that have registered handlers."""
        return [e for e, h in self._handlers.items() if h]


# Standard event names
MODEL_VERSION_CHANGED = "model_version_changed"
TRAJECTORY_ADDED = "trajectory_added"
METRICS_UPDATED = "metrics_updated"
TRAINING_STATE_CHANGED = "training_state_changed"
WORKER_STATUS_CHANGED = "worker_status_changed"
CONNECTION_STATE_CHANGED = "connection_state_changed"

__all__ = [
    "CONNECTION_STATE_CHANGED",
    "METRICS_UPDATED",
    "MODEL_VERSION_CHANGED",
    "TRAINING_STATE_CHANGED",
    "TRAJECTORY_ADDED",
    "WORKER_STATUS_CHANGED",
    "EventRegistry",
]
