"""Preemption handling for Spot/Preemptible VMs.

Provides utilities for graceful shutdown when a Spot VM is preempted.
Google Cloud sends SIGTERM with a 30-second grace period before
terminating preemptible instances.

Usage:
    from alpha_zero.cloud.preemption import PreemptionHandler

    handler = PreemptionHandler()
    handler.register()

    while not handler.is_preempted:
        # Do work
        pass

    # Graceful cleanup
    handler.cleanup()
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PreemptionHandler:
    """Handler for Spot VM preemption signals.

    Registers signal handlers for SIGTERM and SIGINT to enable
    graceful shutdown when a VM is preempted or interrupted.

    Attributes:
        grace_period: Seconds to allow for cleanup (default: 25s, leaving 5s buffer).
        cleanup_callbacks: List of callbacks to run on preemption.
        is_preempted: Whether preemption signal was received.
        preemption_time: Timestamp when preemption was signaled.
    """

    grace_period: float = 25.0
    cleanup_callbacks: list[Callable[[], None]] = field(default_factory=list)
    is_preempted: bool = False
    preemption_time: datetime | None = None

    _shutdown_event: threading.Event = field(default_factory=threading.Event)
    _original_sigterm: Any = None
    _original_sigint: Any = None

    def register(self) -> None:
        """Register signal handlers for preemption.

        Should be called early in the main process to catch signals.
        """
        # Save original handlers
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_signal)

        logger.info(
            f"Preemption handler registered (grace_period={self.grace_period}s)"
        )

    def unregister(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)

        logger.info("Preemption handler unregistered")

    def _handle_signal(self, signum: int, frame: object) -> None:
        """Handle preemption signal.

        Args:
            signum: Signal number received.
            frame: Current stack frame (unused).
        """
        signal_name = signal.Signals(signum).name
        self.preemption_time = datetime.now(UTC)
        self.is_preempted = True
        self._shutdown_event.set()

        logger.warning(
            f"Received {signal_name} - initiating graceful shutdown "
            f"(grace_period={self.grace_period}s)"
        )

        # Run cleanup in a thread to avoid blocking signal handler
        cleanup_thread = threading.Thread(target=self._run_cleanup_with_timeout)
        cleanup_thread.start()
        cleanup_thread.join(timeout=self.grace_period)

        if cleanup_thread.is_alive():
            logger.error("Cleanup did not complete within grace period, forcing exit")

        sys.exit(0)

    def _run_cleanup_with_timeout(self) -> None:
        """Run cleanup callbacks with timeout monitoring."""
        deadline = time.time() + self.grace_period

        for i, callback in enumerate(self.cleanup_callbacks):
            if time.time() >= deadline:
                logger.warning("Cleanup timeout - skipping remaining callbacks")
                break

            try:
                remaining = deadline - time.time()
                logger.info(
                    f"Running cleanup callback {i + 1}/{len(self.cleanup_callbacks)} "
                    f"({remaining:.1f}s remaining)"
                )
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback {i + 1}: {e}")

        logger.info("Cleanup completed")

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback to run on preemption.

        Callbacks are executed in order of registration.
        Each callback should complete quickly (target <5s each).

        Args:
            callback: Function to call on preemption.
        """
        self.cleanup_callbacks.append(callback)
        logger.debug(f"Added cleanup callback (total: {len(self.cleanup_callbacks)})")

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        """Wait for shutdown signal.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if shutdown was signaled, False if timeout occurred.
        """
        return self._shutdown_event.wait(timeout=timeout)

    def check_preemption(self) -> bool:
        """Check if preemption was signaled.

        Returns:
            True if preemption signal was received.
        """
        return self.is_preempted

    def time_since_preemption(self) -> float | None:
        """Get time elapsed since preemption signal.

        Returns:
            Seconds since preemption, or None if not preempted.
        """
        if self.preemption_time is None:
            return None
        return (datetime.now(UTC) - self.preemption_time).total_seconds()


# Global preemption handler instance
_global_handler: PreemptionHandler | None = None


def register_preemption_handler(
    grace_period: float = 25.0,
    cleanup_callbacks: list[Callable[[], None]] | None = None,
) -> PreemptionHandler:
    """Register a global preemption handler.

    This is a convenience function for simple use cases where
    only one handler is needed.

    Args:
        grace_period: Seconds to allow for cleanup.
        cleanup_callbacks: Optional list of cleanup callbacks.

    Returns:
        The registered PreemptionHandler instance.
    """
    global _global_handler

    _global_handler = PreemptionHandler(
        grace_period=grace_period,
        cleanup_callbacks=cleanup_callbacks or [],
    )
    _global_handler.register()

    return _global_handler


def get_preemption_handler() -> PreemptionHandler | None:
    """Get the global preemption handler.

    Returns:
        The global handler, or None if not registered.
    """
    return _global_handler


def is_preempted() -> bool:
    """Check if the global handler was preempted.

    Returns:
        True if preemption signal was received.
    """
    if _global_handler is None:
        return False
    return _global_handler.is_preempted


__all__ = [
    "PreemptionHandler",
    "register_preemption_handler",
    "get_preemption_handler",
    "is_preempted",
]
