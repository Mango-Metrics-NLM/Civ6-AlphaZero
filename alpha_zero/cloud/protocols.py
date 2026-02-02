"""Abstract protocols for cloud infrastructure.

Defines interfaces for:
- CloudStorageProtocol: Checkpoint and artifact storage
- CloudMonitoringProtocol: Metrics and logging
- PreemptionHandlerProtocol: Spot/preemptible instance handling

These protocols enable:
- Backend agnostic cloud operations
- Easy mocking for testing
- Swap between GCP, AWS, Azure, or local implementations
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CloudStorageProtocol(Protocol):
    """Protocol for cloud storage backends.

    Implementations must provide methods for:
    - Uploading and downloading checkpoints
    - Listing available checkpoints
    - Managing checkpoint lifecycle
    """

    def upload_checkpoint(
        self,
        local_path: Path | str,
        remote_key: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload a checkpoint to cloud storage.

        Args:
            local_path: Path to local checkpoint file.
            remote_key: Key/path in cloud storage.
            metadata: Optional metadata to attach.

        Returns:
            URI of uploaded checkpoint.

        Raises:
            CloudStorageError: If upload fails.
        """
        ...

    def download_checkpoint(
        self,
        remote_key: str,
        local_path: Path | str,
    ) -> Path:
        """Download a checkpoint from cloud storage.

        Args:
            remote_key: Key/path in cloud storage.
            local_path: Local path to save checkpoint.

        Returns:
            Path to downloaded file.

        Raises:
            CloudStorageError: If download fails.
        """
        ...

    def list_checkpoints(
        self,
        prefix: str | None = None,
        max_results: int = 100,
    ) -> list[str]:
        """List available checkpoints.

        Args:
            prefix: Optional prefix to filter checkpoints.
            max_results: Maximum number of results to return.

        Returns:
            List of checkpoint keys.
        """
        ...

    def get_latest_checkpoint(self, prefix: str | None = None) -> str | None:
        """Get the most recent checkpoint.

        Args:
            prefix: Optional prefix to filter checkpoints.

        Returns:
            Key of latest checkpoint, or None if no checkpoints exist.
        """
        ...

    def delete_checkpoint(self, remote_key: str) -> bool:
        """Delete a checkpoint from cloud storage.

        Args:
            remote_key: Key/path in cloud storage.

        Returns:
            True if deleted successfully.

        Raises:
            CloudStorageError: If deletion fails.
        """
        ...


@runtime_checkable
class CloudMonitoringProtocol(Protocol):
    """Protocol for cloud monitoring and logging backends.

    Implementations must provide methods for:
    - Logging training metrics
    - Logging events with severity levels
    - Flushing pending logs
    """

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, float],
        labels: dict[str, str] | None = None,
    ) -> None:
        """Log training metrics.

        Args:
            step: Training step number.
            metrics: Dictionary of metric names to values.
            labels: Optional labels for filtering.
        """
        ...

    def log_event(
        self,
        severity: str,
        message: str,
        labels: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Log an event with severity level.

        Args:
            severity: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            message: Event message.
            labels: Optional labels for filtering.
            payload: Optional structured data.
        """
        ...

    def flush(self) -> None:
        """Flush any pending logs to the backend."""
        ...


@runtime_checkable
class PreemptionHandlerProtocol(Protocol):
    """Protocol for preemption detection and handling.

    Implementations must provide methods for:
    - Starting/stopping preemption monitoring
    - Checking preemption status
    - Registering cleanup callbacks
    """

    def start_monitoring(self) -> None:
        """Start monitoring for preemption signals."""
        ...

    def stop_monitoring(self) -> None:
        """Stop monitoring for preemption signals."""
        ...

    def is_preemption_pending(self) -> bool:
        """Check if preemption is pending.

        Returns:
            True if preemption notice has been received.
        """
        ...

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback for preemption.

        Args:
            callback: Function to call during graceful shutdown.
        """
        ...

    def get_time_remaining(self) -> float | None:
        """Get estimated time remaining before termination.

        Returns:
            Seconds remaining, or None if not preempted.
        """
        ...
