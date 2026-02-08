"""Custom exceptions for cloud infrastructure.

Provides a hierarchy of exceptions for cloud operations:
- CloudError: Base class for all cloud errors
- CloudConfigError: Configuration and validation errors
- CloudStorageError: Storage operation failures
- CloudConnectionError: Network and authentication failures
- PreemptionError: Preemption-related errors
"""

from __future__ import annotations


class CloudError(Exception):
    """Base exception for all cloud-related errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize cloud error.

        Args:
            message: Error description.
            cause: Original exception that caused this error.
        """
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


class CloudConfigError(CloudError):
    """Configuration or validation error.

    Raised when:
    - Required environment variables are missing
    - Configuration values are invalid
    - YAML configuration cannot be parsed
    """

    pass


class CloudStorageError(CloudError):
    """Storage operation failed.

    Raised when:
    - Upload/download fails
    - Bucket or object not found
    - Permission denied for storage operations
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        path: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize storage error.

        Args:
            message: Error description.
            operation: Operation that failed (upload, download, list, delete).
            path: Path or key that caused the error.
            cause: Original exception.
        """
        super().__init__(message, cause)
        self.operation = operation
        self.path = path


class CloudConnectionError(CloudError):
    """Network or authentication failure.

    Raised when:
    - Cannot connect to cloud services
    - Authentication/credentials are invalid
    - Network timeout occurs
    """

    def __init__(
        self,
        message: str,
        service: str | None = None,
        cause: Exception | None = None,
    ) -> None:
        """Initialize connection error.

        Args:
            message: Error description.
            service: Cloud service that failed (storage, logging, etc.).
            cause: Original exception.
        """
        super().__init__(message, cause)
        self.service = service


class PreemptionError(CloudError):
    """Preemption-related error.

    Raised when:
    - Preemption detection fails
    - Checkpoint save during preemption fails
    - Cleanup operations fail during shutdown
    """

    def __init__(
        self,
        message: str,
        checkpoint_saved: bool = False,
        cause: Exception | None = None,
    ) -> None:
        """Initialize preemption error.

        Args:
            message: Error description.
            checkpoint_saved: Whether checkpoint was successfully saved.
            cause: Original exception.
        """
        super().__init__(message, cause)
        self.checkpoint_saved = checkpoint_saved
