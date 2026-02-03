"""
Custom exceptions for GCP cloud operations.

Provides a hierarchy of exceptions for handling GCP-specific errors
with actionable error messages and recovery suggestions.
"""

from __future__ import annotations

from typing import Any


class GCPError(Exception):
    """Base exception for all GCP-related errors.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional context.
        recoverable: Whether the error can be recovered from.
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable

    def __str__(self) -> str:
        base = self.message
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base = f"{base} ({detail_str})"
        return base

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, details={self.details!r})"


class ConfigurationError(GCPError):
    """Error in GCP configuration.

    Raised when:
    - Required environment variables are missing
    - Configuration values are invalid
    - Project/region/bucket settings are incorrect
    """

    def __init__(
        self,
        message: str,
        missing_vars: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if missing_vars:
            details["missing_vars"] = missing_vars
        super().__init__(message, details=details, recoverable=False)
        self.missing_vars = missing_vars or []


class GCSStorageError(GCPError):
    """Error in GCS storage operations.

    Raised when:
    - Upload/download fails
    - Object not found
    - Permission denied
    - Bucket does not exist
    """

    def __init__(
        self,
        message: str,
        bucket: str | None = None,
        blob_path: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ) -> None:
        details = details or {}
        if bucket:
            details["bucket"] = bucket
        if blob_path:
            details["blob_path"] = blob_path
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details, recoverable=recoverable)
        self.bucket = bucket
        self.blob_path = blob_path
        self.operation = operation


class VertexJobError(GCPError):
    """Error in Vertex AI job operations.

    Raised when:
    - Job submission fails
    - Job execution fails
    - Resource quota exceeded
    - Preemption without recovery
    """

    def __init__(
        self,
        message: str,
        job_id: str | None = None,
        job_state: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool = False,
    ) -> None:
        details = details or {}
        if job_id:
            details["job_id"] = job_id
        if job_state:
            details["job_state"] = job_state
        super().__init__(message, details=details, recoverable=recoverable)
        self.job_id = job_id
        self.job_state = job_state


class PreemptionError(VertexJobError):
    """Spot/Preemptible VM was preempted.

    This is a recoverable error - the job can be restarted.
    """

    def __init__(
        self,
        message: str = "Spot VM was preempted",
        job_id: str | None = None,
        checkpoint_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if checkpoint_path:
            details["checkpoint_path"] = checkpoint_path
        super().__init__(message, job_id=job_id, details=details, recoverable=True)
        self.checkpoint_path = checkpoint_path


class QuotaExceededError(VertexJobError):
    """Resource quota exceeded in GCP.

    Raised when:
    - GPU quota is exhausted
    - CPU quota is exhausted
    - Too many concurrent jobs
    """

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        quota_limit: int | None = None,
        requested: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if resource_type:
            details["resource_type"] = resource_type
        if quota_limit is not None:
            details["quota_limit"] = quota_limit
        if requested is not None:
            details["requested"] = requested
        super().__init__(message, details=details, recoverable=False)
        self.resource_type = resource_type
        self.quota_limit = quota_limit
        self.requested = requested
