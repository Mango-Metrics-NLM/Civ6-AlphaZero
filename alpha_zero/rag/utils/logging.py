"""Structured logging for RAG system.

Provides:
- Correlation ID tracking for distributed tracing
- Structured log formatting
- Context-aware logging
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

from alpha_zero.rag.config.settings import LoggingConfig, LogLevel

# Context variable for correlation ID
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id",
    default="",
)


def get_correlation_id() -> str:
    """Get current correlation ID.

    Returns:
        Current correlation ID or empty string
    """
    return _correlation_id.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Set correlation ID for current context.

    Args:
        correlation_id: ID to set, or None to generate new

    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())[:8]
    _correlation_id.set(correlation_id)
    return correlation_id


class CorrelationIdFilter(logging.Filter):
    """Logging filter that adds correlation ID to records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id to the log record.

        Args:
            record: Log record

        Returns:
            True (always passes filter)
        """
        record.correlation_id = get_correlation_id() or "-"
        return True


@dataclass
class LogContext:
    """Context for structured logging.

    Attributes:
        correlation_id: Request correlation ID
        component: Component name
        operation: Operation being performed
        metadata: Additional context metadata
    """

    correlation_id: str = field(default_factory=lambda: get_correlation_id() or "-")
    component: str = ""
    operation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __enter__(self) -> LogContext:
        """Enter context and set correlation ID."""
        if not get_correlation_id():
            set_correlation_id(self.correlation_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "correlation_id": self.correlation_id,
            "component": self.component,
            "operation": self.operation,
            **self.metadata,
        }


class RAGLogger:
    """Structured logger for RAG system.

    Provides context-aware logging with automatic correlation ID
    tracking and structured metadata.

    Example:
        logger = RAGLogger("rag.workflow")
        with logger.context(operation="search") as ctx:
            logger.info("Starting search", query=query)
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        config: LoggingConfig | None = None,
    ) -> None:
        """Initialize RAG logger.

        Args:
            name: Logger name
            level: Log level
            config: Logging configuration
        """
        self.name = name
        self.config = config or LoggingConfig()
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.value))

        # Add correlation ID filter
        self._logger.addFilter(CorrelationIdFilter())

    def _format_message(
        self,
        message: str,
        **kwargs: Any,
    ) -> str:
        """Format message with structured data.

        Args:
            message: Base message
            **kwargs: Additional fields

        Returns:
            Formatted message
        """
        if not kwargs:
            return message

        parts = [message]
        for key, value in kwargs.items():
            if isinstance(value, (dict, list)):
                parts.append(f"{key}={value!r}")
            else:
                parts.append(f"{key}={value}")

        return " | ".join(parts)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message.

        Args:
            message: Log message
            **kwargs: Additional structured fields
        """
        self._logger.debug(self._format_message(message, **kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message.

        Args:
            message: Log message
            **kwargs: Additional structured fields
        """
        self._logger.info(self._format_message(message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message.

        Args:
            message: Log message
            **kwargs: Additional structured fields
        """
        self._logger.warning(self._format_message(message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message.

        Args:
            message: Log message
            **kwargs: Additional structured fields
        """
        self._logger.error(self._format_message(message, **kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback.

        Args:
            message: Log message
            **kwargs: Additional structured fields
        """
        self._logger.exception(self._format_message(message, **kwargs))

    def context(
        self,
        operation: str,
        **metadata: Any,
    ) -> LogContext:
        """Create a logging context.

        Args:
            operation: Operation name
            **metadata: Additional context metadata

        Returns:
            LogContext instance
        """
        return LogContext(
            component=self.name,
            operation=operation,
            metadata=metadata,
        )


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Setup logging with configuration.

    Args:
        config: Logging configuration
    """
    config = config or LoggingConfig()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.level.value))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.level.value))

    # Create formatter
    formatter = logging.Formatter(config.log_format)
    console_handler.setFormatter(formatter)

    # Add correlation ID filter
    console_handler.addFilter(CorrelationIdFilter())

    root_logger.addHandler(console_handler)


def get_logger(name: str, config: LoggingConfig | None = None) -> RAGLogger:
    """Get a configured RAG logger.

    Args:
        name: Logger name
        config: Optional logging configuration

    Returns:
        Configured RAGLogger instance
    """
    config = config or LoggingConfig()
    return RAGLogger(name, config.level, config)
