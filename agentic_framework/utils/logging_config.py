"""
Logging Configuration for Agentic Framework

Provides structured logging with multiple handlers, context management,
and JSON output support for debugging and monitoring.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Generator

# Thread-local storage for log context
_context = threading.local()


def get_context() -> dict[str, Any]:
    """Get the current thread's logging context."""
    if not hasattr(_context, "data"):
        _context.data = {}
    return _context.data


def set_context(key: str, value: Any) -> None:
    """Set a value in the current thread's logging context."""
    get_context()[key] = value


def clear_context() -> None:
    """Clear the current thread's logging context."""
    _context.data = {}


@dataclass
class LogContext:
    """Context manager for structured logging context.

    Automatically adds and removes context fields for the duration
    of the context block.
    """

    fields: dict[str, Any] = field(default_factory=dict)
    _previous: dict[str, Any] = field(default_factory=dict, init=False)

    def __enter__(self) -> LogContext:
        """Enter the context and add fields."""
        self._previous = get_context().copy()
        get_context().update(self.fields)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context and restore previous fields."""
        _context.data = self._previous


class ContextFilter(logging.Filter):
    """Logging filter that adds context fields to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context fields to the record."""
        context = get_context()
        for key, value in context.items():
            setattr(record, key, value)

        # Add default values if not present
        if not hasattr(record, "request_id"):
            record.request_id = ""
        if not hasattr(record, "agent"):
            record.agent = ""
        if not hasattr(record, "phase"):
            record.phase = ""

        return True


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured log output."""

    def __init__(
        self,
        fields: Optional[list[str]] = None,
        timestamp_format: str = "%Y-%m-%dT%H:%M:%S.%f",
    ) -> None:
        """Initialize the JSON formatter.

        Args:
            fields: List of fields to include in JSON output.
            timestamp_format: Format string for timestamps.
        """
        super().__init__()
        self.fields = fields or [
            "timestamp",
            "level",
            "logger",
            "message",
            "request_id",
            "agent",
            "phase",
            "filename",
            "lineno",
            "funcName",
        ]
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data: dict[str, Any] = {}

        for field_name in self.fields:
            if field_name == "timestamp":
                log_data[field_name] = datetime.fromtimestamp(record.created).strftime(
                    self.timestamp_format
                )
            elif field_name == "level":
                log_data[field_name] = record.levelname
            elif field_name == "logger":
                log_data[field_name] = record.name
            elif field_name == "message":
                log_data[field_name] = record.getMessage()
            elif hasattr(record, field_name):
                log_data[field_name] = getattr(record, field_name)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class StructuredLogger:
    """Structured logger with context support.

    Provides a convenient interface for logging with automatic
    context management and structured fields.
    """

    def __init__(self, name: str) -> None:
        """Initialize the structured logger.

        Args:
            name: Logger name.
        """
        self._logger = logging.getLogger(name)
        self._name = name

    @property
    def name(self) -> str:
        """Get the logger name."""
        return self._name

    def _log(
        self,
        level: int,
        msg: str,
        *args: Any,
        extra: Optional[dict[str, Any]] = None,
        exc_info: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log a message with extra fields.

        Args:
            level: Log level.
            msg: Log message.
            *args: Message format arguments.
            extra: Extra fields to include.
            exc_info: Exception info.
            **kwargs: Additional fields to include.
        """
        combined_extra = {**(extra or {}), **kwargs}
        self._logger.log(level, msg, *args, exc_info=exc_info, extra=combined_extra)

    def debug(
        self, msg: str, *args: Any, extra: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Log a debug message."""
        self._log(logging.DEBUG, msg, *args, extra=extra, **kwargs)

    def info(
        self, msg: str, *args: Any, extra: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Log an info message."""
        self._log(logging.INFO, msg, *args, extra=extra, **kwargs)

    def warning(
        self, msg: str, *args: Any, extra: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Log a warning message."""
        self._log(logging.WARNING, msg, *args, extra=extra, **kwargs)

    def error(
        self,
        msg: str,
        *args: Any,
        extra: Optional[dict[str, Any]] = None,
        exc_info: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log an error message."""
        self._log(logging.ERROR, msg, *args, extra=extra, exc_info=exc_info, **kwargs)

    def critical(
        self,
        msg: str,
        *args: Any,
        extra: Optional[dict[str, Any]] = None,
        exc_info: Any = None,
        **kwargs: Any,
    ) -> None:
        """Log a critical message."""
        self._log(logging.CRITICAL, msg, *args, extra=extra, exc_info=exc_info, **kwargs)

    def exception(
        self, msg: str, *args: Any, extra: Optional[dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Log an exception message."""
        self._log(logging.ERROR, msg, *args, extra=extra, exc_info=True, **kwargs)

    @contextmanager
    def context(self, **fields: Any) -> Generator[LogContext, None, None]:
        """Context manager for adding fields to log messages.

        Args:
            **fields: Fields to add to the context.

        Yields:
            LogContext instance.
        """
        with LogContext(fields=fields) as ctx:
            yield ctx

    def with_fields(self, **fields: Any) -> StructuredLogger:
        """Create a new logger with default fields.

        Args:
            **fields: Default fields for the new logger.

        Returns:
            New StructuredLogger with fields set in context.
        """
        # Set fields in context
        get_context().update(fields)
        return self


def setup_logging(
    level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers_config: Optional[list[dict[str, Any]]] = None,
    log_dir: Optional[Path] = None,
) -> None:
    """Set up logging for the framework.

    Args:
        level: Default log level.
        log_format: Log message format.
        handlers_config: List of handler configurations.
        log_dir: Directory for log files.
    """
    # Get root logger for the framework
    root_logger = logging.getLogger("agentic_framework")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add context filter
    context_filter = ContextFilter()

    # Default handlers if none specified
    if handlers_config is None:
        handlers_config = [
            {"type": "console", "level": level},
        ]

    # Create log directory if needed
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    # Configure handlers
    for handler_config in handlers_config:
        handler_type = handler_config.get("type", "console")
        handler_level = handler_config.get("level", level)
        destination = handler_config.get("destination")

        handler: logging.Handler

        if handler_type == "console":
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(log_format))

        elif handler_type == "file":
            if destination is None:
                destination = "agentic_framework.log"
            if log_dir is not None:
                destination = str(log_dir / destination)
            else:
                Path(destination).parent.mkdir(parents=True, exist_ok=True)

            handler = logging.FileHandler(destination)
            handler.setFormatter(logging.Formatter(log_format))

        elif handler_type == "rotating_file":
            if destination is None:
                destination = "agentic_framework.log"
            if log_dir is not None:
                destination = str(log_dir / destination)
            else:
                Path(destination).parent.mkdir(parents=True, exist_ok=True)

            max_bytes = handler_config.get("max_bytes", 10 * 1024 * 1024)  # 10MB
            backup_count = handler_config.get("backup_count", 5)

            handler = logging.handlers.RotatingFileHandler(
                destination,
                maxBytes=max_bytes,
                backupCount=backup_count,
            )
            handler.setFormatter(logging.Formatter(log_format))

        elif handler_type == "json":
            if destination is None:
                destination = "agentic_framework.json"
            if log_dir is not None:
                destination = str(log_dir / destination)
            else:
                Path(destination).parent.mkdir(parents=True, exist_ok=True)

            handler = logging.FileHandler(destination)
            handler.setFormatter(JsonFormatter())

        else:
            continue

        handler.setLevel(getattr(logging, handler_level.upper()))
        handler.addFilter(context_filter)
        root_logger.addHandler(handler)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (will be prefixed with 'agentic_framework.').

    Returns:
        StructuredLogger instance.
    """
    full_name = f"agentic_framework.{name}" if not name.startswith("agentic_framework.") else name
    return StructuredLogger(full_name)


def generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return str(uuid.uuid4())[:8]
