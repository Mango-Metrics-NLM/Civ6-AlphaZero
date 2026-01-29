"""
Logging and Debugging Infrastructure.

This module provides a configurable, structured logging system with support
for multiple output formats, context binding, and performance tracking.

Example:
    >>> from template_system.logging_config import configure_logging, get_logger
    >>> configure_logging(level="DEBUG", format="console")
    >>> logger = get_logger(__name__)
    >>> logger.info("Application started", version="1.0.0")
"""
from __future__ import annotations

import atexit
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TextIO,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


class LogFormat(Enum):
    """Available log output formats."""

    JSON = "json"
    CONSOLE = "console"
    STRUCTURED = "structured"


class LogLevel(Enum):
    """Log levels with numeric values."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @classmethod
    def from_string(cls, value: str) -> LogLevel:
        """Create LogLevel from string.

        Args:
            value: String level name.

        Returns:
            Corresponding LogLevel.

        Raises:
            ValueError: If level name is invalid.
        """
        try:
            return cls[value.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid log level: {value}") from e


@dataclass
class LogConfig:
    """Configuration for the logging system.

    Attributes:
        level: Minimum log level.
        format: Output format (json, console, structured).
        output: Output destination (stdout, stderr, or file path).
        include_timestamp: Whether to include timestamps.
        include_caller: Whether to include caller information.
        include_request_id: Whether to include request ID context.
        max_message_length: Maximum message length (0 for unlimited).
        sensitive_fields: Fields to redact in logs.
        context_fields: Default context fields to include.
    """

    level: LogLevel = LogLevel.INFO
    format: LogFormat = LogFormat.CONSOLE
    output: str = "stdout"
    include_timestamp: bool = True
    include_caller: bool = True
    include_request_id: bool = True
    max_message_length: int = 0
    sensitive_fields: list[str] = field(
        default_factory=lambda: ["password", "token", "api_key", "secret"]
    )
    context_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogConfig:
        """Create LogConfig from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            LogConfig instance.
        """
        if "level" in data and isinstance(data["level"], str):
            data["level"] = LogLevel.from_string(data["level"])
        if "format" in data and isinstance(data["format"], str):
            data["format"] = LogFormat(data["format"])
        return cls(**data)

    @classmethod
    def from_env(cls, prefix: str = "LOG_") -> LogConfig:
        """Create LogConfig from environment variables.

        Args:
            prefix: Environment variable prefix.

        Returns:
            LogConfig instance.
        """
        import os

        config = cls()

        level = os.environ.get(f"{prefix}LEVEL")
        if level:
            config.level = LogLevel.from_string(level)

        fmt = os.environ.get(f"{prefix}FORMAT")
        if fmt:
            config.format = LogFormat(fmt.lower())

        output = os.environ.get(f"{prefix}OUTPUT")
        if output:
            config.output = output

        return config


class ContextFilter(logging.Filter):
    """Filter that adds context to log records."""

    def __init__(self, context: dict[str, Any] | None = None) -> None:
        """Initialize the filter.

        Args:
            context: Default context to add to all records.
        """
        super().__init__()
        self.context = context or {}
        self._local_context: dict[str, Any] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to the log record.

        Args:
            record: Log record to modify.

        Returns:
            Always True (record passes filter).
        """
        # Add default context
        for key, value in self.context.items():
            if not hasattr(record, key):
                setattr(record, key, value)

        # Add local context
        for key, value in self._local_context.items():
            setattr(record, key, value)

        return True

    def bind(self, **kwargs: Any) -> None:
        """Bind context to all future log records.

        Args:
            **kwargs: Context key-value pairs.
        """
        self._local_context.update(kwargs)

    def unbind(self, *keys: str) -> None:
        """Remove context from future log records.

        Args:
            *keys: Context keys to remove.
        """
        for key in keys:
            self._local_context.pop(key, None)

    def clear(self) -> None:
        """Clear all local context."""
        self._local_context.clear()


class SensitiveDataFilter(logging.Filter):
    """Filter that redacts sensitive data from log records."""

    def __init__(self, sensitive_fields: list[str]) -> None:
        """Initialize the filter.

        Args:
            sensitive_fields: Field names to redact.
        """
        super().__init__()
        self.sensitive_fields = set(f.lower() for f in sensitive_fields)
        self.redacted_value = "[REDACTED]"

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data in the log record.

        Args:
            record: Log record to modify.

        Returns:
            Always True (record passes filter).
        """
        # Redact in extra fields
        for field_name in list(vars(record)):
            if field_name.lower() in self.sensitive_fields:
                setattr(record, field_name, self.redacted_value)

        # Redact in message args if dict
        if record.args and isinstance(record.args, dict):
            record.args = self._redact_dict(record.args)

        return True

    def _redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive fields in a dictionary.

        Args:
            data: Dictionary to redact.

        Returns:
            Dictionary with sensitive fields redacted.
        """
        result = {}
        for key, value in data.items():
            if key.lower() in self.sensitive_fields:
                result[key] = self.redacted_value
            elif isinstance(value, dict):
                result[key] = self._redact_dict(value)
            else:
                result[key] = value
        return result


class JsonFormatter(logging.Formatter):
    """Formatter that outputs JSON log records."""

    def __init__(
        self,
        include_timestamp: bool = True,
        include_caller: bool = True,
        extra_fields: list[str] | None = None,
    ) -> None:
        """Initialize the formatter.

        Args:
            include_timestamp: Whether to include timestamps.
            include_caller: Whether to include caller info.
            extra_fields: Additional fields to include.
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_caller = include_caller
        self.extra_fields = extra_fields or []
        self._default_keys = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "message", "asctime", "taskName",
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: Log record to format.

        Returns:
            JSON string.
        """
        log_data: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.include_timestamp:
            log_data["timestamp"] = datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat()

        if self.include_caller:
            log_data["caller"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key in dir(record):
            if key not in self._default_keys and not key.startswith("_"):
                value = getattr(record, key)
                if not callable(value):
                    log_data[key] = value

        return json.dumps(log_data, default=str)


class ConsoleFormatter(logging.Formatter):
    """Formatter for human-readable console output with colors."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def __init__(
        self,
        include_timestamp: bool = True,
        use_colors: bool = True,
    ) -> None:
        """Initialize the formatter.

        Args:
            include_timestamp: Whether to include timestamps.
            use_colors: Whether to use ANSI colors.
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record for console display.

        Args:
            record: Log record to format.

        Returns:
            Formatted string.
        """
        parts = []

        if self.include_timestamp:
            timestamp = datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).strftime("%Y-%m-%d %H:%M:%S")
            parts.append(f"[{timestamp}]")

        # Level with color
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            reset = self.COLORS["RESET"]
            parts.append(f"{color}{level:8s}{reset}")
        else:
            parts.append(f"{level:8s}")

        # Logger name
        parts.append(f"[{record.name}]")

        # Message
        parts.append(record.getMessage())

        # Extra context
        extras = []
        for key in dir(record):
            if (
                not key.startswith("_")
                and key not in self._get_default_keys()
                and not callable(getattr(record, key))
            ):
                extras.append(f"{key}={getattr(record, key)}")

        if extras:
            parts.append(f"({', '.join(extras)})")

        # Exception
        if record.exc_info:
            parts.append("\n" + self.formatException(record.exc_info))

        return " ".join(parts)

    def _get_default_keys(self) -> set[str]:
        """Get default LogRecord attribute names."""
        return {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "message", "asctime", "taskName",
        }


class BoundLogger:
    """Logger wrapper with context binding support.

    This class wraps a standard logger and provides methods for binding
    context that persists across log calls.
    """

    def __init__(
        self,
        logger: logging.Logger,
        context_filter: ContextFilter | None = None,
    ) -> None:
        """Initialize the bound logger.

        Args:
            logger: Underlying logger.
            context_filter: Context filter for binding.
        """
        self._logger = logger
        self._context_filter = context_filter
        self._bound_context: dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> BoundLogger:
        """Create a new logger with additional context.

        Args:
            **kwargs: Context key-value pairs.

        Returns:
            New BoundLogger with merged context.
        """
        new_logger = BoundLogger(self._logger, self._context_filter)
        new_logger._bound_context = {**self._bound_context, **kwargs}
        return new_logger

    def unbind(self, *keys: str) -> BoundLogger:
        """Create a new logger without specified context.

        Args:
            *keys: Context keys to remove.

        Returns:
            New BoundLogger with context removed.
        """
        new_logger = BoundLogger(self._logger, self._context_filter)
        new_logger._bound_context = {
            k: v for k, v in self._bound_context.items()
            if k not in keys
        }
        return new_logger

    def _log(
        self,
        level: int,
        msg: str,
        *args: Any,
        exc_info: Any = None,
        **kwargs: Any,
    ) -> None:
        """Internal logging method.

        Args:
            level: Log level.
            msg: Log message.
            *args: Message format arguments.
            exc_info: Exception info.
            **kwargs: Extra context.
        """
        extra = {**self._bound_context, **kwargs}
        self._logger.log(level, msg, *args, exc_info=exc_info, extra=extra)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log at ERROR level with exception info."""
        self._log(logging.ERROR, msg, *args, exc_info=True, **kwargs)


# Module-level state
_configured = False
_context_filter: ContextFilter | None = None
_loggers: dict[str, BoundLogger] = {}


def configure_logging(
    level: str | LogLevel = LogLevel.INFO,
    format: str | LogFormat = LogFormat.CONSOLE,
    output: str | TextIO = "stdout",
    config: LogConfig | None = None,
) -> None:
    """Configure the logging system.

    Args:
        level: Minimum log level.
        format: Output format.
        output: Output destination.
        config: Full configuration (overrides other args).
    """
    global _configured, _context_filter

    if config:
        log_config = config
    else:
        log_config = LogConfig(
            level=LogLevel.from_string(level) if isinstance(level, str) else level,
            format=LogFormat(format) if isinstance(format, str) else format,
            output=output if isinstance(output, str) else "stdout",
        )

    # Get root logger
    root = logging.getLogger()
    root.setLevel(log_config.level.value)

    # Clear existing handlers
    root.handlers.clear()

    # Create handler
    if log_config.output == "stdout":
        handler: logging.Handler = logging.StreamHandler(sys.stdout)
    elif log_config.output == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(log_config.output)

    handler.setLevel(log_config.level.value)

    # Set formatter
    if log_config.format == LogFormat.JSON:
        formatter: logging.Formatter = JsonFormatter(
            include_timestamp=log_config.include_timestamp,
            include_caller=log_config.include_caller,
        )
    else:
        formatter = ConsoleFormatter(
            include_timestamp=log_config.include_timestamp,
            use_colors=log_config.format == LogFormat.CONSOLE,
        )

    handler.setFormatter(formatter)

    # Add filters
    _context_filter = ContextFilter(log_config.context_fields)
    handler.addFilter(_context_filter)

    if log_config.sensitive_fields:
        handler.addFilter(SensitiveDataFilter(log_config.sensitive_fields))

    root.addHandler(handler)
    _configured = True

    # Register shutdown
    atexit.register(logging.shutdown)


def get_logger(name: str) -> BoundLogger:
    """Get a bound logger for a module.

    Args:
        name: Logger name (usually __name__).

    Returns:
        BoundLogger instance.
    """
    if not _configured:
        configure_logging()

    if name not in _loggers:
        _loggers[name] = BoundLogger(
            logging.getLogger(name),
            _context_filter,
        )

    return _loggers[name]


@contextmanager
def log_context(**kwargs: Any) -> Generator[None, None, None]:
    """Context manager for temporary log context.

    Args:
        **kwargs: Context key-value pairs.

    Yields:
        None.
    """
    if _context_filter:
        _context_filter.bind(**kwargs)
        try:
            yield
        finally:
            _context_filter.unbind(*kwargs.keys())
    else:
        yield


def log_timing(
    logger: BoundLogger | None = None,
    level: int = logging.DEBUG,
    message: str = "Operation completed",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to log function execution time.

    Args:
        logger: Logger to use (default: module logger).
        level: Log level for timing message.
        message: Log message template.

    Returns:
        Decorator function.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            log = logger or get_logger(func.__module__)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                log._log(
                    level,
                    message,
                    function=func.__name__,
                    duration_ms=duration * 1000,
                )
                return result
            except Exception:
                duration = time.perf_counter() - start
                log._log(
                    logging.ERROR,
                    f"{message} (failed)",
                    function=func.__name__,
                    duration_ms=duration * 1000,
                )
                raise

        return wrapper
    return decorator


class DebugHelper:
    """Helper class for debugging operations.

    Provides utilities for debugging including state inspection,
    call tracing, and breakpoint management.
    """

    def __init__(self, logger: BoundLogger | None = None) -> None:
        """Initialize the debug helper.

        Args:
            logger: Logger for debug output.
        """
        self._logger = logger or get_logger(__name__)
        self._breakpoints: set[str] = set()

    def inspect(self, obj: Any, name: str = "object") -> None:
        """Log detailed inspection of an object.

        Args:
            obj: Object to inspect.
            name: Name for the object in logs.
        """
        self._logger.debug(
            f"Inspecting {name}",
            type=type(obj).__name__,
            value=repr(obj)[:200],
            attributes=dir(obj),
        )

    def trace_call(
        self,
        func_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        """Log a function call for tracing.

        Args:
            func_name: Function name.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        self._logger.debug(
            f"Calling {func_name}",
            call_args=repr(args)[:100],
            call_kwargs=repr(kwargs)[:100],
        )

    def set_breakpoint(self, name: str) -> None:
        """Set a named breakpoint.

        Args:
            name: Breakpoint name.
        """
        self._breakpoints.add(name)
        self._logger.debug(f"Breakpoint set: {name}")

    def check_breakpoint(self, name: str) -> bool:
        """Check if a breakpoint is set.

        Args:
            name: Breakpoint name.

        Returns:
            True if breakpoint is set.
        """
        if name in self._breakpoints:
            self._logger.info(f"Hit breakpoint: {name}")
            return True
        return False

    def clear_breakpoint(self, name: str) -> None:
        """Clear a breakpoint.

        Args:
            name: Breakpoint name.
        """
        self._breakpoints.discard(name)


# Convenience function for quick debug logging
def debug_log(msg: str, **kwargs: Any) -> None:
    """Quick debug log with context.

    Args:
        msg: Log message.
        **kwargs: Context key-value pairs.
    """
    get_logger("debug").debug(msg, **kwargs)
