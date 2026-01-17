"""
Template Logging and Debugging for Agentic Coding System

Provides comprehensive logging and debugging utilities for template generation
and agent execution. All logging levels and formats are configurable.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator, Optional, TextIO

from ..utils.logging_config import get_logger

logger = get_logger("template_logger")


class LogLevel(Enum):
    """Log levels for template logging."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_string(cls, value: str) -> LogLevel:
        """Create from string.

        Args:
            value: String representation.

        Returns:
            LogLevel enum value.
        """
        return cls[value.upper()]

    def to_python_level(self) -> int:
        """Convert to Python logging level.

        Returns:
            Python logging level constant.
        """
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return level_map.get(self, logging.INFO)


@dataclass
class LogEntry:
    """A log entry with full context.

    Attributes:
        timestamp: When the entry was created
        level: Log level
        message: Log message
        source: Source of the log (module, function, etc.)
        context: Additional context data
        template_id: Associated template ID (if any)
        agent_name: Associated agent name (if any)
        step_name: Associated step name (if any)
        duration_ms: Duration in milliseconds (for timed operations)
    """

    timestamp: float
    level: LogLevel
    message: str
    source: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    template_id: Optional[str] = None
    agent_name: Optional[str] = None
    step_name: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "message": self.message,
            "source": self.source,
            "context": self.context,
            "template_id": self.template_id,
            "agent_name": self.agent_name,
            "step_name": self.step_name,
            "duration_ms": self.duration_ms,
        }

    def format(self, include_context: bool = False) -> str:
        """Format the log entry as a string.

        Args:
            include_context: Whether to include context data.

        Returns:
            Formatted string.
        """
        parts = [
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
            f"[{self.level.value}]",
        ]

        if self.source:
            parts.append(f"({self.source})")

        if self.agent_name:
            parts.append(f"[{self.agent_name}]")

        if self.step_name:
            parts.append(f"[{self.step_name}]")

        parts.append(self.message)

        if self.duration_ms is not None:
            parts.append(f"({self.duration_ms:.2f}ms)")

        result = " ".join(parts)

        if include_context and self.context:
            result += f"\n  Context: {json.dumps(self.context, indent=2)}"

        return result


@dataclass
class DebugContext:
    """Context for debugging operations.

    Thread-safe context that can be nested for tracing execution.

    Attributes:
        name: Context name
        parent: Parent context (for nesting)
        start_time: When context was entered
        metadata: Additional debug metadata
        entries: Log entries within this context
    """

    name: str
    parent: Optional[DebugContext] = None
    start_time: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    entries: list[LogEntry] = field(default_factory=list)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_entry(self, entry: LogEntry) -> None:
        """Add a log entry to this context.

        Args:
            entry: Entry to add.
        """
        with self._lock:
            self.entries.append(entry)

    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return (time.time() - self.start_time) * 1000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "name": self.name,
            "start_time": self.start_time,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "entries": [e.to_dict() for e in self.entries],
            "parent": self.parent.name if self.parent else None,
        }


class TemplateLogger:
    """Comprehensive logger for template generation and agent execution.

    Features:
    - Multiple output handlers (console, file, JSON)
    - Configurable log levels
    - Context-aware logging with nesting
    - Performance timing
    - Log aggregation and analysis
    - Thread-safe operation
    """

    def __init__(
        self,
        name: str = "template",
        level: LogLevel = LogLevel.INFO,
        console_output: bool = True,
        file_output: Optional[Path] = None,
        json_output: Optional[Path] = None,
        max_entries: int = 10000,
        format_string: Optional[str] = None,
    ) -> None:
        """Initialize the template logger.

        Args:
            name: Logger name.
            level: Minimum log level.
            console_output: Whether to output to console.
            file_output: Path for file output.
            json_output: Path for JSON output.
            max_entries: Maximum entries to keep in memory.
            format_string: Custom format string.
        """
        self.name = name
        self.level = level
        self._console_output = console_output
        self._file_output = file_output
        self._json_output = json_output
        self._max_entries = max_entries
        self._format_string = format_string or "{timestamp} [{level}] {message}"

        self._entries: deque[LogEntry] = deque(maxlen=max_entries)
        self._context_stack: list[DebugContext] = []
        self._lock = threading.Lock()

        # File handles
        self._file_handle: Optional[TextIO] = None
        self._json_handle: Optional[TextIO] = None

        # Callbacks
        self._on_log: Optional[Callable[[LogEntry], None]] = None
        self._on_error: Optional[Callable[[LogEntry], None]] = None

        # Initialize file outputs
        if self._file_output:
            self._file_handle = open(self._file_output, "a")
        if self._json_output:
            self._json_handle = open(self._json_output, "a")

    def set_level(self, level: LogLevel | str) -> None:
        """Set the minimum log level.

        Args:
            level: New log level.
        """
        if isinstance(level, str):
            level = LogLevel.from_string(level)
        self.level = level

    def set_callback(
        self,
        on_log: Optional[Callable[[LogEntry], None]] = None,
        on_error: Optional[Callable[[LogEntry], None]] = None,
    ) -> None:
        """Set logging callbacks.

        Args:
            on_log: Callback for all log entries.
            on_error: Callback for error entries.
        """
        self._on_log = on_log
        self._on_error = on_error

    def _should_log(self, level: LogLevel) -> bool:
        """Check if a level should be logged.

        Args:
            level: Level to check.

        Returns:
            True if level should be logged.
        """
        level_order = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]
        return level_order.index(level) >= level_order.index(self.level)

    def _create_entry(
        self,
        level: LogLevel,
        message: str,
        source: str = "",
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> LogEntry:
        """Create a log entry.

        Args:
            level: Log level.
            message: Log message.
            source: Source identifier.
            context: Additional context.
            **kwargs: Additional entry fields.

        Returns:
            LogEntry instance.
        """
        # Get context from stack
        current_context = self._context_stack[-1] if self._context_stack else None

        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            source=source or self.name,
            context=context or {},
            template_id=kwargs.get("template_id") or (current_context.metadata.get("template_id") if current_context else None),
            agent_name=kwargs.get("agent_name") or (current_context.metadata.get("agent_name") if current_context else None),
            step_name=kwargs.get("step_name") or (current_context.metadata.get("step_name") if current_context else None),
            duration_ms=kwargs.get("duration_ms"),
        )

        return entry

    def _write_entry(self, entry: LogEntry) -> None:
        """Write an entry to all outputs.

        Args:
            entry: Entry to write.
        """
        formatted = entry.format(include_context=self.level == LogLevel.DEBUG)

        # Console output
        if self._console_output:
            print(formatted, file=sys.stderr)

        # File output
        if self._file_handle:
            self._file_handle.write(formatted + "\n")
            self._file_handle.flush()

        # JSON output
        if self._json_handle:
            self._json_handle.write(json.dumps(entry.to_dict()) + "\n")
            self._json_handle.flush()

        # Callbacks
        if self._on_log:
            self._on_log(entry)

        if entry.level in (LogLevel.ERROR, LogLevel.CRITICAL) and self._on_error:
            self._on_error(entry)

    def _log(
        self,
        level: LogLevel,
        message: str,
        source: str = "",
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[LogEntry]:
        """Internal logging method.

        Args:
            level: Log level.
            message: Log message.
            source: Source identifier.
            context: Additional context.
            **kwargs: Additional entry fields.

        Returns:
            LogEntry if logged, None otherwise.
        """
        if not self._should_log(level):
            return None

        entry = self._create_entry(level, message, source, context, **kwargs)

        with self._lock:
            self._entries.append(entry)

            # Add to current context if available
            if self._context_stack:
                self._context_stack[-1].add_entry(entry)

        self._write_entry(entry)

        return entry

    def debug(self, message: str, **kwargs: Any) -> Optional[LogEntry]:
        """Log a debug message.

        Args:
            message: Log message.
            **kwargs: Additional arguments.

        Returns:
            LogEntry if logged.
        """
        return self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> Optional[LogEntry]:
        """Log an info message.

        Args:
            message: Log message.
            **kwargs: Additional arguments.

        Returns:
            LogEntry if logged.
        """
        return self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> Optional[LogEntry]:
        """Log a warning message.

        Args:
            message: Log message.
            **kwargs: Additional arguments.

        Returns:
            LogEntry if logged.
        """
        return self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> Optional[LogEntry]:
        """Log an error message.

        Args:
            message: Log message.
            **kwargs: Additional arguments.

        Returns:
            LogEntry if logged.
        """
        return self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> Optional[LogEntry]:
        """Log a critical message.

        Args:
            message: Log message.
            **kwargs: Additional arguments.

        Returns:
            LogEntry if logged.
        """
        return self._log(LogLevel.CRITICAL, message, **kwargs)

    @contextmanager
    def context(
        self,
        name: str,
        **metadata: Any,
    ) -> Generator[DebugContext, None, None]:
        """Enter a debug context.

        Args:
            name: Context name.
            **metadata: Context metadata.

        Yields:
            DebugContext instance.
        """
        parent = self._context_stack[-1] if self._context_stack else None
        ctx = DebugContext(name=name, parent=parent, metadata=metadata)

        self._context_stack.append(ctx)
        self.debug(f"Entering context: {name}", context=metadata)

        try:
            yield ctx
        finally:
            self._context_stack.pop()
            duration = ctx.duration_ms
            self.debug(
                f"Exiting context: {name}",
                duration_ms=duration,
                context={"entries_count": len(ctx.entries)},
            )

    @contextmanager
    def timed(
        self,
        operation: str,
        **kwargs: Any,
    ) -> Generator[None, None, None]:
        """Time an operation.

        Args:
            operation: Operation description.
            **kwargs: Additional log arguments.

        Yields:
            None.
        """
        start = time.time()
        self.debug(f"Starting: {operation}", **kwargs)

        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self.info(
                f"Completed: {operation}",
                duration_ms=duration_ms,
                **kwargs,
            )

    def get_entries(
        self,
        level: Optional[LogLevel] = None,
        source: Optional[str] = None,
        since: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[LogEntry]:
        """Get log entries with filters.

        Args:
            level: Filter by level.
            source: Filter by source.
            since: Filter entries after this timestamp.
            limit: Maximum entries to return.

        Returns:
            List of matching LogEntry objects.
        """
        entries = list(self._entries)

        if level:
            entries = [e for e in entries if e.level == level]

        if source:
            entries = [e for e in entries if e.source == source]

        if since:
            entries = [e for e in entries if e.timestamp >= since]

        if limit:
            entries = entries[-limit:]

        return entries

    def get_errors(self) -> list[LogEntry]:
        """Get all error and critical entries.

        Returns:
            List of error entries.
        """
        return [
            e for e in self._entries
            if e.level in (LogLevel.ERROR, LogLevel.CRITICAL)
        ]

    def summary(self) -> dict[str, Any]:
        """Generate a summary of logged entries.

        Returns:
            Summary dictionary.
        """
        by_level = {level.value: 0 for level in LogLevel}
        for entry in self._entries:
            by_level[entry.level.value] += 1

        return {
            "total_entries": len(self._entries),
            "by_level": by_level,
            "error_count": by_level[LogLevel.ERROR.value] + by_level[LogLevel.CRITICAL.value],
            "oldest_entry": self._entries[0].timestamp if self._entries else None,
            "newest_entry": self._entries[-1].timestamp if self._entries else None,
        }

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._entries.clear()

    def close(self) -> None:
        """Close file handles."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

        if self._json_handle:
            self._json_handle.close()
            self._json_handle = None

    def __enter__(self) -> TemplateLogger:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


# Global logger instance
_global_logger: Optional[TemplateLogger] = None


def setup_template_logging(
    name: str = "template",
    level: LogLevel | str = LogLevel.INFO,
    console_output: bool = True,
    file_output: Optional[Path] = None,
    json_output: Optional[Path] = None,
) -> TemplateLogger:
    """Set up global template logging.

    Args:
        name: Logger name.
        level: Minimum log level.
        console_output: Whether to output to console.
        file_output: Path for file output.
        json_output: Path for JSON output.

    Returns:
        Configured TemplateLogger.
    """
    global _global_logger

    if isinstance(level, str):
        level = LogLevel.from_string(level)

    _global_logger = TemplateLogger(
        name=name,
        level=level,
        console_output=console_output,
        file_output=file_output,
        json_output=json_output,
    )

    return _global_logger


def get_template_logger() -> TemplateLogger:
    """Get the global template logger.

    Returns:
        TemplateLogger instance.
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = TemplateLogger()

    return _global_logger
