"""Centralized logging configuration for Civ6 AlphaZero.

Provides:
- Configurable log levels via environment
- JSON and console formatters
- File and cloud output handlers
- Debug mode support
- Structured logging for metrics

Usage:
    from alpha_zero.utils.logging_config import setup_logging, get_logger

    setup_logging(level="DEBUG", enable_file=True)
    logger = get_logger(__name__)
    logger.info("Training started", extra={"metrics": {"epoch": 1}})
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import timezone, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

# Default configuration from environment
DEFAULT_LOG_LEVEL = os.environ.get("CIV6AZ_LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_DIR = os.environ.get("LOG_DIR", "logs")
DEFAULT_LOG_FORMAT = os.environ.get(
    "CIV6AZ_LOG_FORMAT",
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location info
        if record.levelno >= logging.WARNING:
            log_data["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields (metrics, context, etc.)
        extra_keys = set(record.__dict__.keys()) - {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "message",
            "asctime",
        }
        for key in extra_keys:
            value = getattr(record, key, None)
            if value is not None:
                log_data[key] = value

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str | None = None, use_colors: bool = True) -> None:
        """Initialize formatter.

        Args:
            fmt: Log format string.
            use_colors: Whether to use ANSI colors.
        """
        super().__init__(fmt or DEFAULT_LOG_FORMAT)
        self.use_colors = use_colors and self._supports_colors()

    def _supports_colors(self) -> bool:
        """Check if terminal supports colors."""
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("FORCE_COLOR"):
            return True
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format with optional colors."""
        message = super().format(record)

        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            if color:
                # Color the level name
                message = message.replace(
                    f"[{record.levelname}]",
                    f"{color}[{record.levelname}]{self.RESET}",
                )

        return message


class TrainingMetricsAdapter(logging.LoggerAdapter):
    """Logger adapter that automatically includes training context."""

    def __init__(
        self,
        logger: logging.Logger,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Initialize adapter.

        Args:
            logger: Base logger.
            extra: Default extra fields.
        """
        super().__init__(logger, extra or {})
        self._step: int = 0
        self._epoch: int = 0

    def set_training_context(
        self,
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Update training context.

        Args:
            step: Current training step.
            epoch: Current epoch.
        """
        if step is not None:
            self._step = step
        if epoch is not None:
            self._epoch = epoch

    def process(
        self,
        msg: str,
        kwargs: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Process log message to include training context."""
        extra = kwargs.get("extra", {})
        extra.update(
            {
                "training_step": self._step,
                "training_epoch": self._epoch,
                **self.extra,
            }
        )
        kwargs["extra"] = extra
        return msg, kwargs

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, float],
        level: int = logging.INFO,
    ) -> None:
        """Log training metrics with structured format.

        Args:
            step: Training step.
            metrics: Dictionary of metric names to values.
            level: Log level.
        """
        self._step = step
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.log(level, f"Step {step}: {metrics_str}", extra={"metrics": metrics})


def setup_logging(
    level: str | int = DEFAULT_LOG_LEVEL,
    enable_file: bool = False,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    log_filename: str | None = None,
    enable_json: bool = False,
    enable_colors: bool = True,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        enable_file: Whether to write logs to file.
        log_dir: Directory for log files.
        log_filename: Custom log filename (auto-generated if None).
        enable_json: Use JSON formatting for file logs.
        enable_colors: Use colored console output.
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter(use_colors=enable_colors))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if enable_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"civ6az_{timestamp}.log"

        file_path = log_path / log_filename
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)

        if enable_json:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))

        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def get_training_logger(
    name: str,
    extra: dict[str, Any] | None = None,
) -> TrainingMetricsAdapter:
    """Get a training-aware logger adapter.

    Args:
        name: Logger name (typically __name__).
        extra: Default extra fields to include.

    Returns:
        TrainingMetricsAdapter instance.
    """
    base_logger = logging.getLogger(name)
    return TrainingMetricsAdapter(base_logger, extra)


# Convenience exports
__all__ = [
    "setup_logging",
    "get_logger",
    "get_training_logger",
    "StructuredFormatter",
    "ColoredFormatter",
    "TrainingMetricsAdapter",
    "DEFAULT_LOG_LEVEL",
    "DEFAULT_LOG_DIR",
]
