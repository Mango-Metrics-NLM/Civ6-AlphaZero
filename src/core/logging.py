"""Structured logging for the Strategic Planning SDK.

Provides a structlog-based logging system that wraps stdlib logging so that
existing ``logging.getLogger()`` call sites continue to work while gaining
structured context, correlation IDs, and metrics logging.

Example usage::

    >>> from src.core.logging import configure_logging, get_logger, bind_context
    >>> configure_logging(level="DEBUG", format_="pretty")
    >>> log = get_logger(__name__)
    >>> bind_context(environment_id="mini_resource", iteration=42)
    >>> log.info("training started", batch_size=256)

    >>> from src.core.logging import MetricsLogger
    >>> metrics = MetricsLogger("train", interval=10)
    >>> metrics.log_scalar("loss", 0.42, step=100)
"""

from __future__ import annotations

import logging
import logging.handlers
import uuid
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, cast

import structlog
from structlog.stdlib import BoundLogger

from src.core.constants import LOGFILE_BACKUP_COUNT, MAX_LOGFILE_BYTES

__all__ = [
    "MetricsLogger",
    "bind_context",
    "configure_logging",
    "generate_correlation_id",
    "get_logger",
]


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_configured: bool = False


# ---------------------------------------------------------------------------
# Correlation ID helpers
# ---------------------------------------------------------------------------


def generate_correlation_id() -> str:
    """Generate a new correlation ID.

    Returns:
        A 32-character lowercase hex string derived from a UUID4.

    Example::

        >>> cid = generate_correlation_id()
        >>> len(cid)
        32
    """
    return uuid.uuid4().hex


def _add_correlation_id(
    logger: logging.Logger,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Structlog processor that injects a correlation ID when enabled.

    If a ``correlation_id`` key is already present in the event dict (e.g.
    bound via :func:`bind_context`), it is left untouched.  Otherwise a new
    one is generated automatically.

    Args:
        logger: The wrapped logger instance (unused, required by protocol).
        method_name: The name of the log method called (unused).
        event_dict: The current structured event dictionary.

    Returns:
        The event dictionary with ``correlation_id`` guaranteed to be set.
    """
    if "correlation_id" not in event_dict:
        event_dict["correlation_id"] = generate_correlation_id()
    return event_dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    level: str = "INFO",
    format_: str = "pretty",
    log_dir: str | None = None,
    correlation_id_enabled: bool = True,
) -> None:
    """Configure the structured logging subsystem.

    Sets up structlog's processor chain and wires stdlib logging through
    structlog so that both ``structlog.get_logger()`` and the traditional
    ``logging.getLogger()`` produce structured output.

    This function is idempotent but **not** additive -- calling it a second
    time fully reconfigures the logging stack.

    Args:
        level: Root log level as an uppercase string (e.g. ``"DEBUG"``).
        format_: Output format.  ``"json"`` emits machine-readable JSON;
            ``"pretty"`` uses coloured console output suitable for
            development.
        log_dir: When set, a ``RotatingFileHandler`` is attached to the
            stdlib root logger writing JSON-formatted entries into this
            directory.  The directory is created if it does not exist.
        correlation_id_enabled: When ``True`` every log entry is tagged
            with a ``correlation_id``.  A new ID is generated per event
            unless one has been explicitly bound via :func:`bind_context`.

    Raises:
        ValueError: If *format_* is not ``"json"`` or ``"pretty"``.

    Example::

        >>> configure_logging(level="DEBUG", format_="json", log_dir="/tmp/logs")
    """
    global _configured

    if format_ not in ("json", "pretty"):
        raise ValueError(
            f"format_ must be 'json' or 'pretty', got {format_!r}"
        )

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")

    # -- Build the shared processor chain ------------------------------------
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if correlation_id_enabled:
        shared_processors.insert(1, _add_correlation_id)

    # -- Renderer selection --------------------------------------------------
    if format_ == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # -- Configure structlog -------------------------------------------------
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # -- Configure stdlib root logger ----------------------------------------
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    root_logger = logging.getLogger()
    # Remove any previously installed handlers to stay idempotent.
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)

    # Console handler (always present)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # File output always uses JSON for machine readability.
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        )

        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path / "sdk.log"),
            maxBytes=MAX_LOGFILE_BYTES,
            backupCount=LOGFILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> BoundLogger:
    """Obtain a named, bound structured logger.

    If :func:`configure_logging` has not been called yet, the logger will
    still work (structlog falls back to its defaults), but the output will
    lack the full processor chain.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A :class:`structlog.stdlib.BoundLogger` that supports ``.bind()``,
        ``.info()``, ``.debug()``, and all other standard log methods.

    Example::

        >>> log = get_logger("src.core.mcts")
        >>> log.info("search complete", nodes=1200, depth=8)
    """
    return cast(BoundLogger, structlog.get_logger(name))


def bind_context(**kwargs: Any) -> None:
    """Bind key/value pairs to the current contextvars context.

    Values bound here appear automatically in every subsequent log entry
    produced on the same async task / thread, until explicitly cleared via
    ``structlog.contextvars.clear_contextvars()``.

    Args:
        **kwargs: Arbitrary key/value pairs to attach to future log events.

    Example::

        >>> bind_context(environment_id="mini_resource", iteration=42)
    """
    structlog.contextvars.bind_contextvars(**kwargs)


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------


class MetricsLogger:
    """Lightweight metrics logger that throttles output by step interval.

    Designed for high-frequency training loops where logging every single
    step would be too noisy.  Only steps that are exact multiples of
    *interval* produce log output.

    Args:
        name: Logger name used for the underlying structlog logger.
        interval: Log only every *interval*-th step.  A value of ``1``
            means every step is logged.

    Example::

        >>> ml = MetricsLogger("training", interval=100)
        >>> for step in range(1000):
        ...     ml.log_scalar("loss", compute_loss(), step=step)
    """

    def __init__(self, name: str, interval: int = 1) -> None:
        if interval < 1:
            raise ValueError(
                f"interval must be >= 1, got {interval}"
            )
        self._logger: BoundLogger = get_logger(name)
        self._interval: int = interval

    # -- helpers -------------------------------------------------------------

    def _should_log(self, step: int) -> bool:
        """Return ``True`` when *step* falls on the logging interval.

        Args:
            step: The current training step.

        Returns:
            Whether the metric should be emitted.
        """
        return step % self._interval == 0

    # -- public interface ----------------------------------------------------

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a single scalar metric.

        The entry is silently dropped unless *step* is a multiple of the
        configured *interval*.

        Args:
            key: Metric name (e.g. ``"loss"``, ``"learning_rate"``).
            value: Scalar value to record.
            step: Global training step.
        """
        if self._should_log(step):
            self._logger.info("metric", key=key, value=value, step=step)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        """Log multiple scalar metrics in a single event.

        As with :meth:`log_scalar`, the entry is dropped when *step* does
        not align with the interval.

        Args:
            metrics: Mapping of metric names to values.
            step: Global training step.
        """
        if self._should_log(step):
            self._logger.info("metrics", step=step, **metrics)
