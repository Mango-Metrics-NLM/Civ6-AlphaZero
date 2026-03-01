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

import atexit
import contextlib
import functools
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import numpy as np

from alpha_zero.utils.compat import UTC

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
            "timestamp": datetime.now(UTC).isoformat(),
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

    COLORS: ClassVar[dict[str, str]] = {
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
        kwargs: dict[str, Any],  # type: ignore[override]
    ) -> tuple[str, dict[str, Any]]:
        """Process log message to include training context."""
        extra = kwargs.get("extra", {})
        extra.update(
            {
                "training_step": self._step,
                "training_epoch": self._epoch,
                **self.extra,  # type: ignore[dict-item]
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
    *,
    config: LogConfig | None = None,
    console_output: bool | None = None,
    file_output: bool | None = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        enable_file: Whether to write logs to file.
        log_dir: Directory for log files.
        log_filename: Custom log filename (auto-generated if None).
        enable_json: Use JSON formatting for file logs.
        enable_colors: Use colored console output.
        config: Optional LogConfig object (overrides other params).
        console_output: Deprecated alias for enable_colors.
        file_output: Deprecated alias for enable_file.
    """
    # Support LogConfig object
    if config is not None:
        level = config.level
        log_dir = config.log_dir
        enable_file = config.file_output
        enable_json = config.json_format
        enable_colors = config.console_output

    # Support deprecated kwargs
    if console_output is not None:
        enable_colors = console_output
    if file_output is not None:
        enable_file = file_output
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


# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None  # type: ignore[misc, assignment]

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])


class LogLevel(Enum):
    """Log level enumeration."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogConfig:
    """Configuration for logging system.

    Attributes:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files.
        console_output: Whether to output to console.
        file_output: Whether to output to file.
        json_format: Whether to use JSON format for structured logging.
        max_file_size: Maximum size of each log file in bytes (default 10MB).
        backup_count: Number of backup files to keep.
        include_timestamp: Whether to include timestamps in log messages.
        tensorboard: Whether to enable TensorBoard logging.
        tensorboard_dir: Directory for TensorBoard logs.
    """

    level: str = "INFO"
    log_dir: str = "logs"
    console_output: bool = True
    file_output: bool = True
    json_format: bool = False
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    include_timestamp: bool = True
    tensorboard: bool = False
    tensorboard_dir: str = "logs/tensorboard"


class CircularBuffer:
    """Thread-safe circular buffer for recent log storage.

    Provides memory-efficient storage of recent log entries.
    """

    def __init__(self, maxsize: int = 1000) -> None:
        """Initialize circular buffer.

        Args:
            maxsize: Maximum number of entries to store.
        """
        self._buffer: deque[dict[str, Any]] = deque(maxlen=maxsize)
        self._lock = threading.Lock()

    def append(self, item: dict[str, Any]) -> None:
        """Append item to buffer.

        Args:
            item: Log entry to append.
        """
        with self._lock:
            self._buffer.append(item)

    def get_recent(self, n: int | None = None) -> list[dict[str, Any]]:
        """Get recent entries from buffer.

        Args:
            n: Number of entries to retrieve. If None, returns all.

        Returns:
            List of recent log entries.
        """
        with self._lock:
            if n is None:
                return list(self._buffer)
            return list(self._buffer)[-n:]

    def clear(self) -> None:
        """Clear all entries from buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


class MetricsTracker:
    """Thread-safe metrics tracker for aggregating and reporting training metrics.

    Supports running averages, min/max tracking, and histogram statistics.

    Example:
        >>> tracker = MetricsTracker()
        >>> tracker.update("loss", 0.5)
        >>> tracker.update("loss", 0.4)
        >>> print(tracker.get_summary("loss"))
        {'mean': 0.45, 'std': 0.05, 'min': 0.4, 'max': 0.5, 'count': 2}
    """

    def __init__(self, window_size: int = 100) -> None:
        """Initialize metrics tracker.

        Args:
            window_size: Size of sliding window for running statistics.
        """
        self._window_size = window_size
        self._metrics: dict[str, deque[float]] = {}
        self._all_values: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def update(self, name: str, value: float) -> None:
        """Update metric with new value.

        Args:
            name: Metric name.
            value: Metric value.
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=self._window_size)
                self._all_values[name] = []

            self._metrics[name].append(value)
            self._all_values[name].append(value)

    def update_batch(self, metrics: dict[str, float]) -> None:
        """Update multiple metrics at once.

        Args:
            metrics: Dictionary of metric name to value.
        """
        for name, value in metrics.items():
            self.update(name, value)

    def get_summary(self, name: str) -> dict[str, float]:
        """Get summary statistics for a metric.

        Args:
            name: Metric name.

        Returns:
            Dictionary with mean, std, min, max, count, and last values.
        """
        with self._lock:
            if name not in self._metrics or len(self._metrics[name]) == 0:
                return {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0,
                    "last": 0.0,
                }

            values = list(self._metrics[name])
            arr = np.array(values)

            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(values),
                "last": values[-1],
            }

    def get_running_average(self, name: str) -> float:
        """Get running average for a metric.

        Args:
            name: Metric name.

        Returns:
            Running average value.
        """
        summary = self.get_summary(name)
        return summary["mean"]

    def get_all_metrics(self) -> dict[str, dict[str, float]]:
        """Get summaries for all tracked metrics.

        Returns:
            Dictionary mapping metric names to their summaries.
        """
        with self._lock:
            names = list(self._metrics.keys())

        return {name: self.get_summary(name) for name in names}

    def get_history(self, name: str) -> list[float]:
        """Get full history for a metric.

        Args:
            name: Metric name.

        Returns:
            List of all recorded values.
        """
        with self._lock:
            return list(self._all_values.get(name, []))

    def reset(self, name: str | None = None) -> None:
        """Reset metrics.

        Args:
            name: Metric name to reset. If None, resets all metrics.
        """
        with self._lock:
            if name is None:
                self._metrics.clear()
                self._all_values.clear()
            elif name in self._metrics:
                self._metrics[name].clear()
                self._all_values[name].clear()

    def to_dict(self) -> dict[str, Any]:
        """Export all metrics to dictionary.

        Returns:
            Dictionary with all metric histories and summaries.
        """
        result: dict[str, Any] = {}
        with self._lock:
            for name in self._metrics:
                result[name] = {
                    "history": list(self._all_values[name]),
                    "summary": self.get_summary(name),
                }
        return result


class TrainingLogger:
    """Logger specialized for tracking training progress.

    Provides structured logging of training metrics, with optional TensorBoard
    integration for visualization.

    Example:
        >>> logger = TrainingLogger(log_dir="logs/training")
        >>> logger.log_training_step(
        ...     iteration=100,
        ...     loss=0.5,
        ...     policy_loss=0.3,
        ...     value_loss=0.2,
        ...     learning_rate=0.001,
        ... )
        >>> logger.log_evaluation(iteration=100, win_rate=0.65, games_played=50)
    """

    def __init__(
        self,
        log_dir: str = "logs/training",
        name: str = "training",
        tensorboard: bool = True,
        json_log: bool = True,
    ) -> None:
        """Initialize training logger.

        Args:
            log_dir: Directory for log files.
            name: Logger name.
            tensorboard: Whether to enable TensorBoard logging.
            json_log: Whether to write JSON log files.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.name = name
        self._logger = get_logger(f"training.{name}")
        self._metrics = MetricsTracker()
        self._start_time = time.time()
        self._step = 0

        # TensorBoard writer
        self._tb_writer: SummaryWriter | None = None
        if tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = self.log_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
        elif tensorboard and not TENSORBOARD_AVAILABLE:
            self._logger.warning(
                "TensorBoard requested but not available. Install with: pip install tensorboard"
            )

        # JSON log file
        self._json_log_path: Path | None = None
        self._json_entries: list[dict[str, Any]] = []
        if json_log:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._json_log_path = self.log_dir / f"training_{timestamp}.jsonl"

        # Register cleanup on exit
        atexit.register(self.close)

    def log_training_step(
        self,
        iteration: int,
        loss: float,
        policy_loss: float,
        value_loss: float,
        learning_rate: float | None = None,
        gradient_norm: float | None = None,
        batch_size: int | None = None,
        **extra_metrics: float,
    ) -> None:
        """Log a training step.

        Args:
            iteration: Current training iteration.
            loss: Total loss.
            policy_loss: Policy loss component.
            value_loss: Value loss component.
            learning_rate: Current learning rate.
            gradient_norm: Gradient norm (if clipping).
            batch_size: Batch size used.
            **extra_metrics: Additional metrics to log.
        """
        self._step = iteration
        elapsed = time.time() - self._start_time

        # Update metrics tracker
        self._metrics.update("loss", loss)
        self._metrics.update("policy_loss", policy_loss)
        self._metrics.update("value_loss", value_loss)

        # Build log entry
        entry: dict[str, Any] = {
            "type": "training_step",
            "iteration": iteration,
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

        if learning_rate is not None:
            entry["learning_rate"] = learning_rate
            self._metrics.update("learning_rate", learning_rate)

        if gradient_norm is not None:
            entry["gradient_norm"] = gradient_norm
            self._metrics.update("gradient_norm", gradient_norm)

        if batch_size is not None:
            entry["batch_size"] = batch_size

        for key, value in extra_metrics.items():
            entry[key] = value
            self._metrics.update(key, value)

        # Log to standard logger
        self._logger.info(
            f"Step {iteration}: loss={loss:.4f} (policy={policy_loss:.4f}, value={value_loss:.4f})",
        )

        # Log to TensorBoard
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("Loss/total", loss, iteration)
            self._tb_writer.add_scalar("Loss/policy", policy_loss, iteration)
            self._tb_writer.add_scalar("Loss/value", value_loss, iteration)

            if learning_rate is not None:
                self._tb_writer.add_scalar("Training/learning_rate", learning_rate, iteration)
            if gradient_norm is not None:
                self._tb_writer.add_scalar("Training/gradient_norm", gradient_norm, iteration)

            for key, value in extra_metrics.items():
                self._tb_writer.add_scalar(f"Metrics/{key}", value, iteration)

        # Write to JSON log
        self._write_json_entry(entry)

    def log_selfplay(
        self,
        iteration: int,
        num_games: int,
        total_examples: int,
        avg_game_length: float,
        time_elapsed: float,
        **extra_metrics: float,
    ) -> None:
        """Log self-play statistics.

        Args:
            iteration: Current iteration.
            num_games: Number of games played.
            total_examples: Total training examples generated.
            avg_game_length: Average game length in turns.
            time_elapsed: Time taken for self-play.
            **extra_metrics: Additional metrics.
        """
        entry: dict[str, Any] = {
            "type": "selfplay",
            "iteration": iteration,
            "num_games": num_games,
            "total_examples": total_examples,
            "avg_game_length": avg_game_length,
            "time_elapsed": time_elapsed,
            "games_per_second": num_games / time_elapsed if time_elapsed > 0 else 0,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

        entry.update(extra_metrics)

        self._metrics.update("games_per_second", entry["games_per_second"])
        self._metrics.update("avg_game_length", avg_game_length)

        self._logger.info(
            f"Self-play: {num_games} games, {total_examples} examples, "
            f"avg_len={avg_game_length:.1f}, time={time_elapsed:.1f}s",
        )

        if self._tb_writer is not None:
            self._tb_writer.add_scalar("SelfPlay/num_games", num_games, iteration)
            self._tb_writer.add_scalar("SelfPlay/examples", total_examples, iteration)
            self._tb_writer.add_scalar("SelfPlay/avg_game_length", avg_game_length, iteration)
            self._tb_writer.add_scalar(
                "SelfPlay/games_per_second", entry["games_per_second"], iteration
            )

        self._write_json_entry(entry)

    def log_evaluation(
        self,
        iteration: int,
        win_rate: float,
        games_played: int,
        opponent: str = "random",
        **extra_metrics: float,
    ) -> None:
        """Log evaluation results.

        Args:
            iteration: Current iteration.
            win_rate: Win rate against opponent.
            games_played: Number of evaluation games.
            opponent: Opponent type (random, previous, etc.).
            **extra_metrics: Additional metrics.
        """
        entry: dict[str, Any] = {
            "type": "evaluation",
            "iteration": iteration,
            "win_rate": win_rate,
            "games_played": games_played,
            "opponent": opponent,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

        entry.update(extra_metrics)

        self._metrics.update(f"win_rate_{opponent}", win_rate)

        self._logger.info(
            f"Evaluation vs {opponent}: win_rate={win_rate:.1%} ({games_played} games)",
        )

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(f"Evaluation/win_rate_{opponent}", win_rate, iteration)

        self._write_json_entry(entry)

    def log_checkpoint(
        self,
        iteration: int,
        checkpoint_path: str,
        metrics_summary: dict[str, float] | None = None,
    ) -> None:
        """Log checkpoint save.

        Args:
            iteration: Current iteration.
            checkpoint_path: Path to saved checkpoint.
            metrics_summary: Summary of current metrics.
        """
        entry: dict[str, Any] = {
            "type": "checkpoint",
            "iteration": iteration,
            "checkpoint_path": checkpoint_path,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

        if metrics_summary:
            entry["metrics"] = metrics_summary

        self._logger.info(f"Checkpoint saved: {checkpoint_path}")
        self._write_json_entry(entry)

    def get_metrics_summary(self) -> dict[str, dict[str, float]]:
        """Get summary of all tracked metrics.

        Returns:
            Dictionary of metric summaries.
        """
        return self._metrics.get_all_metrics()

    def _write_json_entry(self, entry: dict[str, Any]) -> None:
        """Write entry to JSON log file.

        Args:
            entry: Log entry to write.
        """
        if self._json_log_path is not None:
            with open(self._json_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def close(self) -> None:
        """Close logger and flush all data."""
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None


class GameLogger:
    """Logger for recording game states for MCTS debugging.

    Captures detailed information about game states, MCTS search, and decisions
    for analysis and debugging.

    Example:
        >>> game_logger = GameLogger(log_dir="logs/games")
        >>> game_logger.start_game(game_id=1)
        >>> game_logger.log_state(turn=1, state=state_dict, action=5, mcts_info=mcts_dict)
        >>> game_logger.end_game(winner=0, total_turns=50)
    """

    def __init__(
        self,
        log_dir: str = "logs/games",
        max_games_in_memory: int = 100,
        save_full_states: bool = False,
    ) -> None:
        """Initialize game logger.

        Args:
            log_dir: Directory for game logs.
            max_games_in_memory: Maximum games to keep in memory buffer.
            save_full_states: Whether to save full state tensors (memory intensive).
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._logger = get_logger("game_logger")
        self._current_game: dict[str, Any] | None = None
        self._game_buffer = CircularBuffer(max_games_in_memory)
        self._save_full_states = save_full_states
        self._lock = threading.Lock()

    def start_game(
        self,
        game_id: int | str,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Start logging a new game.

        Args:
            game_id: Unique game identifier.
            config: Game configuration.
            seed: Random seed used.
        """
        with self._lock:
            self._current_game = {
                "game_id": game_id,
                "config": config or {},
                "seed": seed,
                "start_time": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "turns": [],
                "metadata": {},
            }

        self._logger.debug(f"Started game {game_id}")

    def log_state(
        self,
        turn: int,
        player: int,
        action: int,
        action_probs: np.ndarray | None = None,
        value_estimate: float | None = None,
        valid_actions: np.ndarray | None = None,
        mcts_stats: dict[str, Any] | None = None,
        state_features: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """Log a game state and action.

        Args:
            turn: Current turn number.
            player: Current player.
            action: Action taken.
            action_probs: MCTS action probabilities.
            value_estimate: Value estimate from network/MCTS.
            valid_actions: Mask of valid actions.
            mcts_stats: Statistics from MCTS search.
            state_features: Optional state features for debugging.
            **extra: Additional information to log.
        """
        with self._lock:
            if self._current_game is None:
                self._logger.warning("log_state called without active game")
                return

            turn_data: dict[str, Any] = {
                "turn": turn,
                "player": player,
                "action": int(action),
                "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }

            if action_probs is not None:
                # Store top-k actions for efficiency
                top_k = min(10, len(action_probs))
                top_indices = np.argsort(action_probs)[-top_k:][::-1]
                turn_data["top_actions"] = [
                    {"action": int(idx), "prob": float(action_probs[idx])} for idx in top_indices
                ]
                turn_data["action_prob"] = float(action_probs[action])

            if value_estimate is not None:
                turn_data["value_estimate"] = float(value_estimate)

            if valid_actions is not None:
                turn_data["num_valid_actions"] = int(valid_actions.sum())

            if mcts_stats is not None:
                turn_data["mcts_stats"] = mcts_stats

            if state_features is not None and self._save_full_states:
                turn_data["state_features"] = state_features

            turn_data.update(extra)

            self._current_game["turns"].append(turn_data)

    def log_mcts_search(
        self,
        turn: int,
        root_visits: int,
        max_depth: int,
        search_time: float,
        nodes_expanded: int,
        q_values: dict[int, float] | None = None,
    ) -> None:
        """Log detailed MCTS search statistics.

        Args:
            turn: Current turn.
            root_visits: Total visits to root node.
            max_depth: Maximum search depth reached.
            search_time: Time spent searching.
            nodes_expanded: Number of nodes expanded.
            q_values: Q-values for top actions.
        """
        with self._lock:
            if self._current_game is None:
                return

            mcts_data: dict[str, Any] = {
                "root_visits": root_visits,
                "max_depth": max_depth,
                "search_time": search_time,
                "nodes_expanded": nodes_expanded,
                "visits_per_second": root_visits / search_time if search_time > 0 else 0,
            }

            if q_values is not None:
                mcts_data["q_values"] = {str(k): v for k, v in q_values.items()}

            # Update the last turn entry with MCTS data
            if self._current_game["turns"]:
                self._current_game["turns"][-1]["mcts_search"] = mcts_data

    def end_game(
        self,
        winner: int | None,
        total_turns: int,
        reason: str = "normal",
        **extra: Any,
    ) -> dict[str, Any]:
        """End game logging and return game data.

        Args:
            winner: Winner player index (None for draw).
            total_turns: Total turns played.
            reason: Reason for game end (normal, timeout, elimination, etc.).
            **extra: Additional end-game information.

        Returns:
            Complete game log data.
        """
        with self._lock:
            if self._current_game is None:
                self._logger.warning("end_game called without active game")
                return {}

            self._current_game["end_time"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            self._current_game["winner"] = winner
            self._current_game["total_turns"] = total_turns
            self._current_game["end_reason"] = reason
            self._current_game.update(extra)

            game_data = self._current_game
            self._game_buffer.append(game_data)
            self._current_game = None

        self._logger.debug(
            f"Game {game_data['game_id']} ended: winner={winner}, turns={total_turns}",
        )

        return game_data

    def save_game(self, game_data: dict[str, Any], filename: str | None = None) -> Path:
        """Save game log to file.

        Args:
            game_data: Game data to save.
            filename: Optional filename (auto-generated if not provided).

        Returns:
            Path to saved file.
        """
        if filename is None:
            game_id = game_data.get("game_id", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_{game_id}_{timestamp}.json"

        filepath = self.log_dir / filename

        with open(filepath, "w") as f:
            json.dump(game_data, f, indent=2, default=str)

        self._logger.debug(f"Saved game log to {filepath}")
        return filepath

    def get_recent_games(self, n: int | None = None) -> list[dict[str, Any]]:
        """Get recent games from buffer.

        Args:
            n: Number of games to retrieve.

        Returns:
            List of recent game logs.
        """
        return self._game_buffer.get_recent(n)

    def analyze_game(self, game_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze a game log for insights.

        Args:
            game_data: Game data to analyze.

        Returns:
            Analysis results including statistics and patterns.
        """
        turns = game_data.get("turns", [])

        if not turns:
            return {"error": "No turns recorded"}

        # Collect statistics
        value_estimates = [t.get("value_estimate", 0) for t in turns if "value_estimate" in t]
        action_probs = [t.get("action_prob", 0) for t in turns if "action_prob" in t]

        analysis: dict[str, Any] = {
            "total_turns": len(turns),
            "winner": game_data.get("winner"),
        }

        if value_estimates:
            analysis["value_estimates"] = {
                "mean": float(np.mean(value_estimates)),
                "std": float(np.std(value_estimates)),
                "min": float(np.min(value_estimates)),
                "max": float(np.max(value_estimates)),
            }

        if action_probs:
            analysis["action_confidence"] = {
                "mean": float(np.mean(action_probs)),
                "std": float(np.std(action_probs)),
                "min": float(np.min(action_probs)),
                "max": float(np.max(action_probs)),
            }

        # Analyze MCTS if available
        mcts_times = [
            t.get("mcts_search", {}).get("search_time", 0) for t in turns if "mcts_search" in t
        ]
        if mcts_times:
            analysis["mcts_time"] = {
                "mean": float(np.mean(mcts_times)),
                "total": float(np.sum(mcts_times)),
            }

        return analysis


class PerformanceProfiler:
    """Performance profiler for timing code sections.

    Provides context managers and decorators for measuring execution time
    with aggregation statistics.

    Example:
        >>> profiler = PerformanceProfiler()
        >>> with profiler.section("forward_pass"):
        ...     result = model(x)
        >>> print(profiler.get_summary())
    """

    def __init__(self) -> None:
        """Initialize profiler."""
        self._timings: dict[str, list[float]] = {}
        self._active_sections: dict[str, float] = {}
        self._lock = threading.Lock()
        self._logger: logging.Logger | None = None

    def _get_logger(self) -> logging.Logger:
        """Get logger instance (lazy initialization).

        Returns:
            Logger instance for profiler.
        """
        if self._logger is None:
            self._logger = get_logger("profiler")
        return self._logger

    @contextlib.contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Context manager for timing a code section.

        Args:
            name: Section name for identification.

        Yields:
            None (timing is automatic).
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            with self._lock:
                if name not in self._timings:
                    self._timings[name] = []
                self._timings[name].append(elapsed)

    def start(self, name: str) -> None:
        """Start timing a section (manual timing).

        Args:
            name: Section name.
        """
        with self._lock:
            self._active_sections[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing a section.

        Args:
            name: Section name.

        Returns:
            Elapsed time in seconds.
        """
        end_time = time.perf_counter()
        with self._lock:
            if name not in self._active_sections:
                self._get_logger().warning(f"Section '{name}' was not started")
                return 0.0

            elapsed = end_time - self._active_sections.pop(name)

            if name not in self._timings:
                self._timings[name] = []
            self._timings[name].append(elapsed)

            return elapsed

    def get_summary(self, name: str | None = None) -> dict[str, Any]:
        """Get timing summary.

        Args:
            name: Section name (or None for all sections).

        Returns:
            Dictionary with timing statistics.
        """
        with self._lock:
            if name is not None:
                if name not in self._timings:
                    return {}
                timings = self._timings[name]
                return self._compute_stats(name, timings)

            return {
                name: self._compute_stats(name, timings) for name, timings in self._timings.items()
            }

    def _compute_stats(self, name: str, timings: list[float]) -> dict[str, float]:
        """Compute statistics for timing data.

        Args:
            name: Section name.
            timings: List of timing values.

        Returns:
            Statistics dictionary.
        """
        if not timings:
            return {"count": 0}

        arr = np.array(timings)
        return {
            "count": len(timings),
            "total": float(np.sum(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    def reset(self, name: str | None = None) -> None:
        """Reset timing data.

        Args:
            name: Section to reset (or None for all).
        """
        with self._lock:
            if name is None:
                self._timings.clear()
            elif name in self._timings:
                del self._timings[name]

    def report(self) -> str:
        """Generate a formatted timing report.

        Returns:
            Formatted string report.
        """
        summary = self.get_summary()

        if not summary:
            return "No timing data recorded."

        lines = ["Performance Report", "=" * 60]

        # Sort by total time
        sorted_sections = sorted(
            summary.items(),
            key=lambda x: x[1].get("total", 0),
            reverse=True,
        )

        for name, stats in sorted_sections:
            lines.append(f"\n{name}:")
            lines.append(f"  Calls: {stats.get('count', 0)}")
            lines.append(f"  Total: {stats.get('total', 0):.4f}s")
            lines.append(f"  Mean:  {stats.get('mean', 0):.4f}s")
            lines.append(f"  Std:   {stats.get('std', 0):.4f}s")
            lines.append(f"  Min:   {stats.get('min', 0):.4f}s")
            lines.append(f"  Max:   {stats.get('max', 0):.4f}s")

        return "\n".join(lines)


# Global profiler instance
_profiler = PerformanceProfiler()


def timer(name: str) -> Callable[[F], F]:
    """Decorator for timing function execution.

    Args:
        name: Name for the timing section.

    Returns:
        Decorated function.

    Example:
        >>> @timer("my_function")
        ... def my_function():
        ...     time.sleep(0.1)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _profiler.section(name):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


@contextlib.contextmanager
def timed_section(name: str) -> Iterator[dict[str, float]]:
    """Context manager for timing with result access.

    Args:
        name: Section name.

    Yields:
        Dictionary that will contain 'elapsed' key after context exits.

    Example:
        >>> with timed_section("my_operation") as timing:
        ...     do_something()
        >>> print(f"Took {timing['elapsed']:.3f}s")
    """
    timing: dict[str, float] = {}
    start_time = time.perf_counter()
    try:
        yield timing
    finally:
        timing["elapsed"] = time.perf_counter() - start_time
        with _profiler._lock:
            if name not in _profiler._timings:
                _profiler._timings[name] = []
            _profiler._timings[name].append(timing["elapsed"])


def get_profiler() -> PerformanceProfiler:
    """Get the global profiler instance.

    Returns:
        Global PerformanceProfiler instance.
    """
    return _profiler


def set_level(level: str, logger_name: str | None = None) -> None:
    """Set log level for a specific logger or all loggers.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        logger_name: Specific logger name, or None for root.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if logger_name is None:
        logging.getLogger().setLevel(log_level)
    else:
        logging.getLogger(logger_name).setLevel(log_level)


def configure_from_yaml(yaml_path: str | Path) -> None:
    """Configure logging from a YAML file.

    Args:
        yaml_path: Path to YAML configuration file.

    The YAML file should have a 'logging' section with LogConfig fields.

    Example YAML:
        logging:
          level: DEBUG
          log_dir: logs
          json_format: false
          tensorboard: true
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML required for YAML configuration: pip install pyyaml") from e

    yaml_path = Path(yaml_path)

    with open(yaml_path) as f:
        full_config = yaml.safe_load(f)

    log_config = full_config.get("logging", {})

    config = LogConfig(
        level=log_config.get("level", "INFO"),
        log_dir=log_config.get("log_dir", "logs"),
        console_output=log_config.get("console_output", True),
        file_output=log_config.get("file_output", True),
        json_format=log_config.get("json_format", False),
        max_file_size=log_config.get("max_file_size", 10 * 1024 * 1024),
        backup_count=log_config.get("backup_count", 5),
        tensorboard=log_config.get("tensorboard", False),
        tensorboard_dir=log_config.get("tensorboard_dir", "logs/tensorboard"),
    )

    setup_logging(
        level=config.level,
        enable_file=config.file_output,
        log_dir=config.log_dir,
        enable_json=config.json_format,
        enable_colors=config.console_output,
    )


# Convenience exports
__all__ = [
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_LEVEL",
    "TENSORBOARD_AVAILABLE",
    "CircularBuffer",
    "ColoredFormatter",
    "GameLogger",
    "LogConfig",
    "LogLevel",
    "MetricsTracker",
    "PerformanceProfiler",
    "StructuredFormatter",
    "TrainingLogger",
    "TrainingMetricsAdapter",
    "configure_from_yaml",
    "get_logger",
    "get_profiler",
    "get_training_logger",
    "set_level",
    "setup_logging",
    "timed_section",
    "timer",
]
