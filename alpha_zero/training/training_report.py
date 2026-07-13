"""Training report module for periodic training summaries.

This module provides classes for tracking and generating periodic training reports that
summarize loss, gradient norms, learning rates, safeguard events, and custom metrics
over configurable intervals.
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from alpha_zero.utils.compat import UTC

logger = logging.getLogger(__name__)


@dataclass
class TrainingReportConfig:
    """Configuration for training report generation.

    Attributes:
        report_interval: Number of steps between report generation.
        include_safeguard_events: Whether to include safeguard events in reports.
        include_hyperparameters: Whether to include hyperparameters in reports.
        wandb_prefix: Prefix for W&B logging keys.
    """

    report_interval: int
    include_safeguard_events: bool = True
    include_hyperparameters: bool = True
    wandb_prefix: str = "training_report/"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.report_interval < 1:
            msg = f"report_interval must be >= 1, got {self.report_interval}"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "report_interval": self.report_interval,
            "include_safeguard_events": self.include_safeguard_events,
            "include_hyperparameters": self.include_hyperparameters,
            "wandb_prefix": self.wandb_prefix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingReportConfig:
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation of configuration.

        Returns:
            TrainingReportConfig instance.
        """
        return cls(
            report_interval=data["report_interval"],
            include_safeguard_events=data.get("include_safeguard_events", True),
            include_hyperparameters=data.get("include_hyperparameters", True),
            wandb_prefix=data.get("wandb_prefix", "training_report/"),
        )

    @classmethod
    def from_env(cls) -> TrainingReportConfig:
        """Create configuration from environment variables.

        Environment variables:
            CIV6AZ_REPORT_INTERVAL: Report generation interval (default: 100)
            CIV6AZ_REPORT_SAFEGUARD_EVENTS: Include safeguard events (default: True)
            CIV6AZ_REPORT_HYPERPARAMETERS: Include hyperparameters (default: True)
            CIV6AZ_REPORT_METRICS_PREFIX: metric-key prefix (default: training_report/;
                legacy alias CIV6AZ_REPORT_WANDB_PREFIX honored)

        Returns:
            TrainingReportConfig instance.
        """
        return cls(
            report_interval=int(os.environ.get("CIV6AZ_REPORT_INTERVAL", "100")),
            include_safeguard_events=(
                os.environ.get("CIV6AZ_REPORT_SAFEGUARD_EVENTS", "true").lower() == "true"
            ),
            include_hyperparameters=(
                os.environ.get("CIV6AZ_REPORT_HYPERPARAMETERS", "true").lower() == "true"
            ),
            wandb_prefix=os.environ.get(
                "CIV6AZ_REPORT_METRICS_PREFIX",
                os.environ.get("CIV6AZ_REPORT_WANDB_PREFIX", "training_report/"),
            ),
        )


@dataclass
class TrainingReportData:
    """Data class for a single training report.

    Attributes:
        step_range: Tuple of (start_step, end_step) inclusive.
        num_steps: Number of steps in this report.
        mean_loss: Mean loss over the interval.
        min_loss: Minimum loss over the interval.
        max_loss: Maximum loss over the interval.
        mean_grad_norm: Mean gradient norm over the interval.
        max_grad_norm: Maximum gradient norm over the interval.
        learning_rate: Learning rate (mean over interval).
        throughput_steps_per_sec: Training throughput in steps/second.
        safeguard_events: List of safeguard event dictionaries.
        extra_metrics: Dictionary of extra metric means.
        timestamp: ISO 8601 timestamp of report generation.
    """

    step_range: tuple[int, int]
    num_steps: int
    mean_loss: float
    min_loss: float
    max_loss: float
    mean_grad_norm: float
    max_grad_norm: float
    learning_rate: float
    throughput_steps_per_sec: float
    safeguard_events: list[dict[str, Any]] = field(default_factory=list)
    extra_metrics: dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of report data.
        """
        return {
            "step_range": self.step_range,
            "num_steps": self.num_steps,
            "mean_loss": self.mean_loss,
            "min_loss": self.min_loss,
            "max_loss": self.max_loss,
            "mean_grad_norm": self.mean_grad_norm,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "throughput_steps_per_sec": self.throughput_steps_per_sec,
            "safeguard_events": self.safeguard_events,
            "extra_metrics": self.extra_metrics,
            "timestamp": self.timestamp,
        }

    def to_metrics_dict(self, prefix: str = "training_report/") -> dict[str, Any]:
        """Convert to a flat metrics dictionary for the experiment tracker.

        Args:
            prefix: Prefix to prepend to all keys.

        Returns:
            Flat dictionary with prefixed keys.
        """
        wandb_dict = {
            f"{prefix}step_start": self.step_range[0],
            f"{prefix}step_end": self.step_range[1],
            f"{prefix}num_steps": self.num_steps,
            f"{prefix}mean_loss": self.mean_loss,
            f"{prefix}min_loss": self.min_loss,
            f"{prefix}max_loss": self.max_loss,
            f"{prefix}mean_grad_norm": self.mean_grad_norm,
            f"{prefix}max_grad_norm": self.max_grad_norm,
            f"{prefix}learning_rate": self.learning_rate,
            f"{prefix}throughput_steps_per_sec": self.throughput_steps_per_sec,
            f"{prefix}num_safeguard_events": len(self.safeguard_events),
        }

        # Add extra metrics
        for metric_name, metric_value in self.extra_metrics.items():
            wandb_dict[f"{prefix}extra/{metric_name}"] = metric_value

        return wandb_dict

    def to_wandb_dict(self, prefix: str = "training_report/") -> dict[str, Any]:
        """Deprecated alias for :meth:`to_metrics_dict` (W&B is retired)."""
        return self.to_metrics_dict(prefix=prefix)


class TrainingReport:
    """Accumulates training metrics and generates periodic reports.

    This class tracks losses, gradient norms, learning rates, and custom metrics
    over configurable intervals and generates summary reports.
    """

    def __init__(self, config: TrainingReportConfig | None = None) -> None:
        """Initialize training report tracker.

        Args:
            config: Configuration for report generation. If None, uses from_env().
        """
        self.config = config or TrainingReportConfig.from_env()
        self._logger = logging.getLogger(__name__)

        # Accumulators for current interval
        self._losses: list[float] = []
        self._grad_norms: list[float] = []
        self._learning_rates: list[float] = []
        self._extra_metrics: dict[str, list[float]] = defaultdict(list)
        self._safeguard_events: list[dict[str, Any]] = []

        # Interval tracking
        self._interval_start_step: int | None = None
        self._interval_start_time: float = time.monotonic()

        # Generated reports
        self._reports: list[TrainingReportData] = []

    def record_step(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        lr: float,
        *,
        safeguard_event: dict[str, Any] | None = None,
        **metrics: float,
    ) -> None:
        """Record metrics for a training step.

        Args:
            step: Training step number.
            loss: Loss value for this step.
            grad_norm: Gradient norm for this step.
            lr: Learning rate for this step.
            safeguard_event: Optional safeguard event dictionary.
            **metrics: Additional metrics to track (e.g., policy_loss=0.5).
        """
        # Initialize interval start on first step
        if self._interval_start_step is None:
            self._interval_start_step = step
            self._interval_start_time = time.monotonic()

        # Accumulate metrics
        self._losses.append(loss)
        self._grad_norms.append(grad_norm)
        self._learning_rates.append(lr)

        # Accumulate extra metrics
        for metric_name, metric_value in metrics.items():
            self._extra_metrics[metric_name].append(metric_value)

        # Record safeguard event if provided
        if safeguard_event is not None and self.config.include_safeguard_events:
            self._safeguard_events.append(safeguard_event)

    def should_report(self, step: int) -> bool:
        """Check if a report should be generated.

        Args:
            step: Current training step.

        Returns:
            True if enough steps have been accumulated for a report.
        """
        return len(self._losses) >= self.config.report_interval

    def generate_report(self, step: int) -> TrainingReportData | None:
        """Generate a training report for the current interval.

        Args:
            step: Current training step (end of interval).

        Returns:
            TrainingReportData if metrics are available, None otherwise.
        """
        if not self._losses:
            self._logger.warning("generate_report called with no accumulated metrics")
            return None

        # Compute statistics
        num_steps = len(self._losses)
        mean_loss = statistics.mean(self._losses)
        min_loss = min(self._losses)
        max_loss = max(self._losses)
        mean_grad_norm = statistics.mean(self._grad_norms)
        max_grad_norm = max(self._grad_norms)
        learning_rate = statistics.mean(self._learning_rates)

        # Compute throughput
        elapsed_time = time.monotonic() - self._interval_start_time
        throughput = num_steps / elapsed_time if elapsed_time > 0 else 0.0

        # Compute extra metrics means
        extra_metrics = {
            name: statistics.mean(values) for name, values in self._extra_metrics.items()
        }

        # Build report
        start_step = self._interval_start_step if self._interval_start_step is not None else step
        report = TrainingReportData(
            step_range=(start_step, step),
            num_steps=num_steps,
            mean_loss=mean_loss,
            min_loss=min_loss,
            max_loss=max_loss,
            mean_grad_norm=mean_grad_norm,
            max_grad_norm=max_grad_norm,
            learning_rate=learning_rate,
            throughput_steps_per_sec=throughput,
            safeguard_events=self._safeguard_events.copy(),
            extra_metrics=extra_metrics,
        )

        # Store report
        self._reports.append(report)

        # Reset accumulators for next interval
        self._losses.clear()
        self._grad_norms.clear()
        self._learning_rates.clear()
        self._extra_metrics.clear()
        self._safeguard_events.clear()
        self._interval_start_step = None
        self._interval_start_time = time.monotonic()

        self._logger.info(
            f"Generated training report for steps {report.step_range[0]}-"
            f"{report.step_range[1]}: mean_loss={mean_loss:.4f}, "
            f"throughput={throughput:.2f} steps/sec"
        )

        return report

    def get_latest_report(self) -> TrainingReportData | None:
        """Get the most recent training report.

        Returns:
            Latest TrainingReportData if any reports exist, None otherwise.
        """
        return self._reports[-1] if self._reports else None

    def get_all_reports(self) -> list[TrainingReportData]:
        """Get all generated training reports.

        Returns:
            Copy of list of all TrainingReportData.
        """
        return self._reports.copy()

    def reset(self) -> None:
        """Reset all accumulated metrics and reports."""
        self._losses.clear()
        self._grad_norms.clear()
        self._learning_rates.clear()
        self._extra_metrics.clear()
        self._safeguard_events.clear()
        self._interval_start_step = None
        self._interval_start_time = time.monotonic()
        self._reports.clear()
        self._logger.info("Training report tracker reset")

    @property
    def report_count(self) -> int:
        """Get the number of reports generated.

        Returns:
            Number of reports in history.
        """
        return len(self._reports)
