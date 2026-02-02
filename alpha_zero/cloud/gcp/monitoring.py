"""Google Cloud Logging and Monitoring integration.

Provides:
- CloudLogger: Structured logging to Cloud Logging
- CloudMetricsExporter: Metrics export to Cloud Monitoring

Both implementations follow the force_mock pattern for testing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from alpha_zero.cloud.config import CloudConfig
from alpha_zero.cloud.exceptions import CloudConnectionError
from alpha_zero.cloud.protocols import CloudMonitoringProtocol

logger = logging.getLogger(__name__)

# SDK availability flags
LOGGING_AVAILABLE = False
MONITORING_AVAILABLE = False

try:
    from google.cloud import logging as cloud_logging  # type: ignore[import-untyped]

    LOGGING_AVAILABLE = True
except ImportError:
    cloud_logging = None  # type: ignore[assignment]

try:
    from google.cloud import monitoring_v3  # type: ignore[import-untyped]

    MONITORING_AVAILABLE = True
except ImportError:
    monitoring_v3 = None  # type: ignore[assignment]


# Severity level mapping
SEVERITY_MAP = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


@dataclass
class LogEntry:
    """Buffered log entry."""

    severity: str
    message: str
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


class CloudLogger(CloudMonitoringProtocol):
    """Google Cloud Logging integration.

    Provides structured logging to Cloud Logging with buffering
    and batch submission.

    Example:
        >>> config = CloudConfig(project_id="my-project")
        >>> logger = CloudLogger(config)
        >>> logger.log_event("INFO", "Training started", {"experiment": "v1"})
        >>> logger.log_metrics(100, {"loss": 0.5, "accuracy": 0.9})
        >>> logger.flush()
    """

    def __init__(
        self,
        config: CloudConfig,
        log_name: str = "alphazero-training",
        force_mock: bool = False,
        buffer_size: int = 100,
    ) -> None:
        """Initialize Cloud Logger.

        Args:
            config: Cloud configuration.
            log_name: Name for the log in Cloud Logging.
            force_mock: Force use of mock for testing.
            buffer_size: Number of entries to buffer before flush.
        """
        self.config = config
        self._log_name = log_name
        self._use_mock = force_mock or not LOGGING_AVAILABLE or not config.enable_cloud_logging
        self._buffer_size = buffer_size

        self._buffer: list[LogEntry] = []
        self._client: Any = None
        self._logger: Any = None

        if not self._use_mock:
            try:
                self._client = cloud_logging.Client(project=config.project_id)
                self._logger = self._client.logger(log_name)
            except Exception as e:
                logger.warning("Failed to initialize Cloud Logging: %s", e)
                self._use_mock = True

        logger.info(
            "Cloud Logger initialized (mock=%s, log=%s)",
            self._use_mock,
            log_name,
        )

    @property
    def is_mock(self) -> bool:
        """Check if using mock implementation."""
        return self._use_mock

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, float],
        labels: dict[str, str] | None = None,
    ) -> None:
        """Log training metrics.

        Args:
            step: Training step number.
            metrics: Dictionary of metric names to values.
            labels: Optional labels for filtering.
        """
        combined_labels = {"step": str(step)}
        if labels:
            combined_labels.update(labels)

        payload = {"step": step, "metrics": metrics}

        self._add_to_buffer(LogEntry(
            severity="INFO",
            message=f"Training metrics at step {step}",
            timestamp=time.time(),
            labels=combined_labels,
            payload=payload,
        ))

    def log_event(
        self,
        severity: str,
        message: str,
        labels: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Log an event with severity level.

        Args:
            severity: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            message: Event message.
            labels: Optional labels for filtering.
            payload: Optional structured data.
        """
        self._add_to_buffer(LogEntry(
            severity=SEVERITY_MAP.get(severity.upper(), "INFO"),
            message=message,
            timestamp=time.time(),
            labels=labels or {},
            payload=payload or {},
        ))

    def flush(self) -> None:
        """Flush any pending logs to the backend."""
        if not self._buffer:
            return

        if self._use_mock:
            # Just log locally and clear buffer
            for entry in self._buffer:
                local_level = getattr(logging, entry.severity, logging.INFO)
                logger.log(local_level, "[CloudLog] %s", entry.message)
            self._buffer.clear()
            return

        # Batch write to Cloud Logging
        try:
            for entry in self._buffer:
                self._logger.log_struct(
                    {
                        "message": entry.message,
                        **entry.payload,
                    },
                    severity=entry.severity,
                    labels=entry.labels,
                )
            self._buffer.clear()
        except Exception as e:
            logger.warning("Failed to flush logs to Cloud Logging: %s", e)

    def _add_to_buffer(self, entry: LogEntry) -> None:
        """Add entry to buffer, flushing if full."""
        self._buffer.append(entry)
        if len(self._buffer) >= self._buffer_size:
            self.flush()


class CloudMetricsExporter:
    """Google Cloud Monitoring metrics exporter.

    Exports custom metrics to Cloud Monitoring for dashboards
    and alerting.

    Example:
        >>> config = CloudConfig(project_id="my-project")
        >>> exporter = CloudMetricsExporter(config)
        >>> exporter.export_metric("training/loss", 0.5, {"model": "v1"})
    """

    def __init__(
        self,
        config: CloudConfig,
        metric_prefix: str = "custom.googleapis.com/alphazero",
        force_mock: bool = False,
    ) -> None:
        """Initialize Cloud Metrics Exporter.

        Args:
            config: Cloud configuration.
            metric_prefix: Prefix for custom metric names.
            force_mock: Force use of mock for testing.
        """
        self.config = config
        self._metric_prefix = metric_prefix
        self._use_mock = force_mock or not MONITORING_AVAILABLE or not config.enable_cloud_monitoring

        self._client: Any = None
        self._project_name: str = ""

        if not self._use_mock and config.project_id:
            try:
                self._client = monitoring_v3.MetricServiceClient()
                self._project_name = f"projects/{config.project_id}"
            except Exception as e:
                logger.warning("Failed to initialize Cloud Monitoring: %s", e)
                self._use_mock = True

        logger.info(
            "Cloud Metrics Exporter initialized (mock=%s, prefix=%s)",
            self._use_mock,
            metric_prefix,
        )

    @property
    def is_mock(self) -> bool:
        """Check if using mock implementation."""
        return self._use_mock

    def export_metric(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Export a metric to Cloud Monitoring.

        Args:
            name: Metric name (will be prefixed).
            value: Metric value.
            labels: Optional labels for the metric.
        """
        full_name = f"{self._metric_prefix}/{name}"

        if self._use_mock:
            logger.debug("Mock metric: %s = %f (labels=%s)", full_name, value, labels)
            return

        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = full_name
            series.resource.type = "global"

            if labels:
                for key, val in labels.items():
                    series.metric.labels[key] = val

            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 1e9)

            interval = monitoring_v3.TimeInterval(
                end_time={"seconds": seconds, "nanos": nanos}
            )
            point = monitoring_v3.Point(
                interval=interval,
                value={"double_value": value},
            )
            series.points = [point]

            self._client.create_time_series(
                name=self._project_name,
                time_series=[series],
            )

        except Exception as e:
            logger.warning("Failed to export metric %s: %s", full_name, e)

    def export_training_metrics(
        self,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        """Export multiple training metrics.

        Args:
            step: Training step number.
            metrics: Dictionary of metric names to values.
        """
        labels = {"step": str(step)}
        for name, value in metrics.items():
            self.export_metric(f"training/{name}", value, labels)
