"""Cloud logging integration for GCP.

This module provides:
- CloudLogger: Integration with Google Cloud Logging
- CloudMetricsExporter: Export metrics to Cloud Monitoring
- Structured logging with buffering and batch submission

Usage:
    from alpha_zero.cloud.cloud_logging import CloudLogger, CloudMetricsExporter

    logger = CloudLogger(project_id="my-project")
    logger.setup_python_logging()

    logger.log_metrics(iteration=100, metrics={"loss": 0.5})

    exporter = CloudMetricsExporter(project_id="my-project")
    exporter.export_metric("training/loss", 0.5)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from alpha_zero.cloud.config import CloudLoggingConfig, CloudMonitoringConfig

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Structured log entry.

    Attributes:
        severity: Log severity level.
        message: Log message.
        timestamp: Unix timestamp.
        labels: Log labels.
        payload: Structured JSON payload.
    """

    severity: str
    message: str
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


class MockCloudLoggingClient:
    """Mock Cloud Logging client for testing."""

    def __init__(self, project: str | None = None) -> None:
        """Initialize mock client.

        Args:
            project: GCP project ID.
        """
        self.project = project
        self._logs: list[LogEntry] = []

    def logger(self, name: str) -> MockCloudLogger:
        """Get or create logger.

        Args:
            name: Logger name.

        Returns:
            Mock logger instance.
        """
        return MockCloudLogger(name, self._logs)

    def setup_logging(self) -> None:
        """Set up Python logging integration (no-op for mock)."""
        pass


class MockCloudLogger:
    """Mock Cloud Logging logger."""

    def __init__(self, name: str, logs: list[LogEntry]) -> None:
        """Initialize mock logger.

        Args:
            name: Logger name.
            logs: Shared logs list.
        """
        self.name = name
        self._logs = logs

    def log_struct(
        self,
        info: dict[str, Any],
        severity: str = "INFO",
    ) -> None:
        """Log structured data.

        Args:
            info: Structured data to log.
            severity: Log severity.
        """
        entry = LogEntry(
            severity=severity,
            message=str(info.get("message", "")),
            payload=info,
        )
        self._logs.append(entry)

    def log_text(self, text: str, severity: str = "INFO") -> None:
        """Log text message.

        Args:
            text: Text message.
            severity: Log severity.
        """
        entry = LogEntry(
            severity=severity,
            message=text,
        )
        self._logs.append(entry)


def _get_cloud_logging_client(
    project: str | None = None,
    force_mock: bool = False,
) -> Any:
    """Get Cloud Logging client.

    Args:
        project: GCP project ID.
        force_mock: Force use of mock client.

    Returns:
        Cloud Logging client instance.
    """
    if force_mock:
        return MockCloudLoggingClient(project=project)

    try:
        import google.cloud.logging

        return google.cloud.logging.Client(project=project)
    except ImportError:
        logger.warning("google-cloud-logging not installed, using mock")
        return MockCloudLoggingClient(project=project)
    except Exception as e:
        logger.warning(f"Failed to create Cloud Logging client: {e}")
        return MockCloudLoggingClient(project=project)


class CloudLogger:
    """Logger that integrates with Google Cloud Logging.

    Features:
    - Structured logging to Cloud Logging
    - Log buffering for batch submission
    - Severity level mapping
    - Python logging integration

    Attributes:
        project_id: GCP project ID.
        log_name: Cloud Logging log name.
        buffer_size: Maximum buffer size before flush.
        flush_interval: Seconds between automatic flushes.
    """

    def __init__(
        self,
        project_id: str,
        log_name: str = "civ6-alphazero-training",
        buffer_size: int = 100,
        flush_interval: float = 30.0,
        force_mock: bool = False,
    ) -> None:
        """Initialize cloud logger.

        Args:
            project_id: GCP project ID.
            log_name: Cloud Logging log name.
            buffer_size: Maximum buffer size before flush.
            flush_interval: Seconds between automatic flushes.
            force_mock: Force use of mock client.
        """
        self.project_id = project_id
        self.log_name = log_name
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._force_mock = force_mock

        # Initialize client
        self._client = _get_cloud_logging_client(
            project=project_id,
            force_mock=force_mock,
        )
        self._logger = self._client.logger(log_name)

        # Buffer and threading
        self._buffer: queue.Queue[LogEntry] = queue.Queue(maxsize=buffer_size * 2)
        self._flush_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

    @classmethod
    def from_config(
        cls,
        project_id: str,
        config: CloudLoggingConfig,
        force_mock: bool = False,
    ) -> CloudLogger:
        """Create from configuration.

        Args:
            project_id: GCP project ID.
            config: Cloud logging configuration.
            force_mock: Force use of mock client.

        Returns:
            Configured CloudLogger instance.
        """
        return cls(
            project_id=project_id,
            log_name=config.log_name,
            buffer_size=config.buffer_size,
            flush_interval=config.flush_interval_seconds,
            force_mock=force_mock,
        )

    def setup_python_logging(self) -> None:
        """Configure Python logging to use Cloud Logging."""
        if hasattr(self._client, "setup_logging"):
            self._client.setup_logging()

    def start_background_flush(self) -> None:
        """Start background thread for periodic log flushing."""
        if self._running:
            return

        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="cloud-logger-flush",
        )
        self._flush_thread.start()

    def stop_background_flush(self) -> None:
        """Stop background flush thread."""
        self._running = False
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=self.flush_interval + 1)
        self.flush()

    def log(
        self,
        message: str,
        severity: str = "INFO",
        labels: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Log a message.

        Args:
            message: Log message.
            severity: Log severity (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            labels: Additional labels.
            payload: Structured payload data.
        """
        entry = LogEntry(
            severity=severity.upper(),
            message=message,
            labels=labels or {},
            payload=payload or {},
        )

        try:
            self._buffer.put_nowait(entry)
        except queue.Full:
            # Buffer full, flush and retry
            self.flush()
            try:
                self._buffer.put_nowait(entry)
            except queue.Full:
                logger.warning("Log buffer full, dropping entry")

        # Auto-flush if buffer is large
        if self._buffer.qsize() >= self.buffer_size:
            self.flush()

    def log_metrics(
        self,
        iteration: int,
        metrics: dict[str, float],
        labels: dict[str, str] | None = None,
    ) -> None:
        """Log training metrics.

        Args:
            iteration: Training iteration number.
            metrics: Dictionary of metric name to value.
            labels: Additional labels.
        """
        payload = {
            "type": "training_metrics",
            "iteration": iteration,
            **metrics,
        }

        self.log(
            message=f"Training metrics at iteration {iteration}",
            severity="INFO",
            labels=labels,
            payload=payload,
        )

    def log_checkpoint(
        self,
        iteration: int,
        checkpoint_path: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Log checkpoint event.

        Args:
            iteration: Training iteration number.
            checkpoint_path: Path to saved checkpoint.
            metrics: Optional metrics at checkpoint time.
        """
        payload = {
            "type": "checkpoint",
            "iteration": iteration,
            "checkpoint_path": checkpoint_path,
        }
        if metrics:
            payload["metrics"] = metrics

        self.log(
            message=f"Checkpoint saved at iteration {iteration}",
            severity="INFO",
            payload=payload,
        )

    def log_preemption(
        self,
        iteration: int,
        checkpoint_saved: bool,
        checkpoint_path: str = "",
    ) -> None:
        """Log preemption event.

        Args:
            iteration: Training iteration when preempted.
            checkpoint_saved: Whether checkpoint was saved.
            checkpoint_path: Path to saved checkpoint if any.
        """
        payload = {
            "type": "preemption",
            "iteration": iteration,
            "checkpoint_saved": checkpoint_saved,
            "checkpoint_path": checkpoint_path,
        }

        self.log(
            message=f"Preemption at iteration {iteration}",
            severity="WARNING",
            payload=payload,
        )

    def log_job_event(
        self,
        event_type: str,
        job_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log Vertex AI job event.

        Args:
            event_type: Event type (started, completed, failed, etc.).
            job_id: Vertex AI job ID.
            details: Additional event details.
        """
        payload = {
            "type": "job_event",
            "event_type": event_type,
            "job_id": job_id,
        }
        if details:
            payload.update(details)

        self.log(
            message=f"Job event: {event_type}",
            severity="INFO" if event_type != "failed" else "ERROR",
            payload=payload,
        )

    def flush(self) -> None:
        """Flush buffered logs to Cloud Logging."""
        entries_to_send: list[LogEntry] = []

        while True:
            try:
                entry = self._buffer.get_nowait()
                entries_to_send.append(entry)
            except queue.Empty:
                break

        for entry in entries_to_send:
            try:
                if entry.payload:
                    self._logger.log_struct(
                        {
                            "message": entry.message,
                            **entry.payload,
                        },
                        severity=entry.severity,
                    )
                else:
                    self._logger.log_text(entry.message, severity=entry.severity)
            except Exception as e:
                logger.warning(f"Failed to send log entry: {e}")

    def _flush_loop(self) -> None:
        """Background loop for periodic log flushing."""
        while self._running:
            time.sleep(self.flush_interval)
            if self._running:
                self.flush()


class MockCloudMonitoringClient:
    """Mock Cloud Monitoring client for testing."""

    def __init__(self, project: str | None = None) -> None:
        """Initialize mock client.

        Args:
            project: GCP project ID.
        """
        self.project = project
        self._metrics: list[tuple[str, float, dict[str, str]]] = []

    def create_time_series(self, request: dict[str, Any]) -> None:
        """Create time series (mock implementation).

        Args:
            request: Time series request.
        """
        pass  # No-op for mock


def _get_cloud_monitoring_client(
    project: str | None = None,
    force_mock: bool = False,
) -> Any:
    """Get Cloud Monitoring client.

    Args:
        project: GCP project ID.
        force_mock: Force use of mock client.

    Returns:
        Cloud Monitoring client instance.
    """
    if force_mock:
        return MockCloudMonitoringClient(project=project)

    try:
        from google.cloud import monitoring_v3  # type: ignore

        return monitoring_v3.MetricServiceClient()
    except ImportError:
        logger.warning("google-cloud-monitoring not installed, using mock")
        return MockCloudMonitoringClient(project=project)
    except Exception as e:
        logger.warning(f"Failed to create Cloud Monitoring client: {e}")
        return MockCloudMonitoringClient(project=project)


class CloudMetricsExporter:
    """Exports metrics to Google Cloud Monitoring.

    Features:
    - Custom metric creation
    - Batched metric submission
    - Label support

    Attributes:
        project_id: GCP project ID.
        metric_prefix: Prefix for custom metrics.
    """

    def __init__(
        self,
        project_id: str,
        metric_prefix: str = "custom.googleapis.com/alphazero",
        force_mock: bool = False,
    ) -> None:
        """Initialize metrics exporter.

        Args:
            project_id: GCP project ID.
            metric_prefix: Prefix for custom metrics.
            force_mock: Force use of mock client.
        """
        self.project_id = project_id
        self.metric_prefix = metric_prefix.rstrip("/")
        self._force_mock = force_mock

        self._client = _get_cloud_monitoring_client(
            project=project_id,
            force_mock=force_mock,
        )

        # For mock, track exported metrics
        self._exported_metrics: list[tuple[str, float, dict[str, str]]] = []

    @classmethod
    def from_config(
        cls,
        project_id: str,
        config: CloudMonitoringConfig,
        force_mock: bool = False,
    ) -> CloudMetricsExporter:
        """Create from configuration.

        Args:
            project_id: GCP project ID.
            config: Cloud monitoring configuration.
            force_mock: Force use of mock client.

        Returns:
            Configured CloudMetricsExporter instance.
        """
        return cls(
            project_id=project_id,
            metric_prefix=config.metric_prefix,
            force_mock=force_mock,
        )

    def export_metric(
        self,
        metric_name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Export a single metric.

        Args:
            metric_name: Metric name (without prefix).
            value: Metric value.
            labels: Optional metric labels.
        """
        full_metric_name = f"{self.metric_prefix}/{metric_name}"
        labels = labels or {}

        if self._force_mock or isinstance(self._client, MockCloudMonitoringClient):
            self._exported_metrics.append((full_metric_name, value, labels))
            return

        try:
            from google.cloud import monitoring_v3  # type: ignore
            from google.protobuf import timestamp_pb2

            series = monitoring_v3.TimeSeries()
            series.metric.type = full_metric_name
            series.resource.type = "global"

            for key, val in labels.items():
                series.metric.labels[key] = val

            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 1e9)

            interval = monitoring_v3.TimeInterval(
                end_time=timestamp_pb2.Timestamp(seconds=seconds, nanos=nanos)
            )
            point = monitoring_v3.Point(
                interval=interval,
                value=monitoring_v3.TypedValue(double_value=value),
            )
            series.points = [point]

            self._client.create_time_series(
                request={
                    "name": f"projects/{self.project_id}",
                    "time_series": [series],
                }
            )

        except Exception as e:
            logger.warning(f"Failed to export metric {metric_name}: {e}")

    def export_training_metrics(
        self,
        iteration: int,
        metrics: dict[str, float],
        labels: dict[str, str] | None = None,
    ) -> None:
        """Export training metrics batch.

        Args:
            iteration: Training iteration number.
            metrics: Dictionary of metric name to value.
            labels: Common labels for all metrics.
        """
        combined_labels = {"iteration": str(iteration)}
        if labels:
            combined_labels.update(labels)

        for name, value in metrics.items():
            self.export_metric(
                metric_name=f"training/{name}",
                value=value,
                labels=combined_labels,
            )

    def get_exported_metrics(self) -> list[tuple[str, float, dict[str, str]]]:
        """Get list of exported metrics (for testing).

        Returns:
            List of (metric_name, value, labels) tuples.
        """
        return self._exported_metrics.copy()
