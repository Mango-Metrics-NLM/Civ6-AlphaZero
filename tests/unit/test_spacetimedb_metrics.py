"""Unit tests for SpacetimeDB metrics sink."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from alpha_zero.spacetimedb.client import SpacetimeDBClient
from alpha_zero.spacetimedb.config import SpacetimeDBConfig
from alpha_zero.spacetimedb.metrics_sink import SpacetimeDBMetricsSink
from tests.mocks.spacetimedb_mocks import MockSpacetimeDBClient


@pytest.fixture
def config() -> SpacetimeDBConfig:
    """Create test config."""
    return SpacetimeDBConfig(
        enabled=True,
        host="http://localhost:3000",
        module_address="test",
        connection_timeout=1.0,
        max_weight_blob_mb=10,
    )


@pytest.fixture
def mock_sdk() -> MockSpacetimeDBClient:
    """Create mock SDK client."""
    return MockSpacetimeDBClient()


@pytest.fixture
def client(config: SpacetimeDBConfig, mock_sdk: MockSpacetimeDBClient) -> SpacetimeDBClient:
    """Create connected client."""
    c = SpacetimeDBClient(config)
    c.connect_with_mock(mock_sdk)
    return c


@pytest.fixture
def sink(client: SpacetimeDBClient) -> SpacetimeDBMetricsSink:
    """Create metrics sink with small batch size for testing."""
    return SpacetimeDBMetricsSink(
        client, batch_size=3, flush_interval=0.01,
    )


class TestMetricsSinkInit:
    """Test metrics sink initialization."""

    def test_initial_state(self, sink: SpacetimeDBMetricsSink) -> None:
        assert sink.total_metrics_logged == 0
        assert sink.total_events_logged == 0
        assert sink.flush_count == 0
        assert sink.pending_count == 0


class TestLogMetrics:
    """Test metric logging."""

    def test_log_single_metric(
        self, sink: SpacetimeDBMetricsSink,
    ) -> None:
        sink.log_metrics(step=1, metrics={"loss": 0.5})
        assert sink.total_metrics_logged == 1
        assert sink.pending_count == 1

    def test_log_multiple_metrics(
        self, sink: SpacetimeDBMetricsSink,
    ) -> None:
        sink.log_metrics(step=1, metrics={"loss": 0.5, "accuracy": 0.8})
        sink.log_metrics(step=2, metrics={"loss": 0.4})
        assert sink.total_metrics_logged == 2
        assert sink.pending_count == 2

    def test_auto_flush_on_batch_size(
        self, sink: SpacetimeDBMetricsSink, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        """Buffer auto-flushes when batch_size is reached."""
        # batch_size=3 in fixture
        sink.log_metrics(step=1, metrics={"loss": 0.5})
        sink.log_metrics(step=2, metrics={"loss": 0.4})
        assert sink.pending_count == 2

        # Third metric should trigger auto-flush
        sink.log_metrics(step=3, metrics={"loss": 0.3})
        assert sink.pending_count == 0
        assert mock_sdk.call_counts["log_metrics"] == 3

    def test_log_metrics_with_labels(
        self, sink: SpacetimeDBMetricsSink,
    ) -> None:
        sink.log_metrics(
            step=1, metrics={"loss": 0.5},
            labels={"worker": "actor_0"},
        )
        assert sink.total_metrics_logged == 1


class TestLogEvent:
    """Test event logging."""

    def test_log_info_event(
        self, sink: SpacetimeDBMetricsSink,
    ) -> None:
        sink.log_event("INFO", "Training started")
        assert sink.total_events_logged == 1

    def test_log_error_event(
        self, sink: SpacetimeDBMetricsSink,
    ) -> None:
        sink.log_event("ERROR", "Connection lost", payload={"host": "localhost"})
        assert sink.total_events_logged == 1

    def test_log_event_with_labels(
        self, sink: SpacetimeDBMetricsSink,
    ) -> None:
        sink.log_event(
            "WARNING", "High memory usage",
            labels={"component": "replay_buffer"},
        )
        assert sink.total_events_logged == 1


class TestFlush:
    """Test flush behavior."""

    def test_explicit_flush(
        self, sink: SpacetimeDBMetricsSink, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        sink.log_metrics(step=1, metrics={"loss": 0.5})
        sink.log_metrics(step=2, metrics={"loss": 0.4})
        assert sink.pending_count == 2

        sink.flush()
        assert sink.pending_count == 0
        assert sink.flush_count == 1
        assert mock_sdk.call_counts["log_metrics"] == 2

    def test_flush_empty_buffer(
        self, sink: SpacetimeDBMetricsSink,
    ) -> None:
        sink.flush()
        assert sink.flush_count == 1

    def test_multiple_flushes(
        self, sink: SpacetimeDBMetricsSink, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        sink.log_metrics(step=1, metrics={"loss": 0.5})
        sink.flush()
        sink.log_metrics(step=2, metrics={"loss": 0.4})
        sink.flush()

        assert sink.flush_count == 2
        assert mock_sdk.call_counts["log_metrics"] == 2


class TestShouldFlush:
    """Test should_flush logic."""

    def test_should_flush_empty(
        self, sink: SpacetimeDBMetricsSink,
    ) -> None:
        assert not sink.should_flush()

    def test_should_flush_batch_size(
        self, sink: SpacetimeDBMetricsSink,
    ) -> None:
        # Don't trigger auto-flush, check should_flush
        sink._buffer.append((1, {"loss": 0.5}, None))
        sink._buffer.append((2, {"loss": 0.4}, None))
        sink._buffer.append((3, {"loss": 0.3}, None))
        assert sink.should_flush()

    def test_should_flush_interval(
        self, sink: SpacetimeDBMetricsSink,
    ) -> None:
        sink.log_metrics(step=1, metrics={"loss": 0.5})
        # Set last flush time far in the past
        sink._last_flush_time = 0.0
        assert sink.should_flush()


class TestCloudMonitoringProtocol:
    """Test CloudMonitoringProtocol compliance."""

    def test_has_log_metrics(self, sink: SpacetimeDBMetricsSink) -> None:
        assert hasattr(sink, "log_metrics")
        assert callable(sink.log_metrics)

    def test_has_log_event(self, sink: SpacetimeDBMetricsSink) -> None:
        assert hasattr(sink, "log_event")
        assert callable(sink.log_event)

    def test_has_flush(self, sink: SpacetimeDBMetricsSink) -> None:
        assert hasattr(sink, "flush")
        assert callable(sink.flush)

    def test_runtime_checkable(self, sink: SpacetimeDBMetricsSink) -> None:
        """Verify sink satisfies CloudMonitoringProtocol."""
        from alpha_zero.cloud.protocols import CloudMonitoringProtocol

        assert isinstance(sink, CloudMonitoringProtocol)


class TestEndToEnd:
    """Test complete metrics workflows."""

    def test_training_loop_metrics(
        self, sink: SpacetimeDBMetricsSink, mock_sdk: MockSpacetimeDBClient,
    ) -> None:
        """Simulate a training loop publishing metrics."""
        for step in range(10):
            metrics = {
                "total_loss": 1.0 / (step + 1),
                "policy_loss": 0.5 / (step + 1),
                "value_loss": 0.5 / (step + 1),
                "learning_rate": 0.001,
            }
            sink.log_metrics(step=step, metrics=metrics)

        # Some may have been auto-flushed (batch_size=3)
        sink.flush()

        assert sink.total_metrics_logged == 10
        assert mock_sdk.call_counts["log_metrics"] == 10

    def test_mixed_metrics_and_events(
        self, sink: SpacetimeDBMetricsSink,
    ) -> None:
        sink.log_metrics(step=1, metrics={"loss": 0.5})
        sink.log_event("INFO", "Checkpoint saved")
        sink.log_metrics(step=2, metrics={"loss": 0.4})
        sink.log_event("WARNING", "Slow convergence")

        sink.flush()

        assert sink.total_metrics_logged == 2
        assert sink.total_events_logged == 2
