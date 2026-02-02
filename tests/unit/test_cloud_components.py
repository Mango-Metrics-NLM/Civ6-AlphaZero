"""Unit tests for preemption handler and cloud monitoring."""

from __future__ import annotations

import threading
import time
from typing import Callable
from unittest import mock

import pytest

from alpha_zero.cloud.config import CloudConfig
from alpha_zero.cloud.gcp.preemption import PreemptionHandler, register_preemption_handler
from alpha_zero.cloud.gcp.monitoring import CloudLogger, CloudMetricsExporter


class TestPreemptionHandler:
    """Tests for PreemptionHandler."""

    @pytest.fixture
    def handler(self) -> PreemptionHandler:
        """Create a mock preemption handler."""
        return PreemptionHandler(force_mock=True)

    def test_initialization(self, handler: PreemptionHandler) -> None:
        """Test handler initialization."""
        assert handler.is_preemption_pending() is False
        assert handler.get_time_remaining() is None

    def test_start_stop_monitoring(self, handler: PreemptionHandler) -> None:
        """Test starting and stopping monitoring."""
        handler.start_monitoring()
        
        # Should be running
        assert handler._monitoring is True
        
        handler.stop_monitoring()
        
        # Should be stopped
        assert handler._monitoring is False

    def test_register_cleanup(self, handler: PreemptionHandler) -> None:
        """Test registering cleanup callbacks."""
        cleanup_called = []
        
        def cleanup1() -> None:
            cleanup_called.append(1)
        
        def cleanup2() -> None:
            cleanup_called.append(2)
        
        handler.register_cleanup(cleanup1)
        handler.register_cleanup(cleanup2)
        
        assert len(handler._cleanup_callbacks) == 2

    def test_simulate_preemption(self) -> None:
        """Test simulating preemption."""
        checkpoint_called = False
        cleanup_called = False
        
        def checkpoint() -> None:
            nonlocal checkpoint_called
            checkpoint_called = True
        
        def cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True
        
        handler = PreemptionHandler(
            checkpoint_callback=checkpoint,
            cleanup_callbacks=[cleanup],
            force_mock=True,
        )
        
        handler.simulate_preemption()
        
        assert checkpoint_called is True
        assert cleanup_called is True
        assert handler.is_preemption_pending() is True

    def test_time_remaining_after_preemption(self) -> None:
        """Test time remaining calculation after preemption."""
        handler = PreemptionHandler(force_mock=True)
        
        handler.simulate_preemption()
        
        remaining = handler.get_time_remaining()
        
        assert remaining is not None
        assert remaining >= 0
        assert remaining <= 30.0  # Max is 30 seconds

    def test_preemption_only_triggers_once(self) -> None:
        """Test that preemption callbacks only trigger once."""
        call_count = [0]
        
        def checkpoint() -> None:
            call_count[0] += 1
        
        handler = PreemptionHandler(
            checkpoint_callback=checkpoint,
            force_mock=True,
        )
        
        # Trigger twice
        handler.simulate_preemption()
        handler.simulate_preemption()
        
        # Should only have been called once
        assert call_count[0] == 1

    def test_double_start_warning(self, handler: PreemptionHandler, caplog: pytest.LogCaptureFixture) -> None:
        """Test warning when starting monitoring twice."""
        handler.start_monitoring()
        handler.start_monitoring()  # Should warn
        
        assert "already started" in caplog.text
        
        handler.stop_monitoring()


class TestRegisterPreemptionHandler:
    """Tests for register_preemption_handler convenience function."""

    def test_auto_start(self) -> None:
        """Test auto_start parameter."""
        # With auto_start (default)
        handler = register_preemption_handler(
            checkpoint_callback=lambda: None,
            auto_start=True,
        )
        
        try:
            assert handler._monitoring is True
        finally:
            handler.stop_monitoring()

    def test_no_auto_start(self) -> None:
        """Test with auto_start=False."""
        handler = register_preemption_handler(
            checkpoint_callback=lambda: None,
            auto_start=False,
        )
        
        assert handler._monitoring is False


class TestCloudLogger:
    """Tests for CloudLogger."""

    @pytest.fixture
    def cloud_config(self) -> CloudConfig:
        """Create test cloud config."""
        return CloudConfig(
            project_id="test-project",
            enable_cloud_logging=True,
        )

    @pytest.fixture
    def logger(self, cloud_config: CloudConfig) -> CloudLogger:
        """Create mock cloud logger."""
        return CloudLogger(cloud_config, force_mock=True)

    def test_initialization(self, logger: CloudLogger) -> None:
        """Test logger initialization."""
        assert logger.is_mock is True

    def test_log_metrics(self, logger: CloudLogger) -> None:
        """Test logging metrics."""
        # Should not raise
        logger.log_metrics(
            step=100,
            metrics={"loss": 0.5, "accuracy": 0.9},
            labels={"experiment": "test"},
        )
        
        # Check buffer
        assert len(logger._buffer) == 1

    def test_log_event(self, logger: CloudLogger) -> None:
        """Test logging events."""
        logger.log_event(
            severity="INFO",
            message="Training started",
            labels={"phase": "init"},
            payload={"config": {"batch_size": 32}},
        )
        
        assert len(logger._buffer) == 1
        assert logger._buffer[0].message == "Training started"
        assert logger._buffer[0].severity == "INFO"

    def test_log_severity_levels(self, logger: CloudLogger) -> None:
        """Test different severity levels."""
        severities = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for severity in severities:
            logger.log_event(severity, f"Test {severity}")
        
        assert len(logger._buffer) == 5

    def test_flush(self, logger: CloudLogger) -> None:
        """Test flushing logs."""
        logger.log_event("INFO", "Message 1")
        logger.log_event("INFO", "Message 2")
        
        assert len(logger._buffer) == 2
        
        logger.flush()
        
        assert len(logger._buffer) == 0

    def test_auto_flush_on_buffer_full(self) -> None:
        """Test automatic flush when buffer is full."""
        config = CloudConfig(project_id="test")
        logger = CloudLogger(config, force_mock=True, buffer_size=3)
        
        # Add entries up to buffer size
        logger.log_event("INFO", "1")
        logger.log_event("INFO", "2")
        
        assert len(logger._buffer) == 2
        
        # This should trigger auto-flush
        logger.log_event("INFO", "3")
        
        # Buffer should be empty after flush
        assert len(logger._buffer) == 0


class TestCloudMetricsExporter:
    """Tests for CloudMetricsExporter."""

    @pytest.fixture
    def cloud_config(self) -> CloudConfig:
        """Create test cloud config."""
        return CloudConfig(
            project_id="test-project",
            enable_cloud_monitoring=True,
        )

    @pytest.fixture
    def exporter(self, cloud_config: CloudConfig) -> CloudMetricsExporter:
        """Create mock metrics exporter."""
        return CloudMetricsExporter(cloud_config, force_mock=True)

    def test_initialization(self, exporter: CloudMetricsExporter) -> None:
        """Test exporter initialization."""
        assert exporter.is_mock is True

    def test_export_metric(self, exporter: CloudMetricsExporter) -> None:
        """Test exporting a single metric."""
        # Should not raise
        exporter.export_metric(
            name="training/loss",
            value=0.5,
            labels={"model": "v1"},
        )

    def test_export_training_metrics(self, exporter: CloudMetricsExporter) -> None:
        """Test exporting multiple training metrics."""
        metrics = {
            "loss": 0.5,
            "policy_loss": 0.3,
            "value_loss": 0.2,
            "entropy": 0.8,
        }
        
        # Should not raise
        exporter.export_training_metrics(step=1000, metrics=metrics)

    def test_custom_metric_prefix(self, cloud_config: CloudConfig) -> None:
        """Test custom metric prefix."""
        exporter = CloudMetricsExporter(
            cloud_config,
            metric_prefix="custom/prefix",
            force_mock=True,
        )
        
        assert exporter._metric_prefix == "custom/prefix"
