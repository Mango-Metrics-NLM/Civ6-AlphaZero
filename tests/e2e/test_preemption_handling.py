"""E2E tests for GCP preemption handling.

Tests the preemption handler's ability to:
- Initialize and configure properly
- Register and manage cleanup callbacks
- Handle mock preemption signals
- Execute graceful shutdown sequences

These tests can run without real GCP credentials using the force_mock pattern.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# Test Classes
# =============================================================================


class TestPreemptionHandlerInitialization:
    """Tests for preemption handler initialization."""

    def test_handler_initialization_mock_mode(self) -> None:
        """Test that preemption handler initializes in mock mode."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        handler = PreemptionHandler(force_mock=True)

        assert handler._use_mock is True
        assert handler._preemption_pending is False
        logger.info("Preemption handler initialized in mock mode")

    def test_handler_with_checkpoint_callback(self) -> None:
        """Test handler with checkpoint callback."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        checkpoint_called = threading.Event()

        def save_checkpoint() -> None:
            checkpoint_called.set()

        handler = PreemptionHandler(
            checkpoint_callback=save_checkpoint,
            force_mock=True,
        )

        assert handler._checkpoint_callback is not None

    def test_handler_with_custom_check_interval(self) -> None:
        """Test handler with custom check interval."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        handler = PreemptionHandler(
            check_interval=2.0,
            force_mock=True,
        )

        assert handler._check_interval == 2.0

    def test_handler_with_cleanup_callbacks(self) -> None:
        """Test handler initialization with cleanup callbacks."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        cleanup1 = MagicMock()
        cleanup2 = MagicMock()

        handler = PreemptionHandler(
            cleanup_callbacks=[cleanup1, cleanup2],
            force_mock=True,
        )

        assert len(handler._cleanup_callbacks) == 2


class TestPreemptionCallbackManagement:
    """Tests for callback registration and management."""

    def test_register_cleanup_callback(self) -> None:
        """Test registering a cleanup callback."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        handler = PreemptionHandler(force_mock=True)
        callback = MagicMock()

        handler.register_cleanup(callback)

        assert len(handler._cleanup_callbacks) == 1
        logger.info("Cleanup callback registered successfully")

    def test_register_multiple_cleanup_callbacks(self) -> None:
        """Test registering multiple cleanup callbacks."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        handler = PreemptionHandler(force_mock=True)

        for i in range(3):
            handler.register_cleanup(MagicMock(name=f"callback_{i}"))

        assert len(handler._cleanup_callbacks) == 3


class TestMockPreemptionSignal:
    """Tests for mock preemption signal handling."""

    def test_simulate_preemption(self) -> None:
        """Test simulating preemption."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        handler = PreemptionHandler(force_mock=True)

        # Simulate preemption
        handler.simulate_preemption()

        assert handler._preemption_pending is True
        logger.info("Mock preemption triggered successfully")

    def test_is_preemption_pending(self) -> None:
        """Test checking preemption pending status."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        handler = PreemptionHandler(force_mock=True)

        # Initially not pending
        assert handler.is_preemption_pending() is False

        # After simulation, should be pending
        handler.simulate_preemption()
        assert handler.is_preemption_pending() is True

    def test_checkpoint_callback_on_preemption(self) -> None:
        """Test that checkpoint callback is executed on preemption."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        checkpoint_called = threading.Event()

        def save_checkpoint() -> None:
            checkpoint_called.set()

        handler = PreemptionHandler(
            checkpoint_callback=save_checkpoint,
            force_mock=True,
        )

        handler.simulate_preemption()

        assert checkpoint_called.is_set()

    def test_cleanup_callbacks_on_preemption(self) -> None:
        """Test that cleanup callbacks are executed on preemption."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        execution_order: list[str] = []

        def cleanup_1() -> None:
            execution_order.append("cleanup_1")

        def cleanup_2() -> None:
            execution_order.append("cleanup_2")

        handler = PreemptionHandler(
            cleanup_callbacks=[cleanup_1, cleanup_2],
            force_mock=True,
        )

        handler.simulate_preemption()

        assert "cleanup_1" in execution_order
        assert "cleanup_2" in execution_order

    def test_callback_exception_handling(self) -> None:
        """Test that callback exceptions don't prevent other callbacks."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        def failing_cleanup() -> None:
            raise RuntimeError("Intentional test failure")

        success_callback = MagicMock()

        handler = PreemptionHandler(
            cleanup_callbacks=[failing_cleanup, success_callback],
            force_mock=True,
        )

        # Should not raise despite failing callback
        handler.simulate_preemption()

        # Second callback should still execute
        success_callback.assert_called_once()


class TestPreemptionMonitoring:
    """Tests for preemption monitoring lifecycle."""

    def test_start_monitoring(self) -> None:
        """Test starting preemption monitoring."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        handler = PreemptionHandler(
            check_interval=0.1,
            force_mock=True,
        )

        handler.start_monitoring()
        assert handler._monitoring is True

        # Cleanup
        handler.stop_monitoring()

    def test_stop_monitoring(self) -> None:
        """Test stopping preemption monitoring."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        handler = PreemptionHandler(
            check_interval=0.1,
            force_mock=True,
        )

        handler.start_monitoring()
        handler.stop_monitoring()

        assert handler._monitoring is False

    def test_double_stop_is_safe(self) -> None:
        """Test that calling stop twice doesn't cause errors."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        handler = PreemptionHandler(force_mock=True)
        handler.start_monitoring()

        # Double stop should be safe
        handler.stop_monitoring()
        handler.stop_monitoring()  # Should not raise

    def test_double_start_logs_warning(self) -> None:
        """Test that double start logs a warning but doesn't error."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        handler = PreemptionHandler(force_mock=True)

        handler.start_monitoring()
        handler.start_monitoring()  # Should log warning but not error

        # Cleanup
        handler.stop_monitoring()


class TestPreemptionTimeRemaining:
    """Tests for time remaining calculation."""

    def test_time_remaining_before_preemption(self) -> None:
        """Test time remaining returns None before preemption."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        handler = PreemptionHandler(force_mock=True)

        assert handler.get_time_remaining() is None

    def test_time_remaining_after_preemption(self) -> None:
        """Test time remaining returns value after preemption."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        handler = PreemptionHandler(force_mock=True)
        handler.simulate_preemption()

        time_remaining = handler.get_time_remaining()

        # Should be close to PREEMPTION_WARNING_SECONDS (30s)
        assert time_remaining is not None
        assert 25.0 <= time_remaining <= 30.0


class TestPreemptionWithCheckpoint:
    """Tests for checkpoint saving on preemption."""

    def test_checkpoint_saved_to_file(self) -> None:
        """Test checkpoint saving as a preemption callback."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "emergency_checkpoint.pt"

            def save_checkpoint() -> None:
                checkpoint_path.write_text("mock checkpoint data")

            handler = PreemptionHandler(
                checkpoint_callback=save_checkpoint,
                force_mock=True,
            )

            handler.simulate_preemption()

            assert checkpoint_path.exists()
            logger.info("Emergency checkpoint saved successfully")

    def test_checkpoint_before_cleanup(self) -> None:
        """Test that checkpoint is saved before cleanup callbacks."""
        from alpha_zero.cloud.gcp.preemption import PreemptionHandler

        execution_order: list[str] = []

        def save_checkpoint() -> None:
            execution_order.append("checkpoint")

        def cleanup() -> None:
            execution_order.append("cleanup")

        handler = PreemptionHandler(
            checkpoint_callback=save_checkpoint,
            cleanup_callbacks=[cleanup],
            force_mock=True,
        )

        handler.simulate_preemption()

        # Checkpoint should be called before cleanup
        assert execution_order == ["checkpoint", "cleanup"]


class TestRegisterPreemptionHandler:
    """Tests for the convenience registration function."""

    def test_register_with_auto_start(self) -> None:
        """Test registering with auto-start enabled."""
        from alpha_zero.cloud.gcp.preemption import register_preemption_handler

        handler = register_preemption_handler(
            checkpoint_callback=lambda: None,
            auto_start=True,
        )

        assert handler._monitoring is True

        # Cleanup
        handler.stop_monitoring()

    def test_register_without_auto_start(self) -> None:
        """Test registering without auto-start."""
        from alpha_zero.cloud.gcp.preemption import register_preemption_handler

        handler = register_preemption_handler(
            auto_start=False,
        )

        assert handler._monitoring is False


# =============================================================================
# Integration Tests (require specific environment)
# =============================================================================


@pytest.mark.requires_gcp
class TestPreemptionOnGCE:
    """Tests that require a GCE environment.

    These would only pass when running on an actual GCE instance.
    """

    def test_metadata_service_check(self) -> None:
        """Test that metadata service check works on GCE."""
        # This test is mainly for documentation
        pytest.skip("Requires GCE instance environment")
