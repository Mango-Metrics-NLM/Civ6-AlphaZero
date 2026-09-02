"""Unit tests for the GCP preemption handler."""

from __future__ import annotations

import pytest

from alpha_zero.cloud.gcp.preemption import PreemptionHandler, register_preemption_handler


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

    def test_double_start_warning(
        self, handler: PreemptionHandler, caplog: pytest.LogCaptureFixture
    ) -> None:
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
