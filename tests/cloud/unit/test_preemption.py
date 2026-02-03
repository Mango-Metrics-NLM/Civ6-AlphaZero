"""Unit tests for preemption handling module."""

from __future__ import annotations

import pytest

from alpha_zero.cloud.preemption import (
    MockPreemptionHandler,
    PreemptionHandler,
    PreemptionState,
    create_preemption_handler,
)


class TestPreemptionState:
    """Tests for PreemptionState."""

    def test_defaults(self) -> None:
        """Test default state values."""
        state = PreemptionState()
        assert state.preemption_requested is False
        assert state.current_iteration == 0
        assert state.checkpoint_saved is False
        assert state.checkpoint_path == ""
        assert state.shutdown_requested is False


class TestPreemptionHandler:
    """Tests for PreemptionHandler."""

    @pytest.fixture
    def handler(self) -> PreemptionHandler:
        """Create handler with mock mode."""
        return PreemptionHandler(force_mock=True)

    @pytest.fixture
    def handler_with_callback(
        self, tmp_path
    ) -> tuple[PreemptionHandler, list[int]]:
        """Create handler with checkpoint callback."""
        saved_iterations: list[int] = []

        def checkpoint_callback(iteration: int) -> str:
            saved_iterations.append(iteration)
            return f"{tmp_path}/checkpoint_{iteration}.pt"

        handler = PreemptionHandler(
            checkpoint_callback=checkpoint_callback,
            force_mock=True,
        )
        return handler, saved_iterations

    def test_initial_state(self, handler: PreemptionHandler) -> None:
        """Test initial state is not preempted."""
        assert not handler.should_stop()
        assert not handler.is_preemption_pending()

    def test_update_iteration(self, handler: PreemptionHandler) -> None:
        """Test updating iteration."""
        handler.update_iteration(100)
        state = handler.get_state()
        assert state.current_iteration == 100

    def test_setup_and_shutdown(self, handler: PreemptionHandler) -> None:
        """Test setup and shutdown lifecycle."""
        handler.setup()
        # Should not raise
        handler.shutdown()

    def test_get_state(self, handler: PreemptionHandler) -> None:
        """Test getting state copy."""
        handler.update_iteration(50)
        state = handler.get_state()

        assert state.current_iteration == 50
        assert state.preemption_requested is False

    def test_trigger_checkpoint_without_callback(
        self, handler: PreemptionHandler
    ) -> None:
        """Test triggering checkpoint without callback."""
        result = handler.trigger_checkpoint()
        assert result is None

    def test_trigger_checkpoint_with_callback(
        self, handler_with_callback: tuple[PreemptionHandler, list[int]]
    ) -> None:
        """Test triggering checkpoint with callback."""
        handler, saved_iterations = handler_with_callback

        handler.update_iteration(100)
        result = handler.trigger_checkpoint()

        assert result is not None
        assert "checkpoint_100" in result
        assert 100 in saved_iterations

    def test_checkpoint_only_saved_once(
        self, handler_with_callback: tuple[PreemptionHandler, list[int]]
    ) -> None:
        """Test that checkpoint is only saved once."""
        handler, saved_iterations = handler_with_callback

        handler.update_iteration(100)
        handler.trigger_checkpoint()
        handler.trigger_checkpoint()
        handler.trigger_checkpoint()

        # Should only have saved once
        assert len(saved_iterations) == 1


class TestMockPreemptionHandler:
    """Tests for MockPreemptionHandler."""

    @pytest.fixture
    def mock_handler(self) -> MockPreemptionHandler:
        """Create mock handler."""
        return MockPreemptionHandler()

    @pytest.fixture
    def mock_handler_with_callback(
        self, tmp_path
    ) -> tuple[MockPreemptionHandler, list[int]]:
        """Create mock handler with callback."""
        saved_iterations: list[int] = []

        def checkpoint_callback(iteration: int) -> str:
            saved_iterations.append(iteration)
            return f"{tmp_path}/checkpoint_{iteration}.pt"

        handler = MockPreemptionHandler(checkpoint_callback=checkpoint_callback)
        return handler, saved_iterations

    def test_trigger_preemption(self, mock_handler: MockPreemptionHandler) -> None:
        """Test manually triggering preemption."""
        assert not mock_handler.is_preemption_pending()

        mock_handler.trigger_preemption()

        assert mock_handler.is_preemption_pending()
        assert mock_handler.should_stop()

    def test_trigger_shutdown(self, mock_handler: MockPreemptionHandler) -> None:
        """Test manually triggering shutdown."""
        assert not mock_handler.should_stop()

        mock_handler.trigger_shutdown()

        assert mock_handler.should_stop()
        # Preemption should not be pending for shutdown
        assert not mock_handler.is_preemption_pending()

    def test_trigger_preemption_saves_checkpoint(
        self,
        mock_handler_with_callback: tuple[MockPreemptionHandler, list[int]],
    ) -> None:
        """Test that preemption triggers checkpoint save."""
        handler, saved_iterations = mock_handler_with_callback

        handler.update_iteration(50)
        handler.trigger_preemption()

        assert 50 in saved_iterations

    def test_reset(self, mock_handler: MockPreemptionHandler) -> None:
        """Test resetting mock state."""
        mock_handler.trigger_preemption()
        assert mock_handler.is_preemption_pending()

        mock_handler.reset()

        assert not mock_handler.is_preemption_pending()
        assert not mock_handler.should_stop()

    def test_cleanup_callbacks(
        self, mock_handler: MockPreemptionHandler
    ) -> None:
        """Test cleanup callbacks are called."""
        cleanup_called = []

        def cleanup1() -> None:
            cleanup_called.append(1)

        def cleanup2() -> None:
            cleanup_called.append(2)

        handler = MockPreemptionHandler(
            cleanup_callbacks=[cleanup1, cleanup2],
        )

        handler.trigger_preemption()

        assert 1 in cleanup_called
        assert 2 in cleanup_called


class TestCreatePreemptionHandler:
    """Tests for factory function."""

    def test_create_regular_handler(self) -> None:
        """Test creating regular handler."""
        handler = create_preemption_handler(force_mock=False)
        assert isinstance(handler, PreemptionHandler)

    def test_create_mock_handler(self) -> None:
        """Test creating mock handler."""
        handler = create_preemption_handler(force_mock=True)
        assert isinstance(handler, MockPreemptionHandler)

    def test_create_with_callbacks(self, tmp_path) -> None:
        """Test creating handler with callbacks."""
        saved = []

        def checkpoint_cb(iteration: int) -> str:
            saved.append(iteration)
            return f"{tmp_path}/ckpt_{iteration}.pt"

        def cleanup_cb() -> None:
            pass

        handler = create_preemption_handler(
            checkpoint_callback=checkpoint_cb,
            cleanup_callbacks=[cleanup_cb],
            force_mock=True,
        )

        handler.update_iteration(100)
        handler.trigger_checkpoint()

        assert 100 in saved
