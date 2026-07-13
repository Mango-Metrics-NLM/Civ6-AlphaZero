"""Tests for web application factory, replay manager creation, and replay route integration.

Tests cover:
- create_app() smoke test (basic app creation)
- _create_replay_manager() with various config/import scenarios
- _session_cleanup_loop() async behavior
- Replay route endpoints using a standalone FastAPI app with mocked ReplayManager
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from alpha_zero.web.config import WebConfig
from alpha_zero.web.routes.replay import create_replay_router

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_web_config() -> WebConfig:
    """Create a minimal WebConfig suitable for testing without torch."""
    return WebConfig(
        board_size=8,
        max_turns=50,
        default_thinking_time=0.1,
        max_concurrent_sessions=3,
        session_timeout_seconds=600.0,
        use_mcts=False,
        mcts_simulations=2,
        default_opponent="random",
        debug=True,
        enable_replay=False,
    )


@pytest.fixture
def replay_enabled_config() -> WebConfig:
    """Create a WebConfig with replay enabled."""
    return WebConfig(
        board_size=8,
        max_turns=50,
        default_thinking_time=0.1,
        max_concurrent_sessions=3,
        session_timeout_seconds=600.0,
        use_mcts=False,
        mcts_simulations=2,
        default_opponent="random",
        debug=True,
        enable_replay=True,
        replay_dir="test_replays",
    )


@pytest.fixture
def mock_replay_manager() -> MagicMock:
    """Create a mock ReplayManager with standard methods."""
    manager = MagicMock()
    manager.list_replays.return_value = [
        {"game_id": "game_001", "turns": 10, "winner": "player_0"},
        {"game_id": "game_002", "turns": 25, "winner": "player_1"},
    ]

    mock_replay = MagicMock()
    mock_replay.get_summary.return_value = {
        "total_turns": 10,
        "winner": "player_0",
        "board_size": 8,
    }
    mock_replay.get_turn.return_value = {
        "board_state": [[0, 1], [1, 0]],
        "action": 42,
        "player": 0,
    }
    manager.load_replay.return_value = mock_replay
    manager.delete_replay.return_value = True
    return manager


@pytest.fixture
def replay_app(mock_replay_manager: MagicMock) -> FastAPI:
    """Create a standalone FastAPI app with only the replay router mounted."""
    app = FastAPI(title="Replay Test App")
    router = create_replay_router(mock_replay_manager)
    app.include_router(router, prefix="/api/replay")
    return app


@pytest.fixture
def replay_client(replay_app: FastAPI) -> TestClient:
    """Create a test client for the replay-only app."""
    return TestClient(replay_app)


@pytest.fixture
def disabled_replay_app() -> FastAPI:
    """Create a standalone FastAPI app with replay manager disabled (None)."""
    app = FastAPI(title="Replay Disabled Test App")
    router = create_replay_router(None)
    app.include_router(router, prefix="/api/replay")
    return app


@pytest.fixture
def disabled_replay_client(disabled_replay_app: FastAPI) -> TestClient:
    """Create a test client for the replay-disabled app."""
    return TestClient(disabled_replay_app)


# ---------------------------------------------------------------------------
# Tests: create_app
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Smoke tests for the create_app factory function."""

    def test_create_app_returns_fastapi_instance(self, minimal_web_config: WebConfig) -> None:
        """create_app should return a FastAPI instance."""
        from alpha_zero.web.app import create_app

        app = create_app(minimal_web_config)
        assert isinstance(app, FastAPI)

    def test_create_app_stores_config_on_state(self, minimal_web_config: WebConfig) -> None:
        """The app should store the config on app.state."""
        from alpha_zero.web.app import create_app

        app = create_app(minimal_web_config)
        assert app.state.config is minimal_web_config

    def test_create_app_stores_session_manager_on_state(
        self, minimal_web_config: WebConfig
    ) -> None:
        """The app should store the session_manager on app.state."""
        from alpha_zero.web.app import create_app

        app = create_app(minimal_web_config)
        assert app.state.session_manager is not None

    def test_create_app_has_routes(self, minimal_web_config: WebConfig) -> None:
        """The app should have routes registered."""
        from alpha_zero.web.app import create_app

        app = create_app(minimal_web_config)
        assert len(app.routes) > 0

    def test_create_app_health_endpoint_works(self, minimal_web_config: WebConfig) -> None:
        """The /api/health endpoint should be functional."""
        from alpha_zero.web.app import create_app

        app = create_app(minimal_web_config)
        client = TestClient(app)
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        from alpha_zero.web import __version__

        assert data["status"] == "ok"
        assert data["active_sessions"] == 0
        assert data["version"] == __version__

    def test_create_app_with_none_config_uses_env(self) -> None:
        """When config is None, create_app should use WebConfig.from_env()."""
        from alpha_zero.web.app import create_app

        app = create_app(None)
        assert isinstance(app, FastAPI)
        assert app.state.config is not None


# ---------------------------------------------------------------------------
# Tests: _create_replay_manager
# ---------------------------------------------------------------------------


class TestCreateReplayManager:
    """Tests for the _create_replay_manager helper."""

    def test_returns_none_when_replay_disabled(self, minimal_web_config: WebConfig) -> None:
        """Should return None when enable_replay is False."""
        from alpha_zero.web.app import _create_replay_manager

        assert minimal_web_config.enable_replay is False
        result = _create_replay_manager(minimal_web_config)
        assert result is None

    def test_returns_none_on_import_error(
        self, replay_enabled_config: WebConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should return None and log warning when ReplayManager cannot be imported."""
        import builtins

        from alpha_zero.web.app import _create_replay_manager

        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if "alpha_zero.human_interface.replay" in name:
                raise ImportError("mocked import error")
            return original_import(name, *args, **kwargs)

        # Remove cached module so the import inside the function is re-executed
        import sys

        saved = sys.modules.pop("alpha_zero.human_interface.replay", None)
        try:
            with patch.object(builtins, "__import__", side_effect=mock_import):
                with caplog.at_level(logging.WARNING):
                    result = _create_replay_manager(replay_enabled_config)
                    assert result is None
        finally:
            if saved is not None:
                sys.modules["alpha_zero.human_interface.replay"] = saved

    def test_returns_none_on_init_exception(
        self, replay_enabled_config: WebConfig, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should return None and log exception when ReplayManager init fails."""
        from alpha_zero.web.app import _create_replay_manager

        mock_replay_cls = MagicMock(side_effect=RuntimeError("init failed"))

        with patch.dict(
            "sys.modules",
            {"alpha_zero.human_interface.replay": MagicMock(ReplayManager=mock_replay_cls)},
        ):
            with caplog.at_level(logging.ERROR):
                result = _create_replay_manager(replay_enabled_config)
                assert result is None

    def test_returns_manager_on_success(self, replay_enabled_config: WebConfig) -> None:
        """Should return a ReplayManager instance when everything works."""
        from alpha_zero.web.app import _create_replay_manager

        mock_instance = MagicMock()
        mock_replay_cls = MagicMock(return_value=mock_instance)

        mock_module = MagicMock(ReplayManager=mock_replay_cls)

        with patch.dict(
            "sys.modules",
            {"alpha_zero.human_interface.replay": mock_module},
        ):
            result = _create_replay_manager(replay_enabled_config)
            assert result is mock_instance
            mock_replay_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: _session_cleanup_loop
# ---------------------------------------------------------------------------


class TestSessionCleanupLoop:
    """Tests for the async _session_cleanup_loop background task."""

    async def test_cleanup_loop_calls_cleanup_expired(self) -> None:
        """The loop should call cleanup_expired on the session manager."""
        from alpha_zero.web.app import _session_cleanup_loop

        mock_session_manager = MagicMock()
        mock_session_manager.cleanup_expired.return_value = 0

        # Use a very short interval and cancel after a brief wait
        task = asyncio.create_task(
            _session_cleanup_loop(mock_session_manager, interval_seconds=0.01)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert mock_session_manager.cleanup_expired.call_count >= 1

    async def test_cleanup_loop_handles_cancellation(self) -> None:
        """The loop should exit cleanly on CancelledError (does not propagate)."""
        from alpha_zero.web.app import _session_cleanup_loop

        mock_session_manager = MagicMock()
        mock_session_manager.cleanup_expired.return_value = 0

        task = asyncio.create_task(
            _session_cleanup_loop(mock_session_manager, interval_seconds=0.01)
        )
        await asyncio.sleep(0.03)
        task.cancel()

        # The function catches CancelledError internally and exits gracefully,
        # so awaiting the task should complete without raising.
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Task should be done after cancellation
        assert task.done()

    async def test_cleanup_loop_logs_removed_sessions(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The loop should log when sessions are cleaned up."""
        from alpha_zero.web.app import _session_cleanup_loop

        mock_session_manager = MagicMock()
        mock_session_manager.cleanup_expired.return_value = 3

        task = asyncio.create_task(
            _session_cleanup_loop(mock_session_manager, interval_seconds=0.01)
        )

        with caplog.at_level(logging.INFO, logger="alpha_zero.web.app"):
            await asyncio.sleep(0.05)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert any(
            "3" in record.message and "expired" in record.message for record in caplog.records
        )

    async def test_cleanup_loop_survives_exception(self) -> None:
        """The loop should continue running after a non-cancellation exception."""
        from alpha_zero.web.app import _session_cleanup_loop

        mock_session_manager = MagicMock()
        call_count = 0

        def side_effect() -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return 0

        mock_session_manager.cleanup_expired.side_effect = side_effect

        task = asyncio.create_task(
            _session_cleanup_loop(mock_session_manager, interval_seconds=0.01)
        )
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have been called more than once (survived the exception)
        assert mock_session_manager.cleanup_expired.call_count >= 2

    async def test_cleanup_loop_no_log_when_zero_removed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The loop should not log when no sessions were removed."""
        from alpha_zero.web.app import _session_cleanup_loop

        mock_session_manager = MagicMock()
        mock_session_manager.cleanup_expired.return_value = 0

        task = asyncio.create_task(
            _session_cleanup_loop(mock_session_manager, interval_seconds=0.01)
        )

        with caplog.at_level(logging.INFO, logger="alpha_zero.web.app"):
            await asyncio.sleep(0.05)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        cleanup_log_messages = [
            r
            for r in caplog.records
            if "expired session" in r.message.lower() or "Cleaned up" in r.message
        ]
        assert len(cleanup_log_messages) == 0


# ---------------------------------------------------------------------------
# Tests: Replay Route - List
# ---------------------------------------------------------------------------


class TestReplayListEndpoint:
    """Tests for GET /api/replay/list."""

    def test_list_replays_with_manager(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Should return replays list and enabled=True when manager is present."""
        response = replay_client.get("/api/replay/list")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert len(data["replays"]) == len(mock_replay_manager.list_replays.return_value)
        mock_replay_manager.list_replays.assert_called_once()

    def test_list_replays_returns_replay_data(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """The returned replays should match the manager's response."""
        response = replay_client.get("/api/replay/list")
        data = response.json()
        expected = mock_replay_manager.list_replays.return_value
        assert data["replays"] == expected

    def test_list_replays_with_exception(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Should return empty list and enabled=True when list_replays raises."""
        mock_replay_manager.list_replays.side_effect = RuntimeError("db error")
        response = replay_client.get("/api/replay/list")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is True
        assert data["replays"] == []

    def test_list_replays_without_manager(self, disabled_replay_client: TestClient) -> None:
        """Should return empty list and enabled=False when manager is None."""
        response = disabled_replay_client.get("/api/replay/list")
        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False
        assert data["replays"] == []


# ---------------------------------------------------------------------------
# Tests: Replay Route - Get Replay
# ---------------------------------------------------------------------------


class TestGetReplayEndpoint:
    """Tests for GET /api/replay/{game_id}."""

    def test_get_replay_success(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Should return game_id and summary for a valid replay."""
        test_game_id = "game_001"
        response = replay_client.get(f"/api/replay/{test_game_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["game_id"] == test_game_id
        assert "summary" in data
        mock_replay_manager.load_replay.assert_called_once_with(test_game_id)

    def test_get_replay_summary_content(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Summary should come from the replay object's get_summary method."""
        test_game_id = "game_001"
        response = replay_client.get(f"/api/replay/{test_game_id}")
        data = response.json()
        expected_summary = mock_replay_manager.load_replay.return_value.get_summary.return_value
        assert data["summary"] == expected_summary

    def test_get_replay_not_found(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Should return 404 when the replay file is not found."""
        missing_id = "nonexistent_game"
        mock_replay_manager.load_replay.side_effect = FileNotFoundError("not found")
        response = replay_client.get(f"/api/replay/{missing_id}")
        assert response.status_code == 404
        assert missing_id in response.json()["detail"]

    def test_get_replay_server_error(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Should return 500 on unexpected exceptions from load_replay."""
        error_msg = "corrupt file"
        mock_replay_manager.load_replay.side_effect = RuntimeError(error_msg)
        response = replay_client.get("/api/replay/game_bad")
        assert response.status_code == 500
        assert error_msg in response.json()["detail"]

    def test_get_replay_disabled(self, disabled_replay_client: TestClient) -> None:
        """Should return 404 when replay manager is None."""
        response = disabled_replay_client.get("/api/replay/any_game")
        assert response.status_code == 404
        assert "not enabled" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Tests: Replay Route - Get Replay Turn
# ---------------------------------------------------------------------------


class TestGetReplayTurnEndpoint:
    """Tests for GET /api/replay/{game_id}/turn/{turn_number}."""

    def test_get_replay_turn_success(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Should return turn data for a valid game and turn number."""
        test_game_id = "game_001"
        test_turn = 5
        response = replay_client.get(f"/api/replay/{test_game_id}/turn/{test_turn}")
        assert response.status_code == 200
        data = response.json()
        assert data["game_id"] == test_game_id
        assert data["turn_number"] == test_turn
        assert "data" in data

    def test_get_replay_turn_data_content(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Turn data should come from the replay object's get_turn method."""
        test_game_id = "game_001"
        test_turn = 3
        response = replay_client.get(f"/api/replay/{test_game_id}/turn/{test_turn}")
        data = response.json()
        mock_replay = mock_replay_manager.load_replay.return_value
        mock_replay.get_turn.assert_called_once_with(test_turn)
        expected_data = mock_replay.get_turn.return_value
        assert data["data"] == expected_data

    def test_get_replay_turn_game_not_found(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Should return 404 when the game replay file is not found."""
        missing_id = "nonexistent_game"
        mock_replay_manager.load_replay.side_effect = FileNotFoundError("not found")
        response = replay_client.get(f"/api/replay/{missing_id}/turn/0")
        assert response.status_code == 404
        assert missing_id in response.json()["detail"]

    def test_get_replay_turn_not_found(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Should return 404 when the turn does not exist in the replay."""
        test_game_id = "game_001"
        invalid_turn = 999
        mock_replay_manager.load_replay.return_value.get_turn.return_value = None
        response = replay_client.get(f"/api/replay/{test_game_id}/turn/{invalid_turn}")
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert str(invalid_turn) in detail
        assert test_game_id in detail

    def test_get_replay_turn_disabled(self, disabled_replay_client: TestClient) -> None:
        """Should return 404 when replay manager is None."""
        response = disabled_replay_client.get("/api/replay/any_game/turn/0")
        assert response.status_code == 404
        assert "not enabled" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Tests: Replay Route - Delete Replay
# ---------------------------------------------------------------------------


class TestDeleteReplayEndpoint:
    """Tests for DELETE /api/replay/{game_id}."""

    def test_delete_replay_success(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Should return deleted=True and the game_id on successful deletion."""
        test_game_id = "game_001"
        mock_replay_manager.delete_replay.return_value = True
        response = replay_client.delete(f"/api/replay/{test_game_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert data["game_id"] == test_game_id
        mock_replay_manager.delete_replay.assert_called_once_with(test_game_id)

    def test_delete_replay_not_found(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Should return 404 when delete_replay returns False."""
        missing_id = "nonexistent_game"
        mock_replay_manager.delete_replay.return_value = False
        response = replay_client.delete(f"/api/replay/{missing_id}")
        assert response.status_code == 404
        assert missing_id in response.json()["detail"]

    def test_delete_replay_server_error(
        self, replay_client: TestClient, mock_replay_manager: MagicMock
    ) -> None:
        """Should return 500 on unexpected exceptions from delete_replay."""
        error_msg = "disk IO error"
        mock_replay_manager.delete_replay.side_effect = OSError(error_msg)
        response = replay_client.delete("/api/replay/game_bad")
        assert response.status_code == 500
        assert error_msg in response.json()["detail"]

    def test_delete_replay_disabled(self, disabled_replay_client: TestClient) -> None:
        """Should return 404 when replay manager is None."""
        response = disabled_replay_client.delete("/api/replay/any_game")
        assert response.status_code == 404
        assert "not enabled" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Tests: Replay Router creation
# ---------------------------------------------------------------------------


class TestCreateReplayRouter:
    """Tests for the create_replay_router factory function."""

    def test_creates_api_router(self) -> None:
        """Should return an APIRouter instance."""
        from fastapi import APIRouter

        router = create_replay_router(None)
        assert isinstance(router, APIRouter)

    def test_router_has_replay_tag(self) -> None:
        """The router should be tagged with 'replay'."""
        router = create_replay_router(None)
        assert "replay" in router.tags

    def test_router_accepts_mock_manager(self, mock_replay_manager: MagicMock) -> None:
        """Should accept a MagicMock as the replay_manager argument."""
        from fastapi import APIRouter

        router = create_replay_router(mock_replay_manager)
        assert isinstance(router, APIRouter)

    def test_router_has_expected_routes(self, mock_replay_manager: MagicMock) -> None:
        """The router should have routes for list, get, turn, and delete."""
        router = create_replay_router(mock_replay_manager)
        route_paths = [r.path for r in router.routes]
        assert "/list" in route_paths
        assert "/{game_id}" in route_paths
        assert "/{game_id}/turn/{turn_number}" in route_paths
