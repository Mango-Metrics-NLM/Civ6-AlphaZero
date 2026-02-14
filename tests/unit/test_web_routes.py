"""Tests for web API routes."""

from __future__ import annotations

import pytest

from alpha_zero.web.config import WebConfig


@pytest.fixture
def web_config():
    """Create test web configuration."""
    return WebConfig(
        board_size=8,
        max_turns=50,
        default_thinking_time=0.1,
        max_concurrent_sessions=5,
        session_timeout_seconds=600.0,
        use_mcts=False,
        mcts_simulations=2,
        default_opponent="random",
        debug=True,
    )


@pytest.fixture
def app(web_config):
    """Create test FastAPI application."""
    from alpha_zero.web.app import create_app

    return create_app(web_config)


@pytest.fixture
def client(app):
    """Create test HTTP client."""
    from starlette.testclient import TestClient

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "active_sessions" in data
        assert "version" in data

    def test_health_check_shows_zero_sessions(self, client):
        data = client.get("/api/health").json()
        assert data["active_sessions"] == 0


class TestGameCreateEndpoint:
    """Tests for POST /api/game/new."""

    def test_create_game_default(self, client):
        response = client.post("/api/game/new", json={})
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "state" in data
        assert data["state"]["session_id"] == data["session_id"]

    def test_create_game_with_options(self, client):
        response = client.post(
            "/api/game/new",
            json={
                "human_player": 1,
                "opponent_type": "heuristic",
                "board_size": 8,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["state"]["human_player"] == 1

    def test_create_game_with_seed(self, client):
        response = client.post("/api/game/new", json={"seed": 42})
        assert response.status_code == 200

    def test_create_game_invalid_opponent(self, client):
        response = client.post(
            "/api/game/new",
            json={"opponent_type": "invalid"},
        )
        assert response.status_code == 400

    def test_create_game_invalid_human_player(self, client):
        response = client.post(
            "/api/game/new",
            json={"human_player": 5},
        )
        assert response.status_code == 422  # Pydantic validation

    def test_session_counter_increments(self, client):
        client.post("/api/game/new", json={})
        health = client.get("/api/health").json()
        assert health["active_sessions"] == 1

        client.post("/api/game/new", json={})
        health = client.get("/api/health").json()
        assert health["active_sessions"] == 2


class TestGameStateEndpoint:
    """Tests for GET /api/game/{session_id}."""

    def test_get_state(self, client):
        create_resp = client.post("/api/game/new", json={})
        session_id = create_resp.json()["session_id"]

        response = client.get(f"/api/game/{session_id}")
        assert response.status_code == 200
        state = response.json()["state"]
        assert state["session_id"] == session_id
        assert "phase" in state
        assert "board" in state
        assert "valid_actions" in state
        assert "history" in state

    def test_get_state_nonexistent(self, client):
        response = client.get("/api/game/nonexistent")
        assert response.status_code == 404

    def test_state_has_board_data(self, client):
        create_resp = client.post("/api/game/new", json={})
        session_id = create_resp.json()["session_id"]

        state = client.get(f"/api/game/{session_id}").json()["state"]
        board = state["board"]
        assert "terrain_grid" in board
        assert "units" in board
        assert "cities" in board
        assert "board_size" in board


class TestGameMoveEndpoint:
    """Tests for POST /api/game/{session_id}/move."""

    def _create_game(self, client):
        resp = client.post("/api/game/new", json={"human_player": 0})
        data = resp.json()
        return data["session_id"], data["state"]

    def test_submit_valid_move(self, client):
        session_id, state = self._create_game(client)

        if state["phase"] == "human_turn":
            valid = state["valid_actions"]
            assert len(valid) > 0
            action = valid[0]["index"]

            response = client.post(
                f"/api/game/{session_id}/move",
                json={"action": action},
            )
            assert response.status_code == 200
            data = response.json()
            assert "human_move" in data
            assert "state" in data

    def test_submit_move_nonexistent_session(self, client):
        response = client.post(
            "/api/game/nonexistent/move",
            json={"action": 0},
        )
        assert response.status_code == 404

    def test_submit_invalid_action(self, client):
        session_id, state = self._create_game(client)

        if state["phase"] == "human_turn":
            # Use a very high action index that's unlikely to be valid
            response = client.post(
                f"/api/game/{session_id}/move",
                json={"action": 99999},
            )
            assert response.status_code == 400

    def test_move_triggers_ai_response(self, client):
        session_id, state = self._create_game(client)

        if state["phase"] == "human_turn":
            valid = state["valid_actions"]
            action = valid[0]["index"]

            response = client.post(
                f"/api/game/{session_id}/move",
                json={"action": action},
            )
            data = response.json()
            # Should have AI move if game didn't end
            if not data["state"]["game_over"]:
                # AI should have played if it was AI's turn
                if "ai_move" in data:
                    assert data["ai_move"]["is_human"] is False


class TestGameUndoEndpoint:
    """Tests for POST /api/game/{session_id}/undo."""

    def test_undo(self, client):
        resp = client.post("/api/game/new", json={"human_player": 0})
        session_id = resp.json()["session_id"]

        response = client.post(f"/api/game/{session_id}/undo")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "state" in data

    def test_undo_nonexistent(self, client):
        response = client.post("/api/game/nonexistent/undo")
        assert response.status_code == 404


class TestGameResignEndpoint:
    """Tests for POST /api/game/{session_id}/resign."""

    def test_resign(self, client):
        resp = client.post("/api/game/new", json={"human_player": 0})
        session_id = resp.json()["session_id"]

        response = client.post(f"/api/game/{session_id}/resign")
        assert response.status_code == 200
        data = response.json()
        assert data["state"]["game_over"] is True

    def test_resign_nonexistent(self, client):
        response = client.post("/api/game/nonexistent/resign")
        assert response.status_code == 404


class TestGamePauseResumeEndpoints:
    """Tests for pause and resume endpoints."""

    def test_pause(self, client):
        resp = client.post("/api/game/new", json={})
        session_id = resp.json()["session_id"]

        response = client.post(f"/api/game/{session_id}/pause")
        assert response.status_code == 200
        assert response.json()["state"]["phase"] == "paused"

    def test_resume(self, client):
        resp = client.post("/api/game/new", json={})
        session_id = resp.json()["session_id"]

        client.post(f"/api/game/{session_id}/pause")
        response = client.post(f"/api/game/{session_id}/resume")
        assert response.status_code == 200
        assert response.json()["state"]["phase"] != "paused"


class TestValidActionsEndpoint:
    """Tests for GET /api/game/{session_id}/valid-actions."""

    def test_valid_actions(self, client):
        resp = client.post("/api/game/new", json={})
        session_id = resp.json()["session_id"]

        response = client.get(f"/api/game/{session_id}/valid-actions")
        assert response.status_code == 200
        data = response.json()
        assert "actions" in data
        assert len(data["actions"]) > 0
        assert "index" in data["actions"][0]
        assert "label" in data["actions"][0]

    def test_valid_actions_nonexistent(self, client):
        response = client.get("/api/game/nonexistent/valid-actions")
        assert response.status_code == 404


class TestGameSummaryEndpoint:
    """Tests for GET /api/game/{session_id}/summary."""

    def test_summary(self, client):
        resp = client.post("/api/game/new", json={})
        session_id = resp.json()["session_id"]

        response = client.get(f"/api/game/{session_id}/summary")
        assert response.status_code == 200
        data = response.json()
        assert "total_moves" in data
        assert "human_moves" in data
        assert "ai_moves" in data
        assert "game_over" in data

    def test_summary_nonexistent(self, client):
        response = client.get("/api/game/nonexistent/summary")
        assert response.status_code == 404


class TestReplayEndpoints:
    """Tests for replay API endpoints."""

    def test_list_replays(self, client):
        response = client.get("/api/replay/list")
        assert response.status_code == 200
        data = response.json()
        assert "replays" in data

    def test_get_replay_nonexistent(self, client):
        response = client.get("/api/replay/nonexistent_game")
        assert response.status_code == 404


class TestPageEndpoints:
    """Tests for HTML page routes."""

    def test_index_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "MicroCiv" in response.text or "New Game" in response.text

    def test_game_page(self, client):
        resp = client.post("/api/game/new", json={})
        session_id = resp.json()["session_id"]

        response = client.get(f"/game/{session_id}")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_game_page_nonexistent(self, client):
        response = client.get("/game/nonexistent")
        assert response.status_code == 404


class TestWebSocketEndpoint:
    """Tests for WebSocket game interaction."""

    def test_websocket_connect(self, client):
        resp = client.post("/api/game/new", json={})
        session_id = resp.json()["session_id"]

        with client.websocket_connect(f"/ws/game/{session_id}") as ws:
            # Should receive initial state
            data = ws.receive_json()
            assert data["type"] == "state"
            assert "data" in data

    def test_websocket_ping_pong(self, client):
        resp = client.post("/api/game/new", json={})
        session_id = resp.json()["session_id"]

        with client.websocket_connect(f"/ws/game/{session_id}") as ws:
            ws.receive_json()  # initial state
            ws.send_json({"type": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_websocket_invalid_session(self, client):
        with pytest.raises(  # noqa: B017
            Exception
        ):
            with client.websocket_connect("/ws/game/nonexistent"):
                pass

    def test_websocket_send_move(self, client):
        resp = client.post("/api/game/new", json={"human_player": 0})
        session_id = resp.json()["session_id"]
        state = resp.json()["state"]

        with client.websocket_connect(f"/ws/game/{session_id}") as ws:
            ws.receive_json()  # initial state

            if state["phase"] == "human_turn" and state["valid_actions"]:
                action = state["valid_actions"][0]["index"]
                ws.send_json({"type": "move", "action": action})

                # Should get move result
                data = ws.receive_json()
                assert data["type"] == "move_result"

    def test_websocket_invalid_json(self, client):
        resp = client.post("/api/game/new", json={})
        session_id = resp.json()["session_id"]

        with client.websocket_connect(f"/ws/game/{session_id}") as ws:
            ws.receive_json()  # initial state
            ws.send_text("not json")
            data = ws.receive_json()
            assert data["type"] == "error"

    def test_websocket_unknown_message_type(self, client):
        resp = client.post("/api/game/new", json={})
        session_id = resp.json()["session_id"]

        with client.websocket_connect(f"/ws/game/{session_id}") as ws:
            ws.receive_json()  # initial state
            ws.send_json({"type": "unknown_type"})
            data = ws.receive_json()
            assert data["type"] == "error"


class TestWebSocketMoveErrors:
    """Tests for WebSocket move error paths."""

    def _create_and_connect(self, client):
        """Helper to create a game and return session_id + initial state."""
        resp = client.post("/api/game/new", json={"human_player": 0})
        data = resp.json()
        return data["session_id"], data["state"]

    def test_move_missing_action_field(self, client):
        session_id, _ = self._create_and_connect(client)
        with client.websocket_connect(f"/ws/game/{session_id}") as ws:
            ws.receive_json()  # initial state
            ws.send_json({"type": "move"})  # no 'action' key
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "action" in data["message"].lower()

    def test_move_invalid_action_index(self, client):
        session_id, state = self._create_and_connect(client)
        if state["phase"] == "human_turn":
            with client.websocket_connect(f"/ws/game/{session_id}") as ws:
                ws.receive_json()  # initial state
                ws.send_json({"type": "move", "action": 999999})
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "invalid" in data["message"].lower() or "Invalid" in data["message"]


class TestWebSocketUndoResign:
    """Tests for WebSocket undo and resign messages."""

    def test_undo_via_websocket(self, client):
        resp = client.post("/api/game/new", json={"human_player": 0})
        session_id = resp.json()["session_id"]
        state = resp.json()["state"]

        with client.websocket_connect(f"/ws/game/{session_id}") as ws:
            ws.receive_json()  # initial state

            # Play a move first if possible
            if state["phase"] == "human_turn" and state["valid_actions"]:
                action = state["valid_actions"][0]["index"]
                ws.send_json({"type": "move", "action": action})
                # Drain all messages until we get the final 'state' update
                for _ in range(10):
                    msg = ws.receive_json()
                    if msg["type"] == "state":
                        break
                    if msg["type"] == "game_over":
                        break

            ws.send_json({"type": "undo"})
            data = ws.receive_json()
            assert data["type"] == "state"
            assert "data" in data

    def test_resign_via_websocket(self, client):
        resp = client.post("/api/game/new", json={"human_player": 0})
        session_id = resp.json()["session_id"]

        with client.websocket_connect(f"/ws/game/{session_id}") as ws:
            ws.receive_json()  # initial state
            ws.send_json({"type": "resign"})

            # Should get exactly move_result + game_over (2 messages)
            msg1 = ws.receive_json()
            msg2 = ws.receive_json()
            types = [msg1["type"], msg2["type"]]
            assert "move_result" in types
            assert "game_over" in types


class TestHTMXMoveEndpoint:
    """Tests for HTMX form-based move handler."""

    def test_htmx_move_valid(self, client):
        resp = client.post("/api/game/new", json={"human_player": 0})
        session_id = resp.json()["session_id"]
        state = resp.json()["state"]

        if state["phase"] == "human_turn" and state["valid_actions"]:
            action = state["valid_actions"][0]["index"]
            response = client.post(
                f"/htmx/game/{session_id}/move",
                data={"action": action},
            )
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_htmx_move_session_not_found(self, client):
        response = client.post(
            "/htmx/game/nonexistent/move",
            data={"action": 0},
        )
        assert response.status_code == 404


class TestHTMXUndoEndpoint:
    """Tests for HTMX undo handler."""

    def test_htmx_undo(self, client):
        resp = client.post("/api/game/new", json={"human_player": 0})
        session_id = resp.json()["session_id"]

        response = client.post(f"/htmx/game/{session_id}/undo")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_htmx_undo_not_found(self, client):
        response = client.post("/htmx/game/nonexistent/undo")
        assert response.status_code == 404


class TestHTMXResignEndpoint:
    """Tests for HTMX resign handler."""

    def test_htmx_resign(self, client):
        resp = client.post("/api/game/new", json={"human_player": 0})
        session_id = resp.json()["session_id"]

        response = client.post(f"/htmx/game/{session_id}/resign")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_htmx_resign_not_found(self, client):
        response = client.post("/htmx/game/nonexistent/resign")
        assert response.status_code == 404


class TestGameCreateFormEndpoint:
    """Tests for the HTML form-based game creation."""

    def test_create_game_form(self, client):
        response = client.post(
            "/game/new",
            data={"human_player": 0, "opponent_type": "random", "board_size": 8},
            follow_redirects=False,
        )
        # Should redirect to game page
        assert response.status_code == 303
        assert "/game/" in response.headers["location"]

    def test_create_game_form_invalid_opponent(self, client):
        response = client.post(
            "/game/new",
            data={"human_player": 0, "opponent_type": "invalid", "board_size": 8},
        )
        # Should return 400 with error in HTML
        assert response.status_code == 400


