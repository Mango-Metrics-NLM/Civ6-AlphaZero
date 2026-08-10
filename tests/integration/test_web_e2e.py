"""End-to-end integration tests for the web interface.

Tests complete game flows through the HTTP API and WebSocket.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from alpha_zero.web.config import WebConfig


@pytest.fixture
def web_config():
    """Fast test configuration."""
    return WebConfig(
        board_size=8,
        max_turns=30,
        default_thinking_time=0.05,
        max_concurrent_sessions=5,
        session_timeout_seconds=300.0,
        use_mcts=False,
        mcts_simulations=2,
        default_opponent="random",
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


@pytest.mark.integration
class TestFullGameViaREST:
    """Test playing a complete game through REST API."""

    def test_complete_game_random_opponent(self, client):
        """Play a complete game against random opponent via REST."""
        import numpy as np

        rng = np.random.default_rng(42)

        # Create game
        resp = client.post(
            "/api/game/new",
            json={"human_player": 0, "opponent_type": "random", "board_size": 8, "seed": 42},
        )
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Play moves
        for _move_num in range(200):
            # Get current state
            state_resp = client.get(f"/api/game/{session_id}")
            state = state_resp.json()["state"]

            if state["game_over"]:
                break

            if state["phase"] != "human_turn":
                # Shouldn't happen with auto-AI, but just in case
                continue

            valid = state["valid_actions"]
            if not valid:
                break

            # Pick random valid action
            action = valid[rng.integers(len(valid))]["index"]

            move_resp = client.post(
                f"/api/game/{session_id}/move",
                json={"action": action},
            )
            assert move_resp.status_code == 200

        # Get summary
        summary_resp = client.get(f"/api/game/{session_id}/summary")
        assert summary_resp.status_code == 200
        summary = summary_resp.json()
        assert summary["total_moves"] > 0

    def test_complete_game_heuristic_opponent(self, client):
        """Play a short game against heuristic opponent."""
        import numpy as np

        rng = np.random.default_rng(123)

        resp = client.post(
            "/api/game/new",
            json={"human_player": 0, "opponent_type": "heuristic", "board_size": 8},
        )
        session_id = resp.json()["session_id"]

        for _ in range(50):
            state = client.get(f"/api/game/{session_id}").json()["state"]
            if state["game_over"] or state["phase"] != "human_turn":
                break

            valid = state["valid_actions"]
            if not valid:
                break

            action = valid[rng.integers(len(valid))]["index"]
            client.post(f"/api/game/{session_id}/move", json={"action": action})


@pytest.mark.integration
class TestGameWithUndo:
    """Test undo functionality in a game."""

    def test_undo_and_continue(self, client):
        resp = client.post("/api/game/new", json={"human_player": 0, "board_size": 8})
        session_id = resp.json()["session_id"]
        state = resp.json()["state"]

        if state["phase"] == "human_turn" and state["valid_actions"]:
            action = state["valid_actions"][0]["index"]

            # Make a move
            client.post(f"/api/game/{session_id}/move", json={"action": action})

            # Undo
            undo_resp = client.post(f"/api/game/{session_id}/undo")
            assert undo_resp.status_code == 200

            # Make a different move
            state2 = client.get(f"/api/game/{session_id}").json()["state"]
            if state2["phase"] == "human_turn" and len(state2["valid_actions"]) > 1:
                alt_action = state2["valid_actions"][1]["index"]
                move_resp = client.post(
                    f"/api/game/{session_id}/move",
                    json={"action": alt_action},
                )
                assert move_resp.status_code == 200


@pytest.mark.integration
class TestGameResign:
    """Test resign flow."""

    def test_resign_game(self, client):
        resp = client.post("/api/game/new", json={"human_player": 0})
        session_id = resp.json()["session_id"]

        resign_resp = client.post(f"/api/game/{session_id}/resign")
        assert resign_resp.status_code == 200
        assert resign_resp.json()["state"]["game_over"] is True

        # Verify summary reflects game over
        summary = client.get(f"/api/game/{session_id}/summary").json()
        assert summary["game_over"] is True


@pytest.mark.integration
class TestConcurrentSessions:
    """Test concurrent game sessions."""

    def test_multiple_sessions_isolation(self, client):
        """Multiple sessions should be independent."""
        s1 = client.post("/api/game/new", json={"human_player": 0, "seed": 1}).json()
        s2 = client.post("/api/game/new", json={"human_player": 1, "seed": 2}).json()

        assert s1["session_id"] != s2["session_id"]

        # Play a move in s1
        state1 = s1["state"]
        if state1["phase"] == "human_turn" and state1["valid_actions"]:
            action = state1["valid_actions"][0]["index"]
            client.post(f"/api/game/{s1['session_id']}/move", json={"action": action})

        # s2 should be unchanged
        state2 = client.get(f"/api/game/{s2['session_id']}").json()["state"]
        assert state2["turn_number"] == 0 or state2["turn_number"] <= 1

    def test_three_concurrent_games(self, client):
        """Create and interact with 3 games simultaneously."""
        sessions = []
        for i in range(3):
            resp = client.post(
                "/api/game/new",
                json={"human_player": 0, "seed": i * 100},
            )
            sessions.append(resp.json()["session_id"])

        health = client.get("/api/health").json()
        assert health["active_sessions"] == 3

        # All should be accessible
        for sid in sessions:
            resp = client.get(f"/api/game/{sid}")
            assert resp.status_code == 200


@pytest.mark.integration
class TestWebSocketGameFlow:
    """Test WebSocket game interaction flow."""

    def test_websocket_full_flow(self, client):
        """Test move + AI response via WebSocket."""
        resp = client.post("/api/game/new", json={"human_player": 0, "board_size": 8})
        session_id = resp.json()["session_id"]
        state = resp.json()["state"]

        with client.websocket_connect(f"/ws/game/{session_id}") as ws:
            initial = ws.receive_json()
            assert initial["type"] == "state"

            if state["phase"] == "human_turn" and state["valid_actions"]:
                action = state["valid_actions"][0]["index"]
                ws.send_json({"type": "move", "action": action})

                # Collect all responses
                responses = []
                for _ in range(5):
                    try:
                        data = ws.receive_json()
                        responses.append(data)
                        if data["type"] in ("state", "game_over"):
                            break
                    except Exception:
                        break

                types = [r["type"] for r in responses]
                assert "move_result" in types

    def test_websocket_resign(self, client):
        """Test resign via WebSocket."""
        resp = client.post("/api/game/new", json={"human_player": 0})
        session_id = resp.json()["session_id"]

        with client.websocket_connect(f"/ws/game/{session_id}") as ws:
            ws.receive_json()  # initial state
            ws.send_json({"type": "resign"})

            responses = []
            for _ in range(3):
                try:
                    data = ws.receive_json()
                    responses.append(data)
                    if data["type"] == "game_over":
                        break
                except Exception:
                    break

            types = [r["type"] for r in responses]
            assert "game_over" in types


@pytest.mark.integration
class TestHTMXFlow:
    """Test HTMX page interaction flow."""

    def test_create_game_via_form(self, client):
        """Test creating game via HTML form POST."""
        response = client.post(
            "/game/new",
            data={
                "human_player": "0",
                "opponent_type": "random",
                "board_size": "8",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "/game/" in response.headers["location"]

    def test_htmx_move(self, client):
        """Test move via HTMX form submission."""
        resp = client.post("/api/game/new", json={"human_player": 0, "board_size": 8})
        session_id = resp.json()["session_id"]
        state = resp.json()["state"]

        if state["phase"] == "human_turn" and state["valid_actions"]:
            action = state["valid_actions"][0]["index"]
            response = client.post(
                f"/htmx/game/{session_id}/move",
                data={"action": str(action)},
            )
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_htmx_undo(self, client):
        """Test undo via HTMX."""
        resp = client.post("/api/game/new", json={"human_player": 0, "board_size": 8})
        session_id = resp.json()["session_id"]

        response = client.post(f"/htmx/game/{session_id}/undo")
        assert response.status_code == 200

    def test_htmx_resign(self, client):
        """Test resign via HTMX."""
        resp = client.post("/api/game/new", json={"human_player": 0})
        session_id = resp.json()["session_id"]

        response = client.post(f"/htmx/game/{session_id}/resign")
        assert response.status_code == 200


@pytest.fixture
def neural_web_config(tmp_path):
    """Web config wired to a real, tiny trained-shape checkpoint.

    Deliberately small on every axis so a game genuinely reaches game over
    inside the suite's timeout: MicroCiv only advances ``turn`` when the player
    index wraps, so a 200-turn board needs 400+ end_turn actions before it can
    terminate on its own.
    """
    from alpha_zero.core.civ6_actions import MicroCivActionSpace
    from alpha_zero.core.civ6_network import MicroCivNetwork
    from alpha_zero.web.config import WebConfig

    board_size = 8
    network = MicroCivNetwork(
        board_size=board_size,
        action_space_size=MicroCivActionSpace(board_size=board_size).total_size,
        num_res_blocks=2,
        num_filters=8,
    )
    checkpoint = tmp_path / "tiny.pt"
    network.save(str(checkpoint))

    return WebConfig(
        board_size=board_size,
        max_turns=3,
        default_thinking_time=0.01,
        max_concurrent_sessions=2,
        session_timeout_seconds=300.0,
        use_mcts=True,
        mcts_simulations=2,
        default_opponent="neural",
        checkpoint_path=str(checkpoint),
        device="cpu",
    )


@pytest.fixture
def neural_client(neural_web_config):
    from starlette.testclient import TestClient

    from alpha_zero.web.app import create_app

    return TestClient(create_app(neural_web_config))


@pytest.mark.integration
class TestNeuralOpponentAcceptance:
    """End-to-end acceptance for playing a trained model in the browser.

    This is the test the product claim rests on. Before the checkpoint-loader
    fix the "neural" opponent silently played random weights, and before the
    turn-ownership fix neither side controlled its own units — yet every
    existing e2e test still passed, because they only asserted status codes.
    """

    def test_neural_opponent_uses_the_checkpoint_weights(self, neural_web_config):
        """The session's network must match the checkpoint on disk."""
        from alpha_zero.utils.checkpoint import compute_weight_checksum, load_network
        from alpha_zero.web.session import GameSessionManager

        manager = GameSessionManager(neural_web_config)
        expected = load_network(neural_web_config.checkpoint_path, device="cpu")

        loaded = manager._get_shared_network()

        assert compute_weight_checksum(loaded) == pytest.approx(
            compute_weight_checksum(expected), rel=1e-6
        )

    def test_play_a_complete_game_against_the_neural_opponent(self, neural_client):
        """Play to a real game over, asserting turn ownership every step."""
        import numpy as np

        rng = np.random.default_rng(7)

        resp = neural_client.post(
            "/api/game/new",
            json={"human_player": 0, "opponent_type": "neural", "board_size": 8, "seed": 7},
        )
        assert resp.status_code == 200, resp.text
        session_id = resp.json()["session_id"]

        reached_game_over = False
        for _ in range(400):
            state = neural_client.get(f"/api/game/{session_id}").json()["state"]
            if state["game_over"]:
                reached_game_over = True
                break

            # Whenever the server says it is our turn, it must also agree that
            # the acting player is the human. Disagreement is the ownership bug.
            if state["phase"] == "human_turn":
                assert state["current_player"] == 0, (
                    "server offered the human a turn as the wrong player"
                )

            valid = state["valid_actions"]
            if not valid:
                break

            action = valid[rng.integers(len(valid))]["index"]
            move = neural_client.post(f"/api/game/{session_id}/move", json={"action": action})
            assert move.status_code == 200, move.text

        assert reached_game_over, "game never terminated; the acceptance path is not exercised"

    def test_action_groups_expose_city_actions(self, neural_client):
        """Founding a city must be reachable from the rendered page."""
        resp = neural_client.post(
            "/api/game/new",
            json={"human_player": 0, "opponent_type": "neural", "board_size": 8, "seed": 7},
        )
        session_id = resp.json()["session_id"]

        page = neural_client.get(f"/game/{session_id}")

        assert page.status_code == 200
        assert "Found City" in page.text, (
            "Found City missing from the page; it sits at action index 341+ and "
            "the old flat valid_actions[:20] cut hid it"
        )
