"""Tests for web session management."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from alpha_zero.web.config import WebConfig
from alpha_zero.web.session import GameSession, GameSessionManager


@pytest.fixture
def web_config():
    """Create a test web configuration with small board for speed."""
    return WebConfig(
        board_size=8,
        max_turns=50,
        default_thinking_time=0.1,
        max_concurrent_sessions=3,
        session_timeout_seconds=60.0,
        use_mcts=False,
        mcts_simulations=2,
        default_opponent="random",
    )


@pytest.fixture
def session_manager(web_config):
    """Create a GameSessionManager for testing."""
    return GameSessionManager(web_config)


class TestGameSession:
    """Tests for GameSession dataclass."""

    def test_touch_updates_last_activity(self, session_manager):
        session = session_manager.create_session()
        old_time = session.last_activity
        # Small sleep alternative: just touch and check it changed
        session.touch()
        assert session.last_activity >= old_time

    def test_is_expired_false_when_fresh(self, session_manager):
        session = session_manager.create_session()
        assert not session.is_expired

    def test_is_expired_true_when_old(self, session_manager):
        session = session_manager.create_session()
        # Manually backdate
        session.last_activity = datetime.now(UTC) - timedelta(seconds=3600)
        assert session.is_expired

    def test_age_seconds(self, session_manager):
        session = session_manager.create_session()
        assert session.age_seconds >= 0
        assert session.age_seconds < 5  # Should be nearly instant


class TestGameSessionManager:
    """Tests for GameSessionManager."""

    def test_init(self, web_config):
        manager = GameSessionManager(web_config)
        assert manager.active_count == 0

    def test_create_session_random(self, session_manager):
        session = session_manager.create_session(opponent_type="random")
        assert session is not None
        assert session.session_id
        assert session.opponent_type == "random"
        assert session.controller is not None
        assert session_manager.active_count == 1

    def test_create_session_heuristic(self, session_manager):
        session = session_manager.create_session(opponent_type="heuristic")
        assert session.opponent_type == "heuristic"

    def test_create_session_default_opponent(self, session_manager):
        session = session_manager.create_session()
        assert session.opponent_type == "random"

    def test_create_session_human_player_0(self, session_manager):
        session = session_manager.create_session(human_player=0)
        assert session.human_player == 0

    def test_create_session_human_player_1(self, session_manager):
        session = session_manager.create_session(human_player=1)
        assert session.human_player == 1

    def test_create_session_invalid_human_player(self, session_manager):
        with pytest.raises(ValueError, match="human_player must be 0 or 1"):
            session_manager.create_session(human_player=2)

    def test_create_session_invalid_opponent(self, session_manager):
        with pytest.raises(ValueError, match="Invalid opponent_type"):
            session_manager.create_session(opponent_type="invalid")

    def test_create_session_with_seed(self, session_manager):
        s1 = session_manager.create_session(seed=42)
        s2 = session_manager.create_session(seed=42)
        # Both should start but be different sessions
        assert s1.session_id != s2.session_id

    def test_create_session_custom_board_size(self, session_manager):
        session = session_manager.create_session(board_size=12)
        assert session.adapter.board_size == 12

    def test_create_session_starts_game(self, session_manager):
        session = session_manager.create_session()
        # Game should be started
        state = session.controller.current_state
        assert state.phase.value in ("human_turn", "ai_turn", "in_progress")

    def test_get_session_existing(self, session_manager):
        session = session_manager.create_session()
        found = session_manager.get_session(session.session_id)
        assert found is not None
        assert found.session_id == session.session_id

    def test_get_session_nonexistent(self, session_manager):
        assert session_manager.get_session("nonexistent") is None

    def test_remove_session(self, session_manager):
        session = session_manager.create_session()
        sid = session.session_id
        assert session_manager.remove_session(sid)
        assert session_manager.get_session(sid) is None
        assert session_manager.active_count == 0

    def test_remove_session_nonexistent(self, session_manager):
        assert not session_manager.remove_session("nonexistent")

    def test_max_concurrent_sessions(self, session_manager):
        # Config allows 3 max
        session_manager.create_session()
        session_manager.create_session()
        session_manager.create_session()
        with pytest.raises(RuntimeError, match="Maximum concurrent sessions"):
            session_manager.create_session()

    def test_max_sessions_cleanup_expired(self, session_manager):
        # Create 3 sessions, expire one, should allow 4th
        s1 = session_manager.create_session()
        session_manager.create_session()
        session_manager.create_session()

        # Expire the first session
        s1.last_activity = datetime.now(UTC) - timedelta(seconds=3600)

        # Should succeed because cleanup removes expired
        s4 = session_manager.create_session()
        assert s4 is not None

    def test_cleanup_expired(self, session_manager):
        s1 = session_manager.create_session()
        s2 = session_manager.create_session()

        # Expire s1
        s1.last_activity = datetime.now(UTC) - timedelta(seconds=3600)

        removed = session_manager.cleanup_expired()
        assert removed == 1
        assert session_manager.active_count == 1
        assert session_manager.get_session(s2.session_id) is not None

    def test_cleanup_expired_none_expired(self, session_manager):
        session_manager.create_session()
        removed = session_manager.cleanup_expired()
        assert removed == 0

    def test_list_sessions(self, session_manager):
        session_manager.create_session(opponent_type="random")
        session_manager.create_session(opponent_type="heuristic")

        sessions = session_manager.list_sessions()
        assert len(sessions) == 2
        assert all("session_id" in s for s in sessions)
        assert all("opponent_type" in s for s in sessions)
        assert all("phase" in s for s in sessions)

    def test_list_sessions_empty(self, session_manager):
        assert session_manager.list_sessions() == []

    def test_neural_opponent_requires_checkpoint(self, session_manager):
        with pytest.raises(RuntimeError, match="No checkpoint_path configured"):
            session_manager.create_session(opponent_type="neural")


class TestGameSessionGameplay:
    """Test actual gameplay through sessions."""

    def test_play_human_move(self, session_manager):
        session = session_manager.create_session(human_player=0)
        controller = session.controller

        if controller.is_human_turn:
            valid = controller.valid_actions
            assert len(valid) > 0
            result = controller.play_human_move(valid[0])
            assert result.is_human
            assert result.action == valid[0]

    def test_play_ai_move(self, session_manager):
        session = session_manager.create_session(human_player=0)
        controller = session.controller

        # If human goes first, play a move then AI
        if controller.is_human_turn:
            valid = controller.valid_actions
            controller.play_human_move(valid[0])

        if not controller.is_game_over and not controller.is_human_turn:
            result = controller.play_ai_move()
            assert not result.is_human

    def test_full_game_random_actions(self, session_manager):
        """Play a short game with random moves from both sides."""
        import numpy as np

        session = session_manager.create_session(human_player=0, seed=42)
        controller = session.controller
        rng = np.random.default_rng(42)

        for _ in range(100):
            if controller.is_game_over:
                break

            if controller.is_human_turn:
                valid = controller.valid_actions
                if valid:
                    action = valid[rng.integers(len(valid))]
                    controller.play_human_move(action)
            else:
                controller.play_ai_move()

    def test_undo_move(self, session_manager):
        session = session_manager.create_session(human_player=0)
        controller = session.controller

        if controller.is_human_turn:
            valid = controller.valid_actions
            controller.play_human_move(valid[0])
            success = controller.undo_move()
            assert success

    def test_resign(self, session_manager):
        session = session_manager.create_session(human_player=0)
        result = session.controller.resign()
        assert result.game_over
        assert session.controller.is_game_over

    def test_session_action_space(self, session_manager):
        session = session_manager.create_session(board_size=8)
        assert session.action_space is not None
        assert session.action_space.total_size > 0
