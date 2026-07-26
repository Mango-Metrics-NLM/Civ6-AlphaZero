"""Tests for web session management."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from alpha_zero.utils.compat import UTC
from alpha_zero.web.config import WebConfig
from alpha_zero.web.session import GameSessionManager


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

    def test_get_session_or_error_found(self, session_manager):
        session = session_manager.create_session()
        found = session_manager.get_session_or_error(session.session_id)
        assert found.session_id == session.session_id

    def test_get_session_or_error_missing(self, session_manager):
        with pytest.raises(KeyError, match="Session not found"):
            session_manager.get_session_or_error("nonexistent")

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


class TestCreateAIAgent:
    """Tests for _create_ai_agent routing and neural agent error paths."""

    def test_unknown_opponent_type_raises(self, session_manager):
        with pytest.raises(ValueError, match="Invalid opponent_type"):
            session_manager.create_session(opponent_type="unknown_type")

    def test_neural_no_checkpoint_raises(self, session_manager):
        # Config has no checkpoint_path by default
        with pytest.raises(RuntimeError, match="No checkpoint_path configured"):
            session_manager.create_session(opponent_type="neural")

    def test_neural_checkpoint_missing_file_raises(self, web_config):
        """Neural opponent with non-existent checkpoint path."""
        config = WebConfig(
            **{**web_config.to_dict(), "checkpoint_path": "/tmp/nonexistent_checkpoint.pt"}
        )
        manager = GameSessionManager(config)
        with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
            manager.create_session(opponent_type="neural")

    def test_random_agent_creates_successfully(self, session_manager):
        session = session_manager.create_session(opponent_type="random")
        assert session.ai_agent is not None
        assert session.opponent_type == "random"

    def test_heuristic_agent_creates_successfully(self, session_manager):
        session = session_manager.create_session(opponent_type="heuristic")
        assert session.ai_agent is not None
        assert session.opponent_type == "heuristic"


class TestNeuralAgentCreation:
    """Tests for neural agent creation and internal method error paths."""

    def test_create_neural_agent_no_checkpoint(self, web_config):
        """Session manager with no checkpoint_path raises RuntimeError for neural opponent."""
        # Default config has checkpoint_path=None
        manager = GameSessionManager(web_config)
        env = manager._create_environment(web_config.board_size)
        with pytest.raises(RuntimeError, match="No checkpoint_path configured"):
            manager._create_neural_agent(env)

    def test_create_ai_agent_unknown_type(self, session_manager):
        """Calling _create_ai_agent with unknown type raises ValueError (lines 331-332)."""
        env = session_manager._create_environment(8)
        with pytest.raises(ValueError, match="Unknown opponent type"):
            session_manager._create_ai_agent("alien", env)

    def test_create_baseline_unknown_type(self, session_manager):
        """Calling _create_baseline_agent with unknown type raises ValueError (lines 353-354)."""
        env = session_manager._create_environment(8)
        with pytest.raises(ValueError, match="Unknown baseline type"):
            session_manager._create_baseline_agent("turbo", env)


class TestGetSharedNetwork:
    """Tests for network caching in _get_shared_network."""

    @staticmethod
    def _write_checkpoint(tmp_path, board_size: int = 8):
        """Write a real, deliberately non-default checkpoint.

        Small on purpose (2 blocks / 8 filters) to stay well inside the global
        --timeout=120, and non-default so a loader that quietly built the
        6-block/64-filter default would fail rather than coincidentally pass.
        """
        from alpha_zero.core.civ6_actions import MicroCivActionSpace
        from alpha_zero.core.civ6_network import MicroCivNetwork

        network = MicroCivNetwork(
            board_size=board_size,
            action_space_size=MicroCivActionSpace(board_size=board_size).total_size,
            num_res_blocks=2,
            num_filters=8,
        )
        path = tmp_path / "checkpoint.pt"
        network.save(str(path))
        return path, network

    def test_shared_network_cached(self, web_config, tmp_path):
        """Calling _get_shared_network twice returns same object."""
        path, _ = self._write_checkpoint(tmp_path)
        config = WebConfig(
            **{**web_config.to_dict(), "checkpoint_path": str(path), "board_size": 8}
        )
        manager = GameSessionManager(config)

        net1 = manager._get_shared_network()
        net2 = manager._get_shared_network()
        assert net1 is net2

    def test_shared_network_loads_real_weights(self, web_config, tmp_path):
        """The loaded network must carry the checkpoint's weights.

        This is the regression guard for the bug that made the web UI's
        "neural" opponent play with random weights: _get_shared_network read a
        `.state_dict` attribute that CheckpointLoadResult does not have (the
        field is `.data`), so no weights were ever loaded and it still logged
        success. The old version of this test patched safe_load_checkpoint to
        return a plain dict — a type the real function never produces — so the
        mock exercised a branch the production path could not reach.
        """
        from alpha_zero.utils.checkpoint import compute_weight_checksum

        path, saved = self._write_checkpoint(tmp_path)
        config = WebConfig(
            **{**web_config.to_dict(), "checkpoint_path": str(path), "board_size": 8}
        )
        manager = GameSessionManager(config)

        loaded = manager._get_shared_network()

        assert compute_weight_checksum(loaded) == pytest.approx(
            compute_weight_checksum(saved), rel=1e-6
        )
        assert not loaded.training, "network must be in eval mode for inference"

    def test_shared_network_respects_checkpoint_architecture(self, web_config, tmp_path):
        """The architecture comes from the checkpoint, not from defaults."""
        path, _ = self._write_checkpoint(tmp_path)
        config = WebConfig(
            **{**web_config.to_dict(), "checkpoint_path": str(path), "board_size": 8}
        )

        loaded = GameSessionManager(config)._get_shared_network()

        assert loaded.config.num_res_blocks == 2
        assert loaded.config.num_filters == 8
        assert loaded.config.spatial_size == 8

    def test_board_size_mismatch_is_rejected(self, web_config, tmp_path):
        """A board-size mismatch must fail loudly, not mid-game."""
        path, _ = self._write_checkpoint(tmp_path, board_size=8)
        config = WebConfig(
            **{**web_config.to_dict(), "checkpoint_path": str(path), "board_size": 16}
        )
        manager = GameSessionManager(config)

        with pytest.raises(ValueError, match="board_size"):
            manager._get_shared_network()

    def test_shared_network_no_checkpoint_raises(self, web_config):
        """_get_shared_network raises when checkpoint_path is None."""
        manager = GameSessionManager(web_config)
        with pytest.raises(RuntimeError, match="No checkpoint"):
            manager._get_shared_network()
