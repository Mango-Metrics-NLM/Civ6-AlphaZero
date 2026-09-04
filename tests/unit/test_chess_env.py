"""Tests for ``ChessEnv`` — the MCTSEnvironment-compatible chess wrapper."""

from __future__ import annotations

import pytest

pytest.importorskip("chess", reason="requires the 'chess' extra")

import chess
import numpy as np

from alpha_zero.envs.chess_actions import ACTION_SPACE_SIZE, move_to_index
from alpha_zero.envs.chess_encoder import channels_for_history
from alpha_zero.envs.chess_env import (
    BLACK_PLAYER_INDEX,
    WHITE_PLAYER_INDEX,
    ChessEnv,
)


@pytest.fixture
def env() -> ChessEnv:
    return ChessEnv(history_depth=1)


class TestObservationContract:
    def test_obs_keys(self, env: ChessEnv) -> None:
        obs, _ = env.reset()
        assert set(obs.keys()) == {"spatial", "vector", "action_mask"}

    def test_obs_shapes_default(self, env: ChessEnv) -> None:
        obs, _ = env.reset()
        assert obs["spatial"].shape == (channels_for_history(1), 8, 8)
        assert obs["vector"].shape == (64,)
        assert obs["action_mask"].shape == (ACTION_SPACE_SIZE,)

    def test_obs_dtypes(self, env: ChessEnv) -> None:
        obs, _ = env.reset()
        assert obs["spatial"].dtype == np.float32
        assert obs["vector"].dtype == np.float32
        assert obs["action_mask"].dtype == np.float32

    def test_obs_shape_with_history_8(self) -> None:
        env = ChessEnv(history_depth=8)
        obs, _ = env.reset()
        assert obs["spatial"].shape == (119, 8, 8)

    def test_action_space_size(self, env: ChessEnv) -> None:
        assert env.action_space.n == ACTION_SPACE_SIZE

    def test_action_mask_matches_legal_moves_at_start(self, env: ChessEnv) -> None:
        obs, _ = env.reset()
        # 20 legal moves from the starting position (16 pawn + 4 knight).
        assert int(obs["action_mask"].sum()) == 20


class TestPlayerSemantics:
    def test_starts_with_white(self, env: ChessEnv) -> None:
        env.reset()
        assert env.current_player == WHITE_PLAYER_INDEX

    def test_switches_to_black_after_move(self, env: ChessEnv) -> None:
        env.reset()
        e4_index = move_to_index(chess.Move.from_uci("e2e4"))
        env.step(e4_index)
        assert env.current_player == BLACK_PLAYER_INDEX


class TestStep:
    def test_legal_action(self, env: ChessEnv) -> None:
        env.reset()
        e4_index = move_to_index(chess.Move.from_uci("e2e4"))
        obs, reward, terminated, truncated, info = env.step(e4_index)
        assert reward == 0.0
        assert terminated is False
        assert truncated is False
        assert obs["spatial"].shape[1:] == (8, 8)
        assert info["fullmove_number"] == 1

    def test_illegal_action_raises(self, env: ChessEnv) -> None:
        env.reset()
        # Index 0 corresponds to the a1 square + queen-plane-0 (N by 1) — not legal
        # since the a1 square holds a rook and the rook can't move there from start.
        with pytest.raises(ValueError, match="Illegal action"):
            env.step(0)

    def test_step_after_terminal_raises(self) -> None:
        # Fool's mate setup; black's last move checkmates white.
        env = ChessEnv(history_depth=1)
        env.reset()
        for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            env.step(move_to_index(chess.Move.from_uci(uci)))
        assert env.get_info()["terminated"] is True
        with pytest.raises(RuntimeError, match="terminated/truncated"):
            env.step(0)


class TestTerminalConditions:
    def test_checkmate_winner_and_reward(self) -> None:
        env = ChessEnv(history_depth=1)
        env.reset()
        # Fool's mate: black wins.
        moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
        rewards = []
        for uci in moves:
            _, reward, terminated, _, info = env.step(move_to_index(chess.Move.from_uci(uci)))
            rewards.append(reward)
        assert terminated is True
        # Black just moved and won.
        assert rewards[-1] == 1.0
        assert info["winner"] == BLACK_PLAYER_INDEX

    def test_stalemate_is_draw(self) -> None:
        # Classic stalemate position (black to move).
        env = ChessEnv(history_depth=1, starting_fen="7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        env.reset()
        # Black has no legal moves immediately; mark as terminal via outcome.
        outcome = env.board.outcome(claim_draw=False)
        assert outcome is not None
        assert outcome.winner is None  # stalemate → draw

    def test_truncation_at_max_plies(self) -> None:
        env = ChessEnv(history_depth=1, max_plies=2)
        env.reset()
        env.step(move_to_index(chess.Move.from_uci("e2e4")))
        _, _, terminated, truncated, info = env.step(move_to_index(chess.Move.from_uci("e7e5")))
        assert terminated is False
        assert truncated is True
        assert info["winner"] is None


class TestClone:
    def test_clone_is_independent(self) -> None:
        env = ChessEnv(history_depth=1)
        env.reset()
        env.step(move_to_index(chess.Move.from_uci("e2e4")))
        cloned = env.clone()
        # Mutate clone; original must not change.
        cloned.step(move_to_index(chess.Move.from_uci("e7e5")))
        assert env.board.fen() != cloned.board.fen()

    def test_clone_preserves_state(self) -> None:
        env = ChessEnv(history_depth=2)
        env.reset()
        env.step(move_to_index(chess.Move.from_uci("e2e4")))
        cloned = env.clone()
        assert cloned.board.fen() == env.board.fen()
        assert cloned.current_player == env.current_player
        assert cloned.history_depth == env.history_depth

    def test_clone_preserves_terminal_flags(self) -> None:
        env = ChessEnv(history_depth=1)
        env.reset()
        for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            env.step(move_to_index(chess.Move.from_uci(uci)))
        cloned = env.clone()
        assert cloned.get_info()["terminated"] is True


class TestValidation:
    def test_invalid_history_depth(self) -> None:
        with pytest.raises(ValueError):
            ChessEnv(history_depth=0)

    def test_invalid_max_plies(self) -> None:
        with pytest.raises(ValueError):
            ChessEnv(max_plies=0)

    def test_out_of_range_action(self, env: ChessEnv) -> None:
        env.reset()
        with pytest.raises(ValueError, match="out of range"):
            env.step(ACTION_SPACE_SIZE)


class TestRender:
    def test_ansi_render(self) -> None:
        env = ChessEnv(render_mode="ansi")
        env.reset()
        rendered = env.render()
        assert isinstance(rendered, str)
        assert "r" in rendered  # rook is on the board

    def test_no_render_mode_returns_none(self) -> None:
        env = ChessEnv()
        env.reset()
        assert env.render() is None


class TestFactoryIntegration:
    def test_factory_creates_chess_env(self) -> None:
        from alpha_zero.envs.env_factory import GAMEPLAY_PRESETS, create_game_env

        assert "chess" in GAMEPLAY_PRESETS
        env = create_game_env(gameplay_preset="chess", chess_history_depth=1)
        assert isinstance(env, ChessEnv)
        assert env.history_depth == 1

    def test_factory_unknown_preset_raises(self) -> None:
        from alpha_zero.envs.env_factory import create_game_env

        with pytest.raises(ValueError, match="Unknown gameplay_preset"):
            create_game_env(gameplay_preset="bogus")

    def test_factory_default_preset_unchanged(self) -> None:
        """Backwards-compat: default preset still produces a MicroCivEnv."""
        from alpha_zero.envs.env_factory import create_game_env
        from alpha_zero.envs.microciv import MicroCivEnv

        env = create_game_env(gameplay_preset="default", board_size=8, max_turns=10)
        assert isinstance(env, MicroCivEnv)
