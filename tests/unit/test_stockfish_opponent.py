"""Tests for :class:`StockfishOpponent`.

These tests do **not** require a real Stockfish binary. The engine
subprocess is replaced with a ``MagicMock`` that mimics the
``chess.engine.SimpleEngine`` API surface we use.

A single ``@pytest.mark.integration`` test is provided that requires
the ``STOCKFISH_PATH`` env var; if unset it is skipped.
"""

from __future__ import annotations

import pytest

pytest.importorskip("chess", reason="requires the 'chess' extra")

import os
from unittest.mock import MagicMock, patch

import chess
import chess.engine

from alpha_zero.envs.chess_actions import (
    ACTION_SPACE_SIZE,
    legal_action_mask,
    move_to_index,
)
from alpha_zero.envs.chess_env import ChessEnv
from alpha_zero.opponents.stockfish_opponent import (
    STOCKFISH_ELO_MAX,
    STOCKFISH_SKILL_MAX,
    StockfishOpponent,
    StockfishOpponentConfig,
)


def _make_engine_mock(move: chess.Move) -> MagicMock:
    """Return a MagicMock pretending to be a SimpleEngine returning ``move``."""
    engine = MagicMock(spec=chess.engine.SimpleEngine)
    engine.play.return_value = chess.engine.PlayResult(move=move, ponder=None)
    return engine


class TestConfigValidation:
    def test_skill_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="skill"):
            StockfishOpponentConfig(skill=STOCKFISH_SKILL_MAX + 1)

    def test_elo_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="elo"):
            StockfishOpponentConfig(elo=STOCKFISH_ELO_MAX + 1)

    def test_resolved_path_uses_config(self) -> None:
        cfg = StockfishOpponentConfig(path="/bin/sf")
        assert cfg.resolved_path() == "/bin/sf"

    def test_resolved_path_falls_back_to_env(self) -> None:
        cfg = StockfishOpponentConfig()
        with patch.dict(os.environ, {"STOCKFISH_PATH": "/env/sf"}, clear=True):
            assert cfg.resolved_path() == "/env/sf"

    def test_resolved_path_missing_raises(self) -> None:
        cfg = StockfishOpponentConfig()
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="not configured"):
                cfg.resolved_path()


class TestUCIOptions:
    def test_skill_mode_applied(self) -> None:
        engine = _make_engine_mock(chess.Move.from_uci("e2e4"))
        cfg = StockfishOpponentConfig(skill=3, threads=2, hash_mb=64)
        op = StockfishOpponent(cfg, engine=engine)
        try:
            opts = engine.configure.call_args.args[0]
            assert opts["Skill Level"] == 3
            assert opts["Threads"] == 2
            assert opts["Hash"] == 64
            assert "UCI_LimitStrength" not in opts
        finally:
            op.close()

    def test_elo_mode_applied(self) -> None:
        engine = _make_engine_mock(chess.Move.from_uci("e2e4"))
        cfg = StockfishOpponentConfig(use_elo=True, elo=1800)
        op = StockfishOpponent(cfg, engine=engine)
        try:
            opts = engine.configure.call_args.args[0]
            assert opts["UCI_LimitStrength"] is True
            assert opts["UCI_Elo"] == 1800
            assert "Skill Level" not in opts
        finally:
            op.close()

    def test_set_skill_changes_mode(self) -> None:
        engine = _make_engine_mock(chess.Move.from_uci("e2e4"))
        cfg = StockfishOpponentConfig(use_elo=True, elo=1800)
        op = StockfishOpponent(cfg, engine=engine)
        try:
            engine.configure.reset_mock()
            op.set_skill(7)
            opts = engine.configure.call_args.args[0]
            assert opts["Skill Level"] == 7
            assert "UCI_LimitStrength" not in opts
            assert op.config.use_elo is False
        finally:
            op.close()

    def test_set_elo_changes_mode(self) -> None:
        engine = _make_engine_mock(chess.Move.from_uci("e2e4"))
        cfg = StockfishOpponentConfig(skill=5)
        op = StockfishOpponent(cfg, engine=engine)
        try:
            engine.configure.reset_mock()
            op.set_elo(2200)
            opts = engine.configure.call_args.args[0]
            assert opts["UCI_Elo"] == 2200
            assert opts["UCI_LimitStrength"] is True
            assert op.config.use_elo is True
        finally:
            op.close()

    def test_set_skill_invalid_raises(self) -> None:
        engine = _make_engine_mock(chess.Move.from_uci("e2e4"))
        op = StockfishOpponent(StockfishOpponentConfig(), engine=engine)
        try:
            with pytest.raises(ValueError, match="out of range"):
                op.set_skill(STOCKFISH_SKILL_MAX + 1)
        finally:
            op.close()

    def test_set_elo_invalid_raises(self) -> None:
        engine = _make_engine_mock(chess.Move.from_uci("e2e4"))
        op = StockfishOpponent(StockfishOpponentConfig(), engine=engine)
        try:
            with pytest.raises(ValueError, match="out of range"):
                op.set_elo(STOCKFISH_ELO_MAX + 1)
        finally:
            op.close()


class TestGetAction:
    def test_returns_encoded_legal_move(self) -> None:
        env = ChessEnv(history_depth=1)
        env.reset()
        expected_move = chess.Move.from_uci("e2e4")
        engine = _make_engine_mock(expected_move)
        op = StockfishOpponent(StockfishOpponentConfig(), engine=engine)
        try:
            mask = legal_action_mask(env.board)
            action = op.get_action(env, mask)
            assert action == move_to_index(expected_move)
            assert mask[action] == 1.0
        finally:
            op.close()

    def test_falls_back_on_engine_error(self) -> None:
        env = ChessEnv(history_depth=1)
        env.reset()
        engine = MagicMock(spec=chess.engine.SimpleEngine)
        engine.play.side_effect = chess.engine.EngineError("boom")
        op = StockfishOpponent(StockfishOpponentConfig(), engine=engine)
        try:
            mask = legal_action_mask(env.board)
            action = op.get_action(env, mask)
            assert mask[action] == 1.0
        finally:
            op.close()

    def test_falls_back_when_engine_returns_no_move(self) -> None:
        env = ChessEnv(history_depth=1)
        env.reset()
        engine = MagicMock(spec=chess.engine.SimpleEngine)
        engine.play.return_value = chess.engine.PlayResult(move=None, ponder=None)
        op = StockfishOpponent(StockfishOpponentConfig(), engine=engine)
        try:
            mask = legal_action_mask(env.board)
            action = op.get_action(env, mask)
            assert mask[action] == 1.0
        finally:
            op.close()

    def test_falls_back_on_masked_action(self) -> None:
        env = ChessEnv(history_depth=1)
        env.reset()
        # Pretend engine wants to push pawn e2e4, but our mask forbids it.
        engine = _make_engine_mock(chess.Move.from_uci("e2e4"))
        op = StockfishOpponent(StockfishOpponentConfig(), engine=engine)
        try:
            mask = legal_action_mask(env.board)
            forbidden_idx = move_to_index(chess.Move.from_uci("e2e4"))
            mask[forbidden_idx] = 0.0
            action = op.get_action(env, mask)
            assert mask[action] == 1.0
            assert action != forbidden_idx
        finally:
            op.close()

    def test_fallback_warning_emitted_once(self, caplog: pytest.LogCaptureFixture) -> None:
        env = ChessEnv(history_depth=1)
        env.reset()
        engine = MagicMock(spec=chess.engine.SimpleEngine)
        engine.play.side_effect = chess.engine.EngineError("boom")
        op = StockfishOpponent(StockfishOpponentConfig(), engine=engine)
        try:
            mask = legal_action_mask(env.board)
            with caplog.at_level("WARNING"):
                op.get_action(env, mask)
                op.get_action(env, mask)
                op.get_action(env, mask)
            warnings = [r for r in caplog.records if r.levelname == "WARNING"]
            # Only the first fallback should log a warning.
            assert sum("falling back" in r.message for r in warnings) == 1
        finally:
            op.close()


class TestLifecycle:
    def test_close_quits_engine(self) -> None:
        engine = _make_engine_mock(chess.Move.from_uci("e2e4"))
        op = StockfishOpponent(StockfishOpponentConfig(), engine=engine)
        op.close()
        engine.quit.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        engine = _make_engine_mock(chess.Move.from_uci("e2e4"))
        op = StockfishOpponent(StockfishOpponentConfig(), engine=engine)
        op.close()
        op.close()
        engine.quit.assert_called_once()

    def test_context_manager_closes(self) -> None:
        engine = _make_engine_mock(chess.Move.from_uci("e2e4"))
        with StockfishOpponent(StockfishOpponentConfig(), engine=engine) as op:
            assert op.name == "stockfish"
        engine.quit.assert_called_once()

    def test_get_action_after_close_falls_back(self) -> None:
        env = ChessEnv(history_depth=1)
        env.reset()
        engine = _make_engine_mock(chess.Move.from_uci("e2e4"))
        op = StockfishOpponent(StockfishOpponentConfig(), engine=engine)
        op.close()
        mask = legal_action_mask(env.board)
        action = op.get_action(env, mask)
        assert 0 <= action < ACTION_SPACE_SIZE
        assert mask[action] == 1.0


class TestConfigureLimit:
    def test_configure_limit_overrides(self) -> None:
        engine = _make_engine_mock(chess.Move.from_uci("e2e4"))
        op = StockfishOpponent(StockfishOpponentConfig(depth_limit=10), engine=engine)
        try:
            op.configure_limit(depth=None, time_s=0.5, nodes=10000)
            assert op.config.depth_limit is None
            assert op.config.time_limit_s == 0.5
            assert op.config.node_limit == 10000
            limit = op._build_limit()  # private helper, accessed for white-box check
            assert limit.depth is None
            assert limit.time == 0.5
            assert limit.nodes == 10000
        finally:
            op.close()


@pytest.mark.integration
def test_real_stockfish_smoke() -> None:
    """Runs a real Stockfish round-trip — skipped if binary not available."""
    path = os.getenv("STOCKFISH_PATH")
    if not path or not os.path.exists(path):
        pytest.skip("STOCKFISH_PATH not set or binary missing")
    env = ChessEnv(history_depth=1)
    env.reset()
    cfg = StockfishOpponentConfig(path=path, skill=0, depth_limit=1)
    with StockfishOpponent(cfg) as op:
        mask = legal_action_mask(env.board)
        action = op.get_action(env, mask)
    assert mask[action] == 1.0
