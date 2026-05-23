"""Tests for :class:`ChessAdvisor`."""

from __future__ import annotations

from unittest.mock import MagicMock

import chess
import numpy as np
import pytest

from alpha_zero.envs.chess_actions import ACTION_SPACE_SIZE, move_to_index
from alpha_zero.envs.chess_env import ChessEnv
from alpha_zero.orchestrator.chess_advisor import ChessAdvisor, ChessAdvisorConfig
from alpha_zero.utils.llm_client import LLMProvider, LLMResponse


def _llm_response_for(weights: dict[str, float]) -> LLMResponse:
    import json

    return LLMResponse(
        content=json.dumps(weights),
        model="mock",
        provider=LLMProvider.OPENAI,
    )


def _mock_llm_client(weights: dict[str, float]) -> MagicMock:
    client = MagicMock()
    client.generate_sync.return_value = _llm_response_for(weights)
    return client


class TestDisabledOrMissingClient:
    def test_returns_none_when_disabled(self) -> None:
        adv = ChessAdvisor(ChessAdvisorConfig(enabled=False), llm_client=MagicMock())
        assert adv.prior_for_board(chess.Board()) is None

    def test_returns_none_with_no_client(self) -> None:
        adv = ChessAdvisor(ChessAdvisorConfig(enabled=True))
        assert adv.prior_for_board(chess.Board()) is None


class TestConfigValidation:
    def test_invalid_prior_weight(self) -> None:
        with pytest.raises(ValueError, match="prior_weight"):
            ChessAdvisor(ChessAdvisorConfig(prior_weight=1.5))

    def test_negative_max_ply(self) -> None:
        with pytest.raises(ValueError, match="max_ply_for_prior"):
            ChessAdvisor(ChessAdvisorConfig(max_ply_for_prior=-1))

    def test_zero_cache_size(self) -> None:
        with pytest.raises(ValueError, match="cache_size"):
            ChessAdvisor(ChessAdvisorConfig(cache_size=0))

    def test_double_provided_raises(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            ChessAdvisor(ChessAdvisorConfig(), llm_client=MagicMock(), llm_config=MagicMock())


class TestHappyPath:
    def test_prior_normalised_over_legal(self) -> None:
        board = chess.Board()
        weights = {"e2e4": 4.0, "d2d4": 4.0, "g1f3": 2.0}
        client = _mock_llm_client(weights)
        adv = ChessAdvisor(
            ChessAdvisorConfig(enabled=True),
            llm_client=client,
        )
        prior = adv.prior_for_board(board)
        assert prior is not None
        assert prior.shape == (ACTION_SPACE_SIZE,)
        assert pytest.approx(prior.sum(), abs=1e-6) == 1.0
        # Indices for the suggested moves should hold the mass.
        for uci, w in weights.items():
            action = move_to_index(chess.Move.from_uci(uci))
            assert prior[action] == pytest.approx(w / sum(weights.values()), abs=1e-6)

    def test_illegal_moves_ignored(self) -> None:
        board = chess.Board()
        # h7h5 is illegal for white on first move; should be dropped.
        weights = {"e2e4": 5.0, "h7h5": 100.0}
        client = _mock_llm_client(weights)
        adv = ChessAdvisor(ChessAdvisorConfig(enabled=True), llm_client=client)
        prior = adv.prior_for_board(board)
        assert prior is not None
        legal_action = move_to_index(chess.Move.from_uci("e2e4"))
        assert prior[legal_action] == pytest.approx(1.0, abs=1e-6)


class TestPlyGating:
    def test_returns_none_past_max_ply(self) -> None:
        board = chess.Board()
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6"]:
            board.push(chess.Move.from_uci(uci))
        client = _mock_llm_client({"f1c4": 1.0})
        adv = ChessAdvisor(
            ChessAdvisorConfig(enabled=True, max_ply_for_prior=2),
            llm_client=client,
        )
        assert adv.prior_for_board(board) is None
        client.generate_sync.assert_not_called()


class TestCaching:
    def test_repeated_calls_hit_cache(self) -> None:
        board = chess.Board()
        client = _mock_llm_client({"e2e4": 1.0})
        adv = ChessAdvisor(ChessAdvisorConfig(enabled=True), llm_client=client)
        adv.prior_for_board(board)
        adv.prior_for_board(board)
        adv.prior_for_board(board)
        # Only one LLM call should have been issued; subsequent are cached.
        assert client.generate_sync.call_count == 1

    def test_reset_clears_cache(self) -> None:
        board = chess.Board()
        client = _mock_llm_client({"e2e4": 1.0})
        adv = ChessAdvisor(ChessAdvisorConfig(enabled=True), llm_client=client)
        adv.prior_for_board(board)
        adv.reset()
        adv.prior_for_board(board)
        assert client.generate_sync.call_count == 2


class TestGracefulDegradation:
    def test_invalid_json_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        client = MagicMock()
        client.generate_sync.return_value = LLMResponse(
            content="not valid json at all",
            model="mock",
            provider=LLMProvider.OPENAI,
        )
        adv = ChessAdvisor(ChessAdvisorConfig(enabled=True), llm_client=client)
        with caplog.at_level("WARNING"):
            assert adv.prior_for_board(chess.Board()) is None

    def test_llm_raises_returns_none(self) -> None:
        client = MagicMock()
        client.generate_sync.side_effect = RuntimeError("network down")
        adv = ChessAdvisor(ChessAdvisorConfig(enabled=True), llm_client=client)
        assert adv.prior_for_board(chess.Board()) is None

    def test_all_zero_weights_returns_none(self) -> None:
        client = _mock_llm_client({"e2e4": 0.0, "d2d4": 0.0})
        adv = ChessAdvisor(ChessAdvisorConfig(enabled=True), llm_client=client)
        assert adv.prior_for_board(chess.Board()) is None

    def test_non_dict_json_returns_none(self) -> None:
        client = MagicMock()
        client.generate_sync.return_value = LLMResponse(
            content="[1, 2, 3]",
            model="mock",
            provider=LLMProvider.OPENAI,
        )
        adv = ChessAdvisor(ChessAdvisorConfig(enabled=True), llm_client=client)
        assert adv.prior_for_board(chess.Board()) is None


class TestMCTSCallback:
    def test_callback_returns_zeros_when_disabled(self) -> None:
        env = ChessEnv(history_depth=1)
        env.reset()
        adv = ChessAdvisor(ChessAdvisorConfig(enabled=False), llm_client=MagicMock())
        cb = adv.mcts_callback()
        result = cb(env, env.get_valid_actions())
        assert np.allclose(result, 0.0)

    def test_callback_returns_prior_when_enabled(self) -> None:
        env = ChessEnv(history_depth=1)
        env.reset()
        client = _mock_llm_client({"e2e4": 1.0})
        adv = ChessAdvisor(ChessAdvisorConfig(enabled=True), llm_client=client)
        cb = adv.mcts_callback()
        prior = cb(env, env.get_valid_actions())
        assert prior.sum() == pytest.approx(1.0, abs=1e-6)
        legal = move_to_index(chess.Move.from_uci("e2e4"))
        assert prior[legal] == pytest.approx(1.0, abs=1e-6)

    def test_callback_returns_zeros_for_non_chess_env(self) -> None:
        adv = ChessAdvisor(
            ChessAdvisorConfig(enabled=True),
            llm_client=_mock_llm_client({"e2e4": 1.0}),
        )
        cb = adv.mcts_callback()
        # An env without a .board attribute.
        dummy = MagicMock(spec=[])
        result = cb(dummy, np.zeros(ACTION_SPACE_SIZE, dtype=np.float32))
        assert np.allclose(result, 0.0)
