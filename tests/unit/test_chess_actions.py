"""Tests for the AlphaZero 8x8x73 chess action encoding."""

from __future__ import annotations

import chess
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from alpha_zero.envs.chess_actions import (
    ACTION_SPACE_SIZE,
    BOARD_SIZE,
    NUM_PLANES,
    index_to_move,
    legal_action_mask,
    move_to_index,
)


class TestConstants:
    def test_action_space_size(self) -> None:
        assert ACTION_SPACE_SIZE == 8 * 8 * 73 == 4672

    def test_num_planes(self) -> None:
        assert NUM_PLANES == 73

    def test_board_size(self) -> None:
        assert BOARD_SIZE == 8


class TestMoveRoundtrip:
    """For every legal move from the starting position, encoding then decoding
    via the board context must recover the original move."""

    def test_starting_position_round_trip(self) -> None:
        board = chess.Board()
        for move in board.legal_moves:
            index = move_to_index(move)
            decoded = index_to_move(index, board)
            assert decoded == move, f"Roundtrip failed for {move.uci()}: got {decoded.uci()}"

    def test_knight_move(self) -> None:
        board = chess.Board()
        move = chess.Move.from_uci("g1f3")
        index = move_to_index(move)
        assert index_to_move(index, board) == move

    def test_pawn_double_push(self) -> None:
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        index = move_to_index(move)
        assert index_to_move(index, board) == move

    def test_queen_promotion_is_queen_plane(self) -> None:
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
        queen_promo = chess.Move.from_uci("a7a8q")
        index = move_to_index(queen_promo)
        decoded = index_to_move(index, board)
        # Decoder should auto-promote pawn-to-last-rank to queen.
        assert decoded == queen_promo

    @pytest.mark.parametrize("piece_uci", ["a7a8n", "a7a8b", "a7a8r"])
    def test_underpromotion_round_trip(self, piece_uci: str) -> None:
        board = chess.Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
        move = chess.Move.from_uci(piece_uci)
        index = move_to_index(move)
        assert index_to_move(index, board) == move

    @pytest.mark.parametrize(
        "fen,uci",
        [
            # Capture-left underpromotion to knight (white pawn on b7 takes a8).
            ("r6k/1P6/8/8/8/8/8/K7 w - - 0 1", "b7a8n"),
            # Capture-right underpromotion to rook.
            ("r6k/1P6/8/8/8/8/8/K7 w - - 0 1", "b7c8r"),
        ],
    )
    def test_capture_underpromotion(self, fen: str, uci: str) -> None:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        index = move_to_index(move)
        assert index_to_move(index, board) == move

    def test_black_underpromotion(self) -> None:
        board = chess.Board("k7/8/8/8/8/8/p7/K7 b - - 0 1")
        move = chess.Move.from_uci("a2a1n")
        index = move_to_index(move)
        assert index_to_move(index, board) == move

    def test_castling_kingside_is_queen_plane(self) -> None:
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        castle = chess.Move.from_uci("e1g1")
        index = move_to_index(castle)
        assert index_to_move(index, board) == castle


class TestErrorPaths:
    def test_index_out_of_range_raises(self) -> None:
        board = chess.Board()
        with pytest.raises(ValueError, match="out of range"):
            index_to_move(ACTION_SPACE_SIZE, board)
        with pytest.raises(ValueError, match="out of range"):
            index_to_move(-1, board)

    def test_unencodable_move_raises(self) -> None:
        # (drank=3, dfile=1) is neither queen-like (compass direction) nor a
        # knight offset, so the encoder must refuse it.
        bogus = chess.Move(chess.A1, chess.B4)
        with pytest.raises(ValueError):
            move_to_index(bogus)


class TestLegalActionMask:
    def test_mask_shape_and_dtype(self) -> None:
        board = chess.Board()
        mask = legal_action_mask(board)
        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.dtype == np.float32

    def test_mask_count_equals_legal_moves(self) -> None:
        board = chess.Board()
        mask = legal_action_mask(board)
        assert int(mask.sum()) == board.legal_moves.count()

    def test_mask_zero_at_terminal_position(self) -> None:
        # Fool's mate: black checkmates white.
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        mask = legal_action_mask(board)
        assert mask.sum() == 0


class TestPropertyRoundtrip:
    """Property-based round-trip test over many random positions."""

    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    @given(num_random_moves=st.integers(min_value=0, max_value=40))
    def test_random_position_round_trip(self, num_random_moves: int) -> None:
        board = chess.Board()
        rng = np.random.default_rng(num_random_moves)
        for _ in range(num_random_moves):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(legal[int(rng.integers(0, len(legal)))])
        for move in board.legal_moves:
            index = move_to_index(move)
            assert index_to_move(index, board) == move
