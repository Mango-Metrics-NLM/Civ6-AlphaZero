"""Tests for the configurable chess state encoder."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from alpha_zero.envs.chess_encoder import (
    MAX_HISTORY_DEPTH,
    MIN_HISTORY_DEPTH,
    PLANES_PER_TIMESTEP,
    channels_for_history,
    encode_board,
    encode_vector,
)


class TestChannelCount:
    @pytest.mark.parametrize(
        "history_depth,expected_channels",
        [
            (1, 14 * 1 + 7),  # 21
            (2, 14 * 2 + 7),  # 35
            (4, 14 * 4 + 7),  # 63
            (8, 14 * 8 + 7),  # 119 — AlphaZero paper value
        ],
    )
    def test_channels_for_history(self, history_depth: int, expected_channels: int) -> None:
        assert channels_for_history(history_depth) == expected_channels

    def test_alphazero_paper_119(self) -> None:
        assert channels_for_history(MAX_HISTORY_DEPTH) == 119

    def test_invalid_history_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="history_depth"):
            channels_for_history(0)
        with pytest.raises(ValueError, match="history_depth"):
            channels_for_history(MAX_HISTORY_DEPTH + 1)


class TestEncodeBoardShape:
    @pytest.mark.parametrize("history_depth", [1, 2, 4, 8])
    def test_shape_matches_channels(self, history_depth: int) -> None:
        board = chess.Board()
        spatial = encode_board(board, history_depth=history_depth)
        assert spatial.shape == (channels_for_history(history_depth), 8, 8)
        assert spatial.dtype == np.float32


class TestStartingPosition:
    def test_white_to_move_starting_position(self) -> None:
        board = chess.Board()
        spatial = encode_board(board, history_depth=1)
        # First 6 planes: side-to-move (white) pieces.
        # Pawn plane index = 0; rank 1 (white pawn rank) should be all 1s.
        assert np.all(spatial[0, 1, :] == 1.0)
        # Rook plane index 3 should have white rooks at a1 (rank 0, file 0)
        # and h1 (rank 0, file 7).
        assert spatial[3, 0, 0] == 1.0
        assert spatial[3, 0, 7] == 1.0

    def test_side_to_move_aux_plane(self) -> None:
        board = chess.Board()
        spatial = encode_board(board, history_depth=1)
        aux_start = PLANES_PER_TIMESTEP
        # First aux plane = side-to-move (1.0 for white).
        assert np.all(spatial[aux_start, :, :] == 1.0)
        # Apply a move so it's black's turn.
        board.push(chess.Move.from_uci("e2e4"))
        spatial_black = encode_board(board, history_depth=1)
        assert np.all(spatial_black[aux_start, :, :] == 0.0)

    def test_castling_rights_planes(self) -> None:
        board = chess.Board()
        spatial = encode_board(board, history_depth=1)
        aux_start = PLANES_PER_TIMESTEP
        # All four castling rights should be 1.0 at starting position.
        for offset in range(1, 5):
            assert np.all(spatial[aux_start + offset, :, :] == 1.0)


class TestHistoryEncoding:
    def test_zero_padding_when_history_exceeds_stack(self) -> None:
        # Starting position has empty stack; with history_depth=4, timesteps 1..3
        # are zero-filled.
        board = chess.Board()
        spatial = encode_board(board, history_depth=4)
        # Each subsequent timestep (after current) should be all zeros.
        for t in range(1, 4):
            start = t * PLANES_PER_TIMESTEP
            timestep_block = spatial[start : start + PLANES_PER_TIMESTEP]
            assert np.all(timestep_block == 0.0)

    def test_history_reflects_past_moves(self) -> None:
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        spatial = encode_board(board, history_depth=2)
        # Timestep 0 = after e2e4 (black to move, board rotated for black's POV).
        # Timestep 1 = starting position (white to move). The encoding orients
        # from current side-to-move so timestep 1 will be from black's POV too.
        # We only assert it's non-empty.
        ts1_start = PLANES_PER_TIMESTEP
        ts1_block = spatial[ts1_start : ts1_start + PLANES_PER_TIMESTEP]
        assert ts1_block.sum() > 0


class TestRepetitionPlanes:
    def test_no_repetition_at_start(self) -> None:
        board = chess.Board()
        spatial = encode_board(board, history_depth=1)
        # Repetition planes are the last 2 of the first timestep (indices 12, 13).
        assert np.all(spatial[12, :, :] == 0.0)
        assert np.all(spatial[13, :, :] == 0.0)


class TestEncodeVector:
    def test_default_size(self) -> None:
        board = chess.Board()
        v = encode_vector(board)
        assert v.shape == (64,)
        assert v.dtype == np.float32

    def test_minimum_size_enforced(self) -> None:
        board = chess.Board()
        with pytest.raises(ValueError, match="vector_size"):
            encode_vector(board, vector_size=4)

    def test_side_to_move_first_element(self) -> None:
        board = chess.Board()
        v = encode_vector(board)
        assert v[0] == 1.0  # white to move
        board.push(chess.Move.from_uci("e2e4"))
        v_after = encode_vector(board)
        assert v_after[0] == -1.0  # black to move


class TestBounds:
    def test_min_max_history(self) -> None:
        assert MIN_HISTORY_DEPTH == 1
        assert MAX_HISTORY_DEPTH == 8
