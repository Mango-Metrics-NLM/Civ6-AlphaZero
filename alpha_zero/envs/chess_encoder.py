"""Configurable AlphaZero-style chess state encoder.

AlphaZero's chess input is a stack of 119 planes encoding 8 positions of
history plus auxiliary features. For demo/short-training scenarios this is
expensive; the encoder here accepts a ``history_depth`` parameter T ∈ [1, 8]
and produces a smaller representation when T < 8.

Per-timestep planes (14 planes each, total ``14 * T``):

- 12 piece planes: 6 piece types x 2 colours, oriented from the *side to move*'s
  perspective (so white pieces are always on the "near" ranks).
- 2 repetition planes: a 1.0 on the whole board when the position has occurred
  N ≥ 2 / N ≥ 3 times respectively.

Trailing scalar planes (7 planes):

- side-to-move (1 plane), castling rights (4 planes: K/Q for both colours),
  no-progress count (1 plane, fullmove halfmove clock / 100), total move count
  (1 plane, fullmove number / 100).

Total channels for given T: ``14 * T + 7``. So T=1 → 21 channels, T=8 → 119.

Orientation: when it's black's turn, the board is rotated 180° so the side
to move always plays "upward." This matches AlphaZero. Action indices are
not affected — action encoding stays in absolute coordinates.
"""

from __future__ import annotations

from typing import Final

import chess
import numpy as np

from alpha_zero.envs.chess_actions import BOARD_SIZE

PIECE_TYPES: Final[tuple[int, ...]] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
_PIECE_PLANES_PER_COLOR: Final[int] = len(PIECE_TYPES)  # 6
_PIECE_PLANES: Final[int] = _PIECE_PLANES_PER_COLOR * 2  # 12
_REPETITION_PLANES: Final[int] = 2
PLANES_PER_TIMESTEP: Final[int] = _PIECE_PLANES + _REPETITION_PLANES  # 14
AUX_PLANES: Final[int] = 7
MIN_HISTORY_DEPTH: Final[int] = 1
MAX_HISTORY_DEPTH: Final[int] = 8


def channels_for_history(history_depth: int) -> int:
    """Return the total channel count for a given history depth."""
    _validate_history_depth(history_depth)
    return PLANES_PER_TIMESTEP * history_depth + AUX_PLANES


def _validate_history_depth(history_depth: int) -> None:
    if not MIN_HISTORY_DEPTH <= history_depth <= MAX_HISTORY_DEPTH:
        raise ValueError(
            f"history_depth must be in [{MIN_HISTORY_DEPTH}, {MAX_HISTORY_DEPTH}], "
            f"got {history_depth}"
        )


def _piece_planes(board: chess.Board, side_to_move: bool) -> np.ndarray:
    """Return (12, 8, 8) piece planes oriented from ``side_to_move``'s POV."""
    planes = np.zeros((_PIECE_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        if side_to_move == chess.BLACK:
            rank = BOARD_SIZE - 1 - rank
            file = BOARD_SIZE - 1 - file
        piece_idx = PIECE_TYPES.index(piece.piece_type)
        # First 6 planes: side-to-move's pieces; next 6: opponent's.
        if piece.color == side_to_move:
            channel = piece_idx
        else:
            channel = _PIECE_PLANES_PER_COLOR + piece_idx
        planes[channel, rank, file] = 1.0
    return planes


def _repetition_planes(board: chess.Board) -> np.ndarray:
    """Return (2, 8, 8) constant repetition planes.

    Plane 0 is filled with 1.0 if the position has occurred at least twice;
    plane 1 if at least three times (threefold repetition imminent).
    """
    planes = np.zeros((_REPETITION_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    if board.is_repetition(2):
        planes[0, :, :] = 1.0
    if board.is_repetition(3):
        planes[1, :, :] = 1.0
    return planes


def _aux_planes(board: chess.Board) -> np.ndarray:
    """Return (7, 8, 8) auxiliary planes for side-to-move, castling, and clocks."""
    planes = np.zeros((AUX_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    # 0: side to move (1.0 for white).
    planes[0, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    # 1-4: castling rights for [white K, white Q, black K, black Q].
    planes[1, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[2, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[3, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[4, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    # 5: halfmove clock / 100 (50-move rule progress).
    planes[5, :, :] = min(board.halfmove_clock / 100.0, 1.0)
    # 6: fullmove number / 100 (rough game-progress signal, clipped at 1.0).
    planes[6, :, :] = min(board.fullmove_number / 100.0, 1.0)
    return planes


def encode_board(board: chess.Board, history_depth: int = 1) -> np.ndarray:
    """Encode a ``chess.Board`` into a spatial tensor of shape ``(C, 8, 8)``.

    Args:
        board: Current board (may have a non-empty move stack for history).
        history_depth: Number of past positions to include (1..8). Older
            positions are reconstructed by popping moves; if the stack is
            shorter than ``history_depth - 1``, the missing slots are zero-
            filled (no synthetic positions).

    Returns:
        ``np.ndarray`` of dtype float32 and shape ``(C, 8, 8)`` where
        ``C = channels_for_history(history_depth)``.
    """
    _validate_history_depth(history_depth)
    side_to_move = board.turn
    channels = channels_for_history(history_depth)
    out = np.zeros((channels, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Walk back through history, writing each timestep's planes.
    # Timestep 0 = current; timestep T-1 = oldest in window.
    working = board.copy(stack=True)
    for t in range(history_depth):
        start = t * PLANES_PER_TIMESTEP
        out[start : start + _PIECE_PLANES] = _piece_planes(working, side_to_move)
        out[start + _PIECE_PLANES : start + PLANES_PER_TIMESTEP] = _repetition_planes(working)
        if working.move_stack:
            working.pop()
        else:
            # No more history. Remaining timesteps are zero-filled (already initialized).
            break

    # Auxiliary planes always come from the *current* position.
    aux_start = PLANES_PER_TIMESTEP * history_depth
    out[aux_start : aux_start + AUX_PLANES] = _aux_planes(board)
    return out


def encode_vector(board: chess.Board, vector_size: int = 64) -> np.ndarray:
    """Encode a small fixed-size vector of scalar features.

    Returns ``(vector_size,)`` float32 array; unused tail is zero-padded.
    Caller controls ``vector_size`` to match the network's expected input.
    """
    if vector_size < 8:
        raise ValueError("vector_size must be at least 8 to fit core features")
    v = np.zeros(vector_size, dtype=np.float32)
    v[0] = 1.0 if board.turn == chess.WHITE else -1.0
    v[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    v[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    v[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    v[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    v[5] = min(board.halfmove_clock / 100.0, 1.0)
    v[6] = min(board.fullmove_number / 100.0, 1.0)
    v[7] = 1.0 if board.is_repetition(2) else 0.0
    return v


__all__ = [
    "AUX_PLANES",
    "MAX_HISTORY_DEPTH",
    "MIN_HISTORY_DEPTH",
    "PIECE_TYPES",
    "PLANES_PER_TIMESTEP",
    "channels_for_history",
    "encode_board",
    "encode_vector",
]
