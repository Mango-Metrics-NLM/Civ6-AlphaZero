"""AlphaZero-style 8x8x73 action encoding for chess.

The AlphaZero chess action space is a stack of 73 planes over an 8x8 board,
yielding 4672 discrete actions. The plane at position (rank, file, plane_idx)
describes a move *originating* from square (rank, file).

Plane layout (per AlphaZero, Silver et al. 2017):

- Planes  0..55  — "queen-like" moves: 7 distances x 8 compass directions.
  The 8 directions, in fixed order, are: N, NE, E, SE, S, SW, W, NW.
  Plane index = direction_idx * 7 + (distance - 1).

- Planes 56..63  — Knight moves: 8 fixed offsets in (drank, dfile):
  (+2,+1), (+1,+2), (-1,+2), (-2,+1), (-2,-1), (-1,-2), (+1,-2), (+2,-1).

- Planes 64..72  — Underpromotions: 3 promotion pieces (knight, bishop, rook)
  x 3 pawn-move directions (forward, capture-left, capture-right).
  Plane index = 64 + piece_idx * 3 + direction_idx.

Queen-like promotions (i.e., promotion to queen) are encoded as the underlying
queen-direction move with no separate plane — python-chess will treat a queen
promotion as the default when the move ends on the last rank, so we explicitly
set ``promotion = chess.QUEEN`` when reconstructing such moves.

The board is indexed with rank 0 = White's back rank (chess.square(file, rank))
so action indices are consistent regardless of side to move. Coordinate flips
for the side-to-move-relative view live in the encoder, not here.

No values are hardcoded into callers: ``ACTION_SPACE_SIZE``, ``NUM_PLANES``,
``BOARD_SIZE`` are exposed as module constants.
"""

from __future__ import annotations

from typing import Final

import chess
import numpy as np

BOARD_SIZE: Final[int] = 8
NUM_PLANES: Final[int] = 73
ACTION_SPACE_SIZE: Final[int] = BOARD_SIZE * BOARD_SIZE * NUM_PLANES  # 4672

# Queen-like directions: (drank, dfile). Order is fixed by AlphaZero spec.
_QUEEN_DIRECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (1, 0),  # N
    (1, 1),  # NE
    (0, 1),  # E
    (-1, 1),  # SE
    (-1, 0),  # S
    (-1, -1),  # SW
    (0, -1),  # W
    (1, -1),  # NW
)
_QUEEN_MAX_DISTANCE: Final[int] = 7
_QUEEN_PLANES: Final[int] = len(_QUEEN_DIRECTIONS) * _QUEEN_MAX_DISTANCE  # 56

# Knight offsets in (drank, dfile).
_KNIGHT_OFFSETS: Final[tuple[tuple[int, int], ...]] = (
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
)
_KNIGHT_PLANE_OFFSET: Final[int] = _QUEEN_PLANES  # 56
_KNIGHT_PLANES: Final[int] = len(_KNIGHT_OFFSETS)  # 8

# Underpromotion: 3 pieces x 3 directions = 9 planes.
# Pieces in fixed order: knight, bishop, rook. Queen handled as queen-direction.
_UNDERPROMO_PIECES: Final[tuple[int, ...]] = (chess.KNIGHT, chess.BISHOP, chess.ROOK)
# Directions are file deltas relative to the moving pawn: forward(0), capture-left(-1), capture-right(+1).
_UNDERPROMO_FILE_DELTAS: Final[tuple[int, ...]] = (0, -1, 1)
_UNDERPROMO_PLANE_OFFSET: Final[int] = _KNIGHT_PLANE_OFFSET + _KNIGHT_PLANES  # 64
_UNDERPROMO_PLANES: Final[int] = len(_UNDERPROMO_PIECES) * len(_UNDERPROMO_FILE_DELTAS)  # 9

assert _UNDERPROMO_PLANE_OFFSET + _UNDERPROMO_PLANES == NUM_PLANES


def _square_to_rf(square: int) -> tuple[int, int]:
    return chess.square_rank(square), chess.square_file(square)


def _rf_to_index(rank: int, file: int, plane: int) -> int:
    return (rank * BOARD_SIZE + file) * NUM_PLANES + plane


def _index_to_rfp(index: int) -> tuple[int, int, int]:
    plane = index % NUM_PLANES
    rest = index // NUM_PLANES
    file = rest % BOARD_SIZE
    rank = rest // BOARD_SIZE
    return rank, file, plane


def _queen_plane(drank: int, dfile: int) -> int | None:
    """Return queen-plane index for a sliding move, or None if not queen-like."""
    if drank == 0 and dfile == 0:
        return None
    distance = max(abs(drank), abs(dfile))
    if distance > _QUEEN_MAX_DISTANCE:
        return None
    # Direction must be one of the 8 compass directions: components must be in {-1, 0, +1} after normalization.
    step_r = 0 if drank == 0 else drank // abs(drank)
    step_f = 0 if dfile == 0 else dfile // abs(dfile)
    if step_r * distance != drank or step_f * distance != dfile:
        return None
    try:
        direction_idx = _QUEEN_DIRECTIONS.index((step_r, step_f))
    except ValueError:
        return None
    return direction_idx * _QUEEN_MAX_DISTANCE + (distance - 1)


def _knight_plane(drank: int, dfile: int) -> int | None:
    """Return knight-plane index for a knight move, or None if not a knight offset."""
    try:
        offset_idx = _KNIGHT_OFFSETS.index((drank, dfile))
    except ValueError:
        return None
    return _KNIGHT_PLANE_OFFSET + offset_idx


def _underpromo_plane(promotion: int, dfile: int) -> int | None:
    """Return underpromotion-plane index, or None for queen-promotion."""
    if promotion == chess.QUEEN:
        return None
    if promotion not in _UNDERPROMO_PIECES:
        return None
    if dfile not in _UNDERPROMO_FILE_DELTAS:
        return None
    piece_idx = _UNDERPROMO_PIECES.index(promotion)
    dir_idx = _UNDERPROMO_FILE_DELTAS.index(dfile)
    return _UNDERPROMO_PLANE_OFFSET + piece_idx * len(_UNDERPROMO_FILE_DELTAS) + dir_idx


def move_to_index(move: chess.Move) -> int:
    """Encode a ``chess.Move`` into the AlphaZero 4672-action index.

    Raises:
        ValueError: If the move cannot be represented in the encoding.
    """
    from_rank, from_file = _square_to_rf(move.from_square)
    to_rank, to_file = _square_to_rf(move.to_square)
    drank = to_rank - from_rank
    dfile = to_file - from_file

    # Underpromotion (knight/bishop/rook). Queen promotion → queen-plane (no special plane).
    if move.promotion and move.promotion != chess.QUEEN:
        plane = _underpromo_plane(move.promotion, dfile)
        if plane is None:
            raise ValueError(f"Cannot encode promotion={move.promotion} with dfile={dfile}")
        return _rf_to_index(from_rank, from_file, plane)

    # Knight move.
    plane = _knight_plane(drank, dfile)
    if plane is not None:
        return _rf_to_index(from_rank, from_file, plane)

    # Queen-like (including queen promotion, which is the default at the last rank).
    plane = _queen_plane(drank, dfile)
    if plane is not None:
        return _rf_to_index(from_rank, from_file, plane)

    raise ValueError(f"Cannot encode move {move.uci()}: not a recognised geometry")


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Decode an action index into a ``chess.Move`` legal on ``board``.

    The board context is required because (a) underpromotion planes don't
    distinguish piece colour direction (white moves +1 rank, black -1), and
    (b) a queen-direction move that lands on the last rank is auto-promoted
    to a queen for the side to move.

    Raises:
        ValueError: If the index is out of range or yields no legal move.
    """
    if not 0 <= index < ACTION_SPACE_SIZE:
        raise ValueError(f"Index {index} out of range [0, {ACTION_SPACE_SIZE})")

    rank, file, plane = _index_to_rfp(index)
    from_square = chess.square(file, rank)

    # Underpromotion plane.
    if plane >= _UNDERPROMO_PLANE_OFFSET:
        rel = plane - _UNDERPROMO_PLANE_OFFSET
        piece_idx, dir_idx = divmod(rel, len(_UNDERPROMO_FILE_DELTAS))
        promotion_piece = _UNDERPROMO_PIECES[piece_idx]
        dfile = _UNDERPROMO_FILE_DELTAS[dir_idx]
        # Pawn direction: white +1 rank, black -1 rank.
        drank = 1 if board.turn == chess.WHITE else -1
        to_rank = rank + drank
        to_file = file + dfile
        if not (0 <= to_rank < BOARD_SIZE and 0 <= to_file < BOARD_SIZE):
            raise ValueError(f"Underpromotion target off-board for index {index}")
        to_square = chess.square(to_file, to_rank)
        return chess.Move(from_square, to_square, promotion=promotion_piece)

    # Knight plane.
    if plane >= _KNIGHT_PLANE_OFFSET:
        drank, dfile = _KNIGHT_OFFSETS[plane - _KNIGHT_PLANE_OFFSET]
        to_rank = rank + drank
        to_file = file + dfile
        if not (0 <= to_rank < BOARD_SIZE and 0 <= to_file < BOARD_SIZE):
            raise ValueError(f"Knight target off-board for index {index}")
        return chess.Move(from_square, chess.square(to_file, to_rank))

    # Queen-like plane.
    direction_idx, dist_minus_one = divmod(plane, _QUEEN_MAX_DISTANCE)
    distance = dist_minus_one + 1
    step_r, step_f = _QUEEN_DIRECTIONS[direction_idx]
    to_rank = rank + step_r * distance
    to_file = file + step_f * distance
    if not (0 <= to_rank < BOARD_SIZE and 0 <= to_file < BOARD_SIZE):
        raise ValueError(f"Queen-like target off-board for index {index}")
    to_square = chess.square(to_file, to_rank)

    # If a pawn move lands on the last rank, this is implicitly a queen promotion.
    promotion: int | None = None
    piece = board.piece_at(from_square)
    if piece is not None and piece.piece_type == chess.PAWN:
        if (board.turn == chess.WHITE and to_rank == BOARD_SIZE - 1) or (
            board.turn == chess.BLACK and to_rank == 0
        ):
            promotion = chess.QUEEN

    return chess.Move(from_square, to_square, promotion=promotion)


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Return a float32 mask of shape ``(ACTION_SPACE_SIZE,)`` with 1.0 at legal actions."""
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    for move in board.legal_moves:
        try:
            mask[move_to_index(move)] = 1.0
        except ValueError:  # pragma: no cover - all legal moves are encodable
            continue
    return mask


__all__ = [
    "ACTION_SPACE_SIZE",
    "BOARD_SIZE",
    "NUM_PLANES",
    "index_to_move",
    "legal_action_mask",
    "move_to_index",
]
