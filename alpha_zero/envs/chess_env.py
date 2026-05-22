"""Gymnasium-compatible chess environment for AlphaZero/MCTS training.

This env implements the project's ``MCTSEnvironment`` protocol (declared in
``alpha_zero.core.civ6_mcts``): it exposes ``reset()``, ``step()``,
``clone()``, ``get_observation()``, ``get_info()``, ``get_valid_actions()``,
plus ``action_space`` and ``current_player`` attributes.

Observation dict shape mirrors ``MicroCivEnv`` so downstream code does not
need to special-case chess:

- ``spatial``: ``float32`` array of shape ``(C, 8, 8)`` where
  ``C = channels_for_history(history_depth)``.
- ``vector``: ``float32`` array of shape ``(vector_size,)``.
- ``action_mask``: ``float32`` array of shape ``(ACTION_SPACE_SIZE,)`` with
  1.0 marking legal moves.

Reward convention (sparse, returned at terminal states only):

- ``step()`` returns the reward **from the perspective of the player who
  just moved**. In standard chess, the side that delivers checkmate is the
  side that just moved, so terminal-checkmate is always ``+1`` for the
  mover.
- Draws (stalemate, insufficient material, threefold repetition, fifty-move
  rule, ``max_plies`` cap) return ``0``.

``info["winner"]`` is set to the winning player's index (``0 = white``,
``1 = black``) on checkmate, and ``None`` for draws. This matches the
convention consumed by ``Civ6MCTS`` at the win-check path in
``civ6_mcts.py``.
"""

from __future__ import annotations

from typing import Any, ClassVar, SupportsFloat

import chess
import numpy as np
from gymnasium import spaces

from alpha_zero.envs.chess_actions import (
    ACTION_SPACE_SIZE,
    BOARD_SIZE,
    index_to_move,
    legal_action_mask,
    move_to_index,
)
from alpha_zero.envs.chess_encoder import (
    channels_for_history,
    encode_board,
    encode_vector,
)

WHITE_PLAYER_INDEX: int = 0
BLACK_PLAYER_INDEX: int = 1


def _player_index(turn: bool) -> int:
    return WHITE_PLAYER_INDEX if turn == chess.WHITE else BLACK_PLAYER_INDEX


class ChessEnv:
    """Chess environment compatible with the project's MCTS stack."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["ansi"]}

    action_space: spaces.Discrete
    current_player: int

    def __init__(
        self,
        history_depth: int = 1,
        vector_size: int = 64,
        max_plies: int = 512,
        starting_fen: str | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the chess environment.

        Args:
            history_depth: Number of past plies to encode (1..8).
                T=1 → ~21 channels; T=8 → 119 channels (AlphaZero-full).
            vector_size: Size of the scalar feature vector (padded with zeros).
            max_plies: Truncation cap on game length to prevent unbounded
                self-play games. Default 512 plies (256 moves).
            starting_fen: Optional FEN to start from (default: standard).
            render_mode: One of ``metadata["render_modes"]`` or ``None``.
            seed: Reserved for API symmetry; chess play is deterministic
                given moves, so the seed is recorded but otherwise unused.
        """
        if history_depth < 1:
            raise ValueError("history_depth must be >= 1")
        if max_plies < 1:
            raise ValueError("max_plies must be >= 1")
        # Mirror the lower bound enforced inside encode_vector(); validating
        # here means an invalid vector_size fails at construction time with
        # a clear error instead of much later inside reset()/get_observation().
        if vector_size < 8:
            raise ValueError(f"vector_size must be >= 8 to fit core features (got {vector_size})")
        self.history_depth = history_depth
        self.vector_size = vector_size
        self.max_plies = max_plies
        self.starting_fen = starting_fen
        self.render_mode = render_mode
        self._seed = seed

        self._board = chess.Board(starting_fen) if starting_fen is not None else chess.Board()
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        spatial_channels = channels_for_history(history_depth)
        self.observation_space = spaces.Dict(
            {
                "spatial": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(spatial_channels, BOARD_SIZE, BOARD_SIZE),
                    dtype=np.float32,
                ),
                "vector": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(vector_size,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(ACTION_SPACE_SIZE,),
                    dtype=np.float32,
                ),
            }
        )
        self.current_player = _player_index(self._board.turn)
        self._winner: int | None = None
        self._terminated: bool = False
        self._truncated: bool = False

    # ------------------------------------------------------------------ core API

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset to the starting position (or ``starting_fen`` if set)."""
        del options
        if seed is not None:
            self._seed = seed
        self._board = (
            chess.Board(self.starting_fen) if self.starting_fen is not None else chess.Board()
        )
        self.current_player = _player_index(self._board.turn)
        self._winner = None
        self._terminated = False
        self._truncated = False
        return self.get_observation(), self.get_info()

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply ``action`` (encoded index) and advance the game one ply."""
        if self._terminated or self._truncated:
            raise RuntimeError(
                "step() called on a terminated/truncated episode; call reset() first"
            )

        move = self._resolve_action(action)
        mover_player = self.current_player
        self._board.push(move)

        # Update terminal flags, including the claimable draws called out in
        # the module docstring (threefold repetition, fifty-move rule). With
        # ``claim_draw=False`` the env would silently fall through to the
        # ``max_plies`` truncation branch for those positions instead of
        # producing a proper draw outcome with ``winner=None``.
        outcome = self._board.outcome(claim_draw=True)
        reward: float = 0.0
        if outcome is not None:
            self._terminated = True
            if outcome.winner is None:
                self._winner = None
            else:
                self._winner = _player_index(outcome.winner)
                # Reward is from the perspective of the player who just moved.
                reward = 1.0 if self._winner == mover_player else -1.0
        elif self._board.ply() >= self.max_plies:
            self._truncated = True
            self._winner = None

        self.current_player = _player_index(self._board.turn)
        return (
            self.get_observation(),
            reward,
            self._terminated,
            self._truncated,
            self.get_info(),
        )

    def _resolve_action(self, action: int) -> chess.Move:
        if not 0 <= action < ACTION_SPACE_SIZE:
            raise ValueError(f"Action {action} out of range [0, {ACTION_SPACE_SIZE})")
        move = index_to_move(action, self._board)
        if move not in self._board.legal_moves:
            raise ValueError(f"Illegal action {action} ({move.uci()}) on board {self._board.fen()}")
        return move

    # ------------------------------------------------------------------ accessors

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "spatial": encode_board(self._board, history_depth=self.history_depth),
            "vector": encode_vector(self._board, vector_size=self.vector_size),
            "action_mask": legal_action_mask(self._board),
        }

    def get_info(self) -> dict[str, Any]:
        return {
            "winner": self._winner,
            "current_player": self.current_player,
            "ply": self._board.ply(),
            "fullmove_number": self._board.fullmove_number,
            "halfmove_clock": self._board.halfmove_clock,
            "is_check": self._board.is_check(),
            "fen": self._board.fen(),
            "terminated": self._terminated,
            "truncated": self._truncated,
        }

    def get_valid_actions(self) -> np.ndarray:
        return legal_action_mask(self._board)

    def clone(self) -> ChessEnv:
        """Return a deep-state copy suitable for MCTS simulation."""
        cloned = ChessEnv(
            history_depth=self.history_depth,
            vector_size=self.vector_size,
            max_plies=self.max_plies,
            starting_fen=self.starting_fen,
            render_mode=self.render_mode,
            seed=self._seed,
        )
        cloned._board = self._board.copy(stack=True)
        cloned.current_player = self.current_player
        cloned._winner = self._winner
        cloned._terminated = self._terminated
        cloned._truncated = self._truncated
        return cloned

    @property
    def board(self) -> chess.Board:
        """Read-only access to the underlying ``chess.Board`` for advisors."""
        return self._board

    # ------------------------------------------------------------------ rendering

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return str(self._board)
        return None

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None

    # ------------------------------------------------------------------ utility

    def encode_move(self, move: chess.Move) -> int:
        """Public helper for advisors mapping UCI moves to action indices."""
        return move_to_index(move)


__all__ = [
    "BLACK_PLAYER_INDEX",
    "WHITE_PLAYER_INDEX",
    "ChessEnv",
]
