"""Stockfish UCI opponent adapter.

Wraps ``python-chess``'s :class:`chess.engine.SimpleEngine` and exposes:

- ``set_skill(level)`` — sets the UCI ``Skill Level`` parameter (0..20).
- ``set_elo(elo)`` — enables ``UCI_LimitStrength`` and sets ``UCI_Elo``.
- ``configure_limit(...)`` — depth / movetime / nodes search limits.
- Context-manager semantics with subprocess cleanup via ``atexit`` and
  ``close()``.

Resilience:

- If the engine crashes or returns an unexpected error, we log once and
  fall back to a uniform-random legal move so self-play does not abort.
- No configuration values are hardcoded — every knob lives on
  :class:`StockfishOpponentConfig` and has an env override via
  ``CIV6AZ_CHESS_STOCKFISH_*`` (read by the orchestrator config layer).

The Stockfish binary path is required but never embedded in code — pass
it via :class:`StockfishOpponentConfig` (which the config layer fills
from ``CIV6AZ_CHESS_STOCKFISH_PATH`` / ``STOCKFISH_PATH``).
"""

from __future__ import annotations

import atexit
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

import chess
import chess.engine
import numpy as np

from alpha_zero.envs.chess_actions import index_to_move, move_to_index

logger = logging.getLogger(__name__)

#: Per the Stockfish UCI spec, valid Skill Level range.
STOCKFISH_SKILL_MIN: int = 0
STOCKFISH_SKILL_MAX: int = 20
#: Stockfish's UCI_Elo bounds (approximate). UCI_LimitStrength must be true.
STOCKFISH_ELO_MIN: int = 1320
STOCKFISH_ELO_MAX: int = 3190


@dataclass
class StockfishOpponentConfig:
    """Configuration for :class:`StockfishOpponent`.

    All fields have safe defaults so the dataclass can be instantiated from
    YAML or env overrides without surprises.
    """

    #: Path to the Stockfish binary. None → read from ``STOCKFISH_PATH`` env at use time.
    path: str | None = None
    #: Stockfish Skill Level (0..20). Used when ``use_elo`` is False.
    skill: int = 5
    #: Approximate target ELO. Used when ``use_elo`` is True.
    elo: int = 1500
    #: If True, configure via UCI_Elo + UCI_LimitStrength; otherwise via Skill Level.
    use_elo: bool = False
    #: Search depth limit (plies). None disables depth limiting.
    depth_limit: int | None = 10
    #: Search time limit in seconds. None disables time limiting.
    time_limit_s: float | None = None
    #: Search node limit. None disables node limiting.
    node_limit: int | None = None
    #: Number of threads Stockfish may use.
    threads: int = 1
    #: Hash table size in MB.
    hash_mb: int = 16
    #: Extra UCI options merged on top of the above (advanced users).
    extra_uci_options: dict[str, Any] = field(default_factory=dict)
    #: Identifier shown in tournament tables / logs.
    name: str = "stockfish"

    def __post_init__(self) -> None:
        if not STOCKFISH_SKILL_MIN <= self.skill <= STOCKFISH_SKILL_MAX:
            raise ValueError(
                f"skill {self.skill} out of range [{STOCKFISH_SKILL_MIN}, {STOCKFISH_SKILL_MAX}]"
            )
        if not STOCKFISH_ELO_MIN <= self.elo <= STOCKFISH_ELO_MAX:
            raise ValueError(
                f"elo {self.elo} out of range [{STOCKFISH_ELO_MIN}, {STOCKFISH_ELO_MAX}]"
            )

    def resolved_path(self) -> str:
        path = self.path or os.getenv("STOCKFISH_PATH")
        if not path:
            raise ValueError(
                "Stockfish binary path not configured. Set StockfishOpponentConfig.path "
                "or the STOCKFISH_PATH environment variable."
            )
        return path


class StockfishOpponent:
    """Picks moves by querying Stockfish via UCI.

    Designed for safe use inside a long-running training loop:

    - Subprocess is registered with :func:`atexit.register` and the context
      manager protocol so it cannot leak.
    - On any ``chess.engine.EngineError`` the opponent falls back to a random
      legal move and logs once per session (to avoid log spam).
    """

    def __init__(
        self,
        config: StockfishOpponentConfig | None = None,
        *,
        engine: chess.engine.SimpleEngine | None = None,
    ) -> None:
        self.config = config or StockfishOpponentConfig()
        self.name = self.config.name
        self._engine: chess.engine.SimpleEngine | None = engine
        self._closed: bool = False
        self._fallback_warned: bool = False
        if engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.config.resolved_path())
        self._apply_uci_options()
        atexit.register(self.close)

    # -------------------------------------------------------------- configuration

    def _apply_uci_options(self) -> None:
        assert self._engine is not None
        opts: dict[str, Any] = {
            "Threads": self.config.threads,
            "Hash": self.config.hash_mb,
        }
        if self.config.use_elo:
            opts["UCI_LimitStrength"] = True
            opts["UCI_Elo"] = self.config.elo
        else:
            opts["Skill Level"] = self.config.skill
        opts.update(self.config.extra_uci_options)
        try:
            self._engine.configure(opts)
        except chess.engine.EngineError as exc:  # pragma: no cover - engine-specific
            logger.warning("Stockfish configure() failed: %s", exc)

    def set_skill(self, level: int) -> None:
        """Re-configure to skill-level mode at the given level (0..20)."""
        if not STOCKFISH_SKILL_MIN <= level <= STOCKFISH_SKILL_MAX:
            raise ValueError(
                f"level {level} out of range [{STOCKFISH_SKILL_MIN}, {STOCKFISH_SKILL_MAX}]"
            )
        self.config.skill = level
        self.config.use_elo = False
        self._apply_uci_options()

    def set_elo(self, elo: int) -> None:
        """Re-configure to ELO-limit mode at the given target ELO."""
        if not STOCKFISH_ELO_MIN <= elo <= STOCKFISH_ELO_MAX:
            raise ValueError(f"elo {elo} out of range [{STOCKFISH_ELO_MIN}, {STOCKFISH_ELO_MAX}]")
        self.config.elo = elo
        self.config.use_elo = True
        self._apply_uci_options()

    def configure_limit(
        self,
        *,
        depth: int | None = None,
        time_s: float | None = None,
        nodes: int | None = None,
    ) -> None:
        """Override search limits in place. ``None`` clears that limit."""
        self.config.depth_limit = depth
        self.config.time_limit_s = time_s
        self.config.node_limit = nodes

    def _build_limit(self) -> chess.engine.Limit:
        return chess.engine.Limit(
            depth=self.config.depth_limit,
            time=self.config.time_limit_s,
            nodes=self.config.node_limit,
        )

    # -------------------------------------------------------------- Opponent API

    def get_action(self, env: Any, action_mask: np.ndarray) -> int:
        """Ask Stockfish for a move on ``env.board`` and encode it.

        If anything goes wrong (engine crash, malformed response, encoding
        failure) we fall back to a random legal move and log once.
        """
        board: chess.Board = env.board
        if self._engine is None or self._closed:
            return self._fallback_action(board, action_mask, reason="engine unavailable")
        try:
            result = self._engine.play(board, self._build_limit())
        except chess.engine.EngineError as exc:
            return self._fallback_action(board, action_mask, reason=f"EngineError: {exc}")
        if result.move is None:
            return self._fallback_action(board, action_mask, reason="engine returned no move")
        try:
            action = move_to_index(result.move)
        except ValueError as exc:
            return self._fallback_action(board, action_mask, reason=f"unencodable move: {exc}")
        if not (0 <= action < action_mask.shape[0]) or action_mask[action] == 0:
            return self._fallback_action(
                board,
                action_mask,
                reason=f"engine returned masked action {action} ({result.move.uci()})",
            )
        return action

    def reset(self) -> None:
        """Per-episode reset. UCI has ``ucinewgame``; harmless if it fails."""
        if self._engine is None or self._closed:
            return
        try:
            # SimpleEngine doesn't expose ucinewgame directly; configure() with
            # an empty dict is a no-op handshake that flushes state.
            self._engine.configure({})
        except chess.engine.EngineError as exc:  # pragma: no cover
            logger.debug("Stockfish reset() configure() raised: %s", exc)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._engine is not None:
            try:
                self._engine.quit()
            except (chess.engine.EngineError, OSError) as exc:  # pragma: no cover
                logger.debug("Stockfish quit() raised: %s", exc)
            finally:
                self._engine = None

    def __enter__(self) -> StockfishOpponent:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -------------------------------------------------------------- internals

    def _fallback_action(
        self, board: chess.Board, action_mask: np.ndarray, reason: str
    ) -> int:
        if not self._fallback_warned:
            logger.warning(
                "Stockfish unavailable, falling back to random move: %s",
                reason,
            )
            self._fallback_warned = True
        # Build a list of (move, action_index) for moves that are both
        # legal on the board AND permitted by action_mask. Honouring the
        # mask is essential: callers (e.g. self-play wrappers) may mask
        # out moves that are board-legal for game-rule reasons, and a
        # fallback that ignores the mask would let those slip through.
        candidates: list[tuple[chess.Move, int]] = []
        for move in board.legal_moves:
            try:
                idx = move_to_index(move)
            except ValueError:  # pragma: no cover - all legal moves encode
                continue
            if 0 <= idx < action_mask.shape[0] and action_mask[idx] > 0:
                candidates.append((move, idx))
        if candidates:
            _, idx = random.choice(candidates)  # noqa: S311 - not cryptographic
            return idx
        # No mask-respecting legal move exists. Last-resort: any encodable
        # legal move. If even that's empty we cannot proceed.
        for move in board.legal_moves:
            try:
                return move_to_index(move)
            except ValueError:  # pragma: no cover
                continue
        raise RuntimeError("No legal moves available for fallback")


# Keep ``index_to_move`` import alive for downstream debugging / extensions.
_ = index_to_move
