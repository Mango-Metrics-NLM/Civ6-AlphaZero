"""Centralized environment factory for the training pipeline.

Provides a single ``create_game_env`` function that replaces all hardcoded
``MicroCivEnv(...)`` calls across self-play, evaluation, arena, and benchmark
modules.  The ``gameplay_preset`` parameter selects which environment flavour
to create:

- ``"default"`` — plain ``MicroCivEnv`` (backward compatible).
- ``"enhanced"`` — ``EnhancedMicroCivEnv`` with all gameplay systems enabled
  via ``GameplayConfig.enhanced()``.
- ``"chess"`` — ``ChessEnv`` (full chess vs. external opponents / self-play).
  Returns an object implementing the ``MCTSEnvironment`` protocol but **not**
  a ``MicroCivEnv`` instance. The return annotation is intentionally widened
  to ``Any`` to keep type-checkers honest at chess call sites while preserving
  MicroCiv semantics for the default/enhanced presets.

Usage::

    from alpha_zero.envs.env_factory import create_game_env

    env = create_game_env(board_size=8, max_turns=30)                 # default
    env = create_game_env(board_size=8, max_turns=30, gameplay_preset="enhanced")
    env = create_game_env(gameplay_preset="chess", chess_history_depth=1)
"""

from __future__ import annotations

import logging
from typing import Any

from alpha_zero.envs.microciv import MicroCivEnv

logger = logging.getLogger(__name__)

#: Registry of valid gameplay presets.  Kept as a module-level constant so
#: callers (tests, CLI help text) can discover valid values without importing
#: everything.
GAMEPLAY_PRESETS: tuple[str, ...] = ("default", "enhanced", "chess")

#: Subset of presets that return a ``MicroCivEnv`` (or subclass) with the
#: MicroCiv-specific obs/action shapes and ``board_size``/``max_turns``
#: attributes.  Useful for parametrising MicroCiv-shape-sensitive tests.
MICROCIV_PRESETS: tuple[str, ...] = ("default", "enhanced")


def create_game_env(
    board_size: int = 8,
    max_turns: int = 30,
    gameplay_preset: str = "default",
    seed: int | None = None,
    render_mode: str | None = None,
    *,
    chess_history_depth: int = 1,
    chess_max_plies: int = 512,
    chess_starting_fen: str | None = None,
) -> Any:
    """Create a game environment based on the requested preset.

    Args:
        board_size: Size of the square game board.
        max_turns: Maximum turns before the game ends.
        gameplay_preset: One of :data:`GAMEPLAY_PRESETS`.
            ``"default"`` returns a plain :class:`MicroCivEnv`;
            ``"enhanced"`` returns an :class:`EnhancedMicroCivEnv` with
            all gameplay systems enabled;
            ``"chess"`` returns a :class:`ChessEnv` (not a MicroCiv subclass).
        seed: Optional random seed for reproducibility.
        render_mode: Rendering mode (``'ansi'`` or ``None``).

    Returns:
        A game environment instance implementing the ``MCTSEnvironment``
        protocol. Concrete type depends on ``gameplay_preset``:

        - ``"default"`` → :class:`MicroCivEnv`
        - ``"enhanced"`` → :class:`EnhancedMicroCivEnv` (a ``MicroCivEnv``
          subclass)
        - ``"chess"`` → :class:`ChessEnv` (**not** a ``MicroCivEnv`` subclass;
          presets in :data:`MICROCIV_PRESETS` filter this out for MicroCiv-
          shape-sensitive call sites)

    Raises:
        ValueError: If *gameplay_preset* is not a recognised value.
    """
    if gameplay_preset not in GAMEPLAY_PRESETS:
        raise ValueError(
            f"Unknown gameplay_preset {gameplay_preset!r}. Valid presets: {GAMEPLAY_PRESETS}"
        )

    if gameplay_preset == "chess":
        # Lazy import to keep python-chess optional for non-chess users.
        from alpha_zero.envs.chess_env import ChessEnv

        chess_env = ChessEnv(
            history_depth=chess_history_depth,
            max_plies=chess_max_plies,
            starting_fen=chess_starting_fen,
            render_mode=render_mode,
            seed=seed,
        )
        logger.info(
            "Created ChessEnv (preset=%s, history_depth=%d, max_plies=%d)",
            gameplay_preset,
            chess_history_depth,
            chess_max_plies,
        )
        return chess_env

    if gameplay_preset == "enhanced":
        # Lazy import to avoid pulling in gameplay systems when not needed
        from alpha_zero.envs.microciv_enhanced import EnhancedMicroCivEnv
        from alpha_zero.gameplay.config import GameplayConfig

        env = EnhancedMicroCivEnv(
            board_size=board_size,
            max_turns=max_turns,
            seed=seed,
            render_mode=render_mode,
            gameplay_config=GameplayConfig.enhanced(),
        )
        logger.info(
            "Created EnhancedMicroCivEnv (preset=%s, board=%d, turns=%d)",
            gameplay_preset,
            board_size,
            max_turns,
        )
        return env

    # Default preset — plain MicroCivEnv
    env = MicroCivEnv(
        board_size=board_size,
        max_turns=max_turns,
        seed=seed,
        render_mode=render_mode,
    )
    logger.info(
        "Created MicroCivEnv (preset=%s, board=%d, turns=%d)",
        gameplay_preset,
        board_size,
        max_turns,
    )
    return env
