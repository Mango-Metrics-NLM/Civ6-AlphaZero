"""Centralized environment factory for the training pipeline.

Provides a single ``create_game_env`` function that replaces all hardcoded
``MicroCivEnv(...)`` calls across self-play, evaluation, arena, and benchmark
modules.  The ``gameplay_preset`` parameter selects which environment flavour
to create:

- ``"default"`` — plain ``MicroCivEnv`` (backward compatible).
- ``"enhanced"`` — ``EnhancedMicroCivEnv`` with all gameplay systems enabled
  via ``GameplayConfig.enhanced()``.

Usage::

    from alpha_zero.envs.env_factory import create_game_env

    env = create_game_env(board_size=8, max_turns=30)                 # default
    env = create_game_env(board_size=8, max_turns=30, gameplay_preset="enhanced")
"""

from __future__ import annotations

import logging

from alpha_zero.envs.microciv import MicroCivEnv

logger = logging.getLogger(__name__)

#: Registry of valid gameplay presets.  Kept as a module-level constant so
#: callers (tests, CLI help text) can discover valid values without importing
#: everything.
GAMEPLAY_PRESETS: tuple[str, ...] = ("default", "enhanced")


def create_game_env(
    board_size: int = 8,
    max_turns: int = 30,
    gameplay_preset: str = "default",
    seed: int | None = None,
    render_mode: str | None = None,
) -> MicroCivEnv:
    """Create a game environment based on the requested preset.

    Args:
        board_size: Size of the square game board.
        max_turns: Maximum turns before the game ends.
        gameplay_preset: One of :data:`GAMEPLAY_PRESETS`.
            ``"default"`` returns a plain :class:`MicroCivEnv`;
            ``"enhanced"`` returns an :class:`EnhancedMicroCivEnv` with
            all gameplay systems enabled.
        seed: Optional random seed for reproducibility.
        render_mode: Rendering mode (``'ansi'`` or ``None``).

    Returns:
        A game environment instance (always a ``MicroCivEnv`` or subclass).

    Raises:
        ValueError: If *gameplay_preset* is not a recognised value.
    """
    if gameplay_preset not in GAMEPLAY_PRESETS:
        raise ValueError(
            f"Unknown gameplay_preset {gameplay_preset!r}. Valid presets: {GAMEPLAY_PRESETS}"
        )

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
    default_env = MicroCivEnv(
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
    return default_env
