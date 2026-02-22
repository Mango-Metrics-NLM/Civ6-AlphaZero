"""Environment implementations for AlphaZero training.

This module provides various game environments for training AlphaZero agents:
- MicroCivEnv: Simplified 4X game for Phase 1 training
- CivRealmWrapper: Integration with CivRealm (Freeciv-based RL environment)

CivRealm components are lazily imported to avoid import errors when the CivRealm
server is not running.
"""

from typing import Any

from alpha_zero.envs.env_factory import GAMEPLAY_PRESETS, create_game_env
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.envs.microciv_enhanced import EnhancedMicroCivEnv


def __getattr__(name: str) -> Any:
    """Lazy import CivRealm components to avoid import errors."""
    civrealm_exports = {
        "CivRealmConfig",
        "CivRealmEncoder",
        "CivRealmActionHandler",
        "CivRealmCurriculum",
        "CivRealmWrapper",
        "MiniGameTask",
        "create_civrealm_env",
        "CIVREALM_AVAILABLE",
    }

    if name in civrealm_exports:
        from alpha_zero.envs import civrealm_wrapper

        return getattr(civrealm_wrapper, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CIVREALM_AVAILABLE",
    # Environment factory
    "GAMEPLAY_PRESETS",
    "CivRealmActionHandler",
    # CivRealm (lazy loaded)
    "CivRealmConfig",
    "CivRealmCurriculum",
    "CivRealmEncoder",
    "CivRealmWrapper",
    # MicroCiv
    "EnhancedMicroCivEnv",
    "MicroCivEnv",
    "MiniGameTask",
    "create_civrealm_env",
    "create_game_env",
]
