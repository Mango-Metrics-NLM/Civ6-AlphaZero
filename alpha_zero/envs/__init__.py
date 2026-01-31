"""Environment implementations for AlphaZero training.

This module provides various game environments for training AlphaZero agents:
- MicroCivEnv: Simplified 4X game for Phase 1 training
- CivRealmWrapper: Integration with CivRealm (Freeciv-based RL environment)

CivRealm components are lazily imported to avoid import errors when the CivRealm
server is not running.
"""

from alpha_zero.envs.microciv import MicroCivEnv


def __getattr__(name: str):
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
    # MicroCiv
    "MicroCivEnv",
    # CivRealm (lazy loaded)
    "CivRealmConfig",
    "CivRealmEncoder",
    "CivRealmActionHandler",
    "CivRealmCurriculum",
    "CivRealmWrapper",
    "MiniGameTask",
    "create_civrealm_env",
    "CIVREALM_AVAILABLE",
]
