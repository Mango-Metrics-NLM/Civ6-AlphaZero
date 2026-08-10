"""AlphaZero implementation for Civilization VI.

This package provides the core AlphaZero components for training an AI
agent to play Civilization VI.

Public API:
    Core:
        Civ6MCTS - Monte Carlo Tree Search implementation
        ActionSpace - Action space handling for the game
        HierarchicalAction - Hierarchical action representation
        Civ6Network, MicroCivNetwork - Neural network architectures
        LossConfig, CategoricalSupportConfig - Loss function configuration

    Environments:
        MicroCivEnv - Simplified Civ6 training environment
        RealCiv6Env - Real Civ6 game integration (optional dependencies)

    Configuration:
        TimeoutConfig - Centralized timeout configuration
        RewardConfig - Reward shaping configuration (optional dependencies)

Usage:
    from alpha_zero import Civ6MCTS, ActionSpace, MicroCivEnv

Lazy imports:
    Attributes are resolved on first access (PEP 562) rather than at package
    import time. This keeps ``import alpha_zero`` cheap and, more importantly,
    stops it from dragging numpy and gymnasium into processes that only need a
    leaf subpackage — ``alpha_zero.hexcrystal`` in particular is torch-only and
    must stay importable without them.

    ``RealCiv6Env`` and ``RewardConfig`` are resolvable attributes but are
    deliberately absent from ``__all__``: deciding whether their optional
    dependencies are installed would require performing the very import being
    deferred. ``from alpha_zero import RealCiv6Env`` works and raises a normal
    ``ImportError`` when the dependencies are missing; only ``import *`` no
    longer pulls them in.
"""

from __future__ import annotations

# Keep this a literal assignment: scripts/check_doc_sync.py parses it by regex
# to enforce the version triple (D1).
__version__ = "1.8.0"

# Python <3.11 compatibility: patch datetime.UTC before anything imports it.
# This must stay eager — the patch has to land before any dependent import.
import alpha_zero._compat  # noqa: F401  # isort: skip

from typing import TYPE_CHECKING, Any

# Public name -> module that defines it. Resolved on first attribute access.
_LAZY_EXPORTS: dict[str, str] = {
    # Core
    "ActionSpace": "alpha_zero.core",
    "CategoricalSupportConfig": "alpha_zero.core",
    "Civ6MCTS": "alpha_zero.core",
    "Civ6Network": "alpha_zero.core",
    "HierarchicalAction": "alpha_zero.core",
    "LossConfig": "alpha_zero.core",
    "MicroCivNetwork": "alpha_zero.core",
    # Environments
    "MicroCivEnv": "alpha_zero.envs.microciv",
    # Configuration
    "TimeoutConfig": "alpha_zero.utils.timeout_config",
    "get_timeout_config": "alpha_zero.utils.timeout_config",
    # Optional: require extra dependencies, resolvable but not in __all__
    "RealCiv6Env": "alpha_zero.envs.real_civ6_env",
    "RewardConfig": "alpha_zero.envs.real_civ6_env",
}

# Names that may legitimately fail to import when optional extras are absent.
_OPTIONAL_EXPORTS = frozenset({"RealCiv6Env", "RewardConfig"})

__all__ = [
    "ActionSpace",
    "CategoricalSupportConfig",
    # Core
    "Civ6MCTS",
    "Civ6Network",
    "HierarchicalAction",
    "LossConfig",
    # Environments
    "MicroCivEnv",
    "MicroCivNetwork",
    # Configuration
    "TimeoutConfig",
    # Version
    "__version__",
    "get_timeout_config",
]

if TYPE_CHECKING:
    # Give type checkers and IDEs the real symbols; at runtime these are
    # resolved lazily by __getattr__ below.
    from alpha_zero.core import (
        ActionSpace,
        CategoricalSupportConfig,
        Civ6MCTS,
        Civ6Network,
        HierarchicalAction,
        LossConfig,
        MicroCivNetwork,
    )
    from alpha_zero.envs.microciv import MicroCivEnv

    # Redundant aliases: these two are resolvable attributes but intentionally
    # absent from __all__, so re-export them explicitly for type checkers.
    from alpha_zero.envs.real_civ6_env import RealCiv6Env as RealCiv6Env
    from alpha_zero.envs.real_civ6_env import RewardConfig as RewardConfig
    from alpha_zero.utils.timeout_config import TimeoutConfig, get_timeout_config


def __getattr__(name: str) -> Any:
    """Resolve a public attribute on first access (PEP 562).

    Args:
        name: Attribute being looked up on the ``alpha_zero`` package.

    Returns:
        The requested object, imported from its defining module.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
        ImportError: If ``name`` is an optional export whose extra
            dependencies are not installed.
    """
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        if name in _OPTIONAL_EXPORTS:
            raise ImportError(
                f"{name} requires optional dependencies that are not installed "
                f"(while importing {module_path}): {exc}"
            ) from exc
        raise

    value = getattr(module, name)
    # Cache on the module so subsequent lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List public attributes, including lazily-resolved ones."""
    return sorted(set(__all__) | set(_LAZY_EXPORTS))
