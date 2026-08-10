"""AlphaZero implementation for Civilization VI.

This package provides the core AlphaZero components for training an AI
agent to play Civilization VI.

Public API (PEP 562 lazy-loaded — importing this package does **not** pull in
numpy, gymnasium, or any other heavy dependency until you actually access the
attribute):

    Core:
        Civ6MCTS - Monte Carlo Tree Search implementation
        ActionSpace - Action space handling for the game
        HierarchicalAction - Hierarchical action representation
        Civ6Network, MicroCivNetwork - Neural network architectures
        LossConfig, CategoricalSupportConfig - Loss function configuration

    Environments:
        MicroCivEnv - Simplified Civ6 training environment
        RealCiv6Env - Real Civ6 game integration

    Configuration:
        TimeoutConfig - Centralized timeout configuration
        RewardConfig - Reward shaping configuration

Usage:
    from alpha_zero import Civ6MCTS, ActionSpace, MicroCivEnv
"""

from __future__ import annotations

__version__ = "0.14.0"

# Python <3.11 compatibility: patch datetime.UTC before anything imports it.
# This module must be imported eagerly; it is stdlib-only and side-effect-free.
import alpha_zero._compat  # noqa: F401  # isort: skip

# ---------------------------------------------------------------------------
# PEP 562 lazy attribute loader
# ---------------------------------------------------------------------------
# Import nothing eagerly from alpha_zero.core — that subpackage transitively
# imports numpy and gymnasium.  Instead, resolve each public name on first
# access and cache it on the module so subsequent accesses are O(1).

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # (module_path, attribute_name)
    "ActionSpace": ("alpha_zero.core", "ActionSpace"),
    "CategoricalSupportConfig": ("alpha_zero.core", "CategoricalSupportConfig"),
    "Civ6MCTS": ("alpha_zero.core", "Civ6MCTS"),
    "Civ6Network": ("alpha_zero.core", "Civ6Network"),
    "HierarchicalAction": ("alpha_zero.core", "HierarchicalAction"),
    "LossConfig": ("alpha_zero.core", "LossConfig"),
    "MicroCivNetwork": ("alpha_zero.core", "MicroCivNetwork"),
    "MicroCivEnv": ("alpha_zero.envs.microciv", "MicroCivEnv"),
    "TimeoutConfig": ("alpha_zero.utils.timeout_config", "TimeoutConfig"),
    "get_timeout_config": ("alpha_zero.utils.timeout_config", "get_timeout_config"),
}

_OPTIONAL_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "RealCiv6Env": ("alpha_zero.envs.real_civ6_env", "RealCiv6Env"),
    "RewardConfig": ("alpha_zero.envs.real_civ6_env", "RewardConfig"),
}


def __getattr__(name: str) -> object:
    """Resolve public names lazily on first access (PEP 562)."""
    import importlib

    if name in _LAZY_IMPORTS:
        mod_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(mod_path)
        obj = getattr(mod, attr)
        # Cache on the module so subsequent accesses bypass __getattr__.
        globals()[name] = obj
        return obj

    if name in _OPTIONAL_LAZY_IMPORTS:
        mod_path, attr = _OPTIONAL_LAZY_IMPORTS[name]
        try:
            mod = importlib.import_module(mod_path)
            obj = getattr(mod, attr)
            globals()[name] = obj
            return obj
        except ImportError as exc:
            raise AttributeError(
                f"alpha_zero.{name} requires optional dependencies: {exc}"
            ) from exc

    raise AttributeError(f"module 'alpha_zero' has no attribute {name!r}")


__all__ = [
    # Core
    "ActionSpace",
    "CategoricalSupportConfig",
    "Civ6MCTS",
    "Civ6Network",
    "HierarchicalAction",
    "LossConfig",
    # Environments
    "MicroCivEnv",
    "MicroCivNetwork",
    "RealCiv6Env",
    "RewardConfig",
    # Configuration
    "TimeoutConfig",
    "__version__",
    "get_timeout_config",
]
