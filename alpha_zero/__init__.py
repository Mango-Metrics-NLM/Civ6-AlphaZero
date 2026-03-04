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
        RealCiv6Env - Real Civ6 game integration

    Configuration:
        TimeoutConfig - Centralized timeout configuration
        RewardConfig - Reward shaping configuration

Usage:
    from alpha_zero import Civ6MCTS, ActionSpace, MicroCivEnv
"""

__version__ = "0.14.0"

# Python <3.11 compatibility: patch datetime.UTC before anything imports it
import alpha_zero._compat  # noqa: F401  # isort: skip

# Core components
from alpha_zero.core import (
    ActionSpace,
    CategoricalSupportConfig,
    Civ6MCTS,
    Civ6Network,
    HierarchicalAction,
    LossConfig,
    MicroCivNetwork,
)

# Environments
from alpha_zero.envs.microciv import MicroCivEnv

# Configuration
from alpha_zero.utils.timeout_config import TimeoutConfig, get_timeout_config

# Optional: Real Civ6 environment (may have additional dependencies)
try:
    from alpha_zero.envs.real_civ6_env import RealCiv6Env, RewardConfig  # noqa: F401

    _REAL_CIV6_AVAILABLE = True
except ImportError:
    _REAL_CIV6_AVAILABLE = False

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

# Add optional exports if available
if _REAL_CIV6_AVAILABLE:
    __all__.extend(["RealCiv6Env", "RewardConfig"])
