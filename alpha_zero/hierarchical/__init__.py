"""
Hierarchical Reasoning System for Civ6 AlphaZero.

This module implements a multi-level decision-making system that operates
at strategic (30-100 turns), tactical (5-15 turns), and operational (1-3 turns)
time horizons.

Key Components:
    - Goals: Hierarchical goal management with propagation
    - Encoders: State and goal encoding for neural networks
    - Network: Goal-conditioned hierarchical neural network
    - MCTS: Multi-level Monte Carlo Tree Search with options
    - Controllers: Level-specific decision controllers
    - Controller: Main hierarchical controller

Example:
    >>> from alpha_zero.hierarchical import HierarchicalController, HierarchicalConfig
    >>> config = HierarchicalConfig.full()
    >>> controller = HierarchicalController(config)
    >>> decision = controller.decide(game_state)
    >>> action = decision.action
"""

from __future__ import annotations

# Configuration
from alpha_zero.hierarchical.config import (
    GoalConfig,
    HierarchicalConfig,
    HierarchyLevel,
    LevelConfig,
    LLMGuidanceConfig,
    MCTSLevelConfig,
)

# Main Controller
from alpha_zero.hierarchical.controller import (
    HierarchicalController,
    HierarchicalDecision,
    HierarchicalState,
    create_hierarchical_controller,
)

# Controllers
from alpha_zero.hierarchical.controllers import (
    ControllerDecision,
    ControllerState,
    LevelController,
    OperationalController,
    StrategicController,
    TacticalController,
)

# Encoders
from alpha_zero.hierarchical.encoders import (
    GoalEmbedding,
    GoalEncoder,
    HierarchicalGoalEncoder,
    HierarchicalStateEncoder,
    OperationalStateEncoder,
    StateEncoder,
    StrategicStateEncoder,
    TacticalStateEncoder,
)

# Goals
from alpha_zero.hierarchical.goals import (
    Goal,
    GoalEvaluator,
    GoalHierarchy,
    GoalNode,
    GoalPropagator,
    GoalStatus,
    GoalTarget,
    OperationalGoalType,
    StrategicGoalType,
    TacticalGoalType,
)

# MCTS
from alpha_zero.hierarchical.mcts import (
    HierarchicalMCTS,
    MCTSConfig,
    MCTSNode,
    Option,
    OptionTermination,
    SearchStats,
)

# Network
from alpha_zero.hierarchical.network import (
    GoalConditionedHead,
    HierarchicalNetwork,
    HierarchicalNetworkConfig,
    LevelNetwork,
)

__version__ = "1.0.0"

__all__ = [
    "ControllerDecision",
    "ControllerState",
    # Goals
    "Goal",
    "GoalConditionedHead",
    "GoalConfig",
    "GoalEmbedding",
    "GoalEncoder",
    "GoalEvaluator",
    "GoalHierarchy",
    "GoalNode",
    "GoalPropagator",
    "GoalStatus",
    "GoalTarget",
    # Configuration
    "HierarchicalConfig",
    # Main Controller
    "HierarchicalController",
    "HierarchicalDecision",
    "HierarchicalGoalEncoder",
    # MCTS
    "HierarchicalMCTS",
    # Network
    "HierarchicalNetwork",
    "HierarchicalNetworkConfig",
    "HierarchicalState",
    "HierarchicalStateEncoder",
    "HierarchyLevel",
    "LLMGuidanceConfig",
    "LevelConfig",
    # Controllers
    "LevelController",
    "LevelNetwork",
    "MCTSConfig",
    "MCTSLevelConfig",
    "MCTSNode",
    "OperationalController",
    "OperationalGoalType",
    "OperationalStateEncoder",
    "Option",
    "OptionTermination",
    "SearchStats",
    # Encoders
    "StateEncoder",
    "StrategicController",
    "StrategicGoalType",
    "StrategicStateEncoder",
    "TacticalController",
    "TacticalGoalType",
    "TacticalStateEncoder",
    "create_hierarchical_controller",
]
