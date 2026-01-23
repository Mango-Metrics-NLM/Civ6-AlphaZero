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
    HierarchicalConfig,
    LevelConfig,
    GoalConfig,
    MCTSLevelConfig,
    LLMGuidanceConfig,
    HierarchyLevel,
)

# Goals
from alpha_zero.hierarchical.goals import (
    Goal,
    GoalNode,
    GoalTarget,
    GoalHierarchy,
    GoalEvaluator,
    GoalPropagator,
    GoalStatus,
    StrategicGoalType,
    TacticalGoalType,
    OperationalGoalType,
)

# Encoders
from alpha_zero.hierarchical.encoders import (
    StateEncoder,
    StrategicStateEncoder,
    TacticalStateEncoder,
    OperationalStateEncoder,
    HierarchicalStateEncoder,
    GoalEncoder,
    GoalEmbedding,
    HierarchicalGoalEncoder,
)

# Network
from alpha_zero.hierarchical.network import (
    HierarchicalNetwork,
    HierarchicalNetworkConfig,
    LevelNetwork,
    GoalConditionedHead,
)

# MCTS
from alpha_zero.hierarchical.mcts import (
    HierarchicalMCTS,
    MCTSNode,
    MCTSConfig,
    SearchStats,
    Option,
    OptionTermination,
)

# Controllers
from alpha_zero.hierarchical.controllers import (
    LevelController,
    StrategicController,
    TacticalController,
    OperationalController,
    ControllerState,
    ControllerDecision,
)

# Main Controller
from alpha_zero.hierarchical.controller import (
    HierarchicalController,
    HierarchicalState,
    HierarchicalDecision,
    create_hierarchical_controller,
)

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "HierarchicalConfig",
    "LevelConfig",
    "GoalConfig",
    "MCTSLevelConfig",
    "LLMGuidanceConfig",
    "HierarchyLevel",
    # Goals
    "Goal",
    "GoalNode",
    "GoalTarget",
    "GoalHierarchy",
    "GoalEvaluator",
    "GoalPropagator",
    "GoalStatus",
    "StrategicGoalType",
    "TacticalGoalType",
    "OperationalGoalType",
    # Encoders
    "StateEncoder",
    "StrategicStateEncoder",
    "TacticalStateEncoder",
    "OperationalStateEncoder",
    "HierarchicalStateEncoder",
    "GoalEncoder",
    "GoalEmbedding",
    "HierarchicalGoalEncoder",
    # Network
    "HierarchicalNetwork",
    "HierarchicalNetworkConfig",
    "LevelNetwork",
    "GoalConditionedHead",
    # MCTS
    "HierarchicalMCTS",
    "MCTSNode",
    "MCTSConfig",
    "SearchStats",
    "Option",
    "OptionTermination",
    # Controllers
    "LevelController",
    "StrategicController",
    "TacticalController",
    "OperationalController",
    "ControllerState",
    "ControllerDecision",
    # Main Controller
    "HierarchicalController",
    "HierarchicalState",
    "HierarchicalDecision",
    "create_hierarchical_controller",
]
