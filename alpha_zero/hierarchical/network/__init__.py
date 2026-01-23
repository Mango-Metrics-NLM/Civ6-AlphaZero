"""
Hierarchical Neural Networks for Goal-Conditioned Decision Making.

This module provides neural network architectures that condition their
outputs on hierarchical goals, enabling multi-level planning and
decision making.

Example:
    >>> from alpha_zero.hierarchical.network import HierarchicalNetwork
    >>> network = HierarchicalNetwork(config)
    >>> policy, value = network(state, goal_embedding)
"""
from __future__ import annotations

from alpha_zero.hierarchical.network.hierarchical_network import (
    HierarchicalNetwork,
    HierarchicalNetworkConfig,
    LevelNetwork,
    GoalConditionedHead,
)

__all__ = [
    "HierarchicalNetwork",
    "HierarchicalNetworkConfig",
    "LevelNetwork",
    "GoalConditionedHead",
]
