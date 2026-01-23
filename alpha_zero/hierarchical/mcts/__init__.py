"""
Hierarchical Monte Carlo Tree Search with Options Framework.

This module implements multi-level MCTS that operates at different
abstraction levels with temporal abstraction through options.

Example:
    >>> from alpha_zero.hierarchical.mcts import HierarchicalMCTS, MCTSConfig
    >>> config = MCTSConfig.for_level(HierarchyLevel.TACTICAL)
    >>> mcts = HierarchicalMCTS(network, config)
    >>> action, stats = mcts.search(state, goal)
"""
from __future__ import annotations

from alpha_zero.hierarchical.mcts.hierarchical_mcts import (
    HierarchicalMCTS,
    MCTSNode,
    MCTSConfig,
    SearchStats,
    Option,
    OptionTermination,
)

__all__ = [
    "HierarchicalMCTS",
    "MCTSNode",
    "MCTSConfig",
    "SearchStats",
    "Option",
    "OptionTermination",
]
