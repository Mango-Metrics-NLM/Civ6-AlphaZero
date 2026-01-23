"""
State and Goal Encoders for Hierarchical Reasoning.

This module provides encoders that transform game state and goals into
tensor representations suitable for neural network processing at different
hierarchy levels.

Example:
    >>> from alpha_zero.hierarchical.encoders import (
    ...     StrategicStateEncoder, TacticalStateEncoder, GoalEncoder
    ... )
    >>> encoder = StrategicStateEncoder(config)
    >>> state_tensor = encoder.encode(game_state)
"""
from __future__ import annotations

from alpha_zero.hierarchical.encoders.state_encoder import (
    StateEncoder,
    StrategicStateEncoder,
    TacticalStateEncoder,
    OperationalStateEncoder,
    HierarchicalStateEncoder,
)
from alpha_zero.hierarchical.encoders.goal_encoder import (
    GoalEncoder,
    GoalEmbedding,
    HierarchicalGoalEncoder,
)

__all__ = [
    # State encoders
    "StateEncoder",
    "StrategicStateEncoder",
    "TacticalStateEncoder",
    "OperationalStateEncoder",
    "HierarchicalStateEncoder",
    # Goal encoders
    "GoalEncoder",
    "GoalEmbedding",
    "HierarchicalGoalEncoder",
]
