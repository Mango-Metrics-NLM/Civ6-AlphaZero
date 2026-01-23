"""Agent modules for strategic AI decision-making.

This package contains agent implementations that combine neural network outputs
with strategic reasoning, uncertainty quantification, and hierarchical utility.
"""

from alpha_zero.agents.logic_agent import (
    GameContext,
    LogicAgent,
    LogicAgentConfig,
    StrategicDecision,
    UncertaintyReport,
)

__all__ = [
    "LogicAgent",
    "LogicAgentConfig",
    "GameContext",
    "StrategicDecision",
    "UncertaintyReport",
]
