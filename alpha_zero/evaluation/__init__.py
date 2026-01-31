"""Evaluation module for measuring agent performance.

This module provides baseline agents and evaluation pipelines for
measuring the win rate of trained models against various opponents.
"""

from alpha_zero.evaluation.baseline_agents import (
    BaseAgent,
    GreedyHeuristicAgent,
    RandomAgent,
)
from alpha_zero.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
    EvaluationResult,
    WinRateResult,
)

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "GreedyHeuristicAgent",
    "EvaluationPipeline",
    "EvaluationResult",
    "WinRateResult",
]
