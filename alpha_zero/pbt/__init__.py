"""Population-Based Training (PBT) module for Civ6 AlphaZero.

This module provides adaptive hyperparameter tuning during training using
population-based training techniques.

Components:
- config: Configuration dataclasses for PBT
- population: Population management and agent state tracking
- mutations: Hyperparameter mutation operators
- scheduler: Selection strategies and fitness evaluation
- coordinator: Integration with distributed training
"""

from alpha_zero.pbt.config import (
    AgentState,
    HyperparameterRange,
    HyperparameterSpace,
    PBTConfig,
    SelectionMethod,
)
from alpha_zero.pbt.diversity_tracker import DiversityMetrics, PolicyDiversityTracker
from alpha_zero.pbt.multiplayer_evaluator import (
    MultiplayerPBTEvaluator,
    NPlayerArenaRunner,
    NPlayerMatchResult,
)
from alpha_zero.pbt.muzero_hyperparams import (
    create_muzero_hyperparameter_space,
    merge_with_base_space,
)

__all__ = [
    "AgentState",
    "DiversityMetrics",
    "HyperparameterRange",
    "HyperparameterSpace",
    "MultiplayerPBTEvaluator",
    "NPlayerArenaRunner",
    "NPlayerMatchResult",
    "PBTConfig",
    "PolicyDiversityTracker",
    "SelectionMethod",
    "create_muzero_hyperparameter_space",
    "merge_with_base_space",
]
