"""Core AlphaZero components: network, MCTS, action space, and losses."""

from alpha_zero.core.civ6_network import (
    Civ6Network,
    Civ6NetworkV2,
    MicroCivNetwork,
    NetworkConfig,
    PositionalEncoding2D,
    SpatialSelfAttention,
    CrossModalAttention,
    AttentionSpatialEncoder,
    HierarchicalPolicyHead,
    compute_loss,
)
from alpha_zero.core.civ6_mcts import Civ6MCTS
from alpha_zero.core.civ6_actions import ActionSpace, HierarchicalAction
from alpha_zero.core.losses import (
    # Config and data structures
    LossConfig,
    StepLossOutput,
    TrajectoryLossOutput,
    TrajectoryBatch,
    # Utility functions
    scalar_to_categorical,
    categorical_to_scalar,
    # Core loss functions
    compute_policy_loss,
    compute_value_loss,
    compute_reward_loss,
    # EfficientZero losses
    compute_consistency_loss,
    compute_value_prefix_loss,
    # Combined loss functions
    compute_muzero_step_loss,
    compute_muzero_trajectory_loss,
    compute_muzero_loss,
    # Calculator class
    MuZeroLossCalculator,
    # Protocols
    MuZeroNetworkProtocol,
    AlphaZeroNetworkProtocol,
)

__all__ = [
    # Network components
    "Civ6Network",
    "Civ6NetworkV2",
    "MicroCivNetwork",
    "NetworkConfig",
    "PositionalEncoding2D",
    "SpatialSelfAttention",
    "CrossModalAttention",
    "AttentionSpatialEncoder",
    "HierarchicalPolicyHead",
    "compute_loss",
    # MCTS
    "Civ6MCTS",
    # Actions
    "ActionSpace",
    "HierarchicalAction",
    # Loss config and data structures
    "LossConfig",
    "StepLossOutput",
    "TrajectoryLossOutput",
    "TrajectoryBatch",
    # Loss utility functions
    "scalar_to_categorical",
    "categorical_to_scalar",
    # Core loss functions
    "compute_policy_loss",
    "compute_value_loss",
    "compute_reward_loss",
    # EfficientZero losses
    "compute_consistency_loss",
    "compute_value_prefix_loss",
    # Combined loss functions
    "compute_muzero_step_loss",
    "compute_muzero_trajectory_loss",
    "compute_muzero_loss",
    # Calculator class
    "MuZeroLossCalculator",
    # Protocols
    "MuZeroNetworkProtocol",
    "AlphaZeroNetworkProtocol",
]
