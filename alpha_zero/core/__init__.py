"""Core AlphaZero components: network, MCTS, action space, and losses."""


from alpha_zero.core.civ6_actions import ActionSpace, HierarchicalAction
from alpha_zero.core.civ6_mcts import Civ6MCTS
from alpha_zero.core.civ6_network import (
    AttentionSpatialEncoder,
    Civ6Network,
    Civ6NetworkV2,
    CrossModalAttention,
    HierarchicalPolicyHead,
    MicroCivNetwork,
    NetworkConfig,
    PositionalEncoding2D,
    SpatialSelfAttention,
    compute_loss,
)
from alpha_zero.core.losses import (
    AlphaZeroNetworkProtocol,
    # Config and data structures
    LossConfig,
    # Calculator class
    MuZeroLossCalculator,
    # Protocols
    MuZeroNetworkProtocol,
    StepLossOutput,
    TrajectoryBatch,
    TrajectoryLossOutput,
    categorical_to_scalar,
    # EfficientZero losses
    compute_consistency_loss,
    compute_muzero_loss,
    # Combined loss functions
    compute_muzero_step_loss,
    compute_muzero_trajectory_loss,
    # Core loss functions
    compute_policy_loss,
    compute_reward_loss,
    compute_value_loss,
    compute_value_prefix_loss,
    # Utility functions
    scalar_to_categorical,
)
from alpha_zero.core.transformer_network import (
    AttentionType,
    Civ6TransformerNetwork,
    NormType,
    TransformerConfig,
    compute_transformer_loss,
    create_transformer_network,
)

__all__ = [
    # Original network components
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
    # Transformer network
    "Civ6TransformerNetwork",
    "TransformerConfig",
    "AttentionType",
    "NormType",
    "compute_transformer_loss",
    "create_transformer_network",
    # MCTS and actions
    "Civ6MCTS",
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
