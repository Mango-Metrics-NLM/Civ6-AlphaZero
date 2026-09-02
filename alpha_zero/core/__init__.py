"""Core AlphaZero components: network, MCTS, action space, and losses."""

from alpha_zero.core.civ6_actions import ActionSpace, HierarchicalAction
from alpha_zero.core.civ6_mcts import (
    Civ6MCTS,
    GumbelMCTS,
    GumbelMCTSConfig,
    MCTSConfig,
    MCTSNode,
    MuZeroMCTS,
    MuZeroMCTSConfig,
    SimpleMCTS,
)
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
    CategoricalSupportConfig,
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
    create_transformer_network,
)

# JAX backend (optional - only available if JAX is installed)
try:
    from alpha_zero.core.jax_network import (  # noqa: F401
        JaxCiv6Network,
        JaxMicroCivNetwork,
        JaxNetworkWrapper,
        create_jax_network,
    )

    __all_jax__ = [
        "JaxCiv6Network",
        "JaxMicroCivNetwork",
        "JaxNetworkWrapper",
        "create_jax_network",
    ]
except ImportError:
    __all_jax__ = []

__all__ = [
    *__all_jax__,
    "ActionSpace",
    # Network heads (reusable components)
    "AlphaZeroNetworkProtocol",
    "AttentionSpatialEncoder",
    "AttentionType",
    # Loss config and data structures
    "CategoricalSupportConfig",
    # MCTS and actions
    "Civ6MCTS",
    # Original network components
    "Civ6Network",
    "Civ6NetworkV2",
    # Transformer network
    "Civ6TransformerNetwork",
    "CrossModalAttention",
    "GumbelMCTS",
    "GumbelMCTSConfig",
    "HierarchicalAction",
    "HierarchicalPolicyHead",
    "LossConfig",
    "MCTSConfig",
    "MCTSNode",
    "MicroCivNetwork",
    # Calculator class
    "MuZeroLossCalculator",
    "MuZeroMCTS",
    "MuZeroMCTSConfig",
    # Protocols
    "MuZeroNetworkProtocol",
    "NetworkConfig",
    "NormType",
    "PositionalEncoding2D",
    "SimpleMCTS",
    "SpatialSelfAttention",
    "StepLossOutput",
    "TrajectoryBatch",
    "TrajectoryLossOutput",
    "TransformerConfig",
    "categorical_to_scalar",
    # EfficientZero losses
    "compute_consistency_loss",
    "compute_loss",
    "compute_muzero_loss",
    # Combined loss functions
    "compute_muzero_step_loss",
    "compute_muzero_trajectory_loss",
    # Core loss functions
    "compute_policy_loss",
    "compute_reward_loss",
    "compute_value_loss",
    "compute_value_prefix_loss",
    "create_transformer_network",
    # Loss utility functions
    "scalar_to_categorical",
]
