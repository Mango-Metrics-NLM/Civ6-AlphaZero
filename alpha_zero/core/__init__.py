"""Core AlphaZero components: network, MCTS, action space."""

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
]
