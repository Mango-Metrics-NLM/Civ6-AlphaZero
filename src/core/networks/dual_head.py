"""Complete dual-head network combining all components."""
from __future__ import annotations

__all__ = ["DualHeadNetwork", "build_network"]

import torch
import torch.nn as nn
import torch.nn.functional as f

from src.core.config import NetworkConfig
from src.core.networks.encoders import SpatialEncoder, VectorEncoder
from src.core.networks.heads import PolicyHead, ValueHead


class DualHeadNetwork(nn.Module):
    """Complete AlphaZero-style dual-head network.

    Combines spatial encoder, vector encoder, policy head, and value head
    into a single module that satisfies the NetworkProtocol.

    Args:
        spatial_channels: Number of input spatial channels.
        spatial_size: Height/width of spatial input (assumes square).
        vector_dim: Dimension of vector input.
        action_space_size: Number of output actions.
        config: Network architecture configuration.
    """

    def __init__(
        self,
        spatial_channels: int,
        spatial_size: int,
        vector_dim: int,
        action_space_size: int,
        config: NetworkConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = config or NetworkConfig()

        self.spatial_encoder = SpatialEncoder(
            in_channels=spatial_channels,
            num_filters=cfg.num_filters,
            num_blocks=cfg.num_residual_blocks,
        )
        self.vector_encoder = VectorEncoder(
            input_dim=vector_dim,
            output_dim=cfg.fc_hidden_dim,
            dropout=cfg.dropout_rate,
        )

        self.policy_head = PolicyHead(
            spatial_channels=cfg.num_filters,
            spatial_size=spatial_size,
            vector_dim=cfg.fc_hidden_dim,
            action_space_size=action_space_size,
            hidden_dim=cfg.policy_head_hidden_dim,
        )
        self.value_head = ValueHead(
            spatial_channels=cfg.num_filters,
            spatial_size=spatial_size,
            vector_dim=cfg.fc_hidden_dim,
            hidden_dim=cfg.value_head_hidden_dim,
        )

        # Store dimensions for introspection
        self._spatial_channels = spatial_channels
        self._spatial_size = spatial_size
        self._vector_dim = vector_dim
        self._action_space_size = action_space_size

    def forward(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (policy_logits, value).

        Args:
            spatial: (B, C, H, W) spatial features.
            vector: (B, V) vector features.
            action_mask: Optional (B, A) boolean mask.

        Returns:
            (policy_logits, value) where logits shape is (B, A) and value is (B, 1).
        """
        spatial_features = self.spatial_encoder(spatial)
        vector_features = self.vector_encoder(vector)

        policy_logits = self.policy_head(
            spatial_features, vector_features, action_mask
        )
        value = self.value_head(spatial_features, vector_features)

        return policy_logits, value

    @torch.no_grad()
    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inference pass returning (policy_probs, value). No gradients tracked.

        Returns:
            (policy_probs, value) where probs sum to ~1.0 along action dim.
        """
        self.eval()
        policy_logits, value = self.forward(spatial, vector, action_mask)
        policy_probs = f.softmax(policy_logits, dim=-1)
        return policy_probs, value


def build_network(
    spatial_channels: int,
    spatial_size: int,
    vector_dim: int,
    action_space_size: int,
    config: NetworkConfig | None = None,
) -> DualHeadNetwork:
    """Factory function to create a DualHeadNetwork.

    Args:
        spatial_channels: Number of input spatial channels.
        spatial_size: Spatial dimensions (H=W).
        vector_dim: Vector feature dimension.
        action_space_size: Output action count.
        config: Optional network config.

    Returns:
        Initialized DualHeadNetwork.
    """
    return DualHeadNetwork(
        spatial_channels=spatial_channels,
        spatial_size=spatial_size,
        vector_dim=vector_dim,
        action_space_size=action_space_size,
        config=config,
    )
