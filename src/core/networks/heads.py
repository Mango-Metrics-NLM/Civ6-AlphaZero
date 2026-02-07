"""Policy and value head implementations for dual-head networks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f


class PolicyHead(nn.Module):
    """MLP policy head that outputs logits over a discrete action space.

    Architecture:
        spatial_features (B, C, H, W) -> Conv2d(C, 2, 1x1) -> BN -> ReLU -> Flatten
        concat with vector_features (B, V)
        -> Linear(flat_dim + V, hidden_dim) -> ReLU -> Linear(hidden_dim, action_space_size)

    Args:
        spatial_channels: Number of input spatial feature channels.
        spatial_size: Height/width of the spatial features (assumes square).
        vector_dim: Dimension of the vector features.
        action_space_size: Number of discrete actions.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        spatial_channels: int,
        spatial_size: int,
        vector_dim: int,
        action_space_size: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(spatial_channels, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        flat_dim = 2 * spatial_size * spatial_size
        self.fc1 = nn.Linear(flat_dim + vector_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_space_size)
        self._action_space_size = action_space_size

    @property
    def action_space_size(self) -> int:
        return self._action_space_size

    def forward(
        self,
        spatial_features: torch.Tensor,
        vector_features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute policy logits.

        Args:
            spatial_features: (B, C, H, W) from backbone.
            vector_features: (B, V) encoded vector features.
            action_mask: Optional (B, A) boolean mask. Invalid actions get -1e9.

        Returns:
            Policy logits of shape (B, action_space_size).
        """
        x = f.relu(self.bn(self.conv(spatial_features)))
        x = x.flatten(1)  # (B, 2*H*W)
        x = torch.cat([x, vector_features], dim=1)
        x = f.relu(self.fc1(x))
        logits = self.fc2(x)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)

        return logits


class ValueHead(nn.Module):
    """MLP value head that outputs a scalar value in [-1, 1].

    Architecture:
        spatial_features (B, C, H, W) -> Conv2d(C, 1, 1x1) -> BN -> ReLU -> Flatten
        concat with vector_features (B, V)
        -> Linear(flat_dim + V, hidden_dim) -> ReLU -> Linear(hidden_dim, 1) -> Tanh

    Args:
        spatial_channels: Number of input spatial feature channels.
        spatial_size: Height/width of the spatial features.
        vector_dim: Dimension of the vector features.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        spatial_channels: int,
        spatial_size: int,
        vector_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(spatial_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        flat_dim = 1 * spatial_size * spatial_size
        self.fc1 = nn.Linear(flat_dim + vector_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        spatial_features: torch.Tensor,
        vector_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value estimate.

        Args:
            spatial_features: (B, C, H, W) from backbone.
            vector_features: (B, V) encoded vector features.

        Returns:
            Value tensor of shape (B, 1) in [-1, 1].
        """
        x = f.relu(self.bn(self.conv(spatial_features)))
        x = x.flatten(1)  # (B, H*W)
        x = torch.cat([x, vector_features], dim=1)
        x = f.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))
