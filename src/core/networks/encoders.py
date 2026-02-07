"""Spatial and vector feature encoders."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.core.networks.blocks import ResidualBlock


class SpatialEncoder(nn.Module):
    """ResNet-style spatial feature encoder.

    Architecture: Conv(in -> filters) -> BN -> ReLU -> N x ResidualBlock

    Args:
        in_channels: Number of input channels.
        num_filters: Number of convolutional filters.
        num_blocks: Number of residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int = 64,
        num_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.initial_conv = nn.Conv2d(
            in_channels, num_filters, 3, padding=1, bias=False
        )
        self.initial_bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.blocks = nn.Sequential(
            *(ResidualBlock(num_filters) for _ in range(num_blocks))
        )
        self.num_filters = num_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, C_in, H, W) -> (B, num_filters, H, W)."""
        x = self.relu(self.initial_bn(self.initial_conv(x)))
        result: torch.Tensor = self.blocks(x)
        return result


class VectorEncoder(nn.Module):
    """MLP encoder for non-spatial (vector) features.

    Architecture: Linear -> ReLU -> Dropout -> Linear

    Args:
        input_dim: Dimension of input vector.
        output_dim: Dimension of output embedding.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, input_dim) -> (B, output_dim)."""
        result: torch.Tensor = self.net(x)
        return result
