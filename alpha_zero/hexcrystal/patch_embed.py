"""Vision-Transformer style patch embedding.

This module hosts :class:`PatchEmbed`, the default spatial encoder for
:class:`~alpha_zero.core.transformer_network.Civ6TransformerNetwork`. It lives in
``alpha_zero.hexcrystal`` rather than alongside the transformer so that the
encoder registry can offer both built-in encoders without importing the wider
``alpha_zero.core`` package, which would pull in numpy and gymnasium.

``alpha_zero.core.transformer_network`` re-exports :class:`PatchEmbed` for
backwards compatibility, so historical import paths continue to resolve.
"""

from __future__ import annotations

import logging

import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["PatchEmbed"]


class PatchEmbed(nn.Module):
    """Patch embedding for Vision Transformer style input processing."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        spatial_size: int,
    ) -> None:
        """Initialize patch embedding.

        Args:
            in_channels: Number of input channels.
            embed_dim: Embedding dimension.
            patch_size: Size of each patch.
            spatial_size: Size of spatial input.
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (spatial_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        logger.debug(
            "PatchEmbed: %dx%dx%d -> %d patches x %dD (patch_size=%d)",
            in_channels,
            spatial_size,
            spatial_size,
            self.num_patches,
            embed_dim,
            patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Convert spatial input to patch embeddings.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Patch embeddings (B, num_patches, embed_dim).
        """
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x
