"""Hexagonal spatial encoder for transformer networks.

This module provides a hexagon-aware spatial encoder that replaces PatchEmbed
in the Civ6TransformerNetwork. It uses hexagonal convolutions and optional
crystalline symmetry layers to process spatial features before tokenization.

The encoder maintains API compatibility with PatchEmbed by producing the same
output shape: (B, num_tokens, embed_dim).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch import Tensor

from alpha_zero.core.crystalline_layers import CrystallineLatticeLayer, SymmetryType
from alpha_zero.core.hex_conv import HexConvBlock

logger = logging.getLogger(__name__)


@dataclass
class HexEncoderConfig:
    """Configuration for hexagonal spatial encoder.

    Attributes:
        in_channels: Number of input spatial channels.
        embed_dim: Embedding dimension for output tokens.
        spatial_size: Size of spatial grid (height and width).
        hex_hidden_dim: Hidden dimension for hexagonal convolution stack.
        hex_conv_layers: Number of hexagonal convolution layers.
        lattice_type: Type of crystalline lattice ("hexagonal", "square", etc.).
        num_output_tokens: Number of output tokens (must match PatchEmbed.num_patches).
        dropout: Dropout probability.
        offset_type: Hex coordinate offset type ("even-q", "odd-q", etc.).
        activation: Activation function name.
        use_batch_norm: Whether to use batch normalization in hex convolutions.
        use_crystalline: Whether to use crystalline lattice layer.
    """

    in_channels: int = 13
    embed_dim: int = 64
    spatial_size: int = 8
    hex_hidden_dim: int = 32
    hex_conv_layers: int = 2
    lattice_type: str = "hexagonal"
    num_output_tokens: int = 4
    dropout: float = 0.1
    offset_type: str = "even-q"
    activation: str = "gelu"
    use_batch_norm: bool = True
    use_crystalline: bool = True


class SpatialToTokens(nn.Module):
    """Converts spatial feature maps to a fixed number of tokens.

    Uses adaptive pooling followed by linear projection to produce a fixed
    number of output tokens regardless of input spatial dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        spatial_size: int,
        num_tokens: int,
    ) -> None:
        """Initialize spatial-to-tokens converter.

        Args:
            in_channels: Number of input channels.
            embed_dim: Output embedding dimension.
            spatial_size: Input spatial size (height and width).
            num_tokens: Number of output tokens to produce.
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        self.num_tokens = num_tokens

        # Compute pooling dimensions: pool_size^2 >= num_tokens
        self.pool_size = int(math.ceil(math.sqrt(num_tokens)))
        self.pooled_elements = self.pool_size * self.pool_size

        # Adaptive pooling to fixed grid
        self.adaptive_pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))

        # Linear projection from flattened pooled features to tokens
        self.projection = nn.Linear(
            in_channels * self.pooled_elements,
            num_tokens * embed_dim,
        )

        logger.debug(
            f"SpatialToTokens: {in_channels}x{spatial_size}x{spatial_size} -> "
            f"pool({self.pool_size}x{self.pool_size}) -> {num_tokens} tokens x {embed_dim}D"
        )

    def forward(self, x: Tensor) -> Tensor:
        """Convert spatial features to tokens.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, num_tokens, embed_dim).
        """
        batch_size = x.shape[0]

        # Pool to fixed grid: (B, C, H, W) → (B, C, pool_h, pool_w)
        x = self.adaptive_pool(x)

        # Flatten spatial: (B, C, pool_h, pool_w) → (B, C * pool_h * pool_w)
        x = x.flatten(1)

        # Project to tokens: (B, C * pool_h * pool_w) → (B, num_tokens * embed_dim)
        x = self.projection(x)

        # Reshape to token sequence: (B, num_tokens * embed_dim) → (B, num_tokens, embed_dim)
        x = x.view(batch_size, self.num_tokens, self.embed_dim)

        return x


class HexSpatialEncoder(nn.Module):
    """Hexagon-aware spatial encoder for transformer networks.

    This encoder processes spatial features using hexagonal convolutions,
    converts them to a fixed number of tokens, and optionally applies
    crystalline symmetry transformations. It is designed as a drop-in
    replacement for PatchEmbed with the same output shape.
    """

    def __init__(
        self,
        config: HexEncoderConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize hexagonal spatial encoder.

        Args:
            config: Encoder configuration. If None, created from kwargs.
            **kwargs: Override config parameters or provide them directly.
                Accepted keys: in_channels, embed_dim, spatial_size,
                patch_size (for API compatibility), hex_hidden_dim,
                hex_conv_layers, lattice_type, num_output_tokens, dropout,
                offset_type, activation, use_batch_norm, use_crystalline.
        """
        super().__init__()

        # Handle config initialization
        if config is None:
            # Extract config fields from kwargs
            _valid_fields = {
                "in_channels",
                "embed_dim",
                "spatial_size",
                "hex_hidden_dim",
                "hex_conv_layers",
                "lattice_type",
                "num_output_tokens",
                "dropout",
                "offset_type",
                "activation",
                "use_batch_norm",
                "use_crystalline",
            }
            config_kwargs: dict[str, Any] = {k: v for k, v in kwargs.items() if k in _valid_fields}
            config = HexEncoderConfig(**config_kwargs)

        self.config = config

        # Build hexagonal convolution stack
        hex_layers = []
        for i in range(config.hex_conv_layers):
            in_ch = config.in_channels if i == 0 else config.hex_hidden_dim
            out_ch = config.hex_hidden_dim

            hex_layers.append(
                HexConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    board_size=config.spatial_size,
                    offset_type=config.offset_type,
                    activation=config.activation,
                    use_batch_norm=config.use_batch_norm,
                )
            )

        self.hex_conv_stack = nn.Sequential(*hex_layers)

        # Spatial to tokens conversion
        self.spatial_to_tokens = SpatialToTokens(
            in_channels=config.hex_hidden_dim,
            embed_dim=config.embed_dim,
            spatial_size=config.spatial_size,
            num_tokens=config.num_output_tokens,
        )

        # Optional crystalline lattice layer
        self.use_crystalline = config.use_crystalline
        self.crystalline: CrystallineLatticeLayer | None = None
        if self.use_crystalline:
            # Map lattice_type string to SymmetryType enum
            try:
                symmetry_type = SymmetryType(config.lattice_type)
            except ValueError:
                logger.warning(
                    f"Unknown lattice type '{config.lattice_type}', falling back to HEXAGONAL"
                )
                symmetry_type = SymmetryType.HEXAGONAL

            self.crystalline = CrystallineLatticeLayer(
                dim=config.embed_dim,
                lattice_type=symmetry_type,
                lattice_dims=6,  # Standard for 2D hexagonal lattice
            )

        # Output projection and dropout
        self.output_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        logger.debug(
            f"HexSpatialEncoder initialized: {config.in_channels}ch -> "
            f"{config.hex_conv_layers}xHexConv({config.hex_hidden_dim}) -> "
            f"{config.num_output_tokens} tokens x {config.embed_dim}D, "
            f"crystalline={config.use_crystalline}"
        )

    @property
    def num_patches(self) -> int:
        """Return number of output tokens (for PatchEmbed API compatibility).

        Returns:
            Number of output tokens.
        """
        return self.config.num_output_tokens

    def forward(self, x: Tensor) -> Tensor:
        """Encode spatial features to token sequence.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, num_tokens, embed_dim).
        """
        # Apply hexagonal convolution stack: (B, C, H, W) → (B, hex_hidden_dim, H, W)
        x = self.hex_conv_stack(x)

        # Convert to tokens: (B, hex_hidden_dim, H, W) → (B, num_tokens, embed_dim)
        x = self.spatial_to_tokens(x)

        # Apply crystalline lattice symmetry if enabled
        if self.use_crystalline and self.crystalline is not None:
            x = self.crystalline(x)

        # Output projection and dropout
        x = self.output_proj(x)
        x = self.dropout(x)

        return x


def create_hex_encoder(
    in_channels: int = 13,
    embed_dim: int = 64,
    patch_size: int = 4,
    spatial_size: int = 8,
    **kwargs: Any,
) -> HexSpatialEncoder:
    """Factory function to create a hexagonal spatial encoder.

    This function provides API compatibility with PatchEmbed factories by
    accepting patch_size and computing num_output_tokens to match the
    expected behavior: num_tokens = (spatial_size // patch_size)^2.

    Args:
        in_channels: Number of input spatial channels.
        embed_dim: Embedding dimension for output tokens.
        patch_size: Patch size (used to compute num_output_tokens).
        spatial_size: Size of spatial grid (height and width).
        **kwargs: Additional config parameters (hex_hidden_dim, hex_conv_layers, etc.).

    Returns:
        Initialized HexSpatialEncoder instance.
    """
    # Compute num_output_tokens from patch_size to match PatchEmbed behavior
    num_output_tokens = (spatial_size // patch_size) ** 2

    # Merge computed value with provided kwargs
    factory_kwargs: dict[str, Any] = {
        "in_channels": in_channels,
        "embed_dim": embed_dim,
        "spatial_size": spatial_size,
        "num_output_tokens": num_output_tokens,
        **kwargs,
    }

    logger.debug(
        f"create_hex_encoder: patch_size={patch_size} → num_output_tokens={num_output_tokens}"
    )

    return HexSpatialEncoder(config=HexEncoderConfig(**factory_kwargs))
