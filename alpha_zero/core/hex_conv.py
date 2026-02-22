"""Hexagonal topology-aware convolution for Civ6 AlphaZero.

Civ6 uses hexagonal tiles represented on a rectangular grid using "even-q" offset
coordinates. This module implements proper hexagonal convolution that respects the
actual hex adjacency structure, avoiding the rectangular connectivity artifacts of
standard Conv2d.

Key concepts:
- Even-q offset coordinates: flat-top hexagons, even columns offset down
- 7-tap kernel: center tile + 6 hex neighbors
- Pre-computed neighbor indices for efficient gathering
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class HexConvConfig:
    """Configuration for hexagonal convolution layers.

    Attributes:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        board_size: Size of the square board (height and width).
        offset_type: Hex offset coordinate system ("even-q" for flat-top even columns).
        num_layers: Number of hex convolution layers to stack.
        activation: Activation function name ("gelu", "relu", or "silu").
        use_bias: Whether to use bias in convolution layers.
        use_batch_norm: Whether to use batch normalization.
    """

    in_channels: int = 13
    out_channels: int = 64
    board_size: int = 8
    offset_type: str = "even-q"
    num_layers: int = 2
    activation: str = "gelu"
    use_bias: bool = True
    use_batch_norm: bool = True


class HexAdjacency:
    """Pre-computes hexagonal neighbor indices for offset coordinate grids.

    For "even-q" offset coordinates (flat-top hexagons):
    - Even columns (c % 2 == 0): neighbors at (c-1,r-1), (c-1,r), (c,r-1), (c,r+1), (c+1,r-1), (c+1,r)
    - Odd columns (c % 2 == 1): neighbors at (c-1,r), (c-1,r+1), (c,r-1), (c,r+1), (c+1,r), (c+1,r+1)

    This class is a utility for computing neighbor indices, not an nn.Module.
    """

    def __init__(self, board_size: int, offset_type: str = "even-q") -> None:
        """Initialize hex adjacency computation.

        Args:
            board_size: Size of the square board (height and width).
            offset_type: Hex offset coordinate system (currently only "even-q" supported).

        Raises:
            ValueError: If offset_type is not supported.
        """
        if offset_type != "even-q":
            raise ValueError(f"Only 'even-q' offset type is currently supported, got {offset_type}")

        self.board_size = board_size
        self.offset_type = offset_type
        self.neighbor_indices = self._compute_neighbors()

    def _compute_neighbors(self) -> Tensor:
        """Compute neighbor indices for all board positions.

        Returns:
            Tensor of shape (board_size*board_size, 6) where each entry is a flat
            index into the board, with -1 for invalid (out-of-bounds) neighbors.
        """
        size = self.board_size
        num_positions = size * size
        neighbors = torch.full((num_positions, 6), -1, dtype=torch.long)

        for r in range(size):
            for c in range(size):
                flat_idx = r * size + c

                # Even-q offset coordinates: different neighbors for even/odd columns
                if c % 2 == 0:  # Even column
                    neighbor_offsets = [
                        (c - 1, r - 1),  # NW
                        (c - 1, r),  # SW
                        (c, r - 1),  # N
                        (c, r + 1),  # S
                        (c + 1, r - 1),  # NE
                        (c + 1, r),  # SE
                    ]
                else:  # Odd column
                    neighbor_offsets = [
                        (c - 1, r),  # NW
                        (c - 1, r + 1),  # SW
                        (c, r - 1),  # N
                        (c, r + 1),  # S
                        (c + 1, r),  # NE
                        (c + 1, r + 1),  # SE
                    ]

                # Convert to flat indices, mark out-of-bounds as -1
                for i, (nc, nr) in enumerate(neighbor_offsets):
                    if 0 <= nc < size and 0 <= nr < size:
                        neighbors[flat_idx, i] = nr * size + nc

        return neighbors

    def get_neighbor_indices(self) -> Tensor:
        """Get pre-computed neighbor indices.

        Returns:
            Tensor of shape (board_size*board_size, 6) with neighbor flat indices.
        """
        return self.neighbor_indices


class HexConvolution(nn.Module):
    """Hexagonal convolution layer with 7-tap kernel (center + 6 neighbors).

    This layer learns separate weights for the center tile and each of the 6 hex
    neighbors, applying them via gather operations based on pre-computed adjacency.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        board_size: int = 8,
        offset_type: str = "even-q",
        bias: bool = True,
    ) -> None:
        """Initialize hexagonal convolution layer.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            board_size: Size of the square board (height and width).
            offset_type: Hex offset coordinate system.
            bias: Whether to use bias term.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.board_size = board_size
        self.offset_type = offset_type

        # Weight: (out_channels, in_channels, 7) where 7 = center + 6 neighbors
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 7))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Pre-compute hex adjacency
        hex_adj = HexAdjacency(board_size, offset_type)
        neighbor_indices = hex_adj.get_neighbor_indices()

        # Register buffers for device transfer
        # Type annotations for mypy (register_buffer creates Tensor attributes)
        self.neighbor_indices: Tensor
        self.valid_mask: Tensor
        self.register_buffer("neighbor_indices", neighbor_indices)

        # Valid mask: 1 where neighbor exists, 0 where -1
        valid_mask = (neighbor_indices >= 0).float()
        self.register_buffer("valid_mask", valid_mask)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming normal initialization."""
        # Each position has 7 taps (center + 6 neighbors), treat like kernel_size=7
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply hexagonal convolution.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H, W).
        """
        b, c_in, h, w = x.shape
        assert c_in == self.in_channels
        assert h == self.board_size and w == self.board_size

        num_positions = h * w

        # Flatten spatial dimensions: (B, C_in, H*W)
        x_flat = x.view(b, c_in, num_positions)

        # Gather center features: (B, C_in, H*W, 1)
        center_features = x_flat.unsqueeze(-1)

        # Gather neighbor features
        # neighbor_indices: (H*W, 6)
        # Clamp -1 to 0 to avoid index errors, will mask out later
        neighbor_idx_safe = self.neighbor_indices.clamp(min=0)

        # Expand for batch and channel dims: (B, C_in, H*W, 6)
        neighbor_features = x_flat[:, :, neighbor_idx_safe]  # (B, C_in, H*W, 6)

        # Apply valid mask: zero out invalid neighbors
        # valid_mask: (H*W, 6) -> (1, 1, H*W, 6)
        mask = self.valid_mask.unsqueeze(0).unsqueeze(0)
        neighbor_features = neighbor_features * mask

        # Stack center + neighbors: (B, C_in, H*W, 7)
        features = torch.cat([center_features, neighbor_features], dim=-1)

        # Apply convolution weights
        # weight: (C_out, C_in, 7)
        # features: (B, C_in, H*W, 7)
        # Output: (B, C_out, H*W)
        output = torch.einsum("bcht,oct->boh", features, self.weight)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        # Reshape back to spatial: (B, C_out, H, W)
        output = output.view(b, self.out_channels, h, w)

        return output


class HexConvBlock(nn.Module):
    """Hexagonal convolution block with optional batch norm, activation, and residual.

    Structure: HexConv -> [BatchNorm] -> Activation -> [+Residual]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        board_size: int = 8,
        offset_type: str = "even-q",
        activation: str = "gelu",
        use_batch_norm: bool = True,
        residual: bool = True,
    ) -> None:
        """Initialize hexagonal convolution block.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            board_size: Size of the square board (height and width).
            offset_type: Hex offset coordinate system.
            activation: Activation function name ("gelu", "relu", or "silu").
            use_batch_norm: Whether to use batch normalization.
            residual: Whether to add residual connection.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual

        # Main convolution
        self.hex_conv = HexConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            board_size=board_size,
            offset_type=offset_type,
            bias=not use_batch_norm,  # No bias if using batch norm
        )

        # Optional batch normalization
        self.batch_norm: nn.BatchNorm2d | None = None
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)

        # Activation function
        self.act: nn.Module
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Residual connection (1x1 conv if channels mismatch)
        self.skip_conv: nn.Conv2d | None = None
        if residual and in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply hexagonal convolution block.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H, W).
        """
        identity = x

        # HexConv
        out = self.hex_conv(x)

        # BatchNorm
        if self.batch_norm is not None:
            out = self.batch_norm(out)

        # Activation
        out = self.act(out)

        # Residual
        if self.residual:
            if self.skip_conv is not None:
                identity = self.skip_conv(identity)
            out = out + identity

        return cast(Tensor, out)
