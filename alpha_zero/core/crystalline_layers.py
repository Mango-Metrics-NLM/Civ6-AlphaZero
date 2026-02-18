"""Crystalline lattice layer for symmetry-aware neural network transformations.

This module implements a crystalline lattice layer that applies learned geometric
transformations based on mathematical symmetry groups (hexagonal, cubic, tetrahedral).

CRITICAL FIX: The original implementation used random weights on every forward pass,
making the layer non-deterministic and ungradientable. This version uses learned
nn.Parameter weights for proper gradient flow.

Example:
    >>> layer = CrystallineLatticeLayer(dim=64, lattice_type="hexagonal")
    >>> x = torch.randn(32, 10, 64)  # (batch, seq, dim)
    >>> output = layer(x)  # Same shape as input
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

logger = logging.getLogger(__name__)


class SymmetryType(str, Enum):
    """Supported symmetry types for crystalline lattice transformations."""

    HEXAGONAL = "hexagonal"
    CUBIC = "cubic"
    TETRAHEDRAL = "tetrahedral"


@dataclass
class CrystallineLatticeConfig:
    """Configuration for crystalline lattice layer.

    Attributes:
        dim: Embedding dimension for the lattice transformations.
        lattice_type: Type of symmetry group to use.
        lattice_dims: Number of dimensions in the lattice basis.
        combination_mode: How to combine transformed features. Options:
            - "weighted_sum": Use learned weights with softmax.
            - "attention": Use attention mechanism to weight transformations.
    """

    dim: int = 64
    lattice_type: SymmetryType = SymmetryType.HEXAGONAL
    lattice_dims: int = 6
    combination_mode: Literal["weighted_sum", "attention"] = "weighted_sum"


class CrystallineLatticeLayer(nn.Module):
    """Crystalline lattice layer with learned geometric transformations.

    This layer applies multiple geometric transformations based on a symmetry group,
    then combines them using either learned weights or attention. The transformations
    are initialized to mathematical symmetry operations but are learned during training.

    Args:
        dim: Embedding dimension for the lattice transformations.
        lattice_type: Type of symmetry group ("hexagonal", "cubic", or "tetrahedral").
            Can be a string or SymmetryType enum.
        lattice_dims: Number of dimensions in the lattice basis.
        combination_mode: How to combine transformed features ("weighted_sum" or "attention").

    Raises:
        ValueError: If lattice_type is not recognized.
    """

    def __init__(
        self,
        dim: int,
        lattice_type: str | SymmetryType = "hexagonal",
        lattice_dims: int = 6,
        combination_mode: Literal["weighted_sum", "attention"] = "weighted_sum",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.lattice_dims = lattice_dims
        self.combination_mode = combination_mode

        # Convert string to enum if needed
        if isinstance(lattice_type, str):
            try:
                self.lattice_type = SymmetryType(lattice_type.lower())
            except ValueError as e:
                valid_types = [t.value for t in SymmetryType]
                msg = f"Invalid lattice_type: {lattice_type}. Must be one of {valid_types}"
                raise ValueError(msg) from e
        else:
            self.lattice_type = lattice_type

        # Initialize lattice basis
        self.lattice_basis = nn.Parameter(torch.randn(lattice_dims, dim) / math.sqrt(dim))

        # Initialize transformation matrices based on symmetry type
        rotation_matrices = self._initialize_transforms()
        self.lattice_transform = nn.Parameter(torch.stack(rotation_matrices))
        num_transforms = len(rotation_matrices)

        # Initialize combination mechanism
        self.combination_weights: nn.Parameter | None = None
        self.attention_proj: nn.Linear | None = None

        if combination_mode == "weighted_sum":
            # CRITICAL FIX: Use learned parameter instead of random weights
            self.combination_weights = nn.Parameter(torch.ones(num_transforms) / num_transforms)
        elif combination_mode == "attention":
            self.attention_proj = nn.Linear(dim, 1)
        else:
            msg = f"Invalid combination_mode: {combination_mode}"
            raise ValueError(msg)

        # Layer normalization for stability
        self.lattice_norm = nn.LayerNorm(dim)

        logger.debug(
            f"Initialized CrystallineLatticeLayer: dim={dim}, type={self.lattice_type.value}, "
            f"transforms={num_transforms}, mode={combination_mode}"
        )

    def _initialize_transforms(self) -> list[Tensor]:
        """Initialize transformation matrices based on symmetry type.

        Returns:
            List of transformation matrices (each dim x dim).
        """
        if self.lattice_type == SymmetryType.HEXAGONAL:
            return self._hexagonal_transforms()
        elif self.lattice_type == SymmetryType.CUBIC:
            return self._cubic_transforms()
        elif self.lattice_type == SymmetryType.TETRAHEDRAL:
            return self._tetrahedral_transforms()
        else:
            msg = f"Unknown lattice type: {self.lattice_type}"
            raise ValueError(msg)

    def _hexagonal_transforms(self) -> list[Tensor]:
        """Create hexagonal symmetry transforms (6 rotations of 60 degrees).

        Returns:
            List of 6 rotation matrices.
        """
        transforms = []
        for i in range(6):
            theta = i * 2 * math.pi / 6
            # Create identity matrix
            transform = torch.eye(self.dim)
            # Set top-left 2x2 to rotation matrix
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            transform[0, 0] = cos_theta
            transform[0, 1] = -sin_theta
            transform[1, 0] = sin_theta
            transform[1, 1] = cos_theta
            transforms.append(transform)
        return transforms

    def _cubic_transforms(self) -> list[Tensor]:
        """Create cubic symmetry transforms (8 sign flips).

        Returns:
            List of 8 sign flip matrices.
        """
        transforms = []
        for i in range(8):
            transform = torch.eye(self.dim)
            # Flip signs of first 3 dimensions based on binary representation
            if i & 1:
                transform[0, 0] = -1
            if i & 2:
                transform[1, 1] = -1
            if i & 4:
                transform[2, 2] = -1
            transforms.append(transform)
        return transforms

    def _tetrahedral_transforms(self) -> list[Tensor]:
        """Create tetrahedral symmetry transforms (12 rotations of 30 degrees).

        Returns:
            List of 12 rotation matrices.
        """
        transforms = []
        for i in range(12):
            theta = i * 2 * math.pi / 12
            # Create identity matrix
            transform = torch.eye(self.dim)
            # Set top-left 2x2 to rotation matrix
            cos_theta = math.cos(theta)
            sin_theta = math.sin(theta)
            transform[0, 0] = cos_theta
            transform[0, 1] = -sin_theta
            transform[1, 0] = sin_theta
            transform[1, 1] = cos_theta
            transforms.append(transform)
        return transforms

    def forward(self, x: Tensor) -> Tensor:
        """Apply crystalline lattice transformations to input.

        Args:
            x: Input tensor of shape (B, S, D) or (B, D) where:
                B = batch size
                S = sequence length (optional)
                D = dimension (must equal self.dim)

        Returns:
            Transformed tensor with same shape as input.

        Raises:
            ValueError: If input dimension doesn't match layer dimension.
        """
        # Handle both 2D and 3D inputs
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, D) -> (B, 1, D)
            squeeze_output = True
        elif x.dim() != 3:
            msg = f"Expected 2D or 3D input, got {x.dim()}D"
            raise ValueError(msg)

        # Validate dimension
        if x.shape[-1] != self.dim:
            msg = f"Input dimension {x.shape[-1]} doesn't match layer dimension {self.dim}"
            raise ValueError(msg)

        batch_size, seq_len, _ = x.shape

        # Apply each transformation: (B, S, D) @ (D, D)^T -> (B, S, D)
        transformed_list: list[Tensor] = []
        for i in range(self.lattice_transform.shape[0]):
            transform = self.lattice_transform[i]  # (D, D)
            # Matrix multiply: x @ transform^T
            transformed_i = torch.matmul(x, transform.T)  # (B, S, D)
            transformed_list.append(transformed_i)

        # Stack: (num_transforms, B, S, D)
        transformed = torch.stack(transformed_list, dim=0)

        # Combine transformations
        if self.combination_mode == "weighted_sum" and self.combination_weights is not None:
            # CRITICAL FIX: Use learned weights instead of random
            weights = F.softmax(self.combination_weights, dim=0)  # (num_transforms,)
            # Weighted sum: (num_transforms, 1, 1, 1) * (num_transforms, B, S, D) -> (B, S, D)
            weights = weights.view(-1, 1, 1, 1)
            combined = (weights * transformed).sum(dim=0)
        elif self.combination_mode == "attention" and self.attention_proj is not None:
            # Attention over transformations
            # (num_transforms, B, S, D) -> (num_transforms, B, S, 1)
            attn_scores = self.attention_proj(transformed)
            # Softmax over transforms: (num_transforms, B, S, 1)
            attn_weights = F.softmax(attn_scores, dim=0)
            # Weighted sum: (num_transforms, B, S, 1) * (num_transforms, B, S, D) -> (B, S, D)
            combined = (attn_weights * transformed).sum(dim=0)
        else:
            msg = f"Invalid combination_mode: {self.combination_mode}"
            raise ValueError(msg)

        # Apply layer normalization
        output = self.lattice_norm(combined)

        # Restore original shape if needed
        if squeeze_output:
            output = output.squeeze(1)  # (B, 1, D) -> (B, D)

        return cast(Tensor, output)
