"""Domain-agnostic strategic neural network for multi-objective planning.

Provides a flexible neural network architecture that supports:
- Optional spatial (convolutional) encoding for grid-based domains
- Vector encoding for global/non-spatial features
- Multi-objective value head (one output per objective)
- Action masking for valid action enforcement
- Configurable depth, width, attention, and dropout

The network auto-adapts to the domain: if spatial_channels=0, the
spatial encoder is skipped entirely. When num_objectives=1, the value
output is squeezed to scalar for backward compatibility with standard
AlphaZero training pipelines.

Example:
    >>> net = StrategicNetwork(
    ...     spatial_channels=13, spatial_size=16,
    ...     vector_dim=64, action_space_size=356,
    ...     num_objectives=3, num_res_blocks=4,
    ... )
    >>> spatial = torch.randn(2, 13, 16, 16)
    >>> vector = torch.randn(2, 64)
    >>> policy_logits, values = net(spatial, vector)
    >>> print(policy_logits.shape)  # (2, 356)
    >>> print(values.shape)         # (2, 3)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StrategicNetwork(nn.Module):
    """Domain-agnostic neural network for multi-objective strategic planning.

    Architecture overview:
        Spatial Input (C, H, W) ─► SpatialEncoder ─► spatial_features
                                                          │
        Vector Input (D,) ─────► VectorEncoder ──► vector_features
                                                          │
                                              ┌───── combined ─────┐
                                              │                    │
                                        PolicyHead          MultiValueHead
                                              │                    │
                                     policy_logits (A,)    values (num_obj,)

    When spatial_channels=0, the spatial encoder is omitted and only
    vector features drive the policy and value heads.

    Args:
        spatial_channels: Number of spatial input channels (0 = no spatial).
        spatial_size: Height/width of spatial input (assumes square).
        vector_dim: Dimension of vector (non-spatial) input features.
        action_space_size: Total number of discrete actions.
        num_objectives: Number of value head outputs.
        num_res_blocks: Number of residual blocks in spatial encoder.
        num_filters: Number of convolutional filters.
        value_head_hidden: Hidden layer size in value head.
        policy_head_hidden: Hidden layer size in policy head.
        dropout: Dropout probability (0 = no dropout).
        use_attention: Whether to add spatial self-attention.
    """

    def __init__(
        self,
        spatial_channels: int = 0,
        spatial_size: int = 0,
        vector_dim: int = 64,
        action_space_size: int = 100,
        num_objectives: int = 1,
        num_res_blocks: int = 4,
        num_filters: int = 64,
        value_head_hidden: int = 128,
        policy_head_hidden: int = 128,
        dropout: float = 0.0,
        use_attention: bool = False,
    ) -> None:
        super().__init__()

        self.spatial_channels = spatial_channels
        self.spatial_size = spatial_size
        self.vector_dim = vector_dim
        self.action_space_size = action_space_size
        self.num_objectives = num_objectives
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        self.value_head_hidden = value_head_hidden
        self.policy_head_hidden = policy_head_hidden
        self.dropout_rate = dropout
        self.use_attention = use_attention

        # Build spatial encoder if we have spatial input
        self._has_spatial = spatial_channels > 0 and spatial_size > 0
        if self._has_spatial:
            self.spatial_encoder = self._SpatialEncoder(
                in_channels=spatial_channels,
                num_filters=num_filters,
                num_res_blocks=num_res_blocks,
                use_attention=use_attention,
            )
            spatial_flat_dim = num_filters * spatial_size * spatial_size
        else:
            self.spatial_encoder = None  # type: ignore[assignment]
            spatial_flat_dim = 0

        # Vector encoder (always present)
        vector_hidden = max(vector_dim * 2, 128)
        vector_output_dim = max(vector_dim, 64)
        self.vector_encoder = self._VectorEncoder(
            input_dim=vector_dim,
            hidden_dim=vector_hidden,
            output_dim=vector_output_dim,
            dropout=dropout,
        )

        # Combined feature dimension
        combined_dim = spatial_flat_dim + vector_output_dim

        # Policy head
        self.policy_head = self._PolicyHead(
            input_dim=combined_dim,
            hidden_dim=policy_head_hidden,
            action_space_size=action_space_size,
            dropout=dropout,
        )

        # Multi-objective value head
        self.value_head = self._MultiValueHead(
            input_dim=combined_dim,
            hidden_dim=value_head_hidden,
            num_objectives=num_objectives,
            dropout=dropout,
        )

        logger.info(
            "StrategicNetwork created: spatial=%s (%dx%dx%d), "
            "vector=%d, actions=%d, objectives=%d, "
            "res_blocks=%d, filters=%d, attention=%s",
            self._has_spatial,
            spatial_channels,
            spatial_size,
            spatial_size,
            vector_dim,
            action_space_size,
            num_objectives,
            num_res_blocks,
            num_filters,
            use_attention,
        )

    def forward(
        self,
        spatial: torch.Tensor | None,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            spatial: Spatial input tensor (B, C, H, W), or None if no
                spatial features.
            vector: Vector input tensor (B, D).
            action_mask: Optional valid action mask (B, A). Values of 0
                set corresponding policy logits to -inf.

        Returns:
            Tuple of:
            - policy_logits: Raw logits over actions, shape (B, A).
              Masked actions have logit = -1e8.
            - value: Value estimate. Shape (B, num_objectives) when
              num_objectives > 1, or (B,) when num_objectives == 1
              for backward compatibility.
        """
        # Encode spatial features
        if self._has_spatial and spatial is not None and self.spatial_encoder is not None:
            spatial_features = self.spatial_encoder(spatial)
            # Flatten spatial: (B, C, H, W) -> (B, C*H*W)
            batch_size = spatial_features.size(0)
            spatial_flat = spatial_features.reshape(batch_size, -1)
        else:
            spatial_flat = None

        # Encode vector features
        vector_features = self.vector_encoder(vector)

        # Combine features
        if spatial_flat is not None:
            combined = torch.cat([spatial_flat, vector_features], dim=1)
        else:
            combined = vector_features

        # Policy head
        policy_logits = self.policy_head(combined)

        # Apply action mask
        if action_mask is not None:
            # Set invalid action logits to large negative value
            invalid_mask = (action_mask == 0)
            policy_logits = policy_logits.masked_fill(invalid_mask, -1e8)

        # Value head
        value = self.value_head(combined)

        # Squeeze for single-objective backward compatibility
        if self.num_objectives == 1:
            value = value.squeeze(-1)  # (B, 1) -> (B,)

        return policy_logits, value

    def predict(
        self,
        spatial: torch.Tensor | None,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict action probabilities and value (inference mode).

        Like forward(), but returns softmax probabilities instead of
        raw logits and runs under torch.no_grad().

        Args:
            spatial: Spatial input tensor (B, C, H, W) or None.
            vector: Vector input tensor (B, D).
            action_mask: Optional valid action mask (B, A).

        Returns:
            Tuple of:
            - policy_probs: Softmax probabilities, shape (B, A).
            - value: Value estimate, shape (B, num_objectives) or (B,).
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(spatial, vector, action_mask)
            policy_probs = F.softmax(policy_logits, dim=-1)
        return policy_probs, value

    def save(self, path: str) -> None:
        """Save model weights and configuration to disk.

        Args:
            path: File path for the saved checkpoint.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "spatial_channels": self.spatial_channels,
                "spatial_size": self.spatial_size,
                "vector_dim": self.vector_dim,
                "action_space_size": self.action_space_size,
                "num_objectives": self.num_objectives,
                "num_res_blocks": self.num_res_blocks,
                "num_filters": self.num_filters,
                "value_head_hidden": self.value_head_hidden,
                "policy_head_hidden": self.policy_head_hidden,
                "dropout": self.dropout_rate,
                "use_attention": self.use_attention,
            },
        }
        torch.save(checkpoint, save_path)
        logger.info("Model saved to %s", save_path)

    def load(self, path: str, device: str = "cpu") -> None:
        """Load model weights from a checkpoint.

        Args:
            path: File path to the saved checkpoint.
            device: Device to load weights onto.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            # Try loading as raw state dict
            self.load_state_dict(checkpoint)
        else:
            raise ValueError(
                f"Unexpected checkpoint format from {path}"
            )

        self.to(device)
        logger.info("Model loaded from %s to %s", path, device)

    # ================================================================
    # Inner module classes
    # ================================================================

    class _ResidualBlock(nn.Module):
        """Residual block with two convolutional layers and batch norm.

        Architecture: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
        """

        def __init__(self, num_filters: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(
                num_filters, num_filters, 3, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(num_filters)
            self.conv2 = nn.Conv2d(
                num_filters, num_filters, 3, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(num_filters)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with residual connection.

            Args:
                x: Input tensor (B, C, H, W).

            Returns:
                Output tensor (B, C, H, W).
            """
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = out + residual
            return F.relu(out)

    class _SpatialEncoder(nn.Module):
        """Encodes spatial (grid/map) features using convolutions.

        Architecture:
            Conv2d (in -> filters) -> BN -> ReLU
            -> ResidualBlock x num_res_blocks
            -> (optional) SpatialAttention
        """

        def __init__(
            self,
            in_channels: int,
            num_filters: int,
            num_res_blocks: int,
            use_attention: bool = False,
        ) -> None:
            super().__init__()
            self.conv_in = nn.Conv2d(
                in_channels, num_filters, 3, padding=1, bias=False
            )
            self.bn_in = nn.BatchNorm2d(num_filters)

            self.res_blocks = nn.ModuleList(
                [
                    StrategicNetwork._ResidualBlock(num_filters)
                    for _ in range(num_res_blocks)
                ]
            )

            self.use_attention = use_attention
            if use_attention:
                # Simple channel-wise self-attention
                self.attention = nn.MultiheadAttention(
                    embed_dim=num_filters,
                    num_heads=max(1, num_filters // 16),
                    batch_first=True,
                )
                self.attention_norm = nn.LayerNorm(num_filters)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through spatial encoder.

            Args:
                x: Spatial input (B, C, H, W).

            Returns:
                Encoded features (B, num_filters, H, W).
            """
            out = F.relu(self.bn_in(self.conv_in(x)))

            for block in self.res_blocks:
                out = block(out)

            if self.use_attention:
                b, c, h, w = out.shape
                # Reshape to sequence: (B, H*W, C)
                seq = out.permute(0, 2, 3, 1).reshape(b, h * w, c)
                # Self-attention
                attn_out, _ = self.attention(seq, seq, seq)
                # Residual + norm
                seq = self.attention_norm(seq + attn_out)
                # Reshape back: (B, C, H, W)
                out = seq.reshape(b, h, w, c).permute(0, 3, 1, 2)

            return out

    class _VectorEncoder(nn.Module):
        """Encodes non-spatial (vector) features using an MLP.

        Architecture: Linear -> ReLU -> Dropout -> Linear
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = (
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Vector input (B, input_dim).

            Returns:
                Encoded features (B, output_dim).
            """
            out = F.relu(self.fc1(x))
            out = self.dropout(out)
            out = self.fc2(out)
            return out

    class _PolicyHead(nn.Module):
        """Policy head: maps combined features to action logits.

        Architecture: Linear -> ReLU -> Dropout -> Linear
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            action_space_size: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, action_space_size)
            self.dropout = (
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Combined features (B, input_dim).

            Returns:
                Action logits (B, action_space_size).
            """
            out = F.relu(self.fc1(x))
            out = self.dropout(out)
            out = self.fc2(out)
            return out

    class _MultiValueHead(nn.Module):
        """Multi-objective value head.

        Outputs one value per objective, each passed through tanh
        to bound in [-1, 1].

        Architecture: Linear -> ReLU -> Dropout -> Linear -> Tanh
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_objectives: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_objectives)
            self.dropout = (
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Combined features (B, input_dim).

            Returns:
                Value estimates (B, num_objectives), each in [-1, 1].
            """
            out = F.relu(self.fc1(x))
            out = self.dropout(out)
            out = torch.tanh(self.fc2(out))
            return out


__all__ = [
    "StrategicNetwork",
]
