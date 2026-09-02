"""N-player neural network value head.

Extends the standard single-scalar value head to output per-player
value predictions. The architecture uses a shared feature encoder
with player-specific output heads.

Backward compatible: NPlayerValueHead with num_players=2 produces
output compatible with the standard ValueHead interface.

Example:
    >>> head = NPlayerValueHead(spatial_filters=64, spatial_size=16,
    ...                         vector_dim=256, num_players=4)
    >>> values = head(spatial_features, vector_features)
    >>> values.shape
    torch.Size([B, 4])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from alpha_zero.utils.constants import DEFAULT_ACTION_SPACE_SIZE

logger = logging.getLogger(__name__)


@dataclass
class NPlayerNetworkConfig:
    """Configuration for N-player network components.

    Attributes:
        num_players: Number of players for value head output.
        spatial_channels: Input spatial channels.
        spatial_size: Board size (height/width).
        vector_dim: Input vector feature dimension.
        num_res_blocks: Number of residual blocks.
        num_filters: Convolutional filter count.
        fc_hidden_dim: Hidden layer dimension for FC layers.
        value_head_hidden: Hidden dimension in value head.
        action_space_size: Total action space size.
        shared_value_trunk: If True, share a trunk before per-player heads.
    """

    num_players: int = 2
    spatial_channels: int = 13
    spatial_size: int = 16
    vector_dim: int = 64
    num_res_blocks: int = 6
    num_filters: int = 64
    fc_hidden_dim: int = 256
    value_head_hidden: int = 128
    action_space_size: int = DEFAULT_ACTION_SPACE_SIZE
    shared_value_trunk: bool = True


class NPlayerValueHead(nn.Module):
    """Value head that outputs per-player value predictions.

    Architecture:
        spatial_features (B, C, H, W)
            -> Conv2d(C, 1) + BN + ReLU -> flatten
        concat with vector_features (B, V)
            -> FC(combined, hidden) + ReLU
            -> FC(hidden, num_players)
            -> tanh -> (B, num_players)

    For num_players=1, this degenerates to the standard scalar value head.
    """

    def __init__(
        self,
        spatial_filters: int,
        spatial_size: int,
        vector_dim: int,
        num_players: int = 2,
        hidden_dim: int = 128,
    ) -> None:
        """Initialize N-player value head.

        Args:
            spatial_filters: Number of spatial feature channels.
            spatial_size: Spatial feature map size (H=W).
            vector_dim: Vector feature dimension.
            num_players: Number of player-specific value outputs.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.num_players = num_players

        # Spatial reduction
        self.conv = nn.Conv2d(spatial_filters, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)

        # Combined features
        flat_spatial_dim = spatial_size * spatial_size
        combined_dim = flat_spatial_dim + vector_dim

        # Shared trunk
        self.fc1 = nn.Linear(combined_dim, hidden_dim)

        # Per-player output
        self.fc2 = nn.Linear(hidden_dim, num_players)

        logger.debug(
            "NPlayerValueHead initialized",
            extra={
                "num_players": num_players,
                "spatial_filters": spatial_filters,
                "spatial_size": spatial_size,
                "vector_dim": vector_dim,
                "hidden_dim": hidden_dim,
            },
        )

    def forward(
        self,
        spatial_features: torch.Tensor,
        vector_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass producing per-player values.

        Args:
            spatial_features: Spatial features (B, C, H, W).
            vector_features: Vector features (B, V).

        Returns:
            Per-player values of shape (B, num_players), each in [-1, 1].
        """
        s = F.relu(self.bn(self.conv(spatial_features)))
        s = s.view(s.size(0), -1)
        combined = torch.cat([s, vector_features], dim=1)
        out = F.relu(self.fc1(combined))
        out = torch.tanh(self.fc2(out))
        return out


class NPlayerNetworkAdapter(nn.Module):
    """Wraps an existing Civ6Network to add N-player value output.

    This adapter reuses the spatial and vector encoders from an
    existing trained network, replacing only the value head with
    an N-player variant. Useful for transferring 2-player weights.

    Example:
        >>> base_network = Civ6Network(config)
        >>> adapter = NPlayerNetworkAdapter(base_network, num_players=4)
        >>> policy, values = adapter(spatial, vector)
        >>> values.shape  # (B, 4) instead of (B,)
    """

    def __init__(
        self,
        base_network: nn.Module,
        num_players: int = 2,
        value_hidden_dim: int = 128,
        freeze_encoder: bool = False,
    ) -> None:
        """Initialize adapter wrapping a base network.

        Args:
            base_network: Existing Civ6Network or similar.
            num_players: Number of players for value head.
            value_hidden_dim: Hidden dimension for new value head.
            freeze_encoder: If True, freeze encoder weights (transfer learning).
        """
        super().__init__()
        self._num_players = num_players
        self.base_network = base_network

        # Extract config from base network
        if hasattr(base_network, "config"):
            # nn.Module.__getattr__ types dynamic attributes as Tensor | Module;
            # ``config`` is actually a plain config object with int fields.
            config = cast(Any, base_network.config)
            spatial_filters = int(config.num_filters)
            spatial_size = int(config.spatial_size)
            vector_dim = int(getattr(config, "fc_hidden_dim", 256))
        else:
            # Fallback defaults — base_network has no config attribute
            spatial_filters = 64
            spatial_size = 16
            vector_dim = 256
            logger.warning(
                "Base network has no config attribute; using fallback defaults "
                "(spatial_filters=%d, spatial_size=%d, vector_dim=%d)",
                spatial_filters,
                spatial_size,
                vector_dim,
            )

        # Create new N-player value head
        self.nplayer_value_head = NPlayerValueHead(
            spatial_filters=spatial_filters,
            spatial_size=spatial_size,
            vector_dim=vector_dim,
            num_players=num_players,
            hidden_dim=value_hidden_dim,
        )

        # Optionally freeze encoder weights
        if freeze_encoder:
            self._freeze_encoders()

        logger.info(
            "NPlayerNetworkAdapter created",
            extra={
                "num_players": num_players,
                "freeze_encoder": freeze_encoder,
            },
        )

    @property
    def num_players(self) -> int:
        """Number of players this network supports."""
        return self._num_players

    def _freeze_encoders(self) -> None:
        """Freeze spatial and vector encoder parameters."""
        for name, param in self.base_network.named_parameters():
            if "value_head" not in name:
                param.requires_grad = False

    def forward(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with N-player value output.

        Args:
            spatial: Spatial input (B, C, H, W).
            vector: Vector input (B, V).
            action_mask: Optional action mask (B, A).

        Returns:
            Tuple of (policy_logits, player_values).
            policy_logits shape: (B, A)
            player_values shape: (B, num_players)
        """
        # Get features from base network encoder
        if hasattr(self.base_network, "spatial_encoder") and hasattr(
            self.base_network, "vector_encoder"
        ):
            # nn.Module.__getattr__ types submodules as Tensor | Module; these
            # attributes are callable encoder modules on the base network.
            spatial_features = cast(nn.Module, self.base_network.spatial_encoder)(spatial)
            vector_features = cast(nn.Module, self.base_network.vector_encoder)(vector)
        else:
            # Fallback: run full forward and extract features
            policy_logits, _ = self.base_network(spatial, vector, action_mask)
            # Can't get intermediate features, use mean-pooled spatial as proxy
            spatial_proxy = spatial.mean(dim=1, keepdim=True).expand(
                -1, self.nplayer_value_head.conv.in_channels, -1, -1
            )
            value = self.nplayer_value_head(spatial_proxy, vector)
            return policy_logits, value

        # Policy from base network's policy head (submodule typed Tensor | Module)
        policy_logits = cast(nn.Module, self.base_network.policy_head)(
            spatial_features, vector_features
        )

        # Action masking
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, -1e9)

        # N-player value from new head
        player_values = self.nplayer_value_head(spatial_features, vector_features)

        return policy_logits, player_values

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy probabilities and per-player values.

        Args:
            spatial: Spatial features (B, C, H, W).
            vector: Vector features (B, D).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (policy_probs, player_values).
        """
        self.eval()
        with torch.no_grad():
            policy_logits, player_values = self.forward(spatial, vector, action_mask)
            policy_probs = F.softmax(policy_logits, dim=-1)
        return policy_probs, player_values


def compute_nplayer_value_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    num_players: int,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute value loss for N-player predictions.

    Args:
        predicted: Predicted values (B, num_players) or (B,) for legacy.
        target: Target values (B, num_players) or (B,) for legacy.
        num_players: Number of players.
        reduction: Loss reduction method ('mean', 'sum', 'none').

    Returns:
        Value loss tensor.
    """
    # Handle legacy single-value case
    if predicted.dim() == 1 and target.dim() == 1:
        return F.mse_loss(predicted, target, reduction=reduction)

    # Ensure correct shapes
    if predicted.dim() == 1:
        predicted = predicted.unsqueeze(-1).expand(-1, num_players)
    if target.dim() == 1:
        target = target.unsqueeze(-1).expand(-1, num_players)

    return F.mse_loss(predicted, target, reduction=reduction)
