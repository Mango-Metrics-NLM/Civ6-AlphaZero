"""Neural network architecture for Civ6 AlphaZero.

Implements a ResNet-style architecture with:
- Spatial encoder (convolutional) for map features
- Vector encoder (MLP) for global features
- Policy head with action masking
- Value head for position evaluation
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NetworkConfig:
    """Configuration for Civ6 neural network."""

    # Input dimensions
    spatial_channels: int = 13  # MicroCiv default
    spatial_size: int = 16
    vector_dim: int = 64

    # Network architecture
    num_res_blocks: int = 6
    num_filters: int = 64
    fc_hidden_dim: int = 256

    # Output dimensions
    action_space_size: int = 381  # MicroCiv default
    value_head_hidden: int = 128

    # Regularization
    dropout: float = 0.0


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""

    def __init__(self, num_filters: int):
        """Initialize residual block.

        Args:
            num_filters: Number of convolutional filters.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class SpatialEncoder(nn.Module):
    """Encodes spatial (map) features using convolutional layers."""

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        num_res_blocks: int,
    ):
        """Initialize spatial encoder.

        Args:
            in_channels: Number of input channels.
            num_filters: Number of convolutional filters.
            num_res_blocks: Number of residual blocks.
        """
        super().__init__()

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, num_filters, H, W).
        """
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)
        return out


class VectorEncoder(nn.Module):
    """Encodes non-spatial (vector) features using MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        """Initialize vector encoder.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, input_dim).

        Returns:
            Output tensor of shape (B, output_dim).
        """
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class PolicyHead(nn.Module):
    """Policy head for action prediction."""

    def __init__(
        self,
        spatial_filters: int,
        spatial_size: int,
        vector_dim: int,
        action_space_size: int,
        hidden_dim: int = 256,
    ):
        """Initialize policy head.

        Args:
            spatial_filters: Number of spatial feature filters.
            spatial_size: Spatial feature map size (H=W).
            vector_dim: Vector feature dimension.
            action_space_size: Total number of actions.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        # Spatial reduction
        self.conv = nn.Conv2d(spatial_filters, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(2)

        # Combined features
        flat_spatial_dim = 2 * spatial_size * spatial_size
        combined_dim = flat_spatial_dim + vector_dim

        # Output layers
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_space_size)

    def forward(
        self,
        spatial_features: torch.Tensor,
        vector_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            spatial_features: Spatial features (B, C, H, W).
            vector_features: Vector features (B, V).

        Returns:
            Policy logits of shape (B, action_space_size).
        """
        # Reduce spatial features
        s = F.relu(self.bn(self.conv(spatial_features)))
        s = s.view(s.size(0), -1)

        # Combine with vector features
        combined = torch.cat([s, vector_features], dim=1)

        # Output
        out = F.relu(self.fc1(combined))
        out = self.fc2(out)
        return out


class ValueHead(nn.Module):
    """Value head for position evaluation."""

    def __init__(
        self,
        spatial_filters: int,
        spatial_size: int,
        vector_dim: int,
        hidden_dim: int = 128,
    ):
        """Initialize value head.

        Args:
            spatial_filters: Number of spatial feature filters.
            spatial_size: Spatial feature map size.
            vector_dim: Vector feature dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()

        # Spatial reduction
        self.conv = nn.Conv2d(spatial_filters, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)

        # Combined features
        flat_spatial_dim = spatial_size * spatial_size
        combined_dim = flat_spatial_dim + vector_dim

        # Output layers
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        spatial_features: torch.Tensor,
        vector_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            spatial_features: Spatial features (B, C, H, W).
            vector_features: Vector features (B, V).

        Returns:
            Value estimate of shape (B, 1).
        """
        # Reduce spatial features
        s = F.relu(self.bn(self.conv(spatial_features)))
        s = s.view(s.size(0), -1)

        # Combine with vector features
        combined = torch.cat([s, vector_features], dim=1)

        # Output
        out = F.relu(self.fc1(combined))
        out = torch.tanh(self.fc2(out))
        return out


class Civ6Network(nn.Module):
    """Full AlphaZero network for Civilization VI.

    Architecture:
    1. Spatial encoder (ResNet) processes map features
    2. Vector encoder (MLP) processes global features
    3. Features are combined and fed to policy and value heads
    """

    def __init__(self, config: NetworkConfig | None = None, **kwargs):
        """Initialize network.

        Args:
            config: Network configuration.
            **kwargs: Override config values.
        """
        super().__init__()

        # Handle config
        if config is None:
            config = NetworkConfig()

        # Allow kwargs to override config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(
            in_channels=config.spatial_channels,
            num_filters=config.num_filters,
            num_res_blocks=config.num_res_blocks,
        )

        # Vector encoder
        self.vector_encoder = VectorEncoder(
            input_dim=config.vector_dim,
            hidden_dim=config.fc_hidden_dim,
            output_dim=config.fc_hidden_dim,
            dropout=config.dropout,
        )

        # Policy head
        self.policy_head = PolicyHead(
            spatial_filters=config.num_filters,
            spatial_size=config.spatial_size,
            vector_dim=config.fc_hidden_dim,
            action_space_size=config.action_space_size,
            hidden_dim=config.fc_hidden_dim,
        )

        # Value head
        self.value_head = ValueHead(
            spatial_filters=config.num_filters,
            spatial_size=config.spatial_size,
            vector_dim=config.fc_hidden_dim,
            hidden_dim=config.value_head_hidden,
        )

    def forward(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            spatial: Spatial input (B, C, H, W).
            vector: Vector input (B, V).
            action_mask: Valid action mask (B, A). If None, all actions valid.

        Returns:
            Tuple of (policy_logits, value).
        """
        # Encode features
        spatial_features = self.spatial_encoder(spatial)
        vector_features = self.vector_encoder(vector)

        # Get policy logits and value
        policy_logits = self.policy_head(spatial_features, vector_features)
        value = self.value_head(spatial_features, vector_features)

        # Apply action mask
        if action_mask is not None:
            # Set invalid actions to very negative value
            policy_logits = policy_logits.masked_fill(action_mask == 0, -1e9)

        return policy_logits, value.squeeze(-1)

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy probabilities and value.

        Args:
            spatial: Spatial input (B, C, H, W).
            vector: Vector input (B, V).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (policy_probs, value).
        """
        policy_logits, value = self.forward(spatial, vector, action_mask)
        policy_probs = F.softmax(policy_logits, dim=-1)
        return policy_probs, value

    @torch.no_grad()
    def predict_single(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Predict for a single state (no batch dimension).

        Args:
            spatial: Spatial input (C, H, W).
            vector: Vector input (V,).
            action_mask: Valid action mask (A,).

        Returns:
            Tuple of (policy_probs, value).
        """
        self.eval()

        # Add batch dimension
        spatial = spatial.unsqueeze(0)
        vector = vector.unsqueeze(0)
        if action_mask is not None:
            action_mask = action_mask.unsqueeze(0)

        policy_probs, value = self.predict(spatial, vector, action_mask)

        return policy_probs.squeeze(0), value.item()

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> Civ6Network:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint.
            device: Device to load model to.

        Returns:
            Loaded model.
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(config=checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model


class MicroCivNetwork(Civ6Network):
    """Simplified network for MicroCiv environment.

    Uses smaller architecture for faster training.
    """

    def __init__(
        self,
        board_size: int = 16,
        action_space_size: int = 381,
        num_res_blocks: int = 6,
        num_filters: int = 64,
    ):
        """Initialize MicroCiv network.

        Args:
            board_size: Size of game board.
            action_space_size: Total number of actions.
            num_res_blocks: Number of residual blocks.
            num_filters: Number of convolutional filters.
        """
        config = NetworkConfig(
            spatial_channels=13,  # MicroCivEncoder.TOTAL_CHANNELS
            spatial_size=board_size,
            vector_dim=64,
            num_res_blocks=num_res_blocks,
            num_filters=num_filters,
            fc_hidden_dim=128,
            action_space_size=action_space_size,
            value_head_hidden=64,
        )
        super().__init__(config=config)


def compute_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute AlphaZero loss.

    Args:
        policy_logits: Predicted policy logits (B, A).
        value: Predicted value (B,).
        target_policy: Target policy distribution (B, A).
        target_value: Target value (B,).
        action_mask: Valid action mask (B, A).

    Returns:
        Tuple of (total_loss, policy_loss, value_loss).
    """
    # Value loss (MSE)
    value_loss = F.mse_loss(value, target_value)

    # Policy loss (cross-entropy)
    if action_mask is not None:
        # Mask invalid actions in target
        target_policy = target_policy * action_mask
        # Renormalize
        target_policy = target_policy / (target_policy.sum(dim=-1, keepdim=True) + 1e-8)

    # Use log_softmax for numerical stability
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -torch.sum(target_policy * log_probs, dim=-1).mean()

    # Total loss
    total_loss = policy_loss + value_loss

    return total_loss, policy_loss, value_loss
