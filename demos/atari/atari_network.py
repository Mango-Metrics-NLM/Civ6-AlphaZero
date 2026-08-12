"""Neural network for Atari AlphaZero demo.

Reuses ``ResidualBlock`` from the core project and satisfies the
``MCTSNetwork`` protocol so it can be used with ``Civ6MCTS`` directly.

Architecture::

    Pixel frames  (C, 84, 84)      Auxiliary vector  (V,)
         │                                │
         ▼                                ▼
    Conv2d stem (C → F)           Linear encoder (V → H)
         │                                │
    ResBlock x N  (F filters)             │
         │                                │
         ├────────── concat ──────────────┘
         │
    ┌────┴────┐
    │         │
    Policy   Value
    Head     Head
    │         │
    ▼         ▼
  logits   scalar ∈ [-1, 1]

Example:
    >>> from demos.atari.atari_network import AtariNetwork, AtariNetworkConfig
    >>> net = AtariNetwork(AtariNetworkConfig(), num_actions=6, spatial_size=(84, 84), spatial_channels=4)
    >>> import torch
    >>> spatial = torch.randn(1, 4, 84, 84)
    >>> vector = torch.randn(1, 32)
    >>> logits, value = net(spatial, vector)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from alpha_zero.core.civ6_network import ResidualBlock
from demos.atari.config import AtariNetworkConfig

logger = logging.getLogger(__name__)


class _VectorEncoder(nn.Module):
    """Small MLP that encodes auxiliary vector features."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        return self.fc2(out)  # type: ignore[no-any-return]


class _PolicyHead(nn.Module):
    """Policy head: reduces spatial features, concatenates with vector, outputs logits."""

    def __init__(
        self,
        spatial_filters: int,
        spatial_h: int,
        spatial_w: int,
        vector_dim: int,
        num_actions: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(spatial_filters, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        flat_dim = 2 * spatial_h * spatial_w + vector_dim
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, spatial: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        s = F.relu(self.bn(self.conv(spatial)))
        s = s.view(s.size(0), -1)
        combined = torch.cat([s, vector], dim=1)
        return self.fc2(F.relu(self.fc1(combined)))  # type: ignore[no-any-return]


class _ValueHead(nn.Module):
    """Value head: reduces spatial features, concatenates with vector, outputs scalar."""

    def __init__(
        self,
        spatial_filters: int,
        spatial_h: int,
        spatial_w: int,
        vector_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(spatial_filters, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        flat_dim = spatial_h * spatial_w + vector_dim
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, spatial: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        s = F.relu(self.bn(self.conv(spatial)))
        s = s.view(s.size(0), -1)
        combined = torch.cat([s, vector], dim=1)
        return torch.tanh(self.fc2(F.relu(self.fc1(combined))))


class AtariNetwork(nn.Module):
    """AlphaZero-style network for Atari pixel observations.

    Satisfies the ``MCTSNetwork`` protocol::

        predict(spatial, vector, action_mask) → (policy_probs, value)

    Args:
        config: Network hyper-parameters.
        num_actions: Size of the discrete action space.
        spatial_size: (height, width) of preprocessed frames.
        spatial_channels: Number of input channels (frame_stack * channels_per_frame).
        vector_dim: Dimension of the auxiliary vector input.
    """

    def __init__(
        self,
        config: AtariNetworkConfig,
        num_actions: int,
        spatial_size: tuple[int, int] = (84, 84),
        spatial_channels: int = 4,
        vector_dim: int = 32,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        self.spatial_h, self.spatial_w = spatial_size

        # Convolutional stem -- reduces 84x84 to a manageable feature-map size
        # before the residual tower.  Stride-2 convolutions halve dimensions
        # twice: 84 → 42 → 21.
        self.stem = nn.Sequential(
            nn.Conv2d(spatial_channels, config.num_filters, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.num_filters, config.num_filters, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(config.num_filters),
            nn.ReLU(inplace=True),
        )

        # After stem the spatial dims are ceil(H/4) x ceil(W/4)
        self._feat_h = (self.spatial_h + 3) // 4
        self._feat_w = (self.spatial_w + 3) // 4

        # Residual tower (reuses core ResidualBlock)
        self.res_tower = nn.Sequential(
            *[ResidualBlock(config.num_filters) for _ in range(config.num_res_blocks)]
        )

        # Vector encoder
        self.vector_encoder = _VectorEncoder(
            vector_dim,
            config.fc_hidden_dim,
            dropout=config.dropout,
        )

        # Heads
        self.policy_head = _PolicyHead(
            spatial_filters=config.num_filters,
            spatial_h=self._feat_h,
            spatial_w=self._feat_w,
            vector_dim=config.fc_hidden_dim,
            num_actions=num_actions,
            hidden_dim=config.fc_hidden_dim,
        )
        self.value_head = _ValueHead(
            spatial_filters=config.num_filters,
            spatial_h=self._feat_h,
            spatial_w=self._feat_w,
            vector_dim=config.fc_hidden_dim,
            hidden_dim=config.value_head_hidden,
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "AtariNetwork created: actions=%d, spatial=%dx%d, filters=%d, "
            "res_blocks=%d, params=%s",
            num_actions,
            self.spatial_h,
            self.spatial_w,
            config.num_filters,
            config.num_res_blocks,
            f"{total_params:,}",
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass returning (policy_logits, value).

        Args:
            spatial: Pixel input (B, C, H, W).
            vector: Auxiliary features (B, V).
            action_mask: Valid action mask (B, A).  ``None`` ⇒ all valid.

        Returns:
            ``(policy_logits, value)`` where value has shape ``(B,)``.
        """
        # Spatial path
        feat = self.stem(spatial)
        feat = self.res_tower(feat)

        # Vector path
        vec_feat = self.vector_encoder(vector)

        # Heads
        policy_logits = self.policy_head(feat, vec_feat)
        value = self.value_head(feat, vec_feat)

        # Mask invalid actions (usually a no-op for Atari)
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, -1e9)

        return policy_logits, value.squeeze(-1)

    # ------------------------------------------------------------------
    # MCTSNetwork protocol
    # ------------------------------------------------------------------

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_probs, value) — satisfies ``MCTSNetwork`` protocol.

        Args:
            spatial: (B, C, H, W).
            vector: (B, V).
            action_mask: (B, A) or ``None``.

        Returns:
            ``(policy_probs, value)``.
        """
        logits, value = self.forward(spatial, vector, action_mask)
        return F.softmax(logits, dim=-1), value

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: File path for the checkpoint.
        """
        torch.save(
            {
                "config": {
                    "num_res_blocks": self.config.num_res_blocks,
                    "num_filters": self.config.num_filters,
                    "fc_hidden_dim": self.config.fc_hidden_dim,
                    "value_head_hidden": self.config.value_head_hidden,
                    "dropout": self.config.dropout,
                },
                "num_actions": self.num_actions,
                "spatial_size": (self.spatial_h, self.spatial_w),
                "state_dict": self.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)

    @classmethod
    def load(
        cls,
        path: str,
        spatial_channels: int = 4,
        vector_dim: int = 32,
        device: str = "cpu",
    ) -> AtariNetwork:
        """Load a saved checkpoint.

        Args:
            path: Checkpoint file path.
            spatial_channels: Must match the value used during training.
            vector_dim: Must match the value used during training.
            device: Target device.

        Returns:
            Restored ``AtariNetwork``.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config = AtariNetworkConfig(**checkpoint["config"])
        model = cls(
            config=config,
            num_actions=checkpoint["num_actions"],
            spatial_size=tuple(checkpoint["spatial_size"]),
            spatial_channels=spatial_channels,
            vector_dim=vector_dim,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        logger.info("Checkpoint loaded from %s (device=%s)", path, device)
        return model
