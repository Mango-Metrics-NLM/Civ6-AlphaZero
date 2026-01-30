"""
Hierarchical Neural Network for Goal-Conditioned Decisions.

This module implements neural networks that can condition their policy and value
outputs on hierarchical goals, enabling multi-level planning and decision making.

Example:
    >>> from alpha_zero.hierarchical.network import HierarchicalNetwork, HierarchicalNetworkConfig
    >>> config = HierarchicalNetworkConfig()
    >>> network = HierarchicalNetwork(config)
    >>> policy, value = network(state_dict, goal_embeddings)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class nn:
        class Module:
            pass

from alpha_zero.hierarchical.config import HierarchyLevel

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalNetworkConfig:
    """Configuration for hierarchical neural network.

    Attributes:
        board_size: Game board dimensions.
        spatial_channels: Number of spatial input channels.
        vector_dim: Dimension of vector input features.
        action_space_size: Total action space size.
        embed_dim: Internal embedding dimension.
        num_res_blocks: Number of residual blocks.
        goal_embed_dim: Dimension of goal embeddings.
        num_attention_heads: Number of multi-head attention heads.
        dropout: Dropout probability.
        use_goal_attention: Whether to use attention for goal conditioning.
    """

    board_size: tuple[int, int] = (16, 16)
    spatial_channels: int = 13
    vector_dim: int = 64
    action_space_size: int = 356
    embed_dim: int = 128
    num_res_blocks: int = 6
    goal_embed_dim: int = 64
    num_attention_heads: int = 4
    dropout: float = 0.1
    use_goal_attention: bool = True

    @classmethod
    def for_level(cls, level: HierarchyLevel) -> HierarchicalNetworkConfig:
        """Create level-specific network configuration.

        Args:
            level: Hierarchy level.

        Returns:
            Configuration appropriate for the level.
        """
        configs = {
            HierarchyLevel.STRATEGIC: cls(
                embed_dim=256,
                num_res_blocks=8,
                goal_embed_dim=128,
                num_attention_heads=8,
                action_space_size=100,  # Strategic action space is smaller
            ),
            HierarchyLevel.TACTICAL: cls(
                embed_dim=128,
                num_res_blocks=6,
                goal_embed_dim=64,
                num_attention_heads=4,
                action_space_size=500,  # Tactical actions
            ),
            HierarchyLevel.OPERATIONAL: cls(
                embed_dim=64,
                num_res_blocks=4,
                goal_embed_dim=32,
                num_attention_heads=2,
                action_space_size=356,  # Full action space
            ),
        }
        return configs.get(level, cls())


if TORCH_AVAILABLE:
    class ResidualBlock(nn.Module):
        """Residual block with batch normalization."""

        def __init__(self, channels: int, dropout: float = 0.1) -> None:
            """Initialize residual block.

            Args:
                channels: Number of channels.
                dropout: Dropout probability.
            """
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with residual connection."""
            identity = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            return F.relu(out + identity)

    class GoalConditionedHead(nn.Module):
        """Goal-conditioned policy/value head.

        This module takes encoded state features and goal embeddings
        and produces goal-conditioned outputs.
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            goal_dim: int,
            hidden_dim: int = 256,
            use_attention: bool = True,
            num_heads: int = 4,
        ) -> None:
            """Initialize goal-conditioned head.

            Args:
                input_dim: Input feature dimension.
                output_dim: Output dimension.
                goal_dim: Goal embedding dimension.
                hidden_dim: Hidden layer dimension.
                use_attention: Whether to use attention for goal conditioning.
                num_heads: Number of attention heads.
            """
            super().__init__()
            self.use_attention = use_attention

            if use_attention:
                # Multi-head attention for goal conditioning
                self.goal_attention = nn.MultiheadAttention(
                    embed_dim=input_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                self.goal_proj = nn.Linear(goal_dim, input_dim)
                combined_dim = input_dim * 2
            else:
                # Simple concatenation
                combined_dim = input_dim + goal_dim

            # Output layers
            self.fc1 = nn.Linear(combined_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.layer_norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            state_features: torch.Tensor,
            goal_embedding: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass with goal conditioning.

            Args:
                state_features: State features (batch, input_dim).
                goal_embedding: Goal embedding (batch, goal_dim).

            Returns:
                Conditioned output (batch, output_dim).
            """
            if self.use_attention:
                # Project goal to match state dimension
                goal_proj = self.goal_proj(goal_embedding)

                # Add sequence dimension for attention
                state_seq = state_features.unsqueeze(1)  # (B, 1, D)
                goal_seq = goal_proj.unsqueeze(1)  # (B, 1, D)

                # Cross-attention: state attends to goal
                attended, _ = self.goal_attention(
                    state_seq, goal_seq, goal_seq
                )
                attended = attended.squeeze(1)  # (B, D)

                # Concatenate original and attended features
                combined = torch.cat([state_features, attended], dim=-1)
            else:
                # Simple concatenation
                combined = torch.cat([state_features, goal_embedding], dim=-1)

            # Output projection
            out = F.relu(self.layer_norm(self.fc1(combined)))
            return self.fc2(out)

    class LevelNetwork(nn.Module):
        """Neural network for a specific hierarchy level.

        This network processes state and goal information to produce
        level-appropriate policy and value outputs.
        """

        def __init__(
            self,
            config: HierarchicalNetworkConfig,
            level: HierarchyLevel,
        ) -> None:
            """Initialize level network.

            Args:
                config: Network configuration.
                level: Hierarchy level this network operates at.
            """
            super().__init__()
            self.config = config
            self.level = level

            # Spatial encoder
            self.initial_conv = nn.Sequential(
                nn.Conv2d(config.spatial_channels, config.embed_dim, 3, padding=1),
                nn.BatchNorm2d(config.embed_dim),
                nn.ReLU(),
            )

            # Residual tower
            self.res_blocks = nn.ModuleList([
                ResidualBlock(config.embed_dim, config.dropout)
                for _ in range(config.num_res_blocks)
            ])

            # Spatial to vector
            spatial_out = config.board_size[0] * config.board_size[1] * config.embed_dim
            self.spatial_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(spatial_out // 4, config.embed_dim),  # Pool first
                nn.ReLU(),
            )

            # Pooling for spatial reduction
            self.pool = nn.AdaptiveAvgPool2d((config.board_size[0] // 2, config.board_size[1] // 2))

            # Vector encoder
            self.vector_encoder = nn.Sequential(
                nn.Linear(config.vector_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
            )

            # State combiner
            self.state_combiner = nn.Sequential(
                nn.Linear(config.embed_dim * 2, config.embed_dim),
                nn.LayerNorm(config.embed_dim),
                nn.ReLU(),
            )

            # Goal-conditioned policy head
            self.policy_head = GoalConditionedHead(
                input_dim=config.embed_dim,
                output_dim=config.action_space_size,
                goal_dim=config.goal_embed_dim,
                hidden_dim=config.embed_dim * 2,
                use_attention=config.use_goal_attention,
                num_heads=config.num_attention_heads,
            )

            # Goal-conditioned value head
            self.value_head = GoalConditionedHead(
                input_dim=config.embed_dim,
                output_dim=1,
                goal_dim=config.goal_embed_dim,
                hidden_dim=config.embed_dim,
                use_attention=config.use_goal_attention,
                num_heads=config.num_attention_heads,
            )

            logger.debug(
                "LevelNetwork initialized for %s with embed_dim=%d, action_space=%d",
                level.value,
                config.embed_dim,
                config.action_space_size,
            )

        def forward(
            self,
            spatial: torch.Tensor,
            vector: torch.Tensor,
            goal_embedding: torch.Tensor,
            action_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass for level network.

            Args:
                spatial: Spatial input (batch, channels, height, width).
                vector: Vector input (batch, vector_dim).
                goal_embedding: Goal embedding (batch, goal_dim).
                action_mask: Optional action mask (batch, action_space).

            Returns:
                Tuple of (policy_logits, value).
            """
            # Encode spatial features
            x = self.initial_conv(spatial)
            for block in self.res_blocks:
                x = block(x)

            # Pool and flatten
            x = self.pool(x)
            spatial_features = self.spatial_fc(x.flatten(1))

            # Encode vector features
            vector_features = self.vector_encoder(vector)

            # Combine state features
            state_features = self.state_combiner(
                torch.cat([spatial_features, vector_features], dim=-1)
            )

            # Goal-conditioned policy
            policy_logits = self.policy_head(state_features, goal_embedding)

            # Apply action mask
            if action_mask is not None:
                # Mask invalid actions with large negative value
                policy_logits = policy_logits.masked_fill(
                    action_mask == 0,
                    float("-inf"),
                )

            # Goal-conditioned value
            value = self.value_head(state_features, goal_embedding)
            value = torch.tanh(value)  # Bound to [-1, 1]

            return policy_logits, value.squeeze(-1)

        def get_state_embedding(
            self,
            spatial: torch.Tensor,
            vector: torch.Tensor,
        ) -> torch.Tensor:
            """Get state embedding without goal conditioning.

            Args:
                spatial: Spatial input.
                vector: Vector input.

            Returns:
                State embedding tensor.
            """
            x = self.initial_conv(spatial)
            for block in self.res_blocks:
                x = block(x)

            x = self.pool(x)
            spatial_features = self.spatial_fc(x.flatten(1))
            vector_features = self.vector_encoder(vector)

            return self.state_combiner(
                torch.cat([spatial_features, vector_features], dim=-1)
            )

    class HierarchicalNetwork(nn.Module):
        """Multi-level hierarchical neural network.

        This network maintains separate sub-networks for each hierarchy level,
        with shared components for efficiency and cross-level information flow.
        """

        def __init__(
            self,
            config: HierarchicalNetworkConfig | None = None,
            strategic_config: HierarchicalNetworkConfig | None = None,
            tactical_config: HierarchicalNetworkConfig | None = None,
            operational_config: HierarchicalNetworkConfig | None = None,
        ) -> None:
            """Initialize hierarchical network.

            Args:
                config: Base configuration (used if level-specific not provided).
                strategic_config: Configuration for strategic network.
                tactical_config: Configuration for tactical network.
                operational_config: Configuration for operational network.
            """
            super().__init__()

            # Use level-specific configs or derive from base/defaults
            config or HierarchicalNetworkConfig()

            self.strategic_network = LevelNetwork(
                strategic_config or HierarchicalNetworkConfig.for_level(HierarchyLevel.STRATEGIC),
                HierarchyLevel.STRATEGIC,
            )
            self.tactical_network = LevelNetwork(
                tactical_config or HierarchicalNetworkConfig.for_level(HierarchyLevel.TACTICAL),
                HierarchyLevel.TACTICAL,
            )
            self.operational_network = LevelNetwork(
                operational_config or HierarchicalNetworkConfig.for_level(HierarchyLevel.OPERATIONAL),
                HierarchyLevel.OPERATIONAL,
            )

            self._networks = {
                HierarchyLevel.STRATEGIC: self.strategic_network,
                HierarchyLevel.TACTICAL: self.tactical_network,
                HierarchyLevel.OPERATIONAL: self.operational_network,
            }

            # Cross-level information flow (top-down)
            # Strategic -> Tactical
            self.strategic_to_tactical = nn.Sequential(
                nn.Linear(256, 128),  # Strategic embed -> Tactical embed
                nn.ReLU(),
                nn.Linear(128, 64),  # To goal embedding dim
            )

            # Tactical -> Operational
            self.tactical_to_operational = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
            )

            logger.info(
                "HierarchicalNetwork initialized with %d level networks",
                len(self._networks),
            )

        def forward(
            self,
            spatial: torch.Tensor,
            vector: torch.Tensor,
            goal_embedding: torch.Tensor,
            level: HierarchyLevel,
            action_mask: torch.Tensor | None = None,
            higher_level_embedding: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass for specific hierarchy level.

            Args:
                spatial: Spatial input.
                vector: Vector input.
                goal_embedding: Goal embedding for this level.
                level: Hierarchy level to use.
                action_mask: Optional action mask.
                higher_level_embedding: Optional embedding from higher level.

            Returns:
                Tuple of (policy_logits, value).
            """
            network = self._networks.get(level)
            if network is None:
                raise ValueError(f"Unknown hierarchy level: {level}")

            # Augment goal embedding with higher level context
            if higher_level_embedding is not None:
                if level == HierarchyLevel.TACTICAL:
                    context = self.strategic_to_tactical(higher_level_embedding)
                    goal_embedding = goal_embedding + context
                elif level == HierarchyLevel.OPERATIONAL:
                    context = self.tactical_to_operational(higher_level_embedding)
                    goal_embedding = goal_embedding + context

            return network(spatial, vector, goal_embedding, action_mask)

        def forward_all_levels(
            self,
            spatial: torch.Tensor,
            vector: torch.Tensor,
            goal_embeddings: dict[HierarchyLevel, torch.Tensor],
            action_masks: dict[HierarchyLevel, torch.Tensor] | None = None,
        ) -> dict[HierarchyLevel, tuple[torch.Tensor, torch.Tensor]]:
            """Forward pass for all hierarchy levels with cascading context.

            Args:
                spatial: Spatial input.
                vector: Vector input.
                goal_embeddings: Goal embeddings for each level.
                action_masks: Optional action masks for each level.

            Returns:
                Dictionary mapping levels to (policy_logits, value) tuples.
            """
            action_masks = action_masks or {}
            results = {}
            higher_embed = None

            # Process top-down: strategic -> tactical -> operational
            for level in [HierarchyLevel.STRATEGIC, HierarchyLevel.TACTICAL, HierarchyLevel.OPERATIONAL]:
                goal_emb = goal_embeddings.get(level)
                if goal_emb is None:
                    continue

                mask = action_masks.get(level)
                policy, value = self.forward(
                    spatial, vector, goal_emb, level, mask, higher_embed
                )
                results[level] = (policy, value)

                # Get state embedding for next level
                network = self._networks[level]
                higher_embed = network.get_state_embedding(spatial, vector)

            return results

        def get_network(self, level: HierarchyLevel) -> LevelNetwork:
            """Get network for a specific level.

            Args:
                level: Hierarchy level.

            Returns:
                Level-specific network.
            """
            return self._networks[level]

        def save(self, path: str) -> None:
            """Save network weights.

            Args:
                path: Path to save file.
            """
            torch.save(self.state_dict(), path)
            logger.info("Saved hierarchical network to %s", path)

        def load(self, path: str, device: str | torch.device = "cpu") -> None:
            """Load network weights.

            Args:
                path: Path to saved weights.
                device: Device to load to.
            """
            state_dict = torch.load(path, map_location=device)
            self.load_state_dict(state_dict)
            logger.info("Loaded hierarchical network from %s", path)

else:
    # Stub classes when torch not available
    class ResidualBlock:
        """Placeholder when torch not available."""
        pass

    class GoalConditionedHead:
        """Placeholder when torch not available."""
        pass

    class LevelNetwork:
        """Placeholder when torch not available."""
        pass

    class HierarchicalNetwork:
        """Placeholder when torch not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch required for HierarchicalNetwork")
