"""MuZero neural network architecture for Civ6.

Implements the MuZero architecture with three core functions:
1. Representation function: h(o) -> s (observation to hidden state)
2. Dynamics function: g(s, a) -> (s', r) (world model)
3. Prediction function: f(s) -> (p, v) (policy and value from hidden state)

Includes EfficientZero extensions:
- Self-supervised consistency loss
- Value prefix prediction

Reference:
- MuZero: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
- EfficientZero: "Mastering Atari Games with Limited Data"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import logging
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_zero.core.civ6_network import (
    NetworkConfig,
    PolicyHead,
    ResidualBlock,
    SpatialEncoder,
    ValueHead,
    VectorEncoder,
)

# Import centralized loss functions (canonical implementations)
# These imports provide backward compatibility for code using this module
from alpha_zero.core.losses import (
    categorical_to_scalar,
    compute_consistency_loss,
    compute_muzero_loss,
    scalar_to_categorical,
)
from alpha_zero.utils.checkpoint import safe_load_checkpoint

# Module logger
logger = logging.getLogger(__name__)


@dataclass
class MuZeroConfig:
    """Configuration for MuZero network.

    Extends NetworkConfig with MuZero-specific parameters.
    """

    # Base network config
    spatial_channels: int = 13
    spatial_size: int = 16
    vector_dim: int = 64
    num_res_blocks: int = 6
    num_filters: int = 64
    fc_hidden_dim: int = 256
    action_space_size: int = 381
    value_head_hidden: int = 128
    dropout: float = 0.0

    # MuZero-specific
    hidden_state_channels: int = 64  # Channels in hidden state
    dynamics_num_blocks: int = 4  # ResBlocks in dynamics function
    action_embedding_dim: int = 32  # Dimension for action embedding

    # Reward prediction
    reward_support_size: int = 601  # For categorical reward (-300 to +300)
    use_categorical_reward: bool = True

    # EfficientZero extensions
    use_consistency_loss: bool = True
    projection_dim: int = 256  # Dimension for projection head
    prediction_hidden_dim: int = 128  # Hidden dim for prediction head

    # Value prefix prediction
    use_value_prefix: bool = True
    value_prefix_horizon: int = 5  # Predict sum of next K rewards

    # Training
    unroll_steps: int = 5  # Number of dynamics unroll steps


class MuZeroOutput(NamedTuple):
    """Output from MuZero initial inference."""

    hidden_state: torch.Tensor  # (B, C, H, W)
    policy_logits: torch.Tensor  # (B, A)
    value: torch.Tensor  # (B,)


class DynamicsOutput(NamedTuple):
    """Output from MuZero dynamics function."""

    next_hidden_state: torch.Tensor  # (B, C, H, W)
    reward: torch.Tensor  # (B,) or (B, reward_support_size)


class ActionEmbedding(nn.Module):
    """Embeds discrete actions for conditioning the dynamics network.

    Converts action indices to learned embeddings and broadcasts
    them spatially for concatenation with hidden states.
    """

    def __init__(
        self,
        action_space_size: int,
        embedding_dim: int,
        spatial_size: int,
    ):
        """Initialize action embedding.

        Args:
            action_space_size: Total number of actions.
            embedding_dim: Dimension of action embedding.
            spatial_size: Spatial size (H=W) for broadcasting.
        """
        super().__init__()
        self.embedding = nn.Embedding(action_space_size, embedding_dim)
        self.spatial_size = spatial_size

        # Initialize with small values
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Embed actions and broadcast spatially.

        Args:
            actions: Action indices of shape (B,).

        Returns:
            Spatially broadcast embeddings of shape (B, embedding_dim, H, W).
        """
        # Get embedding: (B,) -> (B, embedding_dim)
        embed = self.embedding(actions)

        # Broadcast spatially: (B, embedding_dim) -> (B, embedding_dim, H, W)
        embed = embed.unsqueeze(-1).unsqueeze(-1)
        embed = embed.expand(-1, -1, self.spatial_size, self.spatial_size)

        return embed


class DynamicsNetwork(nn.Module):
    """Dynamics function (world model) for MuZero.

    Predicts the next hidden state and reward given current hidden state
    and action. This is the learned world model that allows MCTS to
    search without environment simulation.

    Architecture:
    1. Embed action and concatenate with hidden state
    2. Process through residual blocks
    3. Output next hidden state and reward prediction
    """

    def __init__(
        self,
        hidden_channels: int,
        action_space_size: int,
        action_embedding_dim: int,
        spatial_size: int,
        num_blocks: int = 4,
        reward_support_size: int = 601,
        use_categorical_reward: bool = True,
    ):
        """Initialize dynamics network.

        Args:
            hidden_channels: Number of channels in hidden state.
            action_space_size: Total number of actions.
            action_embedding_dim: Dimension for action embedding.
            spatial_size: Spatial size (H=W) of hidden state.
            num_blocks: Number of residual blocks.
            reward_support_size: Size of categorical reward support.
            use_categorical_reward: If True, predict categorical distribution.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.use_categorical_reward = use_categorical_reward

        # Action embedding
        self.action_embedding = ActionEmbedding(
            action_space_size=action_space_size,
            embedding_dim=action_embedding_dim,
            spatial_size=spatial_size,
        )

        # Input projection (hidden_state + action_embedding)
        self.input_conv = nn.Conv2d(
            hidden_channels + action_embedding_dim,
            hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.input_bn = nn.BatchNorm2d(hidden_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels) for _ in range(num_blocks)]
        )

        # Reward prediction head
        self.reward_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.reward_bn = nn.BatchNorm2d(1)

        # Flatten and predict reward
        reward_flat_dim = spatial_size * spatial_size
        if use_categorical_reward:
            self.reward_fc = nn.Sequential(
                nn.Linear(reward_flat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, reward_support_size),
            )
        else:
            self.reward_fc = nn.Sequential(
                nn.Linear(reward_flat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

    def forward(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
    ) -> DynamicsOutput:
        """Forward pass of dynamics function.

        Args:
            hidden_state: Current hidden state (B, C, H, W).
            action: Action indices (B,).

        Returns:
            DynamicsOutput with next_hidden_state and reward.
        """
        # Embed action and concatenate
        action_embed = self.action_embedding(action)
        combined = torch.cat([hidden_state, action_embed], dim=1)

        # Process through network
        x = F.relu(self.input_bn(self.input_conv(combined)))
        for block in self.res_blocks:
            x = block(x)

        # Next hidden state (normalized)
        next_hidden = x

        # Reward prediction
        r = F.relu(self.reward_bn(self.reward_conv(x)))
        r = r.view(r.size(0), -1)
        reward = self.reward_fc(r)

        if not self.use_categorical_reward:
            reward = reward.squeeze(-1)

        return DynamicsOutput(next_hidden_state=next_hidden, reward=reward)


class RewardHead(nn.Module):
    """Standalone reward prediction head.

    Can be used for reward prediction from any hidden state representation.
    """

    def __init__(
        self,
        hidden_channels: int,
        spatial_size: int,
        support_size: int = 601,
        use_categorical: bool = True,
    ):
        """Initialize reward head.

        Args:
            hidden_channels: Number of input channels.
            spatial_size: Spatial size (H=W) of input.
            support_size: Size of categorical reward support.
            use_categorical: If True, output categorical distribution.
        """
        super().__init__()
        self.use_categorical = use_categorical

        self.conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

        flat_dim = spatial_size * spatial_size
        output_dim = support_size if use_categorical else 1

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Predict reward from hidden state.

        Args:
            hidden_state: Hidden state (B, C, H, W).

        Returns:
            Reward prediction (B,) or (B, support_size).
        """
        x = F.relu(self.bn(self.conv(hidden_state)))
        x = x.view(x.size(0), -1)
        reward = self.fc(x)

        if not self.use_categorical:
            reward = reward.squeeze(-1)

        return reward


class ProjectionHead(nn.Module):
    """Projection head for EfficientZero self-supervised learning.

    Projects hidden state to a lower-dimensional embedding space
    for consistency loss computation.
    """

    def __init__(
        self,
        hidden_channels: int,
        spatial_size: int,
        projection_dim: int = 256,
    ):
        """Initialize projection head.

        Args:
            hidden_channels: Number of input channels.
            spatial_size: Spatial size (H=W) of input.
            projection_dim: Dimension of projection output.
        """
        super().__init__()

        # Reduce spatial dimensions
        self.conv = nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Project to embedding
        self.fc = nn.Sequential(
            nn.Linear(32 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden state to embedding.

        Args:
            hidden_state: Hidden state (B, C, H, W).

        Returns:
            Projection embedding (B, projection_dim).
        """
        x = F.relu(self.bn(self.conv(hidden_state)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class PredictionHead(nn.Module):
    """Prediction head for EfficientZero SimSiam-style consistency.

    Predicts target embedding from projection embedding.
    Used in consistency loss: prediction(projection(dynamics_state)) ~ projection(target_state)
    """

    def __init__(
        self,
        projection_dim: int = 256,
        hidden_dim: int = 128,
    ):
        """Initialize prediction head.

        Args:
            projection_dim: Dimension of projection embeddings.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, projection: torch.Tensor) -> torch.Tensor:
        """Predict target embedding.

        Args:
            projection: Projection embedding (B, projection_dim).

        Returns:
            Predicted embedding (B, projection_dim).
        """
        return self.net(projection)


class ValuePrefixHead(nn.Module):
    """Value prefix head for predicting cumulative future rewards.

    Predicts the sum of rewards over the next K steps.
    """

    def __init__(
        self,
        hidden_channels: int,
        spatial_size: int,
        horizon: int = 5,
        support_size: int = 601,
        use_categorical: bool = True,
    ):
        """Initialize value prefix head.

        Args:
            hidden_channels: Number of input channels.
            spatial_size: Spatial size (H=W) of input.
            horizon: Number of future steps to predict.
            support_size: Size of categorical support.
            use_categorical: If True, output categorical distribution.
        """
        super().__init__()
        self.horizon = horizon
        self.use_categorical = use_categorical

        self.conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

        flat_dim = spatial_size * spatial_size
        output_dim = support_size if use_categorical else 1

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Predict cumulative future rewards.

        Args:
            hidden_state: Hidden state (B, C, H, W).

        Returns:
            Value prefix prediction (B,) or (B, support_size).
        """
        x = F.relu(self.bn(self.conv(hidden_state)))
        x = x.view(x.size(0), -1)
        prefix = self.fc(x)

        if not self.use_categorical:
            prefix = prefix.squeeze(-1)

        return prefix


class MuZeroNetwork(nn.Module):
    """MuZero network with representation, dynamics, and prediction functions.

    This network implements the three core functions of MuZero:
    1. Representation: observation -> hidden state
    2. Dynamics: (hidden state, action) -> (next hidden state, reward)
    3. Prediction: hidden state -> (policy, value)

    The hidden state preserves spatial structure (B, C, H, W) to maintain
    map information for Civilization VI.

    Also includes EfficientZero extensions:
    - Projection/prediction heads for consistency loss
    - Value prefix prediction for credit assignment
    """

    def __init__(self, config: MuZeroConfig | None = None, **kwargs):
        """Initialize MuZero network.

        Args:
            config: MuZero configuration.
            **kwargs: Override config values.
        """
        super().__init__()

        if config is None:
            config = MuZeroConfig()

        # Allow kwargs to override config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # =====================================================================
        # Representation function: h(o) -> s
        # =====================================================================
        # Spatial encoder
        self.spatial_encoder = SpatialEncoder(
            in_channels=config.spatial_channels,
            num_filters=config.hidden_state_channels,
            num_res_blocks=config.num_res_blocks,
        )

        # Vector encoder
        self.vector_encoder = VectorEncoder(
            input_dim=config.vector_dim,
            hidden_dim=config.fc_hidden_dim,
            output_dim=config.fc_hidden_dim,
            dropout=config.dropout,
        )

        # Combine vector features into spatial (via channel-wise addition)
        self.vector_to_spatial = nn.Linear(
            config.fc_hidden_dim, config.hidden_state_channels
        )

        # =====================================================================
        # Dynamics function: g(s, a) -> (s', r)
        # =====================================================================
        self.dynamics = DynamicsNetwork(
            hidden_channels=config.hidden_state_channels,
            action_space_size=config.action_space_size,
            action_embedding_dim=config.action_embedding_dim,
            spatial_size=config.spatial_size,
            num_blocks=config.dynamics_num_blocks,
            reward_support_size=config.reward_support_size,
            use_categorical_reward=config.use_categorical_reward,
        )

        # =====================================================================
        # Prediction function: f(s) -> (p, v)
        # =====================================================================
        self.policy_head = PolicyHead(
            spatial_filters=config.hidden_state_channels,
            spatial_size=config.spatial_size,
            vector_dim=config.fc_hidden_dim,
            action_space_size=config.action_space_size,
            hidden_dim=config.fc_hidden_dim,
        )

        self.value_head = ValueHead(
            spatial_filters=config.hidden_state_channels,
            spatial_size=config.spatial_size,
            vector_dim=config.fc_hidden_dim,
            hidden_dim=config.value_head_hidden,
        )

        # Store encoded vector for prediction function
        self._encoded_vector: torch.Tensor | None = None

        # =====================================================================
        # EfficientZero extensions
        # =====================================================================
        self.projection_head: ProjectionHead | None = None
        self.prediction_head: PredictionHead | None = None
        self.value_prefix_head: ValuePrefixHead | None = None

        if config.use_consistency_loss:
            self.projection_head = ProjectionHead(
                hidden_channels=config.hidden_state_channels,
                spatial_size=config.spatial_size,
                projection_dim=config.projection_dim,
            )
            self.prediction_head = PredictionHead(
                projection_dim=config.projection_dim,
                hidden_dim=config.prediction_hidden_dim,
            )

        if config.use_value_prefix:
            self.value_prefix_head = ValuePrefixHead(
                hidden_channels=config.hidden_state_channels,
                spatial_size=config.spatial_size,
                horizon=config.value_prefix_horizon,
                support_size=config.reward_support_size,
                use_categorical=config.use_categorical_reward,
            )

    def representation(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Representation function: h(o) -> s.

        Encodes raw observation into hidden state representation.

        Args:
            spatial: Spatial input (B, C, H, W).
            vector: Vector input (B, V).

        Returns:
            Hidden state (B, hidden_channels, H, W).
        """
        # Encode spatial features
        spatial_features = self.spatial_encoder(spatial)

        # Encode vector features
        vector_features = self.vector_encoder(vector)
        self._encoded_vector = vector_features

        # Add vector information to spatial (broadcast and add)
        vector_spatial = self.vector_to_spatial(vector_features)
        vector_spatial = vector_spatial.unsqueeze(-1).unsqueeze(-1)
        vector_spatial = vector_spatial.expand_as(spatial_features)

        # Combine (residual addition)
        hidden_state = spatial_features + 0.1 * vector_spatial

        return hidden_state

    def dynamics_function(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
    ) -> DynamicsOutput:
        """Dynamics function: g(s, a) -> (s', r).

        Predicts next hidden state and reward from current state and action.

        Args:
            hidden_state: Current hidden state (B, C, H, W).
            action: Action indices (B,).

        Returns:
            DynamicsOutput with next_hidden_state and reward.
        """
        return self.dynamics(hidden_state, action)

    def prediction_function(
        self,
        hidden_state: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prediction function: f(s) -> (p, v).

        Predicts policy and value from hidden state.

        Args:
            hidden_state: Hidden state (B, C, H, W).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (policy_logits, value).
        """
        # Need vector features for policy/value heads
        if self._encoded_vector is None:
            # Use zeros if no vector available (e.g., during dynamics unroll)
            batch_size = hidden_state.size(0)
            device = hidden_state.device
            vector_features = torch.zeros(
                batch_size, self.config.fc_hidden_dim, device=device
            )
        else:
            vector_features = self._encoded_vector

        # Get policy and value
        policy_logits = self.policy_head(hidden_state, vector_features)
        value = self.value_head(hidden_state, vector_features)

        # Apply action mask
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(action_mask == 0, -1e9)

        return policy_logits, value.squeeze(-1)

    def initial_inference(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> MuZeroOutput:
        """Initial inference from raw observation.

        This is called once at the start of MCTS search to get the
        initial hidden state, policy, and value.

        Args:
            spatial: Spatial input (B, C, H, W).
            vector: Vector input (B, V).
            action_mask: Valid action mask (B, A).

        Returns:
            MuZeroOutput with hidden_state, policy_logits, and value.
        """
        # Representation
        hidden_state = self.representation(spatial, vector)

        # Prediction
        policy_logits, value = self.prediction_function(hidden_state, action_mask)

        return MuZeroOutput(
            hidden_state=hidden_state,
            policy_logits=policy_logits,
            value=value,
        )

    def recurrent_inference(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recurrent inference during MCTS search.

        Applies dynamics to get next state, then prediction to get policy/value.

        Args:
            hidden_state: Current hidden state (B, C, H, W).
            action: Action indices (B,).
            action_mask: Valid action mask for next state (B, A).

        Returns:
            Tuple of (next_hidden_state, reward, policy_logits, value).
        """
        # Dynamics
        dynamics_output = self.dynamics_function(hidden_state, action)
        next_hidden = dynamics_output.next_hidden_state
        reward = dynamics_output.reward

        # Prediction for next state
        policy_logits, value = self.prediction_function(next_hidden, action_mask)

        return next_hidden, reward, policy_logits, value

    def forward(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (AlphaZero-compatible interface).

        For compatibility with existing training code, this just returns
        policy logits and value from initial inference.

        Args:
            spatial: Spatial input (B, C, H, W).
            vector: Vector input (B, V).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (policy_logits, value).
        """
        output = self.initial_inference(spatial, vector, action_mask)
        return output.policy_logits, output.value

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy probabilities and value (AlphaZero-compatible).

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

    # =========================================================================
    # EfficientZero methods
    # =========================================================================

    def project(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden state to embedding space.

        Args:
            hidden_state: Hidden state (B, C, H, W).

        Returns:
            Projection embedding (B, projection_dim).

        Raises:
            ValueError: If consistency loss is not enabled.
        """
        if self.projection_head is None:
            raise ValueError("Consistency loss not enabled in config")
        return self.projection_head(hidden_state)

    def predict_projection(self, projection: torch.Tensor) -> torch.Tensor:
        """Predict target embedding from projection.

        Args:
            projection: Projection embedding (B, projection_dim).

        Returns:
            Predicted embedding (B, projection_dim).

        Raises:
            ValueError: If consistency loss is not enabled.
        """
        if self.prediction_head is None:
            raise ValueError("Consistency loss not enabled in config")
        return self.prediction_head(projection)

    def predict_value_prefix(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Predict cumulative future rewards (value prefix).

        Args:
            hidden_state: Hidden state (B, C, H, W).

        Returns:
            Value prefix prediction (B,) or (B, support_size).

        Raises:
            ValueError: If value prefix is not enabled.
        """
        if self.value_prefix_head is None:
            raise ValueError("Value prefix not enabled in config")
        return self.value_prefix_head(hidden_state)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def save(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        from dataclasses import asdict

        config_dict = asdict(self.config)
        for k, v in config_dict.items():
            if hasattr(v, "item"):
                config_dict[k] = v.item()

        torch.save(
            {
                "config": config_dict,
                "state_dict": self.state_dict(),
                "network_type": "muzero",
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> MuZeroNetwork:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint.
            device: Device to load model to.

        Returns:
            Loaded model.

        Note:
            Only load checkpoints from trusted sources.
        """
        result = safe_load_checkpoint(
            path,
            map_location=device,
            strict_security=False,
            allow_unsafe_fallback=True,
        )
        checkpoint = result.data
        config_dict = checkpoint["config"]

        if isinstance(config_dict, dict):
            config = MuZeroConfig(**config_dict)
        else:
            config = config_dict

        model = cls(config=config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        if not result.loaded_safely:
            logger.warning(
                "Checkpoint loaded with weights_only=False. "
                "Ensure checkpoint is from a trusted source."
            )
        return model

    def get_num_parameters(self) -> dict[str, int]:
        """Get parameter counts by component.

        Returns:
            Dictionary with parameter counts.
        """
        counts = {
            "spatial_encoder": sum(
                p.numel() for p in self.spatial_encoder.parameters()
            ),
            "vector_encoder": sum(
                p.numel() for p in self.vector_encoder.parameters()
            ),
            "dynamics": sum(p.numel() for p in self.dynamics.parameters()),
            "policy_head": sum(p.numel() for p in self.policy_head.parameters()),
            "value_head": sum(p.numel() for p in self.value_head.parameters()),
        }

        if self.projection_head is not None:
            counts["projection_head"] = sum(
                p.numel() for p in self.projection_head.parameters()
            )
        if self.prediction_head is not None:
            counts["prediction_head"] = sum(
                p.numel() for p in self.prediction_head.parameters()
            )
        if self.value_prefix_head is not None:
            counts["value_prefix_head"] = sum(
                p.numel() for p in self.value_prefix_head.parameters()
            )

        counts["total"] = sum(p.numel() for p in self.parameters())

        return counts


# =============================================================================
# Loss functions for MuZero training
# =============================================================================
# NOTE: Loss functions have been moved to alpha_zero/core/losses.py for
# centralized, reusable implementations. The following functions are imported
# at the top of this module for backward compatibility:
#
#   - scalar_to_categorical
#   - categorical_to_scalar
#   - compute_muzero_loss
#   - compute_consistency_loss
#
# See alpha_zero/core/losses.py for the canonical implementations.
# =============================================================================
