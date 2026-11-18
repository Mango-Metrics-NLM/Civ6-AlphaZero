"""Neural network architecture for Civ6 AlphaZero.

Implements a ResNet-style architecture with:
- Spatial encoder (convolutional) for map features
- Vector encoder (MLP) for global features
- Policy head with action masking
- Value head for position evaluation
"""

from __future__ import annotations

import copy
import importlib
import logging
import math
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from alpha_zero.utils.atomic_io import write_atomically
from alpha_zero.utils.checkpoint import safe_load_checkpoint, validate_primitive_metadata
from alpha_zero.utils.constants import DEFAULT_ACTION_SPACE_SIZE, NETWORK

# Module logger
logger = logging.getLogger(__name__)

try:
    import numpy as np

    # Register common public types for safe loading
    safe_types: list[Any] = [np.ndarray, np.dtype, np.generic]

    # Try to find the internal scalar type for legacy compatibility
    # This varies between NumPy 1.x and 2.x
    for _mod_name in ("numpy.core.multiarray", "numpy._core.multiarray"):
        try:
            _mod = importlib.import_module(_mod_name)
            _scalar = getattr(_mod, "scalar", None)
            if _scalar is not None and _scalar not in safe_types:
                safe_types.append(_scalar)
        except ImportError:
            pass

    torch.serialization.add_safe_globals(safe_types)
except (ImportError, AttributeError):
    pass


def safe_invalid_action_logit(dtype: torch.dtype) -> float:
    """Return a finite invalid-action mask logit safe for ``dtype``."""
    if dtype.is_floating_point:
        return max(NETWORK.INVALID_ACTION_LOGIT, float(torch.finfo(dtype).min) / 2.0)
    return NETWORK.INVALID_ACTION_LOGIT


def _resolve_network_config(
    config: NetworkConfig | None,
    defaults: NetworkConfig,
    kwargs: dict[str, Any],
    *,
    owner: str,
) -> NetworkConfig:
    """Return a copied ``NetworkConfig`` with ``kwargs`` applied, warning on unknown keys."""
    base = defaults if config is None else config
    resolved = replace(base) if is_dataclass(base) else copy.copy(base)
    unknown: list[str] = []
    for key, value in kwargs.items():
        if hasattr(resolved, key):
            setattr(resolved, key, value)
        else:
            unknown.append(key)
    if unknown:
        unknown_sorted = sorted(unknown)
        logger.warning(
            "%s ignored unknown configuration keyword arguments: %s",
            owner,
            ", ".join(unknown_sorted),
            extra={"network_class": owner, "unknown_kwargs": unknown_sorted},
        )
    return resolved


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
    # MicroCiv default, derived from the environment's own slot arithmetic
    # (see alpha_zero.utils.constants.default_action_space_size). Explicit
    # values still win, so 381-wide legacy checkpoints load when the caller
    # passes 381.
    action_space_size: int = DEFAULT_ACTION_SPACE_SIZE
    value_head_hidden: int = 128

    # Number of players. Load-bearing for the *shape* of the value head:
    # MicroCivNetwork swaps in an NPlayerValueHead when this exceeds 2. It
    # lives on the config so a saved checkpoint can be reconstructed without
    # the caller having to remember how it was trained — omitting it made
    # every N-player checkpoint fail load_state_dict against a scalar head.
    # Defaulted, so checkpoints written before this field still load.
    num_players: int = 2

    # Regularization
    dropout: float = 0.0

    # Attention parameters (Phase 2)
    num_attention_heads: int = 4
    use_attention: bool = True
    attention_dropout: float = 0.1


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""

    def __init__(self, num_filters: int) -> None:
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
    ) -> None:
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
            [ResidualBlock(num_filters) for _ in range(num_res_blocks)],
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
    ) -> None:
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
        out: torch.Tensor = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class PolicyHead(nn.Module):
    """Policy head for action prediction.

    Note:
        For new code, consider using ``HeadFactory.create_cnn_policy_head()``
        from ``alpha_zero.core.network_heads`` which provides a more flexible
        and configurable implementation.
    """

    def __init__(
        self,
        spatial_filters: int,
        spatial_size: int,
        vector_dim: int,
        action_space_size: int,
        hidden_dim: int = 256,
    ) -> None:
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
        out: torch.Tensor = F.relu(self.fc1(combined))
        out = self.fc2(out)
        return out


class ValueHead(nn.Module):
    """Value head for position evaluation.

    Note:
        For new code, consider using ``HeadFactory.create_cnn_value_head()``
        from ``alpha_zero.core.network_heads`` which provides a more flexible
        and configurable implementation.
    """

    def __init__(
        self,
        spatial_filters: int,
        spatial_size: int,
        vector_dim: int,
        hidden_dim: int = 128,
    ) -> None:
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

    def __init__(self, config: NetworkConfig | None = None, **kwargs: Any) -> None:
        """Initialize network.

        Args:
            config: Network configuration.
            **kwargs: Override config values.
        """
        super().__init__()

        config = _resolve_network_config(
            config,
            NetworkConfig(),
            kwargs,
            owner=type(self).__name__,
        )

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

        # Value head (typed as nn.Module so subclasses may swap in e.g. NPlayerValueHead)
        self.value_head: nn.Module = ValueHead(
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
            # Set invalid actions to finite negative value safe for tensor dtype
            policy_logits = policy_logits.masked_fill(
                action_mask == 0, safe_invalid_action_logit(policy_logits.dtype)
            )

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

    def save(self, path: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            metadata: Optional provenance (see ``utils.checkpoint.build_provenance``).
                Primitive values only, so the file still loads under
                ``weights_only=True``; anything else raises ``TypeError``.
        """
        # Convert to dictionary with primitive Python types
        config_dict = asdict(self.config)
        # Ensure all values are primitives (not numpy types) for safe loading
        for k, v in config_dict.items():
            if hasattr(v, "item"):  # Handle numpy scalars
                config_dict[k] = v.item()

        payload: dict[str, Any] = {
            "config": config_dict,
            "state_dict": self.state_dict(),
        }
        if metadata is not None:
            payload["metadata"] = validate_primitive_metadata(dict(metadata))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Through a temporary file: a run killed mid-save must not leave a torn checkpoint.
        write_atomically(path, lambda tmp: torch.save(payload, tmp))

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> Civ6Network:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint.
            device: Device to load model to.

        Returns:
            Loaded model.

        Note:
            Only load checkpoints from trusted sources.
        """
        # No per-call unsafe opt-in: a checkpoint the tensor-only unpickler
        # rejects raises unless CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1 is set.
        result = safe_load_checkpoint(path, map_location=device)
        checkpoint = result.data
        config_dict = checkpoint["config"]
        if isinstance(config_dict, dict):
            config = NetworkConfig(**config_dict)
        else:
            config = config_dict  # Fallback for old checkpoints if any

        model = cls(config=config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)

        if not result.loaded_safely:
            logger.warning(
                "Checkpoint loaded with weights_only=False. "
                "Ensure checkpoint is from a trusted source."
            )
        return model


class MicroCivNetwork(Civ6Network):
    """Simplified network for MicroCiv environment.

    Uses smaller architecture for faster training.

    When ``num_players > 2``, the value head is replaced with an
    :class:`~alpha_zero.multiplayer.network_heads.NPlayerValueHead` that
    outputs per-player value predictions of shape ``(B, num_players)``.
    For 2-player games, the original scalar value head is used.
    """

    def __init__(
        self,
        board_size: int = 16,
        action_space_size: int = DEFAULT_ACTION_SPACE_SIZE,
        num_res_blocks: int = 6,
        num_filters: int = 64,
        spatial_channels: int = 13,
        num_players: int = 2,
        config: NetworkConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MicroCiv network.

        Args:
            board_size: Size of game board.
            action_space_size: Total number of actions.
            num_res_blocks: Number of residual blocks.
            num_filters: Number of convolutional filters.
            spatial_channels: Number of spatial input channels.
                Defaults to 13 (2-player MicroCiv). For N-player, use
                ``3 + 4 * num_players + 2``.
            num_players: Number of players. When > 2, uses NPlayerValueHead.
            config: Optional network configuration.
            **kwargs: Additional arguments passed to Civ6Network.
        """
        if config is None:
            config = NetworkConfig(
                spatial_channels=spatial_channels,
                spatial_size=board_size,
                vector_dim=64,
                num_res_blocks=num_res_blocks,
                num_filters=num_filters,
                fc_hidden_dim=128,
                action_space_size=action_space_size,
                value_head_hidden=64,
                num_players=num_players,
            )

        super().__init__(config=config, **kwargs)
        self._num_players = self.config.num_players

        # Replace the value head with an N-player variant when needed
        if self._num_players > 2:
            try:
                from alpha_zero.multiplayer.network_heads import NPlayerValueHead

                self.value_head = NPlayerValueHead(
                    spatial_filters=self.config.num_filters,
                    spatial_size=self.config.spatial_size,
                    vector_dim=self.config.fc_hidden_dim,
                    num_players=self._num_players,
                    hidden_dim=self.config.value_head_hidden,
                )
                logger.info(
                    "MicroCivNetwork: using NPlayerValueHead (num_players=%d, spatial_channels=%d)",
                    self._num_players,
                    self.config.spatial_channels,
                )
            except ImportError:
                logger.warning(
                    "NPlayerValueHead not available; using scalar value head for %d players",
                    self._num_players,
                )

    @property
    def num_players(self) -> int:
        """Number of players this network supports."""
        return self._num_players


# =============================================================================
# Phase 2: Attention-Enhanced Network Components
# =============================================================================


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial feature maps.

    Adds learnable position information to help the network understand
    spatial relationships on the game board.
    """

    def __init__(
        self,
        num_channels: int,
        max_size: int = 128,
        learned: bool = True,
    ) -> None:
        """Initialize 2D positional encoding.

        Args:
            num_channels: Number of feature channels.
            max_size: Maximum spatial dimension (H and W).
            learned: Whether to use learnable embeddings (vs sinusoidal).
        """
        super().__init__()
        self.num_channels = num_channels
        self.max_size = max_size
        self.learned = learned

        if learned:
            # Learnable row and column embeddings
            self.row_embed = nn.Parameter(torch.randn(max_size, num_channels // 2) * 0.02)
            self.col_embed = nn.Parameter(torch.randn(max_size, num_channels // 2) * 0.02)
        else:
            # Pre-compute sinusoidal encodings
            self.register_buffer(
                "pos_encoding",
                self._create_sinusoidal_encoding(max_size, num_channels),
            )

    def _create_sinusoidal_encoding(
        self,
        max_size: int,
        num_channels: int,
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding.

        Args:
            max_size: Maximum spatial dimension.
            num_channels: Number of channels.

        Returns:
            Positional encoding tensor of shape (num_channels, max_size, max_size).
        """
        half_channels = num_channels // 2
        pe = torch.zeros(num_channels, max_size, max_size)

        # Create position indices
        y_pos = torch.arange(max_size).unsqueeze(1).expand(max_size, max_size)
        x_pos = torch.arange(max_size).unsqueeze(0).expand(max_size, max_size)

        # Create frequency bands
        div_term = torch.exp(
            torch.arange(0, half_channels, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / half_channels),
        )

        # Apply sinusoidal encoding to x and y separately
        for i, freq in enumerate(div_term):
            if 2 * i < half_channels:
                pe[2 * i, :, :] = torch.sin(y_pos.float() * freq)
                pe[2 * i + 1, :, :] = torch.cos(y_pos.float() * freq)
            if half_channels + 2 * i < num_channels:
                pe[half_channels + 2 * i, :, :] = torch.sin(x_pos.float() * freq)
            if half_channels + 2 * i + 1 < num_channels:
                pe[half_channels + 2 * i + 1, :, :] = torch.cos(x_pos.float() * freq)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor with positional encoding added.
        """
        _, _, h, w = x.shape

        if self.learned:
            # Combine row and column embeddings
            row_enc = self.row_embed[:h, :].unsqueeze(1)  # (H, 1, C//2)
            col_enc = self.col_embed[:w, :].unsqueeze(0)  # (1, W, C//2)

            # Broadcast and concatenate
            pos_enc = torch.cat(
                [
                    row_enc.expand(h, w, -1),  # (H, W, C//2)
                    col_enc.expand(h, w, -1),  # (H, W, C//2)
                ],
                dim=-1,
            )  # (H, W, C)
            pos_enc = pos_enc.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        else:
            # pos_encoding is a registered buffer (torch types buffers as Tensor | Module)
            pos_enc = cast(torch.Tensor, self.pos_encoding)[:, :h, :w].unsqueeze(0)  # (1, C, H, W)

        return x + pos_enc.to(x.device)


class SpatialSelfAttention(nn.Module):
    """Self-attention over spatial feature maps.

    Enables the network to attend to different spatial locations
    when processing map features.
    """

    def __init__(
        self,
        num_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_relative_pos: bool = True,
        spatial_size: int | tuple[int, int] | None = None,
    ) -> None:
        """Initialize spatial self-attention.

        Args:
            num_channels: Number of feature channels.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            use_relative_pos: Whether to use relative position bias.
            spatial_size: Optional ``H`` or ``(H, W)`` to eagerly initialize
                relative position parameters and buffers at construction time.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.head_dim = num_channels // num_heads
        self.scale = self.head_dim**-0.5
        self.use_relative_pos = use_relative_pos

        assert num_channels % num_heads == 0, "num_channels must be divisible by num_heads"

        # Query, Key, Value projections
        self.qkv = nn.Linear(num_channels, 3 * num_channels, bias=False)
        self.proj = nn.Linear(num_channels, num_channels)
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm = nn.LayerNorm(num_channels)

        # Relative position bias (if enabled)
        self._spatial_size: int | None = None
        self._spatial_hw: tuple[int, int] | None = None
        if use_relative_pos:
            # Reserve parameter/buffer slots without colliding with __dict__.
            self.register_parameter("relative_pos_bias_table", None)
            self.register_buffer("relative_position_index", None)

        # Initialize weights
        self._init_weights()

        if use_relative_pos and spatial_size is not None:
            if isinstance(spatial_size, tuple):
                h, w = spatial_size
            else:
                h = w = int(spatial_size)
            if h <= 0 or w <= 0:
                raise ValueError(f"spatial_size dimensions must be positive, got {spatial_size!r}")
            self._init_relative_pos_bias(h, w)

    def _init_weights(self) -> None:
        """Initialize attention weights."""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @staticmethod
    def _build_relative_position_index(h: int, w: int, *, device: torch.device) -> torch.Tensor:
        """Compute flattened 2D relative position index tensor of shape ``(H*W * H*W,)``."""
        coords_h = torch.arange(h, device=device)
        coords_w = torch.arange(w, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, H, W)
        coords_flatten = coords.flatten(1)  # (2, H*W)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, H*W, H*W)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (H*W, H*W, 2)

        # Shift to positive indices
        relative_coords[:, :, 0] += h - 1
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1

        return relative_coords.sum(-1).view(-1)  # (H*W * H*W,)

    def _is_valid_relative_pos_table(self, tensor: Any) -> bool:
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 2
            and tensor.shape[0] > 0
            and tensor.shape[1] == self.num_heads
        )

    @staticmethod
    def _is_valid_relative_pos_index(tensor: Any) -> bool:
        return (
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 1
            and tensor.numel() > 0
            and not tensor.is_floating_point()
            and not tensor.is_complex()
            and tensor.dtype != torch.bool
        )

    def _resolve_incoming_relative_pos(
        self, incoming_table: Any, incoming_index: Any
    ) -> tuple[bool, tuple[int, int] | None]:
        """Validate incoming relative-position tensors and infer ``(H, W)``."""
        if not self._is_valid_relative_pos_table(incoming_table):
            return False, None
        if not self._is_valid_relative_pos_index(incoming_index):
            return False, None

        idx_len = int(incoming_index.numel())
        p = math.isqrt(idx_len)
        if p <= 0 or p * p != idx_len:
            return False, None

        t = int(incoming_table.shape[0])
        two_s = 4 * p + 1 - t
        if two_s <= 0 or two_s % 2 != 0:
            return False, None

        s = two_s // 2
        disc = s * s - 4 * p
        if disc < 0:
            return False, None

        d = math.isqrt(disc)
        if d * d != disc or (s - d) % 2 != 0:
            return False, None

        a = (s - d) // 2
        b = (s + d) // 2
        if a <= 0 or b <= 0 or a * b != p or (2 * a - 1) * (2 * b - 1) != t:
            return False, None

        candidates = [(a, a)] if a == b else [(a, b), (b, a)]
        if incoming_index.device.type == "meta":
            if len(candidates) == 1:
                return True, candidates[0]
            if self._spatial_hw in candidates:
                return True, self._spatial_hw
            return True, None

        ordered = (
            [self._spatial_hw, *(c for c in candidates if c != self._spatial_hw)]
            if self._spatial_hw in candidates
            else candidates
        )
        index_i64 = incoming_index.to(dtype=torch.int64)
        for cand_h, cand_w in ordered:
            expected = self._build_relative_position_index(
                cand_h, cand_w, device=incoming_index.device
            )
            if torch.equal(index_i64, expected):
                return True, (cand_h, cand_w)
        return False, None

    def _init_relative_pos_bias(self, h: int, w: int) -> None:
        """Initialize relative position bias for given spatial size.

        Args:
            h: Height of feature map.
            w: Width of feature map.
        """
        cur_table = getattr(self, "relative_pos_bias_table", None)
        cur_index = getattr(self, "relative_position_index", None)
        if (
            cur_table is not None
            and cur_index is not None
            and (self._spatial_hw == (h, w) or ((h == 1 or w == 1) and self._spatial_hw == (w, h)))
        ):
            self._spatial_hw = (h, w)
            return

        spatial_size = h * w
        self._spatial_size = spatial_size
        self._spatial_hw = (h, w)

        device = self.qkv.weight.device
        dtype = self.qkv.weight.dtype

        # Create relative position bias table
        num_relative_positions = (2 * h - 1) * (2 * w - 1)
        table = nn.Parameter(
            torch.zeros(num_relative_positions, self.num_heads, device=device, dtype=dtype),
        )
        nn.init.trunc_normal_(table, std=0.02)
        self.register_parameter("relative_pos_bias_table", table)

        self.register_buffer(
            "relative_position_index",
            self._build_relative_position_index(h, w, device=device),
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.use_relative_pos:
            table_key = prefix + "relative_pos_bias_table"
            index_key = prefix + "relative_position_index"
            has_table = table_key in state_dict
            has_index = index_key in state_dict
            legacy_preforward = (
                not has_table and not has_index and (prefix + "qkv.weight") in state_dict
            )

            cur_table = getattr(self, "relative_pos_bias_table", None)
            cur_index = getattr(self, "relative_position_index", None)
            suppressed_keys: list[str] = []

            if has_table and has_index:
                incoming_table = state_dict[table_key]
                incoming_index = state_dict[index_key]
                valid_hw, resolved_hw = self._resolve_incoming_relative_pos(
                    incoming_table, incoming_index
                )
                if not valid_hw:
                    if (
                        cur_table is None
                        or cur_index is None
                        or not isinstance(incoming_table, torch.Tensor)
                        or not isinstance(incoming_index, torch.Tensor)
                        or cur_table.shape == incoming_table.shape
                        or cur_index.shape == incoming_index.shape
                    ):
                        table_shape = (
                            tuple(incoming_table.shape)
                            if isinstance(incoming_table, torch.Tensor)
                            else type(incoming_table).__name__
                        )
                        index_shape = (
                            tuple(incoming_index.shape)
                            if isinstance(incoming_index, torch.Tensor)
                            else type(incoming_index).__name__
                        )
                        error_msgs.append(
                            f"size/shape mismatch for {table_key}/{index_key}: "
                            f"expected num_heads={self.num_heads} and valid 2D relative position "
                            f"tensors, got table={table_shape}, index={index_shape}."
                        )
                        suppressed_keys.extend((table_key, index_key))
                else:
                    if cur_table is None or cur_table.shape != incoming_table.shape:
                        self.register_parameter(
                            "relative_pos_bias_table",
                            nn.Parameter(
                                torch.empty(
                                    incoming_table.shape,
                                    device=self.qkv.weight.device,
                                    dtype=self.qkv.weight.dtype,
                                )
                            ),
                        )
                    if cur_index is None or cur_index.shape != incoming_index.shape:
                        self.register_buffer(
                            "relative_position_index",
                            torch.empty(
                                incoming_index.shape,
                                device=self.qkv.weight.device,
                                dtype=incoming_index.dtype,
                            ),
                        )
                    self._spatial_hw = resolved_hw
                    self._spatial_size = (
                        resolved_hw[0] * resolved_hw[1] if resolved_hw is not None else None
                    )
            elif has_table and not self._is_valid_relative_pos_table(state_dict[table_key]):
                incoming_table = state_dict[table_key]
                if (
                    cur_table is None
                    or not isinstance(incoming_table, torch.Tensor)
                    or cur_table.shape == incoming_table.shape
                ):
                    table_shape = (
                        tuple(incoming_table.shape)
                        if isinstance(incoming_table, torch.Tensor)
                        else type(incoming_table).__name__
                    )
                    error_msgs.append(
                        f"size/shape mismatch for {table_key}: expected 2D tensor with "
                        f"num_heads={self.num_heads}, got {table_shape}."
                    )
                    suppressed_keys.append(table_key)
            elif has_index and not self._is_valid_relative_pos_index(state_dict[index_key]):
                incoming_index = state_dict[index_key]
                if (
                    cur_index is None
                    or not isinstance(incoming_index, torch.Tensor)
                    or cur_index.shape == incoming_index.shape
                ):
                    index_shape = (
                        tuple(incoming_index.shape)
                        if isinstance(incoming_index, torch.Tensor)
                        else type(incoming_index).__name__
                    )
                    error_msgs.append(
                        f"size/shape mismatch for {index_key}: expected 1D integer tensor, "
                        f"got {index_shape}."
                    )
                    suppressed_keys.append(index_key)

            load_state_dict = state_dict
            if suppressed_keys:
                load_state_dict = dict(state_dict)
                for key in suppressed_keys:
                    load_state_dict.pop(key, None)

            missing_start = len(missing_keys)
            super()._load_from_state_dict(
                load_state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

            if suppressed_keys:
                suppressed_set = set(suppressed_keys)
                missing_keys[missing_start:] = [
                    k for k in missing_keys[missing_start:] if k not in suppressed_set
                ]

            if legacy_preforward:
                logger.warning(
                    "Loading legacy SpatialSelfAttention checkpoint without relative_pos buffers "
                    "at prefix %r; retaining initialized values.",
                    prefix,
                )
                for key in (table_key, index_key):
                    if key in missing_keys:
                        missing_keys.remove(key)
            return

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with self-attention.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, C, H, W).
        """
        b, c, h, w = x.shape
        n = h * w

        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        x_seq = x.flatten(2).transpose(1, 2)

        # Pre-norm
        x_normed = self.norm(x_seq)

        # Compute Q, K, V
        qkv = self.qkv(x_normed).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)

        # Add relative position bias if enabled
        if self.use_relative_pos:
            self._init_relative_pos_bias(h, w)
            bias_table = getattr(self, "relative_pos_bias_table", None)
            pos_index = getattr(self, "relative_position_index", None)
            if bias_table is not None and pos_index is not None:
                relative_pos_bias = bias_table[pos_index].view(n, n, -1)
                relative_pos_bias = relative_pos_bias.permute(2, 0, 1).unsqueeze(0)
                attn = attn + relative_pos_bias.to(device=attn.device, dtype=attn.dtype)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out: torch.Tensor = (attn @ v).transpose(1, 2).reshape(b, n, c)  # (B, N, C)

        # Project and reshape back
        out = self.proj(out)
        out = self.dropout(out)

        # Residual connection
        out = x_seq + out

        # Reshape back to spatial: (B, H*W, C) -> (B, C, H, W)
        out = out.transpose(1, 2).reshape(b, c, h, w)

        return out


class CrossModalAttention(nn.Module):
    """Cross-attention between spatial and vector features.

    Allows the network to attend between map features and global game state.
    """

    def __init__(
        self,
        spatial_channels: int,
        vector_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize cross-modal attention.

        Args:
            spatial_channels: Number of spatial feature channels.
            vector_dim: Dimension of vector features.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.spatial_channels = spatial_channels
        self.vector_dim = vector_dim
        self.num_heads = num_heads
        self.head_dim = spatial_channels // num_heads
        self.scale = self.head_dim**-0.5

        assert spatial_channels % num_heads == 0, "spatial_channels must be divisible by num_heads"

        # Project vector to query dimension
        self.vector_to_query = nn.Linear(vector_dim, spatial_channels)

        # Spatial to key/value
        self.spatial_to_kv = nn.Linear(spatial_channels, 2 * spatial_channels, bias=False)

        # Output projection for spatial features
        self.spatial_proj = nn.Linear(spatial_channels, spatial_channels)

        # Vector receives attended spatial information
        self.vector_proj = nn.Linear(spatial_channels, vector_dim)

        # Layer normalization
        self.spatial_norm = nn.LayerNorm(spatial_channels)
        self.vector_norm = nn.LayerNorm(vector_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize attention weights."""
        nn.init.xavier_uniform_(self.vector_to_query.weight)
        nn.init.zeros_(self.vector_to_query.bias)
        nn.init.xavier_uniform_(self.spatial_to_kv.weight)
        nn.init.xavier_uniform_(self.spatial_proj.weight)
        nn.init.zeros_(self.spatial_proj.bias)
        nn.init.xavier_uniform_(self.vector_proj.weight)
        nn.init.zeros_(self.vector_proj.bias)

    def forward(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with cross-attention.

        The vector features attend to spatial features, and spatial features
        are modulated by vector-attended information.

        Args:
            spatial: Spatial features of shape (B, C, H, W).
            vector: Vector features of shape (B, V).

        Returns:
            Tuple of (updated_spatial, updated_vector).
        """
        b, c, h, w = spatial.shape
        n = h * w

        # Reshape spatial: (B, C, H, W) -> (B, H*W, C)
        spatial_seq = spatial.flatten(2).transpose(1, 2)

        # Normalize
        spatial_normed = self.spatial_norm(spatial_seq)
        vector_normed = self.vector_norm(vector)

        # Vector as query
        q = self.vector_to_query(vector_normed)  # (B, C)
        q = q.view(b, 1, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, num_heads, 1, head_dim)

        # Spatial as key/value
        kv = self.spatial_to_kv(spatial_normed).reshape(b, n, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, num_heads, N, head_dim)
        k, v = kv[0], kv[1]

        # Cross-attention: vector attends to spatial
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, 1, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Attended spatial information for vector
        vector_attended = (attn @ v).squeeze(2)  # (B, num_heads, head_dim)
        vector_attended = vector_attended.reshape(b, c)
        vector_update = self.vector_proj(vector_attended)

        # Also modulate spatial based on vector attention pattern
        # Broadcast attention weights back to spatial
        spatial_attention = attn.mean(dim=1).transpose(1, 2)  # (B, N, 1)
        spatial_gate = torch.sigmoid(spatial_attention)
        spatial_modulated = spatial_seq * spatial_gate
        spatial_update = self.spatial_proj(spatial_modulated)

        # Residual connections
        updated_vector = vector + self.dropout(vector_update)
        updated_spatial = spatial_seq + self.dropout(spatial_update)

        # Reshape spatial back
        updated_spatial = updated_spatial.transpose(1, 2).reshape(b, c, h, w)

        return updated_spatial, updated_vector


class AttentionSpatialEncoder(nn.Module):
    """Enhanced spatial encoder with self-attention layers.

    Combines convolutional residual blocks with self-attention
    for better spatial reasoning.
    """

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        num_res_blocks: int,
        num_heads: int = 4,
        attention_dropout: float = 0.1,
        attention_interval: int = 2,
        max_spatial_size: int = 128,
        spatial_size: int | tuple[int, int] | None = None,
    ) -> None:
        """Initialize attention spatial encoder.

        Args:
            in_channels: Number of input channels.
            num_filters: Number of convolutional filters.
            num_res_blocks: Number of residual blocks.
            num_heads: Number of attention heads.
            attention_dropout: Dropout for attention layers.
            attention_interval: Add attention after every N residual blocks.
            max_spatial_size: Maximum spatial size for positional encoding.
            spatial_size: Optional ``H`` or ``(H, W)`` for eager attention bias init.
        """
        super().__init__()

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(
            num_channels=num_filters,
            max_size=max_spatial_size,
            learned=True,
        )

        # Residual blocks with interleaved attention
        self.layers = nn.ModuleList()
        for i in range(num_res_blocks):
            self.layers.append(ResidualBlock(num_filters))

            # Add attention layer after every attention_interval blocks
            if (i + 1) % attention_interval == 0 and i < num_res_blocks - 1:
                self.layers.append(
                    SpatialSelfAttention(
                        num_channels=num_filters,
                        num_heads=num_heads,
                        dropout=attention_dropout,
                        use_relative_pos=True,
                        spatial_size=spatial_size,
                    ),
                )

        # Final attention layer
        self.final_attention = SpatialSelfAttention(
            num_channels=num_filters,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_relative_pos=True,
            spatial_size=spatial_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, num_filters, H, W).
        """
        # Initial convolution
        out: torch.Tensor = F.relu(self.bn_in(self.conv_in(x)))

        # Add positional encoding
        out = self.pos_encoding(out)

        # Apply residual blocks and attention layers
        for layer in self.layers:
            out = layer(out)

        # Final attention
        out = self.final_attention(out)

        return out


class HierarchicalPolicyHead(nn.Module):
    """Hierarchical policy head for action selection.

    Outputs both category probabilities and per-category action probabilities
    to enable hierarchical action selection.
    """

    # Action categories and their sizes
    NUM_CATEGORIES: ClassVar[int] = 9
    CATEGORY_NAMES: ClassVar[list[str]] = [
        "END_TURN",
        "UNIT_ACTION",
        "CITY_ACTION",
        "RESEARCH_TECH",
        "RESEARCH_CIVIC",
        "DIPLOMACY",
        "ESPIONAGE",
        "GOVERNMENT",
        "GREAT_PERSON",
    ]

    def __init__(
        self,
        spatial_filters: int,
        spatial_size: int,
        vector_dim: int,
        action_space_size: int,
        hidden_dim: int = 256,
        category_hidden_dim: int = 128,
    ) -> None:
        """Initialize hierarchical policy head.

        Args:
            spatial_filters: Number of spatial feature filters.
            spatial_size: Spatial feature map size (H=W).
            vector_dim: Vector feature dimension.
            action_space_size: Total number of flat actions.
            hidden_dim: Hidden dimension for action prediction.
            category_hidden_dim: Hidden dimension for category prediction.
        """
        super().__init__()

        self.spatial_size = spatial_size
        self.action_space_size = action_space_size

        # Spatial reduction for category prediction
        self.category_conv = nn.Conv2d(spatial_filters, 4, 1, bias=False)
        self.category_bn = nn.BatchNorm2d(4)

        # Category prediction layers
        cat_spatial_dim = 4 * spatial_size * spatial_size
        cat_combined_dim = cat_spatial_dim + vector_dim
        self.category_fc1 = nn.Linear(cat_combined_dim, category_hidden_dim)
        self.category_fc2 = nn.Linear(category_hidden_dim, self.NUM_CATEGORIES)

        # Spatial reduction for action prediction
        self.action_conv = nn.Conv2d(spatial_filters, 2, 1, bias=False)
        self.action_bn = nn.BatchNorm2d(2)

        # Action prediction layers (flat output)
        action_spatial_dim = 2 * spatial_size * spatial_size
        action_combined_dim = action_spatial_dim + vector_dim + self.NUM_CATEGORIES
        self.action_fc1 = nn.Linear(action_combined_dim, hidden_dim)
        self.action_fc2 = nn.Linear(hidden_dim, action_space_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize layer weights."""
        for module in [self.category_fc1, self.category_fc2, self.action_fc1, self.action_fc2]:
            if hasattr(module, "weight"):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        spatial_features: torch.Tensor,
        vector_features: torch.Tensor,
        return_category_probs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            spatial_features: Spatial features (B, C, H, W).
            vector_features: Vector features (B, V).
            return_category_probs: If True, also return category probabilities.

        Returns:
            Policy logits (B, action_space_size), optionally with category probs.
        """
        # Category prediction
        cat_s = F.relu(self.category_bn(self.category_conv(spatial_features)))
        cat_s = cat_s.view(cat_s.size(0), -1)
        cat_combined = torch.cat([cat_s, vector_features], dim=1)
        cat_hidden = F.relu(self.category_fc1(cat_combined))
        category_logits = self.category_fc2(cat_hidden)
        category_probs = F.softmax(category_logits, dim=-1)

        # Action prediction (conditioned on category probabilities)
        act_s = F.relu(self.action_bn(self.action_conv(spatial_features)))
        act_s = act_s.view(act_s.size(0), -1)
        act_combined = torch.cat([act_s, vector_features, category_probs], dim=1)
        act_hidden = F.relu(self.action_fc1(act_combined))
        action_logits: torch.Tensor = self.action_fc2(act_hidden)

        if return_category_probs:
            return action_logits, category_probs
        return action_logits


class Civ6NetworkV2(nn.Module):
    """Enhanced AlphaZero network with attention mechanisms for Phase 2.

    Improvements over Civ6Network:
    1. Self-attention layers in spatial encoder
    2. Cross-attention between spatial and vector features
    3. Positional encoding for spatial features
    4. Hierarchical policy head for better action selection

    Maintains backward compatibility with the same forward/predict/save/load interface.
    """

    def __init__(self, config: NetworkConfig | None = None, **kwargs: Any) -> None:
        """Initialize network.

        Args:
            config: Network configuration.
            **kwargs: Override config values.
        """
        super().__init__()

        default_config = NetworkConfig(
            spatial_channels=84,  # Full Civ6StateEncoder
            spatial_size=16,
            vector_dim=512,
            num_res_blocks=10,
            num_filters=128,
            fc_hidden_dim=512,
            action_space_size=50000,  # Full hierarchical space
            value_head_hidden=256,
            num_attention_heads=4,
            use_attention=True,
            attention_dropout=0.1,
        )
        config = _resolve_network_config(
            config,
            default_config,
            kwargs,
            owner=type(self).__name__,
        )

        self.config = config

        # Attention-enhanced spatial encoder - Union type for both possibilities
        self.spatial_encoder: AttentionSpatialEncoder | SpatialEncoder
        if config.use_attention:
            self.spatial_encoder = AttentionSpatialEncoder(
                in_channels=config.spatial_channels,
                num_filters=config.num_filters,
                num_res_blocks=config.num_res_blocks,
                num_heads=config.num_attention_heads,
                attention_dropout=config.attention_dropout,
                attention_interval=2,
                max_spatial_size=max(config.spatial_size * 2, 128),
                spatial_size=config.spatial_size,
            )
        else:
            self.spatial_encoder = SpatialEncoder(
                in_channels=config.spatial_channels,
                num_filters=config.num_filters,
                num_res_blocks=config.num_res_blocks,
            )

        # Enhanced vector encoder
        self.vector_encoder = VectorEncoder(
            input_dim=config.vector_dim,
            hidden_dim=config.fc_hidden_dim,
            output_dim=config.fc_hidden_dim,
            dropout=config.dropout,
        )

        # Cross-modal attention (if enabled)
        self.cross_attention: CrossModalAttention | None = None
        if config.use_attention:
            self.cross_attention = CrossModalAttention(
                spatial_channels=config.num_filters,
                vector_dim=config.fc_hidden_dim,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
            )

        # Hierarchical policy head
        self.policy_head = HierarchicalPolicyHead(
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

        # Initialize attention weights
        self._init_attention_weights()

    def _init_attention_weights(self) -> None:
        """Initialize attention layer weights for stable training."""
        for module in self.modules():
            if isinstance(module, (SpatialSelfAttention, CrossModalAttention)):
                # Attention layers initialize themselves
                pass
            elif isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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

        # Cross-modal attention (if enabled)
        if self.cross_attention is not None:
            spatial_features, vector_features = self.cross_attention(
                spatial_features,
                vector_features,
            )

        # Get policy logits and value
        policy_logits = self.policy_head(spatial_features, vector_features)
        value = self.value_head(spatial_features, vector_features)

        # Apply action mask
        if action_mask is not None:
            # Set invalid actions to finite negative value safe for tensor dtype
            policy_logits = policy_logits.masked_fill(
                action_mask == 0, safe_invalid_action_logit(policy_logits.dtype)
            )

        return policy_logits, value.squeeze(-1)

    def forward_with_categories(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning category probabilities for hierarchical selection.

        Args:
            spatial: Spatial input (B, C, H, W).
            vector: Vector input (B, V).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (policy_logits, category_probs, value).
        """
        # Encode features
        spatial_features = self.spatial_encoder(spatial)
        vector_features = self.vector_encoder(vector)

        # Cross-modal attention (if enabled)
        if self.cross_attention is not None:
            spatial_features, vector_features = self.cross_attention(
                spatial_features,
                vector_features,
            )

        # Get policy logits with category probabilities
        policy_logits, category_probs = self.policy_head(
            spatial_features,
            vector_features,
            return_category_probs=True,
        )
        value = self.value_head(spatial_features, vector_features)

        # Apply action mask
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(
                action_mask == 0, safe_invalid_action_logit(policy_logits.dtype)
            )

        return policy_logits, category_probs, value.squeeze(-1)

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

    @torch.no_grad()
    def predict_hierarchical(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Predict with hierarchical category probabilities.

        Args:
            spatial: Spatial input (C, H, W).
            vector: Vector input (V,).
            action_mask: Valid action mask (A,).

        Returns:
            Tuple of (policy_probs, category_probs, value).
        """
        self.eval()

        # Add batch dimension
        spatial = spatial.unsqueeze(0)
        vector = vector.unsqueeze(0)
        if action_mask is not None:
            action_mask = action_mask.unsqueeze(0)

        policy_logits, category_probs, value = self.forward_with_categories(
            spatial,
            vector,
            action_mask,
        )
        policy_probs = F.softmax(policy_logits, dim=-1)

        return policy_probs.squeeze(0), category_probs.squeeze(0), value.item()

    def save(self, path: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            metadata: Optional provenance (see ``utils.checkpoint.build_provenance``).
                Primitive values only, so the file still loads under
                ``weights_only=True``; anything else raises ``TypeError``.
        """
        # Convert to dictionary with primitive Python types
        config_dict = asdict(self.config)
        # Ensure all values are primitives (not numpy types) for safe loading
        for k, v in config_dict.items():
            if hasattr(v, "item"):  # Handle numpy scalars
                config_dict[k] = v.item()

        payload: dict[str, Any] = {
            "config": config_dict,
            "state_dict": self.state_dict(),
            "network_version": "v2",
        }
        if metadata is not None:
            payload["metadata"] = validate_primitive_metadata(dict(metadata))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        write_atomically(path, lambda tmp: torch.save(payload, tmp))

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> Civ6NetworkV2:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint.
            device: Device to load model to.

        Returns:
            Loaded model.

        Note:
            Only load checkpoints from trusted sources.
        """
        # No per-call unsafe opt-in: a checkpoint the tensor-only unpickler
        # rejects raises unless CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1 is set.
        result = safe_load_checkpoint(path, map_location=device)
        checkpoint = result.data
        config_dict = checkpoint["config"]

        if isinstance(config_dict, dict):
            config = NetworkConfig(**config_dict)
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

    @classmethod
    def from_v1_checkpoint(
        cls,
        path: str,
        device: str = "cpu",
        enable_attention: bool = True,
    ) -> Civ6NetworkV2:
        """Load V2 model from V1 checkpoint (backward compatibility).

        Initializes attention layers fresh while loading pre-trained
        convolutional and MLP layers.

        Args:
            path: Path to V1 checkpoint.
            device: Device to load model to.
            enable_attention: Whether to enable attention in loaded model.

        Returns:
            Loaded V2 model with V1 weights.

        Note:
            Only load checkpoints from trusted sources.
        """
        # No per-call unsafe opt-in: a checkpoint the tensor-only unpickler
        # rejects raises unless CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1 is set.
        result = safe_load_checkpoint(path, map_location=device)
        checkpoint = result.data
        config_dict = checkpoint["config"]

        if isinstance(config_dict, dict):
            # Update config for V2
            config_dict["use_attention"] = enable_attention
            if "num_attention_heads" not in config_dict:
                config_dict["num_attention_heads"] = 4
            if "attention_dropout" not in config_dict:
                config_dict["attention_dropout"] = 0.1
            config = NetworkConfig(**config_dict)
        else:
            config = config_dict

        model = cls(config=config)

        # Load compatible weights
        v1_state = checkpoint["state_dict"]
        v2_state = model.state_dict()

        # Transfer weights that have matching shapes
        transferred = 0
        skipped = 0
        for key in v1_state:
            if key in v2_state and v1_state[key].shape == v2_state[key].shape:
                v2_state[key] = v1_state[key]
                transferred += 1
            else:
                skipped += 1

        model.load_state_dict(v2_state)
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
            "spatial_encoder": sum(p.numel() for p in self.spatial_encoder.parameters()),
            "vector_encoder": sum(p.numel() for p in self.vector_encoder.parameters()),
            "policy_head": sum(p.numel() for p in self.policy_head.parameters()),
            "value_head": sum(p.numel() for p in self.value_head.parameters()),
        }

        if self.cross_attention is not None:
            counts["cross_attention"] = sum(p.numel() for p in self.cross_attention.parameters())

        counts["total"] = sum(p.numel() for p in self.parameters())

        return counts


def compute_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute AlphaZero loss.

    .. deprecated::
        Use :func:`alpha_zero.core.losses.compute_policy_loss` and
        :func:`alpha_zero.core.losses.compute_value_loss` for more
        configurable loss computation with categorical support.

    Args:
        policy_logits: Predicted policy logits (B, A).
        value: Predicted value (B,).
        target_policy: Target policy distribution (B, A).
        target_value: Target value (B,).
        action_mask: Valid action mask (B, A).

    Returns:
        Tuple of (total_loss, policy_loss, value_loss).
    """
    import warnings

    warnings.warn(
        "compute_loss() is deprecated. Use alpha_zero.core.losses.compute_policy_loss() "
        "and alpha_zero.core.losses.compute_value_loss() for more configurable loss computation.",
        DeprecationWarning,
        stacklevel=2,
    )
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
