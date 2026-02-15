"""Reusable network head components.

This module provides configurable, reusable head components for neural
networks. It addresses code duplication between PolicyHead/ValueHead in
civ6_network.py and TransformerPolicyHead/TransformerValueHead in
transformer_network.py.

Components:
    - MLPHeadConfig: Configuration for MLP head architecture
    - MLPHead: Configurable multi-layer perceptron
    - SpatialPreprocessor: CNN-style spatial feature reduction
    - FeatureCombiner: Combines spatial and vector features
    - HeadFactory: Factory for creating standard head configurations

Example:
    >>> config = MLPHeadConfig(
    ...     input_dim=512,
    ...     output_dim=100,
    ...     hidden_dims=[256],
    ...     activation="relu",
    ... )
    >>> head = MLPHead(config)
    >>> output = head(features)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# Module-level logger
logger = logging.getLogger(__name__)


class ActivationType(str, Enum):
    """Supported activation functions."""

    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    NONE = "none"


def get_activation(activation: ActivationType | str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get activation function by name.

    Args:
        activation: Activation type or string name.

    Returns:
        Callable activation function.

    Raises:
        ValueError: If activation type is unknown.
    """
    if isinstance(activation, str):
        activation = ActivationType(activation.lower())

    activations: dict[ActivationType, Callable[[torch.Tensor], torch.Tensor]] = {
        ActivationType.RELU: F.relu,
        ActivationType.GELU: F.gelu,
        ActivationType.TANH: torch.tanh,
        ActivationType.SIGMOID: torch.sigmoid,
        ActivationType.NONE: lambda x: x,
    }

    if activation not in activations:
        raise ValueError(f"Unknown activation: {activation}")

    return activations[activation]


def get_activation_module(activation: ActivationType | str) -> nn.Module:
    """Get activation module by name.

    Args:
        activation: Activation type or string name.

    Returns:
        nn.Module activation layer.

    Raises:
        ValueError: If activation type is unknown.
    """
    if isinstance(activation, str):
        activation = ActivationType(activation.lower())

    modules = {
        ActivationType.RELU: nn.ReLU,
        ActivationType.GELU: nn.GELU,
        ActivationType.TANH: nn.Tanh,
        ActivationType.SIGMOID: nn.Sigmoid,
        ActivationType.NONE: nn.Identity,
    }

    if activation not in modules:
        raise ValueError(f"Unknown activation: {activation}")

    return cast(nn.Module, modules[activation]())


@dataclass
class MLPHeadConfig:
    """Configuration for MLP head.

    Attributes:
        input_dim: Input feature dimension.
        output_dim: Output dimension (e.g., action space size or 1 for value).
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function between layers.
        output_activation: Activation for output layer (e.g., "tanh" for value).
        dropout: Dropout probability (0.0 = no dropout).
        use_layer_norm: Whether to use layer normalization.
        use_batch_norm: Whether to use batch normalization (mutually exclusive with layer_norm).
    """

    input_dim: int
    output_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [256])
    activation: str = "relu"
    output_activation: str = "none"
    dropout: float = 0.0
    use_layer_norm: bool = False
    use_batch_norm: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if any(d <= 0 for d in self.hidden_dims):
            raise ValueError(f"hidden_dims must be positive, got {self.hidden_dims}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.use_layer_norm and self.use_batch_norm:
            raise ValueError("Cannot use both layer_norm and batch_norm")


class MLPHead(nn.Module):
    """Configurable multi-layer perceptron head.

    This module provides a flexible MLP implementation that can be used
    for policy heads, value heads, or any other head type requiring
    an MLP architecture.

    Example:
        >>> config = MLPHeadConfig(input_dim=256, output_dim=100, hidden_dims=[128])
        >>> head = MLPHead(config)
        >>> output = head(torch.randn(32, 256))
        >>> output.shape
        torch.Size([32, 100])
    """

    def __init__(self, config: MLPHeadConfig) -> None:
        """Initialize MLP head.

        Args:
            config: Configuration for the MLP architecture.
        """
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_dim = config.input_dim

        # Build hidden layers
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))

            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(get_activation_module(config.activation))

            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))

            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, config.output_dim))

        # Output activation
        if config.output_activation != "none":
            layers.append(get_activation_module(config.output_activation))

        self.mlp = nn.Sequential(*layers)

        logger.debug(
            f"MLPHead created: {config.input_dim} -> {config.hidden_dims} -> {config.output_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (B, input_dim).

        Returns:
            Output of shape (B, output_dim).
        """
        return self.mlp(x)


@dataclass
class SpatialPreprocessorConfig:
    """Configuration for spatial feature preprocessing.

    Attributes:
        in_channels: Number of input spatial channels.
        out_channels: Number of output channels after reduction.
        spatial_size: Spatial dimension (H=W).
        use_batch_norm: Whether to use batch normalization.
    """

    in_channels: int
    out_channels: int
    spatial_size: int
    use_batch_norm: bool = True


class SpatialPreprocessor(nn.Module):
    """CNN-style spatial feature reduction.

    Reduces spatial feature maps using 1x1 convolution, optionally
    followed by batch normalization and ReLU activation.

    Example:
        >>> config = SpatialPreprocessorConfig(in_channels=128, out_channels=2, spatial_size=16)
        >>> preprocessor = SpatialPreprocessor(config)
        >>> spatial = torch.randn(32, 128, 16, 16)
        >>> output = preprocessor(spatial)
        >>> output.shape
        torch.Size([32, 512])  # 2 * 16 * 16
    """

    def __init__(self, config: SpatialPreprocessorConfig) -> None:
        """Initialize spatial preprocessor.

        Args:
            config: Configuration for the preprocessor.
        """
        super().__init__()
        self.config = config

        self.conv = nn.Conv2d(config.in_channels, config.out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(config.out_channels) if config.use_batch_norm else nn.Identity()

        self.output_dim = config.out_channels * config.spatial_size * config.spatial_size

        logger.debug(
            f"SpatialPreprocessor created: {config.in_channels} -> "
            f"{config.out_channels}, output_dim={self.output_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Spatial features of shape (B, C, H, W).

        Returns:
            Flattened features of shape (B, out_channels * H * W).
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x.view(x.size(0), -1)


class FeatureCombiner(nn.Module):
    """Combines spatial and vector features.

    This module takes preprocessed spatial features and vector features,
    concatenates them, and optionally applies a linear projection.

    Example:
        >>> combiner = FeatureCombiner(spatial_dim=512, vector_dim=256)
        >>> spatial = torch.randn(32, 512)
        >>> vector = torch.randn(32, 256)
        >>> combined = combiner(spatial, vector)
        >>> combined.shape
        torch.Size([32, 768])
    """

    def __init__(
        self,
        spatial_dim: int,
        vector_dim: int,
        projection_dim: int | None = None,
    ) -> None:
        """Initialize feature combiner.

        Args:
            spatial_dim: Dimension of spatial features.
            vector_dim: Dimension of vector features.
            projection_dim: Optional projection dimension. If None, no projection.
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        self.vector_dim = vector_dim
        self.combined_dim = spatial_dim + vector_dim

        self.projection: nn.Linear | None
        if projection_dim is not None:
            self.projection = nn.Linear(self.combined_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.projection = None
            self.output_dim = self.combined_dim

    def forward(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            spatial: Spatial features of shape (B, spatial_dim).
            vector: Vector features of shape (B, vector_dim).

        Returns:
            Combined features of shape (B, output_dim).
        """
        combined = torch.cat([spatial, vector], dim=1)
        if self.projection is not None:
            combined = self.projection(combined)
        return combined


class HeadFactory:
    """Factory for creating standard head configurations.

    Provides convenient methods for creating policy and value heads
    with commonly used configurations.
    """

    @staticmethod
    def create_cnn_policy_head(
        spatial_filters: int,
        spatial_size: int,
        vector_dim: int,
        action_space_size: int,
        hidden_dim: int = 256,
        spatial_reduction_channels: int = 2,
    ) -> nn.Module:
        """Create a CNN-style policy head.

        This creates a head similar to the original PolicyHead in civ6_network.py.

        Args:
            spatial_filters: Number of input spatial filter channels.
            spatial_size: Spatial feature map size (H=W).
            vector_dim: Vector feature dimension.
            action_space_size: Number of actions.
            hidden_dim: Hidden layer dimension.
            spatial_reduction_channels: Channels after spatial reduction.

        Returns:
            Policy head module.
        """
        spatial_config = SpatialPreprocessorConfig(
            in_channels=spatial_filters,
            out_channels=spatial_reduction_channels,
            spatial_size=spatial_size,
        )
        spatial_preprocessor = SpatialPreprocessor(spatial_config)

        combiner = FeatureCombiner(
            spatial_dim=spatial_preprocessor.output_dim,
            vector_dim=vector_dim,
        )

        mlp_config = MLPHeadConfig(
            input_dim=combiner.output_dim,
            output_dim=action_space_size,
            hidden_dims=[hidden_dim],
            activation="relu",
            output_activation="none",
        )
        mlp = MLPHead(mlp_config)

        return _CNNHead(spatial_preprocessor, combiner, mlp)

    @staticmethod
    def create_cnn_value_head(
        spatial_filters: int,
        spatial_size: int,
        vector_dim: int,
        hidden_dim: int = 128,
        spatial_reduction_channels: int = 1,
    ) -> nn.Module:
        """Create a CNN-style value head.

        This creates a head similar to the original ValueHead in civ6_network.py.

        Args:
            spatial_filters: Number of input spatial filter channels.
            spatial_size: Spatial feature map size (H=W).
            vector_dim: Vector feature dimension.
            hidden_dim: Hidden layer dimension.
            spatial_reduction_channels: Channels after spatial reduction.

        Returns:
            Value head module.
        """
        spatial_config = SpatialPreprocessorConfig(
            in_channels=spatial_filters,
            out_channels=spatial_reduction_channels,
            spatial_size=spatial_size,
        )
        spatial_preprocessor = SpatialPreprocessor(spatial_config)

        combiner = FeatureCombiner(
            spatial_dim=spatial_preprocessor.output_dim,
            vector_dim=vector_dim,
        )

        mlp_config = MLPHeadConfig(
            input_dim=combiner.output_dim,
            output_dim=1,
            hidden_dims=[hidden_dim],
            activation="relu",
            output_activation="tanh",
        )
        mlp = MLPHead(mlp_config)

        return _CNNHead(spatial_preprocessor, combiner, mlp)

    @staticmethod
    def create_transformer_policy_head(
        embed_dim: int,
        action_space_size: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> nn.Module:
        """Create a transformer-style policy head.

        This creates a head similar to TransformerPolicyHead in transformer_network.py.

        Args:
            embed_dim: Input embedding dimension.
            action_space_size: Number of actions.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout probability.

        Returns:
            Policy head module.
        """
        config = MLPHeadConfig(
            input_dim=embed_dim,
            output_dim=action_space_size,
            hidden_dims=[hidden_dim, hidden_dim],
            activation="gelu",
            output_activation="none",
            dropout=dropout,
        )
        return MLPHead(config)

    @staticmethod
    def create_transformer_value_head(
        embed_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> nn.Module:
        """Create a transformer-style value head.

        This creates a head similar to TransformerValueHead in transformer_network.py.

        Args:
            embed_dim: Input embedding dimension.
            hidden_dim: Hidden layer dimension.
            dropout: Dropout probability.

        Returns:
            Value head module.
        """
        config = MLPHeadConfig(
            input_dim=embed_dim,
            output_dim=1,
            hidden_dims=[hidden_dim, hidden_dim],
            activation="gelu",
            output_activation="tanh",
            dropout=dropout,
        )
        return MLPHead(config)


class _CNNHead(nn.Module):
    """Internal CNN head combining spatial preprocessor, combiner, and MLP.

    This is an internal class used by HeadFactory to create CNN-style heads.
    It takes spatial and vector features as separate inputs.
    """

    def __init__(
        self,
        spatial_preprocessor: SpatialPreprocessor,
        combiner: FeatureCombiner,
        mlp: MLPHead,
    ) -> None:
        """Initialize CNN head.

        Args:
            spatial_preprocessor: Spatial feature preprocessor.
            combiner: Feature combiner.
            mlp: MLP output head.
        """
        super().__init__()
        self.spatial_preprocessor = spatial_preprocessor
        self.combiner = combiner
        self.mlp = mlp

    def forward(
        self,
        spatial_features: torch.Tensor,
        vector_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            spatial_features: Spatial features of shape (B, C, H, W).
            vector_features: Vector features of shape (B, V).

        Returns:
            Output of shape (B, output_dim).
        """
        spatial = self.spatial_preprocessor(spatial_features)
        combined = self.combiner(spatial, vector_features)
        return self.mlp(combined)
