"""Encoder registry and factory for spatial encoders.

This module provides a registry pattern for selecting between different spatial
encoder implementations (PatchEmbed, HexSpatialEncoder, etc.) based on a string
or enum identifier.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


class EncoderType(str, Enum):
    """Supported spatial encoder types."""

    PATCH_EMBED = "patch_embed"
    HEX_CRYSTALLINE = "hex_crystalline"


# Type for encoder factory functions
EncoderFactory = Callable[..., nn.Module]

# Private registry dict
_ENCODER_REGISTRY: dict[str, EncoderFactory] = {}


def register_encoder(
    encoder_type: str | EncoderType,
) -> Callable[[EncoderFactory], EncoderFactory]:
    """Decorator to register an encoder factory.

    Args:
        encoder_type: The encoder type identifier (string or enum).

    Returns:
        Decorator function that registers the factory and returns it unchanged.

    Example:
        @register_encoder(EncoderType.PATCH_EMBED)
        def _create_patch_embed(...) -> nn.Module:
            ...
    """

    def decorator(factory: EncoderFactory) -> EncoderFactory:
        # Convert enum to string if needed
        type_key = encoder_type.value if isinstance(encoder_type, EncoderType) else encoder_type
        _ENCODER_REGISTRY[type_key] = factory
        logger.debug(f"Registered encoder factory for type: {type_key}")
        return factory

    return decorator


def get_registered_encoders() -> list[str]:
    """Return list of registered encoder type names.

    Returns:
        List of registered encoder type strings.
    """
    return list(_ENCODER_REGISTRY.keys())


def create_spatial_encoder(
    encoder_type: str | EncoderType,
    in_channels: int,
    embed_dim: int,
    patch_size: int,
    spatial_size: int,
    **kwargs: Any,
) -> nn.Module:
    """Create a spatial encoder by type.

    Args:
        encoder_type: Type of encoder to create.
        in_channels: Number of input spatial channels.
        embed_dim: Embedding dimension for output tokens.
        patch_size: Patch size (used by PatchEmbed, and to compute num_output_tokens for hex).
        spatial_size: Spatial input size (H=W).
        **kwargs: Additional kwargs passed to the encoder factory.

    Returns:
        An nn.Module with:
        - forward(x: Tensor) -> Tensor mapping (B, C, H, W) -> (B, num_patches, embed_dim)
        - num_patches property

    Raises:
        ValueError: If encoder_type is not registered.
    """
    # Convert enum to string if needed
    type_key = encoder_type.value if isinstance(encoder_type, EncoderType) else encoder_type

    if type_key not in _ENCODER_REGISTRY:
        available = ", ".join(get_registered_encoders())
        raise ValueError(f"Unknown encoder type: {type_key}. Available types: {available}")

    factory = _ENCODER_REGISTRY[type_key]
    logger.debug(
        f"Creating spatial encoder: type={type_key}, in_channels={in_channels}, "
        f"embed_dim={embed_dim}, patch_size={patch_size}, spatial_size={spatial_size}"
    )

    encoder = factory(
        in_channels=in_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        spatial_size=spatial_size,
        **kwargs,
    )

    return encoder


@register_encoder(EncoderType.PATCH_EMBED)
def _create_patch_embed(
    in_channels: int,
    embed_dim: int,
    patch_size: int,
    spatial_size: int,
    **kwargs: Any,  # absorb unknown kwargs
) -> nn.Module:
    """Create a PatchEmbed encoder.

    Args:
        in_channels: Number of input spatial channels.
        embed_dim: Embedding dimension for output tokens.
        patch_size: Size of each square patch.
        spatial_size: Spatial input size (H=W).
        **kwargs: Additional kwargs (ignored for PatchEmbed).

    Returns:
        PatchEmbed instance.
    """
    # Lazy import to avoid circular dependencies
    from alpha_zero.core.transformer_network import PatchEmbed

    return PatchEmbed(
        in_channels=in_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        spatial_size=spatial_size,
    )


@register_encoder(EncoderType.HEX_CRYSTALLINE)
def _create_hex_encoder(
    in_channels: int,
    embed_dim: int,
    patch_size: int,
    spatial_size: int,
    **kwargs: Any,
) -> nn.Module:
    """Create a HexSpatialEncoder.

    Args:
        in_channels: Number of input spatial channels.
        embed_dim: Embedding dimension for output tokens.
        patch_size: Patch size (used to compute num_output_tokens).
        spatial_size: Spatial input size (H=W).
        **kwargs: Additional kwargs (e.g., num_scales, dropout, etc.).

    Returns:
        HexSpatialEncoder instance.
    """
    # Lazy import to avoid circular dependencies
    from alpha_zero.core.hex_encoder import create_hex_encoder

    return create_hex_encoder(
        in_channels=in_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        spatial_size=spatial_size,
        **kwargs,
    )
