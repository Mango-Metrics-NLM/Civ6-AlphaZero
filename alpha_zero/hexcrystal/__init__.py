"""Hex-topology spatial encoders ΓÇö a torch-only, standalone-reusable subpackage.

This package implements a spatial encoder for hexagonal game boards represented
on a rectangular grid, plus the encoder registry that selects between it and the
conventional Vision-Transformer patch embedding.

**Dependency contract.** Everything in this package depends on ``torch`` and the
Python standard library and nothing else. It deliberately does not import numpy,
gymnasium, or any other part of ``alpha_zero``, so it can be vendored or reused
outside this repository ΓÇö for hex-lattice problems in games, telescope arrays, or
molecular grids. ``tests/unit/test_hexcrystal_isolation.py`` enforces this by
importing the package in a subprocess with numpy and gymnasium blocked.

What the encoder actually does, in order:

1. :class:`~alpha_zero.hexcrystal.hex_conv.HexConvolution` applies a 7-tap kernel
   (centre tile plus its six neighbours) using pre-computed adjacency indices for
   **even-q offset** coordinates. Other offset systems are not supported.
2. :class:`~alpha_zero.hexcrystal.hex_encoder.SpatialToTokens` adaptively pools
   the feature map to a fixed grid and projects it to a token sequence. Note that
   the pooling step discards fine spatial structure.
3. :class:`~alpha_zero.hexcrystal.crystalline_layers.CrystallineLatticeLayer`
   optionally applies a set of learned symmetry-group transforms and combines
   them across the *transform* axis (not spatially).

There is no positional encoding in this package; when used inside
:class:`~alpha_zero.core.transformer_network.Civ6TransformerNetwork`, position
information comes from that network's learned position embedding.

Example:
    >>> from alpha_zero.hexcrystal import HexEncoderConfig, HexSpatialEncoder
    >>> import torch
    >>> encoder = HexSpatialEncoder(HexEncoderConfig(in_channels=13, spatial_size=8))
    >>> encoder(torch.randn(2, 13, 8, 8)).shape
    torch.Size([2, 4, 64])
"""

from __future__ import annotations

from alpha_zero.hexcrystal.crystalline_layers import (
    CrystallineLatticeConfig,
    CrystallineLatticeLayer,
    SymmetryType,
)
from alpha_zero.hexcrystal.encoder_registry import (
    EncoderType,
    create_spatial_encoder,
    get_registered_encoders,
    register_encoder,
)
from alpha_zero.hexcrystal.hex_conv import (
    HexAdjacency,
    HexConvBlock,
    HexConvConfig,
    HexConvolution,
)
from alpha_zero.hexcrystal.hex_encoder import (
    HexEncoderConfig,
    HexSpatialEncoder,
    SpatialToTokens,
    create_hex_encoder,
)
from alpha_zero.hexcrystal.patch_embed import PatchEmbed

__all__ = [
    "CrystallineLatticeConfig",
    "CrystallineLatticeLayer",
    "EncoderType",
    "HexAdjacency",
    "HexConvBlock",
    "HexConvConfig",
    "HexConvolution",
    "HexEncoderConfig",
    "HexSpatialEncoder",
    "PatchEmbed",
    "SpatialToTokens",
    "SymmetryType",
    "create_hex_encoder",
    "create_spatial_encoder",
    "get_registered_encoders",
    "register_encoder",
]
