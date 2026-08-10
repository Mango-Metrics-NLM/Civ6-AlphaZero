"""Backwards-compatible re-exports for the hexagonal spatial encoder.

The implementation moved to :mod:`alpha_zero.hexcrystal.hex_encoder` so that it
can be imported without pulling in the rest of ``alpha_zero`` (and therefore
numpy and gymnasium). This module re-exports the public names so historical
import paths keep working; both paths resolve to the same objects.
"""

from __future__ import annotations

from alpha_zero.hexcrystal.hex_encoder import (
    HexEncoderConfig,
    HexSpatialEncoder,
    SpatialToTokens,
    create_hex_encoder,
)

__all__ = [
    "HexEncoderConfig",
    "HexSpatialEncoder",
    "SpatialToTokens",
    "create_hex_encoder",
]
