"""Backwards-compatible re-exports for the hexagonal spatial encoder.

The implementation moved to :mod:`alpha_zero.hexcrystal.hex_encoder` so that it
can be imported without pulling in the rest of ``alpha_zero`` (and therefore
numpy and gymnasium). This module re-exports the public names so historical
import paths keep working; both paths resolve to the same objects.
"""

from __future__ import annotations

import warnings

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

warnings.warn(
    "alpha_zero.core.hex_encoder is a deprecated re-export shim; import from "
    "alpha_zero.hexcrystal.hex_encoder instead. The shim will be removed one release after 2026-09-02.",
    DeprecationWarning,
    stacklevel=2,
)
