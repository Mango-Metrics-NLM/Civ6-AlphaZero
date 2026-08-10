"""Backwards-compatibility shim: alpha_zero.core.hex_encoder.

The implementation has moved to :mod:`alpha_zero.hexcrystal.hex_encoder`.
This module re-exports the entire public surface so that existing callers
are unaffected by the refactoring.

.. deprecated::
    Import from ``alpha_zero.hexcrystal.hex_encoder`` directly.
"""

from __future__ import annotations

import logging

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

_log = logging.getLogger(__name__)
_log.debug(
    "alpha_zero.core.hex_encoder is a backwards-compat shim; use alpha_zero.hexcrystal.hex_encoder"
)
