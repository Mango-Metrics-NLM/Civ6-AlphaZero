"""Backwards-compatibility shim: alpha_zero.core.hex_conv.

The implementation has moved to :mod:`alpha_zero.hexcrystal.hex_conv`.
This module re-exports the entire public surface so that existing callers
are unaffected by the refactoring.

.. deprecated::
    Import from ``alpha_zero.hexcrystal.hex_conv`` directly.
"""

from __future__ import annotations

import logging

from alpha_zero.hexcrystal.hex_conv import (
    HexAdjacency,
    HexConvBlock,
    HexConvConfig,
    HexConvolution,
)

__all__ = [
    "HexAdjacency",
    "HexConvBlock",
    "HexConvConfig",
    "HexConvolution",
]

_log = logging.getLogger(__name__)
_log.debug(
    "alpha_zero.core.hex_conv is a backwards-compat shim; use alpha_zero.hexcrystal.hex_conv"
)
