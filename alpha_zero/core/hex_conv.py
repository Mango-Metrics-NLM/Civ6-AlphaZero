"""Backwards-compatible re-exports for the hex convolution primitives.

The implementation moved to :mod:`alpha_zero.hexcrystal.hex_conv` so that it can
be imported without pulling in the rest of ``alpha_zero`` (and therefore numpy
and gymnasium). This module re-exports the public names so historical import
paths keep working; both paths resolve to the same objects.

.. deprecated:: 2026-09-02
   Emits :class:`DeprecationWarning` on import; scheduled for removal one
   release later. It has no in-repo consumer outside tests.
"""

from __future__ import annotations

import warnings

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

warnings.warn(
    "alpha_zero.core.hex_conv is a deprecated re-export shim; import from "
    "alpha_zero.hexcrystal.hex_conv instead. The shim will be removed one release after 2026-09-02.",
    DeprecationWarning,
    stacklevel=2,
)
