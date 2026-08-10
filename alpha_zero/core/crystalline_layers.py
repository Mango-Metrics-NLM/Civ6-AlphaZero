"""Backwards-compatibility shim: alpha_zero.core.crystalline_layers.

The implementation has moved to :mod:`alpha_zero.hexcrystal.crystalline_layers`.
This module re-exports the entire public surface so that existing callers
are unaffected by the refactoring.

.. deprecated::
    Import from ``alpha_zero.hexcrystal.crystalline_layers`` directly.
"""

from __future__ import annotations

import logging

from alpha_zero.hexcrystal.crystalline_layers import (
    CrystallineLatticeConfig,
    CrystallineLatticeLayer,
    SymmetryType,
)

__all__ = [
    "CrystallineLatticeConfig",
    "CrystallineLatticeLayer",
    "SymmetryType",
]

_log = logging.getLogger(__name__)
_log.debug(
    "alpha_zero.core.crystalline_layers is a backwards-compat shim; "
    "use alpha_zero.hexcrystal.crystalline_layers"
)
