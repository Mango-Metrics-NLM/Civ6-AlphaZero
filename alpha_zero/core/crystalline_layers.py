"""Backwards-compatible re-exports for the crystalline lattice layer.

The implementation moved to :mod:`alpha_zero.hexcrystal.crystalline_layers` so
that it can be imported without pulling in the rest of ``alpha_zero`` (and
therefore numpy and gymnasium). This module re-exports the public names so
historical import paths keep working; both paths resolve to the same objects.
"""

from __future__ import annotations

import warnings

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

warnings.warn(
    "alpha_zero.core.crystalline_layers is a deprecated re-export shim; import from "
    "alpha_zero.hexcrystal.crystalline_layers instead. The shim will be removed one release after 2026-09-02.",
    DeprecationWarning,
    stacklevel=2,
)
