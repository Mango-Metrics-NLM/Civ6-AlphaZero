"""Backwards-compatible re-exports for the crystalline lattice layer.

The implementation moved to :mod:`alpha_zero.hexcrystal.crystalline_layers` so
that it can be imported without pulling in the rest of ``alpha_zero`` (and
therefore numpy and gymnasium). This module re-exports the public names so
historical import paths keep working; both paths resolve to the same objects.
"""

from __future__ import annotations

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
