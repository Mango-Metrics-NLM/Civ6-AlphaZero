"""Backwards-compatible re-exports for the spatial encoder registry.

The implementation moved to :mod:`alpha_zero.hexcrystal.encoder_registry` so that
it can be imported without pulling in the rest of ``alpha_zero`` (and therefore
numpy and gymnasium). This module re-exports the public names so historical
import paths keep working.

The registry itself is a single module-level dict living in the new location, so
an encoder registered through either import path is visible through both.
"""

from __future__ import annotations

from alpha_zero.hexcrystal.encoder_registry import (
    EncoderFactory,
    EncoderType,
    create_spatial_encoder,
    get_registered_encoders,
    register_encoder,
)

__all__ = [
    "EncoderFactory",
    "EncoderType",
    "create_spatial_encoder",
    "get_registered_encoders",
    "register_encoder",
]
