"""Backwards-compatibility shim: alpha_zero.core.encoder_registry.

The implementation has moved to :mod:`alpha_zero.hexcrystal.encoder_registry`.
This module re-exports the entire public surface so that existing callers
are unaffected by the refactoring.

.. deprecated::
    Import from ``alpha_zero.hexcrystal.encoder_registry`` directly.
"""

from __future__ import annotations

import logging

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

_log = logging.getLogger(__name__)
_log.debug(
    "alpha_zero.core.encoder_registry is a backwards-compat shim; "
    "use alpha_zero.hexcrystal.encoder_registry"
)
