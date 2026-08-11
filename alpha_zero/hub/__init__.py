"""Shared Hugging Face Hub infrastructure for model publishing.

This package provides reusable utilities for publishing models, datasets,
and assets to the Hugging Face Hub. Individual model hub mixins live
co-located with their models (e.g. ``alpha_zero.hexcrystal.hub_mixin``);
this package provides the *shared* plumbing they all use.

**Dependency contract.** This package's public API is importable without
the ``hub`` extra installed — the constants and dataclasses are pure Python.
Only functions that actually *call* the Hub API require ``huggingface_hub``
at runtime and raise ``ImportError`` with a helpful message if it is absent.
"""

from __future__ import annotations

from alpha_zero.hub.constants import (
    DEFAULT_MODEL_TAGS,
    HF_ORG,
    LIBRARY_NAME,
)
from alpha_zero.hub.model_card import (
    ModelCardMetadata,
    generate_model_card,
    write_model_card,
)

__all__ = [
    "DEFAULT_MODEL_TAGS",
    "HF_ORG",
    "LIBRARY_NAME",
    "ModelCardMetadata",
    "generate_model_card",
    "write_model_card",
]
