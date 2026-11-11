"""Shared constants for Hugging Face Hub publishing.

All values are environment-overridable so that CI, staging, and production
can point at different organisations or naming schemes without code changes.

Example::

    HF_ORG=my-org python scripts/upload_to_hub.py --model-type encoder
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "CARD_TEMPLATE_PATH",
    "DEFAULT_DATASET_DESCRIPTION",
    "DEFAULT_LICENSE",
    "DEFAULT_MODEL_TAGS",
    "HF_ORG",
    "LIBRARY_NAME",
]

#: Hugging Face organisation / user that owns the published repos.
HF_ORG: str = os.environ.get("HF_ORG", "ianshank")

#: Library name written into model cards and Hub metadata.
LIBRARY_NAME: str = os.environ.get("HF_LIBRARY_NAME", "alpha-zero-civ6")

#: Tags applied to every published model.  Model-specific tags are appended
#: by each hub mixin.
DEFAULT_MODEL_TAGS: tuple[str, ...] = (
    "pytorch",
    "reinforcement-learning",
    "game-ai",
    "civilization-vi",
)

#: SPDX licence identifier used in model cards.
DEFAULT_LICENSE: str = os.environ.get("HF_LICENSE", "mit")

#: Path to the Markdown model-card template.
CARD_TEMPLATE_PATH: Path = Path(
    os.environ.get(
        "HF_CARD_TEMPLATE",
        str(Path(__file__).parent / "templates" / "model_card.md.template"),
    )
)
#: Default description for published datasets.
DEFAULT_DATASET_DESCRIPTION: str = os.environ.get(
    "HF_DATASET_DESCRIPTION", "Civ6 AlphaZero dataset"
)
