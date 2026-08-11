"""Model card generation for Hugging Face Hub publishing.

Produces a Markdown model card from a :class:`ModelCardMetadata` dataclass
using :class:`string.Template` (stdlib — no Jinja2 dependency).  If a
template file is found at :data:`~alpha_zero.hub.constants.CARD_TEMPLATE_PATH`,
it is used; otherwise the built-in fallback template is used.

Example::

    from alpha_zero.hub.model_card import ModelCardMetadata, generate_model_card

    meta = ModelCardMetadata(
        model_name="civ6-alphazero-encoder",
        description="Hex-crystalline spatial encoder for Civ6 map boards.",
        tags=("hexagonal-grid", "spatial-encoder"),
    )
    card_text = generate_model_card(meta)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from alpha_zero.hub.constants import (
    CARD_TEMPLATE_PATH,
    DEFAULT_LICENSE,
    DEFAULT_MODEL_TAGS,
    LIBRARY_NAME,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ModelCardMetadata",
    "generate_model_card",
    "write_model_card",
]

# ---------------------------------------------------------------------------
# Fallback template — used when the file at CARD_TEMPLATE_PATH is missing.
# ---------------------------------------------------------------------------
_FALLBACK_TEMPLATE = """\
---
library_name: $library_name
tags:
$tags_yaml
license: $license
---

# $model_name

$description

## Model Details

- **Library**: $library_name
- **Architecture**: $architecture
- **Parameters**: $num_parameters

## Training

$training_details

## Metrics

$metrics

## Usage

```python
from $import_path import $class_name

model = $class_name.from_pretrained("$repo_id")
```

## Citation

$citation
"""


@dataclass
class ModelCardMetadata:
    """Metadata driving model-card generation.

    Every field has a sensible default so callers only need to fill in
    what they know.  Unknown fields render as ``"Not specified"`` in
    the generated card.

    Attributes:
        model_name: Human-readable model name.
        description: Short prose description.
        architecture: Architecture summary (e.g. "ResNet + Attention").
        num_parameters: Parameter count string (e.g. "2.3M").
        training_details: Markdown training description.
        metrics: Markdown metrics table or prose.
        tags: Extra tags appended to :data:`DEFAULT_MODEL_TAGS`.
        license: SPDX licence identifier.
        library_name: Library name for the Hub metadata.
        import_path: Python import path for ``from_pretrained`` example.
        class_name: Class name for the usage example.
        repo_id: Full ``org/repo`` Hub identifier.
        citation: BibTeX or prose citation block.
    """

    model_name: str = "alpha-zero-civ6-model"
    description: str = "An AlphaZero model for Civilization VI."
    architecture: str = "Not specified"
    num_parameters: str = "Not specified"
    training_details: str = "Not specified"
    metrics: str = "Not specified"
    tags: tuple[str, ...] = ()
    license: str = ""
    library_name: str = ""
    import_path: str = "alpha_zero"
    class_name: str = "Model"
    repo_id: str = ""
    citation: str = "Not specified"

    # Mutable defaults are deferred to __post_init__ so that env-var
    # overrides on LIBRARY_NAME and DEFAULT_LICENSE are captured at
    # instantiation time, not module-load time.
    _resolved_tags: tuple[str, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.license:
            self.license = DEFAULT_LICENSE
        if not self.library_name:
            self.library_name = LIBRARY_NAME
        # Merge default + caller tags, deduplicated and order-preserved.
        seen: set[str] = set()
        merged: list[str] = []
        for tag in (*DEFAULT_MODEL_TAGS, *self.tags):
            if tag not in seen:
                seen.add(tag)
                merged.append(tag)
        self._resolved_tags = tuple(merged)

    @property
    def all_tags(self) -> tuple[str, ...]:
        """All tags (default + model-specific), deduplicated."""
        return self._resolved_tags

    def tags_yaml(self) -> str:
        """Render tags as YAML list items for the card frontmatter."""
        return "\n".join(f"  - {tag}" for tag in self.all_tags)


def _load_template() -> Template:
    """Load the card template from disk or fall back to the built-in."""
    path = CARD_TEMPLATE_PATH
    if path.is_file():
        logger.debug("Using model-card template from %s", path)
        return Template(path.read_text(encoding="utf-8"))
    logger.debug("Template file not found at %s; using fallback", path)
    return Template(_FALLBACK_TEMPLATE)


def generate_model_card(metadata: ModelCardMetadata) -> str:
    """Generate a Markdown model card from *metadata*.

    Uses :data:`CARD_TEMPLATE_PATH` if the file exists, otherwise the
    built-in fallback template.  Template variables are ``$``-prefixed
    identifiers matching :class:`ModelCardMetadata` field names plus
    ``tags_yaml``.

    Args:
        metadata: The card metadata.

    Returns:
        The rendered Markdown string.
    """
    template = _load_template()
    substitutions = {
        "model_name": metadata.model_name,
        "description": metadata.description,
        "architecture": metadata.architecture,
        "num_parameters": metadata.num_parameters,
        "training_details": metadata.training_details,
        "metrics": metadata.metrics,
        "tags_yaml": metadata.tags_yaml(),
        "license": metadata.license,
        "library_name": metadata.library_name,
        "import_path": metadata.import_path,
        "class_name": metadata.class_name,
        "repo_id": metadata.repo_id,
        "citation": metadata.citation,
    }
    return template.safe_substitute(substitutions)


def write_model_card(
    metadata: ModelCardMetadata,
    output_path: Path,
) -> Path:
    """Generate a model card and write it to *output_path*.

    Creates parent directories if needed.

    Args:
        metadata: The card metadata.
        output_path: Destination file path.

    Returns:
        The resolved output path.
    """
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    card_text = generate_model_card(metadata)
    output_path.write_text(card_text, encoding="utf-8")
    logger.info("Model card written to %s", output_path)
    return output_path
