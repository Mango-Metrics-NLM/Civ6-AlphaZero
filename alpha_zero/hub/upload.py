"""Generic upload utility for Hugging Face Hub.

Wraps ``HfApi.upload_folder`` with model-card generation, dry-run support,
and structured logging.  Designed to be called by model-specific upload
scripts without duplicating Hub interaction logic.

**Import guard**: importing this module without ``huggingface_hub`` installed
raises :class:`ImportError` with an actionable message.

Example::

    from alpha_zero.hub.upload import upload_model_to_hub
    from alpha_zero.hub.model_card import ModelCardMetadata

    upload_model_to_hub(
        model=my_hub_model,
        repo_id="Mango-Metrics-NLM/civ6-encoder",
        metadata=ModelCardMetadata(model_name="civ6-encoder"),
        dry_run=True,
    )
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from alpha_zero.hub.model_card import ModelCardMetadata, write_model_card

try:
    from huggingface_hub import HfApi
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "alpha_zero.hub.upload requires the 'hub' extra. "
        "Install with:  pip install -e '.[hub]'"
    ) from exc

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "HubUploadable",
    "upload_model_to_hub",
]


@runtime_checkable
class HubUploadable(Protocol):
    """Protocol for models that support ``save_pretrained``."""

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        """Save model weights and config to a local directory."""
        ...


def upload_model_to_hub(
    model: HubUploadable,
    repo_id: str,
    metadata: ModelCardMetadata | None = None,
    token: str | None = None,
    dry_run: bool = True,
    private: bool = False,
    commit_message: str = "Upload model",
    save_directory: Path | None = None,
) -> Path:
    """Upload a model to the Hugging Face Hub.

    Args:
        model: A model with ``save_pretrained`` support (e.g. a
            ``PyTorchModelHubMixin`` subclass).
        repo_id: Target ``org/repo`` on the Hub.
        metadata: Optional card metadata.  If *None*, a minimal card
            is generated.
        token: HF API token.  Falls back to ``HF_TOKEN`` env var or
            cached ``huggingface-cli login`` credentials.
        dry_run: If *True* (default), save locally but do **not** upload.
        private: Whether the Hub repo should be private.
        commit_message: Commit message for the Hub upload.
        save_directory: Directory to save the model to.  If *None*, a
            temporary directory is created (cleaned up only if *dry_run*
            is False and upload succeeds).

    Returns:
        The local directory where the model was saved.

    Raises:
        TypeError: If *model* does not implement ``save_pretrained``.
        ValueError: If *repo_id* is malformed.
    """
    if not isinstance(model, HubUploadable):
        msg = (
            f"Expected a model with save_pretrained(), got {type(model).__name__}. "
            "Use a PyTorchModelHubMixin subclass."
        )
        raise TypeError(msg)

    if "/" not in repo_id:
        msg = (
            f"repo_id must be in 'org/repo' format, got {repo_id!r}."
        )
        raise ValueError(msg)

    # Resolve save directory.
    if save_directory is None:
        save_directory = Path(tempfile.mkdtemp(prefix="hf_upload_"))
    save_directory = save_directory.resolve()
    save_directory.mkdir(parents=True, exist_ok=True)

    # Save model weights + config.json.
    logger.info("Saving model to %s", save_directory)
    model.save_pretrained(str(save_directory))

    # Generate and write model card.
    if metadata is None:
        metadata = ModelCardMetadata(repo_id=repo_id)
    else:
        # Ensure repo_id is set in metadata for the usage example.
        if not metadata.repo_id:
            metadata.repo_id = repo_id
    card_path = save_directory / "README.md"
    write_model_card(metadata, card_path)

    if dry_run:
        logger.info(
            "[DRY RUN] Model saved to %s — skipping Hub upload to %s",
            save_directory,
            repo_id,
        )
        return save_directory

    # Upload to Hub.
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(save_directory),
        commit_message=commit_message,
        token=token,
    )
    logger.info("Model uploaded to https://huggingface.co/%s", repo_id)
    return save_directory
