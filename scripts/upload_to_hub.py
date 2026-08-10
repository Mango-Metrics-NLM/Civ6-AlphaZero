#!/usr/bin/env python3
"""Upload any supported model to the Hugging Face Hub.

Supports all model types via a registry — no if/elif chains.
**Dry-run by default**: pass ``--publish`` to actually upload.

Usage::

    # Dry-run (default) — saves locally and prints what would be uploaded
    python scripts/upload_to_hub.py \\
        --model-type encoder \\
        --checkpoint-path ./checkpoints/civ6/best.pt

    # Publish to Hub
    python scripts/upload_to_hub.py \\
        --model-type transformer \\
        --checkpoint-path ./checkpoints/civ6/best.pt \\
        --repo-id Mango-Metrics-NLM/civ6-alphazero-transformer \\
        --publish

Environment Variables:
    HF_TOKEN: Hugging Face API token (or use ``huggingface-cli login``).
    HF_ORG: Override the default organisation name.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from alpha_zero.hub.constants import DEFAULT_MODEL_TAGS, HF_ORG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model type registry — maps CLI names to (module_path, class_name, config_class)
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "encoder": {
        "module": "alpha_zero.hexcrystal.hub_mixin",
        "class": "HexSpatialEncoderForHub",
        "description": "Hex-crystalline spatial encoder",
    },
    "transformer": {
        "module": "alpha_zero.core.transformer_hub_mixin",
        "class": "Civ6TransformerNetworkForHub",
        "description": "Civ6 transformer policy/value network",
    },
    "muzero": {
        "module": "alpha_zero.core.muzero_hub_mixin",
        "class": "MuZeroNetworkForHub",
        "description": "MuZero world-model network",
    },
}


def _import_model_class(model_type: str) -> type:
    """Dynamically import the hub-enabled model class from the registry.

    Args:
        model_type: Key in ``_MODEL_REGISTRY``.

    Returns:
        The hub-enabled model class.

    Raises:
        KeyError: If *model_type* is not registered.
        ImportError: If the hub extra is not installed.
    """
    if model_type not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        msg = f"Unknown model type {model_type!r}. Available: {available}"
        raise KeyError(msg)

    entry = _MODEL_REGISTRY[model_type]
    import importlib

    module = importlib.import_module(entry["module"])
    return getattr(module, entry["class"])  # type: ignore[no-any-return]


def _load_checkpoint(
    checkpoint_path: Path, model_class: type, device: str = "cpu",
) -> Any:
    """Load a checkpoint and wrap it in the hub-enabled model class.

    The checkpoint can be either:
    1. A raw ``state_dict`` — loaded into a fresh model with default config.
    2. A dict with ``"state_dict"`` and optional ``"config"`` keys.

    Args:
        checkpoint_path: Path to the ``.pt`` file.
        model_class: The hub-enabled model class.
        device: Device to load onto.

    Returns:
        An instance of *model_class* with loaded weights.
    """
    logger.info("Loading checkpoint from %s", checkpoint_path)
    data = torch.load(
        str(checkpoint_path), map_location=device, weights_only=False,
    )

    # Detect format.
    if isinstance(data, dict) and "state_dict" in data:
        state_dict = data["state_dict"]
        config_dict = data.get("config", {})
    else:
        # Assume raw state_dict.
        state_dict = data
        config_dict = {}

    # Build the model.
    if config_dict:
        model = model_class(**config_dict)
    else:
        model = model_class()

    model.load_state_dict(state_dict, strict=False)
    logger.info(
        "Loaded %d parameters from checkpoint",
        sum(p.numel() for p in model.parameters()),
    )
    return model


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Upload a model to the Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        required=True,
        choices=sorted(_MODEL_REGISTRY),
        help="Model type to upload",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to a .pt checkpoint file (optional: uses default config if omitted)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hub repo in 'org/repo' format (default: $HF_ORG/civ6-$MODEL_TYPE)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local directory to save model (default: temp directory)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        default=False,
        help="Actually upload to Hub (default: dry-run only)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create a private Hub repo",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload model via upload_to_hub.py",
        help="Commit message for the Hub upload",
    )

    args = parser.parse_args()

    # Resolve repo ID — uses HF_ORG from constants (env-overridable).
    repo_id = args.repo_id or f"{HF_ORG}/civ6-{args.model_type}"

    print("=" * 60)
    print(f"  Model Upload: {_MODEL_REGISTRY[args.model_type]['description']}")
    print("=" * 60)
    print(f"  Model type:     {args.model_type}")
    print(f"  Checkpoint:     {args.checkpoint_path or '(default config)'}")
    print(f"  Repo ID:        {repo_id}")
    print(f"  Publish:        {args.publish}")
    print()

    try:
        model_class = _import_model_class(args.model_type)
    except ImportError as e:
        print(f"\nERROR: {e}")
        return 1

    # Load or create model.
    if args.checkpoint_path:
        if not args.checkpoint_path.exists():
            print(f"\nERROR: Checkpoint not found: {args.checkpoint_path}")
            return 1
        model = _load_checkpoint(args.checkpoint_path, model_class)
    else:
        logger.info("No checkpoint provided — using default config")
        model = model_class()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:     {num_params:,}")

    # Import upload utility (requires hub extra).
    try:
        from alpha_zero.hub.model_card import ModelCardMetadata
        from alpha_zero.hub.upload import upload_model_to_hub
    except ImportError as e:
        print(f"\nERROR: {e}")
        return 1

    entry = _MODEL_REGISTRY[args.model_type]
    metadata = ModelCardMetadata(
        model_name=f"civ6-{args.model_type}",
        description=entry["description"],
        architecture=entry["class"],
        num_parameters=f"{num_params:,}",
        import_path=entry["module"],
        class_name=entry["class"],
        repo_id=repo_id,
        # Only add tags not already in DEFAULT_MODEL_TAGS to avoid duplication.
        tags=tuple(
            tag for tag in (args.model_type, "civ6")
            if tag not in DEFAULT_MODEL_TAGS
        ),
    )

    save_dir = upload_model_to_hub(
        model=model,
        repo_id=repo_id,
        metadata=metadata,
        dry_run=not args.publish,
        private=args.private,
        commit_message=args.commit_message,
        save_directory=args.output_dir,
    )

    print(f"\n  Saved to:       {save_dir}")
    if args.publish:
        print(f"  Published to:   https://huggingface.co/{repo_id}")
    else:
        print("  [DRY RUN] Use --publish to upload to Hub")

    return 0


if __name__ == "__main__":
    sys.exit(main())
