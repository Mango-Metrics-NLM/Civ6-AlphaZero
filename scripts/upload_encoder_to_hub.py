#!/usr/bin/env python3
"""Publish the hex-crystalline encoder to the Hugging Face Hub.

**This tool does not publish by default.** Uploading is a side-effecting call to
an external service, so it runs as a dry run unless ``--publish`` is passed
explicitly: it stages the artifacts locally, prints the manifest, and stops. Run
the dry run, read the manifest, then publish deliberately.

What gets published is the *encoder*, extracted from a trained network — not the
whole game-playing model. The encoder is the reusable piece: it depends only on
torch and applies to any hexagonal-grid problem, whereas the surrounding network
is specific to this game. Weights ship as safetensors; a pickled ``.pt`` is never
uploaded, because loading one executes arbitrary code on the consumer's machine.

The hex arm is published regardless of which encoder won the A/B. The repository
is about the hex-crystalline encoder, a negative result is still evidence, and
the extraction refuses a patch-embed checkpoint outright rather than silently
publishing the wrong architecture.

Credentials come from the ``HF_TOKEN`` environment variable only; there is no
token flag, so a token cannot end up in shell history or a process listing.

Example:
    $ python scripts/upload_encoder_to_hub.py \\
          --checkpoint checkpoints/encoder_ab/hex_crystalline.pt \\
          --repo-id your-name/civ6-hexcrystal-encoder            # dry run
    $ HF_TOKEN=... python scripts/upload_encoder_to_hub.py \\
          --checkpoint checkpoints/encoder_ab/hex_crystalline.pt \\
          --repo-id your-name/civ6-hexcrystal-encoder --publish
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("upload_encoder_to_hub")

#: Only the hex encoder is publishable through this tool.
REQUIRED_ENCODER_TYPE = "hex_crystalline"

#: Files the staged directory is expected to contain before upload.
EXPECTED_ARTIFACTS = ("model.safetensors", "config.json", "README.md")

DEFAULT_MODEL_CARD = Path("docs/model_cards/hexcrystal_encoder.md")
DEFAULT_EVIDENCE_DIR = Path("docs/evidence/hexcrystal_ab")


class UploadError(RuntimeError):
    """Raised when publication cannot proceed safely."""


def extract_encoder(checkpoint_path: Path, device: str = "cpu") -> Any:
    """Rebuild the network from a checkpoint and lift out its hex encoder.

    Args:
        checkpoint_path: A ``{state_dict, model_config}`` checkpoint.
        device: Device to map tensors onto.

    Returns:
        A ``HexSpatialEncoderForHub`` carrying the trained encoder weights.

    Raises:
        UploadError: If the checkpoint is not a hex-encoder checkpoint, or does
            not record its encoder type at all.
    """
    import torch

    from alpha_zero.core.transformer_network import (
        AttentionType,
        Civ6TransformerNetwork,
        NormType,
        TransformerConfig,
    )
    from alpha_zero.hexcrystal.hub_mixin import HexSpatialEncoderForHub

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_config" not in payload:
        raise UploadError(
            f"{checkpoint_path} carries no 'model_config', so its encoder type is "
            "unknowable. Publishing it risks shipping the wrong architecture."
        )

    config_dict = dict(payload["model_config"])
    encoder_type = config_dict.get("encoder_type", "patch_embed")
    if encoder_type != REQUIRED_ENCODER_TYPE:
        raise UploadError(
            f"{checkpoint_path} was trained with {encoder_type!r}, but this tool "
            f"publishes {REQUIRED_ENCODER_TYPE!r} only. Point it at the hex arm."
        )

    if isinstance(config_dict.get("norm_type"), str):
        config_dict["norm_type"] = NormType(config_dict["norm_type"])
    if isinstance(config_dict.get("attention_type"), str):
        config_dict["attention_type"] = AttentionType(config_dict["attention_type"])
    for derived in ("_num_patches", "_head_dim"):
        config_dict.pop(derived, None)

    network = Civ6TransformerNetwork(TransformerConfig(**config_dict))
    network.load_state_dict(payload["state_dict"], strict=True)

    inner = network.encoder.patch_embed
    hub_encoder = HexSpatialEncoderForHub.from_encoder(inner)

    logger.info(
        "Extracted encoder from %s: %d parameters",
        checkpoint_path,
        sum(p.numel() for p in hub_encoder.parameters()),
    )
    return hub_encoder


def stage_artifacts(
    encoder: Any,
    staging_dir: Path,
    model_card: Path | None,
    evidence_dir: Path | None,
) -> list[Path]:
    """Write the upload payload to *staging_dir*.

    ``save_pretrained`` emits a placeholder README, which is overwritten with the
    real model card when one is supplied — an unedited placeholder on a public
    model page is worse than no page.

    Args:
        encoder: The hub-enabled encoder to serialize.
        staging_dir: Destination directory; recreated if it already exists.
        model_card: Optional model card to publish as ``README.md``.
        evidence_dir: Optional directory whose A/B summary table is appended to
            the card, so the published numbers match the committed ones.

    Returns:
        The staged files, sorted by name.
    """
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    encoder.save_pretrained(staging_dir)

    if model_card is not None and model_card.exists():
        card_text = model_card.read_text()
        summary = (evidence_dir / "summary_table.md") if evidence_dir else None
        if summary is not None and summary.exists():
            card_text = card_text.replace("<!-- AB_SUMMARY_TABLE -->", summary.read_text())
        (staging_dir / "README.md").write_text(card_text)
        logger.info("Model card written from %s", model_card)
    else:
        logger.warning(
            "No model card at %s; the placeholder README from save_pretrained will be "
            "published as-is. Write a real card before making this public.",
            model_card,
        )

    return sorted(staging_dir.iterdir())


def verify_staged(staging_dir: Path) -> None:
    """Check the staged payload before anything leaves the machine.

    Args:
        staging_dir: Directory produced by :func:`stage_artifacts`.

    Raises:
        UploadError: If an expected artifact is missing, or a pickled weight
            file is present.
    """
    present = {p.name for p in staging_dir.iterdir()}

    missing = [name for name in EXPECTED_ARTIFACTS if name not in present]
    if missing:
        raise UploadError(f"Staged payload is missing {missing}; refusing to upload.")

    pickled = sorted(n for n in present if n.endswith((".pt", ".pth", ".bin", ".pkl")))
    if pickled:
        raise UploadError(
            f"Refusing to publish pickled weights {pickled}: loading them executes "
            "arbitrary code. Publish safetensors only."
        )

    config = json.loads((staging_dir / "config.json").read_text())
    if not config:
        raise UploadError("config.json is empty; the encoder could not be rebuilt from it.")

    logger.info("Staged payload verified: %s", ", ".join(sorted(present)))


def publish(
    staging_dir: Path,
    repo_id: str,
    token: str,
    private: bool,
    api: Any | None = None,
) -> str:
    """Upload the staged payload to the Hub.

    Idempotent: the repository is created if absent and updated in place
    otherwise, so re-running replaces the artifacts rather than failing.

    Args:
        staging_dir: Verified payload directory.
        repo_id: Target repository, ``owner/name``.
        token: Hugging Face access token.
        private: Whether to create the repository as private.
        api: Optional pre-built API client, primarily for testing.

    Returns:
        The repository URL.
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(staging_dir))

    url = f"https://huggingface.co/{repo_id}"
    logger.info("Published %s", url)
    return url


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="upload_encoder_to_hub.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint of the hex arm (a patch_embed checkpoint is rejected)",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target repository as owner/name, e.g. your-name/civ6-hexcrystal-encoder",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Actually upload. Without this the tool stages locally and stops.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private (applies only on first creation)",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("build/hub_staging"),
        help="Where to assemble the payload (default: build/hub_staging)",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=DEFAULT_MODEL_CARD,
        help=f"Model card published as README.md (default: {DEFAULT_MODEL_CARD})",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE_DIR,
        help=f"A/B evidence directory (default: {DEFAULT_EVIDENCE_DIR})",
    )
    parser.add_argument("--device", default="cpu", help="Device for loading (default: cpu)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser


def main(argv: list[str] | None = None, api: Any | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.
        api: Optional Hub API client injected for testing.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        return 1

    # Fail before doing any work if publication is requested without a token.
    token = os.environ.get("HF_TOKEN", "")
    if args.publish and not token:
        logger.error(
            "--publish requires a token in the HF_TOKEN environment variable. Nothing was uploaded."
        )
        return 1

    try:
        encoder = extract_encoder(args.checkpoint, args.device)
        staged = stage_artifacts(encoder, args.staging_dir, args.model_card, args.evidence_dir)
        verify_staged(args.staging_dir)
    except UploadError as exc:
        logger.error("%s", exc)
        return 1

    print(f"\nStaged in {args.staging_dir}:")
    for path in staged:
        print(f"  {path.name:24s} {path.stat().st_size:>10,d} bytes")

    if not args.publish:
        print(
            "\nDry run — nothing was uploaded.\n"
            f"To publish:  HF_TOKEN=... {Path(sys.argv[0]).name} "
            f"--checkpoint {args.checkpoint} --repo-id {args.repo_id} --publish\n"
        )
        return 0

    try:
        url = publish(args.staging_dir, args.repo_id, token, args.private, api=api)
    except Exception as exc:  # Surface any Hub failure as a clean exit 1
        logger.error("Upload failed: %s", exc)
        return 1

    print(f"\nPublished: {url}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
