#!/usr/bin/env python3
"""Migrate models from 2-player (13 channels, scalar value) to N-player (28 channels, 6-dim value).

Sprint 6 Phase 2 Task 2.3: Model migration utility for N-player support.

This script:
1. Loads a checkpoint from 13-channel encoder
2. Pads spatial encoder input from 13→28 channels (zero-fill new channels)
3. Replicates value head output from 1→6 neurons (broadcast scalar to all players)
4. Saves migrated checkpoint with updated architecture metadata

Usage:
    # Migrate local model
    python scripts/migrate_model_to_nplayer.py \\
        --input models/v218.pt \\
        --output models/v218_migrated.pt

    # Migrate GCS model
    python scripts/migrate_model_to_nplayer.py \\
        --input gs://civ6az-training/models/v218.pt \\
        --output gs://civ6az-training/models/v218_migrated.pt

    # Dry-run (show changes without saving)
    python scripts/migrate_model_to_nplayer.py \\
        --input models/v218.pt \\
        --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_conv_layer_key(state_dict: dict[str, torch.Tensor]) -> str | None:
    """Find the first convolutional layer key in state_dict.

    Args:
        state_dict: Model state dictionary.

    Returns:
        Key for first conv layer, or None if not found.
    """
    # Look for common patterns: conv_in, conv1, spatial_encoder.conv_in, etc.
    candidates = [
        "spatial_encoder.conv_in.weight",
        "conv_in.weight",
        "conv1.weight",
        "encoder.conv_in.weight",
    ]

    for key in candidates:
        if key in state_dict:
            return key

    # Fallback 1: search for any key containing "conv" and "weight"
    for key in state_dict.keys():
        if "conv" in key.lower() and "weight" in key and state_dict[key].dim() == 4:
            # Check if it's the input layer (has channel dim of 13)
            if state_dict[key].shape[1] == 13:
                return key

    # Fallback 2: search for ANY 4D weight with 13 input channels
    for key in state_dict.keys():
        if "weight" in key and state_dict[key].dim() == 4:
            if state_dict[key].shape[1] == 13:
                return key

    return None


def find_value_head_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
    """Find value head layer keys in state_dict.

    Args:
        state_dict: Model state dictionary.

    Returns:
        Dict mapping layer type to key (e.g., {"output_weight": "value_head.fc2.weight"}).
    """
    result = {}

    # Common patterns for value head output layer
    candidates = [
        "value_head.fc2.weight",
        "value_head.fc2.bias",
        "value_head.output.weight",
        "value_head.output.bias",
        "value_head.linear.weight",
        "value_head.linear.bias",
    ]

    for key in candidates:
        if key in state_dict:
            if "weight" in key:
                result["output_weight"] = key
            elif "bias" in key:
                result["output_bias"] = key

    # Fallback: search for value_head with output size 1
    if not result:
        for key in state_dict.keys():
            if "value" in key and "weight" in key:
                tensor = state_dict[key]
                if tensor.dim() == 2 and tensor.shape[0] == 1:
                    result["output_weight"] = key
                    # Try to find corresponding bias
                    bias_key = key.replace(".weight", ".bias")
                    if bias_key in state_dict:
                        result["output_bias"] = bias_key
                elif tensor.dim() == 1 and tensor.shape[0] == 1:
                    result["output_bias"] = key

    return result


def migrate_spatial_encoder(
    state_dict: dict[str, torch.Tensor],
    conv_key: str,
) -> dict[str, torch.Tensor]:
    """Migrate spatial encoder from 13→28 channels.

    Args:
        state_dict: Model state dictionary.
        conv_key: Key for first conv layer.

    Returns:
        Updated state_dict with migrated spatial encoder.
    """
    logger.info(f"Migrating spatial encoder: {conv_key}")

    old_weight = state_dict[conv_key]
    old_shape = old_weight.shape
    logger.info(f"  Old weight shape: {old_shape}")

    # Verify old shape is (out_channels, 13, kernel_h, kernel_w)
    if old_shape[1] != 13:
        raise ValueError(
            f"Expected 13 input channels, got {old_shape[1]}. "
            f"Model may already be migrated or have unexpected architecture."
        )

    # Create new weight with 28 input channels
    new_shape = (old_shape[0], 28, old_shape[2], old_shape[3])
    new_weight = torch.zeros(new_shape, dtype=old_weight.dtype)

    # Copy old channels 0-12, leave 13-27 as zeros
    new_weight[:, :13, :, :] = old_weight

    state_dict[conv_key] = new_weight
    logger.info(f"  New weight shape: {new_shape}")
    logger.info(f"  Channels 0-12: copied from old model")
    logger.info(f"  Channels 13-27: zero-initialized (new per-player channels)")

    return state_dict


def migrate_value_head(
    state_dict: dict[str, torch.Tensor],
    value_keys: dict[str, str],
) -> dict[str, torch.Tensor]:
    """Migrate value head from scalar (1) to 6-dim output.

    Args:
        state_dict: Model state dictionary.
        value_keys: Dict mapping layer type to key.

    Returns:
        Updated state_dict with migrated value head.
    """
    logger.info("Migrating value head: scalar → 6-dim")

    # Migrate output weight
    if "output_weight" in value_keys:
        key = value_keys["output_weight"]
        old_weight = state_dict[key]
        old_shape = old_weight.shape
        logger.info(f"  Old weight shape: {old_shape}")

        # Verify old shape is (1, hidden_dim) or similar
        if old_shape[0] != 1:
            raise ValueError(
                f"Expected 1 output neuron, got {old_shape[0]}. "
                f"Model may already be migrated or have unexpected architecture."
            )

        # Replicate scalar output to 6 dimensions
        new_weight = old_weight.repeat(6, 1)
        state_dict[key] = new_weight
        logger.info(f"  New weight shape: {new_weight.shape}")
        logger.info(f"  Replicated scalar weights to all 6 players")

    # Migrate output bias
    if "output_bias" in value_keys:
        key = value_keys["output_bias"]
        old_bias = state_dict[key]
        old_shape = old_bias.shape
        logger.info(f"  Old bias shape: {old_shape}")

        if old_shape[0] != 1:
            raise ValueError(
                f"Expected 1 output bias, got {old_shape[0]}. "
                f"Model may already be migrated."
            )

        # Replicate scalar bias to 6 dimensions
        new_bias = old_bias.repeat(6)
        state_dict[key] = new_bias
        logger.info(f"  New bias shape: {new_bias.shape}")

    return state_dict


def update_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Update model_config metadata for N-player architecture.

    Args:
        checkpoint: Full checkpoint dict (includes model_state_dict, model_config, etc.).

    Returns:
        Updated checkpoint with modified model_config.
    """
    if "model_config" not in checkpoint:
        logger.warning("No model_config found in checkpoint, skipping metadata update")
        return checkpoint

    config = checkpoint["model_config"]

    # Update spatial channels
    old_channels = config.get("spatial_channels", 13)
    config["spatial_channels"] = 28
    logger.info(f"Updated model_config: spatial_channels {old_channels} → 28")

    # Note: value_head_dim stays at 64 (hidden layer), output neurons are 6
    # The output layer size is not typically in model_config, it's in the network structure

    checkpoint["model_config"] = config
    return checkpoint


def migrate_checkpoint(input_path: str, output_path: str | None, dry_run: bool = False) -> None:
    """Migrate a model checkpoint from 2-player to N-player architecture.

    Args:
        input_path: Path to input checkpoint (local or gs://).
        output_path: Path to save migrated checkpoint (local or gs://), or None for dry-run.
        dry_run: If True, show changes without saving.
    """
    logger.info(f"Loading checkpoint from: {input_path}")

    # Load checkpoint
    if input_path.startswith("gs://"):
        from alpha_zero.cloud.gcp.storage import GCSModelStorage

        bucket_name = input_path.split("/")[2]
        blob_path = "/".join(input_path.split("/")[3:])

        storage = GCSModelStorage(bucket_name=bucket_name)
        checkpoint = storage.backend.download_checkpoint(blob_path)
    else:
        checkpoint = torch.load(input_path, map_location="cpu")

    logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Extract state_dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Assume checkpoint is the state_dict itself
        state_dict = checkpoint

    logger.info(f"State dict has {len(state_dict)} parameters")

    # Find layers to migrate
    conv_key = find_conv_layer_key(state_dict)
    if conv_key is None:
        raise ValueError("Could not find spatial encoder conv layer in state_dict")

    value_keys = find_value_head_keys(state_dict)
    if not value_keys:
        raise ValueError("Could not find value head output layer in state_dict")

    logger.info(f"Found conv layer: {conv_key}")
    logger.info(f"Found value head layers: {value_keys}")

    if dry_run:
        logger.info("\n=== DRY RUN MODE ===")
        logger.info("Would migrate:")
        logger.info(f"  1. {conv_key}: 13 → 28 channels (pad with zeros)")
        for layer_type, key in value_keys.items():
            logger.info(f"  2. {key}: 1 → 6 outputs (replicate)")
        logger.info("No changes saved.")
        return

    # Perform migrations
    state_dict = migrate_spatial_encoder(state_dict, conv_key)
    state_dict = migrate_value_head(state_dict, value_keys)

    # Update checkpoint
    if "model_state_dict" in checkpoint:
        checkpoint["model_state_dict"] = state_dict
    elif "state_dict" in checkpoint:
        checkpoint["state_dict"] = state_dict
    else:
        checkpoint = state_dict

    # Update metadata if present
    if isinstance(checkpoint, dict):
        checkpoint = update_model_config(checkpoint)

        # Add migration marker
        checkpoint["migrated_to_nplayer"] = True
        checkpoint["migration_version"] = "1.5.0"

    # Save migrated checkpoint
    if output_path is None:
        logger.error("No output path specified and not in dry-run mode")
        return

    logger.info(f"Saving migrated checkpoint to: {output_path}")

    if output_path.startswith("gs://"):
        from alpha_zero.cloud.gcp.storage import GCSModelStorage

        bucket_name = output_path.split("/")[2]
        blob_path = "/".join(output_path.split("/")[3:])

        storage = GCSModelStorage(bucket_name=bucket_name)
        storage.backend.upload_checkpoint(checkpoint, blob_path)
    else:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_path)

    logger.info("Migration complete!")
    logger.info("\nNext steps:")
    logger.info("1. Test migrated model with: python -m alpha_zero.core.civ6_network (load test)")
    logger.info("2. Run 10-iteration fine-tuning to adapt to new architecture")
    logger.info("3. Evaluate migrated model vs. original baseline")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate model from 2-player to N-player architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input checkpoint path (local or gs://)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output checkpoint path (local or gs://). Required unless --dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without saving",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dry_run and not args.output:
        parser.error("--output is required unless --dry-run is specified")

    try:
        migrate_checkpoint(args.input, args.output, args.dry_run)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
