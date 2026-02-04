#!/usr/bin/env python3
"""
Upload Initial Model to GCS for Civ6 AlphaZero Training.

This script creates and uploads an initial (random) model to GCS,
which serves as the starting point for self-play training.

Usage:
    # Set environment variables first
    export CIV6AZ_VERTEX_PROJECT=your-project-id
    export CIV6AZ_GCS_BUCKET=your-bucket-name

    # Run the script
    python scripts/upload_initial_model.py

    # Or with custom parameters
    python scripts/upload_initial_model.py --version 0 --board-size 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_initial_model(
    state_channels: int = 64,
    board_size: int = 10,
    num_actions: int = 100,
) -> tuple[dict, dict]:
    """Create an initial random model.

    Args:
        state_channels: Number of state channels.
        board_size: Board size (height and width).
        num_actions: Action space size.

    Returns:
        Tuple of (state_dict, metadata).
    """
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    model_config = TransformerConfig(
        spatial_channels=state_channels,
        spatial_size=board_size,
        vector_dim=state_channels,
        action_space_size=num_actions,
    )
    model = Civ6TransformerNetwork(config=model_config)

    # Initialize weights (already done by PyTorch, but we can customize)
    # Using Xavier initialization for better training dynamics
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            torch.nn.init.xavier_uniform_(param)
        elif "bias" in name:
            torch.nn.init.zeros_(param)

    metadata = {
        "version": 0,
        "initial": True,
        "description": "Initial random model for self-play training",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config": {
            "state_channels": state_channels,
            "board_size": board_size,
            "num_actions": num_actions,
        },
        "elo": 0,
    }

    return model.state_dict(), metadata


def upload_model(
    state_dict: dict,
    metadata: dict,
    version: int = 0,
) -> str:
    """Upload model to GCS.

    Args:
        state_dict: Model state dict.
        metadata: Model metadata.
        version: Model version number.

    Returns:
        GCS URI of uploaded model.
    """
    from alpha_zero.cloud.gcp.config import GCPConfig
    from alpha_zero.cloud.gcp.storage import GCSModelStorage

    config = GCPConfig.from_env()
    config.validate()

    logger.info("Uploading to bucket: gs://%s", config.bucket)

    storage = GCSModelStorage(config)
    uri = storage.save_model(state_dict, version=version, metadata=metadata)

    return uri


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload initial model to GCS for Civ6 AlphaZero training"
    )
    parser.add_argument(
        "--version",
        type=int,
        default=0,
        help="Model version number (default: 0)",
    )
    parser.add_argument(
        "--state-channels",
        type=int,
        default=64,
        help="Number of state channels (default: 64)",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=10,
        help="Board size (default: 10)",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        default=100,
        help="Action space size (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create model but don't upload",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Initial Model Upload for Civ6 AlphaZero")
    print("=" * 50)
    print()

    # Create the model
    print("Creating initial model...")
    print(f"  State channels: {args.state_channels}")
    print(f"  Board size:     {args.board_size}x{args.board_size}")
    print(f"  Action space:   {args.num_actions}")

    try:
        state_dict, metadata = create_initial_model(
            state_channels=args.state_channels,
            board_size=args.board_size,
            num_actions=args.num_actions,
        )
    except ImportError as e:
        print(f"\nERROR: Missing dependency - {e}")
        print("Install with: pip install -e .")
        return 1

    # Count parameters
    num_params = sum(p.numel() for p in state_dict.values())
    print(f"  Parameters:     {num_params:,}")

    if args.dry_run:
        print("\n[DRY RUN] Model created but not uploaded")
        print(f"  Version: {args.version}")
        print(f"  Metadata: {metadata}")
        return 0

    # Upload to GCS
    print(f"\nUploading as version {args.version}...")

    try:
        uri = upload_model(state_dict, metadata, args.version)
        print("\nSuccess! Model uploaded to:")
        print(f"  {uri}")
    except Exception as e:
        print(f"\nERROR: Upload failed - {e}")
        return 1

    print()
    print("Next steps:")
    print("  1. Build container: gcloud builds submit --config=dockerfiles/cloudbuild.yaml")
    print("  2. Start training:  python -m alpha_zero.cloud.gcp run --iterations 50")

    return 0


if __name__ == "__main__":
    sys.exit(main())
