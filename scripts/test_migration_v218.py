#!/usr/bin/env python3
"""End-to-end test for model migration on v218.

This script validates that the migration utility works correctly on production
model v218, testing the full pipeline from checkpoint migration to gameplay.

Usage:
    # Full test (downloads from GCS, migrates, validates)
    python scripts/test_migration_v218.py

    # Test with local checkpoint
    python scripts/test_migration_v218.py --input /tmp/v218.pt

    # Skip download (use existing local file)
    python scripts/test_migration_v218.py --skip-download

Sprint 6 Task A: Migration Utility End-to-End Testing
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def download_v218_from_gcs(output_path: str) -> bool:
    """Download v218 model.pt from GCS bucket.

    Args:
        output_path: Local path to save checkpoint.

    Returns:
        True if download successful, False otherwise.
    """
    try:
        import platform
        import subprocess

        # Use platform-specific gsutil path
        if platform.system() == "Windows":
            gsutil_cmd = r"C:\Users\iansh\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gsutil.cmd"
        else:
            gsutil_cmd = "gsutil"

        # Updated GCS path: models are in versioned subdirectories
        gcs_path = "gs://civ6az-training/models/civ6az/v218/model.pt"

        logger.info(f"Downloading v218 from {gcs_path} to {output_path}...")
        result = subprocess.run(
            [gsutil_cmd, "cp", gcs_path, output_path],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(f"GCS download failed: {result.stderr}")
            return False

        logger.info(f"✅ Downloaded v218 ({Path(output_path).stat().st_size / 1e6:.1f} MB)")
        return True

    except FileNotFoundError:
        logger.error("gsutil not found. Install Google Cloud SDK or use --input flag")
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def validate_checkpoint_loads(checkpoint_path: str) -> bool:
    """Validate that migrated checkpoint loads correctly.

    Args:
        checkpoint_path: Path to migrated checkpoint.

    Returns:
        True if validation passes, False otherwise.
    """
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    logger.info(f"Loading checkpoint from {checkpoint_path}...")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Check migration flag
        if not checkpoint.get("migrated_to_nplayer"):
            logger.error("❌ Checkpoint missing 'migrated_to_nplayer' flag")
            return False

        logger.info(f"✅ Migration flag: {checkpoint['migrated_to_nplayer']}")

        # Check model config
        if "model_config" not in checkpoint:
            logger.error("❌ No model_config found in checkpoint")
            return False

        model_config = checkpoint["model_config"]
        spatial_channels = model_config.get("spatial_channels", 13)
        logger.info(f"   Spatial channels: {spatial_channels}")

        if spatial_channels != 28:
            logger.error(f"❌ Expected 28 channels, got {spatial_channels}")
            return False

        # Extract state_dict (handle both naming conventions)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            logger.error("❌ No state_dict found in checkpoint")
            return False

        # Create network using model_config
        logger.info("Creating Civ6TransformerNetwork with migrated config...")

        from alpha_zero.core.transformer_network import AttentionType, NormType

        model_config_copy = model_config.copy()

        # Convert string enums back to Enum types (matching load() pattern)
        if "norm_type" in model_config_copy and isinstance(model_config_copy["norm_type"], str):
            model_config_copy["norm_type"] = NormType(model_config_copy["norm_type"])
        if "attention_type" in model_config_copy and isinstance(model_config_copy["attention_type"], str):
            model_config_copy["attention_type"] = AttentionType(model_config_copy["attention_type"])

        # Remove computed fields
        model_config_copy.pop("_num_patches", None)
        model_config_copy.pop("_head_dim", None)

        config = TransformerConfig(**model_config_copy)
        network = Civ6TransformerNetwork(config=config)

        # Load state dict
        logger.info("Loading migrated state dict...")
        network.load_state_dict(state_dict)

        # Test forward pass
        logger.info("Testing forward pass...")
        batch_size = 4
        spatial_size = config.spatial_size
        spatial = torch.randn(batch_size, spatial_channels, spatial_size, spatial_size)
        vector = torch.randn(batch_size, config.vector_dim)

        with torch.no_grad():
            policy, value = network(spatial, vector)

        # Validate output shapes
        expected_policy_shape = (batch_size, config.action_space_size)
        expected_value_shape = (batch_size, 6)

        if policy.shape != expected_policy_shape:
            logger.error(f"❌ Policy shape: expected {expected_policy_shape}, got {policy.shape}")
            return False

        if value.shape != expected_value_shape:
            logger.error(f"❌ Value shape: expected {expected_value_shape}, got {value.shape}")
            return False

        logger.info(f"✅ Forward pass successful")
        logger.info(f"   Policy shape: {policy.shape}")
        logger.info(f"   Value shape: {value.shape}")

        return True

    except Exception as e:
        logger.error(f"❌ Checkpoint validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2player_game(checkpoint_path: str) -> bool:
    """Run a quick 2-player game to test backward compatibility.

    Args:
        checkpoint_path: Path to migrated checkpoint.

    Returns:
        True if game runs successfully, False otherwise.
    """
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig
    from alpha_zero.envs.microciv import MicroCivEnv

    logger.info("Testing 2-player game (backward compatibility)...")

    try:
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract state_dict and config
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            raise KeyError("No state_dict found in checkpoint")

        from alpha_zero.core.transformer_network import AttentionType, NormType

        model_config = checkpoint["model_config"].copy()

        # Convert string enums back to Enum types (matching load() pattern)
        if "norm_type" in model_config and isinstance(model_config["norm_type"], str):
            model_config["norm_type"] = NormType(model_config["norm_type"])
        if "attention_type" in model_config and isinstance(model_config["attention_type"], str):
            model_config["attention_type"] = AttentionType(model_config["attention_type"])

        # Remove computed fields
        model_config.pop("_num_patches", None)
        model_config.pop("_head_dim", None)

        config = TransformerConfig(**model_config)

        network = Civ6TransformerNetwork(config=config)
        network.load_state_dict(state_dict)
        network.eval()

        # Create 2-player environment
        env = MicroCivEnv(num_players=2, board_size=8)
        obs = env.reset()

        # Play short game (max 30 turns)
        turns_played = 0
        for _ in range(30):
            if env.done:
                break

            # Get observation
            spatial = torch.tensor(obs["spatial"], dtype=torch.float32).unsqueeze(0)
            vector = torch.tensor(obs["vector"], dtype=torch.float32).unsqueeze(0)

            # Get action from network
            with torch.no_grad():
                policy, value = network(spatial, vector)

            # Select action (simple argmax for testing)
            action = policy.argmax(dim=1).item()

            # Step environment
            obs, reward, done, info = env.step(action)
            turns_played += 1

        logger.info(f"✅ 2-player game completed ({turns_played} turns)")
        logger.info(f"   Winner: Player {env.winner if env.done else 'N/A (ongoing)'}")

        return True

    except Exception as e:
        logger.error(f"❌ 2-player game failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4player_game(checkpoint_path: str) -> bool:
    """Run a quick 4-player game to test N-player support.

    Args:
        checkpoint_path: Path to migrated checkpoint.

    Returns:
        True if game runs successfully, False otherwise.
    """
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig
    from alpha_zero.envs.microciv import MicroCivEnv

    logger.info("Testing 4-player game (N-player validation)...")

    try:
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract state_dict and config
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            raise KeyError("No state_dict found in checkpoint")

        from alpha_zero.core.transformer_network import AttentionType, NormType

        model_config = checkpoint["model_config"].copy()

        # Convert string enums back to Enum types (matching load() pattern)
        if "norm_type" in model_config and isinstance(model_config["norm_type"], str):
            model_config["norm_type"] = NormType(model_config["norm_type"])
        if "attention_type" in model_config and isinstance(model_config["attention_type"], str):
            model_config["attention_type"] = AttentionType(model_config["attention_type"])

        # Remove computed fields
        model_config.pop("_num_patches", None)
        model_config.pop("_head_dim", None)

        config = TransformerConfig(**model_config)

        network = Civ6TransformerNetwork(config=config)
        network.load_state_dict(state_dict)
        network.eval()

        # Create 4-player environment
        env = MicroCivEnv(num_players=4, board_size=8)
        obs = env.reset()

        # Play short game (max 50 turns)
        turns_played = 0
        for _ in range(50):
            if env.done:
                break

            # Get observation
            spatial = torch.tensor(obs["spatial"], dtype=torch.float32).unsqueeze(0)
            vector = torch.tensor(obs["vector"], dtype=torch.float32).unsqueeze(0)

            # Get action from network
            with torch.no_grad():
                policy, value = network(spatial, vector)

            # Verify value shape is 6-dim
            assert value.shape == (1, 6), f"Expected (1, 6), got {value.shape}"

            # Select action
            action = policy.argmax(dim=1).item()

            # Step environment
            obs, reward, done, info = env.step(action)
            turns_played += 1

        logger.info(f"✅ 4-player game completed ({turns_played} turns)")
        logger.info(f"   Winner: Player {env.winner if env.done else 'N/A (ongoing)'}")

        return True

    except Exception as e:
        logger.error(f"❌ 4-player game failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """Main entry point for migration testing."""
    parser = argparse.ArgumentParser(
        description="End-to-end test for model migration on v218",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to v218.pt checkpoint (local or GCS). If not provided, downloads from GCS.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip GCS download, use existing file at /tmp/v218.pt",
    )
    parser.add_argument(
        "--skip-games",
        action="store_true",
        help="Skip gameplay tests (faster for just checking migration)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Migration Test: v218 → N-Player (28ch, 6-dim value)")
    logger.info("=" * 60)

    # Step 1: Get input checkpoint
    if args.skip_download:
        input_path = "/tmp/v218.pt"
        if not Path(input_path).exists():
            logger.error(f"❌ File not found: {input_path}")
            return 1
        logger.info(f"Using existing checkpoint: {input_path}")
    elif args.input:
        if args.input.startswith("gs://"):
            input_path = "/tmp/v218.pt"
            if not download_v218_from_gcs(input_path):
                return 1
        else:
            input_path = args.input
            if not Path(input_path).exists():
                logger.error(f"❌ File not found: {input_path}")
                return 1
    else:
        input_path = "/tmp/v218.pt"
        if not download_v218_from_gcs(input_path):
            return 1

    # Step 2: Run migration
    output_path = "/tmp/v218_migrated.pt"
    logger.info(f"\nMigrating {input_path} → {output_path}...")

    from scripts.migrate_model_to_nplayer import migrate_checkpoint

    try:
        migrate_checkpoint(input_path, output_path, dry_run=False)
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Validate migrated checkpoint
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: Checkpoint Loading")
    logger.info("=" * 60)

    if not validate_checkpoint_loads(output_path):
        logger.error("❌ Checkpoint validation failed")
        return 1

    if args.skip_games:
        logger.info("\n✅ Migration test passed (games skipped)")
        return 0

    # Step 4: Test 2-player game
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: 2-Player Game (Backward Compat)")
    logger.info("=" * 60)

    if not test_2player_game(output_path):
        logger.error("❌ 2-player game test failed")
        return 1

    # Step 5: Test 4-player game
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: 4-Player Game (N-Player Support)")
    logger.info("=" * 60)

    if not test_4player_game(output_path):
        logger.error("❌ 4-player game test failed")
        return 1

    # Success!
    logger.info("\n" + "=" * 60)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("=" * 60)
    logger.info(f"Migrated checkpoint: {output_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"1. Create migration directory: gsutil mkdir gs://civ6az-training/models/civ6az/v218_migrated/")
    logger.info(f"2. Upload migrated model: gsutil cp {output_path} gs://civ6az-training/models/civ6az/v218_migrated/model.pt")
    logger.info(f"3. Create metadata: echo '{{\"migrated_from\": \"v218\", \"migration_version\": \"1.5.0\"}}' | gsutil cp - gs://civ6az-training/models/civ6az/v218_migrated/metadata.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
