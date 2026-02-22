"""Launch A/B comparison training runs for encoder evaluation.

Loads environment variables from .env file and invokes the vertex_orchestrator
with the specified encoder_type and agent_id for namespace isolation.

Usage:
    python scripts/launch_ab_run.py --encoder_type patch_embed --agent_id ab_patch_embed
    python scripts/launch_ab_run.py --encoder_type hex_crystalline --agent_id ab_hex_crystalline
    python scripts/launch_ab_run.py --encoder_type patch_embed --agent_id ab_patch_embed --dry_run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def load_env_file(env_path: Path) -> dict[str, str]:
    """Load environment variables from a .env file.

    Args:
        env_path: Path to the .env file.

    Returns:
        Dictionary of loaded environment variables.
    """
    loaded = {}
    if not env_path.exists():
        print(f"Warning: .env file not found at {env_path}")
        return loaded

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if value and key not in os.environ:
                os.environ[key] = value
                loaded[key] = value

    return loaded


def main() -> None:
    """Launch an A/B training run."""
    parser = argparse.ArgumentParser(description="Launch A/B encoder comparison run")
    parser.add_argument(
        "--encoder_type",
        choices=["patch_embed", "hex_crystalline"],
        required=True,
        help="Encoder type for this run",
    )
    parser.add_argument(
        "--agent_id",
        required=True,
        help="Agent ID for GCS path namespacing",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of training iterations (default: 20)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the orchestrator command without executing",
    )
    args = parser.parse_args()

    # Load .env file
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    loaded = load_env_file(env_path)
    print(f"Loaded {len(loaded)} env vars from {env_path}")

    # Verify required env vars
    required_vars = [
        "CIV6AZ_VERTEX_PROJECT",
        "CIV6AZ_VERTEX_REGION",
        "CIV6AZ_GCS_BUCKET",
        "CIV6AZ_CONTAINER_IMAGE_URI",
    ]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"Error: Missing required env vars: {missing}")
        sys.exit(1)

    print(f"Project: {os.environ['CIV6AZ_VERTEX_PROJECT']}")
    print(f"Region: {os.environ['CIV6AZ_VERTEX_REGION']}")
    print(f"Container: {os.environ['CIV6AZ_CONTAINER_IMAGE_URI']}")
    print(f"Encoder: {args.encoder_type}")
    print(f"Agent ID: {args.agent_id}")
    print(f"Iterations: {args.iterations}")

    # Build orchestrator command
    orchestrator_args = [
        sys.executable,
        "-m",
        "alpha_zero.cloud.gcp.vertex_orchestrator",
        f"--iterations={args.iterations}",
        "--workers=2",
        "--games_per_worker=50",
        "--mcts_type=gumbel",
        "--mcts_simulations=8",
        "--gumbel_num_actions=16",
        "--max_turns=15",
        "--epochs=10",
        "--lr_schedule_type=warmup_cosine",
        "--lr_initial=0.001",
        "--lr_final=0.0001",
        "--lr_warmup_steps=100",
        "--value_lr_multiplier=0.1",
        "--evaluation_games=25",
        "--evaluation_win_threshold=0.55",
        "--trajectory_window=5",
        "--post_run_tournament",
        "--tournament_games=6",
        "--tournament_mcts_sims=25",
        f"--encoder_type={args.encoder_type}",
        f"--agent_id={args.agent_id}",
    ]

    cmd_str = " \\\n    ".join(orchestrator_args)

    if args.dry_run:
        print(f"\n[DRY RUN] Would execute:\n{cmd_str}")
        return

    print("\nLaunching orchestrator...")
    print(f"Command: {cmd_str}\n")

    # Execute by importing and calling main directly (avoids subprocess env issues)
    sys.argv = orchestrator_args[2:]  # Remove python -m prefix
    from alpha_zero.cloud.gcp.vertex_orchestrator import main as orchestrator_main

    orchestrator_main()


if __name__ == "__main__":
    main()
