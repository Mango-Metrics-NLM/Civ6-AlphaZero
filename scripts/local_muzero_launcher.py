"""Local MuZero Training Launcher.

This script mimics the behavior of the Vertex AI orchestrator but runs everything
locally as subprocesses. Useful for debugging and validation when cloud
access is restricted.

Usage:
    python scripts/local_muzero_launcher.py \
        --algorithm=muzero \
        --iterations=50 \
        --num_selfplay_workers=2 \
        --games_per_worker=10 \
        --mcts_simulations=50
"""

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import asdict

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.storage import GCSModelStorage
from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig
from alpha_zero.utils.game_budget import add_budget_arguments, enforce_cli

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[LocalLauncher] %(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Run a command in a subprocess and wait for it to complete."""
    cmd_str = " ".join(cmd)
    logger.info("Running: %s", cmd_str)
    try:
        subprocess.run(
            cmd,
            check=True,
            env=env,
            shell=True if os.name == "nt" else False,
        )
    except subprocess.CalledProcessError as e:
        logger.error("Command failed: %s", e)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MuZero Training Launcher")
    parser.add_argument("--algorithm", type=str, default="muzero", choices=["alphazero", "muzero"])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--num_selfplay_workers", type=int, default=2)
    parser.add_argument("--games_per_worker", type=int, default=10)
    parser.add_argument("--mcts_simulations", type=int, default=50)
    parser.add_argument("--project", type=str, default="civ6-alphazero")
    parser.add_argument("--bucket", type=str, default="civ6-alphazero-training")
    add_budget_arguments(parser)
    args = parser.parse_args()
    enforce_cli(
        "local_muzero_launcher",
        {
            "iterations": args.iterations,
            "num_selfplay_workers": args.num_selfplay_workers,
            "games_per_worker": args.games_per_worker,
        },
        args,
    )

    # Base python command
    python_cmd = sys.executable

    # Environment setup
    os.environ["CIV6AZ_STORAGE_BACKEND"] = "local"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    env["CIV6AZ_STORAGE_BACKEND"] = "local"
    env["CIV6AZ_VERTEX_PROJECT"] = args.project
    env["CIV6AZ_GCS_BUCKET"] = args.bucket
    env["CIV6AZ_VERTEX_PROJECT"] = args.project
    env["CIV6AZ_GCS_BUCKET"] = args.bucket

    logger.info("Starting Local MuZero Training Loop")
    logger.info("Algorithm: %s", args.algorithm)
    logger.info("Iterations: %s", args.iterations)
    logger.info("Workers: %s", args.num_selfplay_workers)

    # Initialize v0 model if needed
    config = GCPConfig(bucket=args.bucket, project_id=args.project)
    storage = GCSModelStorage(config)

    try:
        latest = storage.get_latest_version()
        logger.info("Found existing model version %d", latest)
    except Exception:
        logger.info("No models found. Initializing v0...")
        net_cfg = config.network
        model_config = TransformerConfig(
            spatial_channels=net_cfg.spatial_channels,
            spatial_size=net_cfg.board_size,
            vector_dim=net_cfg.vector_dim,
            action_space_size=net_cfg.action_space_size,
            num_layers=net_cfg.num_layers,
            embed_dim=net_cfg.embed_dim,
            encoder_type=net_cfg.encoder_type,
        )
        model = Civ6TransformerNetwork(config=model_config)
        storage.save_model(
            model, 0, metadata={"iteration": -1, "elo": 1000.0}, model_config=asdict(model_config)
        )
        logger.info("Initialized model v0")

    # Main Loop
    for iteration in range(args.iterations):
        logger.info("=" * 60)
        logger.info("ITERATION %d / %d", iteration, args.iterations)
        logger.info("=" * 60)

        # ---------------------------------------------------------
        # Phase 1: Self-Play (Parallel)
        # ---------------------------------------------------------
        logger.info("Phase 1: Self-Play")

        selfplay_procs = []
        for i in range(args.num_selfplay_workers):
            cmd = [
                python_cmd,
                "-m",
                "alpha_zero.cloud.gcp.selfplay_worker",
                f"--mcts_type={args.algorithm}",  # 'muzero' or 'alphazero' (mapped to civ6/muzero internally)
                f"--iteration={iteration}",
                f"--model_version={iteration}",
                f"--games={args.games_per_worker}",
                f"--worker_id={i}",
                f"--mcts_simulations={args.mcts_simulations}",
                # "--storage_backend=local",  # REMOVED: Managed via env var
            ]

            # Map algorithm to mcts_type expected by worker
            # The worker expects 'civ6', 'gumbel', or 'muzero'
            # If args.algorithm is 'alphazero', we use 'civ6' (default)
            # If args.algorithm is 'muzero', we use 'muzero'
            if args.algorithm == "alphazero":
                cmd[3] = "--mcts_type=civ6"

            logger.info("Launching worker %d...", i)
            # On Windows, we need to use shell=True for some path resolution,
            # but we use Popen to run in parallel
            proc = subprocess.Popen(cmd, env=env, shell=True if os.name == "nt" else False)
            selfplay_procs.append(proc)

        # Wait for all workers
        for i, proc in enumerate(selfplay_procs):
            exit_code = proc.wait()
            if exit_code != 0:
                logger.error("Self-play worker %d failed with code %d", i, exit_code)
                sys.exit(1)

        logger.info("All self-play workers completed.")

        # ---------------------------------------------------------
        # Phase 2: Training
        # ---------------------------------------------------------
        logger.info("Phase 2: Training")

        cmd = [
            python_cmd,
            "-m",
            "alpha_zero.cloud.gcp.training_worker",
            f"--algorithm={args.algorithm}",
            f"--iteration={iteration}",
            f"--trajectory_iterations={iteration}",
            # f"--storage_backend=local", # REMOVED: Managed via env var
            # Ensure we point to the same local bucket path
            "--epochs=1",  # Keep it fast for validation
            "--batch_size=32",
        ]

        run_command(cmd, env=env)
        logger.info("Training completed for iteration %d", iteration)


if __name__ == "__main__":
    main()
