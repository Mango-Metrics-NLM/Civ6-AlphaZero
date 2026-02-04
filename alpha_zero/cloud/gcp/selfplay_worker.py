"""
Self-Play Worker Entry Point for Vertex AI.

This is the containerized entry point for self-play workers.
Each worker:
1. Loads the latest model from GCS
2. Runs self-play games using MCTS
3. Writes trajectories to GCS immediately
4. Exits cleanly (stateless design for Spot VM resilience)

Usage:
    python -m alpha_zero.cloud.gcp.selfplay_worker \
        --worker_id=0 \
        --model_version=1 \
        --iteration=10 \
        --games=100
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import torch

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.storage import GCSModelStorage, GCSTrajectoryStorage

logger = logging.getLogger(__name__)


def setup_logging(worker_id: int) -> None:
    """Configure logging for worker."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Worker-{worker_id:02d}] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Civ6AZ Self-Play Worker")
    parser.add_argument("--worker_id", type=int, required=True, help="Worker ID")
    parser.add_argument("--model_version", type=int, required=True, help="Model version to use")
    parser.add_argument("--iteration", type=int, required=True, help="Training iteration")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--mcts_simulations", type=int, default=400, help="MCTS simulations per move")
    parser.add_argument("--mode", type=str, default="selfplay", help="Worker mode (selfplay)")
    parser.add_argument("--dry_run", action="store_true", help="Run without GCS access")
    return parser.parse_args()


def load_model(config: GCPConfig, version: int, dry_run: bool = False) -> torch.nn.Module:
    """Load model from GCS or create dummy for dry run."""
    if dry_run:
        logger.info("Dry run: creating dummy model")
        # Create a minimal dummy model for testing
        from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

        dummy_config = TransformerConfig(
            spatial_channels=64,
            spatial_size=10,
            vector_dim=64,
            action_space_size=100,
        )
        return Civ6TransformerNetwork(config=dummy_config)

    storage = GCSModelStorage(config)
    state_dict, metadata = storage.load_model(version)

    # Load the actual model architecture
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    model_config = TransformerConfig(
        spatial_channels=64,
        spatial_size=10,
        vector_dim=64,
        action_space_size=100,
    )
    model = Civ6TransformerNetwork(config=model_config)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Loaded model v%d (ELO: %s)", version, metadata.elo)
    return model


def run_selfplay(
    model: torch.nn.Module,
    num_games: int,
    mcts_sims: int,
    dry_run: bool = False,
) -> list[dict]:
    """Run self-play games and collect trajectories.

    Returns a list of trajectory dictionaries, each containing:
    - states: list of game states
    - policies: MCTS search policies (action probability distributions)
    - actions: actions taken at each step
    - rewards: final game outcome (propagated to all states)
    """
    # For dry run, return dummy trajectories without needing actual env/MCTS
    if dry_run:
        logger.info("Dry run: creating dummy trajectories")
        trajectories = []
        for _ in range(num_games):
            trajectory = {
                "states": [[0.0] * (64 * 10 * 10) for _ in range(5)],
                "policies": [[1.0 / 100] * 100 for _ in range(5)],
                "actions": [0] * 5,
                "rewards": [0.5] * 5,
            }
            trajectories.append(trajectory)
        return trajectories

    trajectories = []

    for game_idx in range(num_games):
        start_time = time.time()

        # Create environment and MCTS
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.core.civ6_mcts import Civ6MCTS

        env = MicroCivEnv()
        mcts = Civ6MCTS(
            model=model,
            num_simulations=mcts_sims,
            c_puct=1.4,
        )

        trajectory: dict[str, list] = {
            "states": [],
            "policies": [],
            "actions": [],
        }

        state, _ = env.reset()
        done = False

        while not done:
            # Run MCTS
            action, policy = mcts.search(env.unwrapped)

            # Store experience
            trajectory["states"].append(state.copy())
            trajectory["policies"].append(policy)
            trajectory["actions"].append(action)

            # Take action
            state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

        # Propagate final reward
        final_value = reward
        trajectory["rewards"] = [final_value] * len(trajectory["states"])
        trajectories.append(trajectory)

        elapsed = time.time() - start_time
        logger.debug("Game %d/%d completed in %.1fs", game_idx + 1, num_games, elapsed)

        # Log progress periodically
        if (game_idx + 1) % 10 == 0:
            logger.info("Progress: %d/%d games completed", game_idx + 1, num_games)

    return trajectories


def save_trajectories(
    config: GCPConfig,
    trajectories: list[dict],
    iteration: int,
    worker_id: int,
    model_version: int,
    dry_run: bool = False,
) -> str | None:
    """Save trajectories to GCS."""
    if dry_run:
        logger.info("Dry run: would save %d trajectories", len(trajectories))
        return None

    storage = GCSTrajectoryStorage(config, compress=True)
    uri = storage.save_trajectories(
        trajectories=trajectories,
        iteration=iteration,
        worker_id=worker_id,
        model_version=model_version,
    )
    logger.info("Saved %d trajectories to %s", len(trajectories), uri)
    return uri


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.worker_id)

    logger.info("Starting self-play worker %d for iteration %d", args.worker_id, args.iteration)
    logger.info("Model version: %d, Games: %d, MCTS sims: %d", args.model_version, args.games, args.mcts_simulations)

    try:
        # Load configuration from environment
        config = GCPConfig()
        if not args.dry_run:
            config.validate()

        # Load model
        model = load_model(config, args.model_version, args.dry_run)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info("Using device: %s", device)

        # Run self-play
        start_time = time.time()
        trajectories = run_selfplay(model, args.games, args.mcts_simulations, dry_run=args.dry_run)
        elapsed = time.time() - start_time

        logger.info(
            "Self-play complete: %d trajectories in %.1fs (%.1f games/min)",
            len(trajectories),
            elapsed,
            len(trajectories) / max(elapsed / 60, 0.001),
        )

        # Save trajectories
        save_trajectories(
            config=config,
            trajectories=trajectories,
            iteration=args.iteration,
            worker_id=args.worker_id,
            model_version=args.model_version,
            dry_run=args.dry_run,
        )

        logger.info("Worker %d completed successfully", args.worker_id)
        return 0

    except KeyboardInterrupt:
        logger.warning("Worker %d interrupted", args.worker_id)
        return 130
    except Exception as e:
        logger.exception("Worker %d failed: %s", args.worker_id, e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
