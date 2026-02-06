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
import gc
import logging
import sys
import time
from typing import Any

import psutil
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
    parser.add_argument("--mcts_simulations", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--dry_run", action="store_true", help="Run without GCS access")

    # MCTS configuration arguments
    parser.add_argument("--c_puct", type=float, default=None, help="MCTS exploration constant")
    parser.add_argument("--dirichlet_alpha", type=float, default=None, help="Dirichlet noise alpha")
    parser.add_argument("--dirichlet_epsilon", type=float, default=None, help="Dirichlet noise weight")
    parser.add_argument("--temperature", type=float, default=None, help="Action selection temperature")
    parser.add_argument("--temperature_threshold", type=int, default=None, help="Turn for temperature=0")


    return parser.parse_args()


def load_model(config: GCPConfig, version: int, dry_run: bool = False) -> torch.nn.Module:
    """Load model from GCS or create dummy for dry run.

    Args:
        config: GCP configuration.
        version: Model version to load.
        dry_run: If True, create dummy model without GCS access.

    Returns:
        Loaded neural network model.

    Raises:
        RuntimeError: If model loading fails.
    """
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    if dry_run:
        logger.info("Dry run: creating dummy model")
        # Create a minimal dummy model for testing
        dummy_config = TransformerConfig(
            spatial_channels=64,
            spatial_size=10,
            vector_dim=64,
            action_space_size=100,
        )
        return Civ6TransformerNetwork(config=dummy_config)

    # Load from GCS
    storage = GCSModelStorage(config)
    try:
        state_dict, metadata, model_config_dict = storage.load_model(version)
    except Exception as e:
        logger.error("Failed to download model v%d from GCS: %s", version, e)
        raise RuntimeError(f"Failed to load model v{version} from GCS") from e

    # Build TransformerConfig from saved config or use defaults
    if model_config_dict is not None:
        logger.info("Using model config from checkpoint")
        # Handle enum conversions if needed
        config_copy = model_config_dict.copy()

        # Convert string enums to actual enum values
        if "norm_type" in config_copy and isinstance(config_copy["norm_type"], str):
            from alpha_zero.core.transformer_network import NormType
            config_copy["norm_type"] = NormType(config_copy["norm_type"])
        if "attention_type" in config_copy and isinstance(config_copy["attention_type"], str):
            from alpha_zero.core.transformer_network import AttentionType
            config_copy["attention_type"] = AttentionType(config_copy["attention_type"])

        # Remove computed fields that shouldn't be passed to constructor
        config_copy.pop("_num_patches", None)
        config_copy.pop("_head_dim", None)

        try:
            transformer_config = TransformerConfig(**config_copy)
        except TypeError as e:
            logger.warning("Failed to parse model config, using defaults: %s", e)
            transformer_config = TransformerConfig()
    else:
        # Legacy model without config - use defaults and warn
        logger.warning(
            "Model v%d has no saved config, using defaults. "
            "This may cause shape mismatches if model was trained with different architecture.",
            version,
        )
        transformer_config = TransformerConfig()

    # Create model and load state dict
    try:
        model = Civ6TransformerNetwork(config=transformer_config)
        incompatible = model.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            logger.warning("Missing keys in state_dict: %s", incompatible.missing_keys[:5])
        if incompatible.unexpected_keys:
            logger.warning("Unexpected keys in state_dict: %s", incompatible.unexpected_keys[:5])

    except RuntimeError as e:
        if "size mismatch" in str(e):
            logger.error(
                "Model architecture mismatch for v%d. The saved model has different "
                "dimensions than expected. Ensure model config is saved with checkpoints.",
                version,
            )
        raise RuntimeError(f"Failed to load state_dict for v{version}: {e}") from e

    model.eval()
    logger.info("Loaded model v%d (ELO: %s)", version, metadata.elo)
    return model


def run_selfplay(
    model: torch.nn.Module,
    num_games: int,
    mcts_sims: int,
    mcts_config_override: dict[str, Any] | None = None,
    dry_run: bool = False,
    device: str = "cpu",
) -> list[dict]:
    """Run self-play games and collect trajectories.

    Args:
        model: Neural network model for MCTS evaluation.
        num_games: Number of self-play games to run.
        mcts_sims: Number of MCTS simulations per move.
        mcts_config_override: Optional dict of MCTS config overrides
            (c_puct, dirichlet_alpha, dirichlet_epsilon, temperature, temperature_threshold).
        dry_run: If True, create dummy trajectories without running games.
        device: Device for neural network inference (e.g., "cuda" or "cpu").

    Returns:
        List of trajectory dictionaries, each containing:
        - spatial: list of spatial features (flattened)
        - vector: list of vector features (flattened)
        - policies: MCTS search policies (action probability distributions)
        - actions: actions taken at each step
        - rewards: final game outcome (propagated to all states)
    """
    # For dry run, return dummy trajectories without needing actual env/MCTS
    if dry_run:
        logger.info("Dry run: creating dummy trajectories")
        dummy_trajectories: list[dict[str, Any]] = []
        for _ in range(num_games):
            dummy_traj: dict[str, Any] = {
                "spatial": [[0.0] * (64 * 10 * 10) for _ in range(5)],
                "vector": [[0.0] * 64 for _ in range(5)],
                "policies": [[1.0 / 100] * 100 for _ in range(5)],
                "actions": [0] * 5,
                "rewards": [0.5] * 5,
            }
            dummy_trajectories.append(dummy_traj)
        return dummy_trajectories

    trajectories: list[dict[str, Any]] = []
    failed_games = 0

    for game_idx in range(num_games):
        start_time = time.time()
        game_failed = False

        # Create environment and MCTS
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv()

        # Build MCTS config with optional overrides
        mcts_params: dict[str, Any] = {
            "num_simulations": mcts_sims,
            "c_puct": 1.5,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "temperature": 1.0,
            "temperature_threshold": 30,
        }
        if mcts_config_override:
            mcts_params.update({k: v for k, v in mcts_config_override.items() if v is not None})

        mcts_config = MCTSConfig(**mcts_params)
        mcts = Civ6MCTS(
            network=model,
            config=mcts_config,
            device=device,
        )

        trajectory: dict[str, Any] = {
            "spatial": [],
            "vector": [],
            "policies": [],
            "actions": [],
        }

        env.reset()
        done = False

        while not done:
            # Get observation with spatial, vector, action_mask
            obs = env.get_observation()

            # Run MCTS with error handling
            try:
                # Pass env directly since MicroCivEnv implements MCTSEnvironment protocol
                policy, _ = mcts.search(env)  # type: ignore[arg-type]
                action = mcts.select_action(policy)
            except Exception as e:
                logger.error(
                    "MCTS search failed at step %d: %s",
                    len(trajectory["spatial"]),
                    e,
                )
                if device != "cpu":
                    torch.cuda.empty_cache()
                game_failed = True
                break  # Skip this game, move to next

            # Store experience - flatten spatial and vector for storage
            spatial_flat = obs["spatial"].flatten().tolist()
            vector_flat = obs["vector"].flatten().tolist()
            trajectory["spatial"].append(spatial_flat)
            trajectory["vector"].append(vector_flat)
            trajectory["policies"].append(policy.tolist() if hasattr(policy, "tolist") else policy)
            trajectory["actions"].append(int(action))

            # Take action
            step_result = env.step(action)
            reward = float(step_result[1])
            done = bool(step_result[2]) or bool(step_result[3])  # terminated or truncated

        # Track failed games
        if game_failed:
            failed_games += 1
            logger.warning("Game %d/%d failed, skipping", game_idx + 1, num_games)
        # Propagate final reward to all states (only if we collected data and game didn't fail)
        elif trajectory["spatial"]:
            final_value = reward
            num_steps = len(trajectory["spatial"])
            trajectory["rewards"] = [final_value] * num_steps
            trajectories.append(trajectory)

        elapsed = time.time() - start_time
        if not game_failed:
            logger.debug("Game %d/%d completed in %.1fs", game_idx + 1, num_games, elapsed)

        # Cleanup after each game to prevent memory accumulation
        del trajectory
        del mcts
        del env
        if device != "cpu":
            torch.cuda.empty_cache()

        # Log progress and memory usage periodically
        if (game_idx + 1) % 10 == 0:
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(
                "Progress: %d/%d games, Memory: %.0f MB",
                game_idx + 1,
                num_games,
                mem_mb,
            )
            # Force garbage collection if memory is getting high
            if mem_mb > 6000:  # 6GB threshold for 8GB container
                logger.warning("High memory usage (%.0f MB), forcing GC", mem_mb)
                gc.collect()
                if device != "cpu":
                    torch.cuda.empty_cache()

    # Log final statistics including any failed games
    if failed_games > 0:
        logger.warning(
            "Self-play completed with %d/%d games failed",
            failed_games,
            num_games,
        )

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

        # Build MCTS config overrides from CLI arguments
        mcts_override: dict[str, Any] = {
            "c_puct": args.c_puct,
            "dirichlet_alpha": args.dirichlet_alpha,
            "dirichlet_epsilon": args.dirichlet_epsilon,
            "temperature": args.temperature,
            "temperature_threshold": args.temperature_threshold,
        }

        # Run self-play
        start_time = time.time()
        trajectories = run_selfplay(
            model,
            args.games,
            args.mcts_simulations,
            mcts_config_override=mcts_override,
            dry_run=args.dry_run,
            device=str(device),
        )
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
