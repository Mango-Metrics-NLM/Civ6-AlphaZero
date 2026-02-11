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

# Memory threshold for triggering garbage collection (in MB)
MEMORY_THRESHOLD_MB: int = 6000


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
    parser.add_argument("--agent_id", type=str, default=None, help="PBT Agent ID")
    parser.add_argument("--mode", type=str, default="selfplay", help="Worker mode (selfplay)")

    # MCTS configuration arguments
    parser.add_argument(
        "--mcts_simulations", type=int, default=400, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--mcts_type",
        type=str,
        default="civ6",
        choices=["civ6", "gumbel"],
        help="MCTS variant to use",
    )
    parser.add_argument("--board_size", type=int, default=8, help="Size of the hex board")
    parser.add_argument("--max_turns", type=int, default=30, help="Max turns per game")
    parser.add_argument("--c_puct", type=float, default=1.5, help="MCTS exploration constant")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet noise alpha")
    parser.add_argument(
        "--dirichlet_epsilon", type=float, default=0.25, help="Dirichlet noise weight"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Action selection temperature"
    )
    parser.add_argument(
        "--temperature_threshold", type=int, default=30, help="Turn for temperature=0"
    )
    parser.add_argument(
        "--temperature_schedule",
        type=str,
        default="threshold",
        choices=["threshold", "linear", "cosine"],
        help="Temperature decay schedule (threshold=step at threshold, linear/cosine=smooth decay)",
    )

    # Additional hyperparameters
    parser.add_argument(
        "--gumbel_num_actions", type=int, default=16, help="Gumbel MCTS sampled actions"
    )
    parser.add_argument(
        "--action_space_size", type=int, default=376, help="Size of the action space"
    )
    parser.add_argument(
        "--num_res_blocks", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument("--num_filters", type=int, default=64, help="Model embedding dimension")
    parser.add_argument("--reward_shaping", action="store_true", help="Enable reward shaping")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Reward scaling factor")
    parser.add_argument("--dry_run", action="store_true", help="Run without GCS access")

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
            net_cfg = config.network
            logger.warning("Failed to parse model config (%s), using pipeline defaults", e)
            transformer_config = TransformerConfig(
                spatial_channels=net_cfg.spatial_channels,
                spatial_size=net_cfg.board_size,
                vector_dim=net_cfg.vector_dim,
                action_space_size=net_cfg.action_space_size,
                num_layers=net_cfg.num_layers,
                embed_dim=net_cfg.embed_dim,
            )
    else:
        # Legacy model without config - use NetworkArchitectureConfig as fallback
        net_cfg = config.network
        logger.warning(
            "Model v%d has no saved config. Using pipeline architecture defaults: "
            "spatial_size=%d, action_space=%d, embed_dim=%d, num_layers=%d",
            version,
            net_cfg.board_size,
            net_cfg.action_space_size,
            net_cfg.embed_dim,
            net_cfg.num_layers,
        )
        transformer_config = TransformerConfig(
            spatial_channels=net_cfg.spatial_channels,
            spatial_size=net_cfg.board_size,
            vector_dim=net_cfg.vector_dim,
            action_space_size=net_cfg.action_space_size,
            num_layers=net_cfg.num_layers,
            embed_dim=net_cfg.embed_dim,
        )

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


def compute_temperature(
    turn: int,
    max_turns: int,
    temperature: float,
    temperature_threshold: int,
    schedule: str = "threshold",
) -> float:
    """Compute action selection temperature for a given turn.

    Args:
        turn: Current turn number (0-indexed).
        max_turns: Maximum turns in the game.
        temperature: Base temperature value.
        temperature_threshold: Turn at which threshold schedule drops to 0.
        schedule: Schedule type ("threshold", "linear", "cosine").

    Returns:
        Temperature value for this turn.
    """
    if schedule == "threshold":
        return temperature if turn < temperature_threshold else 0.0
    elif schedule == "linear":
        # Linear decay from temperature to 0 over max_turns
        progress = min(turn / max(max_turns, 1), 1.0)
        return temperature * (1.0 - progress)
    elif schedule == "cosine":
        # Cosine annealing from temperature to 0
        import math

        progress = min(turn / max(max_turns, 1), 1.0)
        return temperature * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        return temperature


def run_selfplay(
    model: torch.nn.Module,
    num_games: int,
    mcts_sims: int,
    mcts_type: str = "civ6",
    board_size: int = 8,
    max_turns: int = 30,
    gumbel_num_actions: int = 16,
    mcts_config_override: dict[str, Any] | None = None,
    dry_run: bool = False,
    device: str = "cpu",
    temperature_schedule: str = "threshold",
) -> list[dict]:
    """Run self-play games and collect trajectories.

    Args:
        model: Neural network model for MCTS evaluation.
        num_games: Number of self-play games to run.
        mcts_sims: Number of MCTS simulations per move.
        mcts_type: MCTS variant to use ("civ6" or "gumbel").
        board_size: Size of the game board.
        max_turns: Maximum turns per game.
        gumbel_num_actions: Number of sampled actions for Gumbel MCTS.
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
        # MicroCivEncoder.TOTAL_CHANNELS = 13 for spatial encoding
        spatial_channels = 13
        vector_dim = 64
        dummy_trajectories: list[dict[str, Any]] = []
        for _ in range(num_games):
            dummy_traj: dict[str, Any] = {
                "spatial": [[0.0] * (spatial_channels * board_size * board_size) for _ in range(5)],
                "vector": [[0.0] * vector_dim for _ in range(5)],
                "policies": [[1.0 / 100] * 100 for _ in range(5)],
                "actions": [0] * 5,
                "rewards": [0.5] * 5,
            }
            dummy_trajectories.append(dummy_traj)
        return dummy_trajectories

    # Extract temperature params from override before the game loop
    # These are action selection params, not MCTS config fields
    sel_temperature = 1.0
    sel_threshold = max_turns
    if mcts_config_override:
        sel_temperature = mcts_config_override.pop("temperature", 1.0)
        sel_threshold = mcts_config_override.pop("temperature_threshold", max_turns)

    logger.info(
        "Temperature schedule: %s (temp=%.2f, threshold=%d, max_turns=%d)",
        temperature_schedule,
        sel_temperature,
        sel_threshold,
        max_turns,
    )

    trajectories: list[dict[str, Any]] = []
    failed_games = 0

    for game_idx in range(num_games):
        start_time = time.time()
        game_failed = False

        # Create environment and MCTS
        from alpha_zero.core.mcts_factory import create_mcts
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=board_size, max_turns=max_turns)

        # Build variant-aware MCTS kwargs to avoid passing invalid params
        puct_variants = {"puct", "civ6"}
        gumbel_variants = {"gumbel"}

        mcts_kwargs: dict[str, Any] = {
            "num_simulations": mcts_sims,
        }

        if mcts_type in puct_variants:
            mcts_kwargs.update(
                {
                    "c_puct": 1.5,
                    "dirichlet_alpha": 0.3,
                    "dirichlet_epsilon": 0.25,
                }
            )
        elif mcts_type in gumbel_variants:
            mcts_kwargs["num_sampled_actions"] = mcts_kwargs.get(
                "num_sampled_actions",
                gumbel_num_actions,
            )

        # Apply remaining MCTS config overrides (temperature already extracted)
        # Filter to only keys valid for the current MCTS variant to avoid
        # passing e.g. c_puct/dirichlet_* to GumbelMCTS
        civ6_only_keys = {"c_puct", "dirichlet_alpha", "dirichlet_epsilon"}
        gumbel_only_keys = {"num_sampled_actions", "c_visit", "c_scale"}
        if mcts_config_override:
            for k, v in mcts_config_override.items():
                if v is None:
                    continue
                # Skip keys that belong to a different MCTS variant
                if mcts_type in gumbel_variants and k in civ6_only_keys:
                    continue
                if mcts_type in puct_variants and k in gumbel_only_keys:
                    continue
                mcts_kwargs[k] = v

        mcts = create_mcts(
            variant=mcts_type,
            network=model,
            device=device,
            **mcts_kwargs,
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
                turn = len(trajectory["spatial"])
                temp = compute_temperature(
                    turn=turn,
                    max_turns=max_turns,
                    temperature=sel_temperature,
                    temperature_threshold=sel_threshold,
                    schedule=temperature_schedule,
                )
                action = mcts.select_action(policy, temperature=temp)
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
            if mem_mb > MEMORY_THRESHOLD_MB:
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
        total_states = sum(len(t["spatial"]) for t in trajectories)
        logger.info(
            "Dry run: would save %d trajectories (%d states)", len(trajectories), total_states
        )
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
    logger.info(
        "Config: model_v%d, %d games, %d MCTS sims (%s), %dx%d board",
        args.model_version,
        args.games,
        args.mcts_simulations,
        args.mcts_type,
        args.board_size,
        args.board_size,
    )

    # Initialize W&B (disabled by default, opt-in via WANDB_ENABLED env var)
    wandb_logger = None
    try:
        from alpha_zero.utils.wandb_logger import WandbLogger

        wandb_logger = WandbLogger.from_env()
        if wandb_logger.is_enabled:
            wandb_logger.init(
                config_dict={
                    "worker_id": args.worker_id,
                    "iteration": args.iteration,
                    "model_version": args.model_version,
                    "mcts_type": args.mcts_type,
                    "mcts_simulations": args.mcts_simulations,
                    "board_size": args.board_size,
                    "max_turns": args.max_turns,
                    "games": args.games,
                    "mode": "selfplay",
                }
            )
            logger.info("W&B logging enabled")
    except ImportError:
        logger.debug("W&B not available")

    try:
        config = GCPConfig()
        if args.agent_id:
            config.paths.agent_id = args.agent_id
            logger.info("Using Agent ID: %s", args.agent_id)

        if not args.dry_run:
            config.validate()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        # Load model
        model = load_model(
            config,
            args.model_version,
            dry_run=args.dry_run,
        )
        model = model.to(device)
        model.eval()

        # Build MCTS override dict from CLI args
        mcts_override = {
            "c_puct": args.c_puct,
            "dirichlet_alpha": args.dirichlet_alpha,
            "dirichlet_epsilon": args.dirichlet_epsilon,
            "temperature": args.temperature,
            "temperature_threshold": args.temperature_threshold,
        }

        # Run self-play
        start_time = time.time()
        trajectories = run_selfplay(
            model=model,
            num_games=args.games,
            mcts_sims=args.mcts_simulations,
            mcts_type=args.mcts_type,
            board_size=args.board_size,
            max_turns=args.max_turns,
            gumbel_num_actions=args.gumbel_num_actions,
            mcts_config_override=mcts_override,
            dry_run=args.dry_run,
            device=str(device),
            temperature_schedule=args.temperature_schedule,
        )
        elapsed = time.time() - start_time

        # Calculate total states generated (using spatial list length)
        total_states = sum(len(t["spatial"]) for t in trajectories)
        logger.info(
            "Self-play complete: %d games, %d states in %.1fs (%.1f games/min)",
            len(trajectories),
            total_states,
            elapsed,
            len(trajectories) / max(elapsed / 60, 0.001),
        )

        # Log aggregate selfplay metrics to W&B
        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "is_enabled")
            and wandb_logger.is_enabled
        ):
            wandb_logger.log_selfplay(
                games_completed=len(trajectories),
                buffer_size=total_states,
                samples_generated=total_states,
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
    finally:
        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "is_enabled")
            and wandb_logger.is_enabled
        ):
            wandb_logger.finish()


if __name__ == "__main__":
    sys.exit(main())
