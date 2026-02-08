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
    parser.add_argument("--agent_id", type=str, default=None, help="PBT Agent ID")
    parser.add_argument(
        "--mcts_simulations", type=int, default=400, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--mcts_type",
        type=str,
        default="civ6",
        choices=["civ6", "gumbel"],
        help="MCTS algorithm variant",
    )
    parser.add_argument(
        "--c_puct", type=float, default=1.5, help="MCTS exploration constant"
    )
    parser.add_argument(
        "--gumbel_num_actions",
        type=int,
        default=16,
        help="Number of sampled actions for Gumbel MCTS",
    )
    parser.add_argument("--board_size", type=int, default=8, help="Board size")
    parser.add_argument("--action_space_size", type=int, default=376, help="Action space size")
    parser.add_argument("--num_res_blocks", type=int, default=6, help="Residual blocks in network")
    parser.add_argument("--num_filters", type=int, default=64, help="Conv filters in network")
    parser.add_argument(
        "--max_turns", type=int, default=30, help="Max game turns before truncation"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Action selection temperature"
    )
    parser.add_argument(
        "--reward_shaping", action="store_true", help="Enable enhanced reward shaping"
    )
    parser.add_argument(
        "--reward_scale", type=float, default=1.0, help="Reward shaping scale factor"
    )
    parser.add_argument("--mode", type=str, default="selfplay", help="Worker mode (selfplay)")
    parser.add_argument("--dry_run", action="store_true", help="Run without GCS access")
    return parser.parse_args()


def load_model(
    config: GCPConfig,
    version: int,
    board_size: int,
    action_space_size: int,
    num_res_blocks: int,
    num_filters: int,
    dry_run: bool = False,
) -> torch.nn.Module:
    """Load model from GCS or create fresh for dry run."""
    from alpha_zero.core.civ6_network import MicroCivNetwork

    model = MicroCivNetwork(
        board_size=board_size,
        action_space_size=action_space_size,
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
    )

    if dry_run:
        logger.info("Dry run: using fresh model (no GCS load)")
        return model

    storage = GCSModelStorage(config)
    state_dict, metadata = storage.load_model(version)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Loaded model v%d from GCS", version)
    return model


def run_selfplay(
    model: torch.nn.Module,
    num_games: int,
    mcts_sims: int,
    board_size: int,
    max_turns: int,
    temperature: float,
    device: torch.device,
    mcts_type: str = "civ6",
    c_puct: float = 1.5,
    gumbel_num_actions: int = 16,
    reward_shaping: bool = False,
    reward_scale: float = 1.0,
    wandb_logger: object | None = None,
    iteration: int = 0,
    worker_id: int = 0,
) -> list[dict]:
    """Run self-play games and collect trajectories.

    Returns a list of trajectory dictionaries, each containing:
    - states: list of observation dicts (spatial, vector, action_mask)
    - policies: MCTS search policies (action probability distributions)
    - values: final game outcome propagated to all states
    """
    from alpha_zero.core.mcts_factory import create_mcts
    from alpha_zero.envs.microciv import MicroCivEnv

    # Set up reward shaper if enabled
    reward_shaper = None
    if reward_shaping:
        try:
            from alpha_zero.utils.enhanced_rewards import create_enhanced_reward_shaper

            reward_shaper = create_enhanced_reward_shaper(
                scale_factor=reward_scale,
                board_size=board_size,
            )
            logger.info("Reward shaping enabled (scale=%.2f)", reward_scale)
        except ImportError:
            logger.warning("Enhanced rewards not available, continuing without shaping")

    # Gumbel MCTS provides its own stochasticity via Gumbel sampling
    use_noise = mcts_type != "gumbel"

    trajectories = []

    for game_idx in range(num_games):
        start_time = time.time()

        env = MicroCivEnv(board_size=board_size, max_turns=max_turns)
        mcts = create_mcts(
            mcts_type=mcts_type,
            network=model,
            device=str(device),
            num_simulations=mcts_sims,
            num_sampled_actions=gumbel_num_actions,
            c_puct=c_puct,
        )

        trajectory: dict[str, list] = {
            "states": [],
            "policies": [],
            "actions": [],
        }

        obs, _ = env.reset()
        done = False
        move_count = 0
        cumulative_shaped_reward = 0.0

        while not done:
            # Run MCTS search
            action_probs, root_value = mcts.search(env, add_noise=use_noise)

            # Store experience
            trajectory["states"].append(
                {
                    "spatial": obs["spatial"].copy(),
                    "vector": obs["vector"].copy(),
                    "action_mask": obs["action_mask"].copy(),
                }
            )
            trajectory["policies"].append(action_probs.copy())

            # Select action with temperature
            action = mcts.select_action(action_probs, temperature=temperature)
            trajectory["actions"].append(action)

            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            move_count += 1

            # Apply reward shaping if enabled
            if reward_shaper is not None:
                shaped = reward_shaper.shape_reward(
                    obs,
                    float(reward),
                    info,
                    terminated=terminated,
                )
                cumulative_shaped_reward += shaped

            if move_count % 10 == 0:
                logger.info("Game %d/%d: move %d", game_idx + 1, num_games, move_count)

        # Propagate final reward to all states
        if reward_shaper is not None:
            final_value = cumulative_shaped_reward / max(len(trajectory["states"]), 1)
        else:
            final_value = float(reward)
        trajectory["values"] = [final_value] * len(trajectory["states"])
        trajectories.append(trajectory)

        elapsed = time.time() - start_time

        # Log game to W&B
        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "is_enabled")
            and wandb_logger.is_enabled
        ):
            wandb_logger.log_game(
                game_id=f"iter{iteration}_w{worker_id}_g{game_idx}",
                winner=1 if final_value > 0 else (0 if final_value < 0 else -1),
                total_turns=move_count,
                reward=final_value,
            )

        if (game_idx + 1) % 10 == 0 or (game_idx + 1) == num_games:
            logger.info(
                "Progress: %d/%d games completed (last: %d moves, %.1fs)",
                game_idx + 1,
                num_games,
                move_count,
                elapsed,
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
        total_states = sum(len(t["states"]) for t in trajectories)
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
            board_size=args.board_size,
            action_space_size=args.action_space_size,
            num_res_blocks=args.num_res_blocks,
            num_filters=args.num_filters,
            dry_run=args.dry_run,
        )
        model = model.to(device)
        model.eval()

        # Run self-play
        start_time = time.time()
        trajectories = run_selfplay(
            model,
            args.games,
            args.mcts_simulations,
            board_size=args.board_size,
            max_turns=args.max_turns,
            temperature=args.temperature,
            device=device,
            mcts_type=args.mcts_type,
            c_puct=args.c_puct,
            gumbel_num_actions=args.gumbel_num_actions,
            reward_shaping=args.reward_shaping,
            reward_scale=args.reward_scale,
            wandb_logger=wandb_logger,
            iteration=args.iteration,
            worker_id=args.worker_id,
        )
        elapsed = time.time() - start_time

        total_states = sum(len(t["states"]) for t in trajectories)
        logger.info(
            "Self-play complete: %d games, %d states in %.1fs (%.1f games/min)",
            len(trajectories),
            total_states,
            elapsed,
            len(trajectories) / (elapsed / 60) if elapsed > 0 else 0,
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
