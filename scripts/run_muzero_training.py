"""MuZero Training Script with experiment tracking.

This script runs full MuZero training with:
- Gumbel MuZero MCTS (efficient search with 16 simulations)
- MuZero network with dynamics model
- EfficientZero consistency loss
- Trajectory-based training with multi-step unrolling
- MLflow experiment tracking (via the compat adapter)

Usage:
    python scripts/run_muzero_training.py --wandb --iterations 10
    python scripts/run_muzero_training.py --quick  # Quick test run
    python scripts/run_muzero_training.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from alpha_zero.core.civ6_mcts import MuZeroMCTS, MuZeroMCTSConfig

# Core imports
from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.training.muzero_targets import outcome_only_targets
from alpha_zero.training.muzero_trainer import (
    GameTrajectory,
    MuZeroTrainer,
    MuZeroTrainingConfig,
    TrajectoryBuffer,
)
from alpha_zero.utils.game_budget import add_budget_arguments, enforce_cli

# MLflow-backed tracker import (optional). Legacy names kept for --wandb.
try:
    from alpha_zero.monitoring.config import MLflowConfig as WandbConfig
    from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter as WandbLogger

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    WandbLogger = None
    WandbConfig = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class MuZeroRunConfig:
    """Complete configuration for MuZero training run."""

    # Environment
    board_size: int = 16
    max_turns: int = 200

    # Network architecture
    num_res_blocks: int = 6
    num_filters: int = 64
    hidden_state_channels: int = 64
    dynamics_num_blocks: int = 4

    # MCTS
    num_simulations: int = 16
    num_sampled_actions: int = 16
    use_gumbel: bool = True
    use_sequential_halving: bool = True

    # Training
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    unroll_steps: int = 5
    td_steps: int = 10

    # Self-play
    num_iterations: int = 50
    num_games_per_iteration: int = 20
    replay_buffer_size: int = 50000
    min_replay_size: int = 500
    num_workers: int = 1  # Parallel self-play workers (1 = sequential)

    # EfficientZero
    use_consistency_loss: bool = True
    use_value_prefix: bool = True
    consistency_loss_weight: float = 2.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints/muzero"
    save_interval: int = 5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # W&B
    wandb_enabled: bool = False
    wandb_project: str = "civ6-muzero"
    wandb_run_name: str | None = None


def create_network(config: MuZeroRunConfig, action_space_size: int) -> MuZeroNetwork:
    """Create MuZero network from config and move to device."""
    network_config = MuZeroConfig(
        spatial_channels=13,  # MicroCiv default
        spatial_size=config.board_size,
        vector_dim=64,  # MicroCiv default
        num_res_blocks=config.num_res_blocks,
        num_filters=config.num_filters,
        hidden_state_channels=config.hidden_state_channels,
        dynamics_num_blocks=config.dynamics_num_blocks,
        action_space_size=action_space_size,
        use_consistency_loss=config.use_consistency_loss,
        use_value_prefix=config.use_value_prefix,
    )
    network = MuZeroNetwork(network_config)
    return network.to(config.device)


def create_mcts(network: MuZeroNetwork, config: MuZeroRunConfig) -> MuZeroMCTS:
    """Create MuZero MCTS from config."""
    mcts_config = MuZeroMCTSConfig(
        num_simulations=config.num_simulations,
        num_sampled_actions=config.num_sampled_actions,
        use_gumbel=config.use_gumbel,
        use_sequential_halving=config.use_sequential_halving,
    )
    return MuZeroMCTS(network, mcts_config, device=config.device)


def self_play_game(
    env: MicroCivEnv,
    mcts: MuZeroMCTS,
    config: MuZeroRunConfig,
) -> GameTrajectory:
    """Play a single self-play game and collect trajectory."""
    observations = []
    actions = []
    players = []
    policies = []
    dones = []

    obs, _ = env.reset()
    done = False

    while not done:
        # Store observation as dict
        observations.append(
            {
                "spatial": obs["spatial"].copy(),
                "vector": obs["vector"].copy(),
                "action_mask": obs["action_mask"].copy(),
            }
        )

        # Get MCTS policy. The root mover anchors the search's signed backup.
        players.append(int(env.current_player))
        action_probs, _root_value = mcts.search(
            obs, obs["action_mask"], add_noise=True, to_play=env.current_player
        )
        policies.append(action_probs.copy())

        # Select action
        temperature = 1.0 if env.turn < 30 else 0.0
        action = mcts.select_action(action_probs, temperature)
        actions.append(action)

        # Take action
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        dones.append(done)

    # Append terminal observation (MuZero requires T+1 observations for T actions)
    observations.append(
        {
            "spatial": obs["spatial"].copy(),
            "vector": obs["vector"].copy(),
            "action_mask": obs["action_mask"].copy(),
        }
    )

    # Outcome-only targets in the search's frames (alpha_zero/training/muzero_targets).
    # Until 2026-11-12 values were labelled from player 0's perspective and
    # summed the env's shaped rewards across player changes, then added the
    # outcome again on top of the terminal reward that already carried it.
    rewards, target_values = outcome_only_targets(
        info.get("winner"), players, int(env.current_player)
    )
    # The absorbing state after the game is worth nothing.
    target_values.append(0.0)

    # Terminal policy (dummy but required for T+1)
    policies.append(np.zeros_like(policies[0]))
    # Dones needs to match others
    dones.append(True)

    return GameTrajectory(
        observations=observations,
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        policies=np.array(policies, dtype=np.float32),
        values=np.array(target_values, dtype=np.float32),
        dones=np.array(dones, dtype=bool),
    )


def _self_play_worker(
    _game_idx: int,
    network_state: dict,
    network_config: MuZeroConfig,
    run_config_dict: dict,
) -> GameTrajectory:
    """Worker function for parallel self-play (runs on CPU)."""
    # Recreate config and network in worker process
    config = MuZeroRunConfig(**run_config_dict)

    # Create network on CPU for worker
    network = MuZeroNetwork(network_config)
    network.load_state_dict(network_state)
    network.eval()

    # Create env and MCTS
    env = MicroCivEnv(board_size=config.board_size, max_turns=config.max_turns)
    mcts = create_mcts(network, config)

    # Run single game
    return self_play_game(env, mcts, config)


def run_self_play(
    network: MuZeroNetwork,
    config: MuZeroRunConfig,
    num_games: int,
) -> list[GameTrajectory]:
    """Run multiple self-play games (parallel if num_workers > 1)."""

    # If parallel workers requested and more than 1 game
    if config.num_workers > 1 and num_games > 1:
        return run_self_play_parallel(network, config, num_games)

    # Sequential self-play (default)
    env = MicroCivEnv(board_size=config.board_size, max_turns=config.max_turns)
    mcts = create_mcts(network, config)

    trajectories = []
    total_turns = 0

    for _ in tqdm(range(num_games), desc="Self-play"):
        trajectory = self_play_game(env, mcts, config)
        trajectories.append(trajectory)
        total_turns += len(trajectory.actions)

    logger.info(
        f"Self-play complete: {num_games} games, avg length: {total_turns / num_games:.1f} turns"
    )
    return trajectories


def run_self_play_parallel(
    network: MuZeroNetwork,
    config: MuZeroRunConfig,
    num_games: int,
) -> list[GameTrajectory]:
    """Run self-play games in parallel using multiple CPU workers."""
    num_workers = min(config.num_workers, num_games, cpu_count())
    logger.info(f"Parallel self-play: {num_games} games on {num_workers} workers")

    # Prepare network state for workers (move to CPU)
    network_state = {k: v.cpu() for k, v in network.state_dict().items()}
    network_config = network.config

    # Convert config to dict for serialization
    run_config_dict = {
        k: v for k, v in config.__dict__.items() if not k.startswith("_") and not callable(v)
    }
    # Force CPU for workers
    run_config_dict["device"] = "cpu"

    trajectories = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _self_play_worker,
                i,
                network_state,
                network_config,
                run_config_dict,
            )
            for i in range(num_games)
        ]

        for future in tqdm(as_completed(futures), total=num_games, desc="Self-play"):
            trajectory = future.result()
            trajectories.append(trajectory)

    total_turns = sum(len(t.actions) for t in trajectories)
    logger.info(
        f"Parallel self-play complete: {num_games} games, avg length: {total_turns / num_games:.1f} turns"
    )
    return trajectories


def train(config: MuZeroRunConfig, wandb_logger: Any = None) -> dict[str, float]:
    """Main training loop."""

    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Initialize environment to get action space
    env = MicroCivEnv(board_size=config.board_size)
    action_space_size = env.action_space.n

    # Create network
    network = create_network(config, action_space_size).to(config.device)
    param_count = sum(p.numel() for p in network.parameters())
    logger.info(f"Network parameters: {param_count:,}")
    logger.info(f"Action space: {action_space_size}")
    logger.info(f"Device: {config.device}")

    # Log config to W&B summary
    if wandb_logger and wandb_logger.is_initialized:
        wandb_logger.summary(
            {
                "network_params": param_count,
                "action_space": action_space_size,
            }
        )

    # Create trainer (trainer manages its own optimizer)
    trainer_config = MuZeroTrainingConfig(
        unroll_steps=config.unroll_steps,
        td_steps=config.td_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        use_consistency_loss=config.use_consistency_loss,
        use_value_prefix=config.use_value_prefix,
        consistency_loss_weight=config.consistency_loss_weight,
    )
    trainer = MuZeroTrainer(network, trainer_config, device=config.device)

    # Create trajectory buffer
    trajectory_buffer = TrajectoryBuffer(max_trajectories=config.replay_buffer_size // 100)

    # Training metrics
    best_avg_length = float("inf")
    training_stats = {
        "total_games": 0,
        "total_steps": 0,
        "best_avg_length": best_avg_length,
    }

    # Training loop
    for iteration in range(config.num_iterations):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Iteration {iteration + 1}/{config.num_iterations}")
        logger.info(f"{'=' * 60}")

        # Self-play
        start_time = time.time()
        trajectories = run_self_play(network, config, config.num_games_per_iteration)
        self_play_time = time.time() - start_time

        # Add to buffer
        for traj in trajectories:
            trajectory_buffer.add_trajectory(traj)

        training_stats["total_games"] += len(trajectories)
        avg_game_length = np.mean([len(t.actions) for t in trajectories])

        logger.info(
            f"Self-play: {len(trajectories)} games in {self_play_time:.1f}s, avg length: {avg_game_length:.1f}"
        )

        # Log self-play metrics
        if wandb_logger:
            wandb_logger.log(
                {
                    "iteration": iteration + 1,
                    "self_play/num_games": len(trajectories),
                    "self_play/avg_game_length": avg_game_length,
                    "self_play/time_seconds": self_play_time,
                    "self_play/buffer_size": len(trajectory_buffer),
                }
            )

        # Training
        if len(trajectory_buffer) >= config.min_replay_size // 100:
            logger.info("Training network...")
            start_time = time.time()

            # Sample and train
            num_batches = max(1, len(trajectory_buffer) * 10 // config.batch_size)
            epoch_losses = []

            for _batch_idx in range(num_batches):
                # Sample batch from buffer
                batch = trajectory_buffer.sample_positions(
                    batch_size=config.batch_size,
                    unroll_steps=config.unroll_steps,
                    device=config.device,
                )

                if batch is None:
                    continue

                # Training step (trainer has internal optimizer)
                losses = trainer.train_step(batch)
                epoch_losses.append(losses)
                training_stats["total_steps"] += 1

            train_time = time.time() - start_time

            # Aggregate losses
            if epoch_losses:
                avg_losses = {
                    key: np.mean([loss[key] for loss in epoch_losses if key in loss])
                    for key in epoch_losses[0].keys()
                }

                logger.info(
                    f"Training: {num_batches} batches in {train_time:.1f}s, "
                    f"total_loss={avg_losses.get('total_loss', 0):.4f}, "
                    f"policy={avg_losses.get('policy_loss', 0):.4f}, "
                    f"value={avg_losses.get('value_loss', 0):.4f}"
                )

                # Log training metrics
                if wandb_logger:
                    wandb_logger.log(
                        {
                            "iteration": iteration + 1,
                            "training/total_loss": avg_losses.get("total_loss", 0),
                            "training/policy_loss": avg_losses.get("policy_loss", 0),
                            "training/value_loss": avg_losses.get("value_loss", 0),
                            "training/reward_loss": avg_losses.get("reward_loss", 0),
                            "training/consistency_loss": avg_losses.get("consistency_loss", 0),
                            "training/num_batches": num_batches,
                            "training/time_seconds": train_time,
                            "training/total_steps": training_stats["total_steps"],
                        }
                    )

        # Track best model
        if avg_game_length < best_avg_length:
            best_avg_length = avg_game_length
            training_stats["best_avg_length"] = best_avg_length

        # Checkpointing
        if (iteration + 1) % config.save_interval == 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"muzero_iter_{iteration + 1}.pt"
            torch.save(
                {
                    "iteration": iteration + 1,
                    "network_state_dict": network.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "config": config.__dict__,
                    "stats": training_stats,
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            if wandb_logger:
                wandb_logger.log_model(str(checkpoint_path), f"muzero_iter_{iteration + 1}")

    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Training Complete!")
    logger.info(f"Total games: {training_stats['total_games']}")
    logger.info(f"Total training steps: {training_stats['total_steps']}")
    logger.info(f"Best avg game length: {training_stats['best_avg_length']:.1f}")
    logger.info(f"{'=' * 60}")

    return training_stats


#: Iterations and games per iteration the ``--quick`` / ``--fast`` presets fix.
QUICK_SHAPE = (2, 5)
FAST_SHAPE = (10, 10)


def resolve_iterations_and_games(args: argparse.Namespace) -> tuple[int, int]:
    """``(iterations, games_per_iteration)`` a parsed command line will play.

    The single source for both the run and its game-budget pricing, so the
    two cannot disagree about which flag wins.
    """
    if args.quick:
        return QUICK_SHAPE
    if args.fast:
        return FAST_SHAPE
    return args.iterations, args.games


def build_parser() -> argparse.ArgumentParser:
    """The runner's own argv parser (also used to price a command, e.g. by a sweep)."""
    parser = argparse.ArgumentParser(description="MuZero Training for Civ6")

    # Quick presets
    parser.add_argument(
        "--quick", action="store_true", help="Quick test run (2 iterations, 5 games)"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Fast training (10 iterations, 10 games)"
    )
    parser.add_argument(
        "--small", action="store_true", help="Use small 8x8 board for faster testing"
    )

    # Training params
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--games", type=int, default=20, help="Games per iteration")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--workers", type=int, default=1, help="Parallel self-play workers (1=sequential)"
    )

    # Network architecture
    parser.add_argument("--res-blocks", type=int, default=None, help="Number of residual blocks")
    parser.add_argument("--filters", type=int, default=None, help="Number of conv filters")

    # MCTS params
    parser.add_argument("--simulations", type=int, default=16, help="MCTS simulations")
    parser.add_argument("--no-gumbel", action="store_true", help="Disable Gumbel MCTS")

    # EfficientZero
    parser.add_argument("--no-consistency", action="store_true", help="Disable consistency loss")
    parser.add_argument("--no-value-prefix", action="store_true", help="Disable value prefix")

    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="civ6-muzero", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name")

    # Other
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/muzero", help="Checkpoint directory"
    )

    add_budget_arguments(parser)
    return parser


def main():
    args = build_parser().parse_args()

    # Build config
    config = MuZeroRunConfig()

    if args.quick:
        config.num_iterations, config.num_games_per_iteration = QUICK_SHAPE
        config.min_replay_size = 100
        config.num_simulations = 8
        logger.info("Quick test mode: 2 iterations, 5 games each")
    elif args.fast:
        config.num_iterations, config.num_games_per_iteration = FAST_SHAPE
        config.min_replay_size = 200
        logger.info("Fast training mode: 10 iterations, 10 games each")
    else:
        config.num_iterations, config.num_games_per_iteration = resolve_iterations_and_games(args)

    # Small board for faster testing
    if args.small:
        config.board_size = 8
        config.max_turns = 50
        config.num_simulations = 4
        config.num_res_blocks = 2
        config.num_filters = 32
        logger.info("Small board mode: 8x8 board, 4 simulations, 50 max turns")

    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.num_simulations = args.simulations
    config.num_workers = args.workers
    config.use_gumbel = not args.no_gumbel
    config.use_consistency_loss = not args.no_consistency
    config.use_value_prefix = not args.no_value_prefix
    config.checkpoint_dir = args.checkpoint_dir

    # Network architecture (override --small if specified)
    if args.res_blocks is not None:
        config.num_res_blocks = args.res_blocks
    if args.filters is not None:
        config.num_filters = args.filters

    if args.device:
        config.device = args.device

    enforce_cli(
        "run_muzero_training",
        {
            "num_iterations": config.num_iterations,
            "games_per_iteration": config.num_games_per_iteration,
        },
        args,
    )

    # Log parallel config
    if config.num_workers > 1:
        logger.info(f"Parallel self-play: {config.num_workers} workers (CPU)")

    config.wandb_enabled = args.wandb
    config.wandb_project = args.wandb_project
    config.wandb_run_name = args.wandb_name

    # Initialize W&B
    wandb_logger = None
    if config.wandb_enabled:
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available. Install with: pip install wandb")
        else:
            try:
                wandb_config = WandbConfig(
                    enabled=True,
                    experiment_name=config.wandb_project,
                    run_name=config.wandb_run_name,
                    tags={
                        "muzero": "true",
                        ("gumbel" if config.use_gumbel else "puct"): "true",
                    },
                )
                wandb_logger = WandbLogger(wandb_config)
                # Pass config dict to init for logging
                config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
                wandb_logger.init(config_dict=config_dict)
                logger.info(f"W&B initialized: {config.wandb_project}")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}")

    try:
        # Run training
        stats = train(config, wandb_logger)

        # Final log
        if wandb_logger:
            wandb_logger.log(
                {
                    "final/total_games": stats["total_games"],
                    "final/total_steps": stats["total_steps"],
                    "final/best_avg_length": stats["best_avg_length"],
                }
            )

        # Machine-readable result line for hyperparameter sweeps: the
        # SubprocessTrainingObjective in
        # alpha_zero/optimizations/hyperparameter_search.py extracts the
        # optimized metric from this marker.
        json_safe_stats = {k: (v.item() if hasattr(v, "item") else v) for k, v in stats.items()}
        print(f"SWEEP_METRICS_JSON={json.dumps(json_safe_stats)}")
    finally:
        if wandb_logger:
            wandb_logger.finish()


if __name__ == "__main__":
    main()
