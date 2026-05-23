"""Enhanced training script with all training improvements.

This script integrates:
- Environment tuning for faster game convergence
- Enhanced reward signals for better learning
- Curriculum learning for progressive difficulty
- Enhanced training configuration with schedules
- Weights & Biases (W&B) integration for experiment tracking

Usage:
    python run_enhanced_training.py --preset fast  # Quick test
    python run_enhanced_training.py --preset balanced  # Medium training
    python run_enhanced_training.py --iterations 50 --games 20  # Custom
    python run_enhanced_training.py --wandb  # Enable W&B logging
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm  # type: ignore

from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig, SimpleMCTS
from alpha_zero.core.civ6_network import MicroCivNetwork, compute_loss
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.utils.curriculum_learning import (
    CurriculumConfig,
    CurriculumManager,
)
from alpha_zero.utils.enhanced_rewards import (
    EnhancedRewardShaper,
    create_enhanced_reward_shaper,
)
from alpha_zero.utils.environment_tuning import (
    EnvironmentTuningConfig,
    create_tuned_environment,
    get_default_balanced_config,
    get_default_fast_config,
)

# Import W&B logger (optional)
try:
    from alpha_zero.utils.wandb_logger import WandbConfig, WandbLogger
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    WandbLogger = None  # type: ignore
    WandbConfig = None  # type: ignore

from alpha_zero.utils.logging import get_logger, setup_logging

# Configure logging
# Configure logging
# setup_logging moved to main to avoid side effects on import
logger = get_logger(__name__)


@dataclass
class TrainingExample:
    """Single training example."""
    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    """Replay buffer for storing training examples."""

    def __init__(self, max_size: int = 50000):
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add(self, example: TrainingExample) -> None:
        self.buffer.append(example)

    def add_game(self, examples: list[TrainingExample]) -> None:
        for example in examples:
            self.add(example)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class ReplayDataset(Dataset):
    """PyTorch dataset for replay buffer."""

    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "spatial": torch.from_numpy(ex.spatial).float(),
            "vector": torch.from_numpy(ex.vector).float(),
            "action_mask": torch.from_numpy(ex.action_mask).float(),
            "policy": torch.from_numpy(ex.policy).float(),
            "value": torch.tensor(ex.value).float(),
        }


def self_play_game_enhanced(
    env,
    mcts: Civ6MCTS | SimpleMCTS,
    reward_shaper: EnhancedRewardShaper | None,
    temperature: float = 1.0,
    temperature_threshold: int = 30,
) -> tuple[list[TrainingExample], dict[str, Any]]:
    """Play a single self-play game with enhanced rewards.

    Returns:
        Tuple of (training examples, game stats)
    """
    examples = []
    # ``info`` must be in scope BEFORE the loop so the step-limit guard below
    # can record ``truncated_by_step_limit`` without raising NameError when
    # the guard fires before the first ``env.step`` returns one. Seeded with
    # reset()'s info dict so any keys the env populates on initial state are
    # preserved (e.g. ``early_termination`` read on line ~216).
    obs, info = env.reset()

    if reward_shaper:
        reward_shaper.reset()

    game_history = []
    total_shaped_reward = 0.0
    total_base_reward = 0.0
    step_count = 0

    # Hard step limit to prevent infinite loops (e.g. if agent never ends turn)
    # Use max_turns if available, otherwise default to a safe value
    max_turns = getattr(env, 'max_turns', 100) or 100
    max_steps = max_turns * 20

    while True:
        if step_count >= max_steps:
            logger.warning(f"Game hit hard step limit ({max_steps}) - forcing termination.")
            # Record truncation in info for statistics
            info["truncated_by_step_limit"] = True
            break

        # Get MCTS policy
        # Use underlying environment for MCTS
        underlying_env = env.env if hasattr(env, 'env') else env
        action_probs, _ = mcts.search(underlying_env)

        # Store state and policy
        game_history.append({
            "spatial": obs["spatial"].copy(),
            "vector": obs["vector"].copy(),
            "action_mask": obs["action_mask"].copy(),
            "policy": action_probs.copy(),
            "player": underlying_env.current_player,
        })

        # Select action with temperature and action mask enforcement
        temp = temperature if underlying_env.turn < temperature_threshold else 0.0
        action = mcts.select_action(
            action_probs, temp, action_mask=obs["action_mask"]
        )

        # Take action
        obs, base_reward, terminated, truncated, info = env.step(action)
        step_count += 1

        # Shape reward if shaper available
        if reward_shaper:
            shaped_reward = reward_shaper.shape_reward(
                obs, base_reward, info, terminated=terminated
            )
            total_shaped_reward += shaped_reward
        total_base_reward += base_reward

        if terminated or truncated:
            break

    # Assign values based on game outcome
    winner = info.get("winner")

    # Bootstrap value for draws using shaped rewards
    # Map shaped reward sum to [-0.8, 0.8] range to leave room for terminal win/loss (1.0/-1.0)
    draw_value = np.tanh(total_shaped_reward) * 0.8 if total_shaped_reward != 0 else 0.0

    for hist in game_history:
        if winner is None:
            value = draw_value
        elif winner == hist["player"]:
            value = 1.0
        else:
            value = -1.0

        examples.append(TrainingExample(
            spatial=hist["spatial"],
            vector=hist["vector"],
            action_mask=hist["action_mask"],
            policy=hist["policy"],
            value=value,
        ))

    game_stats = {
        "steps": step_count,
        "base_reward": total_base_reward,
        "shaped_reward": total_shaped_reward,
        "winner": winner,
        "early_termination": info.get("early_termination", False),
    }

    return examples, game_stats


def run_self_play_enhanced(
    network: torch.nn.Module,
    env_config: EnvironmentTuningConfig,
    curriculum: CurriculumManager,
    reward_shaper: EnhancedRewardShaper | None,
    num_games: int,
    num_simulations: int = 50,
    device: str = "cpu",
    use_simple_mcts: bool = False,
) -> tuple[list[TrainingExample], list[dict[str, Any]]]:
    """Run multiple self-play games with training improvements.

    Returns:
        Tuple of (all examples, list of game stats)
    """
    all_examples = []
    all_stats = []

    # Get curriculum parameters
    curriculum_params = curriculum.get_environment_params()

    # Override env config with curriculum settings
    env_config.board_size = curriculum_params.get("board_size", env_config.board_size)
    env_config.max_turns = curriculum_params.get("max_turns", env_config.max_turns)
    env_config.fog_of_war_enabled = curriculum_params.get("fog_of_war_enabled", env_config.fog_of_war_enabled)

    # Create tuned environment
    env = create_tuned_environment(env_config, board_size=env_config.board_size)

    # Create MCTS
    if use_simple_mcts:
        mcts: Civ6MCTS | SimpleMCTS = SimpleMCTS(
            num_simulations=num_simulations,
            c_puct=1.5,
        )
    else:
        mcts_config = MCTSConfig(
            num_simulations=num_simulations,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        )
        mcts = Civ6MCTS(network, mcts_config, device=device)

    wins = 0
    for _ in tqdm(range(num_games), desc="Self-play"):
        examples, stats = self_play_game_enhanced(
            env, mcts, reward_shaper,
            temperature=1.0,
            temperature_threshold=30,
        )
        all_examples.extend(examples)
        all_stats.append(stats)

        # Record game result in curriculum
        won = stats["winner"] == 0  # Player 0 is the learning agent
        curriculum.record_game(won=won)
        if won:
            wins += 1

    logger.info(f"Self-play: {wins}/{num_games} wins ({100*wins/num_games:.1f}%)")

    return all_examples, all_stats


def train_epoch(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: str,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    """Train network for one epoch."""
    network.train()
    total_loss = 0.0
    policy_loss = 0.0
    value_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        spatial = batch["spatial"].to(device)
        vector = batch["vector"].to(device)
        action_mask = batch["action_mask"].to(device)
        target_policy = batch["policy"].to(device)
        target_value = batch["value"].to(device)

        optimizer.zero_grad()

        # Forward pass
        policy_logits, value = network(spatial, vector, action_mask)

        # Compute loss
        loss, p_loss, v_loss = compute_loss(
            policy_logits, value, target_policy, target_value, action_mask
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(network.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        policy_loss += p_loss.item()
        value_loss += v_loss.item()
        num_batches += 1

    return {
        "total_loss": total_loss / max(num_batches, 1),
        "policy_loss": policy_loss / max(num_batches, 1),
        "value_loss": value_loss / max(num_batches, 1),
    }


def evaluate_network(
    network: torch.nn.Module,
    board_size: int,
    max_turns: int,
    device: str,
    num_games: int = 20,
    num_simulations: int = 25,
) -> dict[str, float]:
    """Evaluate network against random player.

    Returns:
        Dictionary with evaluation metrics.
    """
    tuning_config = get_default_balanced_config()
    tuning_config.max_turns = max_turns
    tuning_config.board_size = board_size
    tuning_config.early_termination.enabled = True

    env = create_tuned_environment(tuning_config)
    mcts_config = MCTSConfig(num_simulations=num_simulations)
    mcts = Civ6MCTS(network, mcts_config, device=device)

    wins = 0
    total_steps = 0
    rng = np.random.default_rng()

    # Safety limit: max steps per game to prevent infinite loops
    max_steps_per_game = max_turns * 4  # Allow some margin

    for game_idx in range(num_games):
        obs, _ = env.reset()
        network_player = game_idx % 2  # Alternate who goes first
        steps = 0

        while steps < max_steps_per_game:
            if env.current_player == network_player:
                # Network's turn
                underlying_env = env.env if hasattr(env, 'env') else env
                action_probs, _ = mcts.search(underlying_env, add_noise=False)
                action = mcts.select_action(
                    action_probs, temperature=0.0, action_mask=obs["action_mask"]
                )
            else:
                # Random player's turn
                mask = env.get_valid_actions()
                valid = np.where(mask > 0)[0]
                action = rng.choice(valid) if len(valid) > 0 else 0

            obs, _, terminated, truncated, info = env.step(action)
            steps += 1

            if terminated or truncated:
                winner = info.get("winner")
                if winner == network_player:
                    wins += 1
                total_steps += steps
                break

        # Handle step limit reached (treat as draw)
        if steps >= max_steps_per_game:
            total_steps += steps
            logger.warning(f"Eval game {game_idx} hit step limit ({max_steps_per_game})")

    return {
        "win_rate": wins / num_games,
        "avg_game_length": total_steps / num_games,
    }


def run_enhanced_training(
    num_iterations: int = 50,
    games_per_iteration: int = 20,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    num_epochs: int = 5,
    num_simulations: int = 30,
    board_size: int = 8,
    max_turns: int = 50,
    checkpoint_dir: str = "checkpoints/enhanced",
    eval_interval: int = 5,
    device: str | None = None,
    use_curriculum: bool = True,
    use_reward_shaping: bool = True,
    use_env_tuning: bool = True,
    wandb_enabled: bool = False,
    wandb_project: str = "civ6-alphazero-enhanced",
    wandb_run_name: str | None = None,
    wandb_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Run enhanced training with all improvements.

    Args:
        num_iterations: Number of training iterations
        games_per_iteration: Self-play games per iteration
        batch_size: Training batch size
        learning_rate: Initial learning rate
        num_epochs: Training epochs per iteration
        num_simulations: MCTS simulations
        board_size: Initial board size
        max_turns: Initial max turns
        checkpoint_dir: Directory for checkpoints
        eval_interval: Iterations between evaluations
        device: Compute device
        use_curriculum: Enable curriculum learning
        use_reward_shaping: Enable reward shaping
        use_env_tuning: Enable environment tuning
        wandb_enabled: Enable W&B logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name (auto-generated if None)
        wandb_tags: W&B tags for the run

    Returns:
        Training history dictionary
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize W&B logger
    wandb_logger: WandbLogger | None = None
    if wandb_enabled and WANDB_AVAILABLE:
        wandb_config = WandbConfig(
            enabled=True,
            project=wandb_project,
            name=wandb_run_name,
            tags=wandb_tags or ["enhanced-training", "curriculum", "reward-shaping"],
            mode="online",
        )
        wandb_logger = WandbLogger(wandb_config)

        # Initialize W&B run with training config
        config_dict = {
            "num_iterations": num_iterations,
            "games_per_iteration": games_per_iteration,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "num_simulations": num_simulations,
            "board_size": board_size,
            "max_turns": max_turns,
            "use_curriculum": use_curriculum,
            "use_reward_shaping": use_reward_shaping,
            "use_env_tuning": use_env_tuning,
            "device": device,
        }
        run = wandb_logger.init(config_dict=config_dict)
        if run:
            logger.info(f"W&B run initialized: {run.name}")
    elif wandb_enabled and not WANDB_AVAILABLE:
        logger.warning("W&B requested but not available. Install with: pip install wandb")

    logger.info("=" * 60)
    logger.info("ENHANCED TRAINING WITH IMPROVEMENTS")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Iterations: {num_iterations}")
    logger.info(f"Games/iteration: {games_per_iteration}")
    logger.info(f"Board size: {board_size}, Max turns: {max_turns}")
    logger.info(f"Curriculum: {use_curriculum}, Reward shaping: {use_reward_shaping}, Env tuning: {use_env_tuning}")
    logger.info("=" * 60)

    # Initialize curriculum
    if use_curriculum:
        curriculum_config = CurriculumConfig.default_three_stage()
        # Reduce min_games for faster progression during short training
        for stage in curriculum_config.stages:
            stage.min_games = max(10, games_per_iteration)
        curriculum_config.save_path = os.path.join(checkpoint_dir, "curriculum_state.json")
        curriculum = CurriculumManager(curriculum_config)
        logger.info(f"Curriculum: {len(curriculum_config.stages)} stages")
    else:
        # Create dummy curriculum that doesn't change
        curriculum_config = CurriculumConfig(
            stages=[CurriculumConfig._default_stages()[0]]
        )
        curriculum_config.stages[0].board_size = board_size
        curriculum_config.stages[0].max_turns = max_turns
        curriculum = CurriculumManager(curriculum_config)

    # Get initial environment params from curriculum
    env_params = curriculum.get_environment_params()
    current_board_size = env_params["board_size"]
    current_max_turns = env_params["max_turns"]

    # Initialize environment config
    if use_env_tuning:
        env_config = get_default_fast_config()
        env_config.board_size = current_board_size
        env_config.max_turns = current_max_turns
    else:
        env_config = EnvironmentTuningConfig(
            enabled=False,
            board_size=current_board_size,
            max_turns=current_max_turns,
        )

    # Initialize environment to get action space size
    temp_env = MicroCivEnv(board_size=current_board_size)
    action_space_size = temp_env.action_space_handler.total_size

    # Initialize network
    network = MicroCivNetwork(
        board_size=current_board_size,
        action_space_size=action_space_size,
        num_res_blocks=4,
        num_filters=32,
    ).to(device)

    param_count = sum(p.numel() for p in network.parameters())
    logger.info(f"Network parameters: {param_count:,}")
    logger.info(f"Action space: {action_space_size}")

    # Initialize reward shaper
    if use_reward_shaping:
        reward_shaper = create_enhanced_reward_shaper(
            scale_factor=1.0,
            board_size=current_board_size,
        )
    else:
        reward_shaper = None

    # Initialize optimizer
    optimizer = optim.Adam(
        network.parameters(),
        lr=learning_rate,
        weight_decay=0.0001,
    )

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=50000)

    # Training history
    history: dict[str, Any] = {
        "iterations": [],
        "losses": [],
        "win_rates": [],
        "curriculum_stages": [],
        "game_stats": [],
    }

    # Training loop
    for iteration in range(num_iterations):
        iter_start = time.time()

        logger.info(f"\n{'='*50}")
        logger.info(f"Iteration {iteration + 1}/{num_iterations}")

        # Check curriculum progress
        progress = curriculum.get_progress()
        logger.info(f"Curriculum: {progress['current_stage']} (level {progress['current_level']})")
        logger.info(f"Win rate: {progress['win_rate']:.2%}")

        history["curriculum_stages"].append(progress["current_stage"])

        # Update environment config from curriculum
        env_params = curriculum.get_environment_params()
        env_config.board_size = env_params["board_size"]
        env_config.max_turns = env_params["max_turns"]

        # Self-play
        logger.info("Running self-play...")
        use_simple = iteration == 0
        examples, game_stats = run_self_play_enhanced(
            network=network,
            env_config=env_config,
            curriculum=curriculum,
            reward_shaper=reward_shaper,
            num_games=games_per_iteration,
            num_simulations=num_simulations,
            device=device,
            use_simple_mcts=use_simple,
        )
        replay_buffer.add_game(examples)
        history["game_stats"].append(game_stats)

        logger.info(f"Generated {len(examples)} examples from {games_per_iteration} games")
        logger.info(f"Buffer size: {len(replay_buffer)}")

        # Log self-play stats to W&B
        if wandb_logger and wandb_logger.is_initialized:
            wins = sum(1 for s in game_stats if s.get("winner") == 0)
            avg_steps = sum(s["steps"] for s in game_stats) / len(game_stats)

            # Use iteration as the step for non-training metrics to keep them sync'd
            wandb_logger.log_selfplay(
                games_completed=curriculum.get_progress()["total_games"],
                buffer_size=len(replay_buffer),
                samples_generated=len(examples),
                win_rate=wins / len(game_stats),
                avg_game_length=avg_steps,
                step=iteration # Explicitly set step to iteration
            )
            # Log curriculum progress
            wandb_logger.log({
                "curriculum/stage": progress["current_level"],
                "curriculum/stage_name": progress["current_stage"],
                "curriculum/win_rate": progress["win_rate"],
                "curriculum/total_games": progress["total_games"],
            }, step=iteration)

        # Training
        if len(replay_buffer) >= batch_size:
            logger.info("Training network...")
            sample_size = min(len(replay_buffer), batch_size * 50)
            train_examples = replay_buffer.sample(sample_size)

            dataset = ReplayDataset(train_examples)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )

            epoch_losses = []
            global_step = iteration * num_epochs
            for epoch in range(num_epochs):
                losses = train_epoch(network, optimizer, dataloader, device)
                epoch_losses.append(losses)
                logger.info(
                    f"Epoch {epoch + 1}: loss={losses['total_loss']:.4f}, "
                    f"policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f}"
                )

                # Log training losses to W&B
                if wandb_logger and wandb_logger.is_initialized:
                    wandb_logger.log_training_step(
                        step=global_step + epoch,
                        loss=losses["total_loss"],
                        policy_loss=losses["policy_loss"],
                        value_loss=losses["value_loss"],
                        learning_rate=learning_rate,
                    )

            history["losses"].append(epoch_losses[-1])
        else:
            logger.info(f"Skipping training (buffer {len(replay_buffer)} < {batch_size})")

        # Evaluation
        if (iteration + 1) % eval_interval == 0:
            logger.info("Evaluating...")
            eval_metrics = evaluate_network(
                network=network,
                board_size=int(env_config.board_size or 8),
                max_turns=int(env_config.max_turns or 100),
                device=device,
                num_games=10,
                num_simulations=num_simulations // 2,
            )
            logger.info(f"Win rate vs random: {eval_metrics['win_rate']:.1%}")
            logger.info(f"Avg game length: {eval_metrics['avg_game_length']:.1f}")
            history["win_rates"].append(eval_metrics["win_rate"])

            # Log evaluation metrics to W&B
            if wandb_logger and wandb_logger.is_initialized:
                wandb_logger.log_evaluation(
                    win_rate=eval_metrics["win_rate"],
                    games_played=10,
                    average_reward=eval_metrics["win_rate"] * 2 - 1,  # Map to [-1, 1]
                    average_turns=eval_metrics["avg_game_length"],
                    step=iteration # Use current iteration as step
                )

            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration + 1}.pt")
            network.save(checkpoint_path)
            logger.info(f"Saved: {checkpoint_path}")

            # Log model artifact to W&B
            if wandb_logger and wandb_logger.is_initialized:
                wandb_logger.log_model(
                    model_path=checkpoint_path,
                    name=f"model-iter-{iteration + 1}",
                    metadata={"iteration": iteration + 1, "win_rate": eval_metrics["win_rate"]},
                )

        history["iterations"].append(iteration + 1)
        iter_time = time.time() - iter_start
        logger.info(f"Iteration time: {iter_time:.1f}s")

    # Final evaluation
    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 50)

    final_metrics = evaluate_network(
        network=network,
        board_size=int(env_config.board_size or 8),
        max_turns=int(env_config.max_turns or 100),
        device=device,
        num_games=20,
        num_simulations=num_simulations,
    )
    logger.info(f"Final win rate vs random: {final_metrics['win_rate']:.1%}")
    logger.info(f"Final avg game length: {final_metrics['avg_game_length']:.1f}")

    # Save final model
    final_path = os.path.join(checkpoint_dir, "latest.pt")
    network.save(final_path)
    logger.info(f"Final model saved: {final_path}")

    # Save training history
    history["final_metrics"] = final_metrics
    curriculum_progress = curriculum.get_progress()
    history["final_curriculum"] = curriculum_progress

    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total games: {curriculum_progress['total_games']}")
    logger.info(f"Final stage: {curriculum_progress['current_stage']}")
    logger.info(f"Final win rate: {final_metrics['win_rate']:.1%}")

    # Log final summary and close W&B
    if wandb_logger and wandb_logger.is_initialized:
        # Log final evaluation
        wandb_logger.log_evaluation(
            win_rate=final_metrics["win_rate"],
            games_played=20,
            average_reward=final_metrics["win_rate"] * 2 - 1,
            average_turns=final_metrics["avg_game_length"],
        )

        # Update summary
        wandb_logger.summary({
            "final_win_rate": final_metrics["win_rate"],
            "final_avg_game_length": final_metrics["avg_game_length"],
            "total_iterations": num_iterations,
            "total_games": curriculum_progress["total_games"],
            "final_curriculum_stage": curriculum_progress["current_stage"],
            "final_curriculum_level": curriculum_progress["current_level"],
        })

        # Log final model
        wandb_logger.log_model(
            model_path=final_path,
            name="model-final",
            metadata={
                "win_rate": final_metrics["win_rate"],
                "total_games": curriculum_progress["total_games"],
            },
        )

        # Finish run
        wandb_logger.finish()
        logger.info("W&B run finished")

    return history


def main():
    setup_logging(log_dir="logs", level="INFO")
    parser = argparse.ArgumentParser(description="Enhanced Civ6 AlphaZero Training")
    parser.add_argument("--preset", choices=["fast", "balanced", "full"], default="fast",
                        help="Training preset (fast=quick test, balanced=medium, full=production)")
    parser.add_argument("--iterations", type=int, default=None, help="Override num iterations")
    parser.add_argument("--games", type=int, default=None, help="Override games per iteration")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--board_size", type=int, default=None, help="Override board size")
    parser.add_argument("--simulations", type=int, default=None, help="Override MCTS simulations")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/enhanced")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--no-reward-shaping", action="store_true", help="Disable reward shaping")
    parser.add_argument("--no-env-tuning", action="store_true", help="Disable environment tuning")
    # W&B arguments
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="civ6-alphazero-enhanced",
                        help="W&B project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None,
                        help="W&B tags for the run")
    args = parser.parse_args()

    # Preset configurations
    presets = {
        "fast": {
            "num_iterations": 10,
            "games_per_iteration": 10,
            "batch_size": 32,
            "num_epochs": 3,
            "num_simulations": 20,
            "board_size": 8,
            "max_turns": 30,
            "eval_interval": 5,
        },
        "balanced": {
            "num_iterations": 50,
            "games_per_iteration": 30,
            "batch_size": 64,
            "num_epochs": 5,
            "num_simulations": 30,
            "board_size": 10,
            "max_turns": 50,
            "eval_interval": 10,
        },
        "full": {
            "num_iterations": 200,
            "games_per_iteration": 100,
            "batch_size": 128,
            "num_epochs": 10,
            "num_simulations": 50,
            "board_size": 12,
            "max_turns": 100,
            "eval_interval": 20,
        },
    }

    config = presets[args.preset]

    # Apply overrides
    if args.iterations:
        config["num_iterations"] = args.iterations
    if args.games:
        config["games_per_iteration"] = args.games
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.board_size:
        config["board_size"] = args.board_size
    if args.simulations:
        config["num_simulations"] = args.simulations

    # Run training
    history = run_enhanced_training(
        num_iterations=config["num_iterations"],
        games_per_iteration=config["games_per_iteration"],
        batch_size=config["batch_size"],
        num_epochs=config["num_epochs"],
        num_simulations=config["num_simulations"],
        board_size=config["board_size"],
        max_turns=config["max_turns"],
        checkpoint_dir=args.checkpoint_dir,
        eval_interval=config["eval_interval"],
        device=args.device,
        use_curriculum=not args.no_curriculum,
        use_reward_shaping=not args.no_reward_shaping,
        use_env_tuning=not args.no_env_tuning,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )

    return history


if __name__ == "__main__":
    main()
