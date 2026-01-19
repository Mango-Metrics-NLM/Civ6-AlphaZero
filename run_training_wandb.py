"""Training script with W&B integration for Civ6 AlphaZero.

This module provides modular training functions for the Civ6 AlphaZero agent.
Functions can be used independently or orchestrated via the main() function.

Enhanced with reward shaping, MCTS exploration, and training diagnostics.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces

from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.utils.wandb_logger import WandbLogger, WandbConfig
from alpha_zero.utils.reward_shaping import RewardShaper, RewardShapingConfig, create_reward_shaper
from alpha_zero.utils.simple_mcts import SimpleMCTS, MCTSConfig, create_mcts
from alpha_zero.utils.training_diagnostics import TrainingDiagnostics, create_diagnostics


@dataclass
class TrainingConfig:
    """Configuration for training."""

    board_size: int = 8
    num_games: int = 20
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 0.001
    max_turns_per_game: int = 100
    eval_games: int = 10
    grad_clip_norm: float = 1.0

    # Enhanced training options
    use_reward_shaping: bool = True
    use_mcts_exploration: bool = True
    temperature: float = 1.0
    temperature_decay: float = 0.95
    min_temperature: float = 0.1
    lr_decay: float = 0.99
    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints"

    # MCTS configuration
    mcts_dirichlet_alpha: float = 0.3
    mcts_dirichlet_epsilon: float = 0.25

    # Reward shaping configuration
    reward_shaping_unit_weight: float = 0.05
    reward_shaping_city_weight: float = 0.1
    reward_shaping_health_weight: float = 0.02
    reward_shaping_territory_weight: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "board_size": self.board_size,
            "num_games": self.num_games,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_turns_per_game": self.max_turns_per_game,
            "eval_games": self.eval_games,
            "grad_clip_norm": self.grad_clip_norm,
            "use_reward_shaping": self.use_reward_shaping,
            "use_mcts_exploration": self.use_mcts_exploration,
            "temperature": self.temperature,
            "temperature_decay": self.temperature_decay,
            "min_temperature": self.min_temperature,
            "lr_decay": self.lr_decay,
            "checkpoint_interval": self.checkpoint_interval,
            "mcts_dirichlet_alpha": self.mcts_dirichlet_alpha,
            "mcts_dirichlet_epsilon": self.mcts_dirichlet_epsilon,
            "reward_shaping_unit_weight": self.reward_shaping_unit_weight,
            "reward_shaping_city_weight": self.reward_shaping_city_weight,
        }


@dataclass
class GameResult:
    """Result of a single game."""

    game_id: str
    turns: int
    total_reward: float
    game_time: float
    experiences: list[dict[str, Any]] = field(default_factory=list)
    shaped_reward: Optional[float] = None
    outcome: str = "timeout"


@dataclass
class TrainingResult:
    """Result of training phase."""

    total_steps: int
    final_loss: float
    final_policy_loss: float
    final_value_loss: float
    epoch_losses: list[float] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Result of evaluation phase."""

    win_rate: float
    games_played: int
    total_wins: int
    average_turns: float
    average_reward: float


def select_action(policy: np.ndarray, mask: np.ndarray) -> Optional[int]:
    """Select action with proper handling of edge cases.

    Args:
        policy: Policy probabilities from network (may contain invalid values).
        mask: Binary mask of valid actions.

    Returns:
        Selected action index, or None if no valid actions.
    """
    valid_indices = np.where(mask > 0)[0]
    if len(valid_indices) == 0:
        return None

    # Get probabilities for valid actions only
    valid_probs = policy[valid_indices]
    valid_probs = np.maximum(valid_probs, 0)  # Ensure non-negative

    prob_sum = valid_probs.sum()
    if prob_sum <= 0 or np.isnan(prob_sum) or np.isinf(prob_sum):
        # Uniform random if probabilities are invalid
        return int(np.random.choice(valid_indices))

    valid_probs = valid_probs / prob_sum
    return int(np.random.choice(valid_indices, p=valid_probs))


def run_self_play(
    env: MicroCivEnv,
    network: MicroCivNetwork,
    config: TrainingConfig,
    wandb_logger: Optional[WandbLogger] = None,
    verbose: bool = True,
    reward_shaper: Optional[RewardShaper] = None,
    mcts: Optional[SimpleMCTS] = None,
    diagnostics: Optional[TrainingDiagnostics] = None,
    current_temperature: float = 1.0,
) -> tuple[list[dict[str, Any]], list[GameResult]]:
    """Run self-play games to generate training data.

    Args:
        env: The game environment.
        network: Neural network for policy/value prediction.
        config: Training configuration.
        wandb_logger: Optional W&B logger for metrics.
        verbose: Whether to print progress.
        reward_shaper: Optional reward shaper for intermediate signals.
        mcts: Optional MCTS for exploration.
        diagnostics: Optional training diagnostics tracker.
        current_temperature: Current temperature for action selection.

    Returns:
        Tuple of (replay_buffer, game_results).
    """
    replay_buffer: list[dict[str, Any]] = []
    game_results: list[GameResult] = []

    network.eval()

    if verbose:
        print(f"Starting self-play ({config.num_games} games, temp={current_temperature:.2f})...")

    for game_idx in range(config.num_games):
        obs, _ = env.reset()
        game_data: list[dict[str, Any]] = []
        total_reward = 0.0
        turn = 0
        done = False

        # Reset reward shaper for new episode
        if reward_shaper is not None:
            reward_shaper.reset()

        start_time = time.time()

        while not done and turn < config.max_turns_per_game:
            # Get network policy (returns logits)
            spatial = torch.tensor(obs["spatial"]).unsqueeze(0).float()
            vector = torch.tensor(obs["vector"]).unsqueeze(0).float()

            with torch.no_grad():
                policy_logits, value = network(spatial, vector)
                # Apply softmax to get probabilities
                policy = torch.softmax(policy_logits, dim=-1)

            policy_np = policy.squeeze().numpy()
            value_np = float(value.squeeze().numpy())
            mask = obs["action_mask"]

            # Use MCTS for exploration if available
            if mcts is not None:
                action, improved_policy = mcts.select_action(
                    policy_np, mask, add_noise=True
                )
                # Use improved policy as training target
                target_policy = improved_policy
            else:
                action = select_action(policy_np, mask)
                if action is None:
                    break
                # Create masked policy for training target
                masked_policy = policy_np * mask
                mask_sum = masked_policy.sum()
                if mask_sum > 0:
                    target_policy = masked_policy / mask_sum
                else:
                    target_policy = mask / mask.sum()

            # Track diagnostics
            if diagnostics is not None:
                diagnostics.on_step(policy_np, value_np, action, mask)

            # Store experience
            game_data.append({
                "spatial": obs["spatial"].copy(),
                "vector": obs["vector"].copy(),
                "policy": target_policy,
                "action": action,
            })

            # Step
            obs, reward, terminated, truncated, info = env.step(action)

            # Apply reward shaping if available
            if reward_shaper is not None:
                shaped_reward = reward_shaper.shape_reward(
                    obs, float(reward), terminated=terminated
                )
            else:
                shaped_reward = float(reward)

            total_reward += float(reward)
            done = terminated or truncated
            turn += 1

        game_time = time.time() - start_time

        # Determine outcome
        if total_reward > 0:
            outcome = "win"
        elif total_reward < 0:
            outcome = "loss"
        elif done and not truncated:
            outcome = "draw"
        else:
            outcome = "timeout"

        # Get shaped reward total if available
        shaped_reward_total = None
        if reward_shaper is not None:
            shaped_reward_total = reward_shaper.episode_shaped_reward_sum

        # Add value targets and append to replay buffer
        # Use shaped reward for value target if available
        value_target = shaped_reward_total if shaped_reward_total is not None else total_reward
        for data in game_data:
            data["value"] = value_target
            replay_buffer.append(data)

        game_result = GameResult(
            game_id=f"game_{game_idx}",
            turns=turn,
            total_reward=total_reward,
            game_time=game_time,
            experiences=game_data,
            shaped_reward=shaped_reward_total,
            outcome=outcome,
        )
        game_results.append(game_result)

        # Track episode end in diagnostics
        if diagnostics is not None:
            diagnostics.on_episode_end(
                episode_id=game_idx,
                turns=turn,
                total_reward=total_reward,
                outcome=outcome,
                shaped_reward=shaped_reward_total,
            )

        # Log game to W&B
        if wandb_logger is not None:
            wandb_logger.log_game(
                game_id=game_result.game_id,
                winner=0 if total_reward > 0 else 1,
                total_turns=turn,
                reward=total_reward,
                game_length_seconds=game_time,
            )

        if verbose:
            shaped_str = f", shaped={shaped_reward_total:.2f}" if shaped_reward_total is not None else ""
            print(f"  Game {game_idx+1}/{config.num_games}: turns={turn}, reward={total_reward:.2f}{shaped_str}, outcome={outcome}")

    # Log self-play summary
    if wandb_logger is not None:
        wandb_logger.log_selfplay(
            games_completed=config.num_games,
            buffer_size=len(replay_buffer),
            samples_generated=len(replay_buffer),
        )

    return replay_buffer, game_results


def train_network(
    network: MicroCivNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: list[dict[str, Any]],
    config: TrainingConfig,
    wandb_logger: Optional[WandbLogger] = None,
    verbose: bool = True,
    initial_step: int = 0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> TrainingResult:
    """Train the network on replay buffer data.

    Args:
        network: Neural network to train.
        optimizer: Optimizer for training.
        replay_buffer: List of experience dictionaries.
        config: Training configuration.
        wandb_logger: Optional W&B logger for metrics.
        verbose: Whether to print progress.
        initial_step: Starting global step count.
        scheduler: Optional learning rate scheduler.

    Returns:
        TrainingResult with metrics.
    """
    if verbose:
        print(f"Starting training ({config.num_epochs} epochs)...")

    network.train()
    global_step = initial_step
    epoch_losses: list[float] = []

    final_loss = 0.0
    final_policy_loss = 0.0
    final_value_loss = 0.0

    for epoch in range(config.num_epochs):
        # Shuffle replay buffer
        buffer_copy = replay_buffer.copy()
        random.shuffle(buffer_copy)

        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0

        for i in range(0, len(buffer_copy), config.batch_size):
            batch = buffer_copy[i:i + config.batch_size]
            if len(batch) < config.batch_size:
                continue

            # Prepare batch
            spatial = torch.stack([torch.tensor(d["spatial"]) for d in batch]).float()
            vector = torch.stack([torch.tensor(d["vector"]) for d in batch]).float()
            target_policy = torch.stack([torch.tensor(d["policy"]) for d in batch]).float()
            target_value = torch.tensor([d["value"] for d in batch]).float()

            # Forward pass
            optimizer.zero_grad()
            policy_logits, value = network(spatial, vector)
            # Apply softmax to get probabilities
            policy = torch.softmax(policy_logits, dim=-1)
            value = value.squeeze()

            # Compute losses
            policy_loss = -torch.mean(torch.sum(target_policy * torch.log(policy + 1e-8), dim=1))
            value_loss = torch.mean((value - target_value) ** 2)
            total_loss = policy_loss + value_loss

            # Check for valid loss
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip_norm)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            num_batches += 1

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Log to W&B
            if wandb_logger is not None:
                wandb_logger.log_training_step(
                    step=global_step,
                    loss=total_loss.item(),
                    policy_loss=policy_loss.item(),
                    value_loss=value_loss.item(),
                    learning_rate=current_lr,
                )
            global_step += 1

        # Step scheduler at end of epoch
        if scheduler is not None:
            scheduler.step()

        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_policy_loss = epoch_policy_loss / num_batches
            avg_value_loss = epoch_value_loss / num_batches
            epoch_losses.append(avg_loss)

            final_loss = avg_loss
            final_policy_loss = avg_policy_loss
            final_value_loss = avg_value_loss

            if verbose:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{config.num_epochs}: loss={avg_loss:.4f}, "
                      f"policy={avg_policy_loss:.4f}, value={avg_value_loss:.4f}, lr={current_lr:.6f}")

    return TrainingResult(
        total_steps=global_step - initial_step,
        final_loss=final_loss,
        final_policy_loss=final_policy_loss,
        final_value_loss=final_value_loss,
        epoch_losses=epoch_losses,
    )


def evaluate_network(
    env: MicroCivEnv,
    network: MicroCivNetwork,
    config: TrainingConfig,
    wandb_logger: Optional[WandbLogger] = None,
    verbose: bool = True,
) -> EvaluationResult:
    """Evaluate the network with greedy action selection.

    Args:
        env: The game environment.
        network: Neural network to evaluate.
        config: Training configuration.
        wandb_logger: Optional W&B logger for metrics.
        verbose: Whether to print progress.

    Returns:
        EvaluationResult with metrics.
    """
    if verbose:
        print("Evaluating against random opponent...")

    network.eval()
    wins = 0
    total_eval_turns = 0

    for _ in range(config.eval_games):
        obs, _ = env.reset()
        done = False
        turn = 0
        game_reward = 0.0

        while not done and turn < config.max_turns_per_game:
            spatial = torch.tensor(obs["spatial"]).unsqueeze(0).float()
            vector = torch.tensor(obs["vector"]).unsqueeze(0).float()

            with torch.no_grad():
                policy_logits, _ = network(spatial, vector)
                # Apply softmax to get probabilities
                policy = torch.softmax(policy_logits, dim=-1)

            mask = obs["action_mask"]
            valid_indices = np.where(mask > 0)[0]

            if len(valid_indices) == 0:
                break

            # Greedy action selection
            policy_np = policy.squeeze().numpy() * mask
            action = int(np.argmax(policy_np))

            obs, reward, terminated, truncated, info = env.step(action)
            game_reward += float(reward)
            done = terminated or truncated
            turn += 1

        if game_reward > 0:
            wins += 1
        total_eval_turns += turn

    win_rate = wins / config.eval_games
    avg_turns = total_eval_turns / config.eval_games
    avg_reward = win_rate * 2 - 1  # Map [0, 1] to [-1, 1]

    # Log to W&B
    if wandb_logger is not None:
        wandb_logger.log_evaluation(
            win_rate=win_rate,
            games_played=config.eval_games,
            average_reward=avg_reward,
            average_turns=avg_turns,
        )

    if verbose:
        print(f"Evaluation: win_rate={win_rate:.2%}, avg_turns={avg_turns:.1f}")

    return EvaluationResult(
        win_rate=win_rate,
        games_played=config.eval_games,
        total_wins=wins,
        average_turns=avg_turns,
        average_reward=avg_reward,
    )


def save_checkpoint(
    network: MicroCivNetwork,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    config: TrainingConfig,
    checkpoint_dir: str,
) -> str:
    """Save a training checkpoint.

    Args:
        network: Neural network to save.
        optimizer: Optimizer state to save.
        iteration: Current iteration number.
        config: Training configuration.
        checkpoint_dir: Directory to save checkpoint.

    Returns:
        Path to saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.pt")

    torch.save({
        'iteration': iteration,
        'network_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.to_dict(),
    }, checkpoint_path)

    return checkpoint_path


def main(
    config: Optional[TrainingConfig] = None,
    wandb_enabled: bool = True,
    wandb_project: str = "civ6-alphazero",
    wandb_tags: Optional[list[str]] = None,
    verbose: bool = True,
    num_iterations: int = 1,
) -> dict[str, Any]:
    """Run the full training pipeline.

    Args:
        config: Training configuration. Uses defaults if None.
        wandb_enabled: Whether to enable W&B logging.
        wandb_project: W&B project name.
        wandb_tags: W&B run tags.
        verbose: Whether to print progress.
        num_iterations: Number of self-play -> train -> eval iterations.

    Returns:
        Dictionary with training results and metrics.
    """
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    if verbose:
        print("Starting Civ6 AlphaZero Training with W&B...")

    # Use default config if none provided
    if config is None:
        config = TrainingConfig()

    # Initialize W&B
    wandb_config = WandbConfig(
        enabled=wandb_enabled,
        project=wandb_project,
        mode="online" if wandb_enabled else "disabled",
        tags=wandb_tags or ["training", "microciv", "v3-enhanced"],
    )
    wandb_logger = WandbLogger(wandb_config)

    run = wandb_logger.init(config_dict=config.to_dict())

    if run and verbose:
        print(f"W&B run started: {run.name}")
        print(f"View at: https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}")

    # Create environment and network
    env = MicroCivEnv(board_size=config.board_size)
    assert isinstance(env.action_space, spaces.Discrete), "Action space must be Discrete"

    network = MicroCivNetwork(
        board_size=config.board_size,
        action_space_size=env.action_space.n,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)

    # Create reward shaper if enabled
    reward_shaper = None
    if config.use_reward_shaping:
        reward_shaper = create_reward_shaper(
            enabled=True,
            board_size=config.board_size,
            unit_advantage_weight=config.reward_shaping_unit_weight,
            city_advantage_weight=config.reward_shaping_city_weight,
            health_advantage_weight=config.reward_shaping_health_weight,
            territory_weight=config.reward_shaping_territory_weight,
        )
        if verbose:
            print("Reward shaping enabled")

    # Create MCTS for exploration if enabled
    mcts = None
    if config.use_mcts_exploration:
        mcts = create_mcts(
            temperature=config.temperature,
            dirichlet_alpha=config.mcts_dirichlet_alpha,
            dirichlet_epsilon=config.mcts_dirichlet_epsilon,
            use_full_mcts=False,
        )
        if verbose:
            print("MCTS exploration enabled")

    # Create training diagnostics
    diagnostics = create_diagnostics(num_actions=env.action_space.n)

    # Track results across iterations
    all_game_results: list[GameResult] = []
    all_training_results: list[TrainingResult] = []
    all_eval_results: list[EvaluationResult] = []

    current_temperature = config.temperature
    global_step = 0

    for iteration in range(num_iterations):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*50}")

        # Update MCTS temperature
        if mcts is not None:
            mcts.config.temperature = current_temperature

        # Self-play phase
        replay_buffer, game_results = run_self_play(
            env=env,
            network=network,
            config=config,
            wandb_logger=wandb_logger,
            verbose=verbose,
            reward_shaper=reward_shaper,
            mcts=mcts,
            diagnostics=diagnostics,
            current_temperature=current_temperature,
        )
        all_game_results.extend(game_results)

        # Training phase
        training_result = train_network(
            network=network,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            config=config,
            wandb_logger=wandb_logger,
            verbose=verbose,
            initial_step=global_step,
            scheduler=scheduler,
        )
        global_step += training_result.total_steps
        all_training_results.append(training_result)

        # Track iteration in diagnostics
        diagnostics.on_iteration_end(
            iteration=iteration,
            policy_loss=training_result.final_policy_loss,
            value_loss=training_result.final_value_loss,
            total_loss=training_result.final_loss,
        )

        # Check for warnings
        warnings = diagnostics.warnings
        if warnings and verbose:
            print("\n  Diagnostics warnings:")
            for warning in warnings[-3:]:  # Show last 3 warnings
                print(f"    - {warning}")
        diagnostics.clear_warnings()

        # Evaluation phase
        eval_result = evaluate_network(
            env=env,
            network=network,
            config=config,
            wandb_logger=wandb_logger,
            verbose=verbose,
        )
        all_eval_results.append(eval_result)

        # Log diagnostic summary to W&B
        if wandb_logger is not None and wandb_enabled:
            try:
                import wandb
                if wandb.run is not None:
                    summary = diagnostics.get_summary()
                    wandb.log({
                        "diagnostics/entropy_current": summary["entropy"]["current"],
                        "diagnostics/entropy_avg": summary["entropy"]["moving_avg"],
                        "diagnostics/value_mae": summary["value_prediction"]["mae"],
                        "diagnostics/value_correlation": summary["value_prediction"]["correlation"],
                        "diagnostics/action_diversity": summary["action_diversity"]["diversity_ratio"],
                        "iteration": iteration,
                    })
            except ImportError:
                pass

        # Save checkpoint if needed
        if (iteration + 1) % config.checkpoint_interval == 0:
            checkpoint_path = save_checkpoint(
                network, optimizer, iteration + 1, config, config.checkpoint_dir
            )
            if verbose:
                print(f"  Saved checkpoint: {checkpoint_path}")

        # Decay temperature
        current_temperature = max(
            config.min_temperature,
            current_temperature * config.temperature_decay
        )

    # Log summary
    final_eval = all_eval_results[-1] if all_eval_results else None
    final_training = all_training_results[-1] if all_training_results else None

    summary_dict = {
        "final_win_rate": final_eval.win_rate if final_eval else 0.0,
        "total_training_steps": global_step,
        "total_games": len(all_game_results),
        "total_iterations": num_iterations,
    }

    # Add learning curves
    curves = diagnostics.get_learning_curves()
    if curves["iterations"]:
        summary_dict["win_rate_improvement"] = (
            curves["win_rate"][-1] - curves["win_rate"][0]
            if len(curves["win_rate"]) > 1 else 0.0
        )

    wandb_logger.summary(summary_dict)

    if verbose:
        print(f"\nTraining complete!")
        print(f"  Total iterations: {num_iterations}")
        print(f"  Total games: {len(all_game_results)}")
        print(f"  Total training steps: {global_step}")
        print(f"  Final win rate: {final_eval.win_rate:.2%}" if final_eval else "")

        # Print diagnostic summary
        summary = diagnostics.get_summary()
        print(f"\nDiagnostic Summary:")
        print(f"  Entropy (current): {summary['entropy']['current']:.4f}")
        print(f"  Value MAE: {summary['value_prediction']['mae']:.4f}")
        print(f"  Value Correlation: {summary['value_prediction']['correlation']:.4f}")
        print(f"  Action Diversity: {summary['action_diversity']['diversity_ratio']:.2%}")

    wandb_logger.finish()

    if verbose:
        print("\nW&B run finished successfully!")

    # Calculate total experiences for backward compatibility
    total_experiences = sum(len(gr.experiences) for gr in all_game_results)

    return {
        "config": config.to_dict(),
        "replay_buffer_size": total_experiences,  # Backward compatibility
        "total_games": len(all_game_results),
        "game_results": all_game_results,
        "training_result": all_training_results[-1] if all_training_results else None,  # Backward compat
        "training_results": all_training_results,
        "eval_result": all_eval_results[-1] if all_eval_results else None,  # Backward compat
        "eval_results": all_eval_results,
        "diagnostics_summary": diagnostics.get_summary(),
        "learning_curves": diagnostics.get_learning_curves(),
    }


if __name__ == "__main__":
    # Run with enhanced configuration
    config = TrainingConfig(
        num_games=50,  # More games per iteration
        num_epochs=15,  # More training epochs
        batch_size=16,  # Larger batches
        use_reward_shaping=True,
        use_mcts_exploration=True,
        temperature=1.0,
        temperature_decay=0.95,
    )
    results = main(config=config, num_iterations=3)
