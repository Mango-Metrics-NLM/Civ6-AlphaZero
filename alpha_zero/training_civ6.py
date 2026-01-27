"""Training pipeline for Civ6 AlphaZero.

Implements the AlphaZero self-play training loop:
1. Self-play: Generate games using MCTS + neural network
2. Training: Update network on generated data
3. Evaluation: Compare new network against previous version
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from alpha_zero.core.civ6_mcts import (
    Civ6MCTS,
    GumbelMCTS,
    GumbelMCTSConfig,
    MCTSConfig,
    SimpleMCTS,
)
from alpha_zero.core.civ6_network import Civ6Network, MicroCivNetwork, compute_loss
from alpha_zero.envs.microciv import MicroCivEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Environment
    board_size: int = 16
    max_turns: int = 200

    # Network
    num_res_blocks: int = 6
    num_filters: int = 64

    # MCTS
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30

    # Gumbel MuZero MCTS (alternative to standard MCTS)
    use_gumbel_mcts: bool = False
    gumbel_num_simulations: int = 16
    gumbel_num_sampled_actions: int = 16
    gumbel_c_visit: float = 50.0
    gumbel_use_sequential_halving: bool = True

    # Training
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    num_epochs: int = 10

    # Self-play
    num_games_per_iteration: int = 100
    num_iterations: int = 100
    replay_buffer_size: int = 50000
    min_replay_size: int = 1000

    # Checkpointing
    checkpoint_dir: str = "checkpoints/civ6"
    log_dir: str = "logs/civ6"
    save_interval: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingExample:
    """Single training example."""

    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    policy: np.ndarray
    value: float


class GameHistoryItem(TypedDict):
    """Type for game history items."""

    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    policy: np.ndarray
    player: int


class ReplayBuffer:
    """Replay buffer for storing training examples."""

    def __init__(self, max_size: int = 50000):
        """Initialize replay buffer.

        Args:
            max_size: Maximum number of examples to store.
        """
        self.buffer: deque[TrainingExample] = deque(maxlen=max_size)

    def add(self, example: TrainingExample) -> None:
        """Add example to buffer."""
        self.buffer.append(example)

    def add_game(self, examples: list[TrainingExample]) -> None:
        """Add all examples from a game."""
        for example in examples:
            self.add(example)

    def sample(self, batch_size: int) -> list[TrainingExample]:
        """Sample batch from buffer.

        Args:
            batch_size: Number of examples to sample.

        Returns:
            List of sampled examples.
        """
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class ReplayDataset(Dataset):
    """PyTorch dataset for replay buffer."""

    def __init__(self, examples: list[TrainingExample]):
        """Initialize dataset.

        Args:
            examples: List of training examples.
        """
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


def self_play_game(
    env: MicroCivEnv,
    mcts: Civ6MCTS | SimpleMCTS | GumbelMCTS,
    config: TrainingConfig,
) -> list[TrainingExample]:
    """Play a single self-play game.

    Args:
        env: Game environment.
        mcts: MCTS instance.
        config: Training configuration.

    Returns:
        List of training examples from the game.
    """
    examples = []
    obs, _ = env.reset()

    game_history: list[GameHistoryItem] = []

    while True:
        # Get MCTS policy
        action_probs, _ = mcts.search(env)

        # Store state and policy
        game_history.append(
            {
                "spatial": obs["spatial"].copy(),
                "vector": obs["vector"].copy(),
                "action_mask": obs["action_mask"].copy(),
                "policy": action_probs.copy(),
                "player": env.current_player,
            },
        )

        # Select action
        temperature = (
            config.temperature if env.turn < config.temperature_threshold else 0.0
        )
        action = mcts.select_action(action_probs, temperature)

        # Take action
        obs, _, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Assign values based on game outcome
    winner = info.get("winner")
    for hist in game_history:
        if winner is None:
            value = 0.0  # Draw
        elif winner == hist["player"]:
            value = 1.0
        else:
            value = -1.0

        examples.append(
            TrainingExample(
                spatial=hist["spatial"],
                vector=hist["vector"],
                action_mask=hist["action_mask"],
                policy=hist["policy"],
                value=value,
            ),
        )

    return examples


def run_self_play(
    network: torch.nn.Module,
    config: TrainingConfig,
    num_games: int,
    use_simple_mcts: bool = False,
) -> list[TrainingExample]:
    """Run multiple self-play games.

    Args:
        network: Neural network.
        config: Training configuration.
        num_games: Number of games to play.
        use_simple_mcts: If True, use SimpleMCTS instead of network-guided MCTS.

    Returns:
        List of training examples from all games.
    """
    examples = []
    env = MicroCivEnv(board_size=config.board_size, max_turns=config.max_turns)

    if use_simple_mcts:
        mcts: Civ6MCTS | SimpleMCTS | GumbelMCTS = SimpleMCTS(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
        )
    elif config.use_gumbel_mcts:
        # Use Gumbel MuZero MCTS for efficient search with few simulations
        gumbel_config = GumbelMCTSConfig(
            num_simulations=config.gumbel_num_simulations,
            num_sampled_actions=config.gumbel_num_sampled_actions,
            c_visit=config.gumbel_c_visit,
            use_sequential_halving=config.gumbel_use_sequential_halving,
            temperature=config.temperature,
            temperature_threshold=config.temperature_threshold,
        )
        mcts = GumbelMCTS(network, gumbel_config, device=config.device)
        logger.info(
            f"Using Gumbel MuZero MCTS with {config.gumbel_num_simulations} simulations"
        )
    else:
        mcts_config = MCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            temperature=config.temperature,
            temperature_threshold=config.temperature_threshold,
        )
        mcts = Civ6MCTS(network, mcts_config, device=config.device)

    for _ in tqdm(range(num_games), desc="Self-play"):
        game_examples = self_play_game(env, mcts, config)
        examples.extend(game_examples)

    return examples


def train_epoch(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    config: TrainingConfig,
) -> dict[str, float]:
    """Train network for one epoch.

    Args:
        network: Neural network.
        optimizer: Optimizer.
        dataloader: Training data loader.
        config: Training configuration.

    Returns:
        Dictionary of loss values.
    """
    network.train()
    total_loss = 0.0
    policy_loss = 0.0
    value_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        spatial = batch["spatial"].to(config.device)
        vector = batch["vector"].to(config.device)
        action_mask = batch["action_mask"].to(config.device)
        target_policy = batch["policy"].to(config.device)
        target_value = batch["value"].to(config.device)

        optimizer.zero_grad()

        # Forward pass
        policy_logits, value = network(spatial, vector, action_mask)

        # Compute loss
        loss, p_loss, v_loss = compute_loss(
            policy_logits, value, target_policy, target_value, action_mask,
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        policy_loss += p_loss.item()
        value_loss += v_loss.item()
        num_batches += 1

    return {
        "total_loss": total_loss / num_batches,
        "policy_loss": policy_loss / num_batches,
        "value_loss": value_loss / num_batches,
    }


def evaluate_against_random(
    network: torch.nn.Module,
    config: TrainingConfig,
    num_games: int = 20,
) -> float:
    """Evaluate network against random player.

    Args:
        network: Neural network.
        config: Training configuration.
        num_games: Number of games to play.

    Returns:
        Win rate against random player.
    """
    env = MicroCivEnv(board_size=config.board_size, max_turns=config.max_turns)
    mcts_config = MCTSConfig(num_simulations=config.num_simulations // 2)
    mcts = Civ6MCTS(network, mcts_config, device=config.device)

    wins = 0
    rng = np.random.default_rng()

    for game_idx in range(num_games):
        obs, _ = env.reset()
        network_player = game_idx % 2  # Alternate who goes first

        while True:
            if env.current_player == network_player:
                # Network's turn
                action_probs, _ = mcts.search(env, add_noise=False)
                action = mcts.select_action(action_probs, temperature=0.0)
            else:
                # Random player's turn
                mask = env.get_valid_actions()
                valid = np.where(mask > 0)[0]
                action = rng.choice(valid) if len(valid) > 0 else 0

            obs, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                winner = info.get("winner")
                if winner == network_player:
                    wins += 1
                break

    return wins / num_games


def train(config: TrainingConfig, dry_run: bool = False) -> None:
    """Main training loop.

    Args:
        config: Training configuration.
        dry_run: If True, run minimal training for testing.
    """
    # Create directories
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Initialize environment to get action space size
    env = MicroCivEnv(board_size=config.board_size)
    action_space_size = env.action_space.n

    # Initialize network
    network = MicroCivNetwork(
        board_size=config.board_size,
        action_space_size=action_space_size,
        num_res_blocks=config.num_res_blocks,
        num_filters=config.num_filters,
    ).to(config.device)

    logger.info(f"Network parameters: {sum(p.numel() for p in network.parameters()):,}")
    logger.info(f"Action space size: {action_space_size}")
    logger.info(f"Device: {config.device}")

    # Initialize optimizer
    optimizer = optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=config.replay_buffer_size)

    # Training loop
    num_iterations = 2 if dry_run else config.num_iterations
    num_games = 2 if dry_run else config.num_games_per_iteration

    for iteration in range(num_iterations):
        logger.info(f"\n{'='*50}")
        logger.info(f"Iteration {iteration + 1}/{num_iterations}")
        logger.info(f"{'='*50}")

        # Self-play
        logger.info("Running self-play...")
        start_time = time.time()

        # Use simple MCTS for first iteration (no trained network)
        use_simple = iteration == 0
        examples = run_self_play(network, config, num_games, use_simple_mcts=use_simple)
        replay_buffer.add_game(examples)

        self_play_time = time.time() - start_time
        logger.info(
            f"Self-play: {len(examples)} examples from {num_games} games "
            f"({self_play_time:.1f}s)",
        )

        # Training
        if len(replay_buffer) >= config.min_replay_size or dry_run:
            logger.info("Training network...")
            start_time = time.time()

            # Sample from replay buffer
            sample_size = min(len(replay_buffer), config.batch_size * 100)
            train_examples = replay_buffer.sample(sample_size)

            # Create dataloader
            dataset = ReplayDataset(train_examples)
            dataloader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=0,
            )

            # Train for multiple epochs
            num_epochs = 1 if dry_run else config.num_epochs
            for epoch in range(num_epochs):
                losses = train_epoch(network, optimizer, dataloader, config)
                logger.info(
                    f"Epoch {epoch + 1}: loss={losses['total_loss']:.4f}, "
                    f"policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f}",
                )

            train_time = time.time() - start_time
            logger.info(f"Training complete ({train_time:.1f}s)")

        # Evaluation
        if (iteration + 1) % config.save_interval == 0 or dry_run:
            logger.info("Evaluating against random...")
            win_rate = evaluate_against_random(
                network, config, num_games=10 if dry_run else 20,
            )
            logger.info(f"Win rate vs random: {win_rate:.1%}")

            # Save checkpoint
            checkpoint_path = os.path.join(
                config.checkpoint_dir, f"checkpoint_{iteration + 1}.pt",
            )
            network.save(checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = os.path.join(config.checkpoint_dir, "latest.pt")
    network.save(final_path)
    logger.info(f"Training complete. Final model: {final_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Civ6 AlphaZero")
    parser.add_argument("--board_size", type=int, default=16)
    parser.add_argument("--num_simulations", type=int, default=50)
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_res_blocks", type=int, default=6)
    parser.add_argument("--num_filters", type=int, default=64)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/civ6")
    parser.add_argument("--dry_run", action="store_true", help="Quick test run")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = TrainingConfig(
        board_size=args.board_size,
        num_simulations=args.num_simulations,
        num_games_per_iteration=args.num_games,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_res_blocks=args.num_res_blocks,
        num_filters=args.num_filters,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )

    train(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
