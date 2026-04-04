"""Imitation pre-training for bootstrapping neural network from expert games.

This module provides infrastructure to generate expert games from heuristic
agents and pre-train the neural network through supervised learning before
self-play begins. This bootstraps the value head with meaningful signal.

Usage:
    from alpha_zero.training.imitation_pretraining import (
        ImitationConfig,
        ImitationPreTrainer,
    )

    # Create pre-trainer
    config = ImitationConfig(num_games=1000, num_epochs=5)
    pretrainer = ImitationPreTrainer(config)

    # Generate expert data and train
    pretrainer.generate_expert_data()
    metrics = pretrainer.train()

    # Save pretrained weights
    pretrainer.save_pretrained("checkpoints/pretrained.pt")
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from alpha_zero.core.civ6_network import MicroCivNetwork, compute_loss
from alpha_zero.distributed.shared_storage import GameTrajectory, TrainingExample
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.evaluation.baseline_agents import BaseAgent, GreedyHeuristicAgent

if TYPE_CHECKING:
    from alpha_zero.utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)


@dataclass
class ImitationConfig:
    """Configuration for imitation pre-training. All values externalized."""

    # Data generation
    num_games: int = 1000
    expert_type: str = "heuristic"  # "heuristic", "random"
    expert_seed: int | None = 42

    # Environment settings
    board_size: int = 16
    max_turns: int = 200

    # Training parameters
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    num_epochs: int = 5
    gradient_clip: float = 1.0

    # Value target computation
    use_n_step_returns: bool = True
    n_steps: int = 5
    gamma: float = 0.99

    # Network settings
    num_res_blocks: int = 6
    num_filters: int = 64

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpointing
    save_interval: int = 1  # Save every N epochs
    checkpoint_dir: str = "checkpoints/imitation"

    # Logging
    log_interval: int = 10  # Log every N batches
    verbose: bool = True

    @classmethod
    def from_env(cls) -> ImitationConfig:
        """Load configuration from environment variables with CIV6AZ_IMITATION_ prefix."""
        return cls(
            num_games=int(os.getenv("CIV6AZ_IMITATION_NUM_GAMES", "1000")),
            expert_type=os.getenv("CIV6AZ_IMITATION_EXPERT_TYPE", "heuristic"),
            expert_seed=int(os.getenv("CIV6AZ_IMITATION_EXPERT_SEED", "42")),
            board_size=int(os.getenv("CIV6AZ_IMITATION_BOARD_SIZE", "16")),
            max_turns=int(os.getenv("CIV6AZ_IMITATION_MAX_TURNS", "200")),
            batch_size=int(os.getenv("CIV6AZ_IMITATION_BATCH_SIZE", "256")),
            learning_rate=float(os.getenv("CIV6AZ_IMITATION_LR", "0.001")),
            num_epochs=int(os.getenv("CIV6AZ_IMITATION_EPOCHS", "5")),
            gradient_clip=float(os.getenv("CIV6AZ_IMITATION_GRAD_CLIP", "1.0")),
            use_n_step_returns=os.getenv("CIV6AZ_IMITATION_N_STEP", "true").lower() == "true",
            n_steps=int(os.getenv("CIV6AZ_IMITATION_N_STEPS", "5")),
            gamma=float(os.getenv("CIV6AZ_IMITATION_GAMMA", "0.99")),
            num_res_blocks=int(os.getenv("CIV6AZ_IMITATION_RES_BLOCKS", "6")),
            num_filters=int(os.getenv("CIV6AZ_IMITATION_FILTERS", "64")),
            device=os.getenv(
                "CIV6AZ_IMITATION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
            ),
            checkpoint_dir=os.getenv("CIV6AZ_IMITATION_CHECKPOINT_DIR", "checkpoints/imitation"),
        )


class ExpertDataGenerator(Protocol):
    """Protocol for expert data generation - enables multiple implementations."""

    def generate_game(self) -> GameTrajectory:
        """Generate a single game trajectory from expert play.

        Returns:
            GameTrajectory containing training examples from the game.
        """
        ...

    def generate_batch(self, n_games: int) -> list[GameTrajectory]:
        """Generate multiple game trajectories.

        Args:
            n_games: Number of games to generate.

        Returns:
            List of GameTrajectory objects.
        """
        ...


class HeuristicExpertGenerator:
    """Generate expert games using GreedyHeuristicAgent for both players."""

    def __init__(
        self,
        config: ImitationConfig,
        agent: BaseAgent | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            config: Imitation configuration.
            agent: Optional custom agent (for testing/dependency injection).
        """
        self.config = config
        self._logger = logging.getLogger(__name__)

        # Use injected agent or create default
        if agent is not None:
            self.agent = agent
        else:
            self.agent = GreedyHeuristicAgent(seed=config.expert_seed)

        self._game_seed = config.expert_seed or 0
        self._game_counter = 0

    def generate_game(self) -> GameTrajectory:
        """Play one game and return trajectory with training examples.

        Returns:
            GameTrajectory with examples from both players' perspectives.
        """
        env = MicroCivEnv(
            board_size=self.config.board_size,
            max_turns=self.config.max_turns,
            seed=self._game_seed + self._game_counter,
        )
        self._game_counter += 1

        # Reset agents
        self.agent.reset()
        opponent = GreedyHeuristicAgent(seed=self._game_seed + self._game_counter + 10000)
        opponent.reset()

        obs, _ = env.reset()

        # Collect game history
        history: list[dict] = []

        while True:
            current_player = env.current_player

            # Select agent based on current player
            current_agent = self.agent if current_player == 0 else opponent

            # Get action from expert
            action = current_agent.select_action(env, obs)

            # Store observation and action for this player
            action_probs = np.zeros(int(env.action_space.n), dtype=np.float32)
            action_probs[action] = 1.0  # Hard target from expert

            history.append(
                {
                    "spatial": obs["spatial"].copy(),
                    "vector": obs["vector"].copy(),
                    "action_mask": obs["action_mask"].copy(),
                    "policy": action_probs,
                    "player": current_player,
                }
            )

            # Execute action
            obs, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        # Determine winner
        winner = info.get("winner")

        # Create training examples with value targets
        examples = self._create_examples(history, winner)

        trajectory = GameTrajectory(
            examples=examples,
            game_id=self._game_counter,
            game_length=env.turn,
            winner=winner,
        )

        return trajectory

    def _create_examples(
        self,
        history: list[dict],
        winner: int | None,
    ) -> list[TrainingExample]:
        """Create training examples with computed value targets.

        Args:
            history: List of game state dictionaries.
            winner: Winner player ID (0, 1, or None for draw).

        Returns:
            List of TrainingExample objects.
        """
        examples = []

        for i, state in enumerate(history):
            player = state["player"]

            # Compute value target
            if self.config.use_n_step_returns:
                value = self._compute_n_step_value(
                    history, i, player, winner, self.config.n_steps, self.config.gamma
                )
            else:
                # Simple final outcome
                if winner is None:
                    value = 0.0
                elif winner == player:
                    value = 1.0
                else:
                    value = -1.0

            examples.append(
                TrainingExample(
                    spatial=state["spatial"],
                    vector=state["vector"],
                    action_mask=state["action_mask"],
                    policy=state["policy"],
                    value=value,
                )
            )

        return examples

    def _compute_n_step_value(
        self,
        history: list[dict],
        start_idx: int,
        player: int,
        winner: int | None,
        n_steps: int,
        gamma: float,
    ) -> float:
        """Compute n-step return value target.

        Uses TD(n) style returns with interpolation toward final outcome.

        Args:
            history: Full game history.
            start_idx: Index of current state.
            player: Player whose perspective we're computing for.
            winner: Final game winner.
            n_steps: Number of steps for n-step return.
            gamma: Discount factor.

        Returns:
            Value target in [-1, 1].
        """
        # Final outcome from this player's perspective
        if winner is None:
            final_value = 0.0
        elif winner == player:
            final_value = 1.0
        else:
            final_value = -1.0

        # For imitation, we use a soft interpolation toward the final outcome
        # based on how close we are to the end of the game
        total_steps = len(history)
        steps_remaining = total_steps - start_idx

        if steps_remaining <= n_steps:
            # Close to end, use final outcome more heavily
            return final_value

        # Discount the final value based on distance
        discount = gamma ** min(steps_remaining, n_steps)
        return discount * final_value

    def generate_batch(self, n_games: int) -> list[GameTrajectory]:
        """Generate multiple game trajectories with progress logging.

        Args:
            n_games: Number of games to generate.

        Returns:
            List of GameTrajectory objects.
        """
        trajectories = []
        start_time = time.time()

        for i in range(n_games):
            trajectory = self.generate_game()
            trajectories.append(trajectory)

            if (i + 1) % 100 == 0 or i == n_games - 1:
                elapsed = time.time() - start_time
                games_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                self._logger.info(
                    f"Generated {i + 1}/{n_games} games ({games_per_sec:.1f} games/sec)"
                )

        return trajectories


class SupervisedDataset(Dataset):
    """PyTorch Dataset wrapper for supervised training from expert games."""

    def __init__(self, trajectories: list[GameTrajectory]) -> None:
        """Initialize dataset from game trajectories.

        Args:
            trajectories: List of GameTrajectory objects.
        """
        self.examples = self._flatten_trajectories(trajectories)
        logger.info(
            f"SupervisedDataset: {len(self.examples)} examples from {len(trajectories)} games"
        )

    def _flatten_trajectories(
        self,
        trajectories: list[GameTrajectory],
    ) -> list[TrainingExample]:
        """Flatten trajectories into list of examples.

        Args:
            trajectories: List of game trajectories.

        Returns:
            Flat list of TrainingExample objects.
        """
        examples = []
        for trajectory in trajectories:
            examples.extend(trajectory.examples)
        return examples

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single training example.

        Args:
            idx: Example index.

        Returns:
            Dictionary with tensors matching learner.py format.
        """
        example = self.examples[idx]
        return {
            "spatial": torch.from_numpy(example.spatial).float(),
            "vector": torch.from_numpy(example.vector).float(),
            "action_mask": torch.from_numpy(example.action_mask).float(),
            "policy": torch.from_numpy(example.policy).float(),
            "value": torch.tensor(example.value, dtype=torch.float32),
        }

    @staticmethod
    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Collate function for DataLoader.

        Args:
            batch: List of example dictionaries.

        Returns:
            Batched dictionary of tensors.
        """
        return {
            "spatial": torch.stack([b["spatial"] for b in batch]),
            "vector": torch.stack([b["vector"] for b in batch]),
            "action_mask": torch.stack([b["action_mask"] for b in batch]),
            "policy": torch.stack([b["policy"] for b in batch]),
            "value": torch.stack([b["value"] for b in batch]),
        }


class ImitationPreTrainer:
    """Main class for imitation pre-training.

    Generates expert games and trains network through supervised learning
    to bootstrap value head before self-play.
    """

    def __init__(
        self,
        config: ImitationConfig | None = None,
        network: MicroCivNetwork | None = None,
        expert_generator: ExpertDataGenerator | None = None,
        wandb_logger: WandbLogger | None = None,
    ) -> None:
        """Initialize pre-trainer.

        Args:
            config: Imitation configuration. Uses defaults if None.
            network: Neural network to train. Creates new one if None.
            expert_generator: Expert data generator. Creates HeuristicExpertGenerator if None.
            wandb_logger: Optional WandB logger for metrics.
        """
        self.config = config or ImitationConfig()
        self._logger = logging.getLogger(__name__)
        self._wandb = wandb_logger

        # Initialize environment to get action space size
        env = MicroCivEnv(board_size=self.config.board_size)
        self.action_space_size = env.action_space.n

        # Initialize or use provided network
        if network is not None:
            self.network = network
        else:
            self.network = MicroCivNetwork(
                board_size=self.config.board_size,
                action_space_size=int(self.action_space_size),
                num_res_blocks=self.config.num_res_blocks,
                num_filters=self.config.num_filters,
            )

        self.network = self.network.to(self.config.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Expert generator
        if expert_generator is not None:
            self._generator = expert_generator
        else:
            self._generator = HeuristicExpertGenerator(self.config)

        # Data storage
        self._trajectories: list[GameTrajectory] = []
        self._dataset: SupervisedDataset | None = None

        # Statistics
        self._training_history: list[dict] = []

        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        param_count = sum(p.numel() for p in self.network.parameters())
        self._logger.info(
            f"ImitationPreTrainer initialized: {param_count:,} parameters, "
            f"device={self.config.device}"
        )

    def generate_expert_data(
        self,
        n_games: int | None = None,
    ) -> list[GameTrajectory]:
        """Generate expert games with progress logging.

        Args:
            n_games: Number of games to generate. Uses config if None.

        Returns:
            List of generated trajectories.
        """
        n_games = n_games or self.config.num_games
        self._logger.info(f"Generating {n_games} expert games...")

        self._trajectories = self._generator.generate_batch(n_games)

        # Create dataset
        self._dataset = SupervisedDataset(self._trajectories)

        # Calculate statistics
        total_examples = len(self._dataset)
        wins = sum(1 for t in self._trajectories if t.winner == 0)
        draws = sum(1 for t in self._trajectories if t.winner is None)

        self._logger.info(
            f"Expert data generated: {total_examples} examples from {n_games} games "
            f"(P0 wins: {wins}, draws: {draws})"
        )

        return self._trajectories

    def train(
        self,
        epochs: int | None = None,
    ) -> dict[str, float]:
        """Train network on expert data.

        Args:
            epochs: Number of training epochs. Uses config if None.

        Returns:
            Dictionary with final training metrics.
        """
        epochs = epochs or self.config.num_epochs

        if self._dataset is None or len(self._dataset) == 0:
            raise ValueError("No training data. Call generate_expert_data() first.")

        # Create data loader
        dataloader = DataLoader(
            self._dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=SupervisedDataset.collate_fn,
            num_workers=0,  # Keep simple for now
            pin_memory=self.config.device == "cuda",
        )

        self._logger.info(
            f"Starting training: {epochs} epochs, "
            f"{len(dataloader)} batches/epoch, "
            f"{len(self._dataset)} total examples"
        )

        start_time = time.time()
        total_steps = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            self.network.train()

            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                spatial = batch["spatial"].to(self.config.device)
                vector = batch["vector"].to(self.config.device)
                action_mask = batch["action_mask"].to(self.config.device)
                target_policy = batch["policy"].to(self.config.device)
                target_value = batch["value"].to(self.config.device)

                # Forward pass
                self.optimizer.zero_grad()
                policy_logits, value = self.network(spatial, vector, action_mask)

                # Compute loss (reuse from core module)
                total_loss, p_loss, v_loss = compute_loss(
                    policy_logits,
                    value,
                    target_policy,
                    target_value,
                    action_mask,
                )

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.config.gradient_clip,
                    )

                self.optimizer.step()

                # Track statistics
                epoch_loss += total_loss.item()
                epoch_policy_loss += p_loss.item()
                epoch_value_loss += v_loss.item()

                # Calculate policy accuracy (top-1)
                with torch.no_grad():
                    pred_actions = policy_logits.argmax(dim=-1)
                    target_actions = target_policy.argmax(dim=-1)
                    epoch_correct += (pred_actions == target_actions).sum().item()
                    epoch_total += target_policy.size(0)

                total_steps += 1

                # Log progress
                if (batch_idx + 1) % self.config.log_interval == 0:
                    self._logger.info(
                        f"Epoch {epoch + 1}/{epochs}, "
                        f"Batch {batch_idx + 1}/{len(dataloader)}: "
                        f"loss={total_loss.item():.4f}"
                    )

            # Epoch statistics
            num_batches = len(dataloader)
            avg_loss = epoch_loss / num_batches
            avg_policy_loss = epoch_policy_loss / num_batches
            avg_value_loss = epoch_value_loss / num_batches
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0

            epoch_metrics = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "policy_accuracy": accuracy,
            }
            self._training_history.append(epoch_metrics)

            # Log to WandB
            if self._wandb is not None:
                self._wandb.log_imitation_training(
                    epoch=epoch + 1,
                    loss=avg_loss,
                    policy_loss=avg_policy_loss,
                    value_loss=avg_value_loss,
                    policy_accuracy=accuracy,
                    games_used=len(self._trajectories),
                    samples_per_epoch=len(self._dataset),
                )

            self._logger.info(
                f"Epoch {epoch + 1}/{epochs}: "
                f"loss={avg_loss:.4f} "
                f"(policy={avg_policy_loss:.4f}, value={avg_value_loss:.4f}), "
                f"accuracy={accuracy:.2%}"
            )

            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(epoch + 1)

        # Final statistics
        elapsed = time.time() - start_time
        self._logger.info(
            f"Training complete: {total_steps} steps in {elapsed:.1f}s "
            f"({total_steps / elapsed:.1f} steps/sec)"
        )

        # Return final metrics
        return self._training_history[-1] if self._training_history else {}

    def _save_checkpoint(self, epoch: int) -> str:
        """Save a training checkpoint.

        Args:
            epoch: Current epoch number.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"imitation_epoch_{epoch}.pt",
        )

        torch.save(
            {
                "epoch": epoch,
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "training_history": self._training_history,
            },
            checkpoint_path,
        )

        self._logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def save_pretrained(self, path: str | None = None) -> str:
        """Save pretrained network weights.

        Args:
            path: Path to save weights. Uses default if None.

        Returns:
            Path where weights were saved.
        """
        if path is None:
            path = os.path.join(self.config.checkpoint_dir, "pretrained.pt")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "config": self.config,
                "training_history": self._training_history,
                "num_games": len(self._trajectories),
                "num_examples": len(self._dataset) if self._dataset else 0,
            },
            path,
        )

        self._logger.info(f"Saved pretrained weights: {path}")
        return path

    def load_pretrained(self, path: str) -> None:
        """Load pretrained weights.

        Args:
            path: Path to pretrained weights.

        Note:
            Uses weights_only=False since checkpoint contains custom config.
            Only load checkpoints from trusted sources.
        """
        # Need weights_only=False since checkpoint contains ImitationConfig dataclass
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self._logger.info(f"Loaded pretrained weights from {path}")

    def get_network(self) -> MicroCivNetwork:
        """Get the trained network.

        Returns:
            The neural network instance.
        """
        return self.network

    def get_training_history(self) -> list[dict]:
        """Get training history.

        Returns:
            List of epoch metrics dictionaries.
        """
        return self._training_history.copy()

    @property
    def num_examples(self) -> int:
        """Get number of training examples."""
        return len(self._dataset) if self._dataset else 0

    @property
    def num_games(self) -> int:
        """Get number of generated games."""
        return len(self._trajectories)


def run_imitation_pretraining(
    config: ImitationConfig | None = None,
    wandb_logger: WandbLogger | None = None,
) -> tuple[MicroCivNetwork, dict[str, float]]:
    """Convenience function to run full imitation pre-training.

    Args:
        config: Imitation configuration. Uses defaults if None.
        wandb_logger: Optional WandB logger.

    Returns:
        Tuple of (trained network, final metrics).
    """
    config = config or ImitationConfig.from_env()
    pretrainer = ImitationPreTrainer(config=config, wandb_logger=wandb_logger)

    # Generate and train
    pretrainer.generate_expert_data()
    metrics = pretrainer.train()

    # Save
    pretrainer.save_pretrained()

    return pretrainer.get_network(), metrics
