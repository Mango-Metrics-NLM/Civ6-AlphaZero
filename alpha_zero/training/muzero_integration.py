"""MuZero integration bridge for connecting MuZero components to training orchestrators.

This module provides a high-level interface for creating and coordinating MuZero networks,
trainers, and MCTS for self-play and training workflows.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

import yaml

logger = logging.getLogger(__name__)


@dataclass
class MuZeroIntegrationConfig:
    """Configuration for MuZero integration with training orchestrators."""

    unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 0.997
    use_reanalyze: bool = False
    reanalyze_fraction: float = 0.5
    consistency_loss_weight: float = 2.0
    reward_loss_weight: float = 1.0
    value_prefix: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.unroll_steps < 1:
            raise ValueError(f"unroll_steps must be >= 1, got {self.unroll_steps}")
        if self.td_steps < 1:
            raise ValueError(f"td_steps must be >= 1, got {self.td_steps}")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError(f"discount must be in [0, 1], got {self.discount}")
        if not 0.0 <= self.reanalyze_fraction <= 1.0:
            raise ValueError(f"reanalyze_fraction must be in [0, 1], got {self.reanalyze_fraction}")
        if self.consistency_loss_weight < 0.0:
            raise ValueError(
                f"consistency_loss_weight must be >= 0, got {self.consistency_loss_weight}"
            )
        if self.reward_loss_weight < 0.0:
            raise ValueError(f"reward_loss_weight must be >= 0, got {self.reward_loss_weight}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "unroll_steps": self.unroll_steps,
            "td_steps": self.td_steps,
            "discount": self.discount,
            "use_reanalyze": self.use_reanalyze,
            "reanalyze_fraction": self.reanalyze_fraction,
            "consistency_loss_weight": self.consistency_loss_weight,
            "reward_loss_weight": self.reward_loss_weight,
            "value_prefix": self.value_prefix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MuZeroIntegrationConfig:
        """Create config from dictionary."""
        return cls(
            unroll_steps=data.get("unroll_steps", 5),
            td_steps=data.get("td_steps", 10),
            discount=data.get("discount", 0.997),
            use_reanalyze=data.get("use_reanalyze", False),
            reanalyze_fraction=data.get("reanalyze_fraction", 0.5),
            consistency_loss_weight=data.get("consistency_loss_weight", 2.0),
            reward_loss_weight=data.get("reward_loss_weight", 1.0),
            value_prefix=data.get("value_prefix", False),
        )

    @classmethod
    def from_env(cls) -> MuZeroIntegrationConfig:
        """Create config from environment variables (prefix: CIV6AZ_MUZERO_)."""
        return cls(
            unroll_steps=int(os.environ.get("CIV6AZ_MUZERO_UNROLL_STEPS", "5")),
            td_steps=int(os.environ.get("CIV6AZ_MUZERO_TD_STEPS", "10")),
            discount=float(os.environ.get("CIV6AZ_MUZERO_DISCOUNT", "0.997")),
            use_reanalyze=os.environ.get("CIV6AZ_MUZERO_REANALYZE", "false").lower() == "true",
            reanalyze_fraction=float(os.environ.get("CIV6AZ_MUZERO_REANALYZE_FRACTION", "0.5")),
            consistency_loss_weight=float(
                os.environ.get("CIV6AZ_MUZERO_CONSISTENCY_LOSS_WEIGHT", "2.0")
            ),
            reward_loss_weight=float(os.environ.get("CIV6AZ_MUZERO_REWARD_LOSS_WEIGHT", "1.0")),
            value_prefix=os.environ.get("CIV6AZ_MUZERO_VALUE_PREFIX", "false").lower() == "true",
        )

    @classmethod
    def from_yaml(cls, path: str) -> MuZeroIntegrationConfig:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class MuZeroTrainingBridge:
    """Bridge for connecting MuZero components to training orchestrators."""

    def __init__(
        self,
        config: MuZeroIntegrationConfig | None = None,
        *,
        mcts_factory_fn: Callable | None = None,
    ) -> None:
        """Initialize the MuZero training bridge.

        Args:
            config: Integration config (defaults to MuZeroIntegrationConfig()).
            mcts_factory_fn: Optional factory function for creating MCTS instances.
                If provided, will be called with variant="muzero" and kwargs.
        """
        self.config = config if config is not None else MuZeroIntegrationConfig()
        self.mcts_factory_fn = mcts_factory_fn
        logger.info(f"Initialized MuZeroTrainingBridge with config: {self.config}")

    def create_network(
        self, spatial_channels: int, vector_dim: int, action_size: int, **kwargs: Any
    ) -> nn.Module:
        """Create a MuZero network instance.

        Args:
            spatial_channels: Number of spatial input channels.
            vector_dim: Dimension of vector input.
            action_size: Size of action space.
            **kwargs: Additional arguments to pass to MuZeroNetwork constructor.

        Returns:
            Initialized MuZeroNetwork instance.

        Raises:
            ImportError: If torch is not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MuZeroNetwork")

        from alpha_zero.core.muzero_network import MuZeroNetwork

        # Pass config values to network constructor
        network_kwargs = {
            "spatial_channels": spatial_channels,
            "vector_dim": vector_dim,
            "action_size": action_size,
            "discount": self.config.discount,
            **kwargs,
        }

        network = MuZeroNetwork(**network_kwargs)
        logger.info(
            f"Created MuZeroNetwork with spatial_channels={spatial_channels}, "
            f"vector_dim={vector_dim}, action_size={action_size}"
        )
        return network

    def create_trainer(self, network: nn.Module, device: str = "cpu", **kwargs: Any) -> Any:
        """Create a MuZero trainer instance.

        Args:
            network: The MuZero network to train.
            device: Device to use for training ("cpu" or "cuda").
            **kwargs: Additional arguments to override training config values.

        Returns:
            Initialized MuZeroTrainer instance.

        Raises:
            ImportError: If torch is not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MuZeroTrainer")

        from alpha_zero.training.muzero_trainer import MuZeroTrainer

        # Build training config from integration config + overrides
        training_config = self.to_training_config(**kwargs)

        trainer = MuZeroTrainer(network=network, config=training_config, device=device)
        logger.info(f"Created MuZeroTrainer on device={device}")
        return trainer

    def create_mcts(
        self,
        network: nn.Module,
        *,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        device: str = "cpu",
    ) -> Any:
        """Create a MuZero MCTS instance.

        Args:
            network: The MuZero network to use for MCTS.
            num_simulations: Number of MCTS simulations per search.
            c_puct: Exploration constant for PUCT algorithm.
            device: Device for neural network inference.

        Returns:
            Initialized MuZeroMCTS instance.
        """
        if self.mcts_factory_fn is not None:
            # Use factory function if provided
            logger.info("Creating MuZero MCTS using factory function")
            return self.mcts_factory_fn(
                variant="muzero",
                network=network,
                num_simulations=num_simulations,
                c_puct=c_puct,
            )

        # Create MuZeroMCTS directly with config object
        from alpha_zero.core.civ6_mcts import MuZeroMCTS, MuZeroMCTSConfig

        mcts_config = MuZeroMCTSConfig(
            num_simulations=num_simulations,
            c_puct=c_puct,
            discount=self.config.discount,
        )
        mcts = MuZeroMCTS(
            network=network,
            config=mcts_config,
            device=device,
        )
        logger.info(
            "Created MuZeroMCTS with num_simulations=%d, c_puct=%.2f",
            num_simulations,
            c_puct,
        )
        return mcts

    def to_training_config(self, **overrides: Any) -> Any:
        """Convert integration config to MuZeroTrainingConfig.

        Args:
            **overrides: Values to override in the training config.

        Returns:
            MuZeroTrainingConfig instance.

        Raises:
            ImportError: If torch is not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MuZeroTrainingConfig")

        from alpha_zero.training.muzero_trainer import MuZeroTrainingConfig

        # Start with integration config values
        config_dict = {
            "unroll_steps": self.config.unroll_steps,
            "td_steps": self.config.td_steps,
            "discount": self.config.discount,
            "consistency_loss_weight": self.config.consistency_loss_weight,
            "reward_loss_weight": self.config.reward_loss_weight,
            "use_value_prefix": self.config.value_prefix,
        }

        # Apply overrides
        config_dict.update(overrides)

        return MuZeroTrainingConfig(**config_dict)  # type: ignore[arg-type]

    def selfplay_step(
        self,
        network: nn.Module,
        env: Any,
        mcts: Any,
        *,
        temperature: float = 1.0,
        max_moves: int = 200,
    ) -> Any:
        """Execute a single self-play game using MuZero MCTS.

        Args:
            network: The MuZero network to use.
            env: The game environment.
            mcts: The MuZero MCTS instance.
            temperature: Temperature for action selection.
            max_moves: Maximum number of moves per game.

        Returns:
            GameTrajectory containing the game data.

        Raises:
            ImportError: If torch is not available.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for self-play")

        from alpha_zero.training.muzero_trainer import GameTrajectory

        # Initialize trajectory storage
        observations = []
        actions = []
        rewards = []
        policies = []
        values = []

        # Reset environment
        obs = env.reset()
        done = False
        move_count = 0

        logger.debug(f"Starting self-play game with max_moves={max_moves}")

        while not done and move_count < max_moves:
            # Get observation and action mask
            observation = obs if isinstance(obs, dict) else {"observation": obs}

            # Get action mask (default to all valid if not provided)
            if hasattr(env, "action_mask"):
                action_mask = env.action_mask()
            elif "action_mask" in observation:
                action_mask = observation["action_mask"]
            else:
                # Assume all actions valid
                import numpy as np

                action_mask = np.ones(env.action_space.n, dtype=bool)

            # MCTS search
            action_probs, value = mcts.search(observation, action_mask, add_noise=(move_count < 30))

            # Select action
            action = mcts.select_action(action_probs, temperature, action_mask)

            # Store trajectory data
            observations.append(observation)
            actions.append(action)
            policies.append(action_probs)
            values.append(value)

            # Execute action
            obs, reward, done, _info = env.step(action)
            rewards.append(reward)

            move_count += 1

        logger.debug(
            f"Self-play game completed: {move_count} moves, "
            f"final_reward={rewards[-1] if rewards else 0.0}"
        )

        # Create and return trajectory
        trajectory = GameTrajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            policies=policies,
            values=values,
        )

        return trajectory

    def training_step(self, trainer: Any, batch: Any, step: int) -> dict[str, float]:
        """Execute a single training step.

        Args:
            trainer: The MuZero trainer instance.
            batch: Batch of training data.
            step: Current training step number.

        Returns:
            Dictionary of training metrics.
        """
        # Call trainer's train_step
        metrics = trainer.train_step(batch)

        # Add step count to metrics
        metrics["step"] = float(step)

        logger.debug(
            f"Training step {step} completed: total_loss={metrics.get('total_loss', 0.0):.4f}"
        )

        return metrics  # type: ignore[no-any-return]
