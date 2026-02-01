"""Cloud-native trainer for Vertex AI.

Consumes game trajectories from storage and trains the neural network.

Usage:
    python -m alpha_zero.cloud.trainer \\
        --training-steps=1000 \\
        --iteration=5 \\
        --replay-window=10

Environment Variables:
    CIV6_GCS_BUCKET: GCS bucket for storage
    CIV6_GCS_PREFIX: Path prefix within bucket
    CIV6_ITERATION: Training iteration
    CIV6_TRAINING_STEPS: Number of training steps
    CIV6_BATCH_SIZE: Training batch size
    CIV6_LEARNING_RATE: Learning rate
    CIV6_DEVICE: PyTorch device (cuda/cpu)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from alpha_zero.storage import StorageConfig, create_storage_backend
from alpha_zero.storage.models import ModelMetadata
from alpha_zero.storage.replay_buffer import ReplayBufferConfig, StorageReplayBuffer

if TYPE_CHECKING:
    from alpha_zero.core.civ6_network import MicroCivNetwork
    from alpha_zero.storage.backend import StorageBackend

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for cloud trainer.

    Attributes:
        training_steps: Number of training steps to run.
        iteration: Current training iteration.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        gradient_clip: Maximum gradient norm.
        replay_window: Number of iterations to include in replay buffer.
        min_examples: Minimum examples before training starts.
        device: PyTorch device.
        use_amp: Whether to use automatic mixed precision.
        gradient_accumulation_steps: Steps to accumulate gradients.
        checkpoint_interval: Steps between checkpoints.
        board_size: Game board size.
        policy_loss_weight: Weight for policy loss.
        value_loss_weight: Weight for value loss.
    """

    training_steps: int = 1000
    iteration: int = 0
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    replay_window: int = 10
    min_examples: int = 1000
    device: str = "cuda"
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    checkpoint_interval: int = 100
    board_size: int = 8
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    @classmethod
    def from_env(cls) -> TrainerConfig:
        """Create configuration from environment variables."""
        return cls(
            training_steps=int(os.environ.get("CIV6_TRAINING_STEPS", "1000")),
            iteration=int(os.environ.get("CIV6_ITERATION", "0")),
            batch_size=int(os.environ.get("CIV6_BATCH_SIZE", "256")),
            learning_rate=float(os.environ.get("CIV6_LEARNING_RATE", "0.001")),
            weight_decay=float(os.environ.get("CIV6_WEIGHT_DECAY", "1e-4")),
            gradient_clip=float(os.environ.get("CIV6_GRADIENT_CLIP", "1.0")),
            replay_window=int(os.environ.get("CIV6_REPLAY_WINDOW", "10")),
            min_examples=int(os.environ.get("CIV6_MIN_EXAMPLES", "1000")),
            device=os.environ.get("CIV6_DEVICE", "cuda"),
            use_amp=os.environ.get("CIV6_USE_AMP", "true").lower() == "true",
            gradient_accumulation_steps=int(
                os.environ.get("CIV6_GRADIENT_ACCUMULATION_STEPS", "1")
            ),
            checkpoint_interval=int(os.environ.get("CIV6_CHECKPOINT_INTERVAL", "100")),
            board_size=int(os.environ.get("CIV6_BOARD_SIZE", "8")),
        )


class CloudTrainer:
    """Cloud-native trainer for Vertex AI.

    Trains the neural network using trajectories stored in GCS.
    Handles:
    - Loading/creating model checkpoints
    - Streaming training data from storage
    - Mixed precision training
    - Gradient accumulation for memory efficiency
    - Checkpoint saving to storage

    Attributes:
        config: Trainer configuration.
        storage: Storage backend.
    """

    def __init__(
        self,
        config: TrainerConfig,
        storage: StorageBackend,
    ) -> None:
        """Initialize cloud trainer.

        Args:
            config: Trainer configuration.
            storage: Storage backend.
        """
        self.config = config
        self.storage = storage

        # Determine device
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # Initialize model and optimizer (set in _initialize_model)
        self._model: MicroCivNetwork | None = None
        self._optimizer: optim.Optimizer | None = None
        self._scaler: GradScaler | None = None
        self._start_step: int = 0
        self._model_version: int = 0

        # Create replay buffer
        buffer_config = ReplayBufferConfig(
            window_size=config.replay_window,
            min_examples=config.min_examples,
            batch_size=config.batch_size,
        )
        self._replay_buffer = StorageReplayBuffer(
            storage=storage,
            config=buffer_config,
            current_iteration=config.iteration,
        )

        logger.info(
            f"CloudTrainer initialized "
            f"(iteration={config.iteration}, steps={config.training_steps})"
        )

    def _initialize_model(self) -> None:
        """Initialize or load model and optimizer."""
        from alpha_zero.core.civ6_network import MicroCivNetwork

        # Try to load latest checkpoint
        checkpoint_data = None
        try:
            result = self.storage.load_latest_checkpoint(map_location=self.device)
            if result:
                checkpoint_data, metadata = result
                self._start_step = metadata.training_step
                self._model_version = metadata.model_version
                logger.info(
                    f"Resuming from checkpoint (step={self._start_step}, "
                    f"version={self._model_version})"
                )
        except Exception as e:
            logger.info(f"No checkpoint found, starting fresh: {e}")

        # Create model
        self._model = MicroCivNetwork(
            board_size=self.config.board_size,
            action_space_size=356,
        ).to(self.device)

        # Load weights if checkpoint exists
        if checkpoint_data and "model_state_dict" in checkpoint_data:
            self._model.load_state_dict(checkpoint_data["model_state_dict"])
            logger.info("Loaded model weights from checkpoint")

        # Create optimizer
        self._optimizer = optim.AdamW(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Load optimizer state if checkpoint exists
        if checkpoint_data and "optimizer_state_dict" in checkpoint_data:
            self._optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            logger.info("Loaded optimizer state from checkpoint")

        # Create gradient scaler for mixed precision
        if self.config.use_amp and self.device.type == "cuda":
            self._scaler = GradScaler()
            if checkpoint_data and "scaler_state_dict" in checkpoint_data:
                self._scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
            logger.info("Mixed precision training enabled")

    def _compute_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute training loss.

        Args:
            batch: Batch dictionary with tensors.

        Returns:
            Tuple of (total_loss, loss_components).
        """
        assert self._model is not None, "Model not initialized"

        spatial = batch["spatial"].to(self.device)
        vector = batch["vector"].to(self.device)
        target_policy = batch["policy"].to(self.device)
        target_value = batch["value"].to(self.device)
        action_mask = batch["action_mask"].to(self.device)

        # Forward pass
        policy_logits, value = self._model(spatial, vector)

        # Policy loss (cross-entropy with soft targets)
        # Mask invalid actions before softmax
        masked_logits = policy_logits.clone()
        masked_logits[action_mask == 0] = float("-inf")
        log_probs = nn.functional.log_softmax(masked_logits, dim=-1)
        policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

        # Value loss (MSE)
        value_loss = nn.functional.mse_loss(value.squeeze(-1), target_value)

        # Total loss
        total_loss = (
            self.config.policy_loss_weight * policy_loss
            + self.config.value_loss_weight * value_loss
        )

        loss_components = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_components

    def _save_checkpoint(self, step: int, metrics: dict[str, Any]) -> str:
        """Save training checkpoint.

        Args:
            step: Current training step.
            metrics: Training metrics.

        Returns:
            Path where checkpoint was saved.
        """
        from alpha_zero.storage.models import CheckpointMetadata

        assert self._model is not None, "Model not initialized"
        assert self._optimizer is not None, "Optimizer not initialized"

        checkpoint_data = {
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "step": step,
            "iteration": self.config.iteration,
            "model_version": self._model_version,
            "config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "board_size": self.config.board_size,
            },
        }

        if self._scaler:
            checkpoint_data["scaler_state_dict"] = self._scaler.state_dict()

        metadata = CheckpointMetadata(
            iteration=self.config.iteration,
            training_step=step,
            model_version=self._model_version,
        )

        path = self.storage.save_checkpoint(
            checkpoint_data=checkpoint_data,
            iteration=self.config.iteration,
            metadata=metadata,
        )

        logger.info(f"Saved checkpoint at step {step}")
        return path

    def _save_model(self, metrics: dict[str, float]) -> str:
        """Save model weights and update champion.

        Args:
            metrics: Training metrics.

        Returns:
            Path where model was saved.
        """
        self._model_version += 1

        metadata = ModelMetadata(
            iteration=self.config.iteration,
            version=self._model_version,
            metrics=metrics,
        )

        assert self._model is not None, "Model not initialized"

        # Save iteration model
        path = self.storage.save_model(
            state_dict=self._model.state_dict(),
            iteration=self.config.iteration,
            metadata=metadata,
        )

        logger.info(f"Saved model v{self._model_version} at iteration {self.config.iteration}")
        return path

    def run(self) -> dict[str, Any]:
        """Run training.

        Returns:
            Dictionary with training statistics.
        """
        start_time = datetime.now(UTC)
        logger.info(
            f"Starting training: {self.config.training_steps} steps, "
            f"iteration {self.config.iteration}"
        )

        # Initialize model
        self._initialize_model()
        assert self._model is not None, "Model not initialized"
        assert self._optimizer is not None, "Optimizer not initialized"
        self._model.train()

        # Check if we have enough data
        stats = self._replay_buffer.get_statistics()
        logger.info(f"Replay buffer stats: {stats}")

        if not self._replay_buffer.is_ready():
            logger.warning(
                f"Not enough training examples "
                f"({stats['total_examples']} < {self.config.min_examples})"
            )
            return {
                "status": "skipped",
                "reason": "insufficient_data",
                "total_examples": stats["total_examples"],
                "min_required": self.config.min_examples,
            }

        # Training loop
        step = self._start_step
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        for batch in self._replay_buffer:
            if step >= self._start_step + self.config.training_steps:
                break

            # Forward and backward pass
            if self.config.use_amp and self._scaler:
                with autocast():
                    loss, loss_components = self._compute_loss(batch)
                    loss = loss / self.config.gradient_accumulation_steps

                self._scaler.scale(loss).backward()
                accumulated_loss += loss.item()

                # Update weights after accumulation
                if (num_batches + 1) % self.config.gradient_accumulation_steps == 0:
                    assert self._optimizer is not None
                    assert self._model is not None
                    self._scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        self.config.gradient_clip,
                    )
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._optimizer.zero_grad()
                    accumulated_loss = 0.0
                    step += 1

            else:
                loss, loss_components = self._compute_loss(batch)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()

                if (num_batches + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        self.config.gradient_clip,
                    )
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                    accumulated_loss = 0.0
                    step += 1

            # Track losses
            total_policy_loss += loss_components["policy_loss"]
            total_value_loss += loss_components["value_loss"]
            total_loss += loss_components["total_loss"]
            num_batches += 1

            # Checkpoint
            if step > 0 and step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(
                    step,
                    {
                        "avg_policy_loss": total_policy_loss / num_batches,
                        "avg_value_loss": total_value_loss / num_batches,
                    },
                )

            # Log progress
            if num_batches % 100 == 0:
                logger.info(
                    f"Step {step}: "
                    f"policy_loss={total_policy_loss / num_batches:.4f}, "
                    f"value_loss={total_value_loss / num_batches:.4f}"
                )

        # Final checkpoint and model save
        final_metrics = {
            "avg_policy_loss": total_policy_loss / max(num_batches, 1),
            "avg_value_loss": total_value_loss / max(num_batches, 1),
            "avg_total_loss": total_loss / max(num_batches, 1),
        }

        self._save_checkpoint(step, final_metrics)
        self._save_model(final_metrics)

        # Calculate statistics
        elapsed = (datetime.now(UTC) - start_time).total_seconds()
        steps_completed = step - self._start_step

        result = {
            "status": "completed",
            "iteration": self.config.iteration,
            "steps_completed": steps_completed,
            "steps_requested": self.config.training_steps,
            "batches_processed": num_batches,
            "avg_policy_loss": final_metrics["avg_policy_loss"],
            "avg_value_loss": final_metrics["avg_value_loss"],
            "avg_total_loss": final_metrics["avg_total_loss"],
            "model_version": self._model_version,
            "elapsed_seconds": elapsed,
            "steps_per_second": steps_completed / max(elapsed, 1),
        }

        logger.info(f"Training complete: {result}")
        return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Civ6 AlphaZero Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--training-steps",
        type=int,
        default=int(os.environ.get("CIV6_TRAINING_STEPS", "1000")),
        help="Number of training steps",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=int(os.environ.get("CIV6_ITERATION", "0")),
        help="Current training iteration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("CIV6_BATCH_SIZE", "256")),
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.environ.get("CIV6_LEARNING_RATE", "0.001")),
        help="Learning rate",
    )
    parser.add_argument(
        "--replay-window",
        type=int,
        default=int(os.environ.get("CIV6_REPLAY_WINDOW", "10")),
        help="Number of iterations in replay window",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=int(os.environ.get("CIV6_BOARD_SIZE", "8")),
        help="Game board size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("CIV6_DEVICE", "cuda"),
        help="PyTorch device (cuda/cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for trainer."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    # Create configuration
    config = TrainerConfig(
        training_steps=args.training_steps,
        iteration=args.iteration,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        replay_window=args.replay_window,
        board_size=args.board_size,
        device=args.device,
    )

    if args.dry_run:
        logger.info("Dry run - configuration validated")
        logger.info(f"Config: {config}")
        return

    # Create storage backend
    storage_config = StorageConfig.from_env()
    storage = create_storage_backend(storage_config)

    # Create and run trainer
    trainer = CloudTrainer(
        config=config,
        storage=storage,
    )

    try:
        result = trainer.run()
        logger.info(f"Trainer completed with result: {result}")
        if result.get("status") == "skipped":
            sys.exit(0)  # Not an error, just insufficient data
    except Exception as e:
        logger.error(f"Trainer failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        storage.close()


if __name__ == "__main__":
    main()
