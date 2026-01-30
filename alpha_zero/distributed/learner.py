"""Central learner for distributed AlphaZero training.

The learner samples from the replay buffer, trains the network, and
publishes updated weights to shared storage.
"""

from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from alpha_zero.core.civ6_network import MicroCivNetwork, compute_loss
from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    SharedStorage,
)
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.utils.checkpoint import safe_load_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class LearnerConfig:
    """Configuration for the learner."""

    # Environment (for network initialization)
    board_size: int = 16

    # Network settings
    num_res_blocks: int = 6
    num_filters: int = 64

    # Training settings
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0

    # Learning rate scheduling
    lr_decay_steps: list[int] | None = None
    lr_decay_rate: float = 0.1

    # Gradient accumulation
    gradient_accumulation_steps: int = 1

    # Training loop
    training_steps_per_update: int = 100
    min_buffer_size: int = 10_000
    max_training_steps: int = 0  # 0 = unlimited

    # Weight publishing
    publish_interval: int = 100  # Steps between weight updates

    # Checkpointing
    checkpoint_dir: str = "checkpoints/civ6"
    checkpoint_interval: int = 1000

    # Performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 10


class LearningRateScheduler:
    """Multi-step learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        decay_steps: list[int] | None = None,
        decay_rate: float = 0.1,
        warmup_steps: int = 0,
        base_lr: float = 0.001,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule.
            decay_steps: Steps at which to decay LR.
            decay_rate: Multiplicative decay factor.
            warmup_steps: Number of warmup steps.
            base_lr: Base learning rate.
        """
        self.optimizer = optimizer
        self.decay_steps = decay_steps or []
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0

    def step(self) -> float:
        """Take a scheduler step.

        Returns:
            Current learning rate.
        """
        self.current_step += 1

        # Compute learning rate
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Apply decay at milestones
            lr = self.base_lr
            for milestone in self.decay_steps:
                if self.current_step >= milestone:
                    lr *= self.decay_rate

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return float(self.optimizer.param_groups[0]["lr"])


class GradientAccumulator:
    """Handles gradient accumulation for effective larger batch sizes."""

    def __init__(
        self,
        network: nn.Module,
        optimizer: optim.Optimizer,
        accumulation_steps: int = 1,
        gradient_clip: float = 1.0,
    ):
        """Initialize accumulator.

        Args:
            network: Neural network.
            optimizer: Optimizer.
            accumulation_steps: Steps to accumulate before update.
            gradient_clip: Gradient clipping value.
        """
        self.network = network
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.gradient_clip = gradient_clip
        self.step_count = 0
        self.accumulated_loss = 0.0

    def accumulate(self, loss: torch.Tensor) -> bool:
        """Accumulate gradients from loss.

        Args:
            loss: Loss to backpropagate.

        Returns:
            True if optimizer step was taken.
        """
        # Scale loss for accumulation
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()

        self.accumulated_loss += loss.item()
        self.step_count += 1

        if self.step_count >= self.accumulation_steps:
            # Clip gradients
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.gradient_clip,
                )

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.step_count = 0
            self.accumulated_loss = 0.0
            return True

        return False


class Learner:
    """Central learner that trains the network.

    Samples batches from the replay buffer, trains the network, and
    publishes updated weights to shared storage.
    """

    def __init__(
        self,
        config: LearnerConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ):
        """Initialize learner.

        Args:
            config: Learner configuration.
            shared_storage: Shared storage for weights.
            replay_buffer: Replay buffer for training data.
        """
        self.config = config
        self.storage = shared_storage
        self.replay_buffer = replay_buffer

        # Initialize environment to get action space size
        env = MicroCivEnv(board_size=config.board_size)
        self.action_space_size = env.action_space.n

        # Initialize network
        self.network = MicroCivNetwork(
            board_size=config.board_size,
            action_space_size=self.action_space_size,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
        ).to(config.device)

        logger.info(
            f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}",
        )

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Initialize scheduler
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            decay_steps=config.lr_decay_steps,
            decay_rate=config.lr_decay_rate,
            base_lr=config.learning_rate,
        )

        # Initialize gradient accumulator
        self.grad_accumulator = GradientAccumulator(
            self.network,
            self.optimizer,
            accumulation_steps=config.gradient_accumulation_steps,
            gradient_clip=config.gradient_clip,
        )

        # Statistics
        self.training_step = 0
        self.total_loss = 0.0
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.start_time = 0.0

        # Control
        self._running = False

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        logger.info(f"Learner initialized on {config.device}")

    def _publish_weights(self) -> None:
        """Publish current weights to shared storage."""
        state_dict = self.network.state_dict()
        version = self.storage.set_model_weights(state_dict)
        logger.debug(f"Published weights v{version}")

    def _save_checkpoint(self, step: int) -> str:
        """Save a training checkpoint.

        Args:
            step: Current training step.

        Returns:
            Checkpoint path.
        """
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{step}.pt",
        )

        torch.save(
            {
                "step": step,
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_step": self.lr_scheduler.current_step,
            },
            checkpoint_path,
        )

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, path: str) -> int:
        """Load a training checkpoint.

        Args:
            path: Checkpoint path.

        Returns:
            Training step from checkpoint.

        Note:
            Checkpoints containing optimizer state may require unsafe loading.
            Only load checkpoints from trusted sources.
        """
        result = safe_load_checkpoint(
            path,
            map_location=self.config.device,
            strict_security=False,
            allow_unsafe_fallback=True,
        )
        checkpoint = result.data

        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.current_step = checkpoint["lr_scheduler_step"]
        step = int(checkpoint["step"])

        if not result.loaded_safely:
            logger.warning(
                "Checkpoint loaded with weights_only=False due to optimizer state. "
                "Ensure checkpoint is from a trusted source."
            )
        logger.info(f"Loaded checkpoint from {path} (step {step})")
        return step

    def train_step(self) -> dict[str, float] | None:
        """Perform a single training step.

        Returns:
            Dictionary of losses, or None if buffer not ready.
        """
        # Process pending trajectories
        self.replay_buffer.process_pending_trajectories()

        # Check if buffer is ready
        if not self.replay_buffer.is_ready():
            return None

        # Sample batch
        batch = self.replay_buffer.sample_batch(self.config.batch_size)
        if not batch:
            return None

        # Move to device
        spatial = batch["spatial"].to(self.config.device)
        vector = batch["vector"].to(self.config.device)
        action_mask = batch["action_mask"].to(self.config.device)
        target_policy = batch["policy"].to(self.config.device)
        target_value = batch["value"].to(self.config.device)

        # Forward pass
        self.network.train()
        policy_logits, value = self.network(spatial, vector, action_mask)

        # Compute loss
        total_loss, p_loss, v_loss = compute_loss(
            policy_logits,
            value,
            target_policy,
            target_value,
            action_mask,
        )

        # Accumulate gradients and step
        did_step = self.grad_accumulator.accumulate(total_loss)

        if did_step:
            self.training_step += 1
            self.lr_scheduler.step()

        # Update statistics
        self.total_loss += total_loss.item()
        self.policy_loss += p_loss.item()
        self.value_loss += v_loss.item()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": p_loss.item(),
            "value_loss": v_loss.item(),
            "learning_rate": self.lr_scheduler.get_lr(),
        }

    def run(self) -> None:
        """Main learner loop.

        Trains the network until stopped or max steps reached.
        """
        self._running = True
        self.start_time = time.time()
        self.storage.set_training(True)

        # Publish initial weights
        self._publish_weights()

        logger.info("Learner: Starting training loop")

        # Wait for buffer to have enough data
        while self._running and not self.storage.should_stop():
            self.replay_buffer.process_pending_trajectories()
            if self.replay_buffer.is_ready():
                break
            logger.info(
                f"Waiting for replay buffer "
                f"({self.replay_buffer.size()}/{self.config.min_buffer_size})",
            )
            time.sleep(1.0)

        if not self._running or self.storage.should_stop():
            self._running = False
            self.storage.set_training(False)
            return

        logger.info("Replay buffer ready, starting training")

        step_losses = {"total": 0.0, "policy": 0.0, "value": 0.0}
        log_steps = 0

        while self._running and not self.storage.should_stop():
            # Check max steps
            if (
                self.config.max_training_steps > 0
                and self.training_step >= self.config.max_training_steps
            ):
                logger.info(
                    f"Reached max training steps ({self.config.max_training_steps})",
                )
                break

            # Training step
            losses = self.train_step()

            if losses is None:
                # Buffer empty, wait for more data
                time.sleep(0.1)
                continue

            # Accumulate for logging
            step_losses["total"] += losses["total_loss"]
            step_losses["policy"] += losses["policy_loss"]
            step_losses["value"] += losses["value_loss"]
            log_steps += 1

            # Logging
            if self.training_step % self.config.log_interval == 0 and log_steps > 0:
                avg_total = step_losses["total"] / log_steps
                avg_policy = step_losses["policy"] / log_steps
                avg_value = step_losses["value"] / log_steps

                elapsed = time.time() - self.start_time
                steps_per_sec = self.training_step / elapsed if elapsed > 0 else 0

                logger.info(
                    f"Step {self.training_step}: "
                    f"loss={avg_total:.4f} "
                    f"(policy={avg_policy:.4f}, value={avg_value:.4f}), "
                    f"lr={losses['learning_rate']:.6f}, "
                    f"{steps_per_sec:.1f} steps/sec",
                )

                # Store in shared storage
                self.storage.add_loss_entry(
                    {
                        "step": self.training_step,
                        "total_loss": avg_total,
                        "policy_loss": avg_policy,
                        "value_loss": avg_value,
                        "learning_rate": losses["learning_rate"],
                    }
                )

                step_losses = {"total": 0.0, "policy": 0.0, "value": 0.0}
                log_steps = 0

            # Publish weights
            if self.training_step % self.config.publish_interval == 0:
                self._publish_weights()
                self.storage.set_training_step(self.training_step)

            # Checkpointing
            if (
                self.config.checkpoint_interval > 0
                and self.training_step % self.config.checkpoint_interval == 0
            ):
                self._save_checkpoint(self.training_step)

        # Final checkpoint
        if self.training_step > 0:
            self._save_checkpoint(self.training_step)
            self._publish_weights()

        self._running = False
        self.storage.set_training(False)
        logger.info(f"Learner: Stopped after {self.training_step} steps")

    def stop(self) -> None:
        """Signal the learner to stop."""
        self._running = False

    def is_running(self) -> bool:
        """Check if learner is running."""
        return self._running

    def get_statistics(self) -> dict:
        """Get learner statistics.

        Returns:
            Dictionary with statistics.
        """
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        return {
            "training_step": self.training_step,
            "elapsed_time": elapsed,
            "steps_per_second": self.training_step / elapsed if elapsed > 0 else 0,
            "model_version": self.storage.get_model_version(),
            "buffer_size": self.replay_buffer.size(),
            "learning_rate": self.lr_scheduler.get_lr(),
            "is_running": self._running,
        }


def learner_process(
    config_dict: dict[str, Any],
    storage: SharedStorage,
    replay_buffer: DistributedReplayBuffer,
) -> None:
    """Entry point for learner subprocess.

    Args:
        config_dict: Learner configuration dictionary.
        storage: Shared storage instance.
        replay_buffer: Replay buffer instance.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Learner] [%(levelname)s] %(message)s",
    )

    # Handle signals
    def signal_handler(signum, frame):
        logger.info(f"Learner: Received signal {signum}")
        storage.request_stop()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Create config
    config = LearnerConfig(**config_dict)

    # Create and run learner
    learner = Learner(config, storage, replay_buffer)

    try:
        learner.run()
    except Exception as e:
        logger.error(f"Learner crashed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        logger.info("Learner: Process exiting")
