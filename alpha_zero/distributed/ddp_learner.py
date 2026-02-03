"""Distributed Data Parallel learner for multi-GPU training.

Extends the base Learner with:
- PyTorch DistributedDataParallel wrapper
- Gradient synchronization across processes
- Distributed sampling coordination
- Checkpoint coordination (only rank 0 saves)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817

from alpha_zero.core.civ6_network import MicroCivNetwork, compute_loss
from alpha_zero.distributed.ddp_config import DDPConfig
from alpha_zero.distributed.learner import (
    GradientAccumulator,
    Learner,
    LearnerConfig,
    LearningRateScheduler,
)
from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    SharedStorage,
)
from alpha_zero.envs.microciv import MicroCivEnv

if TYPE_CHECKING:
    from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


@dataclass
class DDPLearnerConfig(LearnerConfig):
    """Configuration for DDP-enabled learner.

    Extends LearnerConfig with DDP-specific settings.
    """

    # DDP settings (can also be configured via DDPConfig)
    ddp_enabled: bool = False
    ddp_world_size: int = 1
    ddp_rank: int = 0
    ddp_local_rank: int = 0
    ddp_backend: str = "auto"

    # Mixed precision training
    use_amp: bool = False
    amp_dtype: str = "float16"

    # Gradient synchronization
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25

    # Distributed checkpointing
    checkpoint_only_on_main: bool = True

    # Sync settings
    sync_batch_norm: bool = True

    @classmethod
    def from_learner_config(
        cls,
        learner_config: LearnerConfig,
        ddp_config: DDPConfig | None = None,
    ) -> DDPLearnerConfig:
        """Create from base LearnerConfig and optional DDPConfig.

        Args:
            learner_config: Base learner configuration.
            ddp_config: Optional DDP configuration.

        Returns:
            DDPLearnerConfig instance.
        """
        # Start with learner config values
        config = cls(
            board_size=learner_config.board_size,
            num_res_blocks=learner_config.num_res_blocks,
            num_filters=learner_config.num_filters,
            batch_size=learner_config.batch_size,
            learning_rate=learner_config.learning_rate,
            weight_decay=learner_config.weight_decay,
            gradient_clip=learner_config.gradient_clip,
            lr_decay_steps=learner_config.lr_decay_steps,
            lr_decay_rate=learner_config.lr_decay_rate,
            gradient_accumulation_steps=learner_config.gradient_accumulation_steps,
            training_steps_per_update=learner_config.training_steps_per_update,
            min_buffer_size=learner_config.min_buffer_size,
            max_training_steps=learner_config.max_training_steps,
            publish_interval=learner_config.publish_interval,
            checkpoint_dir=learner_config.checkpoint_dir,
            checkpoint_interval=learner_config.checkpoint_interval,
            device=learner_config.device,
            log_interval=learner_config.log_interval,
        )

        # Apply DDP config if provided
        if ddp_config is not None and ddp_config.enabled:
            config.ddp_enabled = True
            config.ddp_world_size = ddp_config.world_size
            config.ddp_rank = ddp_config.rank
            config.ddp_local_rank = ddp_config.local_rank
            config.ddp_backend = ddp_config.get_backend()
            config.use_amp = ddp_config.use_amp
            config.amp_dtype = ddp_config.amp_dtype
            config.find_unused_parameters = ddp_config.find_unused_parameters
            config.broadcast_buffers = ddp_config.broadcast_buffers
            config.bucket_cap_mb = ddp_config.bucket_cap_mb
            config.device = ddp_config.get_device()

        return config


class DDPLearner(Learner):
    """Distributed Data Parallel learner for multi-GPU training.

    Extends the base Learner with:
    - Automatic DDP wrapping of the network
    - Gradient synchronization across processes
    - Mixed precision training support
    - Checkpoint coordination (only rank 0 saves)

    When DDP is disabled (world_size=1), behaves identically to base Learner.
    """

    def __init__(
        self,
        config: DDPLearnerConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
        ddp_config: DDPConfig | None = None,
    ) -> None:
        """Initialize DDP learner.

        Args:
            config: DDP learner configuration.
            shared_storage: Shared storage for weights.
            replay_buffer: Replay buffer for training data.
            ddp_config: Optional separate DDP configuration.
        """
        # Store DDP config before parent init
        self.ddp_config = ddp_config or DDPConfig(
            enabled=config.ddp_enabled,
            world_size=config.ddp_world_size,
            rank=config.ddp_rank,
            local_rank=config.ddp_local_rank,
            backend=config.ddp_backend,
        )

        # Configure device before parent init
        if self.ddp_config.enabled:
            config.device = self.ddp_config.get_device()

        # Initialize process group before network creation
        self._process_group_initialized = False
        if self.ddp_config.enabled:
            self._init_process_group()

        # Parent initialization creates network on correct device
        # We'll wrap it with DDP after
        self.config: DDPLearnerConfig = config
        self.storage = shared_storage
        self.replay_buffer = replay_buffer

        # Initialize environment to get action space size
        env = MicroCivEnv(board_size=config.board_size)
        self.action_space_size = env.action_space.n

        # Create network on the correct device
        self.network = MicroCivNetwork(
            board_size=config.board_size,
            action_space_size=self.action_space_size,
            num_res_blocks=config.num_res_blocks,
            num_filters=config.num_filters,
        ).to(config.device)

        # Apply SyncBatchNorm if enabled and using DDP
        if self.ddp_config.enabled and config.sync_batch_norm:
            self.network = nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            logger.info("Applied SyncBatchNorm conversion")

        # Wrap with DDP if enabled
        if self.ddp_config.enabled:
            self.network = self._wrap_ddp(self.network)

        # Log network info (only on main process)
        if self.is_main_process():
            param_count = sum(p.numel() for p in self.network.parameters())
            logger.info(f"Network parameters: {param_count:,}")

        # Initialize optimizer (on wrapped network)
        self.optimizer = torch.optim.Adam(
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
            self._get_base_network(),
            self.optimizer,
            accumulation_steps=config.gradient_accumulation_steps,
            gradient_clip=config.gradient_clip,
        )

        # AMP scaler for mixed precision
        self.scaler: GradScaler | None = None
        if config.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
            logger.info(f"Mixed precision enabled with {config.amp_dtype}")

        # Statistics
        self.training_step = 0
        self.total_loss = 0.0
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.start_time = 0.0

        # Control
        self._running = False

        # Create checkpoint directory (only on main process)
        if self.is_main_process():
            os.makedirs(config.checkpoint_dir, exist_ok=True)

        if self.is_main_process():
            logger.info(
                f"DDPLearner initialized on {config.device} "
                f"(rank {self.ddp_config.rank}/{self.ddp_config.world_size})"
            )

    def _init_process_group(self) -> None:
        """Initialize the distributed process group."""
        if self._process_group_initialized:
            return

        if dist.is_initialized():
            logger.info("Process group already initialized")
            self._process_group_initialized = True
            return

        backend = self.ddp_config.get_backend()
        timeout = timedelta(seconds=self.ddp_config.timeout_seconds)

        logger.info(
            f"Initializing process group: "
            f"backend={backend}, "
            f"rank={self.ddp_config.rank}, "
            f"world_size={self.ddp_config.world_size}"
        )

        dist.init_process_group(
            backend=backend,
            init_method=self.ddp_config.init_method,
            world_size=self.ddp_config.world_size,
            rank=self.ddp_config.rank,
            timeout=timeout,
        )

        self._process_group_initialized = True
        logger.info("Process group initialized successfully")

    def _wrap_ddp(self, network: nn.Module) -> nn.Module:
        """Wrap network with DistributedDataParallel.

        Args:
            network: Network to wrap.

        Returns:
            DDP-wrapped network.
        """
        device_ids = [self.ddp_config.local_rank] if self.ddp_config.use_cuda else None
        output_device = self.ddp_config.local_rank if self.ddp_config.use_cuda else None

        wrapped = DDP(
            network,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=self.config.find_unused_parameters,
            broadcast_buffers=self.config.broadcast_buffers,
            bucket_cap_mb=self.config.bucket_cap_mb,
            gradient_as_bucket_view=True,
        )

        logger.info(f"Network wrapped with DDP (device_ids={device_ids})")
        return wrapped

    def _get_base_network(self) -> nn.Module:
        """Get the unwrapped base network.

        Returns:
            Base network without DDP wrapper.
        """
        if isinstance(self.network, DDP):
            return self.network.module
        return self.network

    def is_main_process(self) -> bool:
        """Check if this is the main (rank 0) process.

        Returns:
            True if rank 0 or DDP not enabled.
        """
        if not self.ddp_config.enabled:
            return True
        return self.ddp_config.rank == 0

    def train_step(self) -> dict[str, float] | None:
        """Perform a single training step with DDP support.

        Returns:
            Dictionary of losses, or None if buffer not ready.
        """
        # Process pending trajectories (all processes)
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

        # Forward pass with optional AMP
        self.network.train()

        if self.config.use_amp and self.scaler is not None:
            with torch.amp.autocast(
                device_type="cuda",
                dtype=getattr(torch, self.config.amp_dtype),
            ):
                policy_logits, value = self.network(spatial, vector, action_mask)
                total_loss, p_loss, v_loss = compute_loss(
                    policy_logits,
                    value,
                    target_policy,
                    target_value,
                    action_mask,
                )

            # Backward with scaler
            self.scaler.scale(total_loss / self.config.gradient_accumulation_steps).backward()
            self.grad_accumulator.step_count += 1

            if self.grad_accumulator.step_count >= self.config.gradient_accumulation_steps:
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.config.gradient_clip,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.grad_accumulator.step_count = 0
                did_step = True
            else:
                did_step = False
        else:
            # Standard training without AMP
            policy_logits, value = self.network(spatial, vector, action_mask)
            total_loss, p_loss, v_loss = compute_loss(
                policy_logits,
                value,
                target_policy,
                target_value,
                action_mask,
            )
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

    def _publish_weights(self) -> None:
        """Publish current weights to shared storage.

        Only rank 0 publishes weights.
        """
        if not self.is_main_process():
            return

        # Get base network state dict (without DDP wrapper)
        state_dict = self._get_base_network().state_dict()
        version = self.storage.set_model_weights(state_dict)
        logger.debug(f"Published weights v{version}")

    def _save_checkpoint(self, step: int) -> str:
        """Save a training checkpoint.

        Only rank 0 saves checkpoints when checkpoint_only_on_main is True.

        Args:
            step: Current training step.

        Returns:
            Checkpoint path (empty string if not saved).
        """
        if self.config.checkpoint_only_on_main and not self.is_main_process():
            # Synchronize at checkpoint (barrier)
            if self.ddp_config.enabled:
                dist.barrier()
            return ""

        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{step}.pt",
        )

        # Save base network state dict
        save_dict: dict[str, Any] = {
            "step": step,
            "network_state_dict": self._get_base_network().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_step": self.lr_scheduler.current_step,
        }

        if self.scaler is not None:
            save_dict["scaler_state_dict"] = self.scaler.state_dict()

        # Add DDP info
        if self.ddp_config.enabled:
            save_dict["ddp_world_size"] = self.ddp_config.world_size
            save_dict["ddp_rank"] = self.ddp_config.rank

        torch.save(save_dict, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Barrier after save so all processes wait
        if self.ddp_config.enabled:
            dist.barrier()

        return checkpoint_path

    def load_checkpoint(self, path: str) -> int:
        """Load a training checkpoint.

        All processes load the checkpoint for consistency.

        Args:
            path: Checkpoint path.

        Returns:
            Training step from checkpoint.
        """
        from alpha_zero.utils.checkpoint import safe_load_checkpoint

        result = safe_load_checkpoint(
            path,
            map_location=self.config.device,
            strict_security=False,
            allow_unsafe_fallback=True,
        )
        checkpoint = result.data

        # Load into base network
        self._get_base_network().load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.current_step = checkpoint["lr_scheduler_step"]
        step = int(checkpoint["step"])

        # Load scaler if available
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if not result.loaded_safely:
            logger.warning(
                "Checkpoint loaded with weights_only=False due to optimizer state. "
                "Ensure checkpoint is from a trusted source."
            )

        # Synchronize after load
        if self.ddp_config.enabled:
            dist.barrier()

        if self.is_main_process():
            logger.info(f"Loaded checkpoint from {path} (step {step})")

        return step

    def synchronize_weights(self) -> None:
        """Synchronize weights across all processes.

        Broadcasts weights from rank 0 to all other processes.
        """
        if not self.ddp_config.enabled:
            return

        base_network = self._get_base_network()

        for param in base_network.parameters():
            dist.broadcast(param.data, src=0)

        logger.debug("Synchronized weights across processes")

    def all_reduce_losses(
        self,
        losses: dict[str, float],
    ) -> dict[str, float]:
        """Average losses across all processes.

        Args:
            losses: Local loss dictionary.

        Returns:
            Averaged losses.
        """
        if not self.ddp_config.enabled:
            return losses

        # Create tensor for all-reduce
        loss_tensor = torch.tensor(
            [losses["total_loss"], losses["policy_loss"], losses["value_loss"]],
            device=self.config.device,
        )

        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_tensor /= self.ddp_config.world_size

        return {
            "total_loss": loss_tensor[0].item(),
            "policy_loss": loss_tensor[1].item(),
            "value_loss": loss_tensor[2].item(),
            "learning_rate": losses.get("learning_rate", 0.0),
        }

    def run(self) -> None:
        """Main learner loop with DDP support.

        Trains the network until stopped or max steps reached.
        Logging and checkpointing only on main process.
        """
        self._running = True
        self.start_time = time.time()
        self.storage.set_training(True)

        # Publish initial weights (only rank 0)
        self._publish_weights()

        if self.is_main_process():
            logger.info(
                f"DDPLearner: Starting training loop "
                f"(rank {self.ddp_config.rank}/{self.ddp_config.world_size})"
            )

        # Wait for buffer to have enough data
        while self._running and not self.storage.should_stop():
            self.replay_buffer.process_pending_trajectories()
            if self.replay_buffer.is_ready():
                break
            if self.is_main_process():
                logger.info(
                    f"Waiting for replay buffer "
                    f"({self.replay_buffer.size()}/{self.config.min_buffer_size})"
                )
            time.sleep(1.0)

        if not self._running or self.storage.should_stop():
            self._running = False
            self.storage.set_training(False)
            return

        if self.is_main_process():
            logger.info("Replay buffer ready, starting training")

        step_losses = {"total": 0.0, "policy": 0.0, "value": 0.0}
        log_steps = 0

        while self._running and not self.storage.should_stop():
            # Check max steps
            if (
                self.config.max_training_steps > 0
                and self.training_step >= self.config.max_training_steps
            ):
                if self.is_main_process():
                    logger.info(
                        f"Reached max training steps ({self.config.max_training_steps})"
                    )
                break

            # Training step
            losses = self.train_step()

            if losses is None:
                time.sleep(0.1)
                continue

            # Accumulate for logging
            step_losses["total"] += losses["total_loss"]
            step_losses["policy"] += losses["policy_loss"]
            step_losses["value"] += losses["value_loss"]
            log_steps += 1

            # Logging (only on main process)
            if (
                self.training_step % self.config.log_interval == 0
                and log_steps > 0
                and self.is_main_process()
            ):
                avg_total = step_losses["total"] / log_steps
                avg_policy = step_losses["policy"] / log_steps
                avg_value = step_losses["value"] / log_steps

                elapsed = time.time() - self.start_time
                steps_per_sec = self.training_step / elapsed if elapsed > 0 else 0

                # Account for all workers
                effective_throughput = steps_per_sec * self.ddp_config.world_size

                logger.info(
                    f"Step {self.training_step}: "
                    f"loss={avg_total:.4f} "
                    f"(policy={avg_policy:.4f}, value={avg_value:.4f}), "
                    f"lr={losses['learning_rate']:.6f}, "
                    f"{effective_throughput:.1f} steps/sec (x{self.ddp_config.world_size})"
                )

                self.storage.add_loss_entry({
                    "step": self.training_step,
                    "total_loss": avg_total,
                    "policy_loss": avg_policy,
                    "value_loss": avg_value,
                    "learning_rate": losses["learning_rate"],
                })

                step_losses = {"total": 0.0, "policy": 0.0, "value": 0.0}
                log_steps = 0

            # Publish weights
            if self.training_step % self.config.publish_interval == 0:
                self._publish_weights()
                if self.is_main_process():
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

        if self.is_main_process():
            logger.info(f"DDPLearner: Stopped after {self.training_step} steps")

    def cleanup(self) -> None:
        """Clean up distributed resources."""
        if self._process_group_initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._process_group_initialized = False
            logger.info("Destroyed process group")

    def get_statistics(self) -> dict:
        """Get learner statistics.

        Returns:
            Dictionary with statistics.
        """
        elapsed = time.time() - self.start_time if self.start_time > 0 else 0
        steps_per_sec = self.training_step / elapsed if elapsed > 0 else 0

        return {
            "training_step": self.training_step,
            "elapsed_time": elapsed,
            "steps_per_second": steps_per_sec,
            "effective_throughput": steps_per_sec * self.ddp_config.world_size,
            "model_version": self.storage.get_model_version(),
            "buffer_size": self.replay_buffer.size(),
            "learning_rate": self.lr_scheduler.get_lr(),
            "is_running": self._running,
            "ddp_enabled": self.ddp_config.enabled,
            "ddp_rank": self.ddp_config.rank,
            "ddp_world_size": self.ddp_config.world_size,
        }


def create_ddp_learner(
    learner_config: LearnerConfig,
    ddp_config: DDPConfig | None = None,
    shared_storage: SharedStorage | None = None,
    replay_buffer: DistributedReplayBuffer | None = None,
) -> DDPLearner:
    """Factory function to create a DDP learner.

    Args:
        learner_config: Base learner configuration.
        ddp_config: Optional DDP configuration. If None, detects from env.
        shared_storage: Optional shared storage. If None, creates new.
        replay_buffer: Optional replay buffer. If None, creates new.

    Returns:
        Configured DDPLearner instance.
    """
    # Detect DDP environment if not provided
    if ddp_config is None:
        from alpha_zero.distributed.ddp_config import detect_ddp_environment
        ddp_config = detect_ddp_environment()

    # Create combined config
    config = DDPLearnerConfig.from_learner_config(learner_config, ddp_config)

    # Create storage and buffer if not provided
    if shared_storage is None:
        shared_storage = SharedStorage()
    if replay_buffer is None:
        replay_buffer = DistributedReplayBuffer()

    return DDPLearner(
        config=config,
        shared_storage=shared_storage,
        replay_buffer=replay_buffer,
        ddp_config=ddp_config,
    )
