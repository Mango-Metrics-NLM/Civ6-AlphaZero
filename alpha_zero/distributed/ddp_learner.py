"""Distributed Data Parallel (DDP) learner for multi-node training.

Provides DDPLearner for scaling training across multiple GPUs and nodes
using PyTorch's DistributedDataParallel. Integrates with cloud infrastructure
for checkpointing and monitoring.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from alpha_zero.distributed.learner import Learner, LearnerConfig
from alpha_zero.distributed.shared_storage import DistributedReplayBuffer, SharedStorage

logger = logging.getLogger(__name__)


@dataclass
class DDPLearnerConfig(LearnerConfig):
    """Configuration for DDP-enabled learner.

    Extends LearnerConfig with distributed training settings.

    Attributes:
        backend: PyTorch distributed backend (nccl, gloo, mpi).
        init_method: URL for process group initialization.
        world_size: Total number of processes across all nodes.
        rank: Global rank of this process.
        local_rank: Local rank within this node.
        master_addr: Address of master node.
        master_port: Port for master node.
        find_unused_parameters: DDP flag for unused parameters.
        broadcast_buffers: Whether to broadcast buffers in DDP.
        gradient_as_bucket_view: Memory optimization for gradients.
    """

    # Distributed settings
    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = ""
    master_port: str = "29500"

    # DDP settings
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    gradient_as_bucket_view: bool = True

    # Gradient sync
    gradient_sync_interval: int = 1

    @classmethod
    def from_env(cls, **kwargs: Any) -> DDPLearnerConfig:
        """Create config from environment variables.

        Standard environment variables from torch.distributed:
        - WORLD_SIZE
        - RANK
        - LOCAL_RANK
        - MASTER_ADDR
        - MASTER_PORT

        Args:
            **kwargs: Override values for config.

        Returns:
            DDPLearnerConfig instance.
        """
        env_values = {
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "master_addr": os.environ.get("MASTER_ADDR", "localhost"),
            "master_port": os.environ.get("MASTER_PORT", "29500"),
        }
        env_values.update(kwargs)
        return cls(**env_values)


class DDPLearner(Learner):
    """Distributed Data Parallel learner for multi-GPU/multi-node training.

    Extends the base Learner with DDP support for:
    - Multi-GPU training within a single node
    - Multi-node training across machines
    - Automatic gradient synchronization
    - Checkpoint coordination across workers

    Example:
        >>> config = DDPLearnerConfig.from_env()
        >>> learner = DDPLearner(config, shared_storage, replay_buffer)
        >>> learner.initialize()
        >>> learner.run()
        >>> learner.cleanup()
    """

    def __init__(
        self,
        config: DDPLearnerConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
        network: nn.Module | None = None,
    ) -> None:
        """Initialize DDP learner.

        Args:
            config: DDP learner configuration.
            shared_storage: Shared storage for weights.
            replay_buffer: Replay buffer for experience.
            network: Optional pre-initialized network (overrides base class network).
        """
        super().__init__(config, shared_storage, replay_buffer)

        # Override network if provided
        if network is not None:
            self._network = network
        else:
            self._network = self.network  # Use network created by base class

        self._ddp_config = config
        self._device: torch.device = torch.device("cpu")
        self._ddp_model: DDP | None = None
        self._is_initialized = False

    @property
    def rank(self) -> int:
        """Get global rank of this process."""
        return self._ddp_config.rank

    @property
    def world_size(self) -> int:
        """Get total number of processes."""
        return self._ddp_config.world_size

    @property
    def is_main(self) -> bool:
        """Check if this is the main (rank 0) process."""
        return self._ddp_config.rank == 0

    @property
    def local_rank(self) -> int:
        """Get local rank within this node."""
        return self._ddp_config.local_rank

    def initialize(self) -> None:
        """Initialize distributed process group and DDP model."""
        if self._is_initialized:
            return

        # Set up device
        if torch.cuda.is_available():
            torch.cuda.set_device(self._ddp_config.local_rank)
            self._device = torch.device(f"cuda:{self._ddp_config.local_rank}")
        else:
            self._device = torch.device("cpu")

        # Initialize process group
        if self._ddp_config.world_size > 1:
            if not dist.is_initialized():
                # Set environment for init
                if self._ddp_config.master_addr:
                    os.environ.setdefault("MASTER_ADDR", self._ddp_config.master_addr)
                if self._ddp_config.master_port:
                    os.environ.setdefault("MASTER_PORT", self._ddp_config.master_port)

                dist.init_process_group(
                    backend=self._ddp_config.backend,
                    init_method=self._ddp_config.init_method,
                    world_size=self._ddp_config.world_size,
                    rank=self._ddp_config.rank,
                )
                logger.info(
                    "Initialized process group: rank=%d, world_size=%d",
                    self._ddp_config.rank,
                    self._ddp_config.world_size,
                )

        # Move model to device and wrap with DDP
        if self._network is not None:
            self._network = self._network.to(self._device)

            if self._ddp_config.world_size > 1:
                self._ddp_model = DDP(
                    self._network,
                    device_ids=[self._ddp_config.local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=self._ddp_config.find_unused_parameters,
                    broadcast_buffers=self._ddp_config.broadcast_buffers,
                    gradient_as_bucket_view=self._ddp_config.gradient_as_bucket_view,
                )
            else:
                self._ddp_model = None

        self._is_initialized = True
        logger.info("DDPLearner initialized on device %s", self._device)

    def cleanup(self) -> None:
        """Cleanup distributed resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
        self._is_initialized = False
        logger.info("DDPLearner cleanup complete")

    def get_model(self) -> nn.Module:
        """Get the model for training (DDP-wrapped if applicable).

        Returns:
            DDP model or base model.
        """
        if self._ddp_model is not None:
            return self._ddp_model
        return self._network  # type: ignore[return-value]

    def get_base_model(self) -> nn.Module:
        """Get the underlying model (unwrapped from DDP).

        Returns:
            Base network module.
        """
        if self._ddp_model is not None:
            return self._ddp_model.module  # type: ignore[return-value]
        return self._network  # type: ignore[return-value]

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Perform one training step with gradient synchronization.

        Args:
            batch: Training batch with states, policies, values.

        Returns:
            Dictionary of loss values.
        """
        model = self.get_model()
        model.train()

        # Move batch to device
        batch = {k: v.to(self._device) for k, v in batch.items()}

        # Forward pass
        outputs = model(batch["states"])

        # Compute losses (override in subclass for specific loss computation)
        losses = self._compute_losses(outputs, batch)

        # Backward pass
        total_loss = losses.get("total_loss", sum(losses.values()))
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.max_grad_norm,
            )

        # Optimizer step
        if self._optimizer is not None:
            self._optimizer.step()
            self._optimizer.zero_grad()

        # Convert losses to floats for logging
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    def _compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute training losses.

        Override in subclass for specific loss functions.

        Args:
            outputs: Model outputs.
            batch: Training batch.

        Returns:
            Dictionary of loss tensors.
        """
        # Placeholder - subclass should override with actual loss computation
        losses: dict[str, torch.Tensor] = {}
        return losses

    def sync_weights_to_main(self) -> None:
        """Synchronize weights to main process (rank 0).

        Only needed if doing async training with separate weight updates.
        """
        if not dist.is_initialized() or self.world_size == 1:
            return

        model = self.get_base_model()
        for param in model.parameters():
            dist.reduce(param.data, dst=0, op=dist.ReduceOp.SUM)
            if self.is_main:
                param.data /= self.world_size

    def broadcast_weights_from_main(self) -> None:
        """Broadcast weights from main process to all workers."""
        if not dist.is_initialized() or self.world_size == 1:
            return

        model = self.get_base_model()
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    def save_checkpoint(
        self,
        path: str,
        step: int,
        additional_state: dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint (only on main process by default).

        Includes barrier synchronization to ensure all processes reach
        the checkpoint point before main saves. This prevents race conditions
        where workers might try to load an incomplete checkpoint.

        Args:
            path: Path to save checkpoint.
            step: Training step number.
            additional_state: Additional state to save.
        """
        # Synchronize all processes before checkpoint
        self.barrier()

        if not self.is_main:
            logger.debug("Skipping checkpoint save on non-main rank %d", self.rank)
            # Wait for main to finish saving
            self.barrier()
            return

        state = {
            "step": step,
            "model_state_dict": self.get_base_model().state_dict(),
            "rank": self.rank,
            "world_size": self.world_size,
            "config": {
                "backend": self.config.backend,
                "gradient_sync_interval": self.config.gradient_sync_interval,
            },
        }

        optimizer = getattr(self, "_optimizer", None)
        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()

        if additional_state:
            state.update(additional_state)

        torch.save(state, path)
        logger.info("Saved checkpoint to %s at step %d (world_size=%d)", path, step, self.world_size)

        # Signal completion to other processes
        self.barrier()

    def load_checkpoint(
        self,
        path: str,
        strict: bool = True,
        validate_world_size: bool = True,
    ) -> dict[str, Any]:
        """Load checkpoint and restore model state.

        All processes load the same checkpoint. Includes validation
        of distributed training metadata.

        Args:
            path: Path to checkpoint file.
            strict: Whether to strictly enforce state dict matching.
            validate_world_size: Whether to warn if world_size differs.

        Returns:
            Loaded checkpoint state.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
            RuntimeError: If checkpoint is corrupted or incompatible.
        """
        import os

        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        map_location = self._device if torch.cuda.is_available() else "cpu"

        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {e}") from e

        # Validate checkpoint structure
        if "model_state_dict" not in checkpoint:
            raise RuntimeError("Invalid checkpoint: missing 'model_state_dict' key")

        # Validate distributed training metadata
        checkpoint_world_size = checkpoint.get("world_size", 1)
        if validate_world_size and checkpoint_world_size != self.world_size:
            logger.warning(
                "Checkpoint world_size (%d) differs from current world_size (%d). "
                "This may cause issues with optimizer state restoration.",
                checkpoint_world_size,
                self.world_size,
            )

        checkpoint_step = checkpoint.get("step", 0)
        logger.info(
            "Loading checkpoint from step %d (saved with world_size=%d)",
            checkpoint_step,
            checkpoint_world_size,
        )

        self.get_base_model().load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict,
        )

        optimizer = getattr(self, "_optimizer", None)
        if "optimizer_state_dict" in checkpoint and optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                logger.warning(
                    "Failed to load optimizer state: %s. "
                    "Optimizer will start fresh.",
                    e,
                )

        logger.info("Loaded checkpoint from %s (rank=%d)", path, self.rank)
        return checkpoint

    def barrier(self) -> None:
        """Synchronization barrier across all processes."""
        if dist.is_initialized() and self.world_size > 1:
            dist.barrier()

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        """All-reduce a tensor across all processes.

        Args:
            tensor: Tensor to reduce.
            op: Reduction operation.

        Returns:
            Reduced tensor (modified in-place).
        """
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(tensor, op=op)
        return tensor

    def gather(
        self,
        tensor: torch.Tensor,
        dst: int = 0,
    ) -> list[torch.Tensor] | None:
        """Gather tensors from all processes to destination.

        Args:
            tensor: Local tensor to gather.
            dst: Destination rank.

        Returns:
            List of gathered tensors on dst, None on other ranks.
        """
        if not dist.is_initialized() or self.world_size == 1:
            return [tensor]

        if self.rank == dst:
            gathered = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.gather(tensor, gather_list=gathered, dst=dst)
            return gathered
        else:
            dist.gather(tensor, gather_list=None, dst=dst)
            return None
