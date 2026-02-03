"""Configuration for PyTorch Distributed Data Parallel (DDP) training.

Provides environment detection and configuration for multi-GPU/multi-node training.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class DDPBackend(str, Enum):
    """DDP communication backend."""

    NCCL = "nccl"  # Optimized for NVIDIA GPUs
    GLOO = "gloo"  # CPU and cross-platform
    MPI = "mpi"  # HPC environments
    AUTO = "auto"  # Auto-detect based on hardware


@dataclass
class DDPConfig:
    """Configuration for Distributed Data Parallel training.

    This class supports:
    - Environment variable detection for cloud/cluster deployments
    - Manual configuration for local multi-GPU testing
    - Backward compatibility with single-GPU training (world_size=1)

    Environment variables detected:
    - WORLD_SIZE: Total number of processes
    - RANK / LOCAL_RANK: Process rank
    - MASTER_ADDR: Coordinator address
    - MASTER_PORT: Coordinator port
    - CUDA_VISIBLE_DEVICES: GPU assignment
    """

    # Distributed settings
    enabled: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    # Communication
    master_addr: str = "localhost"
    master_port: int = 29500
    backend: str = DDPBackend.AUTO.value

    # Process group settings
    init_method: str = "env://"
    timeout_seconds: int = 1800  # 30 minutes for large model init

    # GPU settings
    use_cuda: bool = True
    device_ids: list[int] = field(default_factory=list)
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True

    # Gradient settings
    gradient_as_bucket_view: bool = True
    bucket_cap_mb: int = 25

    # Mixed precision
    use_amp: bool = False
    amp_dtype: str = "float16"  # float16 or bfloat16

    # Fault tolerance
    elastic_training: bool = False
    min_world_size: int = 1
    max_restarts: int = 3

    @classmethod
    def from_env(cls) -> DDPConfig:
        """Create configuration from environment variables.

        Detects common distributed environment variables set by:
        - torchrun / torch.distributed.launch
        - Vertex AI Training
        - Kubernetes operators (KubeFlow, PyTorch Operator)
        - SLURM

        Returns:
            DDPConfig with detected settings.
        """
        config = cls()

        # Detect world size and rank
        world_size = os.environ.get("WORLD_SIZE")
        if world_size is not None:
            config.world_size = int(world_size)
            config.enabled = config.world_size > 1

        rank = os.environ.get("RANK")
        if rank is not None:
            config.rank = int(rank)

        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            config.local_rank = int(local_rank)

        # Detect master address and port
        master_addr = os.environ.get("MASTER_ADDR")
        if master_addr is not None:
            config.master_addr = master_addr

        master_port = os.environ.get("MASTER_PORT")
        if master_port is not None:
            config.master_port = int(master_port)

        # Detect GPU settings
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            config.device_ids = [int(d) for d in cuda_visible.split(",") if d.strip()]

        # Log detected settings
        if config.enabled:
            logger.info(
                f"DDP environment detected: "
                f"rank={config.rank}/{config.world_size}, "
                f"local_rank={config.local_rank}, "
                f"master={config.master_addr}:{config.master_port}"
            )
        else:
            logger.debug("No DDP environment detected, using single-process mode")

        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DDPConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            DDPConfig instance.
        """
        # Handle nested ddp key
        if "ddp" in data:
            data = data["ddp"]

        return cls(
            enabled=data.get("enabled", False),
            world_size=data.get("world_size", 1),
            rank=data.get("rank", 0),
            local_rank=data.get("local_rank", 0),
            master_addr=data.get("master_addr", "localhost"),
            master_port=data.get("master_port", 29500),
            backend=data.get("backend", DDPBackend.AUTO.value),
            init_method=data.get("init_method", "env://"),
            timeout_seconds=data.get("timeout_seconds", 1800),
            use_cuda=data.get("use_cuda", True),
            device_ids=data.get("device_ids", []),
            find_unused_parameters=data.get("find_unused_parameters", False),
            broadcast_buffers=data.get("broadcast_buffers", True),
            gradient_as_bucket_view=data.get("gradient_as_bucket_view", True),
            bucket_cap_mb=data.get("bucket_cap_mb", 25),
            use_amp=data.get("use_amp", False),
            amp_dtype=data.get("amp_dtype", "float16"),
            elastic_training=data.get("elastic_training", False),
            min_world_size=data.get("min_world_size", 1),
            max_restarts=data.get("max_restarts", 3),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "enabled": self.enabled,
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "backend": self.backend,
            "init_method": self.init_method,
            "timeout_seconds": self.timeout_seconds,
            "use_cuda": self.use_cuda,
            "device_ids": self.device_ids,
            "find_unused_parameters": self.find_unused_parameters,
            "broadcast_buffers": self.broadcast_buffers,
            "gradient_as_bucket_view": self.gradient_as_bucket_view,
            "bucket_cap_mb": self.bucket_cap_mb,
            "use_amp": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "elastic_training": self.elastic_training,
            "min_world_size": self.min_world_size,
            "max_restarts": self.max_restarts,
        }

    def get_device(self) -> str:
        """Get the device string for this process.

        Returns:
            Device string like 'cuda:0' or 'cpu'.
        """
        if not self.use_cuda:
            return "cpu"

        if self.device_ids:
            # Use local rank to index into visible devices
            device_idx = min(self.local_rank, len(self.device_ids) - 1)
            return f"cuda:{self.device_ids[device_idx]}"
        else:
            # Default to local rank
            return f"cuda:{self.local_rank}"

    def get_backend(self) -> str:
        """Get the communication backend to use.

        Returns:
            Backend string ('nccl', 'gloo', etc).
        """
        if self.backend != DDPBackend.AUTO.value:
            return self.backend

        # Auto-detect: NCCL for CUDA, Gloo otherwise
        if self.use_cuda:
            return DDPBackend.NCCL.value
        return DDPBackend.GLOO.value

    def is_main_process(self) -> bool:
        """Check if this is the main (rank 0) process.

        Returns:
            True if rank 0 or DDP not enabled.
        """
        return self.rank == 0 or not self.enabled

    def is_local_main(self) -> bool:
        """Check if this is the local main (local_rank 0) process.

        Returns:
            True if local_rank 0 or DDP not enabled.
        """
        return self.local_rank == 0 or not self.enabled

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation issues (empty if valid).
        """
        issues = []

        if self.enabled:
            if self.world_size < 2:
                issues.append("world_size must be >= 2 when DDP is enabled")

            if self.rank < 0 or self.rank >= self.world_size:
                issues.append(f"rank {self.rank} out of range [0, {self.world_size})")

            if self.local_rank < 0:
                issues.append("local_rank must be >= 0")

            if self.master_port < 1024 or self.master_port > 65535:
                issues.append(f"master_port {self.master_port} out of valid range")

        if self.timeout_seconds < 60:
            issues.append("timeout_seconds should be >= 60 for reliability")

        if self.bucket_cap_mb < 1:
            issues.append("bucket_cap_mb must be >= 1")

        return issues

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Returns:
            True if no validation issues.
        """
        return len(self.validate()) == 0


def detect_ddp_environment() -> DDPConfig:
    """Detect DDP environment and return configuration.

    This is a convenience function that:
    1. Checks environment variables
    2. Validates the detected configuration
    3. Returns a ready-to-use config

    Returns:
        DDPConfig with detected settings.
    """
    config = DDPConfig.from_env()

    issues = config.validate()
    if issues:
        logger.warning(f"DDP configuration issues: {issues}")

    return config


def setup_ddp_environment(config: DDPConfig) -> None:
    """Set up environment variables for DDP.

    Call this before initializing torch.distributed.

    Args:
        config: DDP configuration.
    """
    if not config.enabled:
        return

    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    os.environ["WORLD_SIZE"] = str(config.world_size)
    os.environ["RANK"] = str(config.rank)
    os.environ["LOCAL_RANK"] = str(config.local_rank)

    logger.info(
        f"Set DDP environment: "
        f"MASTER_ADDR={config.master_addr}, "
        f"MASTER_PORT={config.master_port}, "
        f"WORLD_SIZE={config.world_size}, "
        f"RANK={config.rank}"
    )
