"""Mock implementations for distributed training testing.

Provides mocks for:
- PyTorch DDP process group
- Distributed environment detection
- Parameter server operations
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class MockProcessGroupConfig:
    """Configuration for mock process group."""

    world_size: int = 2
    rank: int = 0
    backend: str = "gloo"
    init_method: str = "env://"


class MockProcessGroup:
    """Mock PyTorch distributed process group for testing.

    Simulates distributed operations without actual network communication.
    """

    _instance: MockProcessGroup | None = None
    _initialized: bool = False

    def __init__(self, config: MockProcessGroupConfig | None = None):
        """Initialize mock process group.

        Args:
            config: Process group configuration.
        """
        self.config = config or MockProcessGroupConfig()
        self._barriers: dict[str, int] = {}
        self._broadcast_data: dict[str, torch.Tensor] = {}
        self._reduce_data: dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> MockProcessGroup:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = MockProcessGroup()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset mock state."""
        cls._instance = None
        cls._initialized = False

    def init_process_group(
        self,
        backend: str = "gloo",
        init_method: str = "env://",
        world_size: int = 2,
        rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """Mock init_process_group."""
        self.config = MockProcessGroupConfig(
            world_size=world_size,
            rank=rank,
            backend=backend,
            init_method=init_method,
        )
        MockProcessGroup._initialized = True

    def is_initialized(self) -> bool:
        """Check if process group is initialized."""
        return MockProcessGroup._initialized

    def destroy_process_group(self) -> None:
        """Mock destroy_process_group."""
        MockProcessGroup._initialized = False
        self._barriers.clear()
        self._broadcast_data.clear()
        self._reduce_data.clear()

    def get_rank(self) -> int:
        """Get current rank."""
        return self.config.rank

    def get_world_size(self) -> int:
        """Get world size."""
        return self.config.world_size

    def barrier(self, **kwargs: Any) -> None:
        """Mock barrier - no-op in single process."""
        pass

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        **kwargs: Any,
    ) -> None:
        """Mock broadcast - stores tensor for verification."""
        with self._lock:
            key = f"broadcast_{id(tensor)}"
            self._broadcast_data[key] = tensor.clone()

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: Any = None,
        **kwargs: Any,
    ) -> None:
        """Mock all_reduce - simulates averaging across workers."""
        # Simulate averaging by dividing by world_size
        # (in real DDP, all workers contribute, then we average)
        with self._lock:
            key = f"reduce_{id(tensor)}"
            self._reduce_data[key] = tensor.clone()

    def all_gather(
        self,
        tensor_list: list[torch.Tensor],
        tensor: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Mock all_gather - fills list with copies of tensor."""
        for i in range(len(tensor_list)):
            tensor_list[i] = tensor.clone()


class MockDistributedEnvironment:
    """Mock distributed environment for testing.

    Sets up environment variables and provides utilities for testing
    distributed code without actual multi-process setup.
    """

    def __init__(
        self,
        world_size: int = 2,
        rank: int = 0,
        local_rank: int = 0,
        master_addr: str = "localhost",
        master_port: int = 29500,
    ):
        """Initialize mock environment.

        Args:
            world_size: Total number of processes.
            rank: Global rank of this process.
            local_rank: Local rank on this node.
            master_addr: Master node address.
            master_port: Master node port.
        """
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.master_addr = master_addr
        self.master_port = master_port

        self._original_env: dict[str, str | None] = {}
        self._process_group = MockProcessGroup()

    def __enter__(self) -> MockDistributedEnvironment:
        """Enter context - set environment variables."""
        import os

        # Save original values
        env_vars = [
            "WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "MASTER_ADDR",
            "MASTER_PORT",
        ]
        for var in env_vars:
            self._original_env[var] = os.environ.get(var)

        # Set mock values
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context - restore environment variables."""
        import os

        for var, value in self._original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

        # Reset process group
        self._process_group.reset()


class MockDDP(nn.Module):
    """Mock DistributedDataParallel wrapper.

    Wraps a module without actual distributed synchronization.
    """

    def __init__(
        self,
        module: nn.Module,
        device_ids: list[int] | None = None,
        output_device: int | None = None,
        find_unused_parameters: bool = False,
        **kwargs: Any,
    ):
        """Initialize mock DDP.

        Args:
            module: Module to wrap.
            device_ids: GPU device IDs.
            output_device: Output device.
            find_unused_parameters: Whether to find unused parameters.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__()
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.find_unused_parameters = find_unused_parameters

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass - delegates to wrapped module."""
        return self.module(*args, **kwargs)

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Get state dict from wrapped module."""
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: dict[str, Any], **kwargs: Any) -> None:  # type: ignore[override]
        """Load state dict into wrapped module."""
        self.module.load_state_dict(state_dict, **kwargs)

    def parameters(self, recurse: bool = True):
        """Get parameters from wrapped module."""
        return self.module.parameters(recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
        """Get named parameters from wrapped module."""
        return self.module.named_parameters(prefix, recurse)


@dataclass
class MockTensorboardWriter:
    """Mock TensorBoard writer for testing."""

    log_dir: str = "logs/mock"
    _scalars: dict[str, list[tuple[float, int]]] = field(default_factory=dict)
    _histograms: dict[str, list[tuple[Any, int]]] = field(default_factory=dict)

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
    ) -> None:
        """Mock add_scalar."""
        if tag not in self._scalars:
            self._scalars[tag] = []
        self._scalars[tag].append((scalar_value, global_step or 0))

    def add_histogram(
        self,
        tag: str,
        values: Any,
        global_step: int | None = None,
    ) -> None:
        """Mock add_histogram."""
        if tag not in self._histograms:
            self._histograms[tag] = []
        self._histograms[tag].append((values, global_step or 0))

    def flush(self) -> None:
        """Mock flush."""
        pass

    def close(self) -> None:
        """Mock close."""
        pass


class MockGradScaler:
    """Mock AMP GradScaler for testing."""

    def __init__(self, enabled: bool = True, **kwargs: Any):
        """Initialize mock scaler.

        Args:
            enabled: Whether scaling is enabled.
            **kwargs: Additional arguments (ignored).
        """
        self._enabled = enabled
        self._scale = 1.0
        self._growth_factor = 2.0
        self._backoff_factor = 0.5

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """Mock scale - returns loss unchanged."""
        return loss

    def unscale_(self, optimizer: Any) -> None:
        """Mock unscale_ - no-op."""
        pass

    def step(self, optimizer: Any) -> None:
        """Mock step - calls optimizer.step()."""
        optimizer.step()

    def update(self) -> None:
        """Mock update - no-op."""
        pass

    def state_dict(self) -> dict[str, Any]:
        """Get mock state dict."""
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load mock state dict."""
        self._scale = state_dict.get("scale", 1.0)


def create_mock_network(
    input_channels: int = 5,
    board_size: int = 16,
    action_space_size: int = 100,
) -> nn.Module:
    """Create a simple mock network for testing.

    Args:
        input_channels: Number of input channels.
        board_size: Size of the board.
        action_space_size: Size of action space.

    Returns:
        Simple mock network.
    """

    class MockNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(input_channels, 32, 3, padding=1)
            self.fc_policy = nn.Linear(32 * board_size * board_size, action_space_size)
            self.fc_value = nn.Linear(32 * board_size * board_size, 1)

        def forward(
            self,
            spatial: torch.Tensor,
            vector: torch.Tensor,
            action_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = torch.relu(self.conv(spatial))
            x = x.flatten(1)
            policy = self.fc_policy(x)
            value = torch.tanh(self.fc_value(x))
            return policy, value

        def predict(
            self,
            spatial: torch.Tensor,
            vector: torch.Tensor,
            action_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            policy, value = self.forward(spatial, vector, action_mask)
            # Apply softmax for policy
            policy = torch.softmax(policy, dim=-1)
            return policy, value

    return MockNetwork()


def patch_distributed_for_testing():
    """Patch torch.distributed for testing.

    Returns a context manager that patches distributed functions.
    """
    from unittest.mock import patch

    mock_pg = MockProcessGroup()

    patches = [
        patch("torch.distributed.init_process_group", mock_pg.init_process_group),
        patch("torch.distributed.is_initialized", mock_pg.is_initialized),
        patch("torch.distributed.destroy_process_group", mock_pg.destroy_process_group),
        patch("torch.distributed.get_rank", mock_pg.get_rank),
        patch("torch.distributed.get_world_size", mock_pg.get_world_size),
        patch("torch.distributed.barrier", mock_pg.barrier),
        patch("torch.distributed.broadcast", mock_pg.broadcast),
        patch("torch.distributed.all_reduce", mock_pg.all_reduce),
        patch("torch.distributed.all_gather", mock_pg.all_gather),
    ]

    class PatchContext:
        def __enter__(self):
            for p in patches:
                p.__enter__()
            return mock_pg

        def __exit__(self, *args):
            for p in patches:
                p.__exit__(*args)
            mock_pg.reset()

    return PatchContext()
