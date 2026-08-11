"""Mock distributed training infrastructure for testing.

Provides reusable mocks for torch.distributed operations,
enabling unit and integration testing without actual distributed setup.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class MockProcessGroupState:
    """Tracks state of a simulated distributed process group.

    Attributes:
        world_size: Total number of simulated processes.
        rank: Rank of the current simulated process.
        backend: Simulated backend name.
        is_initialized: Whether the process group is initialized.
        barrier_count: Number of barrier calls.
        all_reduce_count: Number of all_reduce calls.
        broadcast_count: Number of broadcast calls.
        gather_count: Number of gather calls.
        reduce_count: Number of reduce calls.
    """

    world_size: int = 1
    rank: int = 0
    backend: str = "gloo"
    is_initialized: bool = False
    barrier_count: int = 0
    all_reduce_count: int = 0
    broadcast_count: int = 0
    gather_count: int = 0
    reduce_count: int = 0
    stored_tensors: dict[str, Any] = field(default_factory=dict)


class MockDistributedBackend:
    """Mock implementation of torch.distributed operations.

    Simulates distributed training operations without requiring actual
    multi-process setup. Tracks operation counts for verification.

    Example:
        >>> backend = MockDistributedBackend(world_size=4, rank=0)
        >>> with backend.patch():
        ...     dist.init_process_group("gloo")
        ...     dist.barrier()
        >>> assert backend.state.barrier_count == 1
    """

    def __init__(
        self,
        world_size: int = 1,
        rank: int = 0,
        backend: str = "gloo",
    ) -> None:
        """Initialize mock distributed backend.

        Args:
            world_size: Simulated world size.
            rank: Simulated rank of this process.
            backend: Simulated backend name.
        """
        self.state = MockProcessGroupState(
            world_size=world_size,
            rank=rank,
            backend=backend,
        )
        self._patches: list[Any] = []

    def init_process_group(
        self,
        backend: str = "gloo",
        init_method: str | None = None,
        world_size: int = -1,
        rank: int = -1,
        **kwargs: Any,
    ) -> None:
        """Mock process group initialization."""
        if world_size > 0:
            self.state.world_size = world_size
        if rank >= 0:
            self.state.rank = rank
        self.state.backend = backend
        self.state.is_initialized = True
        logger.debug(
            "Mock init_process_group: backend=%s, world_size=%d, rank=%d",
            backend,
            self.state.world_size,
            self.state.rank,
        )

    def destroy_process_group(self) -> None:
        """Mock process group destruction."""
        self.state.is_initialized = False
        logger.debug("Mock destroy_process_group called")

    def is_initialized(self) -> bool:
        """Check if process group is initialized."""
        return self.state.is_initialized

    def get_rank(self) -> int:
        """Get current rank."""
        return self.state.rank

    def get_world_size(self) -> int:
        """Get world size."""
        return self.state.world_size

    def barrier(self, **kwargs: Any) -> None:
        """Mock barrier synchronization."""
        self.state.barrier_count += 1
        logger.debug("Mock barrier called (count=%d)", self.state.barrier_count)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: Any = None,
        **kwargs: Any,
    ) -> None:
        """Mock all-reduce operation.

        In a real scenario, this would sum tensors across all ranks.
        We simulate by multiplying by world_size (as if summing identical tensors).
        """
        self.state.all_reduce_count += 1
        # Simulate sum reduction - each rank contributes the same value
        tensor.mul_(self.state.world_size)
        logger.debug(
            "Mock all_reduce called (count=%d)",
            self.state.all_reduce_count,
        )

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        **kwargs: Any,
    ) -> None:
        """Mock broadcast operation."""
        self.state.broadcast_count += 1
        logger.debug(
            "Mock broadcast called from src=%d (count=%d)",
            src,
            self.state.broadcast_count,
        )

    def reduce(
        self,
        tensor: torch.Tensor,
        dst: int = 0,
        op: Any = None,
        **kwargs: Any,
    ) -> None:
        """Mock reduce operation."""
        self.state.reduce_count += 1
        # Simulate sum reduction on destination
        if self.state.rank == dst:
            tensor.mul_(self.state.world_size)
        logger.debug(
            "Mock reduce called to dst=%d (count=%d)",
            dst,
            self.state.reduce_count,
        )

    def gather(
        self,
        tensor: torch.Tensor,
        gather_list: list[torch.Tensor] | None = None,
        dst: int = 0,
        **kwargs: Any,
    ) -> None:
        """Mock gather operation."""
        self.state.gather_count += 1
        if self.state.rank == dst and gather_list is not None:
            # Fill gather_list with copies of the input tensor
            for _i, t in enumerate(gather_list):
                t.copy_(tensor)
        logger.debug(
            "Mock gather called to dst=%d (count=%d)",
            dst,
            self.state.gather_count,
        )

    def reset_counts(self) -> None:
        """Reset all operation counters."""
        self.state.barrier_count = 0
        self.state.all_reduce_count = 0
        self.state.broadcast_count = 0
        self.state.gather_count = 0
        self.state.reduce_count = 0
        logger.debug("Mock operation counts reset")

    @contextmanager
    def patch(self) -> Generator[MockDistributedBackend, None, None]:
        """Context manager to patch torch.distributed with this mock.

        Yields:
            Self, with torch.distributed patched.
        """
        try:
            self._patches = [
                patch("torch.distributed.init_process_group", self.init_process_group),
                patch("torch.distributed.destroy_process_group", self.destroy_process_group),
                patch("torch.distributed.is_initialized", self.is_initialized),
                patch("torch.distributed.get_rank", self.get_rank),
                patch("torch.distributed.get_world_size", self.get_world_size),
                patch("torch.distributed.barrier", self.barrier),
                patch("torch.distributed.all_reduce", self.all_reduce),
                patch("torch.distributed.broadcast", self.broadcast),
                patch("torch.distributed.reduce", self.reduce),
                patch("torch.distributed.gather", self.gather),
            ]
            for p in self._patches:
                p.start()
            yield self
        finally:
            for p in self._patches:
                p.stop()


class MockDDPWrapper:
    """Mock for torch.nn.parallel.DistributedDataParallel.

    Wraps a module without actual distributed parallelism,
    useful for testing DDP-dependent code paths.

    Attributes:
        module: The wrapped module.
        device_ids: Simulated device IDs.
    """

    def __init__(
        self,
        module: Any,
        device_ids: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize mock DDP wrapper.

        Args:
            module: The module to wrap.
            device_ids: Simulated device IDs.
            **kwargs: Ignored DDP parameters.
        """
        self.module = module
        self.device_ids = device_ids or []
        self._forward_hooks: dict[str, Callable[..., Any]] = {}
        logger.debug("MockDDPWrapper created for module: %s", type(module).__name__)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the wrapped module."""
        return self.module(*args, **kwargs)

    def train(self, mode: bool = True) -> MockDDPWrapper:
        """Set training mode."""
        self.module.train(mode)
        return self

    def eval(self) -> MockDDPWrapper:
        """Set evaluation mode."""
        self.module.eval()
        return self

    def parameters(self) -> Any:
        """Get module parameters."""
        return self.module.parameters()

    def state_dict(self) -> dict[str, Any]:
        """Get module state dict."""
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
        """Load module state dict."""
        return self.module.load_state_dict(state_dict, strict)


@contextmanager
def mock_ddp_context(
    world_size: int = 2,
    rank: int = 0,
    ddp_module_path: str | None = None,
    mock_cuda: bool = True,
) -> Generator[MockDistributedBackend, None, None]:
    """Context manager for mocking DDP environment.

    Patches both torch.distributed and DDP wrapper.  Also patches
    ``torch.cuda.is_available`` to return ``False`` and
    ``alpha_zero.distributed.ddp_config.DDPConfig.get_device`` to return
    ``"cpu"``, so the learner runs on CPU even when the config would
    otherwise select a CUDA device.  This makes tests work in CPU-only
    environments without an NVIDIA driver.

    Args:
        world_size: Simulated world size.
        rank: Simulated rank.
        ddp_module_path: Optional path to patch DDP at (e.g.,
            'alpha_zero.distributed.ddp_learner.DDP'). If not provided,
            patches 'torch.nn.parallel.DistributedDataParallel'.
        mock_cuda: Whether to mock CUDA to be unavailable.

    Yields:
        MockDistributedBackend instance with patches applied.

    Example:
        >>> with mock_ddp_context(world_size=4, rank=0) as mock_dist:
        ...     learner = DDPLearner(config, storage, buffer)
        ...     learner.initialize()
        ...     assert mock_dist.state.is_initialized
    """
    backend = MockDistributedBackend(world_size=world_size, rank=rank)

    # Determine where to patch DDP
    ddp_patch_target = ddp_module_path or "torch.nn.parallel.DistributedDataParallel"

    with backend.patch():
        with patch(ddp_patch_target, MockDDPWrapper):
            if mock_cuda:
                with patch("torch.cuda.is_available", return_value=False):
                    with patch(
                        "alpha_zero.distributed.ddp_config.DDPConfig.get_device",
                        return_value="cpu",
                    ):
                        yield backend
            else:
                yield backend
