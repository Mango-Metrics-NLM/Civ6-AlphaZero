"""Checkpoint utilities for network serialization.

This module provides unified checkpoint saving and loading functions
to eliminate duplication across network implementations.

Example:
    >>> from alpha_zero.core.checkpoint_utils import save_checkpoint, load_checkpoint
    >>>
    >>> # Save a network
    >>> save_checkpoint(network, network.config, "checkpoint.pt")
    >>>
    >>> # Load a network
    >>> checkpoint = load_checkpoint("checkpoint.pt", device="cuda")
    >>> network = NetworkClass.from_checkpoint(checkpoint)

Security:
    Every load goes through :func:`alpha_zero.utils.safe_load.safe_torch_load`
    (re-exported here), which calls ``torch.load(..., weights_only=True)`` and
    **never downgrades on its own**. A checkpoint that the tensor-only
    unpickler rejects raises :class:`UnsafeCheckpointError` naming the file and
    the two ways to opt in: the ``allow_unsafe=True`` keyword, or the
    process-wide environment variable ``CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1`` (read
    at call time). Either opt-in logs a WARNING and loads with
    ``weights_only=False``, which can execute arbitrary code from the file --
    use it only for checkpoints you produced. The helper lives in a leaf module
    so ``alpha_zero.utils.checkpoint`` can share it without a ``utils -> core``
    import cycle.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import nn

from alpha_zero.utils.safe_load import (
    UNSAFE_CHECKPOINT_ENV_VAR,
    UnsafeCheckpointError,
    safe_torch_load,
    unsafe_checkpoint_loading_allowed,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=nn.Module)

__all__ = [
    "UNSAFE_CHECKPOINT_ENV_VAR",
    "VALID_REDUCTIONS",
    "UnsafeCheckpointError",
    "get_checkpoint_info",
    "load_checkpoint",
    "load_state_dict_into_network",
    "safe_torch_load",
    "save_checkpoint",
    "serialize_config",
    "unsafe_checkpoint_loading_allowed",
    "validate_reduction",
]


# =============================================================================
# Type Constants
# =============================================================================

# Valid reduction types for loss functions
VALID_REDUCTIONS = frozenset({"mean", "sum", "none"})


# =============================================================================
# Checkpoint Functions
# =============================================================================


def serialize_config(config: Any) -> dict[str, Any]:
    """Serialize a config dataclass to a dictionary.

    Handles:
    - Numpy scalars (via .item())
    - Enum values (via .value)
    - Nested dataclasses (recursively)

    Args:
        config: A dataclass configuration object.

    Returns:
        Dictionary representation suitable for torch.save().

    Raises:
        TypeError: If config is not a dataclass.
    """
    if not is_dataclass(config):
        raise TypeError(f"config must be a dataclass, got {type(config).__name__}")
    # Ensure config is a dataclass instance, not a class
    if isinstance(config, type):
        raise TypeError("config must be a dataclass instance, not a class")

    config_dict: dict[str, Any] = asdict(config)
    return dict(_convert_to_primitives(config_dict))


def _convert_to_primitives(data: Any) -> Any:
    """Recursively convert data to JSON-safe primitive types.

    Args:
        data: Any data structure to convert.

    Returns:
        Data with all values converted to primitives.
    """
    if isinstance(data, dict):
        return {k: _convert_to_primitives(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_convert_to_primitives(v) for v in data]
    elif isinstance(data, Enum):
        return data.value
    elif hasattr(data, "item"):  # Handle numpy scalars
        return data.item()
    elif hasattr(data, "tolist"):  # Handle numpy arrays
        return data.tolist()
    return data


def save_checkpoint(
    network: nn.Module,
    config: Any,
    path: str | Path,
    *,
    network_type: str | None = None,
    extra_data: dict[str, Any] | None = None,
) -> None:
    """Save a network checkpoint to disk.

    Args:
        network: The neural network module to save.
        config: The configuration dataclass for the network.
        path: Path to save the checkpoint to.
        network_type: Optional string identifier for the network type.
        extra_data: Optional additional data to include in checkpoint.

    Example:
        >>> save_checkpoint(
        ...     network=model,
        ...     config=model.config,
        ...     path="checkpoint.pt",
        ...     network_type="microciv_v1",
        ... )
    """
    checkpoint: dict[str, Any] = {
        "config": serialize_config(config),
        "state_dict": network.state_dict(),
    }

    if network_type is not None:
        checkpoint["network_type"] = network_type

    if extra_data is not None:
        checkpoint.update(extra_data)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)
    logger.debug("Saved checkpoint to %s", path)


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    *,
    weights_only: bool = True,
    allow_unsafe: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint from disk.

    Safe by default (``weights_only=True``). There is **no** automatic downgrade
    to ``weights_only=False``: a checkpoint the safe unpickler rejects raises
    :class:`UnsafeCheckpointError` unless the caller opts in.

    Args:
        path: Path to the checkpoint file.
        device: Device to load tensors to ("cpu", "cuda", etc.).
        weights_only: Deprecated. ``False`` is treated as ``allow_unsafe=True``
            (kept so existing call sites keep working); prefer ``allow_unsafe``.
        allow_unsafe: Authorise a ``weights_only=False`` load if the safe load
            is rejected. ``CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1`` in the environment
            has the same effect for the whole process.

    Returns:
        Checkpoint dictionary with "config" and "state_dict" keys.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        UnsafeCheckpointError: If the safe load is rejected and unsafe loading
            was not authorised.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if not weights_only:
        logger.warning(
            "load_checkpoint(weights_only=False) is deprecated; treating it as "
            "allow_unsafe=True for %s",
            path,
        )
        allow_unsafe = True

    checkpoint = safe_torch_load(path, map_location=device, allow_unsafe=allow_unsafe)

    logger.debug("Loaded checkpoint from %s", path)
    return dict(checkpoint)


def load_state_dict_into_network(
    network: T,
    checkpoint: dict[str, Any],
    *,
    strict: bool = True,
) -> T:
    """Load state dict from checkpoint into network.

    Args:
        network: Network to load weights into.
        checkpoint: Checkpoint dictionary with "state_dict" key.
        strict: If True, strictly match keys in state_dict.

    Returns:
        The network with loaded weights.
    """
    state_dict = checkpoint.get("state_dict", checkpoint)
    network.load_state_dict(state_dict, strict=strict)
    return network


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_reduction(reduction: str) -> None:
    """Validate that reduction is a valid loss reduction type.

    Args:
        reduction: The reduction type to validate.

    Raises:
        ValueError: If reduction is not "mean", "sum", or "none".

    Example:
        >>> validate_reduction("mean")  # OK
        >>> validate_reduction("invalid")  # Raises ValueError
    """
    if reduction not in VALID_REDUCTIONS:
        raise ValueError(
            f"Unknown reduction: {reduction}. Valid options are: {sorted(VALID_REDUCTIONS)}"
        )


# =============================================================================
# Checkpoint Info
# =============================================================================


def get_checkpoint_info(path: str | Path, *, allow_unsafe: bool = False) -> dict[str, Any]:
    """Get metadata about a checkpoint without fully loading it.

    Args:
        path: Path to the checkpoint file.
        allow_unsafe: Authorise ``weights_only=False`` if the safe load is
            rejected (see :func:`safe_torch_load`). Inspecting a file is not a
            reason to run code from it, so the default is the safe loader.

    Returns:
        Dictionary with checkpoint metadata (network_type, config keys, etc.)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        UnsafeCheckpointError: If the safe load is rejected without opt-in.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # map_location="cpu" so inspecting never allocates on a GPU.
    checkpoint = safe_torch_load(path, map_location="cpu", allow_unsafe=allow_unsafe)

    info = {
        "path": str(path),
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "keys": list(checkpoint.keys()),
    }

    if "network_type" in checkpoint:
        info["network_type"] = checkpoint["network_type"]

    # A config saved as a dataclass *instance* (legacy, needs allow_unsafe)
    # has no keys; report its type instead of crashing on the metadata read.
    config = checkpoint.get("config")
    if isinstance(config, dict):
        info["config_keys"] = list(config.keys())
    elif config is not None:
        info["config_type"] = type(config).__name__

    if "state_dict" in checkpoint:
        info["num_parameters"] = sum(p.numel() for p in checkpoint["state_dict"].values())

    return info
