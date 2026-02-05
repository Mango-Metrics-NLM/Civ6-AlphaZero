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
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import nn

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=nn.Module)


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
) -> dict[str, Any]:
    """Load a checkpoint from disk.

    Uses safe loading by default (weights_only=True) to prevent
    arbitrary code execution.

    Args:
        path: Path to the checkpoint file.
        device: Device to load tensors to ("cpu", "cuda", etc.).
        weights_only: If True, only load tensor weights (safer).

    Returns:
        Checkpoint dictionary with "config" and "state_dict" keys.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=weights_only,
        )
    except Exception:
        # Fall back to unsafe loading for legacy checkpoints
        logger.warning("Safe loading failed for %s, falling back to unsafe load", path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)

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


def get_checkpoint_info(path: str | Path) -> dict[str, Any]:
    """Get metadata about a checkpoint without fully loading it.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Dictionary with checkpoint metadata (network_type, config keys, etc.)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load with map_location to avoid loading to GPU
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    info = {
        "path": str(path),
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "keys": list(checkpoint.keys()),
    }

    if "network_type" in checkpoint:
        info["network_type"] = checkpoint["network_type"]

    if "config" in checkpoint:
        info["config_keys"] = list(checkpoint["config"].keys())

    if "state_dict" in checkpoint:
        info["num_parameters"] = sum(p.numel() for p in checkpoint["state_dict"].values())

    return info
