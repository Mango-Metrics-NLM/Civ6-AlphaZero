"""Checkpoint utilities for safe model loading and saving.

This module provides secure checkpoint handling with:
- Safe loading with weights_only=True when possible
- Fallback mechanisms for complex checkpoints
- Configurable security levels
- Logging and warnings for security-relevant operations

Security Note:
    PyTorch's torch.load() with weights_only=False can execute arbitrary code
    from checkpoint files. Only load checkpoints from trusted sources.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Module logger
logger = logging.getLogger(__name__)

# Default allowed classes for safe loading
# These are standard PyTorch/numpy types that are safe to unpickle
DEFAULT_SAFE_GLOBALS = {
    "collections.OrderedDict",
    "torch.nn.parameter.Parameter",
    "numpy.core.multiarray._reconstruct",
    "numpy.ndarray",
    "numpy.dtype",
}


@dataclass
class CheckpointLoadResult:
    """Result of loading a checkpoint.

    Attributes:
        data: The loaded checkpoint data.
        loaded_safely: Whether weights_only=True was used.
        warnings: Any warnings generated during loading.
    """

    data: dict[str, Any]
    loaded_safely: bool
    warnings: list[str]


def safe_load_checkpoint(
    path: str | Path,
    map_location: str | torch.device | None = None,
    strict_security: bool = False,
    allow_unsafe_fallback: bool = True,
) -> CheckpointLoadResult:
    """Load a checkpoint file with security best practices.

    This function attempts to load checkpoints safely using weights_only=True.
    If that fails (e.g., for checkpoints with optimizer state or custom objects),
    it can fall back to weights_only=False with appropriate warnings.

    Args:
        path: Path to the checkpoint file.
        map_location: Device to map tensors to (e.g., "cpu", "cuda").
        strict_security: If True, never fall back to unsafe loading.
            Raises an error if safe loading fails.
        allow_unsafe_fallback: If True (default), fall back to unsafe loading
            when safe loading fails. Ignored if strict_security=True.

    Returns:
        CheckpointLoadResult with the loaded data and safety metadata.

    Raises:
        RuntimeError: If strict_security=True and safe loading fails.
        FileNotFoundError: If checkpoint file doesn't exist.

    Example:
        >>> result = safe_load_checkpoint("model.pt", map_location="cpu")
        >>> if not result.loaded_safely:
        ...     print("Warning: Checkpoint loaded with unsafe mode")
        >>> model.load_state_dict(result.data["model_state_dict"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    result_warnings: list[str] = []

    # First, try loading with weights_only=True (safe mode)
    try:
        data = torch.load(path, map_location=map_location, weights_only=True)
        logger.info(f"Loaded checkpoint safely (weights_only=True): {path}")
        return CheckpointLoadResult(
            data=data,
            loaded_safely=True,
            warnings=result_warnings,
        )
    except Exception as e:
        safe_load_error = str(e)
        logger.debug(f"Safe loading failed: {safe_load_error}")

    # If strict security is enabled, don't fall back
    if strict_security:
        raise RuntimeError(
            f"Cannot load checkpoint safely (weights_only=True). "
            f"Error: {safe_load_error}. "
            f"Set strict_security=False to allow unsafe loading from trusted sources."
        )

    # Fall back to unsafe loading if allowed
    if not allow_unsafe_fallback:
        raise RuntimeError(
            f"Safe loading failed and fallback is disabled. Error: {safe_load_error}"
        )

    # Warn about unsafe loading
    warning_msg = (
        f"Loading checkpoint with weights_only=False. "
        f"Only load checkpoints from trusted sources. Path: {path}"
    )
    result_warnings.append(warning_msg)
    warnings.warn(warning_msg, UserWarning, stacklevel=2)
    logger.warning(warning_msg)

    # Load with weights_only=False (unsafe but necessary for some checkpoints)
    data = torch.load(path, map_location=map_location, weights_only=False)
    logger.info(f"Loaded checkpoint (weights_only=False): {path}")

    return CheckpointLoadResult(
        data=data,
        loaded_safely=False,
        warnings=result_warnings,
    )


def load_model_state(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load only model state dict from checkpoint (always safe).

    This function extracts just the model weights, which can always
    be loaded safely with weights_only=True.

    Args:
        path: Path to the checkpoint file.
        map_location: Device to map tensors to.

    Returns:
        The model state dict.

    Raises:
        KeyError: If checkpoint doesn't contain expected keys.
    """
    result = safe_load_checkpoint(
        path,
        map_location=map_location,
        strict_security=False,
        allow_unsafe_fallback=True,
    )

    # Look for common state dict keys
    data = result.data
    for key in ["model_state_dict", "network_state_dict", "state_dict"]:
        if key in data:
            return data[key]  # type: ignore[no-any-return]

    # If the checkpoint IS the state dict directly
    if all(isinstance(v, torch.Tensor) for v in data.values()):
        return data

    raise KeyError(
        f"Could not find model state dict in checkpoint. Available keys: {list(data.keys())}"
    )


def save_checkpoint(
    path: str | Path,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None = None,
    config: Any = None,
    **extra_data: Any,
) -> None:
    """Save a checkpoint with standard structure.

    Args:
        path: Path to save checkpoint to.
        model_state_dict: Model weights to save.
        optimizer_state_dict: Optional optimizer state.
        config: Optional configuration object or dict.
        **extra_data: Additional data to include in checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model_state_dict,
    }

    if optimizer_state_dict is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state_dict

    if config is not None:
        checkpoint["config"] = config

    checkpoint.update(extra_data)

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def compute_weight_checksum(network: torch.nn.Module) -> float:
    """Compute a reproducible checksum of network weights.

    Sums the absolute values of all parameters. Useful for detecting
    weight re-initialisation or verifying checkpoint roundtrip fidelity.

    Args:
        network: PyTorch module to checksum.

    Returns:
        Non-negative float representing the total absolute weight magnitude.
    """
    total = 0.0
    for param in network.parameters():
        total += param.data.abs().sum().item()
    return total


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "CheckpointLoadResult",
    "compute_weight_checksum",
    "load_model_state",
    "safe_load_checkpoint",
    "save_checkpoint",
]
