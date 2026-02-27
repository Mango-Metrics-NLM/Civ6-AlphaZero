"""Framework abstraction layer for multi-backend ML support.

Provides framework detection, array conversion utilities, and availability
checks that decouple the rest of the codebase from any specific ML framework.

Supported frameworks:
- PyTorch (default, always available)
- JAX (optional, install with: pip install -e '.[jax]')

Usage:
    from alpha_zero.core.framework import (
        FrameworkType,
        get_framework,
        to_numpy,
        from_numpy,
        framework_available,
    )

    fw = get_framework()  # Reads CIV6AZ_FRAMEWORK env var
    arr = to_numpy(tensor)  # Works with torch.Tensor, jax.Array, or np.ndarray
"""

from __future__ import annotations

import enum
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class FrameworkType(enum.Enum):
    """Supported ML frameworks."""

    PYTORCH = "pytorch"
    JAX = "jax"


def get_framework(override: str | None = None) -> FrameworkType:
    """Determine the active ML framework.

    Resolution order:
    1. Explicit override parameter
    2. CIV6AZ_FRAMEWORK environment variable
    3. Default: pytorch

    Args:
        override: Explicit framework name ("pytorch" or "jax").

    Returns:
        The resolved FrameworkType.

    Raises:
        ValueError: If the framework name is not recognized.
    """
    name = override or os.environ.get("CIV6AZ_FRAMEWORK", "pytorch")
    name = name.lower().strip()

    try:
        return FrameworkType(name)
    except ValueError:
        valid = [f.value for f in FrameworkType]
        msg = f"Unknown framework '{name}'. Valid options: {valid}"
        raise ValueError(msg) from None


def framework_available(framework: FrameworkType | str) -> bool:
    """Check if a given framework is installed and importable.

    Args:
        framework: Framework to check (FrameworkType or string name).

    Returns:
        True if the framework and its dependencies are available.
    """
    if isinstance(framework, str):
        try:
            framework = FrameworkType(framework.lower().strip())
        except ValueError:
            return False

    if framework == FrameworkType.PYTORCH:
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    if framework == FrameworkType.JAX:
        try:
            import equinox  # noqa: F401
            import jax  # noqa: F401
            import optax  # noqa: F401

            return True
        except ImportError:
            return False

    return False  # pragma: no cover


def to_numpy(tensor: Any) -> np.ndarray:
    """Convert any framework tensor to a NumPy array.

    Supports torch.Tensor, jax.Array, and np.ndarray (passthrough).
    Uses zero-copy where possible.

    Args:
        tensor: Input tensor from any supported framework.

    Returns:
        NumPy array.

    Raises:
        TypeError: If the input type is not recognized.
    """
    if isinstance(tensor, np.ndarray):
        return tensor

    # Check for PyTorch tensor
    try:
        import torch

        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
    except ImportError:
        pass

    # Check for JAX array — use duck typing via jax.Array base class
    # (jnp.ndarray isinstance check is fragile across JAX versions)
    try:
        import jax

        if isinstance(tensor, jax.Array):
            return np.asarray(tensor)
    except ImportError:
        pass

    # Try generic __array__ protocol
    if hasattr(tensor, "__array__"):
        return np.asarray(tensor)

    raise TypeError(f"Cannot convert {type(tensor).__name__} to numpy array")


def from_numpy(
    array: np.ndarray,
    framework: FrameworkType | str,
    device: str = "cpu",
) -> Any:
    """Convert a NumPy array to a framework-specific tensor.

    Args:
        array: NumPy array to convert.
        framework: Target framework.
        device: Target device (used by PyTorch; JAX uses its own device placement).

    Returns:
        Framework-specific tensor.

    Raises:
        ValueError: If the framework is not recognized.
        ImportError: If the framework is not installed.
    """
    if isinstance(framework, str):
        framework = FrameworkType(framework.lower().strip())

    if framework == FrameworkType.PYTORCH:
        import torch

        tensor = torch.from_numpy(array)
        if device != "cpu":
            tensor = tensor.to(device)
        return tensor

    if framework == FrameworkType.JAX:
        import jax.numpy as jnp

        return jnp.array(array)

    msg = f"Unsupported framework: {framework}"
    raise ValueError(msg)  # pragma: no cover
