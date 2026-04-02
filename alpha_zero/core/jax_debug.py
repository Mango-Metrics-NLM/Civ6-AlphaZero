"""JAX-specific debugging and logging utilities.

Provides structured logging for JAX operations, debug mode toggling,
and output validation for development and troubleshooting.

Usage:
    from alpha_zero.core.jax_debug import (
        JaxDebugLogger,
        jax_debug_mode,
        validate_jax_outputs,
    )

    debug_logger = JaxDebugLogger.from_env()
    debug_logger.log_array("policy", policy_array)

    with jax_debug_mode():
        # JIT is disabled, NaN checking enabled
        result = model(inputs)

Environment variables:
    CIV6AZ_JAX_DEBUG: Set to "true" to enable verbose debug logging.
    CIV6AZ_LOG_LEVEL: Standard log level (DEBUG, INFO, WARNING, ERROR).
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Generator
from typing import Any

logger = logging.getLogger(__name__)


class JaxDebugLogger:
    """Structured logger for JAX operations.

    Logs tensor shapes, dtypes, value ranges, and JIT compilation events
    at configurable verbosity levels.
    """

    def __init__(self, enabled: bool = False, log_level: str = "DEBUG") -> None:
        """Initialize the debug logger.

        Args:
            enabled: Whether debug logging is active.
            log_level: Minimum log level for debug messages.
        """
        self._enabled = enabled
        self._logger = logging.getLogger(f"{__name__}.debug")
        self._log_level = getattr(logging, log_level.upper(), logging.DEBUG)

    @classmethod
    def from_env(cls) -> JaxDebugLogger:
        """Create a debug logger from environment variables.

        Reads CIV6AZ_JAX_DEBUG and CIV6AZ_LOG_LEVEL.
        """
        enabled = os.environ.get("CIV6AZ_JAX_DEBUG", "false").lower() in (
            "true",
            "1",
            "yes",
        )
        log_level = os.environ.get("CIV6AZ_LOG_LEVEL", "DEBUG")
        return cls(enabled=enabled, log_level=log_level)

    @property
    def enabled(self) -> bool:
        """Whether debug logging is currently active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def log_array(self, name: str, arr: Any) -> None:
        """Log shape, dtype, and value statistics for an array.

        Args:
            name: Descriptive name for the array.
            arr: Array-like object (JAX array, NumPy array, etc.).
        """
        if not self._enabled:
            return

        try:
            import numpy as np

            a = np.asarray(arr)
            self._logger.log(
                self._log_level,
                "Array '%s': shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f nan_count=%d",
                name,
                a.shape,
                a.dtype,
                float(a.min()),
                float(a.max()),
                float(a.mean()),
                int(np.isnan(a).sum()),
            )
        except Exception as e:
            self._logger.log(self._log_level, "Array '%s': failed to inspect: %s", name, e)

    def log_params(self, params: Any, label: str = "model") -> None:
        """Log parameter tree structure and total parameter count.

        Args:
            params: Equinox model or JAX parameter pytree.
            label: Descriptive label for the parameter set.
        """
        if not self._enabled:
            return

        try:
            import jax

            leaves = jax.tree.leaves(params)
            array_leaves = [leaf for leaf in leaves if hasattr(leaf, "size")]
            if not array_leaves:
                self._logger.log(self._log_level, "Params '%s': no array leaves found", label)
                return

            total = sum(leaf.size for leaf in array_leaves)
            typed_leaves = [leaf for leaf in array_leaves if hasattr(leaf, "dtype")]
            if len(typed_leaves) < len(array_leaves):
                self._logger.log(
                    self._log_level,
                    "Params '%s': %d/%d leaves lack dtype, memory estimate may be approximate",
                    label,
                    len(array_leaves) - len(typed_leaves),
                    len(array_leaves),
                )
            total_bytes = sum(leaf.size * leaf.dtype.itemsize for leaf in typed_leaves)
            self._logger.log(
                self._log_level,
                "Params '%s': %d leaves, %d total parameters (%.2f MB)",
                label,
                len(array_leaves),
                total,
                total_bytes / (1024 * 1024),
            )
        except Exception as e:
            self._logger.log(self._log_level, "Params '%s': failed to inspect: %s", label, e)

    def log_compilation(self, fn_name: str, input_shapes: dict[str, tuple] | None = None) -> None:
        """Log a JIT compilation event.

        Args:
            fn_name: Name of the function being compiled.
            input_shapes: Optional dict of input name → shape.
        """
        if not self._enabled:
            return

        shapes_str = ""
        if input_shapes:
            shapes_str = ", ".join(f"{k}={v}" for k, v in input_shapes.items())

        self._logger.log(
            self._log_level,
            "JIT compiling '%s'%s",
            fn_name,
            f" with shapes: {shapes_str}" if shapes_str else "",
        )


@contextlib.contextmanager
def jax_debug_mode() -> Generator[None, None, None]:
    """Context manager that disables JIT and enables NaN checking.

    Useful for debugging JAX computations that produce unexpected results.
    Within this context, all JAX operations run eagerly (no XLA compilation)
    and any NaN values immediately raise an error.

    Example:
        with jax_debug_mode():
            result = model(inputs)  # Runs eagerly, NaN → error
    """
    import jax

    # Save current settings (jax.config attributes are runtime-only, not typed)
    old_disable_jit = jax.config.jax_disable_jit
    old_debug_nans = jax.config.jax_debug_nans

    try:
        jax.config.update("jax_disable_jit", True)
        jax.config.update("jax_debug_nans", True)
        logger.info("JAX debug mode enabled: JIT disabled, NaN checking active")
        yield
    finally:
        jax.config.update("jax_disable_jit", old_disable_jit)
        jax.config.update("jax_debug_nans", old_debug_nans)
        logger.info("JAX debug mode disabled: restored previous settings")


def validate_jax_outputs(
    policy: Any,
    value: Any,
    expected_action_size: int | None = None,
) -> None:
    """Validate JAX network outputs for shape and value correctness.

    Args:
        policy: Policy output array, shape (B, A) or (A,).
        value: Value output, shape (B,) or scalar.
        expected_action_size: Expected size of action dimension.

    Raises:
        ValueError: If outputs have unexpected shapes or values.
    """
    import numpy as np

    policy_np = np.asarray(policy)
    value_np = np.asarray(value)

    # Check policy shape
    if policy_np.ndim not in (1, 2):
        msg = f"Policy must be 1D or 2D, got shape {policy_np.shape}"
        raise ValueError(msg)

    if expected_action_size is not None:
        action_dim = policy_np.shape[-1]
        if action_dim != expected_action_size:
            msg = f"Policy action dim {action_dim} != expected {expected_action_size}"
            raise ValueError(msg)

    # Check for NaN
    if np.isnan(policy_np).any():
        msg = f"Policy contains NaN values: {np.isnan(policy_np).sum()} NaN elements"
        raise ValueError(msg)

    if np.isnan(value_np).any():
        msg = "Value contains NaN"
        raise ValueError(msg)

    # Check value range (should be in [-1, 1] for tanh output)
    value_flat = value_np.ravel()
    if len(value_flat) > 0:
        if value_flat.min() < -1.01 or value_flat.max() > 1.01:
            logger.warning(
                "Value outside expected [-1, 1] range: min=%.4f, max=%.4f",
                float(value_flat.min()),
                float(value_flat.max()),
            )
