"""JAX checkpoint serialization and cross-framework conversion.

Provides save/load for JAX/Equinox models and bidirectional conversion
between JAX and PyTorch checkpoint formats.

Usage:
    from alpha_zero.core.jax_checkpoint import (
        save_jax_checkpoint,
        load_jax_checkpoint,
        jax_to_torch_state_dict,
        torch_to_jax_params,
    )

    # Save JAX model
    save_jax_checkpoint(wrapper.model, "checkpoint_100", metadata={"iteration": 100})

    # Load JAX model
    model = load_jax_checkpoint("checkpoint_100", template_model)

    # Cross-framework conversion
    state_dict = jax_to_torch_state_dict(jax_model)
    jax_params = torch_to_jax_params(torch_state_dict, template_model)
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, cast

import numpy as np

from alpha_zero.core.framework import RESERVED_CHECKPOINT_KEYS

logger = logging.getLogger(__name__)

try:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def _require_jax() -> None:
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX checkpoint requires: pip install -e '.[jax]' "
            "(installs jax, jaxlib, equinox, optax)"
        )


def save_jax_checkpoint(
    model: Any,
    path: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a JAX/Equinox model to disk.

    Creates two files:
    - {path}.eqx: Serialized model parameters (Equinox format)
    - {path}.json: Configuration and metadata sidecar

    Args:
        model: Equinox model to save.
        path: Base path (without extension).
        metadata: Optional metadata to include in the JSON sidecar.
    """
    _require_jax()

    model_path = f"{path}.eqx"
    meta_path = f"{path}.json"

    eqx.tree_serialise_leaves(model_path, model)

    meta: dict[str, Any] = {"framework": "jax"}
    if hasattr(model, "config") and hasattr(model.config, "to_dict"):
        meta["config"] = model.config.to_dict()
    if metadata:
        overwritten = set(metadata.keys()) & RESERVED_CHECKPOINT_KEYS
        if overwritten:
            logger.warning(
                "Metadata keys %s are reserved and will not be overwritten",
                overwritten,
            )
        meta.update({k: v for k, v in metadata.items() if k not in RESERVED_CHECKPOINT_KEYS})

    Path(meta_path).write_text(json.dumps(meta, indent=2, default=str))

    logger.info("Saved JAX checkpoint to %s", model_path)


def load_jax_checkpoint(
    path: str,
    template: Any,
) -> Any:
    """Load a JAX/Equinox model from disk.

    Args:
        path: Base path (without extension).
        template: Template model with same architecture (parameters will be overwritten).

    Returns:
        Equinox model with loaded parameters.
    """
    _require_jax()

    model_path = f"{path}.eqx"
    model = eqx.tree_deserialise_leaves(model_path, template)

    logger.info("Loaded JAX checkpoint from %s", model_path)
    return model


# ---------------------------------------------------------------------------
# Cross-framework conversion: JAX → PyTorch
# ---------------------------------------------------------------------------


def _collect_jax_params(model: Any) -> dict[str, np.ndarray]:
    """Recursively collect all array leaves from an Equinox model into a flat dict.

    Uses dot-separated paths as keys (e.g., 'spatial_encoder.conv_in.weight').

    Args:
        model: Equinox model.

    Returns:
        Dictionary mapping parameter names to NumPy arrays.
    """
    _require_jax()

    params: dict[str, np.ndarray] = {}

    def _collect(prefix: str, obj: Any) -> None:
        if eqx.is_array(obj):
            params[prefix] = np.asarray(obj)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _collect(f"{prefix}.{i}", item)
        elif hasattr(obj, "__dict__"):
            for name, value in vars(obj).items():
                if name.startswith("_"):
                    continue
                child_prefix = f"{prefix}.{name}" if prefix else name
                _collect(child_prefix, value)

    _collect("", model)
    return params


def jax_to_torch_state_dict(
    jax_model: Any,
    prefix_map: dict[str, str] | None = None,
) -> OrderedDict[str, Any]:
    """Convert JAX/Equinox model parameters to a PyTorch state_dict.

    The conversion handles:
    - Conv2d weights: Same OIHW layout in both frameworks (no transpose needed)
    - Linear weights: JAX (out, in) → PyTorch (out, in) (same layout)
    - GroupNorm weight/bias: Same 1D layout
    - Skips non-parameter attributes (config, etc.)

    Args:
        jax_model: Equinox model with trained parameters.
        prefix_map: Optional mapping from JAX param names to PyTorch state_dict keys.
            If None, uses a default mapping based on the standard network architecture.

    Returns:
        OrderedDict suitable for torch.nn.Module.load_state_dict().
    """
    _require_jax()
    import torch

    jax_params = _collect_jax_params(jax_model)
    state_dict: OrderedDict[str, Any] = OrderedDict()

    for jax_name, jax_array in jax_params.items():
        # Apply prefix mapping if provided
        if prefix_map and jax_name in prefix_map:
            torch_name = prefix_map[jax_name]
        else:
            torch_name = jax_name

        # Convert to torch tensor
        torch_tensor = torch.from_numpy(jax_array.copy())
        state_dict[torch_name] = torch_tensor

    total_elements = sum(v.numel() for v in state_dict.values())
    logger.info(
        "Converted JAX → PyTorch state_dict: %d parameters (%d total elements)",
        len(state_dict),
        total_elements,
    )

    return state_dict


# ---------------------------------------------------------------------------
# Cross-framework conversion: PyTorch → JAX
# ---------------------------------------------------------------------------


def torch_to_jax_params(
    state_dict: dict[str, Any],
    template: Any,
    prefix_map: dict[str, str] | None = None,
) -> Any:
    """Convert a PyTorch state_dict into JAX parameters and update a template model.

    Args:
        state_dict: PyTorch state_dict (from model.state_dict()).
        template: Template Equinox model with same architecture.
        prefix_map: Optional mapping from PyTorch state_dict keys to JAX param names.

    Returns:
        Equinox model with parameters loaded from the PyTorch state_dict.
    """
    _require_jax()

    # Collect template parameter paths
    template_params = _collect_jax_params(template)

    # Build reverse mapping
    torch_to_jax_name: dict[str, str] = {}
    if prefix_map:
        torch_to_jax_name = prefix_map
    else:
        # Default: assume same names
        for jax_name in template_params:
            torch_to_jax_name[jax_name] = jax_name

    # Build the update dict
    updates: dict[str, jax.Array] = {}
    matched = 0

    # Detect torch.Tensor type robustly (torch may not be installed)
    try:
        import torch as _torch

        _torch_tensor_type: type | None = _torch.Tensor
    except ImportError:
        _torch_tensor_type = None

    for torch_key, torch_tensor in state_dict.items():
        # Skip non-tensor entries (metadata, nested structures)
        if isinstance(torch_tensor, (dict, list, str, int, float, bool)):
            logger.debug(
                "Skipping non-tensor state_dict entry: %s (%s)",
                torch_key,
                type(torch_tensor).__name__,
            )
            continue

        if _torch_tensor_type is not None and isinstance(torch_tensor, _torch_tensor_type):
            np_val = cast(Any, torch_tensor).detach().cpu().numpy()
        else:
            try:
                np_val = np.asarray(torch_tensor)
            except (TypeError, ValueError) as exc:
                logger.warning("Cannot convert state_dict[%s] to array: %s", torch_key, exc)
                continue

        # Try direct match
        jax_name = torch_to_jax_name.get(torch_key, torch_key)
        if jax_name in template_params:
            if template_params[jax_name].shape == np_val.shape:
                updates[jax_name] = jnp.array(np_val)
                matched += 1
            else:
                logger.warning(
                    "Shape mismatch for '%s': JAX %s vs PyTorch %s",
                    jax_name,
                    template_params[jax_name].shape,
                    np_val.shape,
                )

    logger.info(
        "Converted PyTorch → JAX: %d/%d parameters matched (PyTorch: %d, JAX: %d)",
        matched,
        len(state_dict),
        len(state_dict),
        len(template_params),
    )

    # Apply updates to template — one parameter at a time for robustness
    updated_model = template
    applied = 0

    for param_path, param_value in updates.items():
        parts = param_path.split(".")
        if not parts or parts == [""]:
            continue

        def _make_accessor(path_parts: list[str]) -> Any:
            """Build a leaf-accessor closure for eqx.tree_at."""

            def accessor(model: Any) -> Any:
                obj = model
                for part in path_parts:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                return obj

            return accessor

        try:
            accessor_fn = _make_accessor(parts)
            updated_model = eqx.tree_at(accessor_fn, updated_model, param_value)
            applied += 1
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to set parameter '%s': %s: %s",
                param_path,
                type(exc).__name__,
                exc,
            )

    if applied < matched:
        logger.warning(
            "Only applied %d/%d matched parameters to model",
            applied,
            matched,
        )
    else:
        logger.info(
            "Successfully applied %d/%d parameters to JAX model",
            applied,
            matched,
        )

    return updated_model
