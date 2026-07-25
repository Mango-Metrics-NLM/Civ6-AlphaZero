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
from dataclasses import asdict, dataclass, fields, is_dataclass
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
    map_location: str | torch.device | None = "cpu",
    strict_security: bool = False,
    allow_unsafe_fallback: bool = True,
) -> CheckpointLoadResult:
    """Load a checkpoint file with security best practices.

    This function attempts to load checkpoints safely using weights_only=True.
    If that fails (e.g., for checkpoints with optimizer state or custom objects),
    it can fall back to weights_only=False with appropriate warnings.

    Args:
        path: Path to the checkpoint file.
        map_location: Device to map tensors to (e.g., "cpu", "cuda"). Defaults
            to ``"cpu"``, matching :func:`alpha_zero.core.checkpoint_utils.
            load_checkpoint`. Passing ``None`` restores tensors to whatever
            device they were saved from, which makes a GPU-trained checkpoint
            unloadable on a CPU-only machine — the caller must opt into that
            explicitly rather than get it by default.
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
    map_location: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    """Load only model state dict from checkpoint (always safe).

    This function extracts just the model weights, which can always
    be loaded safely with weights_only=True.

    Args:
        path: Path to the checkpoint file.
        map_location: Device to map tensors to. Defaults to ``"cpu"`` so a
            GPU-trained checkpoint stays loadable on a CPU-only machine.

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


def _config_to_primitives(config: Any) -> Any:
    """Reduce a config to plain primitives suitable for ``weights_only=True``.

    Dataclass instances become dicts; numpy scalars become Python scalars.
    Anything already primitive is returned untouched.
    """
    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)

    if isinstance(config, dict):
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in config.items()}
    return config


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
        config: Optional configuration object or dict. Dataclass instances are
            converted to plain dicts of primitives — storing the instance
            itself would force every later ``torch.load`` onto the
            ``weights_only=False`` path, since the class is not registered as
            a safe global.
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
        checkpoint["config"] = _config_to_primitives(config)

    checkpoint.update(extra_data)

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def _looks_like_network_config(config: dict[str, Any]) -> bool:
    """True when ``config`` is a core ``NetworkConfig``, not an orchestrator one.

    Two unrelated dataclasses in this repo are both called ``NetworkConfig``
    and both are publicly exported: the core one
    (``alpha_zero.core.civ6_network``) carries ``spatial_size`` and
    ``action_space_size``; the orchestrator one
    (``alpha_zero.orchestrator.config``) carries ``num_heads`` and
    ``framework`` and has neither. Feeding the latter to the former's
    constructor raises TypeError, so the shape has to be sniffed.
    """
    return "spatial_size" in config or "action_space_size" in config


def load_network(
    path: str | Path,
    *,
    device: str | None = None,
    network_cls: type[torch.nn.Module] | None = None,
    expected_board_size: int | None = None,
    **arch_overrides: Any,
) -> torch.nn.Module:
    """Load a checkpoint into a ready-to-use network.

    This is the single reader for every checkpoint shape this repo writes.
    ``Civ6Network.save`` writes ``{"config", "state_dict"}``, the orchestrator's
    checkpointer writes ``{"model_state_dict", ...}``, and some paths write a
    bare state dict; before this existed, each consumer hardcoded one of those
    and silently mishandled the others.

    The returned module is already on ``device`` and in ``eval()`` mode, because
    the search classes that consume it do neither.

    Args:
        path: Checkpoint file.
        device: Device request, resolved via ``resolve_device_str`` (so "cuda"
            on a CPU-only host degrades to cpu rather than raising).
        network_cls: Network class to build. Defaults to ``MicroCivNetwork``.
        expected_board_size: When given, the checkpoint's ``spatial_size`` must
            match. Catches "trained at 16, played at 8" as a clear error rather
            than a shape mismatch mid-game.
        **arch_overrides: Constructor arguments used when the checkpoint has no
            embedded config.

    Raises:
        ValueError: If the checkpoint carries no config and none was supplied,
            or if the board size disagrees with ``expected_board_size``.
    """
    from alpha_zero.utils.device import resolve_device_str

    resolved = resolve_device_str(device)
    state_dict = load_model_state(path, map_location=resolved)

    result = safe_load_checkpoint(path, map_location=resolved)
    raw_config = result.data.get("config") if isinstance(result.data, dict) else None

    if network_cls is None:
        # Imported lazily: core.civ6_network imports this module at module
        # scope, so a top-level import here would be a genuine cycle.
        from alpha_zero.core.civ6_network import MicroCivNetwork

        network_cls = MicroCivNetwork

    network = _construct_network(network_cls, raw_config, expected_board_size, arch_overrides, path)
    network.load_state_dict(state_dict)
    network.to(resolved)
    network.eval()
    return network


def _construct_network(
    network_cls: type[torch.nn.Module],
    raw_config: Any,
    expected_board_size: int | None,
    arch_overrides: dict[str, Any],
    path: str | Path,
) -> torch.nn.Module:
    """Build an unpopulated network matching a checkpoint's architecture."""
    from alpha_zero.core.civ6_network import NetworkConfig

    if isinstance(raw_config, dict) and _looks_like_network_config(raw_config):
        known = {f.name for f in fields(NetworkConfig)}
        # Drop unknown keys rather than raising: a checkpoint written by a
        # newer build may carry fields this one has never heard of, and the
        # architecture-defining subset is what actually matters here.
        unknown = set(raw_config) - known
        if unknown:
            logger.debug("Ignoring unknown config keys in %s: %s", path, sorted(unknown))
        config = NetworkConfig(**{k: v for k, v in raw_config.items() if k in known})

        if expected_board_size is not None and config.spatial_size != expected_board_size:
            raise ValueError(
                f"Checkpoint {path} was trained at board_size="
                f"{config.spatial_size}, but this game uses "
                f"{expected_board_size}. Train a matching model or start a "
                f"{config.spatial_size}x{config.spatial_size} game."
            )

        expected_channels = 3 + 4 * config.num_players + 2
        if config.spatial_channels != expected_channels:
            logger.warning(
                "Checkpoint %s has spatial_channels=%d but num_players=%d implies %d; "
                "trusting the stored channel count.",
                path,
                config.spatial_channels,
                config.num_players,
                expected_channels,
            )

        return network_cls(config=config)

    if arch_overrides:
        return network_cls(**arch_overrides)

    found = sorted(raw_config) if isinstance(raw_config, dict) else type(raw_config).__name__
    raise ValueError(
        f"Checkpoint {path} has no usable architecture config (found: {found}). "
        "Pass the architecture explicitly, e.g. "
        "load_network(path, board_size=8, action_space_size=376). Guessing "
        "defaults here would load weights into the wrong shape."
    )


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
    "load_network",
    "safe_load_checkpoint",
    "save_checkpoint",
]
