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
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch

from alpha_zero.utils.safe_load import (
    UNSAFE_CHECKPOINT_ENV_VAR,
    UnsafeCheckpointError,
    safe_torch_load,
    unsafe_checkpoint_loading_allowed,
)

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
    allow_unsafe_fallback: bool = False,
) -> CheckpointLoadResult:
    """Load a checkpoint file with security best practices.

    Attempts ``weights_only=True`` first. There is **no automatic downgrade**:
    if torch's tensor-only unpickler rejects the payload, this raises
    :class:`~alpha_zero.utils.safe_load.UnsafeCheckpointError` unless unsafe
    loading has been authorised -- by ``allow_unsafe_fallback=True`` or by
    ``CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1`` in the environment (read at call
    time). An authorised unsafe load is logged at WARNING and reported in the
    result's ``loaded_safely=False`` / ``warnings``.

    Before 2026-09-02 ``allow_unsafe_fallback`` defaulted to ``True`` and the
    six live callers (``MicroCivNetwork.load``/``Civ6Network.load``/
    ``MuZeroNetwork.load``, the two learners and ``MuZeroTrainer``) passed it
    explicitly, so every checkpoint in the repo was unpickled with arbitrary
    code execution the moment the safe path failed.

    Args:
        path: Path to the checkpoint file.
        map_location: Device to map tensors to (e.g., "cpu", "cuda"). Defaults
            to ``"cpu"``, matching
            :func:`alpha_zero.core.checkpoint_utils.load_checkpoint`.
            Passing ``None`` restores tensors to whatever device they were
            saved from, which makes a GPU-trained checkpoint unloadable on a
            CPU-only machine — the caller must opt into that explicitly rather
            than get it by default.
        strict_security: If True, never load unsafely, even when the
            environment variable is set. Raises if safe loading fails.
        allow_unsafe_fallback: Per-call opt-in to ``weights_only=False`` when
            the safe load is rejected. Default ``False``. The name is kept for
            backwards compatibility; it is honoured together with the
            environment variable.

    Returns:
        CheckpointLoadResult with the loaded data and safety metadata.

    Raises:
        UnsafeCheckpointError: (a ``RuntimeError``) if safe loading fails and
            unsafe loading is not authorised, or ``strict_security=True``.
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
        safe_load_error = e
        logger.debug(f"Safe loading failed: {safe_load_error}")

    # strict_security wins over every opt-in, including the environment.
    if strict_security:
        raise UnsafeCheckpointError(
            f"Cannot load checkpoint safely (weights_only=True). "
            f"Error: {safe_load_error}. strict_security=True forbids unsafe loading "
            f"of {path}."
        ) from safe_load_error

    if not unsafe_checkpoint_loading_allowed(allow_unsafe_fallback):
        raise UnsafeCheckpointError(
            f"Safe loading failed and fallback is disabled for {path}. "
            f"Error: {safe_load_error}. Automatic fallback to weights_only=False has "
            "been removed because it executes arbitrary code from the file. If this "
            "checkpoint comes from a trusted source, pass allow_unsafe_fallback=True "
            f"or set {UNSAFE_CHECKPOINT_ENV_VAR}=1."
        ) from safe_load_error

    # Authorised unsafe load. safe_torch_load owns the single weights_only=False
    # site in the repo and logs the WARNING; the UserWarning keeps the
    # historical contract for callers that filter on it.
    authority = (
        "allow_unsafe_fallback=True" if allow_unsafe_fallback else f"{UNSAFE_CHECKPOINT_ENV_VAR}=1"
    )
    warning_msg = (
        f"Loading checkpoint with weights_only=False (authorised by {authority}). "
        f"Only load checkpoints from trusted sources. Path: {path}"
    )
    result_warnings.append(warning_msg)
    warnings.warn(warning_msg, UserWarning, stacklevel=2)

    data = safe_torch_load(path, map_location=map_location, allow_unsafe=True)
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
    return extract_state_dict(result.data)


def extract_state_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Pull the model state dict out of already-loaded checkpoint data.

    Split out of :func:`load_model_state` so callers that already hold the
    loaded blob — :func:`load_network` needs the config from the same blob —
    can reuse the key dispatch without reading the file a second time.

    Args:
        data: Deserialized checkpoint contents.

    Returns:
        The model state dict.

    Raises:
        KeyError: If no recognised state dict is present.
    """
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


class HorizonMismatch(ValueError):  # noqa: N818 - name fixed by the plan (LF-A16)
    """A checkpoint was trained at a different ``max_turns`` than it is read at.

    The file loads and plays; the numbers it produces just do not mean what the
    reader thinks. A ``ValueError`` so existing ``except ValueError`` handlers on
    the resume path keep working.
    """


#: Metadata values must survive ``torch.load(weights_only=True)``.
_PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def validate_primitive_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return ``metadata`` unchanged, or raise ``TypeError`` for a non-primitive value."""
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, _PRIMITIVE_TYPES):
            raise TypeError(
                f"checkpoint metadata must be primitive (str/int/float/bool/None); "
                f"{key!r} is {type(value).__name__}. A richer value would force every "
                "later load onto the unsafe weights_only=False path."
            )
    return metadata


def read_git_sha(start: Path | None = None) -> str | None:
    """Commit at ``HEAD`` by reading ``.git`` directly -- no subprocess.

    Walks up from ``start`` (default: the working directory) to the nearest
    ``.git``; follows a symbolic ``HEAD`` through loose and packed refs. Returns
    ``None`` outside a repository or when the ref cannot be resolved, rather
    than guessing.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        git_dir = candidate / ".git"
        if git_dir.is_file():  # worktree: "gitdir: <path>"
            try:
                pointer = git_dir.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            git_dir = Path(pointer.removeprefix("gitdir:").strip())
        if not git_dir.is_dir():
            continue
        try:
            head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not head.startswith("ref:"):
            return head or None
        ref = head.removeprefix("ref:").strip()
        loose = git_dir / ref
        if loose.is_file():
            try:
                return loose.read_text(encoding="utf-8").strip() or None
            except OSError:
                return None
        packed = git_dir / "packed-refs"
        if packed.is_file():
            try:
                for line in packed.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0]
            except OSError:
                return None
        return None
    return None


def build_provenance(config: Any, *, git_sha: str | None) -> dict[str, Any]:
    """Primitive-only provenance for a checkpoint, read from a *resolved* training config.

    ``config`` is a ``TrainingConfig`` (duck-typed: this module cannot import
    ``training_civ6``). Values are what the run actually used, so a Gumbel flag
    left at its default is recorded as the default, not as ``None``.
    """
    from alpha_zero.envs.microciv import ENV_VERSION

    metadata: dict[str, Any] = {
        "board_size": int(config.board_size),
        "max_turns": int(config.max_turns),
        "num_players": int(config.num_players),
        "mcts": "gumbel" if config.use_gumbel_mcts else "civ6",
        "gumbel_k": int(config.gumbel_num_sampled_actions),
        "gumbel_sims": int(config.gumbel_num_simulations),
        "c_scale": float(config.gumbel_c_scale),
        "policy_target": str(config.gumbel_policy_target),
        "bootstrap": str(config.bootstrap),
        "seed": None if config.seed is None else int(config.seed),
        "git_sha": git_sha,
        "env_version": int(ENV_VERSION),
    }
    return validate_primitive_metadata(metadata)


def check_provenance(
    metadata: dict[str, Any] | None,
    *,
    expected_max_turns: int | None,
    source: str | Path,
) -> None:
    """Refuse a checkpoint whose training horizon differs from the one it is read at.

    Args:
        metadata: The checkpoint's ``metadata`` entry, or ``None`` for a file
            written before provenance existed.
        expected_max_turns: The horizon the reader will play at. ``None`` means
            the caller does not care, and nothing is checked.
        source: Path named in messages.

    Raises:
        HorizonMismatch: ``max_turns`` is recorded and differs.

    A legacy file (no metadata) is logged at WARNING and passes. A board-size
    mismatch is deliberately not handled here: ``load_state_dict`` already fails
    on it with a shape error.
    """
    if expected_max_turns is None:
        return
    if not metadata:
        logger.warning(
            "Checkpoint %s has no provenance metadata (written before LF-A16); "
            "cannot verify it was trained at max_turns=%d.",
            source,
            expected_max_turns,
        )
        return
    trained = metadata.get("max_turns")
    if trained is not None and trained != expected_max_turns:
        raise HorizonMismatch(
            f"{source} was trained at max_turns={trained} but is being read at "
            f"max_turns={expected_max_turns}. Results at a different horizon are not "
            "comparable; train or evaluate at a matching --max_turns."
        )


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
    expected_max_turns: int | None = None,
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
        expected_max_turns: When given, the checkpoint's recorded ``max_turns``
            must match (:class:`HorizonMismatch`); a file with no provenance
            metadata only warns.
        **arch_overrides: Constructor arguments used when the checkpoint has no
            embedded config.

    Raises:
        HorizonMismatch: ``expected_max_turns`` disagrees with the checkpoint.
        ValueError: If the checkpoint carries no config and none was supplied,
            or if the board size disagrees with ``expected_board_size``.
    """
    from alpha_zero.utils.device import resolve_device_str

    resolved = resolve_device_str(device)

    # Read the file once: the weights and the architecture config both come out
    # of the same blob, and web session startup pays this latency directly.
    result = safe_load_checkpoint(path, map_location=resolved)
    state_dict = extract_state_dict(result.data)
    raw_config = result.data.get("config") if isinstance(result.data, dict) else None
    metadata = result.data.get("metadata") if isinstance(result.data, dict) else None
    check_provenance(metadata, expected_max_turns=expected_max_turns, source=path)

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


class EncoderMismatchError(RuntimeError):
    """Raised when a model's weights do not match the expected spatial encoder."""


#: Substrings that appear only in hex-encoder ``state_dict`` keys.
_HEX_ENCODER_KEY_MARKERS = ("hex_conv", "crystalline", "spatial_to_tokens")

#: Prefix under which the spatial encoder's weights live, for both encoder
#: types — the consuming network assigns the encoder to ``self.patch_embed``
#: regardless of which one it built, to keep checkpoint keys stable.
_ENCODER_KEY_PREFIX = "encoder.patch_embed."

#: Encoder identifiers this module knows how to discriminate.
KNOWN_ENCODER_TYPES = ("patch_embed", "hex_crystalline")


def _encoder_keys(keys: Iterable[str]) -> list[str]:
    """Return the subset of *keys* belonging to the spatial encoder.

    Scoped to the encoder subtree so an unrelated module whose name happens to
    contain one of the markers cannot trigger a false positive. Keys that carry
    no prefix at all (a bare encoder ``state_dict``) are all considered
    in-scope.
    """
    keys = list(keys)
    scoped = [k for k in keys if k.startswith(_ENCODER_KEY_PREFIX)]
    return scoped if scoped else keys


def classify_encoder_keys(keys: Iterable[str]) -> str:
    """Infer which spatial encoder a set of ``state_dict`` keys belongs to.

    Args:
        keys: ``state_dict`` key names.

    Returns:
        ``"hex_crystalline"`` if hex-specific weights are present, else
        ``"patch_embed"``.
    """
    scoped = _encoder_keys(keys)
    has_hex = any(marker in k for k in scoped for marker in _HEX_ENCODER_KEY_MARKERS)
    return "hex_crystalline" if has_hex else "patch_embed"


def validate_encoder_state(
    model: torch.nn.Module,
    expected_encoder_type: str,
    *,
    checkpoint_keys: Iterable[str] | None = None,
    strict: bool = True,
) -> bool:
    """Check that a model carries the weights of the expected spatial encoder.

    Several load paths use ``load_state_dict(strict=False)``, which turns an
    architecture mismatch into a warning: the encoder is left randomly
    initialised, the model still runs, and the damage is invisible in aggregate
    metrics — worst of all during an evaluation whose whole purpose is comparing
    encoders.

    Catching that requires comparing the *checkpoint* against the model. A
    model's own ``state_dict`` keys are a property of the module tree that was
    constructed, not of whatever was loaded into it, so inspecting the model
    alone can only confirm what the caller already asked for. Pass
    ``checkpoint_keys`` to get the real check; without it this degrades to a
    build-time assertion, which is why the argument is strongly recommended
    wherever a checkpoint is involved.

    Args:
        model: The model to inspect, after weights have been loaded.
        expected_encoder_type: One of :data:`KNOWN_ENCODER_TYPES`.
        checkpoint_keys: ``state_dict`` keys from the checkpoint that was
            loaded. When given, the checkpoint's encoder identity is compared
            against both the expectation and the model, so a ``strict=False``
            cross-load is detected.
        strict: If True, raise on mismatch. If False, log an error and return
            False so the caller can decide.

    Returns:
        True when the model — and the checkpoint, when supplied — match
        *expected_encoder_type*.

    Raises:
        ValueError: If *expected_encoder_type* is not a known encoder.
        EncoderMismatchError: If ``strict`` and a mismatch is found.
    """
    if expected_encoder_type not in KNOWN_ENCODER_TYPES:
        raise ValueError(
            f"Unknown encoder type {expected_encoder_type!r}; "
            f"expected one of {sorted(KNOWN_ENCODER_TYPES)}"
        )

    model_keys = list(model.state_dict())
    model_type = classify_encoder_keys(model_keys)

    problems: list[str] = []
    if model_type != expected_encoder_type:
        problems.append(
            f"the model was built as {model_type!r}, not the expected {expected_encoder_type!r}"
        )

    if checkpoint_keys is not None:
        checkpoint_keys = list(checkpoint_keys)
        checkpoint_type = classify_encoder_keys(checkpoint_keys)
        if checkpoint_type != expected_encoder_type:
            problems.append(
                f"the checkpoint holds {checkpoint_type!r} weights, not the expected "
                f"{expected_encoder_type!r}"
            )

        # The decisive check: encoder weights present in the model but absent
        # from the checkpoint were left randomly initialised by a lenient load.
        missing = sorted(set(_encoder_keys(model_keys)) - set(checkpoint_keys))
        if missing:
            preview = ", ".join(missing[:4])
            suffix = f" (+{len(missing) - 4} more)" if len(missing) > 4 else ""
            problems.append(
                f"{len(missing)} encoder weight(s) are absent from the checkpoint and "
                f"remain randomly initialised: {preview}{suffix}"
            )
    else:
        logger.debug(
            "validate_encoder_state called without checkpoint_keys; this only confirms "
            "how the model was built, not what was loaded into it."
        )

    if not problems:
        logger.debug("Encoder validation passed: %s", expected_encoder_type)
        return True

    message = (
        "Encoder mismatch: "
        + "; ".join(problems)
        + (
            ". Loading a checkpoint into the wrong architecture leaves parameters "
            "randomly initialised without raising, so this is treated as fatal."
        )
    )
    if strict:
        raise EncoderMismatchError(message)

    logger.error(message)
    return False


__all__ = [
    "KNOWN_ENCODER_TYPES",
    "CheckpointLoadResult",
    "EncoderMismatchError",
    "classify_encoder_keys",
    "compute_weight_checksum",
    "load_model_state",
    "load_network",
    "safe_load_checkpoint",
    "save_checkpoint",
    "validate_encoder_state",
]
