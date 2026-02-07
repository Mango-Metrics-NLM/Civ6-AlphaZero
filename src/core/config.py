"""Pydantic v2 configuration models for the Strategic Planning SDK.

This module defines all configuration dataclasses used across the SDK.
Configuration can be loaded from YAML files, composed via deep-merge
overrides, and patched at runtime through environment variables with a
configurable prefix (default ``SPSDK_``).

Example::

    from src.core.config import SDKConfig, load_config

    # Load from YAML with env-var overrides applied automatically
    config = load_config("configs/training.yaml")

    # Compose base + override files
    config = SDKConfig.from_yaml_with_overrides(
        "configs/base.yaml",
        "configs/experiment_1.yaml",
    )

    # Pure defaults with programmatic overrides
    config = load_config(overrides={"training": {"batch_size": 512}})
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.exceptions import ConfigFileNotFoundError, ConfigValidationError
from src.core.types import (
    SDK_ENV_PREFIX,
    BackboneType,
    DeviceType,
    SelectionPolicyType,
)

__all__ = [
    "CurriculumConfig",
    "CurriculumStage",
    "LoggingConfig",
    "MCTSConfig",
    "NetworkConfig",
    "SDKConfig",
    "TrainingConfig",
    "load_config",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*.

    Scalar values in *override* replace those in *base*.  Nested dicts are
    merged recursively.  Lists and other types are replaced wholesale.

    Args:
        base: The base dictionary.
        override: Dictionary whose values take precedence.

    Returns:
        A new dictionary containing the merged result.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _coerce_value(current: Any, raw: str) -> Any:
    """Coerce a raw string environment-variable value to the type of *current*.

    Args:
        current: The existing value whose type guides the coercion.
        raw: The raw string value from the environment.

    Returns:
        The coerced value.
    """
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    # For enums that are str-backed, attempt construction from the value
    if hasattr(current, "__class__") and issubclass(current.__class__, str):
        enum_cls = current.__class__
        if hasattr(enum_cls, "__members__"):
            return enum_cls(raw)
    return raw


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class MCTSConfig(BaseModel):
    """Configuration for Monte Carlo Tree Search.

    Attributes:
        num_simulations: Number of MCTS simulations per move.
        c_puct: PUCT exploration constant.
        dirichlet_alpha: Alpha parameter for Dirichlet noise at the root.
        dirichlet_epsilon: Mixing weight for Dirichlet noise at the root.
        temperature_schedule: Mapping of game step to temperature value.
            At each step, the temperature of the nearest preceding key is
            used.
        virtual_loss: Virtual loss applied during parallel MCTS to
            discourage multiple threads from exploring the same path.
        progressive_widening_alpha: Controls how many children are
            considered as visit counts grow (N^alpha).
        selection_policy: Child-selection policy variant.
    """

    num_simulations: int = Field(default=200, ge=1)
    c_puct: float = Field(default=1.5, gt=0)
    dirichlet_alpha: float = Field(default=0.3, gt=0)
    dirichlet_epsilon: float = Field(default=0.25, ge=0, le=1)
    temperature_schedule: dict[int, float] = Field(
        default_factory=lambda: {0: 1.0, 30: 0.1}
    )
    virtual_loss: float = Field(default=1.0, ge=0)
    progressive_widening_alpha: float = Field(default=0.5, gt=0)
    selection_policy: SelectionPolicyType = SelectionPolicyType.puct


class NetworkConfig(BaseModel):
    """Configuration for the neural network architecture.

    Attributes:
        backbone_type: Which backbone architecture to use.
        num_residual_blocks: Number of residual blocks in the backbone.
        num_filters: Number of convolutional filters per residual block.
        fc_hidden_dim: Hidden dimension for fully-connected layers
            shared by both heads.
        value_head_hidden_dim: Hidden dimension specific to the value head.
        policy_head_hidden_dim: Hidden dimension specific to the policy head.
        use_attention: Whether to use multi-head self-attention.
        attention_heads: Number of attention heads (used when
            ``use_attention`` is ``True``).
        dropout_rate: Dropout probability applied after attention and
            fully-connected layers.
        weight_init: Weight initialisation scheme (e.g. ``"kaiming"``,
            ``"xavier"``).
        compile_model: Whether to use ``torch.compile`` for optimised
            execution (requires PyTorch 2.0+).
    """

    backbone_type: BackboneType = BackboneType.resnet
    num_residual_blocks: int = Field(default=6, ge=1)
    num_filters: int = Field(default=64, ge=1)
    fc_hidden_dim: int = Field(default=256, ge=1)
    value_head_hidden_dim: int = Field(default=64, ge=1)
    policy_head_hidden_dim: int = Field(default=64, ge=1)
    use_attention: bool = False
    attention_heads: int = Field(default=4, ge=1)
    dropout_rate: float = Field(default=0.0, ge=0, lt=1)
    weight_init: str = "kaiming"
    compile_model: bool = False


class TrainingConfig(BaseModel):
    """Configuration for the training loop.

    Attributes:
        batch_size: Mini-batch size for gradient updates.
        learning_rate: Initial learning rate for the optimiser.
        lr_schedule: Mapping of training step to learning-rate multiplier.
            The multiplier of the nearest preceding key is applied.
        weight_decay: L2 regularisation coefficient.
        max_iterations: Maximum number of training iterations.
        games_per_iteration: Number of self-play games per iteration.
        replay_buffer_size: Maximum number of training examples retained.
        checkpoint_interval: Save a checkpoint every N iterations.
        eval_interval: Evaluate the current model every N iterations.
        eval_games: Number of evaluation games per evaluation round.
        max_steps_per_game: Maximum environment steps before a game is
            terminated.
        num_actors: Number of parallel self-play actors.
        device: Hardware device for tensor computation.
        mixed_precision: Whether to use automatic mixed precision (AMP).
        gradient_accumulation_steps: Number of micro-batches to accumulate
            before a single optimiser step.
    """

    batch_size: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    lr_schedule: dict[int, float] = Field(default_factory=lambda: {0: 1.0})
    weight_decay: float = Field(default=1e-4, ge=0)
    max_iterations: int = Field(default=1000, ge=1)
    games_per_iteration: int = Field(default=25, ge=1)
    replay_buffer_size: int = Field(default=100_000, ge=1)
    checkpoint_interval: int = Field(default=10, ge=1)
    eval_interval: int = Field(default=10, ge=1)
    eval_games: int = Field(default=20, ge=1)
    max_steps_per_game: int = Field(default=200, ge=1)
    num_actors: int = Field(default=1, ge=1)
    device: DeviceType = DeviceType.auto
    mixed_precision: bool = False
    gradient_accumulation_steps: int = Field(default=1, ge=1)


class CurriculumStage(BaseModel):
    """A single stage within a curriculum learning progression.

    Attributes:
        name: Human-readable stage name (e.g. ``"beginner"``).
        environment: Registered environment name to use for this stage.
        min_iterations: Minimum iterations to spend in this stage before
            promotion is considered.
        max_iterations: Maximum iterations in this stage before forced
            promotion.
        baseline_type: Opponent strategy for evaluation
            (e.g. ``"random"``, ``"greedy"``).
    """

    name: str
    environment: str
    min_iterations: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=1000, ge=1)
    baseline_type: str = "random"


class CurriculumConfig(BaseModel):
    """Configuration for curriculum learning across multiple stages.

    Attributes:
        stages: Ordered list of curriculum stages.
        promotion_elo_threshold: Elo rating gain required to advance to
            the next stage.
        promotion_window: Number of recent games used to compute Elo for
            promotion decisions.
        demotion_enabled: Whether to allow demotion to a previous stage
            if performance drops.
        demotion_elo_threshold: Elo rating loss that triggers demotion
            (only when ``demotion_enabled`` is ``True``).
    """

    stages: list[CurriculumStage] = Field(default_factory=list)
    promotion_elo_threshold: float = 100.0
    promotion_window: int = Field(default=50, ge=1)
    demotion_enabled: bool = False
    demotion_elo_threshold: float = -100.0


class LoggingConfig(BaseModel):
    """Configuration for SDK logging and metrics collection.

    Attributes:
        level: Logging level string (e.g. ``"DEBUG"``, ``"INFO"``).
        format_: Output format — ``"pretty"`` for human-readable,
            ``"json"`` for structured logging.  Aliased as ``"format"``
            in YAML / dicts.
        log_dir: Directory for log file output.  ``None`` disables file
            logging.
        log_mcts_trees: Whether to log full MCTS tree dumps (expensive).
        log_gradient_norms: Whether to log per-layer gradient norms.
        log_activation_maps: Whether to log intermediate activation maps.
        metrics_interval: Log aggregated metrics every N steps.
        correlation_id_enabled: Whether to tag log entries with a
            correlation id for distributed tracing.
    """

    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    format_: str = Field(default="pretty", alias="format")
    log_dir: str | None = None
    log_mcts_trees: bool = False
    log_gradient_norms: bool = False
    log_activation_maps: bool = False
    metrics_interval: int = Field(default=10, ge=1)
    correlation_id_enabled: bool = True


# ---------------------------------------------------------------------------
# Top-level SDK configuration
# ---------------------------------------------------------------------------


class SDKConfig(BaseModel):
    """Top-level configuration container for the Strategic Planning SDK.

    Aggregates all sub-configurations into a single object that can be
    loaded from YAML, composed from multiple files, and patched with
    environment variable overrides.

    Attributes:
        mcts: MCTS algorithm settings.
        network: Neural network architecture settings.
        training: Training loop settings.
        curriculum: Curriculum learning settings.
        logging: Logging and metrics settings.
        seed: Global random seed for reproducibility.
        project_name: Human-readable project identifier used in output
            paths, logging, and experiment tracking.
        output_dir: Root directory for all outputs (checkpoints, logs,
            metrics).
    """

    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    seed: int = 42
    project_name: str = "strategic-planning-sdk"
    output_dir: str = "./outputs"

    # -- YAML loading -------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> SDKConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully validated ``SDKConfig`` instance.

        Raises:
            ConfigFileNotFoundError: If *path* does not exist.
            ConfigValidationError: If the YAML content fails validation.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise ConfigFileNotFoundError(
                f"Configuration file not found: {filepath}",
                path=filepath,
            )
        with open(filepath) as fh:
            raw = yaml.safe_load(fh) or {}
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise ConfigValidationError(
                f"Validation failed for {filepath}: {exc}"
            ) from exc

    @classmethod
    def from_yaml_with_overrides(
        cls,
        base_path: str | Path,
        override_path: str | Path,
    ) -> SDKConfig:
        """Load a base YAML config and deep-merge an override file on top.

        Args:
            base_path: Path to the base configuration YAML.
            override_path: Path to the override configuration YAML.

        Returns:
            A fully validated ``SDKConfig`` whose values are the result of
            deep-merging *override_path* into *base_path*.

        Raises:
            ConfigFileNotFoundError: If either path does not exist.
            ConfigValidationError: If the merged content fails validation.
        """
        base_file = Path(base_path)
        override_file = Path(override_path)

        for p in (base_file, override_file):
            if not p.exists():
                raise ConfigFileNotFoundError(
                    f"Configuration file not found: {p}",
                    path=p,
                )

        with open(base_file) as fh:
            base_raw = yaml.safe_load(fh) or {}
        with open(override_file) as fh:
            override_raw = yaml.safe_load(fh) or {}

        merged = _deep_merge(base_raw, override_raw)
        try:
            return cls.model_validate(merged)
        except ValidationError as exc:
            raise ConfigValidationError(
                f"Validation failed after merging {base_file} + "
                f"{override_file}: {exc}"
            ) from exc

    # -- Environment variable overrides -------------------------------------

    def apply_env_overrides(self, prefix: str = SDK_ENV_PREFIX) -> SDKConfig:
        """Create a new config with values overridden from environment variables.

        Environment variables are mapped to nested config fields by
        splitting on ``_`` after stripping the prefix.  For example,
        with the default prefix ``SPSDK_``::

            SPSDK_TRAINING_BATCH_SIZE=512
              -> self.training.batch_size = 512

            SPSDK_MCTS_NUM_SIMULATIONS=400
              -> self.mcts.num_simulations = 400

            SPSDK_SEED=123
              -> self.seed = 123

        The method returns a **new** ``SDKConfig`` instance; the original
        is not mutated.

        Args:
            prefix: Environment variable prefix to scan for.

        Returns:
            A new ``SDKConfig`` with matching environment overrides applied.
        """
        data = self.model_dump(by_alias=False)

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Strip prefix and lowercase: SPSDK_TRAINING_BATCH_SIZE -> training_batch_size
            remainder = env_key[len(prefix) :].lower()
            parts = remainder.split("_")

            if not parts:
                continue

            # Try to navigate the nested config
            # First, check if the first part matches a top-level section
            resolved = False

            # Strategy: try to match the longest top-level key first, then
            # descend.  Top-level sections are single-word keys.
            section_keys = set(data.keys())

            if parts[0] in section_keys:
                section = parts[0]
                field_parts = parts[1:]

                if isinstance(data[section], dict) and field_parts:
                    # Navigate into sub-config; join remaining parts to form
                    # the field name (e.g. batch_size from ["batch", "size"])
                    sub = data[section]
                    field_name = "_".join(field_parts)

                    if field_name in sub:
                        current_value = sub[field_name]
                        try:
                            sub[field_name] = _coerce_value(
                                current_value, env_value
                            )
                            resolved = True
                            logger.debug(
                                "Env override: %s.%s = %r",
                                section,
                                field_name,
                                sub[field_name],
                            )
                        except (ValueError, TypeError) as exc:
                            logger.warning(
                                "Failed to coerce env var %s=%r: %s",
                                env_key,
                                env_value,
                                exc,
                            )
                elif not isinstance(data[section], dict):
                    # Top-level scalar field that happens to share a section
                    # name — unlikely but handle gracefully
                    pass

            # If not resolved via section lookup, try as a top-level field
            # e.g. SPSDK_SEED=123 -> seed
            if not resolved:
                field_name = remainder  # full remainder as field name
                if field_name in data and not isinstance(data[field_name], dict):
                    current_value = data[field_name]
                    try:
                        data[field_name] = _coerce_value(
                            current_value, env_value
                        )
                        logger.debug(
                            "Env override: %s = %r",
                            field_name,
                            data[field_name],
                        )
                    except (ValueError, TypeError) as exc:
                        logger.warning(
                            "Failed to coerce env var %s=%r: %s",
                            env_key,
                            env_value,
                            exc,
                        )

        try:
            return SDKConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigValidationError(
                f"Validation failed after applying env overrides: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> SDKConfig:
    """Load an SDK configuration with optional file, overrides, and env vars.

    This is the recommended entry point for obtaining a configuration
    object.  The loading order is:

    1. If *path* is given, load from YAML; otherwise start from defaults.
    2. Deep-merge *overrides* dict on top.
    3. Apply environment variable overrides (always last).

    Args:
        path: Optional path to a YAML configuration file.
        overrides: Optional dictionary of overrides to deep-merge.  Keys
            should match the nested structure of ``SDKConfig`` (e.g.
            ``{"training": {"batch_size": 512}}``).

    Returns:
        A fully validated ``SDKConfig`` with all overrides applied.

    Raises:
        ConfigFileNotFoundError: If *path* is given but does not exist.
        ConfigValidationError: If the final configuration fails validation.
    """
    # Step 1: base config
    if path is not None:
        filepath = Path(path)
        if not filepath.exists():
            raise ConfigFileNotFoundError(
                f"Configuration file not found: {filepath}",
                path=filepath,
            )
        with open(filepath) as fh:
            raw = yaml.safe_load(fh) or {}
    else:
        raw = {}

    # Step 2: apply programmatic overrides
    if overrides:
        raw = _deep_merge(raw, overrides)

    # Step 3: validate and construct
    try:
        config = SDKConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(
            f"Configuration validation failed: {exc}"
        ) from exc

    # Step 4: apply env var overrides (always last)
    config = config.apply_env_overrides()

    return config
