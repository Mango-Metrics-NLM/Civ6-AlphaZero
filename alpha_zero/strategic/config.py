"""Configuration system for the Strategic Planning SDK.

This module provides a hierarchical configuration system for the strategic
planning SDK, following the same patterns as the Training Orchestrator config.

All values can be:
- Loaded from YAML files
- Loaded from presets (fast, balanced, thorough)
- Overridden via environment variables (STRATEGIC_SDK_ prefix)
- Set programmatically

Usage:
    from alpha_zero.strategic.config import StrategicConfig

    # Load from file
    config = StrategicConfig.from_yaml("configs/strategic.yaml")

    # Load from preset
    config = StrategicConfig.from_preset("balanced")

    # Environment variable overrides
    # STRATEGIC_SDK_PLANNING_NUM_SIMULATIONS=200 will set planning.num_simulations
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PlanningConfig:
    """MCTS planning parameters for strategic search.

    Controls the Monte Carlo Tree Search behavior including exploration
    constants, Dirichlet noise for root exploration, and temperature
    scheduling for action selection.

    Attributes:
        num_simulations: Number of MCTS simulations per planning step.
        c_puct: Exploration constant for PUCT formula.
        dirichlet_alpha: Dirichlet noise alpha parameter for root exploration.
        dirichlet_epsilon: Weight of Dirichlet noise vs. prior policy.
        temperature: Temperature for action selection (higher = more exploration).
        temperature_threshold: Step after which temperature drops to near-zero.
        num_planning_iterations: Number of iterative planning rounds.
        strategies_per_iteration: Number of candidate strategies per iteration.
    """

    num_simulations: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30
    num_planning_iterations: int = 10
    strategies_per_iteration: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanningConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with planning configuration values.

        Returns:
            PlanningConfig instance.
        """
        return cls(
            num_simulations=int(data.get("num_simulations", 100)),
            c_puct=float(data.get("c_puct", 1.5)),
            dirichlet_alpha=float(data.get("dirichlet_alpha", 0.3)),
            dirichlet_epsilon=float(data.get("dirichlet_epsilon", 0.25)),
            temperature=float(data.get("temperature", 1.0)),
            temperature_threshold=int(data.get("temperature_threshold", 30)),
            num_planning_iterations=int(data.get("num_planning_iterations", 10)),
            strategies_per_iteration=int(data.get("strategies_per_iteration", 5)),
        )


@dataclass
class ObjectiveConfig:
    """Configuration for multi-objective optimization.

    Controls how multiple objectives are combined into a scalar value
    for MCTS search, including scalarization method and constraint handling.

    Attributes:
        scalarization_method: Method for combining objectives. One of
            "weighted_sum", "chebyshev", or "epsilon_constraint".
        objective_weights: Weights for each objective. Empty list means
            equal weights will be computed automatically.
        constraint_penalty: Penalty multiplier for constraint violations.
    """

    scalarization_method: str = "weighted_sum"
    objective_weights: list[float] = field(default_factory=list)
    constraint_penalty: float = 10.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectiveConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with objective configuration values.

        Returns:
            ObjectiveConfig instance.
        """
        return cls(
            scalarization_method=data.get("scalarization_method", "weighted_sum"),
            objective_weights=list(data.get("objective_weights", [])),
            constraint_penalty=float(data.get("constraint_penalty", 10.0)),
        )


@dataclass
class ScenarioConfig:
    """Configuration for scenario-based robust planning.

    When enabled, the planner evaluates strategies across multiple
    scenarios to find robust solutions that perform well under
    different possible futures.

    Attributes:
        enabled: Whether scenario analysis is active.
        num_scenarios: Number of scenarios to evaluate.
        robustness_weight: Weight given to worst-case scenario performance
            vs. expected performance (0.0 = pure expected, 1.0 = pure minimax).
    """

    enabled: bool = False
    num_scenarios: int = 5
    robustness_weight: float = 0.3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with scenario configuration values.

        Returns:
            ScenarioConfig instance.
        """
        return cls(
            enabled=data.get("enabled", False),
            num_scenarios=int(data.get("num_scenarios", 5)),
            robustness_weight=float(data.get("robustness_weight", 0.3)),
        )


@dataclass
class StrategicNetworkConfig:
    """Neural network configuration for strategic evaluation.

    Defines architecture parameters for the policy-value network
    used to guide MCTS search. Supports both spatial and vector inputs.

    Attributes:
        spatial_channels: Number of spatial input channels (0 for non-spatial).
        spatial_size: Height/width of spatial input (0 for non-spatial).
        vector_dim: Dimension of the vector (non-spatial) input.
        num_objectives: Number of value heads for multi-objective evaluation.
        num_res_blocks: Number of residual blocks in the backbone.
        num_filters: Number of convolutional filters per layer.
        value_head_hidden: Hidden dimension in the value head MLP.
        policy_head_hidden: Hidden dimension in the policy head MLP.
        dropout: Dropout rate for regularization.
        use_attention: Whether to use multi-head attention in the backbone.
    """

    spatial_channels: int = 0
    spatial_size: int = 0
    vector_dim: int = 64
    num_objectives: int = 1
    num_res_blocks: int = 4
    num_filters: int = 64
    value_head_hidden: int = 128
    policy_head_hidden: int = 128
    dropout: float = 0.0
    use_attention: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategicNetworkConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with network configuration values.

        Returns:
            StrategicNetworkConfig instance.
        """
        return cls(
            spatial_channels=int(data.get("spatial_channels", 0)),
            spatial_size=int(data.get("spatial_size", 0)),
            vector_dim=int(data.get("vector_dim", 64)),
            num_objectives=int(data.get("num_objectives", 1)),
            num_res_blocks=int(data.get("num_res_blocks", 4)),
            num_filters=int(data.get("num_filters", 64)),
            value_head_hidden=int(data.get("value_head_hidden", 128)),
            policy_head_hidden=int(data.get("policy_head_hidden", 128)),
            dropout=float(data.get("dropout", 0.0)),
            use_attention=data.get("use_attention", False),
        )


@dataclass
class ExplainabilityConfig:
    """Configuration for decision trace and explainability features.

    Controls what information is recorded during planning for
    post-hoc analysis and explanation generation.

    Attributes:
        enabled: Whether decision tracing is active.
        record_alternatives: Whether to record alternative actions considered.
        max_alternatives: Maximum number of alternative actions to record per step.
        generate_explanations: Whether to generate natural language explanations
            (requires LLM integration).
        export_format: Format for exporting traces ("json" or "yaml").
    """

    enabled: bool = True
    record_alternatives: bool = True
    max_alternatives: int = 5
    generate_explanations: bool = False
    export_format: str = "json"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExplainabilityConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with explainability configuration values.

        Returns:
            ExplainabilityConfig instance.
        """
        return cls(
            enabled=data.get("enabled", True),
            record_alternatives=data.get("record_alternatives", True),
            max_alternatives=int(data.get("max_alternatives", 5)),
            generate_explanations=data.get("generate_explanations", False),
            export_format=data.get("export_format", "json"),
        )


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: Whether to log to a file.
        log_file: Path to log file (relative to log_dir).
        tensorboard: Whether to enable TensorBoard logging.
        metrics_interval: Steps between metric logging.
    """

    level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "strategic.log"
    tensorboard: bool = False
    metrics_interval: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoggingConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with logging configuration values.

        Returns:
            LoggingConfig instance.
        """
        return cls(
            level=data.get("level", "INFO"),
            log_to_file=data.get("log_to_file", False),
            log_file=data.get("log_file", "strategic.log"),
            tensorboard=data.get("tensorboard", False),
            metrics_interval=int(data.get("metrics_interval", 10)),
        )


@dataclass
class StrategicConfig:
    """Master configuration for the Strategic Planning SDK.

    Combines all sub-configurations and provides convenience methods
    for loading from YAML, presets, and environment variables.

    Usage:
        # From preset
        config = StrategicConfig.from_preset("balanced")

        # From YAML file
        config = StrategicConfig.from_yaml("configs/strategic.yaml")

        # With env overrides (STRATEGIC_SDK_PLANNING_NUM_SIMULATIONS=200)
        config = config.apply_env_overrides()

        # Validate
        warnings = config.validate()

    Attributes:
        mode: Operating mode ("plan", "train", "evaluate", "benchmark").
        experiment_name: Name for this experiment/run.
        random_seed: Random seed for reproducibility (None for random).
        checkpoint_dir: Directory for saving checkpoints.
        log_dir: Directory for log files.
        config_path: Path to the YAML config file this was loaded from.
        planning: MCTS planning parameters.
        objective: Multi-objective optimization settings.
        scenario: Scenario-based planning settings.
        network: Neural network architecture settings.
        explainability: Decision trace and explanation settings.
        logging: Logging and monitoring settings.
    """

    # Top-level settings
    mode: str = "plan"
    experiment_name: str = "strategic_planning_v1"
    random_seed: int | None = None
    checkpoint_dir: str = "checkpoints/strategic"
    log_dir: str = "logs/strategic"
    config_path: str | None = None

    # Sub-configurations
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    network: StrategicNetworkConfig = field(default_factory=StrategicNetworkConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> StrategicConfig:
        """Load configuration from YAML file with env overrides.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Loaded and validated configuration.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        config = cls._from_dict(data)
        config.config_path = str(path)
        config.apply_env_overrides()
        return config

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> StrategicConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            StrategicConfig instance.
        """
        return cls(
            mode=data.get("mode", "plan"),
            experiment_name=data.get("experiment_name", "strategic_planning_v1"),
            random_seed=data.get("random_seed"),
            checkpoint_dir=data.get("checkpoint_dir", "checkpoints/strategic"),
            log_dir=data.get("log_dir", "logs/strategic"),
            planning=PlanningConfig.from_dict(data.get("planning", {})),
            objective=ObjectiveConfig.from_dict(data.get("objective", {})),
            scenario=ScenarioConfig.from_dict(data.get("scenario", {})),
            network=StrategicNetworkConfig.from_dict(data.get("network", {})),
            explainability=ExplainabilityConfig.from_dict(data.get("explainability", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
        )

    @classmethod
    def from_preset(cls, preset: str) -> StrategicConfig:
        """Load from predefined preset.

        Available presets:
        - "fast": Minimal simulations for quick testing and CI.
        - "balanced": Moderate settings for initial experiments.
        - "thorough": High-quality planning with full analysis.

        Args:
            preset: One of "fast", "balanced", "thorough".

        Returns:
            Configuration for the specified preset.

        Raises:
            ValueError: If the preset name is not recognized.
        """
        presets = {
            "fast": cls._fast_preset(),
            "balanced": cls._balanced_preset(),
            "thorough": cls._thorough_preset(),
        }
        if preset not in presets:
            raise ValueError(
                f"Unknown preset: '{preset}'. Choose from: {list(presets.keys())}"
            )
        config = presets[preset]
        config.apply_env_overrides()
        return config

    @classmethod
    def _fast_preset(cls) -> StrategicConfig:
        """Fast preset for quick testing and CI."""
        config = cls()
        config.experiment_name = "fast_test"
        # Minimal MCTS simulations
        config.planning.num_simulations = 10
        config.planning.num_planning_iterations = 2
        config.planning.strategies_per_iteration = 2
        # Smaller network
        config.network.num_res_blocks = 2
        config.network.num_filters = 32
        config.network.value_head_hidden = 64
        config.network.policy_head_hidden = 64
        # Disable expensive features
        config.scenario.enabled = False
        config.explainability.generate_explanations = False
        config.explainability.record_alternatives = False
        config.logging.tensorboard = False
        return config

    @classmethod
    def _balanced_preset(cls) -> StrategicConfig:
        """Balanced preset for initial experiments."""
        config = cls()
        config.experiment_name = "balanced_planning"
        # Moderate MCTS
        config.planning.num_simulations = 100
        config.planning.num_planning_iterations = 10
        config.planning.strategies_per_iteration = 5
        # Standard network
        config.network.num_res_blocks = 4
        config.network.num_filters = 64
        # Enable scenario analysis with moderate settings
        config.scenario.enabled = True
        config.scenario.num_scenarios = 5
        # Enable traces but not LLM explanations
        config.explainability.enabled = True
        config.explainability.record_alternatives = True
        config.explainability.generate_explanations = False
        return config

    @classmethod
    def _thorough_preset(cls) -> StrategicConfig:
        """Thorough preset for high-quality planning."""
        config = cls()
        config.experiment_name = "thorough_planning"
        # Deep MCTS search
        config.planning.num_simulations = 800
        config.planning.c_puct = 2.0
        config.planning.num_planning_iterations = 50
        config.planning.strategies_per_iteration = 10
        # Larger network
        config.network.num_res_blocks = 8
        config.network.num_filters = 128
        config.network.value_head_hidden = 256
        config.network.policy_head_hidden = 256
        config.network.use_attention = True
        # Full scenario analysis
        config.scenario.enabled = True
        config.scenario.num_scenarios = 20
        config.scenario.robustness_weight = 0.5
        # Full explainability
        config.explainability.enabled = True
        config.explainability.record_alternatives = True
        config.explainability.max_alternatives = 10
        config.explainability.generate_explanations = True
        # Detailed logging
        config.logging.tensorboard = True
        config.logging.log_to_file = True
        config.logging.metrics_interval = 1
        return config

    def apply_env_overrides(self) -> StrategicConfig:
        """Apply environment variable overrides with STRATEGIC_SDK_ prefix.

        Environment variables follow the pattern:
        STRATEGIC_SDK_<SECTION>_<KEY>=value

        Examples:
        - STRATEGIC_SDK_PLANNING_NUM_SIMULATIONS=200
        - STRATEGIC_SDK_OBJECTIVE_SCALARIZATION_METHOD=chebyshev
        - STRATEGIC_SDK_SCENARIO_ENABLED=true
        - STRATEGIC_SDK_NETWORK_NUM_RES_BLOCKS=8

        Returns:
            Self for method chaining.
        """
        prefix = "STRATEGIC_SDK_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            parts = key[len(prefix):].lower().split("_", 1)
            if len(parts) != 2:
                continue

            section, attr = parts
            self._apply_override(section, attr, value)

        return self

    def _apply_override(self, section: str, attr: str, value: str) -> None:
        """Apply a single environment variable override.

        Args:
            section: Configuration section name (e.g. "planning").
            attr: Attribute name within the section (e.g. "num_simulations").
            value: String value from the environment variable.
        """
        section_map: dict[str, Any] = {
            "planning": self.planning,
            "objective": self.objective,
            "scenario": self.scenario,
            "network": self.network,
            "explainability": self.explainability,
            "logging": self.logging,
        }

        if section not in section_map:
            return

        target = section_map[section]
        if not hasattr(target, attr):
            return

        current_value = getattr(target, attr)
        try:
            if isinstance(current_value, bool):
                setattr(target, attr, value.lower() in ("true", "1", "yes"))
            elif isinstance(current_value, int):
                setattr(target, attr, int(value))
            elif isinstance(current_value, float):
                setattr(target, attr, float(value))
            elif isinstance(current_value, str):
                setattr(target, attr, value)
            else:
                logger.warning(
                    "Cannot override %s.%s: unsupported type %s",
                    section,
                    attr,
                    type(current_value).__name__,
                )
        except (ValueError, TypeError) as e:
            logger.warning(
                "Failed to apply env override %s.%s=%s: %s",
                section,
                attr,
                value,
                e,
            )

    def validate(self) -> list[str]:
        """Validate configuration, return list of warnings.

        Checks for potentially problematic settings and returns
        human-readable warning messages.

        Returns:
            List of warning messages. Empty list means no issues found.
        """
        warnings: list[str] = []

        # Validate mode
        valid_modes = ("plan", "train", "evaluate", "benchmark")
        if self.mode not in valid_modes:
            warnings.append(
                f"Unknown mode '{self.mode}'. Valid modes: {valid_modes}"
            )

        # Check MCTS simulations
        if self.planning.num_simulations < 1:
            warnings.append(
                f"MCTS simulations ({self.planning.num_simulations}) must be >= 1"
            )
        elif self.planning.num_simulations < 10:
            warnings.append(
                f"MCTS simulations ({self.planning.num_simulations}) is very low; "
                f"results may be unreliable"
            )

        # Check exploration parameters
        if self.planning.c_puct <= 0:
            warnings.append(
                f"c_puct ({self.planning.c_puct}) must be positive"
            )

        if not 0.0 <= self.planning.dirichlet_epsilon <= 1.0:
            warnings.append(
                f"dirichlet_epsilon ({self.planning.dirichlet_epsilon}) "
                f"should be in [0.0, 1.0]"
            )

        if self.planning.dirichlet_alpha <= 0:
            warnings.append(
                f"dirichlet_alpha ({self.planning.dirichlet_alpha}) must be positive"
            )

        # Check temperature
        if self.planning.temperature < 0:
            warnings.append(
                f"temperature ({self.planning.temperature}) must be non-negative"
            )

        # Check objective config
        valid_scalarizations = ("weighted_sum", "chebyshev", "epsilon_constraint")
        if self.objective.scalarization_method not in valid_scalarizations:
            warnings.append(
                f"Unknown scalarization method '{self.objective.scalarization_method}'. "
                f"Valid methods: {valid_scalarizations}"
            )

        if self.objective.constraint_penalty < 0:
            warnings.append(
                f"constraint_penalty ({self.objective.constraint_penalty}) "
                f"should be non-negative"
            )

        # Check objective weights match num_objectives
        if (
            self.objective.objective_weights
            and self.network.num_objectives > 0
            and len(self.objective.objective_weights) != self.network.num_objectives
        ):
            warnings.append(
                f"objective_weights length ({len(self.objective.objective_weights)}) "
                f"does not match num_objectives ({self.network.num_objectives})"
            )

        # Check scenario config
        if self.scenario.enabled and self.scenario.num_scenarios < 1:
            warnings.append(
                f"Scenario analysis enabled but num_scenarios "
                f"({self.scenario.num_scenarios}) must be >= 1"
            )

        if not 0.0 <= self.scenario.robustness_weight <= 1.0:
            warnings.append(
                f"robustness_weight ({self.scenario.robustness_weight}) "
                f"should be in [0.0, 1.0]"
            )

        # Check network config
        if self.network.num_res_blocks < 1:
            warnings.append(
                f"num_res_blocks ({self.network.num_res_blocks}) must be >= 1"
            )

        if self.network.num_filters < 1:
            warnings.append(
                f"num_filters ({self.network.num_filters}) must be >= 1"
            )

        if not 0.0 <= self.network.dropout < 1.0:
            warnings.append(
                f"dropout ({self.network.dropout}) should be in [0.0, 1.0)"
            )

        # Check explainability config
        valid_formats = ("json", "yaml")
        if self.explainability.export_format not in valid_formats:
            warnings.append(
                f"Unknown export_format '{self.explainability.export_format}'. "
                f"Valid formats: {valid_formats}"
            )

        if self.explainability.max_alternatives < 0:
            warnings.append(
                f"max_alternatives ({self.explainability.max_alternatives}) "
                f"must be non-negative"
            )

        return warnings

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the full configuration.
        """
        return {
            "mode": self.mode,
            "experiment_name": self.experiment_name,
            "random_seed": self.random_seed,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "planning": {
                "num_simulations": self.planning.num_simulations,
                "c_puct": self.planning.c_puct,
                "dirichlet_alpha": self.planning.dirichlet_alpha,
                "dirichlet_epsilon": self.planning.dirichlet_epsilon,
                "temperature": self.planning.temperature,
                "temperature_threshold": self.planning.temperature_threshold,
                "num_planning_iterations": self.planning.num_planning_iterations,
                "strategies_per_iteration": self.planning.strategies_per_iteration,
            },
            "objective": {
                "scalarization_method": self.objective.scalarization_method,
                "objective_weights": self.objective.objective_weights,
                "constraint_penalty": self.objective.constraint_penalty,
            },
            "scenario": {
                "enabled": self.scenario.enabled,
                "num_scenarios": self.scenario.num_scenarios,
                "robustness_weight": self.scenario.robustness_weight,
            },
            "network": {
                "spatial_channels": self.network.spatial_channels,
                "spatial_size": self.network.spatial_size,
                "vector_dim": self.network.vector_dim,
                "num_objectives": self.network.num_objectives,
                "num_res_blocks": self.network.num_res_blocks,
                "num_filters": self.network.num_filters,
                "value_head_hidden": self.network.value_head_hidden,
                "policy_head_hidden": self.network.policy_head_hidden,
                "dropout": self.network.dropout,
                "use_attention": self.network.use_attention,
            },
            "explainability": {
                "enabled": self.explainability.enabled,
                "record_alternatives": self.explainability.record_alternatives,
                "max_alternatives": self.explainability.max_alternatives,
                "generate_explanations": self.explainability.generate_explanations,
                "export_format": self.explainability.export_format,
            },
            "logging": {
                "level": self.logging.level,
                "log_to_file": self.logging.log_to_file,
                "log_file": self.logging.log_file,
                "tensorboard": self.logging.tensorboard,
                "metrics_interval": self.logging.metrics_interval,
            },
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save the YAML file. Parent directories
                will be created if they do not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info("Configuration saved to %s", path)
