"""
Configuration for the Hierarchical Reasoning System.

This module defines configuration dataclasses for all hierarchical components,
following the pattern established by the orchestrator configuration system.

Example:
    >>> from alpha_zero.hierarchical.config import HierarchicalConfig
    >>> config = HierarchicalConfig.from_yaml("configs/hierarchical.yaml")
    >>> config = config.apply_env_overrides()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


class HierarchyLevel(Enum):
    """Levels in the hierarchical decision system."""

    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"


@dataclass
class MCTSLevelConfig:
    """MCTS configuration for a specific hierarchy level.

    Attributes:
        num_simulations: Number of MCTS simulations per search.
        c_puct: Exploration constant for UCB.
        horizon: Planning horizon in turns.
        temperature: Temperature for action selection.
        dirichlet_alpha: Dirichlet noise alpha for exploration.
        dirichlet_epsilon: Weight of Dirichlet noise at root.
    """

    num_simulations: int = 100
    c_puct: float = 1.5
    horizon: int = 10
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Level-specific defaults
    STRATEGIC_DEFAULTS: ClassVar[dict[str, Any]] = {
        "num_simulations": 20,
        "c_puct": 2.0,
        "horizon": 50,
        "temperature": 1.2,
    }

    TACTICAL_DEFAULTS: ClassVar[dict[str, Any]] = {
        "num_simulations": 50,
        "c_puct": 1.5,
        "horizon": 15,
        "temperature": 1.0,
    }

    OPERATIONAL_DEFAULTS: ClassVar[dict[str, Any]] = {
        "num_simulations": 200,
        "c_puct": 1.5,
        "horizon": 5,
        "temperature": 1.0,
    }

    @classmethod
    def for_level(cls, level: HierarchyLevel) -> MCTSLevelConfig:
        """Create config with level-specific defaults.

        Args:
            level: Hierarchy level.

        Returns:
            MCTSLevelConfig with appropriate defaults.
        """
        defaults_map = {
            HierarchyLevel.STRATEGIC: cls.STRATEGIC_DEFAULTS,
            HierarchyLevel.TACTICAL: cls.TACTICAL_DEFAULTS,
            HierarchyLevel.OPERATIONAL: cls.OPERATIONAL_DEFAULTS,
        }
        return cls(**defaults_map.get(level, {}))


@dataclass
class GoalConfig:
    """Configuration for goal management.

    Attributes:
        max_active_goals: Maximum active goals per level.
        goal_timeout_turns: Turns before goal expires.
        progress_threshold: Progress needed to consider goal partially complete.
        abandon_threshold: Turns without progress before abandoning.
        reward_shaping_weight: Weight for goal-based reward shaping.
    """

    max_active_goals: int = 5
    goal_timeout_turns: int = 100
    progress_threshold: float = 0.1
    abandon_threshold: int = 20
    reward_shaping_weight: float = 0.1

    # Goal type specific weights
    strategic_value_weight: float = 1.0
    tactical_value_weight: float = 0.5
    operational_value_weight: float = 0.2


@dataclass
class LevelConfig:
    """Configuration for a single hierarchy level.

    Attributes:
        enabled: Whether this level is active.
        update_frequency: How often to update decisions (in turns).
        mcts: MCTS configuration for this level.
        network_embed_dim: Goal embedding dimension for network.
        state_abstract_dim: Abstracted state dimension.
    """

    enabled: bool = True
    update_frequency: int = 1
    mcts: MCTSLevelConfig = field(default_factory=MCTSLevelConfig)
    network_embed_dim: int = 64
    state_abstract_dim: int = 128

    @classmethod
    def for_level(cls, level: HierarchyLevel) -> LevelConfig:
        """Create config with level-specific defaults.

        Args:
            level: Hierarchy level.

        Returns:
            LevelConfig with appropriate defaults.
        """
        configs = {
            HierarchyLevel.STRATEGIC: cls(
                update_frequency=10,
                mcts=MCTSLevelConfig.for_level(level),
                network_embed_dim=128,
                state_abstract_dim=256,
            ),
            HierarchyLevel.TACTICAL: cls(
                update_frequency=3,
                mcts=MCTSLevelConfig.for_level(level),
                network_embed_dim=64,
                state_abstract_dim=128,
            ),
            HierarchyLevel.OPERATIONAL: cls(
                update_frequency=1,
                mcts=MCTSLevelConfig.for_level(level),
                network_embed_dim=32,
                state_abstract_dim=64,
            ),
        }
        return configs.get(level, cls())


@dataclass
class LLMGuidanceConfig:
    """Configuration for LLM-based strategic guidance.

    Attributes:
        enabled: Whether to use LLM guidance.
        analysis_interval: Games between LLM analyses.
        query_budget: Max LLM queries per training run.
        temperature: LLM generation temperature.
        use_rag_context: Whether to augment with RAG retrieval.
        rag_top_k: Number of RAG results to include.
    """

    enabled: bool = True
    analysis_interval: int = 100
    query_budget: int = 1000
    temperature: float = 0.7
    use_rag_context: bool = True
    rag_top_k: int = 5


@dataclass
class HierarchicalConfig:
    """Master configuration for the hierarchical reasoning system.

    Attributes:
        enabled: Whether hierarchical reasoning is active.
        strategic: Strategic level configuration.
        tactical: Tactical level configuration.
        operational: Operational level configuration.
        goals: Goal management configuration.
        llm_guidance: LLM guidance configuration.
        debug: Enable debug logging and visualization.
    """

    enabled: bool = True

    # Level configurations
    strategic: LevelConfig = field(
        default_factory=lambda: LevelConfig.for_level(HierarchyLevel.STRATEGIC)
    )
    tactical: LevelConfig = field(
        default_factory=lambda: LevelConfig.for_level(HierarchyLevel.TACTICAL)
    )
    operational: LevelConfig = field(
        default_factory=lambda: LevelConfig.for_level(HierarchyLevel.OPERATIONAL)
    )

    # Shared configurations
    goals: GoalConfig = field(default_factory=GoalConfig)
    llm_guidance: LLMGuidanceConfig = field(default_factory=LLMGuidanceConfig)

    # Debug and logging
    debug: bool = False
    log_goal_changes: bool = True
    log_level_decisions: bool = False

    # Environment variable prefix for overrides
    ENV_PREFIX: ClassVar[str] = "CIV6AZ_HIERARCHICAL_"

    def get_level_config(self, level: HierarchyLevel) -> LevelConfig:
        """Get configuration for a specific level.

        Args:
            level: Hierarchy level.

        Returns:
            LevelConfig for the level.
        """
        level_map = {
            HierarchyLevel.STRATEGIC: self.strategic,
            HierarchyLevel.TACTICAL: self.tactical,
            HierarchyLevel.OPERATIONAL: self.operational,
        }
        return level_map[level]

    def is_level_enabled(self, level: HierarchyLevel) -> bool:
        """Check if a hierarchy level is enabled.

        Args:
            level: Hierarchy level.

        Returns:
            True if level is enabled.
        """
        if not self.enabled:
            return False
        return self.get_level_config(level).enabled

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HierarchicalConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            HierarchicalConfig instance.
        """
        # Parse nested configurations
        if "strategic" in data:
            if "mcts" in data["strategic"]:
                data["strategic"]["mcts"] = MCTSLevelConfig(**data["strategic"]["mcts"])
            data["strategic"] = LevelConfig(**data["strategic"])

        if "tactical" in data:
            if "mcts" in data["tactical"]:
                data["tactical"]["mcts"] = MCTSLevelConfig(**data["tactical"]["mcts"])
            data["tactical"] = LevelConfig(**data["tactical"])

        if "operational" in data:
            if "mcts" in data["operational"]:
                data["operational"]["mcts"] = MCTSLevelConfig(**data["operational"]["mcts"])
            data["operational"] = LevelConfig(**data["operational"])

        if "goals" in data:
            data["goals"] = GoalConfig(**data["goals"])

        if "llm_guidance" in data:
            data["llm_guidance"] = LLMGuidanceConfig(**data["llm_guidance"])

        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> HierarchicalConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            HierarchicalConfig instance.

        Raises:
            ImportError: If PyYAML not installed.
            FileNotFoundError: If file doesn't exist.
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for YAML config loading")

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        # Extract hierarchical section if present
        if "hierarchical" in data:
            data = data["hierarchical"]

        return cls.from_dict(data)

    def apply_env_overrides(self) -> HierarchicalConfig:
        """Apply environment variable overrides.

        Environment variables should be prefixed with CIV6AZ_HIERARCHICAL_
        and use double underscores for nested keys.

        Example:
            CIV6AZ_HIERARCHICAL_STRATEGIC__MCTS__NUM_SIMULATIONS=50

        Returns:
            Self with overrides applied.
        """
        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue

            # Parse key path
            config_key = key[len(self.ENV_PREFIX) :].lower()
            parts = config_key.split("__")

            # Apply override
            try:
                self._apply_override(parts, value)
            except (AttributeError, ValueError) as e:
                logger.warning(f"Failed to apply env override {key}: {e}")

        return self

    def _apply_override(self, parts: list[str], value: str) -> None:
        """Apply a single environment override.

        Args:
            parts: Key path parts.
            value: Value to set.
        """
        obj: Any = self
        for part in parts[:-1]:
            obj = getattr(obj, part)

        # Parse value
        final_key = parts[-1]
        current_value = getattr(obj, final_key)

        if isinstance(current_value, bool):
            parsed_value = value.lower() in ("true", "1", "yes", "on")
        elif isinstance(current_value, int):
            parsed_value = int(value)
        elif isinstance(current_value, float):
            parsed_value = float(value)
        else:
            parsed_value = value

        setattr(obj, final_key, parsed_value)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as nested dictionary.
        """
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def default(cls) -> HierarchicalConfig:
        """Create configuration with all defaults.

        Returns:
            Default HierarchicalConfig.
        """
        return cls()

    @classmethod
    def minimal(cls) -> HierarchicalConfig:
        """Create minimal configuration (operational only).

        Returns:
            HierarchicalConfig with only operational level.
        """
        return cls(
            strategic=LevelConfig(enabled=False),
            tactical=LevelConfig(enabled=False),
            operational=LevelConfig.for_level(HierarchyLevel.OPERATIONAL),
        )

    @classmethod
    def full(cls) -> HierarchicalConfig:
        """Create full configuration with all levels enabled.

        Returns:
            HierarchicalConfig with all levels enabled.
        """
        return cls(
            enabled=True,
            strategic=LevelConfig.for_level(HierarchyLevel.STRATEGIC),
            tactical=LevelConfig.for_level(HierarchyLevel.TACTICAL),
            operational=LevelConfig.for_level(HierarchyLevel.OPERATIONAL),
            llm_guidance=LLMGuidanceConfig(enabled=True),
        )
