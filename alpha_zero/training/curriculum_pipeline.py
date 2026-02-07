"""Multi-Environment Curriculum Learning Pipeline.

This module implements a comprehensive curriculum learning pipeline that
supports staged training across multiple environments with transfer learning:

MicroCiv (Simple) → Unciv (Intermediate) → Civ6 (Full)

Features:
- Environment-level curriculum progression
- Intra-environment difficulty staging
- Network weight transfer between environments
- Performance-based promotion criteria
- Configurable via dataclasses
- Checkpoint management for curriculum state

Example usage:
    >>> from alpha_zero.training.curriculum_pipeline import (
    ...     CurriculumPipeline,
    ...     CurriculumPipelineConfig,
    ...     EnvironmentTier,
    ... )
    >>> config = CurriculumPipelineConfig.default()
    >>> pipeline = CurriculumPipeline(config)
    >>> env = pipeline.get_current_environment()
    >>> network = pipeline.get_network_for_environment()
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol, TypeVar

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T")


class NetworkProtocol(Protocol):
    """Protocol for neural network compatibility."""

    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


class EnvironmentProtocol(Protocol):
    """Protocol for environment compatibility."""

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]: ...
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...


# =============================================================================
# Environment Tiers
# =============================================================================


class EnvironmentTier(Enum):
    """Environment complexity tiers in the curriculum.

    Each tier represents a level of game complexity:
    - MICROCIV: Simplified 4X game for rapid iteration
    - UNCIV: Open-source Civ5 clone with full mechanics
    - CIV6: Full Civilization 6 game
    """

    MICROCIV = auto()
    UNCIV = auto()
    CIV6 = auto()

    @property
    def complexity_score(self) -> int:
        """Get relative complexity score (higher = more complex)."""
        scores = {
            EnvironmentTier.MICROCIV: 1,
            EnvironmentTier.UNCIV: 5,
            EnvironmentTier.CIV6: 10,
        }
        return scores[self]

    def next_tier(self) -> EnvironmentTier | None:
        """Get the next tier in the curriculum."""
        order = [EnvironmentTier.MICROCIV, EnvironmentTier.UNCIV, EnvironmentTier.CIV6]
        try:
            idx = order.index(self)
            if idx < len(order) - 1:
                return order[idx + 1]
        except ValueError:
            pass
        return None

    def previous_tier(self) -> EnvironmentTier | None:
        """Get the previous tier in the curriculum."""
        order = [EnvironmentTier.MICROCIV, EnvironmentTier.UNCIV, EnvironmentTier.CIV6]
        try:
            idx = order.index(self)
            if idx > 0:
                return order[idx - 1]
        except ValueError:
            pass
        return None


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class DifficultyStage:
    """Configuration for a difficulty stage within an environment tier.

    Attributes:
        name: Human-readable stage name.
        level: Numeric difficulty level (higher = harder).
        board_size: Board/map size for this stage.
        max_turns: Maximum turns per game.
        num_opponents: Number of AI opponents.
        starting_resources: Initial resource multiplier.
        promotion_win_rate: Win rate required to advance.
        demotion_win_rate: Win rate causing demotion.
        min_games: Minimum games before stage change.
        extra_params: Additional environment-specific parameters.
    """

    name: str
    level: int = 0
    board_size: int = 8
    max_turns: int = 50
    num_opponents: int = 1
    starting_resources: float = 1.0
    promotion_win_rate: float = 0.6
    demotion_win_rate: float = 0.3
    min_games: int = 50
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "level": self.level,
            "board_size": self.board_size,
            "max_turns": self.max_turns,
            "num_opponents": self.num_opponents,
            "starting_resources": self.starting_resources,
            "promotion_win_rate": self.promotion_win_rate,
            "demotion_win_rate": self.demotion_win_rate,
            "min_games": self.min_games,
            "extra_params": self.extra_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DifficultyStage:
        """Create from dictionary."""
        return cls(
            name=data.get("name", "default"),
            level=data.get("level", 0),
            board_size=data.get("board_size", 8),
            max_turns=data.get("max_turns", 50),
            num_opponents=data.get("num_opponents", 1),
            starting_resources=data.get("starting_resources", 1.0),
            promotion_win_rate=data.get("promotion_win_rate", 0.6),
            demotion_win_rate=data.get("demotion_win_rate", 0.3),
            min_games=data.get("min_games", 50),
            extra_params=data.get("extra_params", {}),
        )


@dataclass
class EnvironmentTierConfig:
    """Configuration for an environment tier in the curriculum.

    Attributes:
        tier: The environment tier.
        enabled: Whether this tier is enabled.
        difficulty_stages: Stages within this tier.
        network_type: Type of network to use.
        state_channels: Number of spatial state channels.
        vector_size: Size of vector features.
        action_space_size: Size of action space.
        promotion_criteria: Criteria to promote to next tier.
        demotion_criteria: Criteria to demote to previous tier.
        weight_transfer_strategy: How to transfer weights from previous tier.
    """

    tier: EnvironmentTier
    enabled: bool = True
    difficulty_stages: list[DifficultyStage] = field(default_factory=list)
    network_type: str = "resnet"
    state_channels: int = 13
    vector_size: int = 64
    action_space_size: int = 356
    promotion_criteria: dict[str, float] = field(default_factory=dict)
    demotion_criteria: dict[str, float] = field(default_factory=dict)
    weight_transfer_strategy: str = "partial"

    def __post_init__(self) -> None:
        """Initialize with default stages if none provided."""
        if not self.difficulty_stages:
            self.difficulty_stages = self._default_stages()

        if not self.promotion_criteria:
            self.promotion_criteria = {
                "min_games": 500,
                "win_rate": 0.65,
                "elo_rating": 1500,
            }

        if not self.demotion_criteria:
            self.demotion_criteria = {
                "min_games": 100,
                "win_rate": 0.2,
            }

    def _default_stages(self) -> list[DifficultyStage]:
        """Get default difficulty stages for this tier."""
        if self.tier == EnvironmentTier.MICROCIV:
            return [
                DifficultyStage(
                    name="microciv_tutorial",
                    level=0,
                    board_size=6,
                    max_turns=20,
                    promotion_win_rate=0.8,
                ),
                DifficultyStage(
                    name="microciv_beginner",
                    level=1,
                    board_size=8,
                    max_turns=40,
                    promotion_win_rate=0.7,
                ),
                DifficultyStage(
                    name="microciv_intermediate",
                    level=2,
                    board_size=12,
                    max_turns=80,
                    promotion_win_rate=0.6,
                ),
                DifficultyStage(
                    name="microciv_advanced",
                    level=3,
                    board_size=16,
                    max_turns=150,
                    promotion_win_rate=0.55,
                ),
            ]
        elif self.tier == EnvironmentTier.UNCIV:
            return [
                DifficultyStage(
                    name="unciv_tiny",
                    level=0,
                    board_size=20,
                    max_turns=100,
                    num_opponents=1,
                    promotion_win_rate=0.65,
                ),
                DifficultyStage(
                    name="unciv_small",
                    level=1,
                    board_size=30,
                    max_turns=200,
                    num_opponents=2,
                    promotion_win_rate=0.6,
                ),
                DifficultyStage(
                    name="unciv_standard",
                    level=2,
                    board_size=40,
                    max_turns=300,
                    num_opponents=3,
                    promotion_win_rate=0.55,
                ),
                DifficultyStage(
                    name="unciv_large",
                    level=3,
                    board_size=50,
                    max_turns=400,
                    num_opponents=5,
                    promotion_win_rate=0.5,
                ),
            ]
        elif self.tier == EnvironmentTier.CIV6:
            return [
                DifficultyStage(
                    name="civ6_settler",
                    level=0,
                    board_size=44,
                    max_turns=300,
                    num_opponents=2,
                    extra_params={"difficulty": "Settler"},
                ),
                DifficultyStage(
                    name="civ6_chieftain",
                    level=1,
                    board_size=54,
                    max_turns=400,
                    num_opponents=4,
                    extra_params={"difficulty": "Chieftain"},
                ),
                DifficultyStage(
                    name="civ6_prince",
                    level=2,
                    board_size=64,
                    max_turns=500,
                    num_opponents=5,
                    extra_params={"difficulty": "Prince"},
                ),
                DifficultyStage(
                    name="civ6_king",
                    level=3,
                    board_size=74,
                    max_turns=500,
                    num_opponents=6,
                    extra_params={"difficulty": "King"},
                ),
                DifficultyStage(
                    name="civ6_deity",
                    level=4,
                    board_size=84,
                    max_turns=500,
                    num_opponents=7,
                    extra_params={"difficulty": "Deity"},
                ),
            ]
        return [DifficultyStage(name="default")]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tier": self.tier.name,
            "enabled": self.enabled,
            "difficulty_stages": [s.to_dict() for s in self.difficulty_stages],
            "network_type": self.network_type,
            "state_channels": self.state_channels,
            "vector_size": self.vector_size,
            "action_space_size": self.action_space_size,
            "promotion_criteria": self.promotion_criteria,
            "demotion_criteria": self.demotion_criteria,
            "weight_transfer_strategy": self.weight_transfer_strategy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentTierConfig:
        """Create from dictionary."""
        tier = EnvironmentTier[data.get("tier", "MICROCIV")]
        stages = [DifficultyStage.from_dict(s) for s in data.get("difficulty_stages", [])]
        return cls(
            tier=tier,
            enabled=data.get("enabled", True),
            difficulty_stages=stages,
            network_type=data.get("network_type", "resnet"),
            state_channels=data.get("state_channels", 13),
            vector_size=data.get("vector_size", 64),
            action_space_size=data.get("action_space_size", 356),
            promotion_criteria=data.get("promotion_criteria", {}),
            demotion_criteria=data.get("demotion_criteria", {}),
            weight_transfer_strategy=data.get("weight_transfer_strategy", "partial"),
        )


@dataclass
class TransferLearningConfig:
    """Configuration for weight transfer between environment tiers.

    Attributes:
        strategy: Transfer strategy ('none', 'partial', 'full', 'progressive').
        freeze_layers: Layers to freeze after transfer.
        learning_rate_multiplier: LR multiplier for transferred layers.
        feature_alignment: Whether to align feature dimensions.
        reinitialize_heads: Whether to reinitialize policy/value heads.
        warmup_games: Games to play before unfreezing layers.
    """

    strategy: str = "partial"
    freeze_layers: list[str] = field(default_factory=list)
    learning_rate_multiplier: float = 0.1
    feature_alignment: bool = True
    reinitialize_heads: bool = True
    warmup_games: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy,
            "freeze_layers": self.freeze_layers,
            "learning_rate_multiplier": self.learning_rate_multiplier,
            "feature_alignment": self.feature_alignment,
            "reinitialize_heads": self.reinitialize_heads,
            "warmup_games": self.warmup_games,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransferLearningConfig:
        """Create from dictionary."""
        return cls(
            strategy=data.get("strategy", "partial"),
            freeze_layers=data.get("freeze_layers", []),
            learning_rate_multiplier=data.get("learning_rate_multiplier", 0.1),
            feature_alignment=data.get("feature_alignment", True),
            reinitialize_heads=data.get("reinitialize_heads", True),
            warmup_games=data.get("warmup_games", 100),
        )


@dataclass
class CurriculumPipelineConfig:
    """Complete configuration for the curriculum pipeline.

    Attributes:
        enabled: Whether curriculum learning is active.
        environment_tiers: Configuration for each environment tier.
        transfer_learning: Transfer learning configuration.
        allow_tier_demotion: Whether to allow demotion between tiers.
        metrics_window: Number of games for metrics calculation.
        checkpoint_dir: Directory for curriculum checkpoints.
        auto_save: Whether to auto-save state after changes.
    """

    enabled: bool = True
    environment_tiers: list[EnvironmentTierConfig] = field(default_factory=list)
    transfer_learning: TransferLearningConfig = field(
        default_factory=TransferLearningConfig,
    )
    allow_tier_demotion: bool = False
    metrics_window: int = 100
    checkpoint_dir: str = "checkpoints/curriculum"
    auto_save: bool = True

    def __post_init__(self) -> None:
        """Initialize with default tiers if none provided."""
        if not self.environment_tiers:
            self.environment_tiers = [
                EnvironmentTierConfig(
                    tier=EnvironmentTier.MICROCIV,
                    state_channels=13,
                    vector_size=64,
                    action_space_size=356,
                ),
                EnvironmentTierConfig(
                    tier=EnvironmentTier.UNCIV,
                    state_channels=48,
                    vector_size=256,
                    action_space_size=10000,
                ),
                EnvironmentTierConfig(
                    tier=EnvironmentTier.CIV6,
                    state_channels=84,
                    vector_size=512,
                    action_space_size=50000,
                ),
            ]

    @classmethod
    def default(cls) -> CurriculumPipelineConfig:
        """Create default curriculum pipeline configuration."""
        return cls(enabled=True)

    @classmethod
    def fast_iteration(cls) -> CurriculumPipelineConfig:
        """Create fast iteration config (smaller games, quicker promotion)."""
        config = cls(enabled=True)
        for tier_config in config.environment_tiers:
            for stage in tier_config.difficulty_stages:
                stage.min_games = 20
                stage.promotion_win_rate = 0.5
                stage.max_turns = min(stage.max_turns, 50)
            tier_config.promotion_criteria["min_games"] = 100
            tier_config.promotion_criteria["win_rate"] = 0.5
        return config

    @classmethod
    def production(cls) -> CurriculumPipelineConfig:
        """Create production config (standard settings)."""
        return cls(enabled=True, allow_tier_demotion=True)

    def get_tier_config(self, tier: EnvironmentTier) -> EnvironmentTierConfig | None:
        """Get configuration for a specific tier."""
        for tier_config in self.environment_tiers:
            if tier_config.tier == tier and tier_config.enabled:
                return tier_config
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "environment_tiers": [t.to_dict() for t in self.environment_tiers],
            "transfer_learning": self.transfer_learning.to_dict(),
            "allow_tier_demotion": self.allow_tier_demotion,
            "metrics_window": self.metrics_window,
            "checkpoint_dir": self.checkpoint_dir,
            "auto_save": self.auto_save,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumPipelineConfig:
        """Create from dictionary."""
        tiers = [EnvironmentTierConfig.from_dict(t) for t in data.get("environment_tiers", [])]
        transfer = TransferLearningConfig.from_dict(
            data.get("transfer_learning", {}),
        )
        return cls(
            enabled=data.get("enabled", True),
            environment_tiers=tiers,
            transfer_learning=transfer,
            allow_tier_demotion=data.get("allow_tier_demotion", False),
            metrics_window=data.get("metrics_window", 100),
            checkpoint_dir=data.get("checkpoint_dir", "checkpoints/curriculum"),
            auto_save=data.get("auto_save", True),
        )

    @classmethod
    def from_yaml(cls, path: str) -> CurriculumPipelineConfig:
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("curriculum_pipeline", data))

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump({"curriculum_pipeline": self.to_dict()}, f, default_flow_style=False)


# =============================================================================
# Curriculum State
# =============================================================================


@dataclass
class CurriculumState:
    """State of the curriculum pipeline progression.

    Attributes:
        current_tier: Current environment tier.
        current_stage_index: Index of current difficulty stage within tier.
        games_at_stage: Games played at current stage.
        games_at_tier: Games played at current tier.
        total_games: Total games played across all tiers.
        metrics_history: Recent game outcomes (win/loss).
        tier_history: History of tier transitions.
        stage_history: History of stage transitions.
        elo_rating: Current estimated ELO rating.
    """

    current_tier: EnvironmentTier = EnvironmentTier.MICROCIV
    current_stage_index: int = 0
    games_at_stage: int = 0
    games_at_tier: int = 0
    total_games: int = 0
    metrics_history: list[bool] = field(default_factory=list)
    tier_history: list[dict[str, Any]] = field(default_factory=list)
    stage_history: list[dict[str, Any]] = field(default_factory=list)
    elo_rating: float = 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_tier": self.current_tier.name,
            "current_stage_index": self.current_stage_index,
            "games_at_stage": self.games_at_stage,
            "games_at_tier": self.games_at_tier,
            "total_games": self.total_games,
            "metrics_history": self.metrics_history,
            "tier_history": self.tier_history,
            "stage_history": self.stage_history,
            "elo_rating": self.elo_rating,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurriculumState:
        """Create from dictionary."""
        return cls(
            current_tier=EnvironmentTier[data.get("current_tier", "MICROCIV")],
            current_stage_index=data.get("current_stage_index", 0),
            games_at_stage=data.get("games_at_stage", 0),
            games_at_tier=data.get("games_at_tier", 0),
            total_games=data.get("total_games", 0),
            metrics_history=data.get("metrics_history", []),
            tier_history=data.get("tier_history", []),
            stage_history=data.get("stage_history", []),
            elo_rating=data.get("elo_rating", 1000.0),
        )


# =============================================================================
# Weight Transfer Manager
# =============================================================================


class WeightTransferManager:
    """Manages weight transfer between network architectures.

    Handles the complexities of transferring learned weights when moving
    between environment tiers with different state/action spaces.
    """

    def __init__(self, config: TransferLearningConfig) -> None:
        """Initialize weight transfer manager.

        Args:
            config: Transfer learning configuration.
        """
        self.config = config
        self._logger = logging.getLogger(f"{__name__}.WeightTransferManager")

    def transfer_weights(
        self,
        source_network: NetworkProtocol,
        target_network: NetworkProtocol,
        source_tier: EnvironmentTier,
        target_tier: EnvironmentTier,
    ) -> dict[str, Any]:
        """Transfer weights from source to target network.

        Args:
            source_network: Network trained on source tier.
            target_network: Network for target tier (may have different architecture).
            source_tier: Source environment tier.
            target_tier: Target environment tier.

        Returns:
            Dictionary with transfer statistics.
        """
        if self.config.strategy == "none":
            return self._no_transfer(target_network)
        elif self.config.strategy == "full":
            return self._full_transfer(source_network, target_network)
        elif self.config.strategy == "partial":
            return self._partial_transfer(source_network, target_network)
        elif self.config.strategy == "progressive":
            return self._progressive_transfer(
                source_network,
                target_network,
                source_tier,
                target_tier,
            )
        else:
            self._logger.warning(f"Unknown transfer strategy: {self.config.strategy}")
            return self._no_transfer(target_network)

    def _no_transfer(self, target_network: NetworkProtocol) -> dict[str, Any]:
        """No transfer - start from scratch."""
        self._logger.info("No weight transfer - using random initialization")
        return {"transferred_layers": 0, "total_layers": 0, "strategy": "none"}

    def _full_transfer(
        self,
        source_network: NetworkProtocol,
        target_network: NetworkProtocol,
    ) -> dict[str, Any]:
        """Full transfer - copy all compatible weights."""
        source_state = source_network.state_dict()
        target_state = target_network.state_dict()

        transferred = 0
        total = len(target_state)

        for name, param in source_state.items():
            if name in target_state:
                if param.shape == target_state[name].shape:
                    target_state[name] = param
                    transferred += 1
                else:
                    self._logger.debug(
                        f"Shape mismatch for {name}: {param.shape} vs {target_state[name].shape}",
                    )

        target_network.load_state_dict(target_state)

        self._logger.info(
            f"Full transfer: {transferred}/{total} layers transferred",
        )
        return {
            "transferred_layers": transferred,
            "total_layers": total,
            "strategy": "full",
        }

    def _partial_transfer(
        self,
        source_network: NetworkProtocol,
        target_network: NetworkProtocol,
    ) -> dict[str, Any]:
        """Partial transfer - copy only shared backbone weights."""
        source_state = source_network.state_dict()
        target_state = target_network.state_dict()

        transferred = 0
        total = len(target_state)

        # Transfer backbone/residual layers but not heads
        head_keywords = ["policy", "value", "head", "fc_out", "output"]

        for name, param in source_state.items():
            # Skip head layers
            if any(kw in name.lower() for kw in head_keywords):
                continue

            if name in target_state:
                target_shape = target_state[name].shape
                source_shape = param.shape

                if source_shape == target_shape:
                    target_state[name] = param
                    transferred += 1
                elif self.config.feature_alignment:
                    # Try to align by copying shared dimensions
                    aligned = self._align_tensor(param, target_shape)
                    if aligned is not None:
                        target_state[name] = aligned
                        transferred += 1
                        self._logger.debug(f"Aligned {name}: {source_shape} -> {target_shape}")

        # Reinitialize heads if configured
        if self.config.reinitialize_heads:
            self._logger.info("Reinitializing policy and value heads")

        target_network.load_state_dict(target_state)

        self._logger.info(
            f"Partial transfer: {transferred}/{total} layers transferred",
        )
        return {
            "transferred_layers": transferred,
            "total_layers": total,
            "strategy": "partial",
            "reinitialize_heads": self.config.reinitialize_heads,
        }

    def _progressive_transfer(
        self,
        source_network: NetworkProtocol,
        target_network: NetworkProtocol,
        source_tier: EnvironmentTier,
        target_tier: EnvironmentTier,
    ) -> dict[str, Any]:
        """Progressive transfer - adapt based on tier complexity difference."""
        complexity_diff = target_tier.complexity_score - source_tier.complexity_score

        if complexity_diff <= 0:
            # Moving to simpler environment - transfer everything
            return self._full_transfer(source_network, target_network)
        elif complexity_diff <= 2:
            # Small step up - partial transfer with more layers
            return self._partial_transfer(source_network, target_network)
        else:
            # Large step up - minimal transfer, mostly reinitialize
            self._logger.info(
                f"Large complexity jump ({complexity_diff}), minimal transfer",
            )
            return self._partial_transfer(source_network, target_network)

    def _align_tensor(self, source: Any, target_shape: tuple) -> Any | None:
        """Attempt to align source tensor to target shape.

        Args:
            source: Source tensor.
            target_shape: Target shape to match.

        Returns:
            Aligned tensor or None if alignment not possible.
        """
        try:
            import torch

            if len(source.shape) != len(target_shape):
                return None

            # Handle different input channel sizes
            if source.shape[0] != target_shape[0]:
                # Can't easily align output channels
                return None

            # For convolutional layers with different input channels
            if len(source.shape) == 4:  # Conv weight: [out, in, h, w]
                if source.shape[1] < target_shape[1]:
                    # Target has more input channels - pad with zeros
                    new_weight = torch.zeros(target_shape)
                    new_weight[:, : source.shape[1], :, :] = source
                    return new_weight
                elif source.shape[1] > target_shape[1]:
                    # Target has fewer input channels - truncate
                    return source[:, : target_shape[1], :, :]

            return None
        except Exception as e:
            self._logger.debug(f"Tensor alignment failed: {e}")
            return None

    def get_frozen_layers(self) -> list[str]:
        """Get list of layer names that should be frozen."""
        return self.config.freeze_layers

    def should_unfreeze(self, games_played: int) -> bool:
        """Check if frozen layers should be unfrozen.

        Args:
            games_played: Number of games played since transfer.

        Returns:
            True if layers should be unfrozen.
        """
        return games_played >= self.config.warmup_games


# =============================================================================
# Environment Factory
# =============================================================================


class EnvironmentFactory(ABC):
    """Abstract factory for creating environments."""

    @abstractmethod
    def create_environment(
        self,
        tier: EnvironmentTier,
        stage: DifficultyStage,
    ) -> EnvironmentProtocol:
        """Create an environment instance.

        Args:
            tier: Environment tier.
            stage: Difficulty stage configuration.

        Returns:
            Environment instance.
        """


class DefaultEnvironmentFactory(EnvironmentFactory):
    """Default factory using built-in environments."""

    def __init__(
        self,
        microciv_class: type | None = None,
        unciv_interface: Any | None = None,
        civ6_interface: Any | None = None,
    ) -> None:
        """Initialize factory.

        Args:
            microciv_class: Class for MicroCiv environment.
            unciv_interface: Interface for Unciv.
            civ6_interface: Interface for Civ6.
        """
        self._microciv_class = microciv_class
        self._unciv_interface = unciv_interface
        self._civ6_interface = civ6_interface

    def create_environment(
        self,
        tier: EnvironmentTier,
        stage: DifficultyStage,
    ) -> EnvironmentProtocol:
        """Create an environment instance."""
        if tier == EnvironmentTier.MICROCIV:
            return self._create_microciv(stage)
        elif tier == EnvironmentTier.UNCIV:
            return self._create_unciv(stage)
        elif tier == EnvironmentTier.CIV6:
            return self._create_civ6(stage)
        else:
            raise ValueError(f"Unknown tier: {tier}")

    def _create_microciv(self, stage: DifficultyStage) -> EnvironmentProtocol:
        """Create MicroCiv environment."""
        if self._microciv_class is None:
            try:
                from alpha_zero.envs.microciv import MicroCivEnv

                self._microciv_class = MicroCivEnv
            except ImportError as e:
                raise RuntimeError("MicroCiv environment not available") from e

        env: EnvironmentProtocol = self._microciv_class(
            board_size=stage.board_size,
            max_turns=stage.max_turns,
        )
        return env

    def _create_unciv(self, stage: DifficultyStage) -> EnvironmentProtocol:
        """Create Unciv-wrapped environment."""
        # Unciv requires special handling - wrap MicroCiv for now
        # In production, this would interface with actual Unciv
        logger.warning(
            "Unciv environment not fully implemented, using MicroCiv with Unciv-like settings",
        )
        return self._create_microciv(stage)

    def _create_civ6(self, stage: DifficultyStage) -> EnvironmentProtocol:
        """Create Civ6-wrapped environment."""
        # Civ6 requires special handling
        logger.warning(
            "Civ6 environment not fully implemented, using MicroCiv with Civ6-like settings",
        )
        return self._create_microciv(stage)


# =============================================================================
# Main Curriculum Pipeline
# =============================================================================


class CurriculumPipeline:
    """Main curriculum learning pipeline.

    Manages the complete curriculum progression from simple environments
    to complex ones, handling stage transitions, tier promotions, and
    weight transfer between network architectures.

    Example:
        >>> config = CurriculumPipelineConfig.default()
        >>> pipeline = CurriculumPipeline(config)
        >>>
        >>> # Training loop
        >>> while not pipeline.is_complete():
        ...     env = pipeline.get_current_environment()
        ...     # ... play games ...
        ...     pipeline.record_game(won=True)
        ...
        ...     if pipeline.should_promote():
        ...         pipeline.promote()
    """

    def __init__(
        self,
        config: CurriculumPipelineConfig,
        environment_factory: EnvironmentFactory | None = None,
    ) -> None:
        """Initialize curriculum pipeline.

        Args:
            config: Pipeline configuration.
            environment_factory: Factory for creating environments.
        """
        self.config = config
        self._state = CurriculumState()
        self._env_factory = environment_factory or DefaultEnvironmentFactory()
        self._weight_manager = WeightTransferManager(config.transfer_learning)
        self._logger = logging.getLogger(f"{__name__}.CurriculumPipeline")

        # Callbacks for events
        self._on_stage_change: list[Callable[[str, str], None]] = []
        self._on_tier_change: list[Callable[[EnvironmentTier, EnvironmentTier], None]] = []

        # Load saved state if exists
        self._load_state()

    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------

    @property
    def current_tier(self) -> EnvironmentTier:
        """Get current environment tier."""
        return self._state.current_tier

    @property
    def current_stage(self) -> DifficultyStage:
        """Get current difficulty stage."""
        tier_config = self.config.get_tier_config(self._state.current_tier)
        if tier_config and self._state.current_stage_index < len(tier_config.difficulty_stages):
            return tier_config.difficulty_stages[self._state.current_stage_index]
        return DifficultyStage(name="default")

    @property
    def total_games(self) -> int:
        """Get total games played."""
        return self._state.total_games

    @property
    def win_rate(self) -> float:
        """Get win rate over metrics window."""
        if not self._state.metrics_history:
            return 0.0
        window = self._state.metrics_history[-self.config.metrics_window :]
        return sum(window) / len(window)

    @property
    def elo_rating(self) -> float:
        """Get current ELO rating."""
        return self._state.elo_rating

    # -------------------------------------------------------------------------
    # Environment Management
    # -------------------------------------------------------------------------

    def get_current_environment(self) -> EnvironmentProtocol:
        """Get environment for current curriculum stage.

        Returns:
            Environment instance configured for current stage.
        """
        return self._env_factory.create_environment(
            self._state.current_tier,
            self.current_stage,
        )

    def get_environment_params(self) -> dict[str, Any]:
        """Get parameters for current environment.

        Returns:
            Dictionary of environment parameters.
        """
        stage = self.current_stage
        return {
            "tier": self._state.current_tier.name,
            "stage": stage.name,
            "board_size": stage.board_size,
            "max_turns": stage.max_turns,
            "num_opponents": stage.num_opponents,
            "starting_resources": stage.starting_resources,
            **stage.extra_params,
        }

    # -------------------------------------------------------------------------
    # Game Recording
    # -------------------------------------------------------------------------

    def record_game(
        self,
        won: bool,
        opponent_elo: float = 1000.0,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a completed game.

        Args:
            won: Whether the agent won.
            opponent_elo: ELO rating of opponent.
            metrics: Optional additional metrics.

        Returns:
            Dictionary with status and any changes.
        """
        self._state.games_at_stage += 1
        self._state.games_at_tier += 1
        self._state.total_games += 1

        # Update metrics history
        self._state.metrics_history.append(won)
        if len(self._state.metrics_history) > self.config.metrics_window * 2:
            self._state.metrics_history = self._state.metrics_history[-self.config.metrics_window :]

        # Update ELO
        self._update_elo(won, opponent_elo)

        # Check for stage/tier changes
        result = self._check_progression()

        # Auto-save if configured
        if self.config.auto_save:
            self._save_state()

        return result

    def _update_elo(self, won: bool, opponent_elo: float) -> None:
        """Update ELO rating based on game result."""
        k_factor = 32  # Standard K-factor
        expected = 1 / (1 + 10 ** ((opponent_elo - self._state.elo_rating) / 400))
        actual = 1.0 if won else 0.0
        self._state.elo_rating += k_factor * (actual - expected)

    # -------------------------------------------------------------------------
    # Progression Checks
    # -------------------------------------------------------------------------

    def _check_progression(self) -> dict[str, Any]:
        """Check if stage or tier should change.

        Returns:
            Dictionary with change status and details.
        """
        # First check stage progression
        stage_result = self._check_stage_progression()
        if stage_result.get("changed"):
            return stage_result

        # Then check tier progression
        tier_result = self._check_tier_progression()
        return tier_result

    def _check_stage_progression(self) -> dict[str, Any]:
        """Check if difficulty stage should change."""
        stage = self.current_stage

        # Need minimum games
        if self._state.games_at_stage < stage.min_games:
            return {
                "changed": False,
                "stage": stage.name,
                "win_rate": self.win_rate,
                "games_remaining": stage.min_games - self._state.games_at_stage,
            }

        # Check for promotion
        if self.win_rate >= stage.promotion_win_rate:
            return self._promote_stage()

        # Check for demotion
        if self.win_rate <= stage.demotion_win_rate:
            return self._demote_stage()

        return {
            "changed": False,
            "stage": stage.name,
            "win_rate": self.win_rate,
        }

    def _check_tier_progression(self) -> dict[str, Any]:
        """Check if environment tier should change."""
        tier_config = self.config.get_tier_config(self._state.current_tier)
        if not tier_config:
            return {"changed": False}

        criteria = tier_config.promotion_criteria

        # Check promotion criteria
        if (
            self._state.games_at_tier >= criteria.get("min_games", 500)
            and self.win_rate >= criteria.get("win_rate", 0.65)
            and self._state.elo_rating >= criteria.get("elo_rating", 1500)
            and self._state.current_stage_index >= len(tier_config.difficulty_stages) - 1
        ):
            next_tier = self._state.current_tier.next_tier()
            if next_tier and self.config.get_tier_config(next_tier):
                return self._promote_tier()

        # Check demotion criteria
        if self.config.allow_tier_demotion:
            demotion = tier_config.demotion_criteria
            if self._state.games_at_tier >= demotion.get(
                "min_games", 100
            ) and self.win_rate <= demotion.get("win_rate", 0.2):
                prev_tier = self._state.current_tier.previous_tier()
                if prev_tier and self.config.get_tier_config(prev_tier):
                    return self._demote_tier()

        return {"changed": False}

    # -------------------------------------------------------------------------
    # Stage Transitions
    # -------------------------------------------------------------------------

    def _promote_stage(self) -> dict[str, Any]:
        """Promote to next difficulty stage."""
        tier_config = self.config.get_tier_config(self._state.current_tier)
        if not tier_config:
            return {"changed": False}

        old_stage = self.current_stage

        if self._state.current_stage_index < len(tier_config.difficulty_stages) - 1:
            self._state.current_stage_index += 1
            new_stage = self.current_stage

            # Record history
            self._state.stage_history.append(
                {
                    "type": "promotion",
                    "from": old_stage.name,
                    "to": new_stage.name,
                    "games": self._state.total_games,
                    "win_rate": self.win_rate,
                }
            )

            # Reset stage counters
            self._state.games_at_stage = 0
            self._state.metrics_history.clear()

            # Notify callbacks
            for callback in self._on_stage_change:
                callback(old_stage.name, new_stage.name)

            self._logger.info(
                f"Stage promotion: {old_stage.name} -> {new_stage.name}",
            )

            return {
                "changed": True,
                "type": "stage_promotion",
                "from_stage": old_stage.name,
                "to_stage": new_stage.name,
            }

        return {"changed": False, "reason": "at_final_stage"}

    def _demote_stage(self) -> dict[str, Any]:
        """Demote to previous difficulty stage."""
        old_stage = self.current_stage

        if self._state.current_stage_index > 0:
            self._state.current_stage_index -= 1
            new_stage = self.current_stage

            # Record history
            self._state.stage_history.append(
                {
                    "type": "demotion",
                    "from": old_stage.name,
                    "to": new_stage.name,
                    "games": self._state.total_games,
                    "win_rate": self.win_rate,
                }
            )

            # Reset stage counters
            self._state.games_at_stage = 0
            self._state.metrics_history.clear()

            # Notify callbacks
            for callback in self._on_stage_change:
                callback(old_stage.name, new_stage.name)

            self._logger.info(
                f"Stage demotion: {old_stage.name} -> {new_stage.name}",
            )

            return {
                "changed": True,
                "type": "stage_demotion",
                "from_stage": old_stage.name,
                "to_stage": new_stage.name,
            }

        return {"changed": False, "reason": "at_first_stage"}

    # -------------------------------------------------------------------------
    # Tier Transitions
    # -------------------------------------------------------------------------

    def _promote_tier(self) -> dict[str, Any]:
        """Promote to next environment tier."""
        old_tier = self._state.current_tier
        new_tier = old_tier.next_tier()

        if not new_tier:
            return {"changed": False, "reason": "at_final_tier"}

        new_tier_config = self.config.get_tier_config(new_tier)
        if not new_tier_config:
            return {"changed": False, "reason": "tier_not_configured"}

        # Record history
        self._state.tier_history.append(
            {
                "type": "promotion",
                "from": old_tier.name,
                "to": new_tier.name,
                "games": self._state.total_games,
                "win_rate": self.win_rate,
                "elo": self._state.elo_rating,
            }
        )

        # Update state
        self._state.current_tier = new_tier
        self._state.current_stage_index = 0
        self._state.games_at_stage = 0
        self._state.games_at_tier = 0
        self._state.metrics_history.clear()

        # Notify callbacks
        for callback in self._on_tier_change:
            callback(old_tier, new_tier)

        self._logger.info(
            f"Tier promotion: {old_tier.name} -> {new_tier.name}",
        )

        return {
            "changed": True,
            "type": "tier_promotion",
            "from_tier": old_tier.name,
            "to_tier": new_tier.name,
            "transfer_required": True,
        }

    def _demote_tier(self) -> dict[str, Any]:
        """Demote to previous environment tier."""
        old_tier = self._state.current_tier
        new_tier = old_tier.previous_tier()

        if not new_tier:
            return {"changed": False, "reason": "at_first_tier"}

        new_tier_config = self.config.get_tier_config(new_tier)
        if not new_tier_config:
            return {"changed": False, "reason": "tier_not_configured"}

        # Record history
        self._state.tier_history.append(
            {
                "type": "demotion",
                "from": old_tier.name,
                "to": new_tier.name,
                "games": self._state.total_games,
                "win_rate": self.win_rate,
                "elo": self._state.elo_rating,
            }
        )

        # Update state - go to last stage of previous tier
        self._state.current_tier = new_tier
        self._state.current_stage_index = len(new_tier_config.difficulty_stages) - 1
        self._state.games_at_stage = 0
        self._state.games_at_tier = 0
        self._state.metrics_history.clear()

        # Notify callbacks
        for callback in self._on_tier_change:
            callback(old_tier, new_tier)

        self._logger.info(
            f"Tier demotion: {old_tier.name} -> {new_tier.name}",
        )

        return {
            "changed": True,
            "type": "tier_demotion",
            "from_tier": old_tier.name,
            "to_tier": new_tier.name,
            "transfer_required": True,
        }

    # -------------------------------------------------------------------------
    # Weight Transfer
    # -------------------------------------------------------------------------

    def transfer_weights(
        self,
        source_network: NetworkProtocol,
        target_network: NetworkProtocol,
        source_tier: EnvironmentTier,
        target_tier: EnvironmentTier,
    ) -> dict[str, Any]:
        """Transfer weights between networks for tier transition.

        Args:
            source_network: Network from source tier.
            target_network: Network for target tier.
            source_tier: Source environment tier.
            target_tier: Target environment tier.

        Returns:
            Dictionary with transfer statistics.
        """
        return self._weight_manager.transfer_weights(
            source_network,
            target_network,
            source_tier,
            target_tier,
        )

    # -------------------------------------------------------------------------
    # Progress Information
    # -------------------------------------------------------------------------

    def get_progress(self) -> dict[str, Any]:
        """Get comprehensive progress summary.

        Returns:
            Dictionary with progress information.
        """
        tier_config = self.config.get_tier_config(self._state.current_tier)

        return {
            "current_tier": self._state.current_tier.name,
            "current_stage": self.current_stage.name,
            "stage_index": self._state.current_stage_index,
            "total_stages": len(tier_config.difficulty_stages) if tier_config else 0,
            "games_at_stage": self._state.games_at_stage,
            "games_at_tier": self._state.games_at_tier,
            "total_games": self._state.total_games,
            "win_rate": self.win_rate,
            "elo_rating": self._state.elo_rating,
            "promotion_threshold": self.current_stage.promotion_win_rate,
            "tier_history": self._state.tier_history[-5:],
            "stage_history": self._state.stage_history[-10:],
        }

    def is_complete(self) -> bool:
        """Check if curriculum is complete.

        Returns:
            True if at final tier and final stage.
        """
        final_tier = EnvironmentTier.CIV6
        tier_config = self.config.get_tier_config(final_tier)

        if self._state.current_tier != final_tier:
            return False

        if tier_config:
            return self._state.current_stage_index >= len(tier_config.difficulty_stages) - 1

        return False

    # -------------------------------------------------------------------------
    # State Persistence
    # -------------------------------------------------------------------------

    def _save_state(self) -> None:
        """Save curriculum state to checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state_path = checkpoint_dir / "curriculum_state.json"
        with open(state_path, "w") as f:
            json.dump(self._state.to_dict(), f, indent=2)

    def _load_state(self) -> None:
        """Load curriculum state from checkpoint."""
        state_path = Path(self.config.checkpoint_dir) / "curriculum_state.json"

        if state_path.exists():
            try:
                with open(state_path) as f:
                    data = json.load(f)
                self._state = CurriculumState.from_dict(data)
                self._logger.info(
                    f"Loaded curriculum state: tier={self._state.current_tier.name}, "
                    f"stage={self._state.current_stage_index}, "
                    f"games={self._state.total_games}",
                )
            except Exception as e:
                self._logger.warning(f"Failed to load curriculum state: {e}")

    def reset(self) -> None:
        """Reset curriculum to initial state."""
        self._state = CurriculumState()

        # Remove saved state
        state_path = Path(self.config.checkpoint_dir) / "curriculum_state.json"
        if state_path.exists():
            state_path.unlink()

        self._logger.info("Curriculum reset to initial state")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_stage_change(self, callback: Callable[[str, str], None]) -> None:
        """Register callback for stage changes.

        Args:
            callback: Function called with (old_stage, new_stage).
        """
        self._on_stage_change.append(callback)

    def on_tier_change(
        self,
        callback: Callable[[EnvironmentTier, EnvironmentTier], None],
    ) -> None:
        """Register callback for tier changes.

        Args:
            callback: Function called with (old_tier, new_tier).
        """
        self._on_tier_change.append(callback)

    # -------------------------------------------------------------------------
    # Manual Controls
    # -------------------------------------------------------------------------

    def force_stage(self, stage_index: int) -> dict[str, Any]:
        """Force transition to specific stage.

        Args:
            stage_index: Index of target stage.

        Returns:
            Transition result.
        """
        tier_config = self.config.get_tier_config(self._state.current_tier)
        if not tier_config:
            raise ValueError("Current tier not configured")

        if stage_index < 0 or stage_index >= len(tier_config.difficulty_stages):
            raise ValueError(f"Invalid stage index: {stage_index}")

        old_stage = self.current_stage
        self._state.current_stage_index = stage_index
        new_stage = self.current_stage

        self._state.games_at_stage = 0
        self._state.metrics_history.clear()

        if self.config.auto_save:
            self._save_state()

        self._logger.info(f"Forced stage change: {old_stage.name} -> {new_stage.name}")

        return {
            "changed": True,
            "type": "forced",
            "from_stage": old_stage.name,
            "to_stage": new_stage.name,
        }

    def force_tier(self, tier: EnvironmentTier) -> dict[str, Any]:
        """Force transition to specific tier.

        Args:
            tier: Target environment tier.

        Returns:
            Transition result.
        """
        tier_config = self.config.get_tier_config(tier)
        if not tier_config:
            raise ValueError(f"Tier not configured: {tier}")

        old_tier = self._state.current_tier
        self._state.current_tier = tier
        self._state.current_stage_index = 0
        self._state.games_at_stage = 0
        self._state.games_at_tier = 0
        self._state.metrics_history.clear()

        # Notify callbacks if tier actually changed
        if old_tier != tier:
            for callback in self._on_tier_change:
                callback(old_tier, tier)

        if self.config.auto_save:
            self._save_state()

        self._logger.info(f"Forced tier change: {old_tier.name} -> {tier.name}")

        return {
            "changed": True,
            "type": "forced",
            "from_tier": old_tier.name,
            "to_tier": tier.name,
            "transfer_required": old_tier != tier,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_curriculum_pipeline(
    preset: str = "default",
    **kwargs: Any,
) -> CurriculumPipeline:
    """Factory function to create curriculum pipeline.

    Args:
        preset: Configuration preset ('default', 'fast', 'production').
        **kwargs: Additional configuration overrides.

    Returns:
        CurriculumPipeline instance.
    """
    if preset == "fast":
        config = CurriculumPipelineConfig.fast_iteration()
    elif preset == "production":
        config = CurriculumPipelineConfig.production()
    else:
        config = CurriculumPipelineConfig.default()

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return CurriculumPipeline(config)


def create_environment_factory(
    microciv_class: type | None = None,
    unciv_interface: Any | None = None,
    civ6_interface: Any | None = None,
) -> EnvironmentFactory:
    """Factory function to create environment factory.

    Args:
        microciv_class: Class for MicroCiv environment.
        unciv_interface: Interface for Unciv.
        civ6_interface: Interface for Civ6.

    Returns:
        EnvironmentFactory instance.
    """
    return DefaultEnvironmentFactory(
        microciv_class=microciv_class,
        unciv_interface=unciv_interface,
        civ6_interface=civ6_interface,
    )
