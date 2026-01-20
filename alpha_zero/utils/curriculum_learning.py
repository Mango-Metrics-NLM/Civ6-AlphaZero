"""Curriculum learning utilities for progressive difficulty training.

This module provides curriculum learning capabilities that gradually increase
training difficulty as the agent improves, enabling faster convergence and
better final performance.

Features:
- Configurable difficulty stages with clear progression criteria
- Multiple difficulty dimensions (board size, opponents, time limits)
- Automatic promotion/demotion based on performance metrics
- Adaptive difficulty adjustment

Example usage:
    >>> from alpha_zero.utils.curriculum_learning import (
    ...     CurriculumConfig,
    ...     CurriculumManager,
    ...     create_curriculum_manager,
    ... )
    >>> config = CurriculumConfig.default_three_stage()
    >>> manager = CurriculumManager(config)
    >>> stage = manager.get_current_stage()
    >>> env_params = manager.get_environment_params()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

from alpha_zero.utils.logging import get_logger

logger = get_logger(__name__)


class DifficultyDimension(Enum):
    """Dimensions along which difficulty can vary."""

    BOARD_SIZE = "board_size"
    MAX_TURNS = "max_turns"
    NUM_OPPONENTS = "num_opponents"
    STARTING_UNITS = "starting_units"
    STARTING_CITIES = "starting_cities"
    PRODUCTION_RATE = "production_rate"
    FOG_OF_WAR = "fog_of_war"
    TERRAIN_COMPLEXITY = "terrain_complexity"


@dataclass
class CurriculumStage:
    """Configuration for a single curriculum stage.

    Attributes:
        name: Human-readable stage name.
        level: Numeric difficulty level (0 = easiest).
        board_size: Board size for this stage.
        max_turns: Maximum turns per game.
        num_opponents: Number of opponent players.
        starting_units: Initial unit count per player.
        starting_cities: Initial city count per player.
        production_multiplier: Production speed multiplier.
        fog_of_war_enabled: Whether fog of war is active.
        promotion_threshold: Win rate required to advance.
        demotion_threshold: Win rate causing demotion.
        min_games: Minimum games before stage change.
        description: Optional stage description.
    """

    name: str
    level: int = 0
    board_size: int = 8
    max_turns: int = 50
    num_opponents: int = 1
    starting_units: int = 2
    starting_cities: int = 0
    production_multiplier: float = 2.0
    fog_of_war_enabled: bool = False
    promotion_threshold: float = 0.6
    demotion_threshold: float = 0.3
    min_games: int = 50
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert stage to dictionary.

        Returns:
            Stage as dictionary.
        """
        return {
            "name": self.name,
            "level": self.level,
            "board_size": self.board_size,
            "max_turns": self.max_turns,
            "num_opponents": self.num_opponents,
            "starting_units": self.starting_units,
            "starting_cities": self.starting_cities,
            "production_multiplier": self.production_multiplier,
            "fog_of_war_enabled": self.fog_of_war_enabled,
            "promotion_threshold": self.promotion_threshold,
            "demotion_threshold": self.demotion_threshold,
            "min_games": self.min_games,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumStage":
        """Create stage from dictionary.

        Args:
            data: Stage dictionary.

        Returns:
            CurriculumStage instance.
        """
        return cls(
            name=data.get("name", "default"),
            level=data.get("level", 0),
            board_size=data.get("board_size", 8),
            max_turns=data.get("max_turns", 50),
            num_opponents=data.get("num_opponents", 1),
            starting_units=data.get("starting_units", 2),
            starting_cities=data.get("starting_cities", 0),
            production_multiplier=data.get("production_multiplier", 2.0),
            fog_of_war_enabled=data.get("fog_of_war_enabled", False),
            promotion_threshold=data.get("promotion_threshold", 0.6),
            demotion_threshold=data.get("demotion_threshold", 0.3),
            min_games=data.get("min_games", 50),
            description=data.get("description", ""),
        )


@dataclass
class CurriculumConfig:
    """Complete curriculum configuration.

    Attributes:
        enabled: Whether curriculum learning is active.
        stages: List of curriculum stages in order of difficulty.
        allow_demotion: Whether to allow demotion to easier stages.
        metrics_window: Number of games to consider for promotion/demotion.
        adaptive_enabled: Enable adaptive difficulty adjustment.
        adaptive_step_size: How much to adjust difficulty adaptively.
        save_path: Path to save curriculum progress.
    """

    enabled: bool = True
    stages: list[CurriculumStage] = field(default_factory=list)
    allow_demotion: bool = True
    metrics_window: int = 100
    adaptive_enabled: bool = False
    adaptive_step_size: float = 0.1
    save_path: str | None = None

    def __post_init__(self) -> None:
        """Validate and set up stages."""
        if not self.stages:
            self.stages = self._default_stages()

        # Sort stages by level
        self.stages.sort(key=lambda s: s.level)

        # Validate monotonic difficulty
        for i in range(1, len(self.stages)):
            if self.stages[i].level <= self.stages[i-1].level:
                logger.warning(
                    f"Stage levels should be strictly increasing: "
                    f"{self.stages[i-1].name} (level {self.stages[i-1].level}) "
                    f"and {self.stages[i].name} (level {self.stages[i].level})"
                )

    @staticmethod
    def _default_stages() -> list[CurriculumStage]:
        """Get default curriculum stages.

        Returns:
            List of default stages.
        """
        return [
            CurriculumStage(
                name="beginner",
                level=0,
                board_size=8,
                max_turns=30,
                promotion_threshold=0.7,
                description="Small board, short games",
            ),
            CurriculumStage(
                name="intermediate",
                level=1,
                board_size=12,
                max_turns=60,
                production_multiplier=1.5,
                promotion_threshold=0.6,
                description="Medium board, longer games",
            ),
            CurriculumStage(
                name="advanced",
                level=2,
                board_size=16,
                max_turns=100,
                production_multiplier=1.0,
                fog_of_war_enabled=True,
                promotion_threshold=0.55,
                description="Full board, fog of war",
            ),
        ]

    @classmethod
    def default_three_stage(cls) -> "CurriculumConfig":
        """Create default three-stage curriculum.

        Returns:
            CurriculumConfig with three stages.
        """
        return cls(
            enabled=True,
            stages=cls._default_stages(),
        )

    @classmethod
    def default_five_stage(cls) -> "CurriculumConfig":
        """Create five-stage curriculum for gradual progression.

        Returns:
            CurriculumConfig with five stages.
        """
        return cls(
            enabled=True,
            stages=[
                CurriculumStage(
                    name="tutorial",
                    level=0,
                    board_size=6,
                    max_turns=20,
                    production_multiplier=3.0,
                    promotion_threshold=0.8,
                    min_games=30,
                    description="Very simple games",
                ),
                CurriculumStage(
                    name="beginner",
                    level=1,
                    board_size=8,
                    max_turns=30,
                    production_multiplier=2.0,
                    promotion_threshold=0.7,
                    description="Small board",
                ),
                CurriculumStage(
                    name="intermediate",
                    level=2,
                    board_size=10,
                    max_turns=50,
                    production_multiplier=1.5,
                    promotion_threshold=0.65,
                    description="Medium board",
                ),
                CurriculumStage(
                    name="advanced",
                    level=3,
                    board_size=14,
                    max_turns=80,
                    production_multiplier=1.2,
                    fog_of_war_enabled=True,
                    promotion_threshold=0.6,
                    description="Larger board with fog",
                ),
                CurriculumStage(
                    name="expert",
                    level=4,
                    board_size=16,
                    max_turns=100,
                    production_multiplier=1.0,
                    fog_of_war_enabled=True,
                    promotion_threshold=0.55,
                    description="Full difficulty",
                ),
            ],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumConfig":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            CurriculumConfig instance.
        """
        stages_data = data.get("stages", [])
        stages = [CurriculumStage.from_dict(s) for s in stages_data]

        return cls(
            enabled=data.get("enabled", True),
            stages=stages,
            allow_demotion=data.get("allow_demotion", True),
            metrics_window=data.get("metrics_window", 100),
            adaptive_enabled=data.get("adaptive_enabled", False),
            adaptive_step_size=data.get("adaptive_step_size", 0.1),
            save_path=data.get("save_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary.
        """
        return {
            "enabled": self.enabled,
            "stages": [s.to_dict() for s in self.stages],
            "allow_demotion": self.allow_demotion,
            "metrics_window": self.metrics_window,
            "adaptive_enabled": self.adaptive_enabled,
            "adaptive_step_size": self.adaptive_step_size,
            "save_path": self.save_path,
        }


@dataclass
class CurriculumState:
    """State of curriculum progression.

    Attributes:
        current_stage_index: Index of current stage.
        games_at_stage: Number of games played at current stage.
        total_games: Total games played across all stages.
        wins_at_stage: Wins at current stage (within metrics window).
        stage_history: History of stage changes.
        metrics_history: Recent game outcomes.
    """

    current_stage_index: int = 0
    games_at_stage: int = 0
    total_games: int = 0
    wins_at_stage: int = 0
    stage_history: list[dict[str, Any]] = field(default_factory=list)
    metrics_history: list[bool] = field(default_factory=list)  # True = win


class CurriculumManager:
    """Manages curriculum progression during training."""

    def __init__(self, config: CurriculumConfig | None = None):
        """Initialize curriculum manager.

        Args:
            config: Curriculum configuration.
        """
        self.config = config or CurriculumConfig()
        self._state = CurriculumState()
        self._logger = get_logger(f"{__name__}.CurriculumManager")

        # Load state if save path exists
        if self.config.save_path:
            self._load_state()

    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage.

        Returns:
            Current CurriculumStage.
        """
        if not self.config.stages:
            return CurriculumStage(name="default")

        idx = min(self._state.current_stage_index, len(self.config.stages) - 1)
        return self.config.stages[idx]

    def get_environment_params(self) -> dict[str, Any]:
        """Get environment parameters for current stage.

        Returns:
            Dictionary of environment parameters.
        """
        stage = self.get_current_stage()

        return {
            "board_size": stage.board_size,
            "max_turns": stage.max_turns,
            "production_multiplier": stage.production_multiplier,
            "fog_of_war_enabled": stage.fog_of_war_enabled,
            "starting_units": stage.starting_units,
            "starting_cities": stage.starting_cities,
        }

    def record_game(self, won: bool, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record a completed game.

        Args:
            won: Whether the agent won.
            metrics: Optional additional metrics.

        Returns:
            Dictionary with status and any stage changes.
        """
        self._state.games_at_stage += 1
        self._state.total_games += 1

        if won:
            self._state.wins_at_stage += 1

        # Track in metrics window
        self._state.metrics_history.append(won)
        if len(self._state.metrics_history) > self.config.metrics_window:
            self._state.metrics_history.pop(0)

        # Check for stage change
        result = self._check_stage_change()

        # Save state
        if self.config.save_path:
            self._save_state()

        return result

    def _check_stage_change(self) -> dict[str, Any]:
        """Check if stage should change.

        Returns:
            Dictionary with change status.
        """
        stage = self.get_current_stage()

        # Need minimum games before considering change
        if self._state.games_at_stage < stage.min_games:
            return {
                "changed": False,
                "stage": stage.name,
                "win_rate": self._get_window_win_rate(),
                "games_remaining": stage.min_games - self._state.games_at_stage,
            }

        win_rate = self._get_window_win_rate()

        # Check for promotion
        if win_rate >= stage.promotion_threshold:
            if self._state.current_stage_index < len(self.config.stages) - 1:
                return self._promote()

        # Check for demotion
        if self.config.allow_demotion and win_rate <= stage.demotion_threshold:
            if self._state.current_stage_index > 0:
                return self._demote()

        return {
            "changed": False,
            "stage": stage.name,
            "win_rate": win_rate,
            "to_promotion": stage.promotion_threshold - win_rate,
        }

    def _get_window_win_rate(self) -> float:
        """Get win rate over metrics window.

        Returns:
            Win rate as fraction.
        """
        if not self._state.metrics_history:
            return 0.0

        return sum(self._state.metrics_history) / len(self._state.metrics_history)

    def _promote(self) -> dict[str, Any]:
        """Promote to next stage.

        Returns:
            Promotion result dictionary.
        """
        old_stage = self.get_current_stage()
        self._state.current_stage_index += 1
        new_stage = self.get_current_stage()

        # Record history
        self._state.stage_history.append({
            "type": "promotion",
            "from_stage": old_stage.name,
            "to_stage": new_stage.name,
            "games": self._state.total_games,
            "win_rate": self._get_window_win_rate(),
        })

        # Reset stage counters
        self._state.games_at_stage = 0
        self._state.wins_at_stage = 0
        self._state.metrics_history.clear()

        self._logger.info(
            f"Promoted from {old_stage.name} (level {old_stage.level}) "
            f"to {new_stage.name} (level {new_stage.level})"
        )

        return {
            "changed": True,
            "type": "promotion",
            "from_stage": old_stage.name,
            "to_stage": new_stage.name,
            "new_params": self.get_environment_params(),
        }

    def _demote(self) -> dict[str, Any]:
        """Demote to previous stage.

        Returns:
            Demotion result dictionary.
        """
        old_stage = self.get_current_stage()
        self._state.current_stage_index -= 1
        new_stage = self.get_current_stage()

        # Record history
        self._state.stage_history.append({
            "type": "demotion",
            "from_stage": old_stage.name,
            "to_stage": new_stage.name,
            "games": self._state.total_games,
            "win_rate": self._get_window_win_rate(),
        })

        # Reset stage counters
        self._state.games_at_stage = 0
        self._state.wins_at_stage = 0
        self._state.metrics_history.clear()

        self._logger.info(
            f"Demoted from {old_stage.name} (level {old_stage.level}) "
            f"to {new_stage.name} (level {new_stage.level})"
        )

        return {
            "changed": True,
            "type": "demotion",
            "from_stage": old_stage.name,
            "to_stage": new_stage.name,
            "new_params": self.get_environment_params(),
        }

    def force_stage(self, stage_index: int) -> dict[str, Any]:
        """Force transition to specific stage.

        Args:
            stage_index: Index of target stage.

        Returns:
            Transition result dictionary.
        """
        if stage_index < 0 or stage_index >= len(self.config.stages):
            raise ValueError(f"Invalid stage index: {stage_index}")

        old_stage = self.get_current_stage()
        self._state.current_stage_index = stage_index
        new_stage = self.get_current_stage()

        # Reset counters
        self._state.games_at_stage = 0
        self._state.wins_at_stage = 0
        self._state.metrics_history.clear()

        self._logger.info(f"Forced stage change to {new_stage.name}")

        # Save state after force change
        if self.config.save_path:
            self._save_state()

        return {
            "changed": True,
            "type": "forced",
            "from_stage": old_stage.name,
            "to_stage": new_stage.name,
            "new_params": self.get_environment_params(),
        }

    def get_progress(self) -> dict[str, Any]:
        """Get curriculum progress summary.

        Returns:
            Progress summary dictionary.
        """
        stage = self.get_current_stage()

        return {
            "current_stage": stage.name,
            "current_level": stage.level,
            "total_stages": len(self.config.stages),
            "games_at_stage": self._state.games_at_stage,
            "total_games": self._state.total_games,
            "win_rate": self._get_window_win_rate(),
            "promotion_threshold": stage.promotion_threshold,
            "demotion_threshold": stage.demotion_threshold if self.config.allow_demotion else None,
            "stage_history": self._state.stage_history[-10:],  # Last 10 changes
            "at_final_stage": self._state.current_stage_index >= len(self.config.stages) - 1,
        }

    def _save_state(self) -> None:
        """Save curriculum state to file."""
        if not self.config.save_path:
            return

        save_path = Path(self.config.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = {
            "current_stage_index": self._state.current_stage_index,
            "games_at_stage": self._state.games_at_stage,
            "total_games": self._state.total_games,
            "wins_at_stage": self._state.wins_at_stage,
            "stage_history": self._state.stage_history,
            "metrics_history": self._state.metrics_history,
        }

        with open(save_path, "w") as f:
            json.dump(state_dict, f, indent=2)

    def _load_state(self) -> None:
        """Load curriculum state from file."""
        if not self.config.save_path:
            return

        save_path = Path(self.config.save_path)
        if not save_path.exists():
            return

        try:
            with open(save_path) as f:
                state_dict = json.load(f)

            self._state = CurriculumState(
                current_stage_index=state_dict.get("current_stage_index", 0),
                games_at_stage=state_dict.get("games_at_stage", 0),
                total_games=state_dict.get("total_games", 0),
                wins_at_stage=state_dict.get("wins_at_stage", 0),
                stage_history=state_dict.get("stage_history", []),
                metrics_history=state_dict.get("metrics_history", []),
            )

            self._logger.info(f"Loaded curriculum state from {save_path}")
        except Exception as e:
            self._logger.warning(f"Failed to load curriculum state: {e}")

    def reset(self) -> None:
        """Reset curriculum to initial state."""
        self._state = CurriculumState()

        if self.config.save_path:
            save_path = Path(self.config.save_path)
            if save_path.exists():
                save_path.unlink()

        self._logger.info("Curriculum reset to initial state")


class AdaptiveDifficultyController:
    """Provides fine-grained adaptive difficulty adjustment."""

    def __init__(
        self,
        config: CurriculumConfig,
        target_win_rate: float = 0.55,
        adjustment_rate: float = 0.05,
    ):
        """Initialize adaptive difficulty controller.

        Args:
            config: Curriculum configuration.
            target_win_rate: Target win rate to maintain.
            adjustment_rate: Rate of difficulty adjustment.
        """
        self.config = config
        self.target_win_rate = target_win_rate
        self.adjustment_rate = adjustment_rate

        self._difficulty_modifier: float = 0.0
        self._recent_outcomes: list[bool] = []
        self._window_size: int = 50
        self._logger = get_logger(f"{__name__}.AdaptiveDifficultyController")

    def record_outcome(self, won: bool) -> None:
        """Record game outcome for adaptive adjustment.

        Args:
            won: Whether the agent won.
        """
        self._recent_outcomes.append(won)
        if len(self._recent_outcomes) > self._window_size:
            self._recent_outcomes.pop(0)

        self._update_difficulty()

    def _update_difficulty(self) -> None:
        """Update difficulty modifier based on recent performance."""
        if len(self._recent_outcomes) < 10:
            return

        win_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)

        # Adjust difficulty to maintain target win rate
        error = win_rate - self.target_win_rate

        # Increase difficulty if winning too much, decrease if losing too much
        adjustment = error * self.adjustment_rate
        self._difficulty_modifier = np.clip(
            self._difficulty_modifier + adjustment,
            -0.5,
            0.5,
        )

        if abs(adjustment) > 0.01:
            self._logger.debug(
                f"Difficulty adjustment: modifier={self._difficulty_modifier:.3f}, "
                f"win_rate={win_rate:.2f}"
            )

    def apply_modifier(self, base_params: dict[str, Any]) -> dict[str, Any]:
        """Apply difficulty modifier to environment parameters.

        Args:
            base_params: Base environment parameters.

        Returns:
            Modified environment parameters.
        """
        params = base_params.copy()

        # Adjust max_turns (more difficult = more turns to win in)
        base_turns = params.get("max_turns", 100)
        params["max_turns"] = int(base_turns * (1 + self._difficulty_modifier * 0.5))

        # Adjust production (more difficult = slower production)
        base_prod = params.get("production_multiplier", 1.0)
        params["production_multiplier"] = base_prod * (1 - self._difficulty_modifier * 0.3)

        return params

    def get_difficulty_modifier(self) -> float:
        """Get current difficulty modifier.

        Returns:
            Difficulty modifier value.
        """
        return self._difficulty_modifier

    def reset(self) -> None:
        """Reset adaptive state."""
        self._difficulty_modifier = 0.0
        self._recent_outcomes.clear()


def create_curriculum_manager(
    num_stages: int = 3,
    **kwargs: Any,
) -> CurriculumManager:
    """Factory function to create curriculum manager.

    Args:
        num_stages: Number of curriculum stages (3 or 5).
        **kwargs: Additional configuration parameters.

    Returns:
        CurriculumManager instance.
    """
    if num_stages == 5:
        config = CurriculumConfig.default_five_stage()
    else:
        config = CurriculumConfig.default_three_stage()

    # Apply kwargs overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return CurriculumManager(config)
