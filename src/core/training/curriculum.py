"""Curriculum scheduler for multi-stage training progression.

Manages stage transitions based on Elo ratings with support for
promotion, demotion, and stage-specific configuration overrides.

Layer 3 -- depends on config, types, logging.
"""
from __future__ import annotations

__all__ = ["CurriculumScheduler", "StageStats"]

import logging
from dataclasses import dataclass, field

from src.core.config import CurriculumConfig, CurriculumStage
from src.core.constants import INITIAL_ELO_RATING

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StageStats:
    """Statistics tracked per curriculum stage."""

    stage_name: str
    games_played: int = 0
    iterations_completed: int = 0
    current_elo: float = INITIAL_ELO_RATING
    peak_elo: float = INITIAL_ELO_RATING
    promotion_count: int = 0
    demotion_count: int = 0
    elo_history: list[float] = field(default_factory=list)

    def record_elo(self, elo: float) -> None:
        """Record an Elo measurement."""
        self.current_elo = elo
        self.peak_elo = max(self.peak_elo, elo)
        self.elo_history.append(elo)


class CurriculumScheduler:
    """Manages multi-stage curriculum progression for training.

    The scheduler tracks training progress through a sequence of
    difficulty stages.  Promotion and demotion are governed by Elo
    thresholds computed from a sliding window of recent evaluations.

    Args:
        config: Curriculum configuration specifying stages and thresholds.
        initial_stage_index: Starting stage index (default 0).
    """

    def __init__(
        self,
        config: CurriculumConfig,
        initial_stage_index: int = 0,
    ) -> None:
        if not config.stages:
            raise ValueError("CurriculumConfig must have at least one stage")
        if initial_stage_index < 0 or initial_stage_index >= len(config.stages):
            raise ValueError(
                f"initial_stage_index {initial_stage_index} out of range "
                f"[0, {len(config.stages)})"
            )

        self._config = config
        self._current_index = initial_stage_index
        self._stage_stats: dict[str, StageStats] = {
            stage.name: StageStats(stage_name=stage.name)
            for stage in config.stages
        }

    @property
    def current_stage(self) -> CurriculumStage:
        """The currently active curriculum stage."""
        return self._config.stages[self._current_index]

    @property
    def current_stage_index(self) -> int:
        return self._current_index

    @property
    def current_stage_name(self) -> str:
        return self.current_stage.name

    @property
    def num_stages(self) -> int:
        return len(self._config.stages)

    @property
    def is_final_stage(self) -> bool:
        return self._current_index >= self.num_stages - 1

    @property
    def current_stats(self) -> StageStats:
        """Stats for the current stage."""
        return self._stage_stats[self.current_stage.name]

    def get_stage_stats(self, stage_name: str) -> StageStats | None:
        """Get stats for a specific stage by name."""
        return self._stage_stats.get(stage_name)

    def record_iteration(
        self,
        elo: float,
        games_played: int = 0,
    ) -> str | None:
        """Record completion of a training iteration.

        Updates stage stats and checks promotion/demotion criteria.

        Args:
            elo: Current Elo rating from evaluation.
            games_played: Number of games played in this iteration.

        Returns:
            New stage name if a transition occurred, or None if no change.
        """
        stats = self.current_stats
        stats.iterations_completed += 1
        stats.games_played += games_played
        stats.record_elo(elo)
        logger.debug(
            "Curriculum iteration recorded: stage=%s, elo=%.1f, iter=%d",
            self.current_stage_name, elo, stats.iterations_completed,
        )

        # Check for promotion
        if self._should_promote(stats):
            return self._promote()

        # Check for demotion
        if self._should_demote(stats):
            return self._demote()

        return None

    def _should_promote(self, stats: StageStats) -> bool:
        """Check if current stage meets promotion criteria."""
        if self.is_final_stage:
            return False

        stage = self.current_stage
        # Must meet minimum iterations
        if stats.iterations_completed < stage.min_iterations:
            return False

        # Check Elo threshold over sliding window
        window = self._config.promotion_window
        if len(stats.elo_history) < window:
            return False

        recent_elos = stats.elo_history[-window:]
        avg_elo = sum(recent_elos) / len(recent_elos)

        return avg_elo >= self._config.promotion_elo_threshold

    def _should_demote(self, stats: StageStats) -> bool:
        """Check if current stage meets demotion criteria."""
        if not self._config.demotion_enabled:
            return False

        if self._current_index == 0:
            return False

        # Check Elo threshold over sliding window
        window = self._config.promotion_window
        if len(stats.elo_history) < window:
            return False

        recent_elos = stats.elo_history[-window:]
        avg_elo = sum(recent_elos) / len(recent_elos)

        return avg_elo < self._config.demotion_elo_threshold

    def _promote(self) -> str | None:
        """Advance to the next curriculum stage."""
        if self.is_final_stage:
            return None

        old_name = self.current_stage_name
        self.current_stats.promotion_count += 1
        self._current_index += 1
        new_name = self.current_stage_name

        logger.info(
            "Curriculum promotion: %s → %s (index %d)",
            old_name,
            new_name,
            self._current_index,
        )
        return new_name

    def _demote(self) -> str | None:
        """Revert to the previous curriculum stage."""
        if self._current_index == 0:
            return None

        old_name = self.current_stage_name
        self.current_stats.demotion_count += 1
        self._current_index -= 1
        new_name = self.current_stage_name

        logger.info(
            "Curriculum demotion: %s → %s (index %d)",
            old_name,
            new_name,
            self._current_index,
        )
        return new_name

    def force_stage(self, stage_index: int) -> str:
        """Force transition to a specific stage.

        Args:
            stage_index: Target stage index.

        Returns:
            Name of the new stage.

        Raises:
            ValueError: If stage_index is out of range.
        """
        if stage_index < 0 or stage_index >= self.num_stages:
            raise ValueError(
                f"stage_index {stage_index} out of range [0, {self.num_stages})"
            )
        self._current_index = stage_index
        logger.info(
            "Forced curriculum stage: %s (index %d)",
            self.current_stage_name,
            stage_index,
        )
        return self.current_stage_name

    def get_summary(self) -> dict[str, object]:
        """Return a summary of curriculum state for logging/metrics."""
        return {
            "current_stage": self.current_stage_name,
            "current_index": self._current_index,
            "num_stages": self.num_stages,
            "is_final": self.is_final_stage,
            "iterations_at_stage": self.current_stats.iterations_completed,
            "games_at_stage": self.current_stats.games_played,
            "elo_at_stage": self.current_stats.current_elo,
            "peak_elo_at_stage": self.current_stats.peak_elo,
        }
