"""Atari-specific adapter for the 3-layer hierarchical reasoning system.

Maps the core hierarchical controller (``alpha_zero.hierarchical``) onto
Atari games using domain-agnostic goal types.  The three levels are:

- **Strategic** (long horizon): Overall game-phase goals like
  MAXIMIZE_SCORE, SURVIVE, REACH_STAGE.
- **Tactical** (medium horizon): Multi-step sub-objectives like
  CLEAR_AREA, ACCUMULATE_POWERUP, AVOID_HAZARD.
- **Operational** (every step): Primitive actions (LEFT, RIGHT, FIRE …).

The adapter creates Atari-flavoured goal enums, a lightweight state encoder,
and a controller facade that wraps ``HierarchicalController``.

Example:
    >>> from demos.atari.hierarchical import AtariHierarchicalAdapter, AtariHierarchicalConfig
    >>> adapter = AtariHierarchicalAdapter(AtariHierarchicalConfig())
    >>> decision = adapter.decide(observation, valid_actions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from demos.atari.config import AtariHierarchicalConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Atari-generic goal enums
# ============================================================================


class AtariStrategicGoal(Enum):
    """Strategic-level goals applicable to any Atari game."""

    MAXIMIZE_SCORE = auto()
    SURVIVE = auto()
    REACH_NEXT_STAGE = auto()
    AGGRESSIVE_PLAY = auto()
    DEFENSIVE_PLAY = auto()
    BALANCED_PLAY = auto()


class AtariTacticalGoal(Enum):
    """Tactical-level goals — medium-horizon sub-objectives."""

    CLEAR_AREA = auto()
    AVOID_HAZARD = auto()
    ACCUMULATE_POWERUP = auto()
    POSITION_ADVANTAGE = auto()
    CHAIN_COMBO = auto()
    RECOVER_LIVES = auto()
    RUSH_TARGET = auto()
    HOLD_POSITION = auto()


class AtariOperationalGoal(Enum):
    """Operational-level — effectively action categories."""

    MOVE_LEFT = auto()
    MOVE_RIGHT = auto()
    MOVE_UP = auto()
    MOVE_DOWN = auto()
    FIRE = auto()
    NOOP = auto()


# ============================================================================
# Lightweight goal representation
# ============================================================================


@dataclass
class AtariGoal:
    """A single goal in the Atari hierarchy.

    Attributes:
        level: One of ``"strategic"``, ``"tactical"``, ``"operational"``.
        goal_type: The enum member for this goal.
        progress: Estimated progress toward completion (0-1).
        active: Whether the goal is currently being pursued.
        created_step: Environment step when the goal was created.
        metadata: Optional extra information.
    """

    level: str
    goal_type: AtariStrategicGoal | AtariTacticalGoal | AtariOperationalGoal
    progress: float = 0.0
    active: bool = True
    created_step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Hierarchical Adapter
# ============================================================================


@dataclass
class _LevelState:
    """Mutable state for a single hierarchy level."""

    active_goals: list[AtariGoal] = field(default_factory=list)
    last_update_step: int = 0
    cumulative_reward: float = 0.0


class AtariHierarchicalAdapter:
    """Adapts the 3-layer hierarchical reasoning system for Atari environments.

    This is a *lightweight* adapter that mirrors the interfaces of the core
    ``HierarchicalController`` without pulling in Civ6-specific encoders and
    goal types.  It can be used standalone or as a drop-in strategy layer
    inside the Atari training loop.

    Args:
        config: Hierarchical configuration.
        num_actions: Size of the Atari action space.
    """

    def __init__(
        self,
        config: AtariHierarchicalConfig | None = None,
        num_actions: int = 6,
    ) -> None:
        self.config = config or AtariHierarchicalConfig()
        self.num_actions = num_actions

        # Per-level mutable state
        self._strategic = _LevelState()
        self._tactical = _LevelState()
        self._operational = _LevelState()

        self._step: int = 0
        self._episode_reward: float = 0.0
        self._rng = np.random.default_rng()

        logger.info(
            "AtariHierarchicalAdapter initialised: enabled=%s, "
            "strategic_freq=%d, tactical_freq=%d",
            self.config.enabled,
            self.config.strategic_update_freq,
            self.config.tactical_update_freq,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        observation: dict[str, np.ndarray],
        valid_actions: np.ndarray | None = None,
        base_policy: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Make a hierarchical decision.

        When the hierarchy is *disabled*, this is a trivial pass-through that
        returns the ``base_policy`` unchanged.

        Args:
            observation: ``{spatial, vector, action_mask}`` dict.
            valid_actions: Action validity mask.
            base_policy: Policy from the flat MCTS / network (modified in-place).

        Returns:
            Dict with keys ``policy``, ``strategic_goals``, ``tactical_goals``,
            ``reward_shaping``, ``metadata``.
        """
        if not self.config.enabled:
            return {
                "policy": base_policy,
                "strategic_goals": [],
                "tactical_goals": [],
                "reward_shaping": 0.0,
                "metadata": {},
            }

        self._step += 1

        # --- Strategic layer (infrequent) ---
        if self._should_update_strategic():
            self._update_strategic(observation)

        # --- Tactical layer (moderate frequency) ---
        if self._should_update_tactical():
            self._update_tactical(observation)

        # --- Operational layer (every step) ---
        policy_adjustment = self._compute_operational_adjustment(observation, base_policy)

        # Compute goal-progress reward shaping
        reward_shaping = self._compute_reward_shaping(observation)

        return {
            "policy": policy_adjustment,
            "strategic_goals": [g.goal_type.name for g in self._strategic.active_goals],
            "tactical_goals": [g.goal_type.name for g in self._tactical.active_goals],
            "reward_shaping": reward_shaping,
            "metadata": {
                "step": self._step,
                "episode_reward": self._episode_reward,
            },
        }

    def update(self, reward: float, done: bool) -> None:
        """Update internal state after an environment step.

        Args:
            reward: Reward received from the environment.
            done: Whether the episode ended.
        """
        self._episode_reward += reward

        # Update goal progress heuristics
        self._update_goal_progress(reward)

        if done:
            self.reset()

    def reset(self) -> None:
        """Reset all hierarchy state for a new episode."""
        self._strategic = _LevelState()
        self._tactical = _LevelState()
        self._operational = _LevelState()
        self._step = 0
        self._episode_reward = 0.0
        logger.debug("Hierarchical adapter reset")

    def get_active_goals(self) -> dict[str, list[str]]:
        """Return a summary of currently active goals at each level.

        Returns:
            Dict mapping level names to lists of goal-type names.
        """
        return {
            "strategic": [g.goal_type.name for g in self._strategic.active_goals if g.active],
            "tactical": [g.goal_type.name for g in self._tactical.active_goals if g.active],
            "operational": [g.goal_type.name for g in self._operational.active_goals if g.active],
        }

    # ------------------------------------------------------------------
    # Strategic layer
    # ------------------------------------------------------------------

    def _should_update_strategic(self) -> bool:
        return (self._step - self._strategic.last_update_step) >= self.config.strategic_update_freq

    def _update_strategic(self, observation: dict[str, np.ndarray]) -> None:
        """Select / refresh strategic goals based on current game state."""
        self._strategic.last_update_step = self._step

        # Heuristic: choose strategy based on reward trajectory
        if self._episode_reward > 0:
            goal_type = AtariStrategicGoal.AGGRESSIVE_PLAY
        elif self._episode_reward < 0:
            goal_type = AtariStrategicGoal.DEFENSIVE_PLAY
        else:
            goal_type = AtariStrategicGoal.BALANCED_PLAY

        # Always keep MAXIMIZE_SCORE as the overarching strategic goal
        new_goals = [
            AtariGoal("strategic", AtariStrategicGoal.MAXIMIZE_SCORE, created_step=self._step),
            AtariGoal("strategic", goal_type, created_step=self._step),
        ]
        self._strategic.active_goals = new_goals[: self.config.max_active_goals_per_level]

        logger.debug(
            "Strategic update at step %d: %s",
            self._step,
            [g.goal_type.name for g in self._strategic.active_goals],
        )

    # ------------------------------------------------------------------
    # Tactical layer
    # ------------------------------------------------------------------

    def _should_update_tactical(self) -> bool:
        return (self._step - self._tactical.last_update_step) >= self.config.tactical_update_freq

    def _update_tactical(self, observation: dict[str, np.ndarray]) -> None:
        """Decompose strategic goals into tactical sub-goals."""
        self._tactical.last_update_step = self._step

        tactical_goals: list[AtariGoal] = []

        for s_goal in self._strategic.active_goals:
            if s_goal.goal_type == AtariStrategicGoal.AGGRESSIVE_PLAY:
                tactical_goals.append(
                    AtariGoal("tactical", AtariTacticalGoal.RUSH_TARGET, created_step=self._step)
                )
                tactical_goals.append(
                    AtariGoal("tactical", AtariTacticalGoal.CHAIN_COMBO, created_step=self._step)
                )
            elif s_goal.goal_type == AtariStrategicGoal.DEFENSIVE_PLAY:
                tactical_goals.append(
                    AtariGoal("tactical", AtariTacticalGoal.AVOID_HAZARD, created_step=self._step)
                )
                tactical_goals.append(
                    AtariGoal(
                        "tactical", AtariTacticalGoal.HOLD_POSITION, created_step=self._step
                    )
                )
            elif s_goal.goal_type == AtariStrategicGoal.MAXIMIZE_SCORE:
                tactical_goals.append(
                    AtariGoal("tactical", AtariTacticalGoal.CLEAR_AREA, created_step=self._step)
                )

        self._tactical.active_goals = tactical_goals[: self.config.max_active_goals_per_level]

        logger.debug(
            "Tactical update at step %d: %s",
            self._step,
            [g.goal_type.name for g in self._tactical.active_goals],
        )

    # ------------------------------------------------------------------
    # Operational layer
    # ------------------------------------------------------------------

    def _compute_operational_adjustment(
        self,
        observation: dict[str, np.ndarray],
        base_policy: np.ndarray | None,
    ) -> np.ndarray | None:
        """Optionally bias the base policy toward tactical objectives.

        Currently returns the base policy unchanged; a future extension
        could train a goal-conditioned policy head and merge logits here.
        """
        return base_policy

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _compute_reward_shaping(self, observation: dict[str, np.ndarray]) -> float:
        """Compute a small bonus/penalty based on goal progress.

        Returns:
            Shaped reward component (added to the environment reward).
        """
        if not self._tactical.active_goals:
            return 0.0

        progress_sum = sum(g.progress for g in self._tactical.active_goals if g.active)
        return float(progress_sum * self.config.goal_reward_shaping_weight)

    def _update_goal_progress(self, reward: float) -> None:
        """Heuristically update goal progress based on reward signals."""
        tactical_increment = self.config.tactical_progress_increment
        strategic_increment = self.config.strategic_progress_increment

        for goal in self._tactical.active_goals:
            if not goal.active:
                continue

            # Positive reward -> we're making progress on most goals
            if reward > 0:
                goal.progress = min(1.0, goal.progress + tactical_increment)

            # Mark completed goals
            if goal.progress >= 1.0:
                goal.active = False

        for goal in self._strategic.active_goals:
            if not goal.active:
                continue
            if reward > 0:
                goal.progress = min(1.0, goal.progress + strategic_increment)
