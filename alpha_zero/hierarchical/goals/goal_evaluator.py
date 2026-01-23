"""
Goal Evaluator for Hierarchical Reasoning.

This module provides progress tracking, completion detection, and
reward shaping based on goal achievement.

Example:
    >>> from alpha_zero.hierarchical.goals import GoalEvaluator, GoalHierarchy
    >>> evaluator = GoalEvaluator()
    >>> progress = evaluator.evaluate_progress(goal, game_state)
    >>> reward = evaluator.compute_goal_reward(goal, progress)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from alpha_zero.hierarchical.goals.goal_types import (
    Goal,
    GoalStatus,
    HierarchyLevel,
    OperationalGoalType,
    StrategicGoalType,
    TacticalGoalType,
)

if TYPE_CHECKING:
    from alpha_zero.hierarchical.goals.goal_hierarchy import GoalHierarchy

logger = logging.getLogger(__name__)


class GameStateProtocol(Protocol):
    """Protocol for game state objects."""

    def get_player_score(self, player_id: int) -> float: ...
    def get_city_count(self, player_id: int) -> int: ...
    def get_unit_count(self, player_id: int) -> int: ...
    def get_tech_progress(self, tech_id: str) -> float: ...
    def get_position(self, entity_id: str) -> tuple[int, int] | None: ...


@dataclass
class EvaluationResult:
    """Result of goal evaluation.

    Attributes:
        goal_id: ID of evaluated goal.
        progress: Progress value (0.0 to 1.0).
        is_complete: Whether goal is complete.
        is_failed: Whether goal has failed.
        reward_delta: Change in reward due to progress.
        metrics: Additional evaluation metrics.
    """

    goal_id: str
    progress: float
    is_complete: bool = False
    is_failed: bool = False
    reward_delta: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class GoalEvaluator:
    """Evaluator for goal progress and completion.

    This class provides methods for evaluating goal progress based on
    game state, detecting completion conditions, and computing
    goal-based reward shaping.

    Attributes:
        progress_weight: Weight for progress-based rewards.
        completion_bonus: Bonus reward for goal completion.
        failure_penalty: Penalty for goal failure.
        timeout_penalty: Penalty for goal timeout.
    """

    progress_weight: float = 0.1
    completion_bonus: float = 1.0
    failure_penalty: float = -0.5
    timeout_penalty: float = -0.2

    # Level-specific weights
    level_weights: dict[HierarchyLevel, float] = field(
        default_factory=lambda: {
            HierarchyLevel.STRATEGIC: 1.0,
            HierarchyLevel.TACTICAL: 0.5,
            HierarchyLevel.OPERATIONAL: 0.1,
        }
    )

    def evaluate_progress(
        self,
        goal: Goal,
        game_state: dict[str, Any],
        previous_progress: float | None = None,
    ) -> EvaluationResult:
        """Evaluate progress for a goal.

        Args:
            goal: Goal to evaluate.
            game_state: Current game state.
            previous_progress: Previous progress value for delta calculation.

        Returns:
            EvaluationResult with progress and metrics.
        """
        # Route to appropriate evaluator based on level
        if goal.level == HierarchyLevel.STRATEGIC:
            progress = self._evaluate_strategic(goal, game_state)
        elif goal.level == HierarchyLevel.TACTICAL:
            progress = self._evaluate_tactical(goal, game_state)
        else:
            progress = self._evaluate_operational(goal, game_state)

        # Clamp progress
        progress = max(0.0, min(1.0, progress))

        # Check completion
        is_complete = progress >= 1.0

        # Check failure conditions
        is_failed = self._check_failure(goal, game_state)

        # Compute reward delta
        prev = previous_progress if previous_progress is not None else goal.progress
        reward_delta = self._compute_reward_delta(
            goal, progress, prev, is_complete, is_failed
        )

        return EvaluationResult(
            goal_id=goal.id,
            progress=progress,
            is_complete=is_complete,
            is_failed=is_failed,
            reward_delta=reward_delta,
            metrics={
                "level_weight": self.level_weights[goal.level],
                "previous_progress": prev,
                "progress_delta": progress - prev,
            },
        )

    def _evaluate_strategic(
        self,
        goal: Goal,
        game_state: dict[str, Any],
    ) -> float:
        """Evaluate strategic goal progress.

        Args:
            goal: Strategic goal.
            game_state: Game state.

        Returns:
            Progress value (0.0 to 1.0).
        """
        if not isinstance(goal.goal_type, StrategicGoalType):
            return goal.progress

        goal_type = goal.goal_type

        # Victory progress
        if goal_type == StrategicGoalType.DOMINATION_VICTORY:
            return self._evaluate_domination(game_state)
        if goal_type == StrategicGoalType.SCIENCE_VICTORY:
            return self._evaluate_science_victory(game_state)
        if goal_type == StrategicGoalType.CULTURE_VICTORY:
            return self._evaluate_culture_victory(game_state)

        # Posture progress
        if goal_type == StrategicGoalType.MILITARY_SUPREMACY:
            return self._evaluate_military_supremacy(game_state)
        if goal_type == StrategicGoalType.ECONOMIC_DOMINANCE:
            return self._evaluate_economic_dominance(game_state)
        if goal_type == StrategicGoalType.TERRITORIAL_EXPANSION:
            return self._evaluate_territorial_expansion(game_state)

        # Default: return current progress
        return goal.progress

    def _evaluate_tactical(
        self,
        goal: Goal,
        game_state: dict[str, Any],
    ) -> float:
        """Evaluate tactical goal progress.

        Args:
            goal: Tactical goal.
            game_state: Game state.

        Returns:
            Progress value (0.0 to 1.0).
        """
        if not isinstance(goal.goal_type, TacticalGoalType):
            return goal.progress

        goal_type = goal.goal_type
        target = goal.target

        # Military operations
        if goal_type == TacticalGoalType.CONQUER_CITY:
            return self._evaluate_city_conquest(target, game_state)
        if goal_type == TacticalGoalType.BUILD_ARMY:
            return self._evaluate_army_building(target, game_state)
        if goal_type == TacticalGoalType.DEFEND_CITY:
            return self._evaluate_city_defense(target, game_state)

        # Infrastructure
        if goal_type == TacticalGoalType.DEVELOP_CITY:
            return self._evaluate_city_development(target, game_state)
        if goal_type == TacticalGoalType.ESTABLISH_OUTPOST:
            return self._evaluate_outpost_establishment(target, game_state)

        # Research
        if goal_type == TacticalGoalType.RESEARCH_TECH:
            return self._evaluate_tech_research(target, game_state)
        if goal_type == TacticalGoalType.RESEARCH_CIVIC:
            return self._evaluate_civic_research(target, game_state)

        return goal.progress

    def _evaluate_operational(
        self,
        goal: Goal,
        game_state: dict[str, Any],
    ) -> float:
        """Evaluate operational goal progress.

        Args:
            goal: Operational goal.
            game_state: Game state.

        Returns:
            Progress value (0.0 to 1.0).
        """
        if not isinstance(goal.goal_type, OperationalGoalType):
            return goal.progress

        goal_type = goal.goal_type
        target = goal.target

        # Movement goals
        if goal_type == OperationalGoalType.MOVE_TO_POSITION:
            return self._evaluate_movement(target, game_state)
        if goal_type == OperationalGoalType.ATTACK_TARGET:
            return self._evaluate_attack(target, game_state)

        # Production goals
        if goal_type == OperationalGoalType.PRODUCE_UNIT:
            return self._evaluate_production(target, game_state)
        if goal_type == OperationalGoalType.PRODUCE_BUILDING:
            return self._evaluate_production(target, game_state)

        # Turn management (always complete on execution)
        if goal_type in {OperationalGoalType.END_TURN, OperationalGoalType.SKIP_TURN}:
            return 1.0

        return goal.progress

    # ==========================================================================
    # Strategic Evaluation Helpers
    # ==========================================================================

    def _evaluate_domination(self, state: dict[str, Any]) -> float:
        """Evaluate domination victory progress."""
        player_id = state.get("current_player", 0)
        total_capitals = state.get("total_capitals", 1)
        captured_capitals = state.get("captured_capitals", {}).get(player_id, 0)

        if total_capitals <= 1:
            return 1.0 if captured_capitals >= total_capitals else 0.0

        return captured_capitals / total_capitals

    def _evaluate_science_victory(self, state: dict[str, Any]) -> float:
        """Evaluate science victory progress."""
        player_id = state.get("current_player", 0)
        tech_levels = state.get("tech_levels", {})
        player_tech = tech_levels.get(player_id, 0)
        max_tech = state.get("max_tech_level", 100)

        return min(player_tech / max_tech, 1.0)

    def _evaluate_culture_victory(self, state: dict[str, Any]) -> float:
        """Evaluate culture victory progress."""
        player_id = state.get("current_player", 0)
        tourism = state.get("tourism", {}).get(player_id, 0)
        required_tourism = state.get("required_tourism", 1000)

        return min(tourism / required_tourism, 1.0)

    def _evaluate_military_supremacy(self, state: dict[str, Any]) -> float:
        """Evaluate military supremacy progress."""
        player_id = state.get("current_player", 0)
        military_strengths = state.get("military_strength", {})

        player_strength = military_strengths.get(player_id, 0)
        max_other = max(
            (s for pid, s in military_strengths.items() if pid != player_id),
            default=1,
        )

        # Progress is ratio of our strength to strongest opponent
        if max_other <= 0:
            return 1.0

        return min(player_strength / (max_other * 1.5), 1.0)

    def _evaluate_economic_dominance(self, state: dict[str, Any]) -> float:
        """Evaluate economic dominance progress."""
        player_id = state.get("current_player", 0)
        gold_incomes = state.get("gold_income", {})

        player_income = gold_incomes.get(player_id, 0)
        max_other = max(
            (g for pid, g in gold_incomes.items() if pid != player_id),
            default=1,
        )

        if max_other <= 0:
            return 1.0

        return min(player_income / (max_other * 1.2), 1.0)

    def _evaluate_territorial_expansion(self, state: dict[str, Any]) -> float:
        """Evaluate territorial expansion progress."""
        player_id = state.get("current_player", 0)
        city_counts = state.get("city_count", {})
        target_cities = state.get("target_city_count", 5)

        player_cities = city_counts.get(player_id, 0)
        return min(player_cities / target_cities, 1.0)

    # ==========================================================================
    # Tactical Evaluation Helpers
    # ==========================================================================

    def _evaluate_city_conquest(
        self,
        target: Any,
        state: dict[str, Any],
    ) -> float:
        """Evaluate city conquest progress."""
        if target.target_id is None:
            return 0.0

        captured_cities = state.get("captured_cities", [])
        if target.target_id in captured_cities:
            return 1.0

        # Check siege progress
        city_hp = state.get("city_health", {}).get(target.target_id, 100)
        return 1.0 - (city_hp / 100.0)

    def _evaluate_army_building(
        self,
        target: Any,
        state: dict[str, Any],
    ) -> float:
        """Evaluate army building progress."""
        player_id = state.get("current_player", 0)
        current_units = state.get("unit_count", {}).get(player_id, 0)
        target_units = target.metadata.get("target_units", 10)

        return min(current_units / target_units, 1.0)

    def _evaluate_city_defense(
        self,
        target: Any,
        state: dict[str, Any],
    ) -> float:
        """Evaluate city defense progress."""
        if target.target_id is None:
            return 1.0  # No specific city to defend

        # City still exists and hasn't been captured
        player_cities = state.get("player_cities", {}).get(
            state.get("current_player", 0), []
        )
        if target.target_id in player_cities:
            return 1.0

        return 0.0  # City lost

    def _evaluate_city_development(
        self,
        target: Any,
        state: dict[str, Any],
    ) -> float:
        """Evaluate city development progress."""
        if target.target_id is None:
            return 0.0

        city_pop = state.get("city_population", {}).get(target.target_id, 1)
        target_pop = target.metadata.get("target_population", 10)

        return min(city_pop / target_pop, 1.0)

    def _evaluate_outpost_establishment(
        self,
        target: Any,
        state: dict[str, Any],
    ) -> float:
        """Evaluate outpost establishment progress."""
        if target.position is None:
            return 0.0

        # Check if city exists at target position
        cities_by_pos = state.get("cities_by_position", {})
        if target.position in cities_by_pos:
            return 1.0

        # Check if settler is moving towards position
        settler_positions = state.get("settler_positions", [])
        if not settler_positions:
            return 0.0

        # Calculate distance progress
        min_dist = min(
            abs(s[0] - target.position[0]) + abs(s[1] - target.position[1])
            for s in settler_positions
        )
        max_dist = state.get("map_size", 32)

        return max(0.0, 1.0 - (min_dist / max_dist))

    def _evaluate_tech_research(
        self,
        target: Any,
        state: dict[str, Any],
    ) -> float:
        """Evaluate tech research progress."""
        if target.target_id is None:
            return 0.0

        tech_progress = state.get("tech_progress", {}).get(target.target_id, 0.0)
        return tech_progress

    def _evaluate_civic_research(
        self,
        target: Any,
        state: dict[str, Any],
    ) -> float:
        """Evaluate civic research progress."""
        if target.target_id is None:
            return 0.0

        civic_progress = state.get("civic_progress", {}).get(target.target_id, 0.0)
        return civic_progress

    # ==========================================================================
    # Operational Evaluation Helpers
    # ==========================================================================

    def _evaluate_movement(
        self,
        target: Any,
        state: dict[str, Any],
    ) -> float:
        """Evaluate movement goal progress."""
        if target.position is None:
            return 1.0  # No specific position

        unit_id = target.metadata.get("unit_id")
        if unit_id is None:
            return 0.0

        unit_pos = state.get("unit_positions", {}).get(unit_id)
        if unit_pos is None:
            return 0.0

        # Check if at target
        if unit_pos == target.position:
            return 1.0

        # Calculate distance
        dist = abs(unit_pos[0] - target.position[0]) + abs(unit_pos[1] - target.position[1])
        initial_dist = target.metadata.get("initial_distance", dist + 1)

        return max(0.0, 1.0 - (dist / initial_dist))

    def _evaluate_attack(
        self,
        target: Any,
        state: dict[str, Any],
    ) -> float:
        """Evaluate attack goal progress."""
        if target.target_id is None:
            return 0.0

        # Check if target destroyed
        destroyed_units = state.get("destroyed_units", [])
        if target.target_id in destroyed_units:
            return 1.0

        # Check target health
        target_hp = state.get("unit_health", {}).get(target.target_id, 100)
        return 1.0 - (target_hp / 100.0)

    def _evaluate_production(
        self,
        target: Any,
        state: dict[str, Any],
    ) -> float:
        """Evaluate production goal progress."""
        city_id = target.metadata.get("city_id")
        item_id = target.target_id

        if city_id is None or item_id is None:
            return 0.0

        # Check if item completed
        completed_items = state.get("completed_production", {}).get(city_id, [])
        if item_id in completed_items:
            return 1.0

        # Check production progress
        production_queue = state.get("production_queue", {}).get(city_id, {})
        if item_id in production_queue:
            return production_queue[item_id].get("progress", 0.0)

        return 0.0

    def _check_failure(
        self,
        goal: Goal,
        state: dict[str, Any],
    ) -> bool:
        """Check if goal has failed.

        Args:
            goal: Goal to check.
            state: Game state.

        Returns:
            True if goal has failed.
        """
        # Check timeout
        current_turn = state.get("turn", 0)
        if goal.deadline_turn is not None and current_turn > goal.deadline_turn:
            return True

        # Check for irreversible failure conditions
        # (e.g., target city captured by someone else)
        if isinstance(goal.goal_type, TacticalGoalType):
            if goal.goal_type == TacticalGoalType.CONQUER_CITY:
                target_city = goal.target.target_id
                if target_city:
                    owner = state.get("city_owners", {}).get(target_city)
                    player = state.get("current_player", 0)
                    other_captured = state.get("captured_by_others", [])
                    if target_city in other_captured:
                        return True

        return False

    def _compute_reward_delta(
        self,
        goal: Goal,
        new_progress: float,
        old_progress: float,
        is_complete: bool,
        is_failed: bool,
    ) -> float:
        """Compute reward change based on goal evaluation.

        Args:
            goal: Goal being evaluated.
            new_progress: New progress value.
            old_progress: Previous progress value.
            is_complete: Whether goal is complete.
            is_failed: Whether goal has failed.

        Returns:
            Reward delta value.
        """
        level_weight = self.level_weights[goal.level]
        reward = 0.0

        # Progress-based reward
        progress_delta = new_progress - old_progress
        reward += progress_delta * self.progress_weight * level_weight

        # Completion bonus
        if is_complete and old_progress < 1.0:
            reward += self.completion_bonus * level_weight

        # Failure penalty
        if is_failed:
            reward += self.failure_penalty * level_weight

        return reward

    def evaluate_batch(
        self,
        goals: list[Goal],
        game_state: dict[str, Any],
    ) -> list[EvaluationResult]:
        """Evaluate multiple goals at once.

        Args:
            goals: List of goals to evaluate.
            game_state: Current game state.

        Returns:
            List of evaluation results.
        """
        return [
            self.evaluate_progress(goal, game_state)
            for goal in goals
        ]

    def compute_total_goal_reward(
        self,
        hierarchy: GoalHierarchy,
        game_state: dict[str, Any],
    ) -> float:
        """Compute total goal-based reward for all active goals.

        Args:
            hierarchy: Goal hierarchy.
            game_state: Current game state.

        Returns:
            Total reward contribution from goals.
        """
        active_goals = hierarchy.get_active_goals()
        total_reward = 0.0

        for goal in active_goals:
            result = self.evaluate_progress(goal, game_state, goal.progress)
            total_reward += result.reward_delta

            # Update goal progress
            if result.is_complete:
                hierarchy.complete_goal(goal.id)
            elif result.is_failed:
                hierarchy.fail_goal(goal.id)
            else:
                hierarchy.update_progress(goal.id, result.progress)

        return total_reward
