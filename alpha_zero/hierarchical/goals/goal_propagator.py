"""
Goal propagation across hierarchy levels.

This module handles cross-level goal communication, decomposition of high-level
goals into sub-goals, and aggregation of progress from lower levels upward.

Example:
    >>> from alpha_zero.hierarchical.goals import GoalPropagator, GoalHierarchy
    >>> hierarchy = GoalHierarchy()
    >>> propagator = GoalPropagator(hierarchy)
    >>> propagator.propagate_down(strategic_goal, game_state)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from alpha_zero.hierarchical.goals.goal_types import (
    Goal,
    GoalStatus,
    GoalTarget,
    HierarchyLevel,
    OperationalGoalType,
    StrategicGoalType,
    TacticalGoalType,
)

if TYPE_CHECKING:
    from alpha_zero.hierarchical.goals.goal_hierarchy import GoalHierarchy

logger = logging.getLogger(__name__)


@dataclass
class PropagationResult:
    """Result of a goal propagation operation.

    Attributes:
        success: Whether propagation succeeded.
        goals_created: New goals created during propagation.
        goals_updated: Existing goals that were updated.
        messages: Log messages from propagation.
    """

    success: bool = True
    goals_created: list[Goal] = field(default_factory=list)
    goals_updated: list[Goal] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def merge(self, other: PropagationResult) -> PropagationResult:
        """Merge another result into this one.

        Args:
            other: Result to merge.

        Returns:
            Self for chaining.
        """
        self.success = self.success and other.success
        self.goals_created.extend(other.goals_created)
        self.goals_updated.extend(other.goals_updated)
        self.messages.extend(other.messages)
        return self


class DecompositionStrategy(ABC):
    """Abstract base class for goal decomposition strategies."""

    @abstractmethod
    def decompose(
        self,
        goal: Goal,
        game_state: dict[str, Any],
        context: dict[str, Any],
    ) -> list[Goal]:
        """Decompose a goal into sub-goals.

        Args:
            goal: Goal to decompose.
            game_state: Current game state.
            context: Additional context for decomposition.

        Returns:
            List of sub-goals.
        """
        pass

    @abstractmethod
    def can_handle(self, goal: Goal) -> bool:
        """Check if this strategy can handle the goal.

        Args:
            goal: Goal to check.

        Returns:
            True if strategy can decompose this goal.
        """
        pass


class StrategicDecomposition(DecompositionStrategy):
    """Decomposition strategy for strategic goals."""

    # Mapping of strategic goals to tactical sub-goals
    DECOMPOSITION_MAP: dict[StrategicGoalType, list[TacticalGoalType]] = {
        StrategicGoalType.DOMINATION_VICTORY: [
            TacticalGoalType.CAPTURE_CITY,
            TacticalGoalType.BUILD_ARMY,
            TacticalGoalType.FORTIFY_POSITION,
            TacticalGoalType.ELIMINATE_UNIT,
        ],
        StrategicGoalType.SCIENCE_VICTORY: [
            TacticalGoalType.RESEARCH_TECH,
            TacticalGoalType.BUILD_CAMPUS,
            TacticalGoalType.DEVELOP_CITY,
        ],
        StrategicGoalType.CULTURE_VICTORY: [
            TacticalGoalType.BUILD_THEATER,
            TacticalGoalType.GENERATE_TOURISM,
            TacticalGoalType.BUILD_WONDER,
        ],
        StrategicGoalType.RELIGIOUS_VICTORY: [
            TacticalGoalType.SPREAD_RELIGION,
            TacticalGoalType.BUILD_HOLY_SITE,
        ],
        StrategicGoalType.DIPLOMATIC_VICTORY: [
            TacticalGoalType.FORM_ALLIANCE,
            TacticalGoalType.BUILD_WONDER,
        ],
        StrategicGoalType.EXPAND_TERRITORY: [
            TacticalGoalType.SETTLE_CITY,
            TacticalGoalType.IMPROVE_TILE,
            TacticalGoalType.SECURE_RESOURCE,
        ],
        StrategicGoalType.ECONOMIC_DOMINANCE: [
            TacticalGoalType.BUILD_COMMERCIAL,
            TacticalGoalType.ESTABLISH_TRADE,
            TacticalGoalType.DEVELOP_CITY,
        ],
        StrategicGoalType.MILITARY_SUPREMACY: [
            TacticalGoalType.BUILD_ARMY,
            TacticalGoalType.BUILD_ENCAMPMENT,
            TacticalGoalType.UPGRADE_UNIT,
        ],
        StrategicGoalType.DEFEND_EMPIRE: [
            TacticalGoalType.BUILD_ARMY,
            TacticalGoalType.FORTIFY_POSITION,
            TacticalGoalType.DEFEND_CITY,
        ],
        StrategicGoalType.ESTABLISH_ALLIANCE: [
            TacticalGoalType.FORM_ALLIANCE,
        ],
        StrategicGoalType.SABOTAGE_ENEMY: [
            TacticalGoalType.PILLAGE_IMPROVEMENT,
            TacticalGoalType.CAPTURE_CITY,
            TacticalGoalType.ELIMINATE_UNIT,
        ],
        StrategicGoalType.CONTROL_CONTINENT: [
            TacticalGoalType.SETTLE_CITY,
            TacticalGoalType.CAPTURE_CITY,
            TacticalGoalType.SECURE_RESOURCE,
        ],
        StrategicGoalType.TECHNOLOGICAL_LEAD: [
            TacticalGoalType.RESEARCH_TECH,
            TacticalGoalType.BUILD_CAMPUS,
        ],
        StrategicGoalType.CULTURAL_INFLUENCE: [
            TacticalGoalType.BUILD_THEATER,
            TacticalGoalType.GENERATE_TOURISM,
        ],
        StrategicGoalType.RELIGIOUS_SPREAD: [
            TacticalGoalType.SPREAD_RELIGION,
            TacticalGoalType.BUILD_HOLY_SITE,
        ],
        StrategicGoalType.RESOURCE_MONOPOLY: [
            TacticalGoalType.SECURE_RESOURCE,
            TacticalGoalType.ESTABLISH_TRADE,
        ],
    }

    def can_handle(self, goal: Goal) -> bool:
        """Check if goal is strategic level."""
        return goal.level == HierarchyLevel.STRATEGIC

    def decompose(
        self,
        goal: Goal,
        game_state: dict[str, Any],
        context: dict[str, Any],
    ) -> list[Goal]:
        """Decompose strategic goal into tactical sub-goals."""
        if not isinstance(goal.goal_type, StrategicGoalType):
            return []

        tactical_types = self.DECOMPOSITION_MAP.get(goal.goal_type, [])
        sub_goals = []

        for i, tactic_type in enumerate(tactical_types):
            # Create tactical goal with inherited properties
            sub_goal = Goal(
                id=f"{goal.id}_tactic_{i}",
                level=HierarchyLevel.TACTICAL,
                goal_type=tactic_type,
                target=self._derive_target(goal, tactic_type, game_state),
                priority=goal.priority * 0.8,  # Slightly lower priority
                parent_goal_id=goal.id,
                created_turn=context.get("current_turn", 0),
                deadline_turn=goal.deadline_turn,
                metadata={
                    "derived_from": goal.goal_type.name,
                    "decomposition_index": i,
                },
            )
            sub_goals.append(sub_goal)

        logger.debug(
            "Decomposed strategic goal %s into %d tactical goals",
            goal.id,
            len(sub_goals),
        )
        return sub_goals

    def _derive_target(
        self,
        parent: Goal,
        tactic_type: TacticalGoalType,
        game_state: dict[str, Any],
    ) -> GoalTarget:
        """Derive target for tactical goal from parent and game state."""
        # Default target inherits from parent
        target = GoalTarget(
            target_type=tactic_type.name.lower(),
        )

        # Specialize target based on tactic type and game state
        game_state.get("cities", [])
        game_state.get("units", [])

        if tactic_type == TacticalGoalType.CAPTURE_CITY and parent.target.position:
            target.position = parent.target.position
        elif tactic_type == TacticalGoalType.BUILD_ARMY:
            target.quantity = max(1, parent.target.quantity // 3) if parent.target.quantity else 3
        elif tactic_type == TacticalGoalType.SETTLE_CITY:
            target.quantity = 1
        elif tactic_type == TacticalGoalType.RESEARCH_TECH:
            target.quantity = 1
            target.entity_id = parent.target.entity_id  # Tech name

        return target


class TacticalDecomposition(DecompositionStrategy):
    """Decomposition strategy for tactical goals."""

    # Mapping of tactical goals to operational sub-goals
    DECOMPOSITION_MAP: dict[TacticalGoalType, list[OperationalGoalType]] = {
        TacticalGoalType.CAPTURE_CITY: [
            OperationalGoalType.MOVE_UNIT,
            OperationalGoalType.ATTACK_TARGET,
            OperationalGoalType.SIEGE_CITY,
        ],
        TacticalGoalType.DEFEND_CITY: [
            OperationalGoalType.MOVE_UNIT,
            OperationalGoalType.FORTIFY_UNIT,
            OperationalGoalType.HEAL_UNIT,
        ],
        TacticalGoalType.BUILD_ARMY: [
            OperationalGoalType.PRODUCE_UNIT,
            OperationalGoalType.UPGRADE_UNIT,
        ],
        TacticalGoalType.SETTLE_CITY: [
            OperationalGoalType.MOVE_UNIT,
            OperationalGoalType.FOUND_CITY,
        ],
        TacticalGoalType.DEVELOP_CITY: [
            OperationalGoalType.BUILD_BUILDING,
            OperationalGoalType.BUILD_IMPROVEMENT,
        ],
        TacticalGoalType.RESEARCH_TECH: [
            OperationalGoalType.SELECT_RESEARCH,
        ],
        TacticalGoalType.BUILD_CAMPUS: [
            OperationalGoalType.BUILD_DISTRICT,
            OperationalGoalType.BUILD_BUILDING,
        ],
        TacticalGoalType.BUILD_THEATER: [
            OperationalGoalType.BUILD_DISTRICT,
            OperationalGoalType.BUILD_BUILDING,
        ],
        TacticalGoalType.BUILD_HOLY_SITE: [
            OperationalGoalType.BUILD_DISTRICT,
            OperationalGoalType.BUILD_BUILDING,
        ],
        TacticalGoalType.BUILD_COMMERCIAL: [
            OperationalGoalType.BUILD_DISTRICT,
            OperationalGoalType.BUILD_BUILDING,
        ],
        TacticalGoalType.BUILD_ENCAMPMENT: [
            OperationalGoalType.BUILD_DISTRICT,
            OperationalGoalType.BUILD_BUILDING,
        ],
        TacticalGoalType.BUILD_WONDER: [
            OperationalGoalType.BUILD_BUILDING,
        ],
        TacticalGoalType.IMPROVE_TILE: [
            OperationalGoalType.MOVE_UNIT,
            OperationalGoalType.BUILD_IMPROVEMENT,
        ],
        TacticalGoalType.SECURE_RESOURCE: [
            OperationalGoalType.BUILD_IMPROVEMENT,
            OperationalGoalType.MOVE_UNIT,
        ],
        TacticalGoalType.ESTABLISH_TRADE: [
            OperationalGoalType.ESTABLISH_TRADE_ROUTE,
        ],
        TacticalGoalType.SPREAD_RELIGION: [
            OperationalGoalType.MOVE_UNIT,
            OperationalGoalType.SPREAD_RELIGION,
        ],
        TacticalGoalType.ELIMINATE_UNIT: [
            OperationalGoalType.MOVE_UNIT,
            OperationalGoalType.ATTACK_TARGET,
        ],
        TacticalGoalType.FORTIFY_POSITION: [
            OperationalGoalType.MOVE_UNIT,
            OperationalGoalType.FORTIFY_UNIT,
        ],
        TacticalGoalType.PILLAGE_IMPROVEMENT: [
            OperationalGoalType.MOVE_UNIT,
            OperationalGoalType.PILLAGE,
        ],
        TacticalGoalType.GENERATE_TOURISM: [
            OperationalGoalType.BUILD_BUILDING,
            OperationalGoalType.CREATE_GREAT_WORK,
        ],
        TacticalGoalType.FORM_ALLIANCE: [
            OperationalGoalType.SEND_DELEGATION,
            OperationalGoalType.MAKE_DEAL,
        ],
        TacticalGoalType.UPGRADE_UNIT: [
            OperationalGoalType.UPGRADE_UNIT,
        ],
    }

    def can_handle(self, goal: Goal) -> bool:
        """Check if goal is tactical level."""
        return goal.level == HierarchyLevel.TACTICAL

    def decompose(
        self,
        goal: Goal,
        game_state: dict[str, Any],
        context: dict[str, Any],
    ) -> list[Goal]:
        """Decompose tactical goal into operational sub-goals."""
        if not isinstance(goal.goal_type, TacticalGoalType):
            return []

        operational_types = self.DECOMPOSITION_MAP.get(goal.goal_type, [])
        sub_goals = []

        for i, op_type in enumerate(operational_types):
            sub_goal = Goal(
                id=f"{goal.id}_op_{i}",
                level=HierarchyLevel.OPERATIONAL,
                goal_type=op_type,
                target=self._derive_target(goal, op_type, game_state),
                priority=goal.priority * 0.9,
                parent_goal_id=goal.id,
                created_turn=context.get("current_turn", 0),
                deadline_turn=min(
                    goal.deadline_turn or float("inf"),
                    context.get("current_turn", 0) + 5,  # Operational goals are short-term
                ),
                metadata={
                    "derived_from": goal.goal_type.name,
                    "decomposition_index": i,
                },
            )
            sub_goals.append(sub_goal)

        logger.debug(
            "Decomposed tactical goal %s into %d operational goals",
            goal.id,
            len(sub_goals),
        )
        return sub_goals

    def _derive_target(
        self,
        parent: Goal,
        op_type: OperationalGoalType,
        game_state: dict[str, Any],
    ) -> GoalTarget:
        """Derive target for operational goal."""
        target = GoalTarget(
            target_type=op_type.name.lower(),
        )

        # Inherit location from parent if applicable
        if parent.target.position:
            target.position = parent.target.position

        # Set quantities for production goals
        if op_type in (OperationalGoalType.PRODUCE_UNIT, OperationalGoalType.BUILD_BUILDING):
            target.quantity = 1

        return target


class GoalPropagator:
    """Handles cross-level goal communication and propagation.

    This class manages:
    - Downward propagation: Decomposing high-level goals into sub-goals
    - Upward propagation: Aggregating progress from sub-goals to parents
    - Lateral propagation: Coordinating related goals at the same level

    Attributes:
        hierarchy: The goal hierarchy to propagate within.
        strategies: List of decomposition strategies.
        propagation_hooks: Callbacks for propagation events.
    """

    def __init__(
        self,
        hierarchy: GoalHierarchy,
        strategies: list[DecompositionStrategy] | None = None,
    ) -> None:
        """Initialize the goal propagator.

        Args:
            hierarchy: Goal hierarchy to manage.
            strategies: Custom decomposition strategies. If None, uses defaults.
        """
        self.hierarchy = hierarchy
        self.strategies = strategies or [
            StrategicDecomposition(),
            TacticalDecomposition(),
        ]
        self.propagation_hooks: list[Callable[[PropagationResult], None]] = []

        logger.info(
            "Initialized GoalPropagator with %d strategies",
            len(self.strategies),
        )

    def register_hook(
        self,
        hook: Callable[[PropagationResult], None],
    ) -> None:
        """Register a hook to be called after propagation.

        Args:
            hook: Callback function.
        """
        self.propagation_hooks.append(hook)

    def propagate_down(
        self,
        goal: Goal,
        game_state: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> PropagationResult:
        """Propagate a goal downward by decomposing into sub-goals.

        Args:
            goal: Goal to decompose.
            game_state: Current game state for context.
            context: Additional context (current_turn, etc.).

        Returns:
            PropagationResult with created sub-goals.
        """
        context = context or {}
        result = PropagationResult()

        # Skip if goal is already at operational level
        if goal.level == HierarchyLevel.OPERATIONAL:
            result.messages.append(f"Goal {goal.id} at operational level, no decomposition needed")
            return result

        # Find appropriate strategy
        strategy = self._find_strategy(goal)
        if not strategy:
            result.success = False
            result.messages.append(f"No decomposition strategy for goal {goal.id}")
            logger.warning("No decomposition strategy for goal %s", goal.id)
            return result

        # Decompose
        sub_goals = strategy.decompose(goal, game_state, context)

        # Add sub-goals to hierarchy
        for sub_goal in sub_goals:
            try:
                self.hierarchy.add_goal(sub_goal, parent_id=goal.id)
                result.goals_created.append(sub_goal)
            except Exception as e:
                result.messages.append(f"Failed to add sub-goal {sub_goal.id}: {e}")
                logger.error("Failed to add sub-goal %s: %s", sub_goal.id, e)

        # Update parent goal
        goal.child_goal_ids.extend([g.id for g in sub_goals])
        result.goals_updated.append(goal)
        result.messages.append(
            f"Decomposed {goal.id} into {len(sub_goals)} sub-goals"
        )

        # Call hooks
        for hook in self.propagation_hooks:
            try:
                hook(result)
            except Exception as e:
                logger.error("Propagation hook error: %s", e)

        return result

    def propagate_up(
        self,
        goal: Goal,
        game_state: dict[str, Any],
    ) -> PropagationResult:
        """Propagate progress from a goal upward to its parent.

        Args:
            goal: Goal with updated progress.
            game_state: Current game state.

        Returns:
            PropagationResult with updated parent goals.
        """
        result = PropagationResult()

        if not goal.parent_goal_id:
            result.messages.append(f"Goal {goal.id} has no parent")
            return result

        # Get parent
        parent = self.hierarchy.get_goal(goal.parent_goal_id)
        if not parent:
            result.success = False
            result.messages.append(f"Parent {goal.parent_goal_id} not found")
            return result

        # Aggregate progress from all children
        children = [
            self.hierarchy.get_goal(cid)
            for cid in parent.child_goal_ids
            if self.hierarchy.get_goal(cid)
        ]

        if children:
            # Weight by priority
            total_priority = sum(c.priority for c in children)
            if total_priority > 0:
                weighted_progress = sum(
                    c.progress * c.priority for c in children
                ) / total_priority

                old_progress = parent.progress
                parent.progress = weighted_progress

                result.goals_updated.append(parent)
                result.messages.append(
                    f"Updated {parent.id} progress: {old_progress:.2f} -> {weighted_progress:.2f}"
                )

        # Check if parent should be completed
        if parent.progress >= 1.0:
            parent.status = GoalStatus.COMPLETED
            result.messages.append(f"Goal {parent.id} completed via sub-goal progress")

        # Recursively propagate up
        if parent.parent_goal_id:
            sub_result = self.propagate_up(parent, game_state)
            result.merge(sub_result)

        return result

    def propagate_lateral(
        self,
        goal: Goal,
        game_state: dict[str, Any],
    ) -> PropagationResult:
        """Propagate goal effects to related goals at the same level.

        This handles synergies and conflicts between goals.

        Args:
            goal: Goal that was updated.
            game_state: Current game state.

        Returns:
            PropagationResult with affected goals.
        """
        result = PropagationResult()

        # Get sibling goals (same level, same parent or no parent)
        siblings = self.hierarchy.get_active_goals(goal.level)
        siblings = [s for s in siblings if s.id != goal.id]

        for sibling in siblings:
            # Check for synergy or conflict
            relationship = self._analyze_relationship(goal, sibling, game_state)

            if relationship == "synergy":
                # Boost sibling progress slightly
                boost = min(0.1, goal.progress * 0.1)
                sibling.progress = min(1.0, sibling.progress + boost)
                result.goals_updated.append(sibling)
                result.messages.append(
                    f"Synergy boost to {sibling.id}: +{boost:.2f}"
                )
            elif relationship == "conflict":
                # Reduce sibling priority
                sibling.priority *= 0.9
                result.goals_updated.append(sibling)
                result.messages.append(
                    f"Conflict penalty to {sibling.id} priority"
                )

        return result

    def full_propagation(
        self,
        goal: Goal,
        game_state: dict[str, Any],
        context: dict[str, Any] | None = None,
        direction: str = "both",
    ) -> PropagationResult:
        """Perform full propagation in specified direction(s).

        Args:
            goal: Starting goal for propagation.
            game_state: Current game state.
            context: Additional context.
            direction: "down", "up", "lateral", or "both" (down + up).

        Returns:
            Combined PropagationResult.
        """
        result = PropagationResult()
        context = context or {}

        if direction in ("down", "both"):
            down_result = self.propagate_down(goal, game_state, context)
            result.merge(down_result)

            # Recursively propagate down to created sub-goals
            for sub_goal in down_result.goals_created:
                sub_result = self.full_propagation(
                    sub_goal, game_state, context, "down"
                )
                result.merge(sub_result)

        if direction in ("up", "both"):
            up_result = self.propagate_up(goal, game_state)
            result.merge(up_result)

        if direction == "lateral":
            lateral_result = self.propagate_lateral(goal, game_state)
            result.merge(lateral_result)

        return result

    def synchronize_hierarchy(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> PropagationResult:
        """Synchronize entire hierarchy after game state changes.

        This performs:
        1. Progress updates based on current state
        2. Upward propagation of all leaf goals
        3. Status updates for completed/failed goals
        4. Cleanup of abandoned goals

        Args:
            game_state: Current game state.
            context: Additional context.

        Returns:
            PropagationResult summarizing all changes.
        """
        context = context or {}
        result = PropagationResult()
        current_turn = context.get("current_turn", 0)

        # Process from bottom up (operational -> tactical -> strategic)
        for level in [HierarchyLevel.OPERATIONAL, HierarchyLevel.TACTICAL, HierarchyLevel.STRATEGIC]:
            goals = self.hierarchy.get_active_goals(level)

            for goal in goals:
                # Check for deadline
                if goal.deadline_turn and current_turn >= goal.deadline_turn:
                    if goal.progress < 1.0:
                        goal.status = GoalStatus.FAILED
                        result.goals_updated.append(goal)
                        result.messages.append(f"Goal {goal.id} failed (deadline)")
                        continue

                # Check for abandonment (no progress)
                turns_since_progress = current_turn - goal.last_progress_turn
                abandon_threshold = context.get("abandon_threshold", 20)
                if turns_since_progress > abandon_threshold and goal.progress < 0.5:
                    goal.status = GoalStatus.ABANDONED
                    result.goals_updated.append(goal)
                    result.messages.append(f"Goal {goal.id} abandoned (no progress)")
                    continue

                # Propagate up if leaf goal with parent
                if not goal.child_goal_ids and goal.parent_goal_id:
                    up_result = self.propagate_up(goal, game_state)
                    result.merge(up_result)

        return result

    def _find_strategy(self, goal: Goal) -> DecompositionStrategy | None:
        """Find decomposition strategy for a goal."""
        for strategy in self.strategies:
            if strategy.can_handle(goal):
                return strategy
        return None

    def _analyze_relationship(
        self,
        goal1: Goal,
        goal2: Goal,
        game_state: dict[str, Any],
    ) -> str:
        """Analyze relationship between two goals.

        Returns:
            "synergy", "conflict", or "neutral".
        """
        # Define synergistic and conflicting goal type pairs
        synergies: set[tuple[Any, Any]] = {
            (StrategicGoalType.SCIENCE_VICTORY, StrategicGoalType.TECHNOLOGICAL_LEAD),
            (StrategicGoalType.CULTURE_VICTORY, StrategicGoalType.CULTURAL_INFLUENCE),
            (StrategicGoalType.RELIGIOUS_VICTORY, StrategicGoalType.RELIGIOUS_SPREAD),
            (StrategicGoalType.DOMINATION_VICTORY, StrategicGoalType.MILITARY_SUPREMACY),
            (TacticalGoalType.BUILD_ARMY, TacticalGoalType.BUILD_ENCAMPMENT),
            (TacticalGoalType.BUILD_CAMPUS, TacticalGoalType.RESEARCH_TECH),
            (TacticalGoalType.BUILD_THEATER, TacticalGoalType.GENERATE_TOURISM),
        }

        conflicts: set[tuple[Any, Any]] = {
            (StrategicGoalType.DOMINATION_VICTORY, StrategicGoalType.DIPLOMATIC_VICTORY),
            (StrategicGoalType.EXPAND_TERRITORY, StrategicGoalType.DEFEND_EMPIRE),
            (TacticalGoalType.BUILD_ARMY, TacticalGoalType.DEVELOP_CITY),  # Resource conflict
        }

        pair = (goal1.goal_type, goal2.goal_type)
        reverse_pair = (goal2.goal_type, goal1.goal_type)

        if pair in synergies or reverse_pair in synergies:
            return "synergy"
        if pair in conflicts or reverse_pair in conflicts:
            return "conflict"

        return "neutral"
