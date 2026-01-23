"""
Goal Hierarchy Manager for Hierarchical Reasoning.

This module provides the main interface for managing the goal hierarchy,
including creation, querying, and lifecycle management of goals.

Example:
    >>> from alpha_zero.hierarchical.goals import GoalHierarchy, Goal, StrategicGoalType
    >>> hierarchy = GoalHierarchy()
    >>> strategic = Goal.strategic(StrategicGoalType.DOMINATION_VICTORY)
    >>> hierarchy.add_goal(strategic)
    >>> active = hierarchy.get_active_goals(HierarchyLevel.STRATEGIC)
"""
from __future__ import annotations

import logging
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
from alpha_zero.hierarchical.goals.goal_node import GoalNode

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class GoalHierarchy:
    """Manager for the goal hierarchy.

    This class provides the main interface for working with the goal
    hierarchy, including adding, removing, and querying goals at
    different levels.

    Attributes:
        roots: Root nodes (strategic goals with no parent).
        nodes_by_id: Index of all nodes by goal ID.
        current_turn: Current game turn for temporal tracking.
        max_active_per_level: Maximum active goals per level.
    """

    roots: list[GoalNode] = field(default_factory=list)
    nodes_by_id: dict[str, GoalNode] = field(default_factory=dict)
    current_turn: int = 0
    max_active_per_level: dict[HierarchyLevel, int] = field(
        default_factory=lambda: {
            HierarchyLevel.STRATEGIC: 3,
            HierarchyLevel.TACTICAL: 5,
            HierarchyLevel.OPERATIONAL: 10,
        }
    )

    # Statistics tracking
    _total_goals_created: int = 0
    _total_goals_completed: int = 0
    _total_goals_failed: int = 0

    def add_goal(
        self,
        goal: Goal,
        parent_id: str | None = None,
    ) -> GoalNode:
        """Add a goal to the hierarchy.

        Args:
            goal: Goal to add.
            parent_id: Optional parent goal ID.

        Returns:
            Created GoalNode.

        Raises:
            ValueError: If parent not found or level mismatch.
        """
        # Set creation turn
        goal.created_turn = self.current_turn

        # Create node
        node = GoalNode(goal=goal)

        # Add to index
        self.nodes_by_id[goal.id] = node

        # Connect to parent or add as root
        if parent_id is not None:
            parent_node = self.nodes_by_id.get(parent_id)
            if parent_node is None:
                raise ValueError(f"Parent goal not found: {parent_id}")

            # Validate level hierarchy
            if not self._validate_level_order(parent_node.level, goal.level):
                raise ValueError(
                    f"Invalid level order: {parent_node.level.value} -> {goal.level.value}"
                )

            parent_node.add_child(node)
        else:
            # Root nodes should be strategic
            if goal.level != HierarchyLevel.STRATEGIC:
                logger.warning(
                    f"Adding non-strategic goal {goal.id} as root"
                )
            self.roots.append(node)

        self._total_goals_created += 1
        logger.info(
            f"Added goal: {goal.goal_type.name} "
            f"(level={goal.level.value}, id={goal.id})"
        )
        return node

    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal and its subtree from the hierarchy.

        Args:
            goal_id: ID of goal to remove.

        Returns:
            True if goal was found and removed.
        """
        node = self.nodes_by_id.get(goal_id)
        if node is None:
            return False

        # Remove from parent's children
        if node.parent is not None:
            node.parent.remove_child(goal_id)
        else:
            # Remove from roots
            self.roots = [r for r in self.roots if r.id != goal_id]

        # Remove entire subtree from index
        for descendant in node.traverse_preorder():
            self.nodes_by_id.pop(descendant.id, None)

        logger.info(f"Removed goal: {goal_id}")
        return True

    def get_goal(self, goal_id: str) -> Goal | None:
        """Get a goal by ID.

        Args:
            goal_id: Goal ID.

        Returns:
            Goal if found, None otherwise.
        """
        node = self.nodes_by_id.get(goal_id)
        return node.goal if node else None

    def get_node(self, goal_id: str) -> GoalNode | None:
        """Get a node by goal ID.

        Args:
            goal_id: Goal ID.

        Returns:
            GoalNode if found, None otherwise.
        """
        return self.nodes_by_id.get(goal_id)

    def get_goals_by_level(self, level: HierarchyLevel) -> list[Goal]:
        """Get all goals at a specific level.

        Args:
            level: Hierarchy level.

        Returns:
            List of goals at the level.
        """
        return [
            node.goal for node in self.nodes_by_id.values()
            if node.level == level
        ]

    def get_active_goals(self, level: HierarchyLevel | None = None) -> list[Goal]:
        """Get all active goals, optionally filtered by level.

        Args:
            level: Optional level filter.

        Returns:
            List of active goals.
        """
        goals = [
            node.goal for node in self.nodes_by_id.values()
            if node.goal.status == GoalStatus.ACTIVE
        ]
        if level is not None:
            goals = [g for g in goals if g.level == level]
        return goals

    def get_pending_goals(self, level: HierarchyLevel | None = None) -> list[Goal]:
        """Get all pending goals, optionally filtered by level.

        Args:
            level: Optional level filter.

        Returns:
            List of pending goals.
        """
        goals = [
            node.goal for node in self.nodes_by_id.values()
            if node.goal.status == GoalStatus.PENDING
        ]
        if level is not None:
            goals = [g for g in goals if g.level == level]
        return goals

    def get_parent_goal(self, goal_id: str) -> Goal | None:
        """Get the parent goal of a goal.

        Args:
            goal_id: Child goal ID.

        Returns:
            Parent goal if exists, None otherwise.
        """
        node = self.nodes_by_id.get(goal_id)
        if node is None or node.parent is None:
            return None
        return node.parent.goal

    def get_child_goals(self, goal_id: str) -> list[Goal]:
        """Get all direct child goals of a goal.

        Args:
            goal_id: Parent goal ID.

        Returns:
            List of child goals.
        """
        node = self.nodes_by_id.get(goal_id)
        if node is None:
            return []
        return [c.goal for c in node.children]

    def get_ancestor_goals(self, goal_id: str) -> list[Goal]:
        """Get all ancestor goals from root to parent.

        Args:
            goal_id: Goal ID.

        Returns:
            List of ancestor goals (root first).
        """
        node = self.nodes_by_id.get(goal_id)
        if node is None:
            return []
        return [n.goal for n in node.get_ancestors()]

    def activate_goal(self, goal_id: str) -> bool:
        """Activate a goal.

        Args:
            goal_id: Goal ID to activate.

        Returns:
            True if goal was activated.
        """
        node = self.nodes_by_id.get(goal_id)
        if node is None:
            return False

        # Check if we can activate more goals at this level
        active_count = len(self.get_active_goals(node.level))
        max_active = self.max_active_per_level.get(node.level, 5)

        if active_count >= max_active:
            logger.warning(
                f"Cannot activate goal {goal_id}: "
                f"max active ({max_active}) reached for {node.level.value}"
            )
            return False

        node.update_status(GoalStatus.ACTIVE)
        return True

    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed.

        Args:
            goal_id: Goal ID to complete.

        Returns:
            True if goal was completed.
        """
        node = self.nodes_by_id.get(goal_id)
        if node is None:
            return False

        node.update_status(GoalStatus.COMPLETED)
        self._total_goals_completed += 1
        return True

    def fail_goal(self, goal_id: str) -> bool:
        """Mark a goal as failed.

        Args:
            goal_id: Goal ID to fail.

        Returns:
            True if goal was marked as failed.
        """
        node = self.nodes_by_id.get(goal_id)
        if node is None:
            return False

        node.update_status(GoalStatus.FAILED)
        self._total_goals_failed += 1
        return True

    def update_progress(self, goal_id: str, progress: float) -> bool:
        """Update progress for a goal.

        Args:
            goal_id: Goal ID.
            progress: New progress value (0.0 to 1.0).

        Returns:
            True if progress was updated.
        """
        node = self.nodes_by_id.get(goal_id)
        if node is None:
            return False

        old_progress = node.goal.progress
        node.goal.update_progress(progress)

        logger.debug(
            f"Goal {goal_id} progress: {old_progress:.1%} -> {progress:.1%}"
        )
        return True

    def set_turn(self, turn: int) -> None:
        """Set the current game turn.

        Args:
            turn: Current turn number.
        """
        self.current_turn = turn

    def get_highest_priority_goal(
        self,
        level: HierarchyLevel,
        status: GoalStatus | None = None,
    ) -> Goal | None:
        """Get the highest priority goal at a level.

        Args:
            level: Hierarchy level.
            status: Optional status filter.

        Returns:
            Highest priority goal, or None if none found.
        """
        goals = self.get_goals_by_level(level)
        if status is not None:
            goals = [g for g in goals if g.status == status]

        if not goals:
            return None

        return max(goals, key=lambda g: g.priority)

    def get_context_for_level(
        self,
        level: HierarchyLevel,
    ) -> dict[str, Any]:
        """Get goal context for a specific level.

        This method provides the information that lower levels need
        to condition their decisions.

        Args:
            level: Level to get context for.

        Returns:
            Dictionary containing parent goals and constraints.
        """
        context: dict[str, Any] = {
            "level": level.value,
            "turn": self.current_turn,
            "active_goals": [],
            "parent_goals": [],
        }

        # Get active goals at this level
        active = self.get_active_goals(level)
        context["active_goals"] = [g.to_dict() for g in active]

        # Get parent level goals
        if level == HierarchyLevel.TACTICAL:
            parent_level = HierarchyLevel.STRATEGIC
        elif level == HierarchyLevel.OPERATIONAL:
            parent_level = HierarchyLevel.TACTICAL
        else:
            return context

        parent_active = self.get_active_goals(parent_level)
        context["parent_goals"] = [g.to_dict() for g in parent_active]

        return context

    def decompose_goal(
        self,
        goal_id: str,
        subgoals: list[Goal],
    ) -> list[GoalNode]:
        """Decompose a goal into subgoals.

        Args:
            goal_id: Parent goal ID.
            subgoals: List of subgoals to add.

        Returns:
            List of created child nodes.
        """
        parent_node = self.nodes_by_id.get(goal_id)
        if parent_node is None:
            raise ValueError(f"Parent goal not found: {goal_id}")

        created_nodes = []
        for subgoal in subgoals:
            node = self.add_goal(subgoal, parent_id=goal_id)
            created_nodes.append(node)

        logger.info(
            f"Decomposed goal {goal_id} into {len(subgoals)} subgoals"
        )
        return created_nodes

    def prune_completed(self) -> int:
        """Remove all completed goal subtrees.

        Returns:
            Number of goals removed.
        """
        removed = 0
        roots_to_remove = []

        for root in self.roots:
            if root.goal.status == GoalStatus.COMPLETED:
                roots_to_remove.append(root.id)

        for root_id in roots_to_remove:
            node = self.nodes_by_id.get(root_id)
            if node:
                removed += node.subtree_size
                self.remove_goal(root_id)

        return removed

    def prune_failed(self) -> int:
        """Remove all failed goal subtrees.

        Returns:
            Number of goals removed.
        """
        removed = 0
        roots_to_remove = []

        for root in self.roots:
            if root.goal.status in {GoalStatus.FAILED, GoalStatus.ABANDONED}:
                roots_to_remove.append(root.id)

        for root_id in roots_to_remove:
            node = self.nodes_by_id.get(root_id)
            if node:
                removed += node.subtree_size
                self.remove_goal(root_id)

        return removed

    def clear(self) -> None:
        """Clear all goals from the hierarchy."""
        self.roots.clear()
        self.nodes_by_id.clear()
        logger.info("Cleared goal hierarchy")

    def _validate_level_order(
        self,
        parent_level: HierarchyLevel,
        child_level: HierarchyLevel,
    ) -> bool:
        """Validate that child level is below parent level.

        Args:
            parent_level: Parent's hierarchy level.
            child_level: Child's hierarchy level.

        Returns:
            True if ordering is valid.
        """
        level_order = {
            HierarchyLevel.STRATEGIC: 0,
            HierarchyLevel.TACTICAL: 1,
            HierarchyLevel.OPERATIONAL: 2,
        }
        return level_order[child_level] > level_order[parent_level]

    @property
    def total_goals(self) -> int:
        """Get total number of goals in hierarchy."""
        return len(self.nodes_by_id)

    @property
    def statistics(self) -> dict[str, Any]:
        """Get hierarchy statistics.

        Returns:
            Dictionary of statistics.
        """
        by_level = {level: 0 for level in HierarchyLevel}
        by_status = {status: 0 for status in GoalStatus}

        for node in self.nodes_by_id.values():
            by_level[node.level] += 1
            by_status[node.goal.status] += 1

        return {
            "total_goals": self.total_goals,
            "total_roots": len(self.roots),
            "by_level": {k.value: v for k, v in by_level.items()},
            "by_status": {k.name: v for k, v in by_status.items()},
            "total_created": self._total_goals_created,
            "total_completed": self._total_goals_completed,
            "total_failed": self._total_goals_failed,
        }

    def visualize(self) -> str:
        """Create a visual representation of the hierarchy.

        Returns:
            Multi-line string visualization.
        """
        if not self.roots:
            return "Empty goal hierarchy"

        lines = ["Goal Hierarchy:"]
        for root in self.roots:
            lines.append(root.visualize())
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert hierarchy to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "roots": [r.to_dict() for r in self.roots],
            "current_turn": self.current_turn,
            "statistics": self.statistics,
        }
