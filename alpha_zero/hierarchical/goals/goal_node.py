"""
Goal Node Tree Structure for Hierarchical Reasoning.

This module provides the tree structure for organizing goals hierarchically,
with support for traversal, modification, and state management.

Example:
    >>> from alpha_zero.hierarchical.goals import GoalNode, Goal, StrategicGoalType
    >>> root = GoalNode(Goal.strategic(StrategicGoalType.DOMINATION_VICTORY))
    >>> child = GoalNode(Goal.tactical(TacticalGoalType.CONQUER_CITY))
    >>> root.add_child(child)
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from alpha_zero.hierarchical.goals.goal_types import (
    Goal,
    GoalStatus,
    HierarchyLevel,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class GoalNode:
    """Node in the goal hierarchy tree.

    This class represents a single node containing a goal, with links to
    parent and children nodes. It provides methods for tree traversal
    and manipulation.

    Attributes:
        goal: The goal at this node.
        parent: Parent node reference (None for root).
        children: List of child nodes.
        metadata: Node-specific metadata.
    """

    goal: Goal
    parent: GoalNode | None = None
    children: list[GoalNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Update goal hierarchy links after initialization."""
        # Ensure goal references match node structure
        if self.parent is not None:
            self.goal.parent_goal_id = self.parent.goal.id
        else:
            self.goal.parent_goal_id = None

    @property
    def id(self) -> str:
        """Get the goal ID for this node."""
        return self.goal.id

    @property
    def level(self) -> HierarchyLevel:
        """Get the hierarchy level of this node's goal."""
        return self.goal.level

    @property
    def is_root(self) -> bool:
        """Check if this is a root node (no parent)."""
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0

    @property
    def depth(self) -> int:
        """Get the depth of this node in the tree."""
        if self.parent is None:
            return 0
        return self.parent.depth + 1

    @property
    def height(self) -> int:
        """Get the height of the subtree rooted at this node."""
        if self.is_leaf:
            return 0
        return 1 + max(child.height for child in self.children)

    @property
    def subtree_size(self) -> int:
        """Get the total number of nodes in this subtree."""
        return 1 + sum(child.subtree_size for child in self.children)

    def add_child(self, child: GoalNode) -> None:
        """Add a child node.

        Args:
            child: Child node to add.
        """
        child.parent = self
        child.goal.parent_goal_id = self.goal.id
        self.children.append(child)
        self.goal.child_goal_ids.append(child.goal.id)
        logger.debug(f"Added child {child.id} to node {self.id}")

    def remove_child(self, child_id: str) -> GoalNode | None:
        """Remove a child node by ID.

        Args:
            child_id: ID of child to remove.

        Returns:
            Removed child node, or None if not found.
        """
        for i, child in enumerate(self.children):
            if child.id == child_id:
                removed = self.children.pop(i)
                removed.parent = None
                removed.goal.parent_goal_id = None
                if child_id in self.goal.child_goal_ids:
                    self.goal.child_goal_ids.remove(child_id)
                logger.debug(f"Removed child {child_id} from node {self.id}")
                return removed
        return None

    def find_child(self, child_id: str) -> GoalNode | None:
        """Find a direct child by ID.

        Args:
            child_id: ID of child to find.

        Returns:
            Child node, or None if not found.
        """
        for child in self.children:
            if child.id == child_id:
                return child
        return None

    def find_descendant(self, goal_id: str) -> GoalNode | None:
        """Find a descendant node by ID (including self).

        Args:
            goal_id: ID of goal to find.

        Returns:
            Node with matching ID, or None if not found.
        """
        if self.id == goal_id:
            return self
        for child in self.children:
            result = child.find_descendant(goal_id)
            if result is not None:
                return result
        return None

    def get_ancestors(self) -> list[GoalNode]:
        """Get all ancestor nodes from root to parent.

        Returns:
            List of ancestor nodes, starting from root.
        """
        if self.parent is None:
            return []
        ancestors = self.parent.get_ancestors()
        ancestors.append(self.parent)
        return ancestors

    def get_siblings(self) -> list[GoalNode]:
        """Get sibling nodes (same parent, excluding self).

        Returns:
            List of sibling nodes.
        """
        if self.parent is None:
            return []
        return [c for c in self.parent.children if c.id != self.id]

    def get_leaves(self) -> list[GoalNode]:
        """Get all leaf nodes in this subtree.

        Returns:
            List of leaf nodes.
        """
        if self.is_leaf:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def get_active_children(self) -> list[GoalNode]:
        """Get children with active goals.

        Returns:
            List of children with ACTIVE status.
        """
        return [c for c in self.children if c.goal.status == GoalStatus.ACTIVE]

    def get_pending_children(self) -> list[GoalNode]:
        """Get children with pending goals.

        Returns:
            List of children with PENDING status.
        """
        return [c for c in self.children if c.goal.status == GoalStatus.PENDING]

    def get_children_by_level(self, level: HierarchyLevel) -> list[GoalNode]:
        """Get all descendants at a specific hierarchy level.

        Args:
            level: Hierarchy level to filter by.

        Returns:
            List of nodes at the specified level.
        """
        result = []
        if self.level == level:
            result.append(self)
        for child in self.children:
            result.extend(child.get_children_by_level(level))
        return result

    def traverse_preorder(self) -> Iterator[GoalNode]:
        """Traverse subtree in pre-order (root before children).

        Yields:
            Nodes in pre-order.
        """
        yield self
        for child in self.children:
            yield from child.traverse_preorder()

    def traverse_postorder(self) -> Iterator[GoalNode]:
        """Traverse subtree in post-order (children before root).

        Yields:
            Nodes in post-order.
        """
        for child in self.children:
            yield from child.traverse_postorder()
        yield self

    def traverse_level_order(self) -> Iterator[GoalNode]:
        """Traverse subtree in level-order (breadth-first).

        Yields:
            Nodes in level-order.
        """
        queue = [self]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(node.children)

    def map(self, fn: Callable[[Goal], Goal]) -> GoalNode:
        """Apply a function to all goals in subtree.

        Args:
            fn: Function to apply to each goal.

        Returns:
            New GoalNode tree with transformed goals.
        """
        new_goal = fn(self.goal)
        new_node = GoalNode(goal=new_goal)
        for child in self.children:
            new_child = child.map(fn)
            new_node.add_child(new_child)
        return new_node

    def filter(
        self,
        predicate: Callable[[Goal], bool],
    ) -> list[GoalNode]:
        """Filter nodes by a predicate.

        Args:
            predicate: Function that returns True for nodes to include.

        Returns:
            List of nodes matching the predicate.
        """
        result = []
        if predicate(self.goal):
            result.append(self)
        for child in self.children:
            result.extend(child.filter(predicate))
        return result

    def update_status(self, new_status: GoalStatus) -> None:
        """Update the goal status and propagate if needed.

        Args:
            new_status: New status to set.
        """
        old_status = self.goal.status
        self.goal.status = new_status

        # Log status change
        logger.debug(
            f"Goal {self.id} status: {old_status.name} -> {new_status.name}"
        )

        # Handle status transitions
        if new_status == GoalStatus.COMPLETED:
            # Check if parent can be completed
            if self.parent and all(
                c.goal.status == GoalStatus.COMPLETED
                for c in self.parent.children
            ):
                logger.info(f"All children complete, parent {self.parent.id} eligible")

        elif new_status == GoalStatus.FAILED:
            # Fail all children
            for child in self.children:
                if not child.goal.is_terminal:
                    child.update_status(GoalStatus.FAILED)

        elif new_status == GoalStatus.ABANDONED:
            # Abandon all children
            for child in self.children:
                if not child.goal.is_terminal:
                    child.update_status(GoalStatus.ABANDONED)

    def aggregate_progress(self) -> float:
        """Calculate aggregate progress including children.

        Returns:
            Weighted average progress of subtree.
        """
        if self.is_leaf:
            return self.goal.progress

        if not self.children:
            return self.goal.progress

        # Weight by child priority
        total_weight = sum(c.goal.priority for c in self.children)
        if total_weight == 0:
            return self.goal.progress

        weighted_progress = sum(
            c.aggregate_progress() * c.goal.priority
            for c in self.children
        ) / total_weight

        # Combine with self progress (self is 30%, children are 70%)
        return 0.3 * self.goal.progress + 0.7 * weighted_progress

    def to_dict(self) -> dict[str, Any]:
        """Convert node and subtree to dictionary.

        Returns:
            Nested dictionary representation.
        """
        return {
            "goal": self.goal.to_dict(),
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoalNode:
        """Create node and subtree from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            GoalNode with children.
        """
        # This would need goal deserialization logic
        raise NotImplementedError("Use GoalHierarchy.from_dict instead")

    def __repr__(self) -> str:
        """String representation of node."""
        return (
            f"GoalNode(id={self.id}, "
            f"level={self.level.value}, "
            f"children={len(self.children)})"
        )

    def visualize(self, indent: int = 0) -> str:
        """Create a visual tree representation.

        Args:
            indent: Current indentation level.

        Returns:
            Multi-line string visualization.
        """
        prefix = "  " * indent
        status_emoji = {
            GoalStatus.PENDING: "⏳",
            GoalStatus.ACTIVE: "▶️",
            GoalStatus.PAUSED: "⏸️",
            GoalStatus.COMPLETED: "✅",
            GoalStatus.FAILED: "❌",
            GoalStatus.ABANDONED: "🚫",
        }
        emoji = status_emoji.get(self.goal.status, "❓")

        lines = [
            f"{prefix}{emoji} [{self.level.value[:3].upper()}] "
            f"{self.goal.goal_type.name} "
            f"(progress: {self.goal.progress:.1%})"
        ]

        for child in self.children:
            lines.append(child.visualize(indent + 1))

        return "\n".join(lines)
