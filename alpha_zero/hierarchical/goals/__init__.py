"""
Goal Hierarchy System for Hierarchical Reasoning.

This module provides the goal management infrastructure for the hierarchical
decision system, including goal types, tree structures, evaluation, and
cross-level propagation.

Example:
    >>> from alpha_zero.hierarchical.goals import GoalHierarchy, Goal, StrategicGoalType
    >>> hierarchy = GoalHierarchy()
    >>> goal = Goal.strategic(StrategicGoalType.DOMINATION_VICTORY)
    >>> hierarchy.add_goal(goal)
"""

from __future__ import annotations

from alpha_zero.hierarchical.goals.goal_evaluator import GoalEvaluator
from alpha_zero.hierarchical.goals.goal_hierarchy import GoalHierarchy
from alpha_zero.hierarchical.goals.goal_node import GoalNode
from alpha_zero.hierarchical.goals.goal_propagator import GoalPropagator
from alpha_zero.hierarchical.goals.goal_types import (
    Goal,
    GoalStatus,
    GoalTarget,
    HierarchyLevel,
    OperationalGoalType,
    StrategicGoalType,
    TacticalGoalType,
)

__all__ = [
    # Types and enums
    "GoalStatus",
    "HierarchyLevel",
    "StrategicGoalType",
    "TacticalGoalType",
    "OperationalGoalType",
    # Core classes
    "Goal",
    "GoalTarget",
    "GoalNode",
    "GoalHierarchy",
    "GoalEvaluator",
    "GoalPropagator",
]
