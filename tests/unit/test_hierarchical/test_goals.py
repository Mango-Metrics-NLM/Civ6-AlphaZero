"""Tests for hierarchical goal management system."""
from __future__ import annotations

import pytest
import numpy as np

from alpha_zero.hierarchical.goals.goal_types import (
    Goal,
    GoalTarget,
    GoalStatus,
    HierarchyLevel,
    StrategicGoalType,
    TacticalGoalType,
    OperationalGoalType,
)
from alpha_zero.hierarchical.goals.goal_node import GoalNode
from alpha_zero.hierarchical.goals.goal_hierarchy import GoalHierarchy
from alpha_zero.hierarchical.goals.goal_evaluator import GoalEvaluator
from alpha_zero.hierarchical.goals.goal_propagator import (
    GoalPropagator,
    StrategicDecomposition,
    TacticalDecomposition,
    PropagationResult,
)


class TestGoal:
    """Tests for Goal dataclass."""

    def test_strategic_goal_creation(self) -> None:
        """Test creating a strategic goal."""
        goal = Goal.strategic(
            StrategicGoalType.DOMINATION_VICTORY,
            priority=1.0,
            created_turn=0,
        )

        assert goal.level == HierarchyLevel.STRATEGIC
        assert goal.goal_type == StrategicGoalType.DOMINATION_VICTORY
        assert goal.priority == 1.0
        assert goal.status == GoalStatus.PENDING
        assert goal.progress == 0.0

    def test_tactical_goal_creation(self) -> None:
        """Test creating a tactical goal."""
        goal = Goal.tactical(
            TacticalGoalType.CAPTURE_CITY,
            priority=0.8,
            created_turn=10,
        )

        assert goal.level == HierarchyLevel.TACTICAL
        assert goal.goal_type == TacticalGoalType.CAPTURE_CITY
        assert goal.priority == 0.8

    def test_operational_goal_creation(self) -> None:
        """Test creating an operational goal."""
        goal = Goal.operational(
            OperationalGoalType.MOVE_UNIT,
            priority=0.5,
            created_turn=20,
        )

        assert goal.level == HierarchyLevel.OPERATIONAL
        assert goal.goal_type == OperationalGoalType.MOVE_UNIT
        assert goal.priority == 0.5

    def test_goal_target(self) -> None:
        """Test GoalTarget dataclass."""
        target = GoalTarget(
            target_type="city",
            target_id="city_123",
            position=(10, 15),
            quantity=1,
        )

        assert target.target_type == "city"
        assert target.entity_id == "city_123"
        assert target.location == (10, 15)
        assert target.quantity == 1

    def test_goal_with_target(self) -> None:
        """Test creating goal with target."""
        target = GoalTarget(position=(5, 5))
        goal = Goal.tactical(
            TacticalGoalType.SETTLE_CITY,
            target=target,
        )

        assert goal.target.location == (5, 5)


class TestGoalNode:
    """Tests for GoalNode tree structure."""

    def test_node_creation(self) -> None:
        """Test creating a goal node."""
        goal = Goal.strategic(StrategicGoalType.SCIENCE_VICTORY)
        node = GoalNode(goal)

        assert node.goal == goal
        assert node.parent is None
        assert len(node.children) == 0

    def test_add_child(self) -> None:
        """Test adding child nodes."""
        parent_goal = Goal.strategic(StrategicGoalType.SCIENCE_VICTORY)
        parent = GoalNode(parent_goal)

        child_goal = Goal.tactical(TacticalGoalType.RESEARCH_TECH)
        child = parent.add_child(child_goal)

        assert len(parent.children) == 1
        assert child.parent == parent
        assert child.goal == child_goal

    def test_remove_child(self) -> None:
        """Test removing child nodes."""
        parent_goal = Goal.strategic(StrategicGoalType.DOMINATION_VICTORY)
        parent = GoalNode(parent_goal)

        child_goal = Goal.tactical(TacticalGoalType.BUILD_ARMY)
        child = parent.add_child(child_goal)

        removed = parent.remove_child(child.goal.id)

        assert removed is True
        assert len(parent.children) == 0

    def test_traverse_preorder(self) -> None:
        """Test pre-order traversal."""
        root = GoalNode(Goal.strategic(StrategicGoalType.SCIENCE_VICTORY))
        child1 = root.add_child(Goal.tactical(TacticalGoalType.RESEARCH_TECH))
        child2 = root.add_child(Goal.tactical(TacticalGoalType.BUILD_CAMPUS))
        grandchild = child1.add_child(Goal.operational(OperationalGoalType.SELECT_RESEARCH))

        visited = []
        for node in root.traverse_preorder():
            visited.append(node.goal.goal_type)

        assert len(visited) == 4
        assert visited[0] == StrategicGoalType.SCIENCE_VICTORY  # Root first

    def test_aggregate_progress(self) -> None:
        """Test progress aggregation."""
        root = GoalNode(Goal.strategic(StrategicGoalType.DOMINATION_VICTORY))
        child1 = root.add_child(Goal.tactical(TacticalGoalType.CAPTURE_CITY))
        child2 = root.add_child(Goal.tactical(TacticalGoalType.BUILD_ARMY))

        child1.goal.progress = 0.5
        child2.goal.progress = 0.7

        aggregated = root.aggregate_progress()
        # Should be weighted average
        assert 0.5 <= aggregated <= 0.7


class TestGoalHierarchy:
    """Tests for GoalHierarchy management."""

    def test_create_hierarchy(self) -> None:
        """Test creating goal hierarchy."""
        hierarchy = GoalHierarchy()

        assert hierarchy.root is not None
        assert len(hierarchy.get_all_goals()) == 0

    def test_add_goal(self) -> None:
        """Test adding goals to hierarchy."""
        hierarchy = GoalHierarchy()

        goal = Goal.strategic(StrategicGoalType.CULTURE_VICTORY)
        hierarchy.add_goal(goal)

        assert len(hierarchy.get_all_goals()) == 1
        assert hierarchy.get_goal(goal.id) == goal

    def test_add_child_goal(self) -> None:
        """Test adding child goals."""
        hierarchy = GoalHierarchy()

        parent = Goal.strategic(StrategicGoalType.SCIENCE_VICTORY)
        hierarchy.add_goal(parent)

        child = Goal.tactical(TacticalGoalType.RESEARCH_TECH)
        hierarchy.add_goal(child, parent_id=parent.id)

        assert child.parent_goal_id == parent.id
        assert child.id in parent.child_goal_ids

    def test_get_active_goals(self) -> None:
        """Test getting active goals by level."""
        hierarchy = GoalHierarchy()

        strategic = Goal.strategic(StrategicGoalType.DOMINATION_VICTORY)
        strategic.status = GoalStatus.ACTIVE
        hierarchy.add_goal(strategic)

        tactical = Goal.tactical(TacticalGoalType.BUILD_ARMY)
        tactical.status = GoalStatus.ACTIVE
        hierarchy.add_goal(tactical)

        completed = Goal.tactical(TacticalGoalType.CAPTURE_CITY)
        completed.status = GoalStatus.COMPLETED
        hierarchy.add_goal(completed)

        active_tactical = hierarchy.get_active_goals(HierarchyLevel.TACTICAL)
        assert len(active_tactical) == 1
        assert active_tactical[0] == tactical

    def test_remove_goal(self) -> None:
        """Test removing goals."""
        hierarchy = GoalHierarchy()

        goal = Goal.strategic(StrategicGoalType.RELIGIOUS_VICTORY)
        hierarchy.add_goal(goal)

        hierarchy.remove_goal(goal.id)

        assert hierarchy.get_goal(goal.id) is None

    def test_goal_lookup_index(self) -> None:
        """Test goal lookup by ID."""
        hierarchy = GoalHierarchy()

        goals = [
            Goal.strategic(StrategicGoalType.SCIENCE_VICTORY),
            Goal.tactical(TacticalGoalType.BUILD_CAMPUS),
            Goal.operational(OperationalGoalType.BUILD_DISTRICT),
        ]

        for goal in goals:
            hierarchy.add_goal(goal)

        for goal in goals:
            assert hierarchy.get_goal(goal.id) == goal


class TestGoalEvaluator:
    """Tests for GoalEvaluator."""

    def test_evaluator_creation(self) -> None:
        """Test creating goal evaluator."""
        evaluator = GoalEvaluator()
        assert evaluator is not None

    def test_evaluate_strategic_goal(self) -> None:
        """Test evaluating strategic goal progress."""
        evaluator = GoalEvaluator()
        goal = Goal.strategic(StrategicGoalType.DOMINATION_VICTORY)

        game_state = {
            "victory_progress": {"domination": 0.5},
            "cities": [{"player": 0}, {"player": 0}],
        }

        result = evaluator.evaluate_strategic_goal(goal, game_state)

        assert result.progress >= 0.0
        assert result.progress <= 1.0

    def test_evaluate_tactical_goal(self) -> None:
        """Test evaluating tactical goal progress."""
        evaluator = GoalEvaluator()
        goal = Goal.tactical(TacticalGoalType.BUILD_ARMY)
        goal.target = GoalTarget(quantity=5)

        game_state = {
            "units": [{"player": 0, "class": "melee"} for _ in range(3)],
            "current_player": 0,
        }

        result = evaluator.evaluate_tactical_goal(goal, game_state)

        assert result.progress >= 0.0

    def test_evaluate_operational_goal(self) -> None:
        """Test evaluating operational goal progress."""
        evaluator = GoalEvaluator()
        goal = Goal.operational(OperationalGoalType.MOVE_UNIT)
        goal.target = GoalTarget(position=(5, 5))

        game_state = {
            "units": [{"player": 0, "x": 4, "y": 5}],
            "current_player": 0,
        }

        result = evaluator.evaluate_operational_goal(goal, game_state)

        assert result is not None

    def test_compute_goal_reward(self) -> None:
        """Test computing goal-based reward."""
        evaluator = GoalEvaluator()

        # Create goal with progress
        goal = Goal.strategic(StrategicGoalType.SCIENCE_VICTORY)
        goal.progress = 0.3

        prev_state = {"victory_progress": {"science": 0.3}}
        next_state = {"victory_progress": {"science": 0.4}}

        reward = evaluator.compute_goal_reward(
            goal, prev_state, next_state, base_reward=1.0
        )

        # Should have positive bonus due to progress increase
        assert reward >= 1.0


class TestGoalPropagator:
    """Tests for GoalPropagator."""

    def test_propagator_creation(self) -> None:
        """Test creating goal propagator."""
        hierarchy = GoalHierarchy()
        propagator = GoalPropagator(hierarchy)

        assert propagator.hierarchy == hierarchy
        assert len(propagator.strategies) > 0

    def test_strategic_decomposition(self) -> None:
        """Test decomposing strategic goals."""
        hierarchy = GoalHierarchy()
        propagator = GoalPropagator(hierarchy)

        strategic_goal = Goal.strategic(StrategicGoalType.DOMINATION_VICTORY)
        hierarchy.add_goal(strategic_goal)

        game_state = {"turn": 0, "cities": [], "units": []}
        context = {"current_turn": 0}

        result = propagator.propagate_down(strategic_goal, game_state, context)

        assert result.success
        assert len(result.goals_created) > 0
        # Should create tactical goals
        for goal in result.goals_created:
            assert goal.level == HierarchyLevel.TACTICAL

    def test_tactical_decomposition(self) -> None:
        """Test decomposing tactical goals."""
        hierarchy = GoalHierarchy()
        propagator = GoalPropagator(hierarchy)

        tactical_goal = Goal.tactical(TacticalGoalType.CAPTURE_CITY)
        hierarchy.add_goal(tactical_goal)

        game_state = {"turn": 0}
        context = {"current_turn": 0}

        result = propagator.propagate_down(tactical_goal, game_state, context)

        assert result.success
        # Should create operational goals
        for goal in result.goals_created:
            assert goal.level == HierarchyLevel.OPERATIONAL

    def test_propagate_up(self) -> None:
        """Test upward progress propagation."""
        hierarchy = GoalHierarchy()
        propagator = GoalPropagator(hierarchy)

        # Create parent and child
        parent = Goal.strategic(StrategicGoalType.SCIENCE_VICTORY)
        hierarchy.add_goal(parent)

        child = Goal.tactical(TacticalGoalType.RESEARCH_TECH)
        hierarchy.add_goal(child, parent_id=parent.id)

        # Set child progress
        child.progress = 0.8

        result = propagator.propagate_up(child, {})

        assert result.success
        assert parent.progress > 0.0  # Parent should have updated progress

    def test_synchronize_hierarchy(self) -> None:
        """Test full hierarchy synchronization."""
        hierarchy = GoalHierarchy()
        propagator = GoalPropagator(hierarchy)

        # Add goals
        goal = Goal.tactical(TacticalGoalType.BUILD_ARMY)
        goal.status = GoalStatus.ACTIVE
        goal.created_turn = 0
        goal.deadline_turn = 5
        hierarchy.add_goal(goal)

        # Simulate time passing beyond deadline
        context = {"current_turn": 10}
        result = propagator.synchronize_hierarchy({}, context)

        # Goal should be marked failed
        assert goal.status == GoalStatus.FAILED


class TestDecompositionStrategies:
    """Tests for decomposition strategies."""

    def test_strategic_decomposition_can_handle(self) -> None:
        """Test strategic decomposition handles strategic goals."""
        strategy = StrategicDecomposition()

        strategic = Goal.strategic(StrategicGoalType.CULTURE_VICTORY)
        tactical = Goal.tactical(TacticalGoalType.BUILD_THEATER)

        assert strategy.can_handle(strategic) is True
        assert strategy.can_handle(tactical) is False

    def test_tactical_decomposition_can_handle(self) -> None:
        """Test tactical decomposition handles tactical goals."""
        strategy = TacticalDecomposition()

        strategic = Goal.strategic(StrategicGoalType.CULTURE_VICTORY)
        tactical = Goal.tactical(TacticalGoalType.BUILD_THEATER)

        assert strategy.can_handle(strategic) is False
        assert strategy.can_handle(tactical) is True

    def test_strategic_decomposition_map(self) -> None:
        """Test strategic decomposition produces correct tactical types."""
        strategy = StrategicDecomposition()

        goal = Goal.strategic(StrategicGoalType.SCIENCE_VICTORY)
        sub_goals = strategy.decompose(goal, {}, {"current_turn": 0})

        # Should produce research-related tactical goals
        tactical_types = [g.goal_type for g in sub_goals]
        assert TacticalGoalType.RESEARCH_TECH in tactical_types or \
               TacticalGoalType.BUILD_CAMPUS in tactical_types
