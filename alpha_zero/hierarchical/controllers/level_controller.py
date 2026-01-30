"""
Level Controllers for Hierarchical Decision Making.

This module implements controllers for each hierarchy level:
- Strategic: Long-term planning and victory path selection
- Tactical: Medium-term goals and resource allocation
- Operational: Immediate action selection

Example:
    >>> from alpha_zero.hierarchical.controllers import StrategicController
    >>> controller = StrategicController(config, network, goal_hierarchy)
    >>> decision = controller.decide(game_state, context)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from alpha_zero.hierarchical.config import (
    HierarchyLevel,
    LevelConfig,
)
from alpha_zero.hierarchical.encoders.goal_encoder import (
    GoalEncoder,
)
from alpha_zero.hierarchical.encoders.state_encoder import (
    StateEncoder,
)
from alpha_zero.hierarchical.goals.goal_evaluator import GoalEvaluator
from alpha_zero.hierarchical.goals.goal_hierarchy import GoalHierarchy
from alpha_zero.hierarchical.goals.goal_propagator import GoalPropagator
from alpha_zero.hierarchical.goals.goal_types import (
    Goal,
    GoalStatus,
    StrategicGoalType,
    TacticalGoalType,
)
from alpha_zero.hierarchical.mcts.hierarchical_mcts import (
    HierarchicalMCTS,
    MCTSConfig,
    Option,
    SearchStats,
)

logger = logging.getLogger(__name__)


@dataclass
class ControllerState:
    """State maintained by a level controller.

    Attributes:
        active_goals: Currently active goals at this level.
        last_decision_turn: Turn when last decision was made.
        decision_history: Recent decisions for analysis.
        cumulative_reward: Reward accumulated since last reset.
        episodes_completed: Number of goal completions.
    """

    active_goals: list[Goal] = field(default_factory=list)
    last_decision_turn: int = -9999  # Sentinel: ensures first should_update returns True
    decision_history: list[ControllerDecision] = field(default_factory=list)
    cumulative_reward: float = 0.0
    episodes_completed: int = 0


@dataclass
class ControllerDecision:
    """Decision made by a level controller.

    Attributes:
        level: Hierarchy level of decision.
        turn: Turn when decision was made.
        goals_created: New goals created.
        goals_completed: Goals marked as completed.
        action: Primitive action (for operational level).
        option: Selected option (for tactical/strategic).
        value_estimate: Estimated value of decision.
        search_stats: MCTS search statistics.
        reasoning: Human-readable decision reasoning.
    """

    level: HierarchyLevel
    turn: int
    goals_created: list[Goal] = field(default_factory=list)
    goals_completed: list[str] = field(default_factory=list)
    action: int | None = None
    option: Option | None = None
    value_estimate: float = 0.0
    search_stats: SearchStats | None = None
    reasoning: str = ""


class LevelController(ABC):
    """Abstract base class for hierarchy level controllers.

    Each level controller manages:
    - Goal selection and prioritization
    - MCTS-based planning
    - Communication with adjacent levels
    """

    def __init__(
        self,
        level: HierarchyLevel,
        config: LevelConfig,
        network: Any | None = None,
        state_encoder: StateEncoder | None = None,
        goal_encoder: GoalEncoder | None = None,
        goal_hierarchy: GoalHierarchy | None = None,
    ) -> None:
        """Initialize level controller.

        Args:
            level: Hierarchy level for this controller.
            config: Level-specific configuration.
            network: Neural network for this level.
            state_encoder: State encoder.
            goal_encoder: Goal encoder.
            goal_hierarchy: Shared goal hierarchy.
        """
        self.level = level
        self.config = config
        self.network = network
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.goal_hierarchy = goal_hierarchy or GoalHierarchy()

        # Controller state
        self.state = ControllerState()

        # MCTS instance
        mcts_config = MCTSConfig.from_level_config(config.mcts)
        self.mcts = HierarchicalMCTS(
            network=network,
            state_encoder=state_encoder,
            goal_encoder=goal_encoder,
            config=mcts_config,
            level=level,
        )

        # Goal evaluator
        self.goal_evaluator = GoalEvaluator()

        logger.info(
            "Initialized %s controller with update_frequency=%d",
            level.value,
            config.update_frequency,
        )

    @abstractmethod
    def decide(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ControllerDecision:
        """Make a decision for the current game state.

        Args:
            game_state: Current game state.
            context: Additional context (higher level guidance, etc.).

        Returns:
            ControllerDecision with selected goals/actions.
        """
        pass

    @abstractmethod
    def update(
        self,
        game_state: dict[str, Any],
        reward: float,
        done: bool,
    ) -> None:
        """Update controller state after action execution.

        Args:
            game_state: New game state.
            reward: Reward received.
            done: Whether episode ended.
        """
        pass

    def should_update(self, current_turn: int) -> bool:
        """Check if controller should make a new decision.

        Args:
            current_turn: Current game turn.

        Returns:
            True if controller should decide.
        """
        turns_since_decision = current_turn - self.state.last_decision_turn
        return turns_since_decision >= self.config.update_frequency

    def get_active_goals(self) -> list[Goal]:
        """Get currently active goals at this level."""
        return [
            g
            for g in self.state.active_goals
            if g.status in (GoalStatus.ACTIVE, GoalStatus.PENDING)
        ]

    def reset(self) -> None:
        """Reset controller state for new episode."""
        self.state = ControllerState()
        logger.debug("Reset %s controller state", self.level.value)


class StrategicController(LevelController):
    """Controller for strategic level decisions.

    Handles long-term planning including:
    - Victory path selection
    - Major policy decisions
    - Alliance/war decisions
    - Long-term resource allocation
    """

    def __init__(
        self,
        config: LevelConfig,
        network: Any | None = None,
        state_encoder: StateEncoder | None = None,
        goal_encoder: GoalEncoder | None = None,
        goal_hierarchy: GoalHierarchy | None = None,
        llm_advisor: Any | None = None,
    ) -> None:
        """Initialize strategic controller.

        Args:
            config: Level configuration.
            network: Neural network.
            state_encoder: State encoder.
            goal_encoder: Goal encoder.
            goal_hierarchy: Goal hierarchy.
            llm_advisor: Optional LLM strategy advisor.
        """
        super().__init__(
            level=HierarchyLevel.STRATEGIC,
            config=config,
            network=network,
            state_encoder=state_encoder,
            goal_encoder=goal_encoder,
            goal_hierarchy=goal_hierarchy,
        )
        self.llm_advisor = llm_advisor

        # Victory path tracking
        self.current_victory_path: StrategicGoalType | None = None
        self.victory_path_confidence: float = 0.0

    def decide(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ControllerDecision:
        """Make strategic decision.

        Args:
            game_state: Current game state.
            context: Additional context.

        Returns:
            ControllerDecision with strategic goals.
        """
        context = context or {}
        current_turn = game_state.get("turn", 0)

        decision = ControllerDecision(
            level=self.level,
            turn=current_turn,
        )

        # Evaluate current strategic position
        position = self._evaluate_strategic_position(game_state)

        # Check if we need to update victory path
        if self._should_reconsider_victory_path(game_state, position):
            victory_path = self._select_victory_path(game_state, position)
            self._set_victory_path(victory_path, game_state)
            decision.reasoning += f"Selected victory path: {victory_path.name}. "

        # Create/update strategic goals based on victory path
        new_goals = self._generate_strategic_goals(game_state, position)
        for goal in new_goals:
            self.goal_hierarchy.add_goal(goal)
            decision.goals_created.append(goal)

        # Run MCTS to evaluate strategic options
        active_goals = self.get_active_goals()
        if active_goals and self.network:
            action, stats = self.mcts.search(game_state, active_goals)
            decision.value_estimate = stats.root_value
            decision.search_stats = stats

        # Update state
        self.state.last_decision_turn = current_turn
        self.state.active_goals = active_goals
        self.state.decision_history.append(decision)

        # Keep history bounded
        if len(self.state.decision_history) > 100:
            self.state.decision_history = self.state.decision_history[-50:]

        logger.debug(
            "Strategic decision at turn %d: %d goals, value=%.3f",
            current_turn,
            len(new_goals),
            decision.value_estimate,
        )

        return decision

    def update(
        self,
        game_state: dict[str, Any],
        reward: float,
        done: bool,
    ) -> None:
        """Update strategic controller state."""
        self.state.cumulative_reward += reward

        # Update goal progress
        for goal in self.state.active_goals:
            progress = self.goal_evaluator.evaluate_strategic_goal(goal, game_state)
            goal.progress = progress.progress
            goal.last_progress_turn = game_state.get("turn", 0)

            if progress.achieved:
                goal.status = GoalStatus.COMPLETED
                self.state.episodes_completed += 1

        if done:
            logger.info(
                "Strategic controller episode ended: reward=%.3f, goals_completed=%d",
                self.state.cumulative_reward,
                self.state.episodes_completed,
            )

    def _evaluate_strategic_position(
        self,
        game_state: dict[str, Any],
    ) -> dict[str, float]:
        """Evaluate current strategic position."""
        game_state.get("victory_progress", {})
        military = game_state.get("military", {})
        resources = game_state.get("resources", {})
        tech = game_state.get("technology", {})

        return {
            "domination_viability": self._assess_domination_viability(game_state),
            "science_viability": self._assess_science_viability(game_state),
            "culture_viability": self._assess_culture_viability(game_state),
            "religious_viability": self._assess_religious_viability(game_state),
            "diplomatic_viability": self._assess_diplomatic_viability(game_state),
            "military_strength": military.get("strength", 0) / 1000.0,
            "economic_health": min(resources.get("gold", 0) / 1000.0, 1.0),
            "tech_lead": tech.get("era", 0) / 8.0,
        }

    def _assess_domination_viability(self, game_state: dict[str, Any]) -> float:
        """Assess viability of domination victory."""
        military = game_state.get("military", {})
        victory = game_state.get("victory_progress", {})

        strength = military.get("strength", 0)
        avg_enemy_strength = military.get("avg_enemy_strength", 100)

        ratio = strength / (avg_enemy_strength + 1)
        current_progress = victory.get("domination", 0)

        return min(1.0, ratio * 0.3 + current_progress * 0.7)

    def _assess_science_viability(self, game_state: dict[str, Any]) -> float:
        """Assess viability of science victory."""
        game_state.get("technology", {})
        victory = game_state.get("victory_progress", {})

        science_per_turn = game_state.get("yields", {}).get("science_per_turn", 0)
        current_progress = victory.get("science", 0)

        return min(1.0, (science_per_turn / 100.0) * 0.3 + current_progress * 0.7)

    def _assess_culture_viability(self, game_state: dict[str, Any]) -> float:
        """Assess viability of culture victory."""
        victory = game_state.get("victory_progress", {})
        culture_per_turn = game_state.get("yields", {}).get("culture_per_turn", 0)
        tourism = game_state.get("tourism", 0)

        return min(
            1.0,
            (culture_per_turn / 100.0) * 0.2
            + (tourism / 500.0) * 0.2
            + victory.get("culture", 0) * 0.6,
        )

    def _assess_religious_viability(self, game_state: dict[str, Any]) -> float:
        """Assess viability of religious victory."""
        victory = game_state.get("victory_progress", {})
        faith_per_turn = game_state.get("yields", {}).get("faith_per_turn", 0)

        return min(1.0, (faith_per_turn / 50.0) * 0.3 + victory.get("religious", 0) * 0.7)

    def _assess_diplomatic_viability(self, game_state: dict[str, Any]) -> float:
        """Assess viability of diplomatic victory."""
        victory = game_state.get("victory_progress", {})
        diplomacy = game_state.get("diplomacy", {})
        favor = diplomacy.get("favor", 0)

        return min(1.0, (favor / 100.0) * 0.3 + victory.get("diplomatic", 0) * 0.7)

    def _should_reconsider_victory_path(
        self,
        game_state: dict[str, Any],
        position: dict[str, float],
    ) -> bool:
        """Check if we should reconsider current victory path."""
        if self.current_victory_path is None:
            return True

        # Reconsider if confidence drops too low
        if self.victory_path_confidence < 0.3:
            return True

        # Reconsider periodically (every 50 turns)
        current_turn = game_state.get("turn", 0)
        if current_turn % 50 == 0:
            return True

        return False

    def _select_victory_path(
        self,
        game_state: dict[str, Any],
        position: dict[str, float],
    ) -> StrategicGoalType:
        """Select best victory path based on current position."""
        viabilities = {
            StrategicGoalType.DOMINATION_VICTORY: position["domination_viability"],
            StrategicGoalType.SCIENCE_VICTORY: position["science_viability"],
            StrategicGoalType.CULTURE_VICTORY: position["culture_viability"],
            StrategicGoalType.RELIGIOUS_VICTORY: position["religious_viability"],
            StrategicGoalType.DIPLOMATIC_VICTORY: position["diplomatic_viability"],
        }

        best_path = max(viabilities, key=viabilities.get)
        self.victory_path_confidence = viabilities[best_path]

        return best_path

    def _set_victory_path(
        self,
        path: StrategicGoalType,
        game_state: dict[str, Any],
    ) -> None:
        """Set new victory path and create corresponding goal."""
        self.current_victory_path = path
        game_state.get("turn", 0)

        # Remove old victory goal if exists
        for goal in self.state.active_goals:
            if goal.goal_type in (
                StrategicGoalType.DOMINATION_VICTORY,
                StrategicGoalType.SCIENCE_VICTORY,
                StrategicGoalType.CULTURE_VICTORY,
                StrategicGoalType.RELIGIOUS_VICTORY,
                StrategicGoalType.DIPLOMATIC_VICTORY,
            ):
                goal.status = GoalStatus.ABANDONED

    def _generate_strategic_goals(
        self,
        game_state: dict[str, Any],
        position: dict[str, float],
    ) -> list[Goal]:
        """Generate strategic goals based on current situation."""
        goals = []
        current_turn = game_state.get("turn", 0)

        # Main victory goal
        if self.current_victory_path:
            victory_goal = Goal.strategic(
                self.current_victory_path,
                priority=1.0,
                created_turn=current_turn,
            )
            goals.append(victory_goal)

        # Supporting goals based on position
        if position["military_strength"] < 0.3:
            defense_goal = Goal.strategic(
                StrategicGoalType.DEFEND_EMPIRE,
                priority=0.8,
                created_turn=current_turn,
            )
            goals.append(defense_goal)

        if position["economic_health"] < 0.3:
            econ_goal = Goal.strategic(
                StrategicGoalType.ECONOMIC_DOMINANCE,
                priority=0.7,
                created_turn=current_turn,
            )
            goals.append(econ_goal)

        return goals


class TacticalController(LevelController):
    """Controller for tactical level decisions.

    Handles medium-term planning including:
    - City development priorities
    - Military campaigns
    - Resource acquisition
    - District/building choices
    """

    def __init__(
        self,
        config: LevelConfig,
        network: Any | None = None,
        state_encoder: StateEncoder | None = None,
        goal_encoder: GoalEncoder | None = None,
        goal_hierarchy: GoalHierarchy | None = None,
    ) -> None:
        """Initialize tactical controller."""
        super().__init__(
            level=HierarchyLevel.TACTICAL,
            config=config,
            network=network,
            state_encoder=state_encoder,
            goal_encoder=goal_encoder,
            goal_hierarchy=goal_hierarchy,
        )

        # Tactical state
        self.active_campaigns: list[str] = []  # Campaign IDs
        self.city_priorities: dict[str, TacticalGoalType] = {}

    def decide(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ControllerDecision:
        """Make tactical decision.

        Args:
            game_state: Current game state.
            context: Context including strategic goals.

        Returns:
            ControllerDecision with tactical goals.
        """
        context = context or {}
        current_turn = game_state.get("turn", 0)
        strategic_goals = context.get("strategic_goals", [])

        decision = ControllerDecision(
            level=self.level,
            turn=current_turn,
        )

        # Decompose strategic goals into tactical goals
        tactical_goals = self._decompose_strategic_goals(strategic_goals, game_state)

        # Add situational tactical goals
        situational = self._identify_situational_goals(game_state)
        tactical_goals.extend(situational)

        # Prioritize goals
        prioritized = self._prioritize_goals(tactical_goals, game_state)

        # Add to hierarchy and decision
        for goal in prioritized:
            self.goal_hierarchy.add_goal(goal)
            decision.goals_created.append(goal)

        # Run MCTS for tactical options
        active_goals = self.get_active_goals()
        if active_goals and self.network:
            action, stats = self.mcts.search(game_state, active_goals)
            decision.value_estimate = stats.root_value
            decision.search_stats = stats

        # Update state
        self.state.last_decision_turn = current_turn
        self.state.active_goals = active_goals
        self.state.decision_history.append(decision)

        logger.debug(
            "Tactical decision at turn %d: %d goals created",
            current_turn,
            len(decision.goals_created),
        )

        return decision

    def update(
        self,
        game_state: dict[str, Any],
        reward: float,
        done: bool,
    ) -> None:
        """Update tactical controller state."""
        self.state.cumulative_reward += reward
        current_turn = game_state.get("turn", 0)

        # Update goal progress
        for goal in self.state.active_goals:
            progress = self.goal_evaluator.evaluate_tactical_goal(goal, game_state)
            goal.progress = progress.progress
            goal.last_progress_turn = current_turn

            if progress.achieved:
                goal.status = GoalStatus.COMPLETED
                self.state.episodes_completed += 1

    def _decompose_strategic_goals(
        self,
        strategic_goals: list[Goal],
        game_state: dict[str, Any],
    ) -> list[Goal]:
        """Decompose strategic goals into tactical sub-goals."""
        tactical_goals = []
        current_turn = game_state.get("turn", 0)

        for s_goal in strategic_goals:
            if not isinstance(s_goal.goal_type, StrategicGoalType):
                continue

            # Use goal propagator for decomposition
            propagator = GoalPropagator(self.goal_hierarchy)
            result = propagator.propagate_down(s_goal, game_state, {"current_turn": current_turn})
            tactical_goals.extend(result.goals_created)

        return tactical_goals

    def _identify_situational_goals(
        self,
        game_state: dict[str, Any],
    ) -> list[Goal]:
        """Identify situational tactical goals."""
        goals = []
        current_turn = game_state.get("turn", 0)

        # Check for immediate threats
        enemies_nearby = self._count_nearby_enemies(game_state)
        if enemies_nearby > 0:
            defend_goal = Goal.tactical(
                TacticalGoalType.DEFEND_CITY,
                priority=0.9,
                created_turn=current_turn,
                deadline_turn=current_turn + 10,
            )
            goals.append(defend_goal)

        # Check for expansion opportunities
        if self._should_expand(game_state):
            expand_goal = Goal.tactical(
                TacticalGoalType.SETTLE_CITY,
                priority=0.7,
                created_turn=current_turn,
            )
            goals.append(expand_goal)

        # Check for resource needs
        resources = game_state.get("resources", {})
        if resources.get("gold", 0) < 0:
            econ_goal = Goal.tactical(
                TacticalGoalType.BUILD_COMMERCIAL,
                priority=0.8,
                created_turn=current_turn,
            )
            goals.append(econ_goal)

        return goals

    def _count_nearby_enemies(self, game_state: dict[str, Any]) -> int:
        """Count enemy units near own cities."""
        units = game_state.get("units", [])
        cities = game_state.get("cities", [])
        own_player = game_state.get("current_player", 0)

        own_cities = [c for c in cities if c.get("player") == own_player]
        enemies = [u for u in units if u.get("player") != own_player]

        count = 0
        for city in own_cities:
            cx, cy = city.get("x", 0), city.get("y", 0)
            for enemy in enemies:
                ex, ey = enemy.get("x", 0), enemy.get("y", 0)
                if abs(ex - cx) <= 5 and abs(ey - cy) <= 5:
                    count += 1

        return count

    def _should_expand(self, game_state: dict[str, Any]) -> bool:
        """Check if we should expand (settle new city)."""
        cities = game_state.get("cities", [])
        own_player = game_state.get("current_player", 0)
        own_cities = [c for c in cities if c.get("player") == own_player]

        # Simple heuristic: expand if < 6 cities and have settler capacity
        units = game_state.get("units", [])
        settlers = [
            u for u in units if u.get("player") == own_player and u.get("type") == "settler"
        ]

        return len(own_cities) < 6 or len(settlers) > 0

    def _prioritize_goals(
        self,
        goals: list[Goal],
        game_state: dict[str, Any],
    ) -> list[Goal]:
        """Prioritize tactical goals based on urgency and importance."""

        # Sort by priority and deadline
        def score(g: Goal) -> float:
            urgency = 0.0
            if g.deadline_turn:
                turns_left = g.deadline_turn - game_state.get("turn", 0)
                urgency = max(0, 1.0 - turns_left / 20.0)
            return g.priority + urgency * 0.5

        return sorted(goals, key=score, reverse=True)


class OperationalController(LevelController):
    """Controller for operational level decisions.

    Handles immediate action selection including:
    - Unit movement and combat
    - Production queue management
    - Technology/civic selection
    - Diplomatic actions
    """

    def __init__(
        self,
        config: LevelConfig,
        network: Any | None = None,
        state_encoder: StateEncoder | None = None,
        goal_encoder: GoalEncoder | None = None,
        goal_hierarchy: GoalHierarchy | None = None,
    ) -> None:
        """Initialize operational controller."""
        super().__init__(
            level=HierarchyLevel.OPERATIONAL,
            config=config,
            network=network,
            state_encoder=state_encoder,
            goal_encoder=goal_encoder,
            goal_hierarchy=goal_hierarchy,
        )

        # Action tracking
        self.pending_actions: list[int] = []
        self.executed_actions: list[tuple[int, float]] = []  # (action, reward)

    def decide(
        self,
        game_state: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> ControllerDecision:
        """Make operational decision (select action).

        Args:
            game_state: Current game state.
            context: Context including tactical goals.

        Returns:
            ControllerDecision with selected action.
        """
        context = context or {}
        current_turn = game_state.get("turn", 0)
        tactical_goals = context.get("tactical_goals", [])

        decision = ControllerDecision(
            level=self.level,
            turn=current_turn,
        )

        # Get valid actions
        valid_actions = game_state.get("action_mask", np.ones(356))

        # Decompose tactical goals into operational goals if needed
        operational_goals = self._get_operational_goals(tactical_goals, game_state)

        # Run MCTS to select action
        active_goals = operational_goals or self.get_active_goals()

        if self.network and np.any(valid_actions > 0):
            action, stats = self.mcts.search(
                game_state,
                active_goals,
                valid_actions,
                self._simulate_action,
            )
            decision.action = action
            decision.value_estimate = stats.root_value
            decision.search_stats = stats
        else:
            # Fallback: random valid action
            valid_indices = np.where(valid_actions > 0)[0]
            if len(valid_indices) > 0:
                decision.action = int(np.random.choice(valid_indices))

        # Update state
        self.state.last_decision_turn = current_turn
        self.state.decision_history.append(decision)

        if len(self.state.decision_history) > 1000:
            self.state.decision_history = self.state.decision_history[-500:]

        logger.debug(
            "Operational decision at turn %d: action=%s, value=%.3f",
            current_turn,
            decision.action,
            decision.value_estimate,
        )

        return decision

    def update(
        self,
        game_state: dict[str, Any],
        reward: float,
        done: bool,
    ) -> None:
        """Update operational controller state."""
        self.state.cumulative_reward += reward

        # Track action outcomes
        if self.state.decision_history:
            last = self.state.decision_history[-1]
            if last.action is not None:
                self.executed_actions.append((last.action, reward))

        # Keep executed actions bounded
        if len(self.executed_actions) > 10000:
            self.executed_actions = self.executed_actions[-5000:]

        # Update operational goal progress
        for goal in self.state.active_goals:
            progress = self.goal_evaluator.evaluate_operational_goal(goal, game_state)
            goal.progress = progress.progress

            if progress.achieved:
                goal.status = GoalStatus.COMPLETED
                self.state.episodes_completed += 1

    def _get_operational_goals(
        self,
        tactical_goals: list[Goal],
        game_state: dict[str, Any],
    ) -> list[Goal]:
        """Get operational goals from tactical goals."""
        operational = []
        current_turn = game_state.get("turn", 0)

        for t_goal in tactical_goals:
            if not isinstance(t_goal.goal_type, TacticalGoalType):
                continue

            # Create operational sub-goals
            propagator = GoalPropagator(self.goal_hierarchy)
            result = propagator.propagate_down(t_goal, game_state, {"current_turn": current_turn})
            operational.extend(result.goals_created)

        return operational

    def _simulate_action(
        self,
        state: dict[str, Any],
        action: int,
    ) -> dict[str, Any]:
        """Simulate action effect (placeholder).

        In practice, this should use a game model to predict
        the next state after taking an action.
        """
        # Return copy of state (no actual simulation)
        return dict(state)
