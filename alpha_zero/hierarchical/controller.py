"""
Hierarchical Controller - Main Coordinator for Multi-Level Decision Making.

This module provides the main HierarchicalController class that coordinates
all hierarchy levels (strategic, tactical, operational) for unified
decision making in Civilization 6.

Example:
    >>> from alpha_zero.hierarchical import HierarchicalController, HierarchicalConfig
    >>> config = HierarchicalConfig.full()
    >>> controller = HierarchicalController(config)
    >>> action = controller.decide(game_state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from alpha_zero.hierarchical.config import (
    HierarchicalConfig,
    HierarchyLevel,
)
from alpha_zero.hierarchical.controllers.level_controller import (
    ControllerDecision,
    OperationalController,
    StrategicController,
    TacticalController,
)
from alpha_zero.hierarchical.encoders.goal_encoder import HierarchicalGoalEncoder
from alpha_zero.hierarchical.encoders.state_encoder import HierarchicalStateEncoder
from alpha_zero.hierarchical.goals.goal_evaluator import GoalEvaluator
from alpha_zero.hierarchical.goals.goal_hierarchy import GoalHierarchy
from alpha_zero.hierarchical.goals.goal_propagator import GoalPropagator
from alpha_zero.hierarchical.goals.goal_types import Goal, GoalStatus
from alpha_zero.hierarchical.network.hierarchical_network import (
    HierarchicalNetwork,
    HierarchicalNetworkConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalState:
    """State maintained by the hierarchical controller.

    Attributes:
        current_turn: Current game turn.
        strategic_decisions: Recent strategic decisions.
        tactical_decisions: Recent tactical decisions.
        operational_decisions: Recent operational decisions.
        total_reward: Accumulated reward.
        episodes: Number of episodes completed.
    """

    current_turn: int = 0
    strategic_decisions: list[ControllerDecision] = field(default_factory=list)
    tactical_decisions: list[ControllerDecision] = field(default_factory=list)
    operational_decisions: list[ControllerDecision] = field(default_factory=list)
    total_reward: float = 0.0
    episodes: int = 0


@dataclass
class HierarchicalDecision:
    """Combined decision from all hierarchy levels.

    Attributes:
        action: Selected primitive action.
        strategic_context: Active strategic goals.
        tactical_context: Active tactical goals.
        operational_goals: Active operational goals.
        value_estimates: Value estimates by level.
        reasoning: Combined reasoning from all levels.
    """

    action: int
    strategic_context: list[Goal] = field(default_factory=list)
    tactical_context: list[Goal] = field(default_factory=list)
    operational_goals: list[Goal] = field(default_factory=list)
    value_estimates: dict[HierarchyLevel, float] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action,
            "strategic_goals": [g.id for g in self.strategic_context],
            "tactical_goals": [g.id for g in self.tactical_context],
            "operational_goals": [g.id for g in self.operational_goals],
            "value_estimates": {k.value: v for k, v in self.value_estimates.items()},
            "reasoning": self.reasoning,
        }


class HierarchicalController:
    """Main controller that coordinates all hierarchy levels.

    This class manages the interaction between strategic, tactical, and
    operational controllers, ensuring coherent decision-making across
    all abstraction levels.

    Attributes:
        config: Hierarchical configuration.
        goal_hierarchy: Shared goal hierarchy.
        strategic: Strategic level controller.
        tactical: Tactical level controller.
        operational: Operational level controller.
    """

    def __init__(
        self,
        config: HierarchicalConfig | None = None,
        network: HierarchicalNetwork | None = None,
        llm_advisor: Any | None = None,
    ) -> None:
        """Initialize hierarchical controller.

        Args:
            config: Hierarchical configuration.
            network: Neural network (created if not provided).
            llm_advisor: Optional LLM strategy advisor.
        """
        self.config = config or HierarchicalConfig.default()
        self.llm_advisor = llm_advisor

        # Shared components
        self.goal_hierarchy = GoalHierarchy()
        self.goal_evaluator = GoalEvaluator()
        self.goal_propagator = GoalPropagator(self.goal_hierarchy)

        # Encoders
        self.state_encoder = HierarchicalStateEncoder()
        self.goal_encoder = HierarchicalGoalEncoder()

        # Network
        self.network = network
        if self.network is None and TORCH_AVAILABLE:
            self._create_network()

        # Level controllers
        self.strategic = self._create_strategic_controller()
        self.tactical = self._create_tactical_controller()
        self.operational = self._create_operational_controller()

        # Controller state
        self.state = HierarchicalState()

        logger.info(
            "HierarchicalController initialized with levels: strategic=%s, tactical=%s, operational=%s",
            self.config.is_level_enabled(HierarchyLevel.STRATEGIC),
            self.config.is_level_enabled(HierarchyLevel.TACTICAL),
            self.config.is_level_enabled(HierarchyLevel.OPERATIONAL),
        )

    def decide(
        self,
        game_state: dict[str, Any],
        valid_actions: np.ndarray | None = None,
    ) -> HierarchicalDecision:
        """Make a hierarchical decision for the current game state.

        This method coordinates all hierarchy levels:
        1. Strategic controller sets long-term goals (every N turns)
        2. Tactical controller decomposes into medium-term goals
        3. Operational controller selects the immediate action

        Args:
            game_state: Current game state.
            valid_actions: Mask of valid actions.

        Returns:
            HierarchicalDecision with selected action and context.
        """
        current_turn = game_state.get("turn", 0)
        self.state.current_turn = current_turn

        decision = HierarchicalDecision(action=0)
        reasoning_parts = []

        # Strategic level decision
        if self.config.is_level_enabled(HierarchyLevel.STRATEGIC):
            if self.strategic.should_update(current_turn):
                strategic_decision = self.strategic.decide(game_state)
                self.state.strategic_decisions.append(strategic_decision)
                decision.value_estimates[HierarchyLevel.STRATEGIC] = (
                    strategic_decision.value_estimate
                )
                reasoning_parts.append(f"Strategic: {strategic_decision.reasoning}")

        # Get strategic context for tactical level
        strategic_goals = self.strategic.get_active_goals()
        decision.strategic_context = strategic_goals

        # Tactical level decision
        if self.config.is_level_enabled(HierarchyLevel.TACTICAL):
            if self.tactical.should_update(current_turn):
                tactical_context = {"strategic_goals": strategic_goals}
                tactical_decision = self.tactical.decide(game_state, tactical_context)
                self.state.tactical_decisions.append(tactical_decision)
                decision.value_estimates[HierarchyLevel.TACTICAL] = tactical_decision.value_estimate
                reasoning_parts.append(f"Tactical: {tactical_decision.reasoning}")

        # Get tactical context for operational level
        tactical_goals = self.tactical.get_active_goals()
        decision.tactical_context = tactical_goals

        # Operational level decision (always runs)
        operational_context = {
            "strategic_goals": strategic_goals,
            "tactical_goals": tactical_goals,
        }
        operational_decision = self.operational.decide(game_state, operational_context)
        self.state.operational_decisions.append(operational_decision)

        decision.action = operational_decision.action or 0
        decision.value_estimates[HierarchyLevel.OPERATIONAL] = operational_decision.value_estimate
        decision.operational_goals = self.operational.get_active_goals()

        # Apply action mask
        if valid_actions is not None and decision.action is not None:
            if valid_actions[decision.action] == 0:
                # Selected action invalid, fall back
                valid_indices = np.where(valid_actions > 0)[0]
                if len(valid_indices) > 0:
                    decision.action = int(valid_indices[0])
                    reasoning_parts.append("(action overridden due to validity)")

        decision.reasoning = " | ".join(reasoning_parts)

        if self.config.log_level_decisions:
            logger.debug(
                "Turn %d decision: action=%d, values=%s",
                current_turn,
                decision.action,
                decision.value_estimates,
            )

        return decision

    def update(
        self,
        game_state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any],
        done: bool,
    ) -> dict[str, float]:
        """Update all controllers after action execution.

        Args:
            game_state: State before action.
            action: Action taken.
            reward: Reward received.
            next_state: State after action.
            done: Whether episode ended.

        Returns:
            Dictionary with shaped rewards for each level.
        """
        self.state.total_reward += reward

        # Compute goal-based reward shaping
        shaped_rewards = self._compute_shaped_rewards(game_state, action, reward, next_state)

        # Update each controller
        self.operational.update(
            next_state, shaped_rewards.get(HierarchyLevel.OPERATIONAL, reward), done
        )

        if self.config.is_level_enabled(HierarchyLevel.TACTICAL):
            self.tactical.update(
                next_state, shaped_rewards.get(HierarchyLevel.TACTICAL, reward), done
            )

        if self.config.is_level_enabled(HierarchyLevel.STRATEGIC):
            self.strategic.update(
                next_state, shaped_rewards.get(HierarchyLevel.STRATEGIC, reward), done
            )

        # Synchronize goal hierarchy
        context = {"current_turn": next_state.get("turn", 0)}
        self.goal_propagator.synchronize_hierarchy(next_state, context)

        if done:
            self.state.episodes += 1
            if self.config.log_goal_changes:
                self._log_episode_summary()

        return {k.value: v for k, v in shaped_rewards.items()}

    def reset(self) -> None:
        """Reset controller for new episode."""
        self.state = HierarchicalState()
        self.goal_hierarchy = GoalHierarchy()
        self.goal_propagator = GoalPropagator(self.goal_hierarchy)

        self.strategic.reset()
        self.tactical.reset()
        self.operational.reset()

        logger.debug("HierarchicalController reset for new episode")

    def get_training_data(self) -> dict[str, Any]:
        """Get training data from recent decisions.

        Returns:
            Dictionary with training samples for each level.
        """
        data: dict[str, list[dict[str, Any]]] = {
            "strategic": [],
            "tactical": [],
            "operational": [],
        }

        # Collect from operational decisions (most frequent)
        for decision in self.state.operational_decisions[-1000:]:
            if decision.search_stats:
                data["operational"].append(
                    {
                        "action": decision.action,
                        "value": decision.value_estimate,
                        "visit_counts": decision.search_stats.visit_counts,
                        "prior_probs": decision.search_stats.prior_probs,
                    }
                )

        # Collect from tactical decisions
        for decision in self.state.tactical_decisions[-100:]:
            if decision.search_stats:
                data["tactical"].append(
                    {
                        "goals": [g.id for g in decision.goals_created],
                        "value": decision.value_estimate,
                    }
                )

        # Collect from strategic decisions
        for decision in self.state.strategic_decisions[-20:]:
            data["strategic"].append(
                {
                    "goals": [g.id for g in decision.goals_created],
                    "value": decision.value_estimate,
                    "reasoning": decision.reasoning,
                }
            )

        return data

    def save(self, path: str | Path) -> None:
        """Save controller state and network weights.

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save network
        if self.network and TORCH_AVAILABLE:
            self.network.save(str(path / "hierarchical_network.pt"))

        # Save goal hierarchy
        self.goal_hierarchy.save(str(path / "goal_hierarchy.json"))

        logger.info("Saved HierarchicalController to %s", path)

    def load(self, path: str | Path, device: str = "cpu") -> None:
        """Load controller state and network weights.

        Args:
            path: Directory to load from.
            device: Device to load network to.
        """
        path = Path(path)

        # Load network
        network_path = path / "hierarchical_network.pt"
        if network_path.exists() and self.network and TORCH_AVAILABLE:
            self.network.load(str(network_path), device)

        # Load goal hierarchy
        hierarchy_path = path / "goal_hierarchy.json"
        if hierarchy_path.exists():
            self.goal_hierarchy.load(str(hierarchy_path))

        logger.info("Loaded HierarchicalController from %s", path)

    def _create_network(self) -> None:
        """Create hierarchical neural network."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, network not created")
            return

        self.network = HierarchicalNetwork(
            strategic_config=HierarchicalNetworkConfig.for_level(HierarchyLevel.STRATEGIC),
            tactical_config=HierarchicalNetworkConfig.for_level(HierarchyLevel.TACTICAL),
            operational_config=HierarchicalNetworkConfig.for_level(HierarchyLevel.OPERATIONAL),
        )

    def _create_strategic_controller(self) -> StrategicController:
        """Create strategic level controller."""
        return StrategicController(
            config=self.config.strategic,
            network=self.network.strategic_network if self.network else None,
            state_encoder=self.state_encoder.strategic,
            goal_encoder=self.goal_encoder,
            goal_hierarchy=self.goal_hierarchy,
            llm_advisor=self.llm_advisor,
        )

    def _create_tactical_controller(self) -> TacticalController:
        """Create tactical level controller."""
        return TacticalController(
            config=self.config.tactical,
            network=self.network.tactical_network if self.network else None,
            state_encoder=self.state_encoder.tactical,
            goal_encoder=self.goal_encoder,
            goal_hierarchy=self.goal_hierarchy,
        )

    def _create_operational_controller(self) -> OperationalController:
        """Create operational level controller."""
        return OperationalController(
            config=self.config.operational,
            network=self.network.operational_network if self.network else None,
            state_encoder=self.state_encoder.operational,
            goal_encoder=self.goal_encoder,
            goal_hierarchy=self.goal_hierarchy,
        )

    def _compute_shaped_rewards(
        self,
        game_state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any],
    ) -> dict[HierarchyLevel, float]:
        """Compute goal-shaped rewards for each level.

        Args:
            game_state: State before action.
            action: Action taken.
            reward: Environment reward.
            next_state: State after action.

        Returns:
            Shaped rewards for each level.
        """
        shaped = {}
        weight = self.config.goals.reward_shaping_weight

        # Operational: reward + immediate goal progress
        operational_goals = self.operational.get_active_goals()
        op_bonus = self._compute_goal_bonus(operational_goals, game_state, next_state)
        shaped[HierarchyLevel.OPERATIONAL] = reward + weight * op_bonus

        # Tactical: reward + tactical goal progress
        tactical_goals = self.tactical.get_active_goals()
        tac_bonus = self._compute_goal_bonus(tactical_goals, game_state, next_state)
        shaped[HierarchyLevel.TACTICAL] = reward + weight * tac_bonus * 0.5

        # Strategic: reward + strategic goal progress
        strategic_goals = self.strategic.get_active_goals()
        strat_bonus = self._compute_goal_bonus(strategic_goals, game_state, next_state)
        shaped[HierarchyLevel.STRATEGIC] = reward + weight * strat_bonus * 0.25

        return shaped

    def _compute_goal_bonus(
        self,
        goals: list[Goal],
        prev_state: dict[str, Any],
        next_state: dict[str, Any],
    ) -> float:
        """Compute bonus from goal progress.

        Args:
            goals: Active goals.
            prev_state: Previous state.
            next_state: Next state.

        Returns:
            Progress-based bonus.
        """
        if not goals:
            return 0.0

        total_bonus = 0.0
        for goal in goals:
            prev_progress = goal.progress
            # Evaluate new progress
            if goal.level == HierarchyLevel.STRATEGIC:
                result = self.goal_evaluator.evaluate_strategic_goal(goal, next_state)
            elif goal.level == HierarchyLevel.TACTICAL:
                result = self.goal_evaluator.evaluate_tactical_goal(goal, next_state)
            else:
                result = self.goal_evaluator.evaluate_operational_goal(goal, next_state)

            progress_delta = result.progress - prev_progress
            total_bonus += progress_delta * goal.priority * goal.reward_modifier

        return total_bonus

    def _log_episode_summary(self) -> None:
        """Log episode summary."""
        completed = sum(
            1 for g in self.goal_hierarchy.get_all_goals() if g.status == GoalStatus.COMPLETED
        )
        failed = sum(
            1 for g in self.goal_hierarchy.get_all_goals() if g.status == GoalStatus.FAILED
        )
        total_goals = len(self.goal_hierarchy.get_all_goals())

        logger.info(
            "Episode %d summary: reward=%.3f, goals=%d/%d completed, %d failed",
            self.state.episodes,
            self.state.total_reward,
            completed,
            total_goals,
            failed,
        )


def create_hierarchical_controller(
    config_path: str | Path | None = None,
    preset: str = "default",
    **kwargs: Any,
) -> HierarchicalController:
    """Factory function to create HierarchicalController.

    Args:
        config_path: Optional path to YAML configuration.
        preset: Configuration preset ("default", "minimal", "full").
        **kwargs: Override configuration values.

    Returns:
        Configured HierarchicalController.
    """
    if config_path:
        config = HierarchicalConfig.from_yaml(config_path)
    elif preset == "minimal":
        config = HierarchicalConfig.minimal()
    elif preset == "full":
        config = HierarchicalConfig.full()
    else:
        config = HierarchicalConfig.default()

    # Apply environment overrides
    config = config.apply_env_overrides()

    return HierarchicalController(config=config, **kwargs)
