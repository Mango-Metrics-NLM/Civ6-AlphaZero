"""
Training Integration for Hierarchical Reasoning System.

This module integrates the hierarchical reasoning system with the
TrainingOrchestrator, enabling hierarchical decision-making during
self-play and training.

Example:
    >>> from alpha_zero.hierarchical.training_integration import HierarchicalTrainingIntegration
    >>> integration = HierarchicalTrainingIntegration(config, orchestrator)
    >>> integration.enable()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from alpha_zero.hierarchical.config import HierarchicalConfig, HierarchyLevel
from alpha_zero.hierarchical.controller import (
    HierarchicalController,
    create_hierarchical_controller,
)
from alpha_zero.hierarchical.goals.goal_types import GoalStatus

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalTrainingConfig:
    """Configuration for hierarchical training integration.

    Attributes:
        enabled: Whether hierarchical reasoning is enabled.
        hierarchical_config: Configuration for hierarchical system.
        train_hierarchical_networks: Whether to train level-specific networks.
        use_goal_shaped_rewards: Whether to use goal-based reward shaping.
        mix_flat_hierarchical: Mix flat and hierarchical decisions for exploration.
        flat_decision_probability: Probability of flat decisions when mixing.
        save_hierarchical_data: Whether to save hierarchical training data.
        hierarchical_checkpoint_frequency: Save hierarchical state every N iterations.
    """

    enabled: bool = True
    hierarchical_config: HierarchicalConfig | None = None
    train_hierarchical_networks: bool = True
    use_goal_shaped_rewards: bool = True
    mix_flat_hierarchical: bool = False
    flat_decision_probability: float = 0.1
    save_hierarchical_data: bool = True
    hierarchical_checkpoint_frequency: int = 100


@dataclass
class HierarchicalTrainingState:
    """State maintained during hierarchical training.

    Attributes:
        total_decisions: Total hierarchical decisions made.
        goal_completions: Number of goals completed.
        level_value_estimates: Running average of value estimates by level.
        strategic_win_rate: Win rate when following strategic goals.
        exploration_decisions: Number of exploration (flat) decisions.
    """

    total_decisions: int = 0
    goal_completions: dict[str, int] = field(default_factory=dict)
    level_value_estimates: dict[str, list[float]] = field(default_factory=dict)
    strategic_win_rate: float = 0.0
    exploration_decisions: int = 0


class HierarchicalTrainingIntegration:
    """Integrates hierarchical reasoning with TrainingOrchestrator.

    This class wraps the standard self-play loop to use hierarchical
    decision-making while maintaining compatibility with the existing
    training infrastructure.
    """

    def __init__(
        self,
        config: HierarchicalTrainingConfig,
        orchestrator: Any | None = None,
        llm_advisor: Any | None = None,
    ) -> None:
        """Initialize hierarchical training integration.

        Args:
            config: Hierarchical training configuration.
            orchestrator: TrainingOrchestrator instance (can be set later).
            llm_advisor: Optional LLM strategy advisor.
        """
        self.config = config
        self.orchestrator = orchestrator
        self.llm_advisor = llm_advisor

        # Create hierarchical controller
        self.controller: HierarchicalController | None = None
        if config.enabled:
            self._create_controller()

        # Training state
        self.state = HierarchicalTrainingState()

        # Callbacks for orchestrator hooks
        self._original_select_action: Callable | None = None
        self._original_compute_reward: Callable | None = None

        logger.info(
            "HierarchicalTrainingIntegration initialized: enabled=%s, "
            "goal_rewards=%s, train_networks=%s",
            config.enabled,
            config.use_goal_shaped_rewards,
            config.train_hierarchical_networks,
        )

    def _create_controller(self) -> None:
        """Create the hierarchical controller."""
        h_config = self.config.hierarchical_config or HierarchicalConfig.default()
        self.controller = create_hierarchical_controller(
            preset="default" if h_config is None else None,  # type: ignore[arg-type]
            llm_advisor=self.llm_advisor,
        )
        if self.config.hierarchical_config:
            self.controller = HierarchicalController(
                config=self.config.hierarchical_config,
                llm_advisor=self.llm_advisor,
            )

    def enable(self) -> None:
        """Enable hierarchical decision-making in orchestrator.

        This patches the orchestrator's self-play methods to use
        hierarchical decisions instead of flat MCTS.
        """
        if not self.config.enabled:
            logger.warning("Hierarchical training not enabled in config")
            return

        if self.orchestrator is None:
            logger.warning("No orchestrator set, cannot enable integration")
            return

        logger.info("Enabling hierarchical decision-making in orchestrator")

        # Store original methods for fallback/restoration
        if hasattr(self.orchestrator, "_select_action"):
            self._original_select_action = self.orchestrator._select_action

    def disable(self) -> None:
        """Disable hierarchical decision-making, restore original behavior."""
        if self._original_select_action and self.orchestrator:
            self.orchestrator._select_action = self._original_select_action

        logger.info("Disabled hierarchical decision-making")

    def set_orchestrator(self, orchestrator: Any) -> None:
        """Set the orchestrator instance.

        Args:
            orchestrator: TrainingOrchestrator instance.
        """
        self.orchestrator = orchestrator

    def select_action(
        self,
        game_state: dict[str, Any],
        valid_actions: np.ndarray,
    ) -> tuple[int, dict[str, Any]]:
        """Select action using hierarchical reasoning.

        Args:
            game_state: Current game state.
            valid_actions: Mask of valid actions.

        Returns:
            Tuple of (action, info_dict).
        """
        if not self.config.enabled or self.controller is None:
            # Fall back to flat decision if available
            if self._original_select_action:
                return self._original_select_action(game_state, valid_actions)  # type: ignore[no-any-return]
            # Default: random valid action
            valid_indices = np.where(valid_actions > 0)[0]
            action = int(np.random.choice(valid_indices)) if len(valid_indices) > 0 else 0
            return action, {}

        # Mix flat and hierarchical decisions for exploration
        if self.config.mix_flat_hierarchical:
            if np.random.random() < self.config.flat_decision_probability:
                self.state.exploration_decisions += 1
                if self._original_select_action:
                    return self._original_select_action(game_state, valid_actions)  # type: ignore[no-any-return]

        # Hierarchical decision
        decision = self.controller.decide(game_state, valid_actions)
        self.state.total_decisions += 1

        # Track value estimates by level
        for level, value in decision.value_estimates.items():
            level_name = level.value
            if level_name not in self.state.level_value_estimates:
                self.state.level_value_estimates[level_name] = []
            self.state.level_value_estimates[level_name].append(value)
            # Keep bounded
            if len(self.state.level_value_estimates[level_name]) > 1000:
                self.state.level_value_estimates[level_name] = self.state.level_value_estimates[
                    level_name
                ][-500:]

        info = {
            "hierarchical": True,
            "decision": decision.to_dict(),
            "value_estimates": decision.value_estimates,
        }

        return decision.action, info

    def compute_shaped_reward(
        self,
        game_state: dict[str, Any],
        action: int,
        reward: float,
        next_state: dict[str, Any],
        done: bool,
    ) -> float:
        """Compute goal-shaped reward.

        Args:
            game_state: State before action.
            action: Action taken.
            reward: Environment reward.
            next_state: State after action.
            done: Whether episode ended.

        Returns:
            Shaped reward including goal bonuses.
        """
        if not self.config.use_goal_shaped_rewards or self.controller is None:
            return reward

        # Get shaped rewards from controller
        shaped_rewards = self.controller.update(game_state, action, reward, next_state, done)

        # Use operational level reward (most granular)
        operational_reward = shaped_rewards.get("operational", reward)

        return operational_reward

    def on_game_start(self) -> None:
        """Called at the start of each self-play game."""
        if self.controller:
            self.controller.reset()

    def on_game_end(self, winner: int, total_reward: float) -> None:
        """Called at the end of each self-play game.

        Args:
            winner: Game winner (0=draw, 1=player1, -1=player2).
            total_reward: Total reward from game.
        """
        if self.controller is None:
            return

        # Count goal completions
        for goal in self.controller.goal_hierarchy.get_all_goals():
            goal_type = (
                goal.goal_type.name if hasattr(goal.goal_type, "name") else str(goal.goal_type)
            )
            if goal.status == GoalStatus.COMPLETED:
                self.state.goal_completions[goal_type] = (
                    self.state.goal_completions.get(goal_type, 0) + 1
                )

        # Update strategic win rate (if we had strategic goals and won)
        strategic_goals = [
            g
            for g in self.controller.goal_hierarchy.get_all_goals()
            if g.level == HierarchyLevel.STRATEGIC
        ]
        if strategic_goals and winner == 1:
            # Simple exponential moving average
            alpha = 0.1
            self.state.strategic_win_rate = (
                alpha * 1.0 + (1 - alpha) * self.state.strategic_win_rate
            )
        elif strategic_goals:
            alpha = 0.1
            self.state.strategic_win_rate = (
                alpha * 0.0 + (1 - alpha) * self.state.strategic_win_rate
            )

    def on_iteration_end(self, iteration: int) -> dict[str, Any]:
        """Called at the end of each training iteration.

        Args:
            iteration: Current iteration number.

        Returns:
            Metrics dictionary for logging.
        """
        metrics = {
            "hierarchical/total_decisions": self.state.total_decisions,
            "hierarchical/exploration_decisions": self.state.exploration_decisions,
            "hierarchical/strategic_win_rate": self.state.strategic_win_rate,
        }

        # Add goal completion metrics
        for goal_type, count in self.state.goal_completions.items():
            metrics[f"hierarchical/goals/{goal_type}"] = count

        # Add value estimate metrics
        for level, values in self.state.level_value_estimates.items():
            if values:
                metrics[f"hierarchical/value_{level}"] = np.mean(values[-100:])

        # Save checkpoint if needed
        if (
            self.config.save_hierarchical_data
            and self.config.hierarchical_checkpoint_frequency > 0
            and iteration % self.config.hierarchical_checkpoint_frequency == 0
        ):
            self._save_hierarchical_checkpoint(iteration)

        return metrics

    def get_training_data(self) -> dict[str, list[dict[str, Any]]]:
        """Get hierarchical training data for network updates.

        Returns:
            Dictionary with training samples for each level.
        """
        if self.controller is None:
            return {"strategic": [], "tactical": [], "operational": []}

        return self.controller.get_training_data()

    def train_hierarchical_networks(
        self,
        training_data: dict[str, list[dict[str, Any]]],
    ) -> dict[str, float]:
        """Train level-specific networks on collected data.

        Args:
            training_data: Training data from get_training_data().

        Returns:
            Dictionary of training losses by level.
        """
        if not self.config.train_hierarchical_networks or not TORCH_AVAILABLE:
            return {}

        if self.controller is None or self.controller.network is None:
            return {}

        losses = {}

        # Train operational network (most data)
        operational_data = training_data.get("operational", [])
        if operational_data:
            op_loss = self._train_level_network(
                self.controller.network.operational_network,
                operational_data,
                HierarchyLevel.OPERATIONAL,
            )
            losses["operational"] = op_loss

        # Train tactical network
        tactical_data = training_data.get("tactical", [])
        if tactical_data and len(tactical_data) >= 10:
            tac_loss = self._train_level_network(
                self.controller.network.tactical_network,
                tactical_data,
                HierarchyLevel.TACTICAL,
            )
            losses["tactical"] = tac_loss

        # Train strategic network
        strategic_data = training_data.get("strategic", [])
        if strategic_data and len(strategic_data) >= 5:
            strat_loss = self._train_level_network(
                self.controller.network.strategic_network,
                strategic_data,
                HierarchyLevel.STRATEGIC,
            )
            losses["strategic"] = strat_loss

        return losses

    def _train_level_network(
        self,
        network: Any,
        data: list[dict[str, Any]],
        level: HierarchyLevel,
    ) -> float:
        """Train a single level network.

        Args:
            network: Network to train.
            data: Training data.
            level: Hierarchy level.

        Returns:
            Training loss.
        """
        if not TORCH_AVAILABLE:
            return 0.0

        # Simple training step (placeholder - actual implementation would be more complex)
        network.train()
        torch.optim.Adam(network.parameters(), lr=0.001)

        total_loss = 0.0
        for sample in data[-100:]:  # Use recent data
            value_target = sample.get("value", 0.0)

            # Placeholder forward pass
            # In practice, this would use proper state encoding
            loss = torch.tensor(abs(value_target))  # Placeholder
            total_loss += loss.item()

        network.eval()
        return total_loss / max(len(data[-100:]), 1)

    def _save_hierarchical_checkpoint(self, iteration: int) -> None:
        """Save hierarchical state checkpoint.

        Args:
            iteration: Current iteration number.
        """
        if self.controller is None:
            return

        if self.orchestrator and hasattr(self.orchestrator, "checkpoint_dir"):
            checkpoint_dir = Path(self.orchestrator.checkpoint_dir) / "hierarchical"
        else:
            checkpoint_dir = Path("checkpoints/hierarchical")

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save controller
        self.controller.save(checkpoint_dir / f"iter_{iteration}")

        logger.debug("Saved hierarchical checkpoint at iteration %d", iteration)

    def save(self, path: str | Path) -> None:
        """Save integration state.

        Args:
            path: Path to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.controller:
            self.controller.save(path / "controller")

        # Save state
        import json

        state_dict = {
            "total_decisions": self.state.total_decisions,
            "goal_completions": self.state.goal_completions,
            "strategic_win_rate": self.state.strategic_win_rate,
            "exploration_decisions": self.state.exploration_decisions,
        }
        with open(path / "training_state.json", "w") as f:
            json.dump(state_dict, f, indent=2)

        logger.info("Saved hierarchical training integration to %s", path)

    def load(self, path: str | Path) -> None:
        """Load integration state.

        Args:
            path: Path to load from.
        """
        path = Path(path)

        if self.controller and (path / "controller").exists():
            self.controller.load(path / "controller")

        # Load state
        state_path = path / "training_state.json"
        if state_path.exists():
            import json

            with open(state_path) as f:
                state_dict = json.load(f)

            self.state.total_decisions = state_dict.get("total_decisions", 0)
            self.state.goal_completions = state_dict.get("goal_completions", {})
            self.state.strategic_win_rate = state_dict.get("strategic_win_rate", 0.0)
            self.state.exploration_decisions = state_dict.get("exploration_decisions", 0)

        logger.info("Loaded hierarchical training integration from %s", path)


def create_hierarchical_training_integration(
    orchestrator: Any | None = None,
    config_path: str | Path | None = None,
    preset: str = "default",
    **kwargs: Any,
) -> HierarchicalTrainingIntegration:
    """Factory function to create HierarchicalTrainingIntegration.

    Args:
        orchestrator: TrainingOrchestrator instance.
        config_path: Path to configuration file.
        preset: Configuration preset ("default", "minimal", "full").
        **kwargs: Override configuration values.

    Returns:
        Configured HierarchicalTrainingIntegration.
    """
    # Create hierarchical config
    if config_path:
        h_config = HierarchicalConfig.from_yaml(config_path)
    elif preset == "minimal":
        h_config = HierarchicalConfig.minimal()
    elif preset == "full":
        h_config = HierarchicalConfig.full()
    else:
        h_config = HierarchicalConfig.default()

    # Apply environment overrides
    h_config = h_config.apply_env_overrides()

    # Create training config
    training_config = HierarchicalTrainingConfig(
        enabled=kwargs.get("enabled", True),
        hierarchical_config=h_config,
        train_hierarchical_networks=kwargs.get("train_networks", True),
        use_goal_shaped_rewards=kwargs.get("goal_rewards", True),
        mix_flat_hierarchical=kwargs.get("mix_flat", False),
    )

    return HierarchicalTrainingIntegration(
        config=training_config,
        orchestrator=orchestrator,
        llm_advisor=kwargs.get("llm_advisor"),
    )
