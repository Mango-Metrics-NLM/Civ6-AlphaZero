"""Tests for hierarchical controller and level controllers."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.hierarchical.config import (
    HierarchicalConfig,
    HierarchyLevel,
    LevelConfig,
)
from alpha_zero.hierarchical.controller import (
    HierarchicalController,
    HierarchicalDecision,
    create_hierarchical_controller,
)
from alpha_zero.hierarchical.controllers.level_controller import (
    ControllerDecision,
    ControllerState,
    OperationalController,
    StrategicController,
    TacticalController,
)
from alpha_zero.hierarchical.goals.goal_types import (
    Goal,
    StrategicGoalType,
)


class TestControllerState:
    """Tests for ControllerState."""

    def test_default_state(self) -> None:
        """Test default controller state."""
        state = ControllerState()

        assert state.active_goals == []
        assert state.last_decision_turn == -9999  # Sentinel for first update
        assert state.cumulative_reward == 0.0
        assert state.episodes_completed == 0


class TestControllerDecision:
    """Tests for ControllerDecision."""

    def test_decision_creation(self) -> None:
        """Test creating controller decision."""
        decision = ControllerDecision(
            level=HierarchyLevel.TACTICAL,
            turn=10,
            action=5,
        )

        assert decision.level == HierarchyLevel.TACTICAL
        assert decision.turn == 10
        assert decision.action == 5


class TestStrategicController:
    """Tests for StrategicController."""

    def test_controller_creation(self) -> None:
        """Test creating strategic controller."""
        config = LevelConfig.for_level(HierarchyLevel.STRATEGIC)
        controller = StrategicController(config=config)

        assert controller.level == HierarchyLevel.STRATEGIC
        assert controller.state is not None

    def test_should_update(self) -> None:
        """Test update frequency check."""
        config = LevelConfig.for_level(HierarchyLevel.STRATEGIC)
        config.update_frequency = 10
        controller = StrategicController(config=config)

        # Initially should update
        assert controller.should_update(0) is True

        # After decision at turn 0
        controller.state.last_decision_turn = 0
        assert controller.should_update(5) is False  # Too soon
        assert controller.should_update(10) is True  # Time to update

    def test_decide(self) -> None:
        """Test making strategic decision."""
        config = LevelConfig.for_level(HierarchyLevel.STRATEGIC)
        controller = StrategicController(config=config)

        game_state = {
            "turn": 0,
            "victory_progress": {"domination": 0.1, "science": 0.2},
            "military": {"strength": 100, "avg_enemy_strength": 100},
            "resources": {"gold": 500},
            "yields": {"science_per_turn": 20, "culture_per_turn": 10},
            "technology": {"era": 1},
        }

        decision = controller.decide(game_state)

        assert isinstance(decision, ControllerDecision)
        assert decision.level == HierarchyLevel.STRATEGIC
        assert decision.turn == 0

    def test_update(self) -> None:
        """Test updating controller state."""
        config = LevelConfig.for_level(HierarchyLevel.STRATEGIC)
        controller = StrategicController(config=config)

        # First make a decision to create goals
        game_state = {
            "turn": 0,
            "victory_progress": {"science": 0.3},
            "military": {"strength": 100},
            "resources": {"gold": 500},
            "yields": {"science_per_turn": 30},
            "technology": {"era": 2},
        }
        controller.decide(game_state)

        # Now update
        controller.update(game_state, reward=1.0, done=False)

        assert controller.state.cumulative_reward == 1.0

    def test_reset(self) -> None:
        """Test resetting controller."""
        config = LevelConfig.for_level(HierarchyLevel.STRATEGIC)
        controller = StrategicController(config=config)

        controller.state.cumulative_reward = 100.0
        controller.state.last_decision_turn = 50

        controller.reset()

        assert controller.state.cumulative_reward == 0.0
        assert controller.state.last_decision_turn == -9999  # Reset to sentinel


class TestTacticalController:
    """Tests for TacticalController."""

    def test_controller_creation(self) -> None:
        """Test creating tactical controller."""
        config = LevelConfig.for_level(HierarchyLevel.TACTICAL)
        controller = TacticalController(config=config)

        assert controller.level == HierarchyLevel.TACTICAL

    def test_decide_with_strategic_context(self) -> None:
        """Test tactical decision with strategic goals."""
        config = LevelConfig.for_level(HierarchyLevel.TACTICAL)
        controller = TacticalController(config=config)

        strategic_goal = Goal.strategic(StrategicGoalType.DOMINATION_VICTORY)
        context = {"strategic_goals": [strategic_goal]}

        game_state = {
            "turn": 5,
            "units": [],
            "cities": [{"player": 0, "x": 5, "y": 5}],
            "current_player": 0,
            "resources": {"gold": 100},
        }

        decision = controller.decide(game_state, context)

        assert isinstance(decision, ControllerDecision)
        assert decision.level == HierarchyLevel.TACTICAL

    def test_identify_threats(self) -> None:
        """Test threat identification."""
        config = LevelConfig.for_level(HierarchyLevel.TACTICAL)
        controller = TacticalController(config=config)

        game_state = {
            "turn": 10,
            "units": [
                {"player": 0, "x": 5, "y": 5},  # Own unit
                {"player": 1, "x": 6, "y": 5},  # Enemy nearby
            ],
            "cities": [{"player": 0, "x": 5, "y": 5}],
            "current_player": 0,
            "resources": {},
        }

        enemies_nearby = controller._count_nearby_enemies(game_state)
        assert enemies_nearby > 0


class TestOperationalController:
    """Tests for OperationalController."""

    def test_controller_creation(self) -> None:
        """Test creating operational controller."""
        config = LevelConfig.for_level(HierarchyLevel.OPERATIONAL)
        controller = OperationalController(config=config)

        assert controller.level == HierarchyLevel.OPERATIONAL

    def test_decide_selects_action(self) -> None:
        """Test operational decision selects action."""
        config = LevelConfig.for_level(HierarchyLevel.OPERATIONAL)
        controller = OperationalController(config=config)

        game_state = {
            "turn": 0,
            "action_mask": np.ones(356),  # All actions valid
        }

        decision = controller.decide(game_state)

        assert isinstance(decision, ControllerDecision)
        assert decision.action is not None
        assert 0 <= decision.action < 356

    def test_decide_with_masked_actions(self) -> None:
        """Test decision respects action mask."""
        config = LevelConfig.for_level(HierarchyLevel.OPERATIONAL)
        controller = OperationalController(config=config)

        # Only actions 0, 10, 20 are valid
        action_mask = np.zeros(356)
        action_mask[0] = 1
        action_mask[10] = 1
        action_mask[20] = 1

        game_state = {
            "turn": 0,
            "action_mask": action_mask,
        }

        decision = controller.decide(game_state)

        assert decision.action in [0, 10, 20]


class TestHierarchicalConfig:
    """Tests for HierarchicalConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = HierarchicalConfig.default()

        assert config.enabled is True
        assert config.strategic is not None
        assert config.tactical is not None
        assert config.operational is not None

    def test_minimal_config(self) -> None:
        """Test minimal configuration."""
        config = HierarchicalConfig.minimal()

        assert config.strategic.enabled is False
        assert config.tactical.enabled is False
        assert config.operational.enabled is True

    def test_full_config(self) -> None:
        """Test full configuration."""
        config = HierarchicalConfig.full()

        assert config.enabled is True
        assert config.strategic.enabled is True
        assert config.tactical.enabled is True
        assert config.operational.enabled is True
        assert config.llm_guidance.enabled is True

    def test_is_level_enabled(self) -> None:
        """Test level enabled check."""
        config = HierarchicalConfig.minimal()

        assert config.is_level_enabled(HierarchyLevel.STRATEGIC) is False
        assert config.is_level_enabled(HierarchyLevel.OPERATIONAL) is True


@pytest.fixture
def fast_config() -> HierarchicalConfig:
    """HierarchicalConfig with MCTS simulations reduced to 1 for fast tests."""
    cfg = HierarchicalConfig.default()
    cfg.strategic.mcts.num_simulations = 1
    cfg.tactical.mcts.num_simulations = 1
    cfg.operational.mcts.num_simulations = 1
    return cfg


class TestHierarchicalController:
    """Tests for HierarchicalController."""

    def test_controller_creation(self) -> None:
        """Test creating hierarchical controller."""
        config = HierarchicalConfig.default()
        controller = HierarchicalController(config=config)

        assert controller.strategic is not None
        assert controller.tactical is not None
        assert controller.operational is not None

    def test_decide(self, fast_config: HierarchicalConfig) -> None:
        """Test making hierarchical decision."""
        controller = HierarchicalController(config=fast_config)

        game_state = {
            "turn": 0,
            "victory_progress": {},
            "military": {},
            "resources": {},
            "yields": {},
            "technology": {},
            "action_mask": np.ones(356),
        }

        decision = controller.decide(game_state)

        assert isinstance(decision, HierarchicalDecision)
        assert decision.action is not None

    def test_decide_with_valid_actions(self, fast_config: HierarchicalConfig) -> None:
        """Test decision respects valid actions."""
        controller = HierarchicalController(config=fast_config)

        valid_actions = np.zeros(356)
        valid_actions[42] = 1

        game_state = {
            "turn": 0,
            "victory_progress": {},
            "military": {},
            "resources": {},
            "yields": {},
            "technology": {},
            "action_mask": valid_actions,
        }

        decision = controller.decide(game_state, valid_actions=valid_actions)

        assert decision.action == 42

    def test_update(self) -> None:
        """Test updating controller."""
        config = HierarchicalConfig.default()
        controller = HierarchicalController(config=config)

        game_state = {"turn": 0, "action_mask": np.ones(356)}
        next_state = {"turn": 1, "action_mask": np.ones(356)}

        shaped_rewards = controller.update(
            game_state, action=0, reward=1.0, next_state=next_state, done=False
        )

        assert isinstance(shaped_rewards, dict)
        assert controller.state.total_reward == 1.0

    def test_reset(self) -> None:
        """Test resetting controller."""
        config = HierarchicalConfig.default()
        controller = HierarchicalController(config=config)

        controller.state.total_reward = 100.0
        controller.state.episodes = 5

        controller.reset()

        assert controller.state.total_reward == 0.0
        assert controller.state.episodes == 0

    def test_get_training_data(self, fast_config: HierarchicalConfig) -> None:
        """Test getting training data."""
        controller = HierarchicalController(config=fast_config)

        # Make a decision to populate training data buffers
        game_state = {"turn": 0, "action_mask": np.ones(356)}
        controller.decide(game_state)

        data = controller.get_training_data()

        assert "strategic" in data
        assert "tactical" in data
        assert "operational" in data


class TestCreateHierarchicalController:
    """Tests for factory function."""

    def test_create_default(self) -> None:
        """Test creating with default preset."""
        controller = create_hierarchical_controller(preset="default")

        assert isinstance(controller, HierarchicalController)

    def test_create_minimal(self) -> None:
        """Test creating with minimal preset."""
        controller = create_hierarchical_controller(preset="minimal")

        assert isinstance(controller, HierarchicalController)
        assert controller.config.strategic.enabled is False

    def test_create_full(self) -> None:
        """Test creating with full preset."""
        controller = create_hierarchical_controller(preset="full")

        assert isinstance(controller, HierarchicalController)
        assert controller.config.strategic.enabled is True


class TestHierarchicalDecision:
    """Tests for HierarchicalDecision."""

    def test_decision_creation(self) -> None:
        """Test creating hierarchical decision."""
        decision = HierarchicalDecision(action=10)

        assert decision.action == 10
        assert decision.strategic_context == []
        assert decision.tactical_context == []

    def test_to_dict(self) -> None:
        """Test converting decision to dict."""
        strategic = Goal.strategic(StrategicGoalType.SCIENCE_VICTORY)
        decision = HierarchicalDecision(
            action=5,
            strategic_context=[strategic],
            value_estimates={HierarchyLevel.STRATEGIC: 0.5},
        )

        d = decision.to_dict()

        assert d["action"] == 5
        assert strategic.id in d["strategic_goals"]
        assert "strategic" in d["value_estimates"]
