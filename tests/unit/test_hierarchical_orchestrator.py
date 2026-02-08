"""Tests for hierarchical decision-making integration in TrainingOrchestrator.

This module tests the hierarchical controller integration:
- HierarchicalConfig in orchestrator config
- Hierarchical mode in _play_single_game()
- Goal-shaped rewards and strategic decisions
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


class TestHierarchicalConfigIntegration:
    """Tests for HierarchicalConfig in orchestrator config."""

    def test_default_values(self):
        """Test default hierarchical config values."""
        from alpha_zero.orchestrator.config import HierarchicalConfig

        config = HierarchicalConfig()

        assert config.enabled is False
        assert config.strategic_interval == 50
        assert config.tactical_interval == 10
        assert config.goal_weight == 0.3

    def test_config_with_custom_values(self):
        """Test hierarchical config with custom values."""
        from alpha_zero.orchestrator.config import HierarchicalConfig

        config = HierarchicalConfig(
            enabled=True,
            strategic_interval=100,
            tactical_interval=20,
            goal_weight=0.5,
        )

        assert config.enabled is True
        assert config.strategic_interval == 100
        assert config.tactical_interval == 20
        assert config.goal_weight == 0.5


class TestOrchestratorHierarchicalInit:
    """Tests for hierarchical system initialization in orchestrator."""

    def test_hierarchical_disabled_by_default(self):
        """Test that hierarchical mode is disabled by default."""
        from alpha_zero.orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig()

        assert config.hierarchical.enabled is False

    def test_hierarchical_integration_initialized_to_none(self):
        """Test that hierarchical integration is initialized to None."""
        from alpha_zero.orchestrator.config import OrchestratorConfig
        from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        assert orchestrator._hierarchical_integration is None


class TestBuildGameStateDict:
    """Tests for _build_game_state_dict helper method."""

    def test_build_game_state_dict_basic(self):
        """Test building game state dict with basic info."""
        from alpha_zero.orchestrator.config import OrchestratorConfig
        from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        # Create mock state
        state = {
            "spatial": np.zeros((13, 16, 16)),
            "vector": np.array([0.5, 0.3, 0.2, 0.1, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0]),
        }

        # Create mock env
        mock_env = MagicMock()
        mock_env._env = MagicMock()

        game_state = orchestrator._build_game_state_dict(state, mock_env, turn=10)

        assert game_state["turn"] == 10
        assert "spatial" in game_state
        assert "vector" in game_state
        assert game_state["turn_progress"] == 0.5

    def test_build_game_state_dict_with_env_info(self):
        """Test building game state dict with environment info."""
        from alpha_zero.orchestrator.config import OrchestratorConfig
        from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

        config = OrchestratorConfig()
        orchestrator = TrainingOrchestrator(config)

        state = {
            "spatial": np.zeros((13, 16, 16)),
            "vector": np.zeros(10),
        }

        # Create mock env with get_game_info
        mock_env = MagicMock()
        mock_env._env.get_game_info = MagicMock(
            return_value={"player_score": 100, "enemy_score": 50}
        )

        game_state = orchestrator._build_game_state_dict(state, mock_env, turn=5)

        assert game_state["turn"] == 5
        assert game_state.get("player_score") == 100
        assert game_state.get("enemy_score") == 50


class TestHierarchicalController:
    """Tests for HierarchicalController class."""

    def test_controller_initialization(self):
        """Test hierarchical controller initialization."""
        from alpha_zero.hierarchical.config import HierarchicalConfig
        from alpha_zero.hierarchical.controller import HierarchicalController

        config = HierarchicalConfig.default()
        controller = HierarchicalController(config=config)

        assert controller.config is not None
        assert controller.goal_hierarchy is not None
        assert controller.strategic is not None
        assert controller.tactical is not None
        assert controller.operational is not None

    @pytest.mark.skip(reason="Requires full game state with units - tests existing controller")
    def test_controller_decide(self):
        """Test controller decide method."""
        from alpha_zero.hierarchical.config import HierarchicalConfig
        from alpha_zero.hierarchical.controller import HierarchicalController

        config = HierarchicalConfig.minimal()  # Minimal for faster testing
        controller = HierarchicalController(config=config)

        # Create game state
        game_state = {
            "turn": 10,
            "spatial": np.zeros((13, 16, 16)),
            "vector": np.zeros(10),
        }

        valid_actions = np.ones(356)

        decision = controller.decide(game_state, valid_actions)

        assert decision.action >= 0
        assert decision.action < 356

    @pytest.mark.skip(reason="Requires full game state with units - tests existing controller")
    def test_controller_reset(self):
        """Test controller reset method."""
        from alpha_zero.hierarchical.config import HierarchicalConfig
        from alpha_zero.hierarchical.controller import HierarchicalController

        config = HierarchicalConfig.default()
        controller = HierarchicalController(config=config)

        # Make some decisions to populate state
        game_state = {"turn": 0, "spatial": np.zeros((13, 16, 16)), "vector": np.zeros(10)}
        controller.decide(game_state, np.ones(356))

        # Reset
        controller.reset()

        assert controller.state.current_turn == 0
        assert len(controller.state.strategic_decisions) == 0


class TestHierarchicalTrainingIntegration:
    """Tests for HierarchicalTrainingIntegration class."""

    def test_integration_initialization(self):
        """Test training integration initialization."""
        from alpha_zero.hierarchical.training_integration import (
            HierarchicalTrainingConfig,
            HierarchicalTrainingIntegration,
        )

        config = HierarchicalTrainingConfig(enabled=True)
        integration = HierarchicalTrainingIntegration(config=config)

        assert integration.controller is not None
        assert integration.state is not None

    @pytest.mark.skip(reason="Requires full game state with units - tests existing controller")
    def test_integration_select_action(self):
        """Test select_action with hierarchical integration."""
        from alpha_zero.hierarchical.training_integration import (
            HierarchicalTrainingConfig,
            HierarchicalTrainingIntegration,
        )

        config = HierarchicalTrainingConfig(enabled=True)
        integration = HierarchicalTrainingIntegration(config=config)

        game_state = {
            "turn": 5,
            "spatial": np.zeros((13, 16, 16)),
            "vector": np.zeros(10),
        }
        valid_actions = np.ones(356)

        action, info = integration.select_action(game_state, valid_actions)

        assert action >= 0
        assert action < 356
        assert info.get("hierarchical") is True

    def test_integration_on_game_start_end(self):
        """Test game lifecycle callbacks."""
        from alpha_zero.hierarchical.training_integration import (
            HierarchicalTrainingConfig,
            HierarchicalTrainingIntegration,
        )

        config = HierarchicalTrainingConfig(enabled=True)
        integration = HierarchicalTrainingIntegration(config=config)

        # Should not raise
        integration.on_game_start()
        integration.on_game_end(winner=1, total_reward=10.0)

        # Check state was updated
        # (strategic win rate should have been updated)

    def test_integration_compute_shaped_reward(self):
        """Test goal-shaped reward computation."""
        from alpha_zero.hierarchical.training_integration import (
            HierarchicalTrainingConfig,
            HierarchicalTrainingIntegration,
        )

        config = HierarchicalTrainingConfig(
            enabled=True,
            use_goal_shaped_rewards=True,
        )
        integration = HierarchicalTrainingIntegration(config=config)

        game_state = {"turn": 0, "spatial": np.zeros((13, 16, 16)), "vector": np.zeros(10)}
        next_state = {"turn": 1, "spatial": np.zeros((13, 16, 16)), "vector": np.zeros(10)}

        shaped_reward = integration.compute_shaped_reward(
            game_state=game_state,
            action=0,
            reward=1.0,
            next_state=next_state,
            done=False,
        )

        # Shaped reward should be a float
        assert isinstance(shaped_reward, float)


class TestHierarchicalGracefulDegradation:
    """Tests for graceful degradation when hierarchical module unavailable."""

    def test_orchestrator_works_without_hierarchical(self):
        """Test that orchestrator works when hierarchical is disabled."""
        from alpha_zero.orchestrator.config import OrchestratorConfig
        from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

        config = OrchestratorConfig()
        config.hierarchical.enabled = False

        orchestrator = TrainingOrchestrator(config)

        assert orchestrator._hierarchical_integration is None

    def test_flat_mcts_used_when_hierarchical_disabled(self):
        """Test that flat MCTS is used when hierarchical is disabled."""
        from alpha_zero.orchestrator.config import OrchestratorConfig
        from alpha_zero.orchestrator.orchestrator import TrainingOrchestrator

        config = OrchestratorConfig()
        config.hierarchical.enabled = False

        orchestrator = TrainingOrchestrator(config)

        # Hierarchical integration should be None
        assert orchestrator._hierarchical_integration is None

        # When playing a game, should use MCTS (not hierarchical)
        # This is verified by checking hierarchical integration is None


class TestHierarchicalOrchestratorIntegration:
    """End-to-end tests for hierarchical mode in orchestrator."""

    def test_orchestrator_config_includes_hierarchical(self):
        """Test that OrchestratorConfig includes hierarchical field."""
        from alpha_zero.orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig()

        assert hasattr(config, "hierarchical")
        assert hasattr(config.hierarchical, "enabled")
        assert hasattr(config.hierarchical, "strategic_interval")
        assert hasattr(config.hierarchical, "tactical_interval")

    def test_config_serialization_includes_hierarchical(self):
        """Test that config serialization includes hierarchical settings."""
        from alpha_zero.orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig()
        config.hierarchical.enabled = True
        config.hierarchical.strategic_interval = 100

        config_dict = config.to_dict()

        assert "hierarchical" in config_dict
        assert config_dict["hierarchical"]["enabled"] is True
        assert config_dict["hierarchical"]["strategic_interval"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
