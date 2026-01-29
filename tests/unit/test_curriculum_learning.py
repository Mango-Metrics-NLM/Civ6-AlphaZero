"""Unit tests for curriculum learning module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from alpha_zero.utils.curriculum_learning import (
    AdaptiveDifficultyController,
    CurriculumConfig,
    CurriculumManager,
    CurriculumStage,
    CurriculumState,
    DifficultyDimension,
    create_curriculum_manager,
)


class TestCurriculumStage:
    """Tests for CurriculumStage."""

    def test_default_stage(self):
        """Test default stage values."""
        stage = CurriculumStage(name="test")

        assert stage.name == "test"
        assert stage.level == 0
        assert stage.board_size == 8
        assert stage.max_turns == 50
        assert stage.promotion_threshold == 0.6

    def test_to_dict(self):
        """Test stage to dictionary conversion."""
        stage = CurriculumStage(
            name="beginner",
            level=1,
            board_size=10,
            promotion_threshold=0.7,
        )

        data = stage.to_dict()

        assert data["name"] == "beginner"
        assert data["level"] == 1
        assert data["board_size"] == 10
        assert data["promotion_threshold"] == 0.7

    def test_from_dict(self):
        """Test stage from dictionary."""
        data = {
            "name": "advanced",
            "level": 2,
            "board_size": 16,
            "max_turns": 100,
            "fog_of_war_enabled": True,
        }

        stage = CurriculumStage.from_dict(data)

        assert stage.name == "advanced"
        assert stage.level == 2
        assert stage.board_size == 16
        assert stage.fog_of_war_enabled is True


class TestCurriculumConfig:
    """Tests for CurriculumConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CurriculumConfig()

        assert config.enabled is True
        assert len(config.stages) == 3  # Default three stages
        assert config.allow_demotion is True

    def test_stages_sorted_by_level(self):
        """Test that stages are sorted by level."""
        stages = [
            CurriculumStage(name="b", level=2),
            CurriculumStage(name="a", level=0),
            CurriculumStage(name="c", level=1),
        ]

        config = CurriculumConfig(stages=stages)

        assert config.stages[0].level == 0
        assert config.stages[1].level == 1
        assert config.stages[2].level == 2

    def test_default_three_stage(self):
        """Test three-stage curriculum factory."""
        config = CurriculumConfig.default_three_stage()

        assert len(config.stages) == 3
        assert config.stages[0].name == "beginner"
        assert config.stages[2].name == "advanced"

    def test_default_five_stage(self):
        """Test five-stage curriculum factory."""
        config = CurriculumConfig.default_five_stage()

        assert len(config.stages) == 5
        assert config.stages[0].name == "tutorial"
        assert config.stages[4].name == "expert"

    def test_to_dict(self):
        """Test config to dictionary."""
        config = CurriculumConfig.default_three_stage()

        data = config.to_dict()

        assert data["enabled"] is True
        assert len(data["stages"]) == 3
        assert data["allow_demotion"] is True

    def test_from_dict(self):
        """Test config from dictionary."""
        data = {
            "enabled": True,
            "stages": [
                {"name": "easy", "level": 0, "board_size": 6},
                {"name": "hard", "level": 1, "board_size": 12},
            ],
            "allow_demotion": False,
            "metrics_window": 50,
        }

        config = CurriculumConfig.from_dict(data)

        assert len(config.stages) == 2
        assert config.allow_demotion is False
        assert config.metrics_window == 50


class TestCurriculumManager:
    """Tests for CurriculumManager."""

    @pytest.fixture
    def manager(self):
        """Create curriculum manager."""
        config = CurriculumConfig.default_three_stage()
        return CurriculumManager(config)

    def test_get_current_stage(self, manager):
        """Test getting current stage."""
        stage = manager.get_current_stage()

        assert stage.name == "beginner"
        assert stage.level == 0

    def test_get_environment_params(self, manager):
        """Test getting environment parameters."""
        params = manager.get_environment_params()

        assert "board_size" in params
        assert "max_turns" in params
        assert "production_multiplier" in params

    def test_record_game_win(self, manager):
        """Test recording a win."""
        result = manager.record_game(won=True)

        assert result["changed"] is False
        assert "win_rate" in result

    def test_record_game_loss(self, manager):
        """Test recording a loss."""
        result = manager.record_game(won=False)

        assert result["changed"] is False

    def test_promotion(self, manager):
        """Test stage promotion."""
        stage = manager.get_current_stage()

        # Record many wins to trigger promotion
        for _ in range(stage.min_games):
            manager.record_game(won=True)

        # Should have promoted
        new_stage = manager.get_current_stage()
        assert new_stage.level > 0

    def test_demotion(self, manager):
        """Test stage demotion."""
        # First promote
        stage = manager.get_current_stage()
        for _ in range(stage.min_games):
            manager.record_game(won=True)

        # Now at intermediate
        assert manager.get_current_stage().level == 1

        # Record many losses
        stage = manager.get_current_stage()
        for _ in range(stage.min_games):
            manager.record_game(won=False)

        # Should have demoted
        assert manager.get_current_stage().level == 0

    def test_demotion_disabled(self):
        """Test that demotion can be disabled."""
        config = CurriculumConfig.default_three_stage()
        config.allow_demotion = False
        manager = CurriculumManager(config)

        # Promote first
        stage = manager.get_current_stage()
        for _ in range(stage.min_games):
            manager.record_game(won=True)

        assert manager.get_current_stage().level == 1

        # Now record losses
        stage = manager.get_current_stage()
        for _ in range(stage.min_games + 10):
            manager.record_game(won=False)

        # Should NOT have demoted
        assert manager.get_current_stage().level == 1

    def test_force_stage(self, manager):
        """Test forcing stage change."""
        result = manager.force_stage(2)

        assert result["changed"] is True
        assert result["type"] == "forced"
        assert manager.get_current_stage().level == 2

    def test_force_invalid_stage(self, manager):
        """Test forcing invalid stage index."""
        with pytest.raises(ValueError):
            manager.force_stage(100)

    def test_get_progress(self, manager):
        """Test getting progress summary."""
        # Record some games
        for _ in range(10):
            manager.record_game(won=True)
        for _ in range(5):
            manager.record_game(won=False)

        progress = manager.get_progress()

        assert progress["current_stage"] == "beginner"
        assert progress["games_at_stage"] == 15
        assert progress["total_games"] == 15
        assert "win_rate" in progress

    def test_reset(self, manager):
        """Test curriculum reset."""
        # Record some games and promote
        for _ in range(100):
            manager.record_game(won=True)

        assert manager.get_current_stage().level > 0

        manager.reset()

        assert manager.get_current_stage().level == 0
        assert manager.get_progress()["total_games"] == 0


class TestCurriculumManagerPersistence:
    """Tests for curriculum state persistence."""

    def test_save_and_load_state(self):
        """Test saving and loading curriculum state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "curriculum_state.json"

            # Create manager with save path
            config = CurriculumConfig.default_three_stage()
            config.save_path = str(save_path)
            manager = CurriculumManager(config)

            # Record some games
            for _ in range(20):
                manager.record_game(won=True)

            # Force promotion
            manager.force_stage(1)

            # Create new manager from same save path
            manager2 = CurriculumManager(config)

            # Should have same state
            assert manager2.get_current_stage().level == 1
            assert manager2.get_progress()["total_games"] == 20


class TestAdaptiveDifficultyController:
    """Tests for AdaptiveDifficultyController."""

    @pytest.fixture
    def controller(self):
        """Create adaptive controller."""
        config = CurriculumConfig()
        return AdaptiveDifficultyController(config, target_win_rate=0.55)

    def test_initial_modifier(self, controller):
        """Test initial difficulty modifier is zero."""
        assert controller.get_difficulty_modifier() == 0.0

    def test_modifier_increases_with_wins(self, controller):
        """Test that difficulty increases with many wins."""
        # Record many wins
        for _ in range(50):
            controller.record_outcome(won=True)

        # Modifier should be positive (increase difficulty)
        assert controller.get_difficulty_modifier() > 0

    def test_modifier_decreases_with_losses(self, controller):
        """Test that difficulty decreases with many losses."""
        # Record many losses
        for _ in range(50):
            controller.record_outcome(won=False)

        # Modifier should be negative (decrease difficulty)
        assert controller.get_difficulty_modifier() < 0

    def test_apply_modifier(self, controller):
        """Test applying modifier to parameters."""
        # Set up some history
        for _ in range(30):
            controller.record_outcome(won=True)

        base_params = {
            "max_turns": 100,
            "production_multiplier": 1.0,
        }

        modified = controller.apply_modifier(base_params)

        # Max turns should increase (harder)
        assert modified["max_turns"] >= base_params["max_turns"]
        # Production should decrease (harder)
        assert modified["production_multiplier"] <= base_params["production_multiplier"]

    def test_reset(self, controller):
        """Test controller reset."""
        for _ in range(20):
            controller.record_outcome(won=True)

        controller.reset()

        assert controller.get_difficulty_modifier() == 0.0


class TestDifficultyDimension:
    """Tests for DifficultyDimension enum."""

    def test_all_dimensions_exist(self):
        """Test all dimensions exist."""
        assert hasattr(DifficultyDimension, "BOARD_SIZE")
        assert hasattr(DifficultyDimension, "MAX_TURNS")
        assert hasattr(DifficultyDimension, "NUM_OPPONENTS")
        assert hasattr(DifficultyDimension, "FOG_OF_WAR")

    def test_dimension_values(self):
        """Test dimension string values."""
        assert DifficultyDimension.BOARD_SIZE.value == "board_size"
        assert DifficultyDimension.MAX_TURNS.value == "max_turns"


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_curriculum_manager_three_stage(self):
        """Test creating three-stage manager."""
        manager = create_curriculum_manager(num_stages=3)

        assert len(manager.config.stages) == 3

    def test_create_curriculum_manager_five_stage(self):
        """Test creating five-stage manager."""
        manager = create_curriculum_manager(num_stages=5)

        assert len(manager.config.stages) == 5

    def test_create_curriculum_manager_with_kwargs(self):
        """Test creating manager with custom kwargs."""
        manager = create_curriculum_manager(
            num_stages=3,
            allow_demotion=False,
            metrics_window=200,
        )

        assert manager.config.allow_demotion is False
        assert manager.config.metrics_window == 200


class TestCurriculumState:
    """Tests for CurriculumState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = CurriculumState()

        assert state.current_stage_index == 0
        assert state.games_at_stage == 0
        assert state.total_games == 0
        assert len(state.stage_history) == 0

    def test_custom_state(self):
        """Test custom state values."""
        state = CurriculumState(
            current_stage_index=2,
            games_at_stage=50,
            total_games=150,
        )

        assert state.current_stage_index == 2
        assert state.games_at_stage == 50
        assert state.total_games == 150
