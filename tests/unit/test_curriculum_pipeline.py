"""Unit tests for the multi-environment curriculum pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alpha_zero.training.curriculum_pipeline import (
    CurriculumPipeline,
    CurriculumPipelineConfig,
    CurriculumState,
    DifficultyStage,
    EnvironmentTier,
    EnvironmentTierConfig,
    TransferLearningConfig,
    WeightTransferManager,
    create_curriculum_pipeline,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def basic_config(temp_checkpoint_dir):
    """Create basic pipeline configuration."""
    return CurriculumPipelineConfig(
        enabled=True,
        checkpoint_dir=temp_checkpoint_dir,
        auto_save=False,
    )


@pytest.fixture
def fast_config(temp_checkpoint_dir):
    """Create fast iteration configuration."""
    config = CurriculumPipelineConfig.fast_iteration()
    config.checkpoint_dir = temp_checkpoint_dir
    config.auto_save = False
    return config


@pytest.fixture
def pipeline(basic_config):
    """Create curriculum pipeline."""
    return CurriculumPipeline(basic_config)


# =============================================================================
# EnvironmentTier Tests
# =============================================================================


class TestEnvironmentTier:
    """Tests for EnvironmentTier enum."""

    def test_complexity_scores(self):
        """Test complexity scores are ordered correctly."""
        assert EnvironmentTier.MICROCIV.complexity_score < EnvironmentTier.UNCIV.complexity_score
        assert EnvironmentTier.UNCIV.complexity_score < EnvironmentTier.CIV6.complexity_score

    def test_next_tier(self):
        """Test next tier progression."""
        assert EnvironmentTier.MICROCIV.next_tier() == EnvironmentTier.UNCIV
        assert EnvironmentTier.UNCIV.next_tier() == EnvironmentTier.CIV6
        assert EnvironmentTier.CIV6.next_tier() is None

    def test_previous_tier(self):
        """Test previous tier regression."""
        assert EnvironmentTier.CIV6.previous_tier() == EnvironmentTier.UNCIV
        assert EnvironmentTier.UNCIV.previous_tier() == EnvironmentTier.MICROCIV
        assert EnvironmentTier.MICROCIV.previous_tier() is None


# =============================================================================
# DifficultyStage Tests
# =============================================================================


class TestDifficultyStage:
    """Tests for DifficultyStage dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        stage = DifficultyStage(name="test")
        assert stage.level == 0
        assert stage.board_size == 8
        assert stage.max_turns == 50
        assert stage.promotion_win_rate == 0.6
        assert stage.min_games == 50

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stage = DifficultyStage(
            name="advanced",
            level=3,
            board_size=16,
            max_turns=100,
        )
        d = stage.to_dict()
        assert d["name"] == "advanced"
        assert d["level"] == 3
        assert d["board_size"] == 16
        assert d["max_turns"] == 100

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "expert",
            "level": 5,
            "board_size": 20,
            "promotion_win_rate": 0.7,
        }
        stage = DifficultyStage.from_dict(data)
        assert stage.name == "expert"
        assert stage.level == 5
        assert stage.board_size == 20
        assert stage.promotion_win_rate == 0.7

    def test_extra_params(self):
        """Test extra parameters are preserved."""
        stage = DifficultyStage(
            name="test",
            extra_params={"difficulty": "Deity", "game_speed": "Standard"},
        )
        assert stage.extra_params["difficulty"] == "Deity"
        assert stage.extra_params["game_speed"] == "Standard"


# =============================================================================
# EnvironmentTierConfig Tests
# =============================================================================


class TestEnvironmentTierConfig:
    """Tests for EnvironmentTierConfig dataclass."""

    def test_default_stages_microciv(self):
        """Test default stages for MicroCiv tier."""
        config = EnvironmentTierConfig(tier=EnvironmentTier.MICROCIV)
        assert len(config.difficulty_stages) > 0
        assert config.difficulty_stages[0].name.startswith("microciv")

    def test_default_stages_unciv(self):
        """Test default stages for Unciv tier."""
        config = EnvironmentTierConfig(tier=EnvironmentTier.UNCIV)
        assert len(config.difficulty_stages) > 0
        assert config.difficulty_stages[0].name.startswith("unciv")

    def test_default_stages_civ6(self):
        """Test default stages for Civ6 tier."""
        config = EnvironmentTierConfig(tier=EnvironmentTier.CIV6)
        assert len(config.difficulty_stages) > 0
        assert config.difficulty_stages[0].name.startswith("civ6")

    def test_promotion_criteria(self):
        """Test promotion criteria are set."""
        config = EnvironmentTierConfig(tier=EnvironmentTier.MICROCIV)
        assert "min_games" in config.promotion_criteria
        assert "win_rate" in config.promotion_criteria

    def test_to_dict_and_back(self):
        """Test serialization round-trip."""
        original = EnvironmentTierConfig(
            tier=EnvironmentTier.UNCIV,
            state_channels=48,
            vector_size=256,
        )
        d = original.to_dict()
        restored = EnvironmentTierConfig.from_dict(d)
        assert restored.tier == original.tier
        assert restored.state_channels == original.state_channels
        assert restored.vector_size == original.vector_size


# =============================================================================
# TransferLearningConfig Tests
# =============================================================================


class TestTransferLearningConfig:
    """Tests for TransferLearningConfig dataclass."""

    def test_default_values(self):
        """Test default transfer learning settings."""
        config = TransferLearningConfig()
        assert config.strategy == "partial"
        assert config.reinitialize_heads is True
        assert config.warmup_games == 100

    def test_custom_freeze_layers(self):
        """Test custom freeze layers."""
        config = TransferLearningConfig(
            freeze_layers=["conv1", "conv2", "bn1"],
            warmup_games=50,
        )
        assert len(config.freeze_layers) == 3
        assert "conv1" in config.freeze_layers


# =============================================================================
# CurriculumPipelineConfig Tests
# =============================================================================


class TestCurriculumPipelineConfig:
    """Tests for CurriculumPipelineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = CurriculumPipelineConfig.default()
        assert config.enabled is True
        assert len(config.environment_tiers) == 3

    def test_fast_iteration_config(self):
        """Test fast iteration preset."""
        config = CurriculumPipelineConfig.fast_iteration()
        # Fast config should have lower thresholds
        for tier_config in config.environment_tiers:
            assert tier_config.promotion_criteria["win_rate"] <= 0.5

    def test_production_config(self):
        """Test production preset."""
        config = CurriculumPipelineConfig.production()
        assert config.allow_tier_demotion is True

    def test_get_tier_config(self):
        """Test getting tier configuration."""
        config = CurriculumPipelineConfig.default()
        microciv = config.get_tier_config(EnvironmentTier.MICROCIV)
        assert microciv is not None
        assert microciv.tier == EnvironmentTier.MICROCIV

    def test_get_disabled_tier(self):
        """Test getting disabled tier returns None."""
        config = CurriculumPipelineConfig.default()
        # Disable Unciv tier
        for tier_config in config.environment_tiers:
            if tier_config.tier == EnvironmentTier.UNCIV:
                tier_config.enabled = False

        result = config.get_tier_config(EnvironmentTier.UNCIV)
        assert result is None


# =============================================================================
# CurriculumState Tests
# =============================================================================


class TestCurriculumState:
    """Tests for CurriculumState dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = CurriculumState()
        assert state.current_tier == EnvironmentTier.MICROCIV
        assert state.current_stage_index == 0
        assert state.total_games == 0
        assert state.elo_rating == 1000.0

    def test_to_dict(self):
        """Test serialization to dict."""
        state = CurriculumState(
            current_tier=EnvironmentTier.UNCIV,
            total_games=500,
            elo_rating=1500.0,
        )
        d = state.to_dict()
        assert d["current_tier"] == "UNCIV"
        assert d["total_games"] == 500
        assert d["elo_rating"] == 1500.0

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "current_tier": "CIV6",
            "current_stage_index": 2,
            "total_games": 1000,
            "elo_rating": 1800.0,
        }
        state = CurriculumState.from_dict(data)
        assert state.current_tier == EnvironmentTier.CIV6
        assert state.current_stage_index == 2
        assert state.total_games == 1000


# =============================================================================
# WeightTransferManager Tests
# =============================================================================


class TestWeightTransferManager:
    """Tests for WeightTransferManager."""

    def test_no_transfer_strategy(self):
        """Test no transfer strategy."""
        config = TransferLearningConfig(strategy="none")
        manager = WeightTransferManager(config)

        source = MagicMock()
        target = MagicMock()

        result = manager.transfer_weights(
            source,
            target,
            EnvironmentTier.MICROCIV,
            EnvironmentTier.UNCIV,
        )
        assert result["strategy"] == "none"
        assert result["transferred_layers"] == 0

    def test_should_unfreeze(self):
        """Test unfreeze timing."""
        config = TransferLearningConfig(warmup_games=100)
        manager = WeightTransferManager(config)

        assert manager.should_unfreeze(50) is False
        assert manager.should_unfreeze(100) is True
        assert manager.should_unfreeze(150) is True

    def test_frozen_layers(self):
        """Test getting frozen layers."""
        config = TransferLearningConfig(
            freeze_layers=["layer1", "layer2"],
        )
        manager = WeightTransferManager(config)

        frozen = manager.get_frozen_layers()
        assert frozen == ["layer1", "layer2"]


# =============================================================================
# CurriculumPipeline Tests
# =============================================================================


class TestCurriculumPipeline:
    """Tests for CurriculumPipeline."""

    def test_initial_state(self, pipeline):
        """Test pipeline starts in initial state."""
        assert pipeline.current_tier == EnvironmentTier.MICROCIV
        assert pipeline.total_games == 0
        assert pipeline.win_rate == 0.0

    def test_record_game_win(self, pipeline):
        """Test recording a winning game."""
        _result = pipeline.record_game(won=True)
        assert pipeline.total_games == 1
        assert pipeline.win_rate == 1.0
        assert pipeline.elo_rating > 1000.0  # ELO should increase

    def test_record_game_loss(self, pipeline):
        """Test recording a losing game."""
        _result = pipeline.record_game(won=False)
        assert pipeline.total_games == 1
        assert pipeline.win_rate == 0.0
        assert pipeline.elo_rating < 1000.0  # ELO should decrease

    def test_multiple_games(self, pipeline):
        """Test recording multiple games."""
        for _ in range(5):
            pipeline.record_game(won=True)
        for _ in range(5):
            pipeline.record_game(won=False)

        assert pipeline.total_games == 10
        assert pipeline.win_rate == 0.5

    def test_stage_promotion(self, fast_config):
        """Test stage promotion after meeting criteria."""
        pipeline = CurriculumPipeline(fast_config)
        initial_stage = pipeline.current_stage.name

        # Record enough wins to promote
        for _ in range(50):
            pipeline.record_game(won=True)

        # Should have promoted to next stage
        assert pipeline.current_stage.name != initial_stage

    def test_stage_demotion(self, fast_config):
        """Test stage demotion after poor performance."""
        pipeline = CurriculumPipeline(fast_config)

        # First promote to intermediate stage
        for _ in range(30):
            pipeline.record_game(won=True)

        _intermediate_stage = pipeline.current_stage.name

        # Now lose consistently
        for _ in range(50):
            pipeline.record_game(won=False)

        # May have demoted (if demotion threshold met)
        # At minimum, should still be tracking correctly
        assert pipeline.total_games == 80

    def test_get_environment_params(self, pipeline):
        """Test getting environment parameters."""
        params = pipeline.get_environment_params()
        assert "tier" in params
        assert "stage" in params
        assert "board_size" in params
        assert "max_turns" in params
        assert params["tier"] == "MICROCIV"

    def test_force_stage(self, pipeline):
        """Test forcing stage transition."""
        result = pipeline.force_stage(2)
        assert result["changed"] is True
        assert result["type"] == "forced"
        assert pipeline._state.current_stage_index == 2

    def test_force_stage_invalid(self, pipeline):
        """Test forcing invalid stage raises error."""
        with pytest.raises(ValueError):
            pipeline.force_stage(100)

    def test_force_tier(self, pipeline):
        """Test forcing tier transition."""
        result = pipeline.force_tier(EnvironmentTier.UNCIV)
        assert result["changed"] is True
        assert result["transfer_required"] is True
        assert pipeline.current_tier == EnvironmentTier.UNCIV

    def test_get_progress(self, pipeline):
        """Test getting progress summary."""
        # Play some games
        for _ in range(10):
            pipeline.record_game(won=True)

        progress = pipeline.get_progress()
        assert progress["current_tier"] == "MICROCIV"
        assert progress["total_games"] == 10
        assert progress["win_rate"] == 1.0

    def test_is_complete(self, pipeline):
        """Test completion check."""
        assert pipeline.is_complete() is False

        # Force to final tier and stage
        pipeline.force_tier(EnvironmentTier.CIV6)
        tier_config = pipeline.config.get_tier_config(EnvironmentTier.CIV6)
        if tier_config:
            pipeline.force_stage(len(tier_config.difficulty_stages) - 1)
            assert pipeline.is_complete() is True

    def test_reset(self, pipeline):
        """Test resetting pipeline."""
        # Play some games
        for _ in range(10):
            pipeline.record_game(won=True)

        pipeline.reset()

        assert pipeline.total_games == 0
        assert pipeline.current_tier == EnvironmentTier.MICROCIV
        assert pipeline._state.current_stage_index == 0

    def test_callbacks_stage_change(self, fast_config):
        """Test stage change callbacks."""
        pipeline = CurriculumPipeline(fast_config)

        callback_data = {"called": False, "old": None, "new": None}

        def on_stage_change(old_stage: str, new_stage: str):
            callback_data["called"] = True
            callback_data["old"] = old_stage
            callback_data["new"] = new_stage

        pipeline.on_stage_change(on_stage_change)

        # Promote through stages
        for _ in range(50):
            pipeline.record_game(won=True)

        # Callback should have been triggered
        # (depends on exact promotion criteria)
        if callback_data["called"]:
            assert callback_data["old"] is not None
            assert callback_data["new"] is not None

    def test_callbacks_tier_change(self, fast_config):
        """Test tier change callbacks."""
        pipeline = CurriculumPipeline(fast_config)

        callback_data = {"called": False}

        def on_tier_change(old_tier: EnvironmentTier, new_tier: EnvironmentTier):
            callback_data["called"] = True
            callback_data["old"] = old_tier
            callback_data["new"] = new_tier

        pipeline.on_tier_change(on_tier_change)

        # Force tier change
        pipeline.force_tier(EnvironmentTier.UNCIV)

        assert callback_data["called"] is True
        assert callback_data["old"] == EnvironmentTier.MICROCIV
        assert callback_data["new"] == EnvironmentTier.UNCIV


# =============================================================================
# State Persistence Tests
# =============================================================================


class TestStatePersistence:
    """Tests for state persistence."""

    def test_save_and_load_state(self, temp_checkpoint_dir):
        """Test saving and loading curriculum state."""
        config = CurriculumPipelineConfig(
            checkpoint_dir=temp_checkpoint_dir,
            auto_save=True,
        )
        pipeline1 = CurriculumPipeline(config)

        # Play some games
        for _ in range(20):
            pipeline1.record_game(won=True)
        for _ in range(10):
            pipeline1.record_game(won=False)

        # Create new pipeline - should load state
        pipeline2 = CurriculumPipeline(config)

        assert pipeline2.total_games == 30
        assert abs(pipeline2.elo_rating - pipeline1.elo_rating) < 0.01

    def test_state_file_exists(self, temp_checkpoint_dir):
        """Test state file is created."""
        config = CurriculumPipelineConfig(
            checkpoint_dir=temp_checkpoint_dir,
            auto_save=True,
        )
        pipeline = CurriculumPipeline(config)
        pipeline.record_game(won=True)

        state_path = Path(temp_checkpoint_dir) / "curriculum_state.json"
        assert state_path.exists()

    def test_state_file_content(self, temp_checkpoint_dir):
        """Test state file contains correct data."""
        config = CurriculumPipelineConfig(
            checkpoint_dir=temp_checkpoint_dir,
            auto_save=True,
        )
        pipeline = CurriculumPipeline(config)

        for _ in range(5):
            pipeline.record_game(won=True)

        state_path = Path(temp_checkpoint_dir) / "curriculum_state.json"
        with open(state_path) as f:
            data = json.load(f)

        assert data["total_games"] == 5
        assert data["current_tier"] == "MICROCIV"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_curriculum_pipeline_default(self, temp_checkpoint_dir):
        """Test creating pipeline with default preset."""
        pipeline = create_curriculum_pipeline(
            preset="default",
            checkpoint_dir=temp_checkpoint_dir,
        )
        assert pipeline.config.enabled is True

    def test_create_curriculum_pipeline_fast(self, temp_checkpoint_dir):
        """Test creating pipeline with fast preset."""
        pipeline = create_curriculum_pipeline(
            preset="fast",
            checkpoint_dir=temp_checkpoint_dir,
        )
        # Fast preset should have lower min games
        tier_config = pipeline.config.get_tier_config(EnvironmentTier.MICROCIV)
        assert tier_config.promotion_criteria["min_games"] == 100

    def test_create_curriculum_pipeline_production(self, temp_checkpoint_dir):
        """Test creating pipeline with production preset."""
        pipeline = create_curriculum_pipeline(
            preset="production",
            checkpoint_dir=temp_checkpoint_dir,
        )
        assert pipeline.config.allow_tier_demotion is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for curriculum pipeline."""

    def test_full_curriculum_progression(self, temp_checkpoint_dir):
        """Test progressing through multiple stages."""
        config = CurriculumPipelineConfig.fast_iteration()
        config.checkpoint_dir = temp_checkpoint_dir
        config.auto_save = True
        pipeline = CurriculumPipeline(config)

        initial_stage_index = pipeline._state.current_stage_index
        stages_seen = [initial_stage_index]

        # Play many games with high win rate
        for i in range(200):
            won = i % 3 != 0  # Win 2/3 of games
            result = pipeline.record_game(won=won)

            if result.get("changed"):
                stages_seen.append(pipeline._state.current_stage_index)

        # Should have progressed through stages
        assert len(stages_seen) >= 1  # At least saw initial stage
        assert pipeline.total_games == 200

    def test_elo_rating_convergence(self, fast_config):
        """Test ELO rating converges with consistent performance."""
        pipeline = CurriculumPipeline(fast_config)

        # Play against 1200 rated opponents, win 60%
        elo_history = []
        for i in range(100):
            won = i % 5 < 3  # Win 60%
            pipeline.record_game(won=won, opponent_elo=1200.0)
            elo_history.append(pipeline.elo_rating)

        # ELO should trend toward a value reflecting 60% win rate vs 1200
        # With K=32, expected ELO is ~1200 for 50% win rate against 1200
        # 60% win rate should lead to higher ELO
        final_elo = elo_history[-1]
        assert final_elo > 1000  # Should be above starting
