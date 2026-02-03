"""Unit tests for evaluation configuration."""

import os
import tempfile
from pathlib import Path

import pytest

from alpha_zero.evaluation.config import (
    BaselineConfig,
    BaselineType,
    DiversityConfig,
    EvaluationConfig,
    HeuristicWeights,
)


class TestHeuristicWeights:
    """Tests for HeuristicWeights dataclass."""

    def test_default_values(self):
        """Test default weight values."""
        weights = HeuristicWeights()
        assert weights.attack_weight == 1.0
        assert weights.defense_weight == 0.8
        assert weights.expansion_weight == 1.2
        assert weights.production_weight == 0.5
        assert weights.exploration_weight == 0.3
        assert weights.city_proximity_weight == 0.7

    def test_custom_values(self):
        """Test custom weight values."""
        weights = HeuristicWeights(
            attack_weight=2.0,
            defense_weight=0.5,
        )
        assert weights.attack_weight == 2.0
        assert weights.defense_weight == 0.5

    def test_negative_weight_raises(self):
        """Test that negative weights raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            HeuristicWeights(attack_weight=-1.0)

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"attack_weight": 1.5, "defense_weight": 0.9}
        weights = HeuristicWeights.from_dict(data)
        assert weights.attack_weight == 1.5
        assert weights.defense_weight == 0.9

    def test_to_dict(self):
        """Test converting to dictionary."""
        weights = HeuristicWeights(attack_weight=2.0)
        result = weights.to_dict()
        assert result["attack_weight"] == 2.0
        assert "defense_weight" in result


class TestBaselineConfig:
    """Tests for BaselineConfig dataclass."""

    def test_default_random_baseline(self):
        """Test default random baseline configuration."""
        config = BaselineConfig()
        assert config.type == BaselineType.RANDOM
        assert config.name == "random"
        assert config.num_games == 100
        assert config.enabled is True

    def test_heuristic_baseline(self):
        """Test heuristic baseline configuration."""
        config = BaselineConfig(
            type=BaselineType.HEURISTIC,
            name="aggressive",
            num_games=50,
            heuristic_weights=HeuristicWeights(attack_weight=2.0),
        )
        assert config.type == BaselineType.HEURISTIC
        assert config.heuristic_weights.attack_weight == 2.0

    def test_invalid_num_games_raises(self):
        """Test that num_games < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_games must be at least 1"):
            BaselineConfig(num_games=0)

    def test_checkpoint_required_for_model_baseline(self):
        """Test that checkpoint path is required for model baselines."""
        with pytest.raises(ValueError, match="checkpoint_path required"):
            BaselineConfig(type=BaselineType.SPECIFIC_CHECKPOINT)

    def test_from_dict_with_type_string(self):
        """Test creating from dict with string type."""
        data = {"type": "heuristic", "name": "test", "num_games": 50}
        config = BaselineConfig.from_dict(data)
        assert config.type == BaselineType.HEURISTIC
        assert config.name == "test"

    def test_from_dict_with_heuristic_params(self):
        """Test legacy heuristic_params key support."""
        data = {
            "type": "heuristic",
            "name": "test",
            "heuristic_params": {"attack_weight": 1.5},
        }
        config = BaselineConfig.from_dict(data)
        assert config.heuristic_weights.attack_weight == 1.5

    def test_to_dict(self):
        """Test converting to dictionary."""
        config = BaselineConfig(type=BaselineType.RANDOM, name="test")
        result = config.to_dict()
        assert result["type"] == "random"
        assert result["name"] == "test"


class TestDiversityConfig:
    """Tests for DiversityConfig dataclass."""

    def test_default_values(self):
        """Test default diversity configuration."""
        config = DiversityConfig()
        assert config.enabled is True
        assert config.track_opening_moves is True
        assert config.opening_depth == 10
        assert config.track_game_lengths is True

    def test_disabled_config(self):
        """Test disabled diversity tracking."""
        config = DiversityConfig(enabled=False)
        assert config.enabled is False

    def test_invalid_opening_depth_raises(self):
        """Test that opening_depth < 1 raises ValueError."""
        with pytest.raises(ValueError, match="opening_depth must be at least 1"):
            DiversityConfig(opening_depth=0)

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {"enabled": False, "opening_depth": 5}
        config = DiversityConfig.from_dict(data)
        assert config.enabled is False
        assert config.opening_depth == 5

    def test_to_dict(self):
        """Test converting to dictionary."""
        config = DiversityConfig(opening_depth=15)
        result = config.to_dict()
        assert result["opening_depth"] == 15


class TestEvaluationConfig:
    """Tests for EvaluationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EvaluationConfig()
        assert config.experiment_name == "model_evaluation"
        assert config.elo_enabled is True
        assert config.elo_initial_rating == 1500.0
        assert len(config.baselines) == 2
        assert config.diversity.enabled is True

    def test_custom_baselines(self):
        """Test configuration with custom baselines."""
        baselines = [
            BaselineConfig(type=BaselineType.RANDOM, name="random", num_games=10),
        ]
        config = EvaluationConfig(baselines=baselines)
        assert len(config.baselines) == 1
        assert config.baselines[0].name == "random"

    def test_invalid_elo_rating_raises(self):
        """Test that out-of-range ELO rating raises ValueError."""
        with pytest.raises(ValueError, match="elo_initial_rating must be between"):
            EvaluationConfig(elo_initial_rating=50)

        with pytest.raises(ValueError, match="elo_initial_rating must be between"):
            EvaluationConfig(elo_initial_rating=5000)

    def test_invalid_parallel_games_raises(self):
        """Test that parallel_games < 1 raises ValueError."""
        with pytest.raises(ValueError, match="parallel_games must be at least 1"):
            EvaluationConfig(parallel_games=0)

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "experiment_name": "test_eval",
            "elo_enabled": False,
            "baselines": [
                {"type": "random", "name": "random", "num_games": 5},
            ],
            "diversity": {"enabled": False},
        }
        config = EvaluationConfig.from_dict(data)
        assert config.experiment_name == "test_eval"
        assert config.elo_enabled is False
        assert len(config.baselines) == 1
        assert config.diversity.enabled is False

    def test_from_yaml(self):
        """Test loading from YAML file."""
        yaml_content = """
experiment_name: yaml_test
elo_enabled: true
baselines:
  - type: random
    name: random_test
    num_games: 20
diversity:
  enabled: true
  opening_depth: 5
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml_content)

            config = EvaluationConfig.from_yaml(path)
            assert config.experiment_name == "yaml_test"
            assert config.baselines[0].num_games == 20
            assert config.diversity.opening_depth == 5

    def test_from_preset_quick(self):
        """Test quick preset."""
        config = EvaluationConfig.from_preset("quick")
        assert config.experiment_name == "quick_eval"
        assert len(config.baselines) == 1
        assert config.baselines[0].num_games == 10
        assert config.diversity.enabled is False

    def test_from_preset_standard(self):
        """Test standard preset."""
        config = EvaluationConfig.from_preset("standard")
        assert config.experiment_name == "standard_eval"
        assert len(config.baselines) == 2
        assert config.diversity.enabled is True

    def test_from_preset_comprehensive(self):
        """Test comprehensive preset."""
        config = EvaluationConfig.from_preset("comprehensive")
        assert config.experiment_name == "comprehensive_eval"
        assert len(config.baselines) == 4

    def test_unknown_preset_raises(self):
        """Test that unknown preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            EvaluationConfig.from_preset("invalid")

    def test_to_dict(self):
        """Test converting to dictionary."""
        config = EvaluationConfig(experiment_name="test")
        result = config.to_dict()
        assert result["experiment_name"] == "test"
        assert "baselines" in result
        assert "diversity" in result
        assert "ENV_PREFIX" not in result  # Should be excluded

    def test_save_and_load(self):
        """Test saving and loading configuration."""
        config = EvaluationConfig(
            experiment_name="save_test",
            baselines=[BaselineConfig(type=BaselineType.RANDOM, num_games=5)],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.save(path)

            loaded = EvaluationConfig.from_yaml(path)
            assert loaded.experiment_name == "save_test"
            assert len(loaded.baselines) == 1

    def test_get_enabled_baselines(self):
        """Test filtering enabled baselines."""
        config = EvaluationConfig(
            baselines=[
                BaselineConfig(name="enabled", enabled=True),
                BaselineConfig(name="disabled", enabled=False),
            ]
        )
        enabled = config.get_enabled_baselines()
        assert len(enabled) == 1
        assert enabled[0].name == "enabled"

    def test_validate_warnings(self):
        """Test validation warnings."""
        config = EvaluationConfig(
            baselines=[
                BaselineConfig(name="test", enabled=False),
            ]
        )
        warnings = config.validate()
        assert any("No baselines are enabled" in w for w in warnings)

    def test_from_env(self):
        """Test environment variable overrides."""
        os.environ["CIV6AZ_EVAL_EXPERIMENT_NAME"] = "env_test"
        os.environ["CIV6AZ_EVAL_ELO_ENABLED"] = "false"

        try:
            config = EvaluationConfig.from_env()
            assert config.experiment_name == "env_test"
            assert config.elo_enabled is False
        finally:
            del os.environ["CIV6AZ_EVAL_EXPERIMENT_NAME"]
            del os.environ["CIV6AZ_EVAL_ELO_ENABLED"]
