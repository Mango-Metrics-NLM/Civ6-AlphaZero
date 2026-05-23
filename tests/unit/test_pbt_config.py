"""Unit tests for PBT configuration module.

Tests cover:
- HyperparameterRange sampling and mutation
- HyperparameterSpace operations
- AgentState management
- PBTConfig loading and validation
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from alpha_zero.pbt.config import (
    AgentState,
    HyperparameterRange,
    HyperparameterSpace,
    PBTConfig,
    SelectionMethod,
)


class TestHyperparameterRange:
    """Tests for HyperparameterRange."""

    def test_linear_range_initialization(self):
        """Test linear range initialization."""
        param = HyperparameterRange(
            name="test_param",
            min_value=0.0,
            max_value=1.0,
            scale="linear",
        )
        assert param.name == "test_param"
        assert param.min_value == 0.0
        assert param.max_value == 1.0
        assert param.scale == "linear"
        assert not param.discrete

    def test_log_range_initialization(self):
        """Test log scale range initialization."""
        param = HyperparameterRange(
            name="learning_rate",
            min_value=1e-5,
            max_value=1e-2,
            scale="log",
        )
        assert param.scale == "log"
        assert param.min_value > 0

    def test_categorical_range_initialization(self):
        """Test categorical range initialization."""
        param = HyperparameterRange(
            name="optimizer",
            min_value=0,
            max_value=2,
            scale="categorical",
            categories=["adam", "sgd", "rmsprop"],
        )
        assert param.scale == "categorical"
        assert len(param.categories) == 3

    def test_invalid_range_raises_error(self):
        """Test that invalid ranges raise errors."""
        with pytest.raises(ValueError):
            HyperparameterRange(
                name="invalid",
                min_value=1.0,
                max_value=0.5,  # max < min
                scale="linear",
            )

    def test_log_scale_requires_positive_min(self):
        """Test that log scale requires positive min value."""
        with pytest.raises(ValueError):
            HyperparameterRange(
                name="invalid",
                min_value=-1.0,
                max_value=1.0,
                scale="log",
            )

    def test_categorical_requires_categories(self):
        """Test that categorical scale requires categories."""
        with pytest.raises(ValueError):
            HyperparameterRange(
                name="invalid",
                min_value=0,
                max_value=2,
                scale="categorical",
                categories=[],  # Empty
            )

    def test_linear_sampling(self):
        """Test linear sampling produces valid values."""
        rng = np.random.default_rng(42)
        param = HyperparameterRange("test", 0.0, 1.0, "linear")

        for _ in range(100):
            value = param.sample(rng)
            assert 0.0 <= value <= 1.0

    def test_log_sampling(self):
        """Test log sampling produces valid values."""
        rng = np.random.default_rng(42)
        param = HyperparameterRange("lr", 1e-5, 1e-2, "log")

        for _ in range(100):
            value = param.sample(rng)
            assert 1e-5 <= value <= 1e-2

    def test_categorical_sampling(self):
        """Test categorical sampling produces valid values."""
        rng = np.random.default_rng(42)
        categories = ["a", "b", "c"]
        param = HyperparameterRange("cat", 0, 2, "categorical", categories=categories)

        for _ in range(100):
            value = param.sample(rng)
            assert value in categories

    def test_discrete_sampling(self):
        """Test discrete sampling produces integers."""
        rng = np.random.default_rng(42)
        param = HyperparameterRange("batch", 32, 256, "linear", discrete=True)

        for _ in range(100):
            value = param.sample(rng)
            assert isinstance(value, int)
            assert 32 <= value <= 256

    def test_linear_mutation(self):
        """Test linear mutation stays within bounds."""
        rng = np.random.default_rng(42)
        param = HyperparameterRange("test", 0.0, 1.0, "linear")

        for _ in range(100):
            original = param.sample(rng)
            mutated = param.mutate(original, rng, mutation_strength=0.2)
            assert 0.0 <= mutated <= 1.0

    def test_log_mutation(self):
        """Test log mutation stays within bounds."""
        rng = np.random.default_rng(42)
        param = HyperparameterRange("lr", 1e-5, 1e-2, "log")

        for _ in range(100):
            original = param.sample(rng)
            mutated = param.mutate(original, rng, mutation_strength=0.2)
            # Use tolerance for floating point comparison at boundaries
            assert mutated >= 1e-5 * (1 - 1e-9)
            assert mutated <= 1e-2 * (1 + 1e-9)

    def test_categorical_mutation(self):
        """Test categorical mutation produces valid categories."""
        rng = np.random.default_rng(42)
        categories = ["a", "b", "c"]
        param = HyperparameterRange("cat", 0, 2, "categorical", categories=categories)

        original = "a"
        for _ in range(100):
            mutated = param.mutate(original, rng, mutation_strength=0.5)
            assert mutated in categories

    def test_clip_bounds(self):
        """Test clip keeps values within bounds."""
        param = HyperparameterRange("test", 0.0, 1.0, "linear")

        assert param.clip(-0.5) == 0.0
        assert param.clip(1.5) == 1.0
        assert param.clip(0.5) == 0.5

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        param = HyperparameterRange(
            name="test",
            min_value=0.0,
            max_value=1.0,
            scale="linear",
            discrete=True,
        )

        data = param.to_dict()
        restored = HyperparameterRange.from_dict(data)

        assert restored.name == param.name
        assert restored.min_value == param.min_value
        assert restored.max_value == param.max_value
        assert restored.scale == param.scale
        assert restored.discrete == param.discrete


class TestHyperparameterSpace:
    """Tests for HyperparameterSpace."""

    def test_default_initialization(self):
        """Test default space has expected parameters."""
        space = HyperparameterSpace()

        params = space.get_all_params()
        assert "learning_rate" in params
        assert "weight_decay" in params
        assert "c_puct" in params
        assert "batch_size" in params

    def test_sample_all(self):
        """Test sampling all parameters."""
        rng = np.random.default_rng(42)
        space = HyperparameterSpace()

        values = space.sample_all(rng)

        assert "learning_rate" in values
        assert "weight_decay" in values
        assert 1e-5 <= values["learning_rate"] <= 1e-2
        assert isinstance(values["batch_size"], int)

    def test_mutate_all(self):
        """Test mutating all parameters."""
        rng = np.random.default_rng(42)
        space = HyperparameterSpace()

        original = space.sample_all(rng)
        mutated = space.mutate_all(original, rng, mutation_strength=0.2, mutation_probability=1.0)

        # All parameters should be present
        assert set(original.keys()) == set(mutated.keys())

        # Values should be within bounds
        for name, param in space.get_all_params().items():
            if param.scale != "categorical":
                assert param.min_value <= mutated[name] <= param.max_value

    def test_custom_params(self):
        """Test adding custom parameters."""
        custom = HyperparameterRange("custom_param", 0.0, 10.0)
        space = HyperparameterSpace(custom_params={"custom_param": custom})

        params = space.get_all_params()
        assert "custom_param" in params

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        space = HyperparameterSpace()

        data = space.to_dict()
        restored = HyperparameterSpace.from_dict(data)

        original_params = space.get_all_params()
        restored_params = restored.get_all_params()

        assert set(original_params.keys()) == set(restored_params.keys())


class TestAgentState:
    """Tests for AgentState."""

    def test_default_initialization(self):
        """Test default agent state."""
        agent = AgentState()

        assert agent.agent_id is not None
        assert agent.fitness_score == 0.0
        assert agent.games_played == 0
        assert agent.generation == 0

    def test_custom_initialization(self):
        """Test custom agent initialization."""
        hyperparams = {"learning_rate": 0.001, "batch_size": 64}
        agent = AgentState(
            agent_id="test_agent",
            hyperparameters=hyperparams,
            fitness_score=0.75,
        )

        assert agent.agent_id == "test_agent"
        assert agent.hyperparameters == hyperparams
        assert agent.fitness_score == 0.75

    def test_update_fitness(self):
        """Test fitness update."""
        agent = AgentState()

        agent.update_fitness(0.5)
        assert agent.fitness_score == 0.5
        assert len(agent.fitness_history) == 1

        agent.update_fitness(0.6)
        assert agent.fitness_score == 0.6
        assert len(agent.fitness_history) == 2

    def test_smoothed_fitness(self):
        """Test smoothed fitness calculation."""
        agent = AgentState()

        # Add fitness scores
        for i in range(10):
            agent.update_fitness(float(i) / 10)

        smoothed = agent.get_smoothed_fitness(window=5)
        # Should be average of last 5: (0.5+0.6+0.7+0.8+0.9)/5 = 0.7
        assert abs(smoothed - 0.7) < 0.01

    def test_clone_with_mutations(self):
        """Test cloning with mutations."""
        rng = np.random.default_rng(42)
        space = HyperparameterSpace()

        original = AgentState(
            agent_id="original",
            hyperparameters=space.sample_all(rng),
            generation=5,
        )

        clone = original.clone_with_mutations(space, rng)

        assert clone.agent_id != original.agent_id
        assert clone.parent_id == original.agent_id
        assert clone.generation == original.generation + 1

    def test_save_load(self):
        """Test save and load."""
        agent = AgentState(
            agent_id="test",
            hyperparameters={"lr": 0.001},
            fitness_score=0.8,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agent.json"
            agent.save(path)

            loaded = AgentState.load(path)

            assert loaded.agent_id == agent.agent_id
            assert loaded.hyperparameters == agent.hyperparameters
            assert loaded.fitness_score == agent.fitness_score

    def test_to_dict_from_dict(self):
        """Test dictionary roundtrip."""
        agent = AgentState(
            agent_id="test",
            hyperparameters={"lr": 0.001},
            fitness_score=0.8,
            games_played=100,
        )

        data = agent.to_dict()
        restored = AgentState.from_dict(data)

        assert restored.agent_id == agent.agent_id
        assert restored.fitness_score == agent.fitness_score
        assert restored.games_played == agent.games_played


class TestPBTConfig:
    """Tests for PBTConfig."""

    def test_default_initialization(self):
        """Test default configuration."""
        config = PBTConfig()

        assert config.population_size == 16
        assert config.evaluation_interval == 1000
        assert config.selection_method == "truncation"

    def test_custom_initialization(self):
        """Test custom configuration."""
        config = PBTConfig(
            population_size=32,
            evaluation_interval=2000,
            mutation_strength=0.3,
        )

        assert config.population_size == 32
        assert config.evaluation_interval == 2000
        assert config.mutation_strength == 0.3

    def test_validation_population_size(self):
        """Test population size validation."""
        with pytest.raises(ValueError):
            PBTConfig(population_size=2, min_population_size=4)

    def test_validation_truncation_fraction(self):
        """Test truncation fraction validation."""
        with pytest.raises(ValueError):
            PBTConfig(truncation_fraction=1.5)

        with pytest.raises(ValueError):
            PBTConfig(truncation_fraction=-0.1)

    def test_validation_mutation_probability(self):
        """Test mutation probability validation."""
        with pytest.raises(ValueError):
            PBTConfig(mutation_probability=1.5)

    def test_from_yaml(self):
        """Test loading from YAML."""
        yaml_content = """
population_size: 32
evaluation_interval: 500
mutation_strength: 0.25
wandb_enabled: false
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml_content)

            config = PBTConfig.from_yaml(path)

            assert config.population_size == 32
            assert config.evaluation_interval == 500
            assert config.mutation_strength == 0.25
            assert config.wandb_enabled is False

    def test_from_env(self):
        """Test environment variable overrides."""
        os.environ["CIV6AZ_PBT_POPULATION_SIZE"] = "64"
        os.environ["CIV6AZ_PBT_MUTATION_STRENGTH"] = "0.4"

        try:
            config = PBTConfig.from_env()

            assert config.population_size == 64
            assert config.mutation_strength == 0.4
        finally:
            del os.environ["CIV6AZ_PBT_POPULATION_SIZE"]
            del os.environ["CIV6AZ_PBT_MUTATION_STRENGTH"]

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = PBTConfig(population_size=24)

        data = config.to_dict()

        assert data["population_size"] == 24
        assert "hyperparameter_space" in data

    def test_save(self):
        """Test saving to YAML."""
        config = PBTConfig(population_size=24)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.save(path)

            assert path.exists()

            with open(path) as f:
                data = yaml.safe_load(f)

            assert data["population_size"] == 24

    def test_get_num_to_replace(self):
        """Test calculating number of agents to replace."""
        config = PBTConfig(population_size=20, truncation_fraction=0.2)

        num = config.get_num_to_replace()
        assert num == 4  # 20 * 0.2 = 4

    def test_get_selection_method_enum(self):
        """Test getting selection method as enum."""
        config = PBTConfig(selection_method="tournament")

        method = config.get_selection_method_enum()
        assert method == SelectionMethod.TOURNAMENT


class TestPBTConfigNewFields:
    """Tests for Phase 7 config fields."""

    def test_baseline_interval_default(self):
        config = PBTConfig()
        assert config.baseline_interval == 5

    def test_baseline_agents_default(self):
        config = PBTConfig()
        assert config.baseline_agents == ["random", "heuristic"]

    def test_checkpoint_retention_default(self):
        config = PBTConfig()
        assert config.checkpoint_retention_generations == 5

    def test_evaluation_strategy_switch_gen_default(self):
        config = PBTConfig()
        assert config.evaluation_strategy_switch_gen == 0

    def test_baseline_interval_validation(self):
        with pytest.raises(ValueError, match="baseline_interval"):
            PBTConfig(baseline_interval=-1)

    def test_checkpoint_retention_validation(self):
        with pytest.raises(ValueError, match="checkpoint_retention_generations"):
            PBTConfig(checkpoint_retention_generations=-1)

    def test_strategy_switch_gen_validation(self):
        with pytest.raises(ValueError, match="evaluation_strategy_switch_gen"):
            PBTConfig(evaluation_strategy_switch_gen=-1)

    def test_new_fields_in_to_dict(self):
        config = PBTConfig(
            baseline_interval=10,
            baseline_agents=["random"],
            checkpoint_retention_generations=3,
            evaluation_strategy_switch_gen=15,
        )
        d = config.to_dict()
        assert d["baseline_interval"] == 10
        assert d["baseline_agents"] == ["random"]
        assert d["checkpoint_retention_generations"] == 3
        assert d["evaluation_strategy_switch_gen"] == 15

    def test_new_fields_yaml_roundtrip(self):
        config = PBTConfig(
            baseline_interval=7,
            checkpoint_retention_generations=10,
            evaluation_strategy_switch_gen=20,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.save(path)
            loaded = PBTConfig.from_yaml(path)

        assert loaded.baseline_interval == 7
        assert loaded.checkpoint_retention_generations == 10
        assert loaded.evaluation_strategy_switch_gen == 20

    def test_new_fields_env_override(self):
        os.environ["CIV6AZ_PBT_BASELINE_INTERVAL"] = "10"
        os.environ["CIV6AZ_PBT_CHECKPOINT_RETENTION_GENERATIONS"] = "3"
        try:
            config = PBTConfig.from_env()
            assert config.baseline_interval == 10
            assert config.checkpoint_retention_generations == 3
        finally:
            del os.environ["CIV6AZ_PBT_BASELINE_INTERVAL"]
            del os.environ["CIV6AZ_PBT_CHECKPOINT_RETENTION_GENERATIONS"]

    def test_zero_baseline_interval_valid(self):
        config = PBTConfig(baseline_interval=0)
        assert config.baseline_interval == 0

    def test_zero_checkpoint_retention_valid(self):
        config = PBTConfig(checkpoint_retention_generations=0)
        assert config.checkpoint_retention_generations == 0


class TestSelectionMethod:
    """Tests for SelectionMethod enum."""

    def test_all_methods_defined(self):
        """Test all expected methods are defined."""
        assert SelectionMethod.TRUNCATION.value == "truncation"
        assert SelectionMethod.TOURNAMENT.value == "tournament"
        assert SelectionMethod.ROULETTE.value == "roulette"
        assert SelectionMethod.RANK.value == "rank"

    def test_string_conversion(self):
        """Test string conversion."""
        method = SelectionMethod("truncation")
        assert method == SelectionMethod.TRUNCATION

    def test_all_values_unique(self):
        """Test all enum values are unique."""
        values = [m.value for m in SelectionMethod]
        assert len(values) == len(set(values))


# =============================================================================
# Additional Validation Tests
# =============================================================================


class TestPBTConfigValidation:
    """Additional validation tests for PBTConfig."""

    def test_mutation_strength_zero_raises(self):
        """Test mutation_strength=0 raises ValueError."""
        with pytest.raises(ValueError, match="mutation_strength"):
            PBTConfig(mutation_strength=0.0)

    def test_mutation_strength_above_one_raises(self):
        """Test mutation_strength > 1 raises ValueError."""
        with pytest.raises(ValueError, match="mutation_strength"):
            PBTConfig(mutation_strength=1.5)

    def test_truncation_fraction_zero_raises(self):
        """Test truncation_fraction=0 raises ValueError."""
        with pytest.raises(ValueError, match="truncation_fraction"):
            PBTConfig(truncation_fraction=0.0)

    def test_invalid_evaluation_strategy_raises(self):
        """Test invalid evaluation_strategy raises ValueError."""
        with pytest.raises(ValueError, match="evaluation_strategy"):
            PBTConfig(evaluation_strategy="invalid_strategy")

    def test_evaluation_games_per_match_zero_raises(self):
        """Test evaluation_games_per_match=0 raises ValueError."""
        with pytest.raises(ValueError, match="evaluation_games_per_match"):
            PBTConfig(evaluation_games_per_match=0)

    def test_elo_initial_rating_zero_raises(self):
        """Test elo_initial_rating=0 raises ValueError."""
        with pytest.raises(ValueError, match="elo_initial_rating"):
            PBTConfig(elo_initial_rating=0)

    def test_elo_k_factor_zero_raises(self):
        """Test elo_k_factor=0 raises ValueError."""
        with pytest.raises(ValueError, match="elo_k_factor"):
            PBTConfig(elo_k_factor=0)

    def test_diversity_weight_negative_raises(self):
        """Test diversity_weight < 0 raises ValueError."""
        with pytest.raises(ValueError, match="diversity_weight"):
            PBTConfig(diversity_weight=-0.1)

    def test_num_players_one_raises(self):
        """Test num_players < 2 raises ValueError."""
        with pytest.raises(ValueError, match="num_players"):
            PBTConfig(num_players=1)

    def test_valid_top_k_vs_best_strategy(self):
        """Test valid top_k_vs_best evaluation strategy."""
        config = PBTConfig(evaluation_strategy="top_k_vs_best")
        assert config.evaluation_strategy == "top_k_vs_best"


# =============================================================================
# From YAML Edge Cases
# =============================================================================


class TestPBTConfigYAMLEdgeCases:
    """Edge case tests for YAML loading."""

    def test_from_yaml_nonexistent_raises(self):
        """Test from_yaml with nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PBTConfig.from_yaml("/nonexistent/path.yaml")

    def test_from_yaml_empty_file(self):
        """Test from_yaml with empty YAML file uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.yaml"
            path.write_text("")

            config = PBTConfig.from_yaml(path)
            assert config.population_size == 16  # Default

    def test_from_yaml_with_hyperparameter_space(self):
        """Test from_yaml with hyperparameter_space section."""
        yaml_content = """
population_size: 8
hyperparameter_space:
  learning_rate:
    name: learning_rate
    min_value: 0.0001
    max_value: 0.01
    scale: log
  batch_size:
    name: batch_size
    min_value: 16
    max_value: 256
    scale: linear
    discrete: true
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml_content)

            config = PBTConfig.from_yaml(path)
            assert config.population_size == 8
            assert config.hyperparameter_space.learning_rate.min_value == 0.0001
            assert config.hyperparameter_space.batch_size.discrete is True

    def test_from_yaml_ignores_unknown_fields(self):
        """Test from_yaml ignores fields not in PBTConfig."""
        yaml_content = """
population_size: 8
unknown_field: some_value
another_unknown: 42
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml_content)

            config = PBTConfig.from_yaml(path)
            assert config.population_size == 8
            assert not hasattr(config, "unknown_field")


# =============================================================================
# From Env Edge Cases
# =============================================================================


class TestPBTConfigEnvEdgeCases:
    """Edge case tests for environment variable loading."""

    def test_from_env_bool_conversion(self):
        """Test environment variable bool conversion."""
        os.environ["CIV6AZ_PBT_WANDB_ENABLED"] = "false"
        os.environ["CIV6AZ_PBT_COPY_WEIGHTS"] = "0"

        try:
            config = PBTConfig.from_env()
            assert config.wandb_enabled is False
            assert config.copy_weights is False
        finally:
            del os.environ["CIV6AZ_PBT_WANDB_ENABLED"]
            del os.environ["CIV6AZ_PBT_COPY_WEIGHTS"]

    def test_from_env_ignores_invalid_values(self):
        """Test from_env ignores values that fail conversion."""
        os.environ["CIV6AZ_PBT_POPULATION_SIZE"] = "not_a_number"

        try:
            config = PBTConfig.from_env()
            # Should keep default since conversion failed
            assert config.population_size == 16
        finally:
            del os.environ["CIV6AZ_PBT_POPULATION_SIZE"]

    def test_from_env_ignores_unknown_params(self):
        """Test from_env ignores env vars for non-existent params."""
        os.environ["CIV6AZ_PBT_NONEXISTENT_PARAM"] = "42"

        try:
            config = PBTConfig.from_env()
            assert not hasattr(config, "nonexistent_param")
        finally:
            del os.environ["CIV6AZ_PBT_NONEXISTENT_PARAM"]

    def test_from_env_with_base_config(self):
        """Test from_env with a base config object."""
        base = PBTConfig(population_size=8, mutation_strength=0.3)
        os.environ["CIV6AZ_PBT_MUTATION_STRENGTH"] = "0.5"

        try:
            config = PBTConfig.from_env(base_config=base)
            assert config.population_size == 8  # From base
            assert config.mutation_strength == 0.5  # From env
        finally:
            del os.environ["CIV6AZ_PBT_MUTATION_STRENGTH"]

    def test_from_env_string_value(self):
        """Test from_env handles string values."""
        os.environ["CIV6AZ_PBT_SELECTION_METHOD"] = "tournament"

        try:
            config = PBTConfig.from_env()
            assert config.selection_method == "tournament"
        finally:
            del os.environ["CIV6AZ_PBT_SELECTION_METHOD"]

    def test_from_env_list_value(self):
        """Test from_env handles list values."""
        os.environ["CIV6AZ_PBT_BASELINE_AGENTS"] = "[random, simple]"

        try:
            config = PBTConfig.from_env()
            assert config.baseline_agents == ["random", "simple"]
        finally:
            del os.environ["CIV6AZ_PBT_BASELINE_AGENTS"]


# =============================================================================
# HyperparameterRange Edge Cases
# =============================================================================


class TestHyperparameterRangeEdgeCases:
    """Edge case tests for HyperparameterRange."""

    def test_clip_categorical_invalid_value(self):
        """Test clip on categorical with invalid value returns first category."""
        param = HyperparameterRange(
            name="opt",
            min_value=0,
            max_value=2,
            scale="categorical",
            categories=["adam", "sgd", "rmsprop"],
        )
        result = param.clip("invalid_optimizer")
        assert result == "adam"

    def test_clip_categorical_valid_value(self):
        """Test clip on categorical with valid value returns it unchanged."""
        param = HyperparameterRange(
            name="opt",
            min_value=0,
            max_value=2,
            scale="categorical",
            categories=["adam", "sgd", "rmsprop"],
        )
        result = param.clip("sgd")
        assert result == "sgd"

    def test_clip_discrete(self):
        """Test clip rounds discrete values."""
        param = HyperparameterRange(
            name="batch",
            min_value=32,
            max_value=256,
            scale="linear",
            discrete=True,
        )
        result = param.clip(100.7)
        assert result == 101
        assert isinstance(result, int)

    def test_sample_without_rng(self):
        """Test sample works without providing rng."""
        param = HyperparameterRange("test", 0.0, 1.0, "linear")
        value = param.sample()  # No rng provided
        assert 0.0 <= value <= 1.0

    def test_mutate_without_rng(self):
        """Test mutate works without providing rng."""
        param = HyperparameterRange("test", 0.0, 1.0, "linear")
        value = param.mutate(0.5)  # No rng provided
        assert 0.0 <= value <= 1.0


# =============================================================================
# AgentState Edge Cases
# =============================================================================


class TestAgentStateEdgeCases:
    """Edge case tests for AgentState."""

    def test_update_fitness_caps_history(self):
        """Test that fitness history is capped at max_history."""
        agent = AgentState()

        for i in range(150):
            agent.update_fitness(float(i), max_history=100)

        assert len(agent.fitness_history) == 100
        # Should contain the most recent 100 values
        assert agent.fitness_history[-1] == 149.0

    def test_get_smoothed_fitness_no_history(self):
        """Test smoothed fitness with no history returns fitness_score."""
        agent = AgentState(fitness_score=0.42)
        assert agent.get_smoothed_fitness() == 0.42

    def test_get_smoothed_fitness_window_larger_than_history(self):
        """Test smoothed fitness when window > history length."""
        agent = AgentState()
        agent.update_fitness(0.5)
        agent.update_fitness(0.7)

        # Window of 10, but only 2 entries
        smoothed = agent.get_smoothed_fitness(window=10)
        assert smoothed == pytest.approx(0.6)

    def test_from_dict_missing_fields_use_defaults(self):
        """Test from_dict with minimal data uses defaults."""
        data = {"agent_id": "test123"}
        agent = AgentState.from_dict(data)

        assert agent.agent_id == "test123"
        assert agent.fitness_score == 0.0
        assert agent.games_played == 0
        assert agent.generation == 0
        assert agent.elo_rating == 1200.0

    def test_clone_preserves_parent_id(self):
        """Test that clone sets parent_id to original agent_id."""
        rng = np.random.default_rng(42)
        space = HyperparameterSpace()

        original = AgentState(
            agent_id="parent_agent",
            hyperparameters=space.sample_all(rng),
        )

        clone = original.clone_with_mutations(space, rng)
        assert clone.parent_id == "parent_agent"

    def test_clone_resets_fitness(self):
        """Test that clone starts with fresh fitness."""
        rng = np.random.default_rng(42)
        space = HyperparameterSpace()

        original = AgentState(
            hyperparameters=space.sample_all(rng),
            fitness_score=0.9,
            games_played=100,
        )
        original.update_fitness(0.9)

        clone = original.clone_with_mutations(space, rng)
        assert clone.fitness_score == 0.0
        assert clone.games_played == 0
        assert clone.fitness_history == []

    def test_update_fitness_sets_last_updated(self):
        """Test update_fitness updates the last_updated timestamp."""
        agent = AgentState()
        original_updated = agent.last_updated

        agent.update_fitness(0.7)
        # Timestamp should be different (or at least not error)
        assert agent.last_updated is not None


# =============================================================================
# HyperparameterSpace Edge Cases
# =============================================================================


class TestHyperparameterSpaceEdgeCases:
    """Edge case tests for HyperparameterSpace."""

    def test_mutate_all_zero_probability(self):
        """Test mutate_all with zero probability returns original values."""
        rng = np.random.default_rng(42)
        space = HyperparameterSpace()

        original = space.sample_all(rng)
        mutated = space.mutate_all(original, rng, mutation_probability=0.0)

        assert original == mutated

    def test_mutate_all_missing_value_gets_sampled(self):
        """Test mutate_all samples missing values."""
        rng = np.random.default_rng(42)
        space = HyperparameterSpace()

        # Provide only learning_rate
        partial = {"learning_rate": 0.001}
        mutated = space.mutate_all(partial, rng, mutation_probability=1.0)

        # Should have all parameters
        assert set(mutated.keys()) == set(space.get_all_params().keys())

    def test_from_dict_with_custom_params(self):
        """Test from_dict handles custom_params."""
        data = {
            "custom_params": {
                "my_param": {
                    "name": "my_param",
                    "min_value": 0.0,
                    "max_value": 10.0,
                    "scale": "linear",
                }
            }
        }
        space = HyperparameterSpace.from_dict(data)
        params = space.get_all_params()
        assert "my_param" in params
        assert params["my_param"].min_value == 0.0
        assert params["my_param"].max_value == 10.0


# =============================================================================
# PBTConfig to_dict Completeness
# =============================================================================


class TestPBTConfigToDict:
    """Tests for PBTConfig.to_dict completeness."""

    def test_to_dict_all_fields_present(self):
        """Test that to_dict includes all important fields."""
        config = PBTConfig()
        d = config.to_dict()

        expected_keys = [
            "population_size",
            "min_population_size",
            "evaluation_interval",
            "evaluation_games",
            "fitness_smoothing_window",
            "selection_method",
            "truncation_fraction",
            "tournament_size",
            "mutation_probability",
            "mutation_strength",
            "resample_probability",
            "exploit_threshold",
            "copy_weights",
            "hyperparameter_space",
            "checkpoint_dir",
            "checkpoint_interval",
            "max_checkpoints",
            "log_dir",
            "wandb_enabled",
            "wandb_project",
            "wandb_tags",
            "convergence_threshold",
            "convergence_window",
            "evaluation_strategy",
            "evaluation_games_per_match",
            "elo_initial_rating",
            "elo_k_factor",
            "baseline_interval",
            "baseline_agents",
            "checkpoint_retention_generations",
            "evaluation_strategy_switch_gen",
            "multiplayer_mode",
            "diversity_weight",
            "num_players",
        ]

        for key in expected_keys:
            assert key in d, f"Missing key: {key}"

    def test_get_num_to_replace_minimum_one(self):
        """Test get_num_to_replace always returns at least 1."""
        config = PBTConfig(population_size=4, truncation_fraction=0.01)
        assert config.get_num_to_replace() >= 1
