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
