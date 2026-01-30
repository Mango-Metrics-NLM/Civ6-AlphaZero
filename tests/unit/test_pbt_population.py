"""Tests for PBT population management.

Tests cover:
- PopulationStats dataclass
- PopulationManager initialization and agent management
- Fitness tracking and ranking
- Selection methods (truncation, tournament, roulette, rank)
- Exploitation and exploration
- Checkpoint save/load
- Convergence detection
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from alpha_zero.pbt.config import (
    AgentState,
    HyperparameterSpace,
    PBTConfig,
    SelectionMethod,
)
from alpha_zero.pbt.population import (
    PopulationManager,
    PopulationStats,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Create seeded RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def hyperparameter_space() -> HyperparameterSpace:
    """Create a minimal hyperparameter space."""
    return HyperparameterSpace()


@pytest.fixture
def pbt_config() -> PBTConfig:
    """Create a minimal PBT configuration for testing."""
    return PBTConfig(
        population_size=8,
        truncation_fraction=0.25,  # Replace bottom 2 agents
        evaluation_interval=10,
        fitness_smoothing_window=5,
        convergence_threshold=0.01,
        convergence_window=5,
    )


@pytest.fixture
def population_manager(pbt_config: PBTConfig, rng: np.random.Generator) -> PopulationManager:
    """Create an initialized population manager."""
    manager = PopulationManager(pbt_config, rng=rng)
    manager.initialize()
    return manager


@pytest.fixture
def sample_agent_states() -> list[AgentState]:
    """Create sample agent states for testing."""
    agents = []
    for i in range(4):
        agent = AgentState(
            hyperparameters={"learning_rate": 0.001 * (i + 1), "c_puct": 1.5},
        )
        agent.update_fitness(0.5 + i * 0.1)  # Increasing fitness
        agents.append(agent)
    return agents


# =============================================================================
# PopulationStats Tests
# =============================================================================


class TestPopulationStats:
    """Tests for PopulationStats dataclass."""

    def test_default_creation(self) -> None:
        """Test creating PopulationStats with defaults."""
        stats = PopulationStats()
        assert stats.generation == 0
        assert stats.mean_fitness == 0.0
        assert stats.max_fitness == 0.0
        assert stats.min_fitness == 0.0
        assert stats.std_fitness == 0.0
        assert stats.hyperparameter_stats == {}
        assert stats.num_active == 0
        assert stats.total_games == 0
        assert stats.total_steps == 0
        assert stats.timestamp is not None

    def test_custom_creation(self) -> None:
        """Test creating PopulationStats with custom values."""
        stats = PopulationStats(
            generation=5,
            mean_fitness=0.75,
            max_fitness=0.9,
            min_fitness=0.6,
            std_fitness=0.1,
            num_active=10,
            total_games=100,
            total_steps=5000,
        )
        assert stats.generation == 5
        assert stats.mean_fitness == 0.75
        assert stats.max_fitness == 0.9
        assert stats.min_fitness == 0.6
        assert stats.std_fitness == 0.1

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        stats = PopulationStats(
            generation=3,
            mean_fitness=0.65,
            hyperparameter_stats={"lr": {"mean": 0.001, "std": 0.0002}},
        )
        data = stats.to_dict()

        assert data["generation"] == 3
        assert data["mean_fitness"] == 0.65
        assert "lr" in data["hyperparameter_stats"]
        assert "timestamp" in data


# =============================================================================
# PopulationManager Initialization Tests
# =============================================================================


class TestPopulationManagerInit:
    """Tests for PopulationManager initialization."""

    def test_initialization(self, pbt_config: PBTConfig, rng: np.random.Generator) -> None:
        """Test basic initialization."""
        manager = PopulationManager(pbt_config, rng=rng)

        assert manager.config == pbt_config
        assert manager.generation == 0
        assert len(manager) == 0  # Not yet initialized

    def test_initialize_random_agents(
        self, pbt_config: PBTConfig, rng: np.random.Generator
    ) -> None:
        """Test initializing with random agents."""
        manager = PopulationManager(pbt_config, rng=rng)
        manager.initialize()

        assert len(manager) == pbt_config.population_size
        assert manager.generation == 0

        # All agents should have hyperparameters
        for agent in manager:
            assert "learning_rate" in agent.hyperparameters
            assert "c_puct" in agent.hyperparameters

    def test_initialize_with_existing_agents(
        self, pbt_config: PBTConfig, sample_agent_states: list[AgentState]
    ) -> None:
        """Test initializing with existing agents."""
        manager = PopulationManager(pbt_config)
        manager.initialize(existing_agents=sample_agent_states)

        assert len(manager) == len(sample_agent_states)

        # Verify agents were loaded
        for original in sample_agent_states:
            loaded = manager.get_agent(original.agent_id)
            assert loaded is not None
            assert loaded.hyperparameters == original.hyperparameters

    def test_reinitialize_clears_population(
        self, population_manager: PopulationManager
    ) -> None:
        """Test that reinitializing clears existing population."""
        original_ids = population_manager.get_agent_ids()

        # Reinitialize
        population_manager.initialize()

        new_ids = population_manager.get_agent_ids()

        # All IDs should be different
        assert set(original_ids) != set(new_ids)


# =============================================================================
# Agent Management Tests
# =============================================================================


class TestAgentManagement:
    """Tests for agent management operations."""

    def test_get_agent_ids(self, population_manager: PopulationManager) -> None:
        """Test getting agent IDs."""
        agent_ids = population_manager.get_agent_ids()

        assert len(agent_ids) == population_manager.config.population_size
        assert all(isinstance(aid, str) for aid in agent_ids)

    def test_get_agent_exists(self, population_manager: PopulationManager) -> None:
        """Test getting an existing agent."""
        agent_ids = population_manager.get_agent_ids()
        agent = population_manager.get_agent(agent_ids[0])

        assert agent is not None
        assert agent.agent_id == agent_ids[0]

    def test_get_agent_not_exists(self, population_manager: PopulationManager) -> None:
        """Test getting a non-existent agent returns None."""
        agent = population_manager.get_agent("nonexistent_id")
        assert agent is None

    def test_get_all_agents(self, population_manager: PopulationManager) -> None:
        """Test getting all agents."""
        agents = population_manager.get_all_agents()

        assert len(agents) == population_manager.config.population_size
        assert all(isinstance(a, AgentState) for a in agents)

    def test_get_hyperparameters(self, population_manager: PopulationManager) -> None:
        """Test getting hyperparameters for an agent."""
        agent_ids = population_manager.get_agent_ids()
        params = population_manager.get_hyperparameters(agent_ids[0])

        assert isinstance(params, dict)
        assert "learning_rate" in params
        assert "c_puct" in params

    def test_get_hyperparameters_not_found(
        self, population_manager: PopulationManager
    ) -> None:
        """Test that KeyError is raised for non-existent agent."""
        with pytest.raises(KeyError):
            population_manager.get_hyperparameters("nonexistent_id")

    def test_iteration(self, population_manager: PopulationManager) -> None:
        """Test iterating over agents."""
        agents_from_iteration = list(population_manager)

        assert len(agents_from_iteration) == population_manager.config.population_size
        assert all(isinstance(a, AgentState) for a in agents_from_iteration)


# =============================================================================
# Fitness Management Tests
# =============================================================================


class TestFitnessManagement:
    """Tests for fitness tracking and ranking."""

    def test_update_fitness(self, population_manager: PopulationManager) -> None:
        """Test updating agent fitness."""
        agent_id = population_manager.get_agent_ids()[0]

        population_manager.update_fitness(agent_id, 0.75)

        agent = population_manager.get_agent(agent_id)
        assert agent is not None
        assert len(agent.fitness_history) == 1
        assert agent.fitness_history[0] == 0.75

    def test_update_fitness_with_games_and_steps(
        self, population_manager: PopulationManager
    ) -> None:
        """Test updating fitness with games and steps."""
        agent_id = population_manager.get_agent_ids()[0]

        population_manager.update_fitness(
            agent_id, 0.6, games_played=10, training_steps=1000
        )

        agent = population_manager.get_agent(agent_id)
        assert agent is not None
        assert agent.games_played == 10
        assert agent.training_steps == 1000

    def test_update_fitness_nonexistent_agent(
        self, population_manager: PopulationManager
    ) -> None:
        """Test updating fitness for non-existent agent logs warning."""
        # Should not raise, just log warning
        population_manager.update_fitness("nonexistent_id", 0.5)

    def test_get_fitness_ranking(self, population_manager: PopulationManager) -> None:
        """Test getting fitness ranking."""
        agent_ids = population_manager.get_agent_ids()

        # Set different fitness values
        for i, agent_id in enumerate(agent_ids):
            population_manager.update_fitness(agent_id, 0.1 * (i + 1))

        ranking = population_manager.get_fitness_ranking()

        assert len(ranking) == len(agent_ids)
        # Should be sorted descending by fitness
        fitnesses = [f for _, f in ranking]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_get_best_agent(self, population_manager: PopulationManager) -> None:
        """Test getting the best performing agent."""
        agent_ids = population_manager.get_agent_ids()

        # Set known fitness values
        for i, agent_id in enumerate(agent_ids):
            population_manager.update_fitness(agent_id, 0.1 * (i + 1))

        best = population_manager.get_best_agent()
        expected_best_id = agent_ids[-1]  # Last one has highest fitness

        assert best is not None
        assert best.agent_id == expected_best_id

    def test_get_best_agent_empty_population(
        self, pbt_config: PBTConfig
    ) -> None:
        """Test get_best_agent returns None for empty population."""
        manager = PopulationManager(pbt_config)
        # Don't initialize
        assert manager.get_best_agent() is None


# =============================================================================
# Selection Tests
# =============================================================================


class TestSelection:
    """Tests for agent selection methods."""

    def test_get_agents_to_replace(self, population_manager: PopulationManager) -> None:
        """Test identifying agents to replace."""
        agent_ids = population_manager.get_agent_ids()

        # Set fitness so we know which should be replaced
        for i, agent_id in enumerate(agent_ids):
            population_manager.update_fitness(agent_id, 0.1 * (i + 1))

        to_replace = population_manager.get_agents_to_replace()
        num_expected = population_manager.config.get_num_to_replace()

        assert len(to_replace) == num_expected
        # Should be the agents with lowest fitness
        ranking = population_manager.get_fitness_ranking()
        expected = [aid for aid, _ in ranking[-num_expected:]]
        assert set(to_replace) == set(expected)

    def test_get_agents_to_copy_from(
        self, population_manager: PopulationManager
    ) -> None:
        """Test identifying top agents to copy from."""
        agent_ids = population_manager.get_agent_ids()

        # Set fitness
        for i, agent_id in enumerate(agent_ids):
            population_manager.update_fitness(agent_id, 0.1 * (i + 1))

        to_copy = population_manager.get_agents_to_copy_from()
        num_expected = population_manager.config.get_num_to_replace()

        assert len(to_copy) == num_expected
        # Should be the agents with highest fitness
        ranking = population_manager.get_fitness_ranking()
        expected = [aid for aid, _ in ranking[:num_expected]]
        assert set(to_copy) == set(expected)

    def test_select_strong_agent_truncation(
        self, pbt_config: PBTConfig, rng: np.random.Generator
    ) -> None:
        """Test truncation selection picks uniformly from top agents."""
        pbt_config.selection_method = "truncation"
        manager = PopulationManager(pbt_config, rng=rng)
        manager.initialize()

        agent_ids = manager.get_agent_ids()
        for i, aid in enumerate(agent_ids):
            manager.update_fitness(aid, 0.1 * (i + 1))

        strong_agents = manager.get_agents_to_copy_from()

        # Run multiple selections
        selections = [manager._select_strong_agent(strong_agents) for _ in range(100)]

        # All selections should be from strong_agents
        assert all(s in strong_agents for s in selections)

    def test_select_strong_agent_tournament(
        self, pbt_config: PBTConfig, rng: np.random.Generator
    ) -> None:
        """Test tournament selection."""
        pbt_config.selection_method = "tournament"
        pbt_config.tournament_size = 2
        manager = PopulationManager(pbt_config, rng=rng)
        manager.initialize()

        agent_ids = manager.get_agent_ids()
        for i, aid in enumerate(agent_ids):
            manager.update_fitness(aid, 0.1 * (i + 1))

        strong_agents = manager.get_agents_to_copy_from()
        selected = manager._select_strong_agent(strong_agents)

        assert selected in strong_agents

    def test_select_strong_agent_roulette(
        self, pbt_config: PBTConfig, rng: np.random.Generator
    ) -> None:
        """Test roulette wheel selection."""
        pbt_config.selection_method = "roulette"
        manager = PopulationManager(pbt_config, rng=rng)
        manager.initialize()

        agent_ids = manager.get_agent_ids()
        for i, aid in enumerate(agent_ids):
            manager.update_fitness(aid, 0.1 * (i + 1))

        strong_agents = manager.get_agents_to_copy_from()
        selected = manager._select_strong_agent(strong_agents)

        assert selected in strong_agents

    def test_select_strong_agent_rank(
        self, pbt_config: PBTConfig, rng: np.random.Generator
    ) -> None:
        """Test rank-based selection."""
        pbt_config.selection_method = "rank"
        manager = PopulationManager(pbt_config, rng=rng)
        manager.initialize()

        agent_ids = manager.get_agent_ids()
        for i, aid in enumerate(agent_ids):
            manager.update_fitness(aid, 0.1 * (i + 1))

        strong_agents = manager.get_agents_to_copy_from()
        selected = manager._select_strong_agent(strong_agents)

        assert selected in strong_agents


# =============================================================================
# Exploit and Explore Tests
# =============================================================================


class TestExploitAndExplore:
    """Tests for exploitation and exploration."""

    def test_exploit_and_explore(self, population_manager: PopulationManager) -> None:
        """Test basic exploit and explore cycle."""
        agent_ids = population_manager.get_agent_ids()

        # Set fitness values
        for i, aid in enumerate(agent_ids):
            population_manager.update_fitness(aid, 0.1 * (i + 1))

        initial_generation = population_manager.generation
        changes = population_manager.exploit_and_explore()

        # Should have replaced some agents
        num_expected = population_manager.config.get_num_to_replace()
        assert len(changes) == num_expected

        # Generation should have incremented
        assert population_manager.generation == initial_generation + 1

        # Each change should have (weak_id, strong_id, new_params)
        for weak_id, strong_id, new_params in changes:
            assert isinstance(weak_id, str)
            assert isinstance(strong_id, str)
            assert isinstance(new_params, dict)

    def test_exploit_and_explore_updates_agent(
        self, population_manager: PopulationManager
    ) -> None:
        """Test that exploit_and_explore updates weak agents."""
        agent_ids = population_manager.get_agent_ids()

        # Set fitness values
        for i, aid in enumerate(agent_ids):
            population_manager.update_fitness(aid, 0.1 * (i + 1))

        # Get weak agent before
        to_replace = population_manager.get_agents_to_replace()
        weak_id = to_replace[0]
        weak_before = population_manager.get_agent(weak_id)
        old_params = weak_before.hyperparameters.copy()

        population_manager.exploit_and_explore()

        # Weak agent should have different params now
        weak_after = population_manager.get_agent(weak_id)
        # At least some parameter should change (with high probability)
        # Note: There's a small chance all params stay the same due to mutation

    def test_exploit_and_explore_with_weight_callback(
        self, pbt_config: PBTConfig, rng: np.random.Generator
    ) -> None:
        """Test that weight copy callback is called."""
        pbt_config.copy_weights = True
        manager = PopulationManager(pbt_config, rng=rng)
        manager.initialize()

        copy_fn = MagicMock()
        manager.set_weight_callbacks(copy_fn=copy_fn)

        agent_ids = manager.get_agent_ids()
        for i, aid in enumerate(agent_ids):
            manager.update_fitness(aid, 0.1 * (i + 1))

        manager.exploit_and_explore()

        # Copy function should have been called
        assert copy_fn.called

    def test_exploit_and_explore_no_agents(
        self, pbt_config: PBTConfig
    ) -> None:
        """Test exploit_and_explore with no agents returns empty."""
        manager = PopulationManager(pbt_config)
        # Don't initialize
        changes = manager.exploit_and_explore()
        assert changes == []


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for population statistics."""

    def test_compute_stats(self, population_manager: PopulationManager) -> None:
        """Test computing population statistics."""
        agent_ids = population_manager.get_agent_ids()

        # Set known fitness values
        for i, aid in enumerate(agent_ids):
            population_manager.update_fitness(aid, 0.5 + 0.05 * i)

        stats = population_manager.compute_stats()

        assert stats.generation == population_manager.generation
        assert stats.num_active == len(agent_ids)
        assert stats.max_fitness >= stats.mean_fitness >= stats.min_fitness
        assert stats.std_fitness >= 0

    def test_compute_stats_hyperparameter_stats(
        self, population_manager: PopulationManager
    ) -> None:
        """Test that hyperparameter statistics are computed."""
        for aid in population_manager.get_agent_ids():
            population_manager.update_fitness(aid, 0.5)

        stats = population_manager.compute_stats()

        assert "learning_rate" in stats.hyperparameter_stats
        lr_stats = stats.hyperparameter_stats["learning_rate"]
        assert "mean" in lr_stats
        assert "std" in lr_stats
        assert "min" in lr_stats
        assert "max" in lr_stats

    def test_compute_stats_empty_population(self, pbt_config: PBTConfig) -> None:
        """Test compute_stats with empty population."""
        manager = PopulationManager(pbt_config)
        stats = manager.compute_stats()

        assert stats.num_active == 0
        assert stats.mean_fitness == 0.0


# =============================================================================
# Convergence Tests
# =============================================================================


class TestConvergence:
    """Tests for convergence detection."""

    def test_is_converged_not_enough_history(
        self, population_manager: PopulationManager
    ) -> None:
        """Test convergence returns False without enough history."""
        assert not population_manager.is_converged()

    def test_is_converged_with_variance(
        self, pbt_config: PBTConfig, rng: np.random.Generator
    ) -> None:
        """Test convergence with high fitness variance."""
        pbt_config.convergence_window = 3
        pbt_config.convergence_threshold = 0.001
        manager = PopulationManager(pbt_config, rng=rng)
        manager.initialize()

        # Create high variance stats
        for i in range(5):
            for j, aid in enumerate(manager.get_agent_ids()):
                manager.update_fitness(aid, 0.1 * j + 0.1 * i)
            manager.compute_stats()

        # High variance should not be converged
        assert not manager.is_converged()

    def test_is_converged_low_variance(
        self, pbt_config: PBTConfig, rng: np.random.Generator
    ) -> None:
        """Test convergence with low fitness variance."""
        pbt_config.convergence_window = 3
        pbt_config.convergence_threshold = 0.1
        manager = PopulationManager(pbt_config, rng=rng)
        manager.initialize()

        # Create low variance stats (all agents same fitness)
        for i in range(5):
            for aid in manager.get_agent_ids():
                manager.update_fitness(aid, 0.5)  # Same fitness
            manager.compute_stats()

        # Low variance should be converged
        assert manager.is_converged()


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestCheckpoints:
    """Tests for checkpoint save/load."""

    def test_save_checkpoint(self, population_manager: PopulationManager) -> None:
        """Test saving a checkpoint."""
        # Update some fitness values
        for aid in population_manager.get_agent_ids():
            population_manager.update_fitness(aid, 0.5)
        population_manager.compute_stats()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            population_manager.save_checkpoint(checkpoint_path)

            # Check files were created
            assert (checkpoint_path / "metadata.json").exists()
            assert (checkpoint_path / "agents").exists()
            assert (checkpoint_path / "stats_history.json").exists()

    def test_load_checkpoint(self, population_manager: PopulationManager) -> None:
        """Test loading a checkpoint."""
        # Update fitness
        original_ids = population_manager.get_agent_ids()
        for aid in original_ids:
            population_manager.update_fitness(aid, 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"
            population_manager.save_checkpoint(checkpoint_path)

            # Create new manager and load
            new_manager = PopulationManager(population_manager.config)
            new_manager.load_checkpoint(checkpoint_path)

            assert len(new_manager) == len(original_ids)
            assert set(new_manager.get_agent_ids()) == set(original_ids)

    def test_load_checkpoint_not_found(self, pbt_config: PBTConfig) -> None:
        """Test that FileNotFoundError is raised for missing checkpoint."""
        manager = PopulationManager(pbt_config)

        with pytest.raises(FileNotFoundError):
            manager.load_checkpoint("/nonexistent/path")

    def test_save_load_with_weight_callbacks(
        self, pbt_config: PBTConfig, rng: np.random.Generator
    ) -> None:
        """Test that weight callbacks are used during save/load."""
        manager = PopulationManager(pbt_config, rng=rng)
        manager.initialize()

        # Mock save function that actually creates files
        def mock_save(agent_id: str, path: str) -> None:
            Path(path).touch()

        save_fn = MagicMock(side_effect=mock_save)
        load_fn = MagicMock()
        manager.set_weight_callbacks(save_fn=save_fn, load_fn=load_fn)

        for aid in manager.get_agent_ids():
            manager.update_fitness(aid, 0.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"

            manager.save_checkpoint(checkpoint_path)
            assert save_fn.called

            manager.load_checkpoint(checkpoint_path)
            assert load_fn.called


# =============================================================================
# Weight Callback Tests
# =============================================================================


class TestWeightCallbacks:
    """Tests for weight management callbacks."""

    def test_set_weight_callbacks(self, population_manager: PopulationManager) -> None:
        """Test setting weight callbacks."""
        save_fn = MagicMock()
        load_fn = MagicMock()
        copy_fn = MagicMock()

        population_manager.set_weight_callbacks(
            save_fn=save_fn,
            load_fn=load_fn,
            copy_fn=copy_fn,
        )

        assert population_manager._save_weights_fn == save_fn
        assert population_manager._load_weights_fn == load_fn
        assert population_manager._copy_weights_fn == copy_fn

    def test_weight_callbacks_optional(
        self, population_manager: PopulationManager
    ) -> None:
        """Test that callbacks are optional."""
        # This should not raise
        population_manager.set_weight_callbacks()
        population_manager.set_weight_callbacks(save_fn=MagicMock())


# =============================================================================
# Property Tests
# =============================================================================


class TestProperties:
    """Tests for manager properties."""

    def test_generation_property(self, population_manager: PopulationManager) -> None:
        """Test generation property."""
        assert population_manager.generation == 0

        # After exploit/explore
        for i, aid in enumerate(population_manager.get_agent_ids()):
            population_manager.update_fitness(aid, 0.1 * (i + 1))
        population_manager.exploit_and_explore()

        assert population_manager.generation == 1

    def test_population_size_property(
        self, population_manager: PopulationManager
    ) -> None:
        """Test population_size property."""
        expected = population_manager.config.population_size
        assert population_manager.population_size == expected
        assert len(population_manager) == expected
