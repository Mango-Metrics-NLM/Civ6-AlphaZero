"""Integration tests for Population-Based Training.

Tests cover:
- Full PBT loop with small population
- Fitness evaluation integration
- Selection and mutation integration
- Checkpoint save/restore
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from alpha_zero.pbt.config import (
    AgentState,
    HyperparameterSpace,
    PBTConfig,
)
from alpha_zero.pbt.mutations import create_mutation_operator
from alpha_zero.pbt.population import PopulationManager


class TestPBTIntegration:
    """Integration tests for PBT system."""

    @pytest.fixture
    def config(self):
        """Create small PBT configuration for testing."""
        return PBTConfig(
            population_size=4,
            min_population_size=2,
            evaluation_interval=10,
            evaluation_games=2,
            truncation_fraction=0.25,
            mutation_probability=0.8,
            mutation_strength=0.2,
        )

    @pytest.fixture
    def population_manager(self, config):
        """Create population manager."""
        return PopulationManager(config)

    def test_population_initialization_and_evaluation(self, population_manager):
        """Test population initialization and fitness updates."""
        # Initialize population
        population_manager.initialize()
        assert population_manager.population_size == 4

        # Update fitness scores for all agents
        for i, agent_id in enumerate(population_manager.get_agent_ids()):
            fitness = 0.5 + i * 0.1  # 0.5, 0.6, 0.7, 0.8
            population_manager.update_fitness(agent_id, fitness, games_played=10)

        # Check rankings - get_fitness_ranking returns [(agent_id, fitness), ...]
        ranking = population_manager.get_fitness_ranking()
        assert len(ranking) == 4
        # Best fitness (0.8) should be first in ranking
        best_agent_id, best_fitness = ranking[0]
        assert best_fitness == pytest.approx(0.8, rel=0.01)

    def test_exploit_and_explore_cycle(self, population_manager):
        """Test full exploit/explore cycle."""
        population_manager.initialize()

        # Assign varying fitness
        agents = list(population_manager.get_agent_ids())
        for i, agent_id in enumerate(agents):
            population_manager.update_fitness(
                agent_id, fitness=float(i) / len(agents), games_played=10
            )

        # Run exploit/explore
        changes = population_manager.exploit_and_explore()

        # Should have replaced the worst agent (truncation_fraction=0.25)
        # With 4 agents, 0.25 * 4 = 1 agent replaced
        assert len(changes) == 1

        # Verify the worst agent was replaced
        # Changes format: (old_id, new_id, new_hyperparams)
        old_id, new_id, _ = changes[0]
        assert old_id == agents[0]  # Worst agent
        assert new_id != old_id

    def test_checkpoint_save_restore(self, population_manager, config):
        """Test saving and restoring population state."""
        population_manager.initialize()

        # Update some fitness
        for agent_id in population_manager.get_agent_ids():
            population_manager.update_fitness(agent_id, np.random.random())

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "population.json"

            # Save
            population_manager.save_checkpoint(checkpoint_path)
            assert checkpoint_path.exists()

            # Create new manager and restore
            new_manager = PopulationManager(config)
            new_manager.load_checkpoint(checkpoint_path)

            # Verify state restored
            assert new_manager.population_size == population_manager.population_size

            for agent_id in population_manager.get_agent_ids():
                original = population_manager.get_agent(agent_id)
                restored = new_manager.get_agent(agent_id)
                if original and restored:
                    assert original.fitness_score == restored.fitness_score

    def test_mutation_operator_integration(self):
        """Test mutation operator with hyperparameter space."""
        space = HyperparameterSpace()
        rng = np.random.default_rng(42)
        operator = create_mutation_operator(strategy="gaussian", strength=0.2, rng=rng)

        # Sample initial values
        values = space.sample_all(rng)

        # Mutate all
        mutated = operator.mutate_all(values, space, mutation_probability=1.0)

        # All values should be present and within bounds
        for name, param in space.get_all_params().items():
            assert name in mutated
            if param.scale != "categorical":
                # Use tolerance for floating point comparison
                assert mutated[name] >= param.min_value * (1 - 1e-9)
                assert mutated[name] <= param.max_value * (1 + 1e-9)

    def test_agent_cloning_with_mutations(self):
        """Test agent cloning preserves lineage."""
        rng = np.random.default_rng(42)
        space = HyperparameterSpace()

        parent = AgentState(
            agent_id="parent",
            hyperparameters=space.sample_all(rng),
            fitness_score=0.8,
            generation=3,
        )

        child = parent.clone_with_mutations(space, rng)

        assert child.parent_id == parent.agent_id
        assert child.generation == parent.generation + 1
        assert child.agent_id != parent.agent_id
        assert child.fitness_score == 0.0  # Reset for new agent

    def test_multiple_generations(self, config):
        """Test running multiple generations."""
        manager = PopulationManager(config)
        manager.initialize()

        initial_generation = manager.generation

        # Run several generations
        for _gen in range(5):
            # Update fitness
            for _i, agent_id in enumerate(manager.get_agent_ids()):
                fitness = np.random.uniform(0.3, 0.9)
                manager.update_fitness(agent_id, fitness, games_played=5)

            # Exploit/explore increments generation internally
            manager.exploit_and_explore()

        # Population should have evolved - generation incremented by exploit_and_explore
        assert manager.generation >= initial_generation + 5

    def test_fitness_smoothing(self, config):
        """Test fitness smoothing over window."""
        manager = PopulationManager(config)
        manager.initialize()

        agent_id = list(manager.get_agent_ids())[0]

        # Add multiple fitness values
        for fitness in [0.5, 0.6, 0.7, 0.8, 0.9]:
            manager.update_fitness(agent_id, fitness)

        agent = manager.get_agent(agent_id)
        smoothed = agent.get_smoothed_fitness(window=3)

        # Smoothed should be average of last 3: (0.7 + 0.8 + 0.9) / 3 = 0.8
        assert abs(smoothed - 0.8) < 0.01

    def test_hyperparameter_bounds_after_mutation(self):
        """Test all hyperparameters stay within bounds after mutations."""
        rng = np.random.default_rng(42)
        space = HyperparameterSpace()
        operator = create_mutation_operator(strategy="gaussian", strength=0.5, rng=rng)

        # Run many mutations
        values = space.sample_all(rng)
        for _ in range(100):
            values = operator.mutate_all(values, space, mutation_probability=1.0)

            # Check all values
            for name, param in space.get_all_params().items():
                if param.scale != "categorical":
                    # Allow small tolerance for floating point
                    assert values[name] >= param.min_value * (1 - 1e-8)
                    assert values[name] <= param.max_value * (1 + 1e-8)

    def test_population_get_stats(self, population_manager):
        """Test population statistics computation."""
        population_manager.initialize()

        # Add varied fitness
        for i, agent_id in enumerate(population_manager.get_agent_ids()):
            population_manager.update_fitness(agent_id, fitness=0.5 + i * 0.1, games_played=10 + i)

        # compute_stats() returns a PopulationStats object
        stats = population_manager.compute_stats()

        # Check stats attributes exist and have valid values
        assert stats.mean_fitness > 0
        assert stats.std_fitness >= 0
        assert stats.max_fitness >= stats.mean_fitness
        assert stats.min_fitness <= stats.mean_fitness
        assert stats.num_active == 4
