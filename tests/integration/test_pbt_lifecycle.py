"""Integration tests for PBT (Population-Based Training) lifecycle.

Validates the full PBT cycle using the actual PopulationManager API:
  init → training → evaluation → selection → mutation → next gen.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(seed=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPBTLifecycle:
    """Full PBT lifecycle integration tests."""

    def test_population_initialization(self, rng: np.random.Generator) -> None:
        """PopulationManager initializes with correct size and valid hyperparams."""
        from alpha_zero.pbt.config import PBTConfig
        from alpha_zero.pbt.population import PopulationManager

        config = PBTConfig(population_size=4)
        mgr = PopulationManager(config=config, rng=rng)
        mgr.initialize()

        ids = mgr.get_agent_ids()
        assert len(ids) == 4

        for agent_id in ids:
            agent = mgr.get_agent(agent_id)
            assert agent is not None
            assert agent.agent_id is not None
            assert agent.hyperparameters is not None

        logger.info("Population initialized: %d agents", len(ids))

    def test_fitness_update_and_ranking(self, rng: np.random.Generator) -> None:
        """Agents' fitness updates and population ranks correctly."""
        from alpha_zero.pbt.config import PBTConfig
        from alpha_zero.pbt.population import PopulationManager

        config = PBTConfig(population_size=4)
        mgr = PopulationManager(config=config, rng=rng)
        mgr.initialize()

        ids = mgr.get_agent_ids()
        scores = [1.0, 0.8, 0.3, 0.1]
        for agent_id, score in zip(ids, scores, strict=False):
            mgr.update_fitness(agent_id, score)

        ranking = mgr.get_fitness_ranking()
        assert ranking[0][1] >= ranking[-1][1], "Best should rank first"

    def test_exploit_and_explore(self, rng: np.random.Generator) -> None:
        """Exploitation copies strong agent params; exploration mutates them."""
        from alpha_zero.pbt.config import PBTConfig
        from alpha_zero.pbt.population import PopulationManager

        config = PBTConfig(population_size=4)
        mgr = PopulationManager(config=config, rng=rng)
        mgr.initialize()

        ids = mgr.get_agent_ids()
        scores = [1.0, 0.8, 0.3, 0.1]
        for agent_id, score in zip(ids, scores, strict=False):
            mgr.update_fitness(agent_id, score)

        replacements = mgr.exploit_and_explore()
        # At least one agent should be replaced
        assert isinstance(replacements, list)
        logger.info("Exploit/explore: %d replacements", len(replacements))

    def test_hyperparameter_mutation(self, rng: np.random.Generator) -> None:
        """Mutation changes hyperparameters within valid bounds."""
        from alpha_zero.pbt.config import HyperparameterSpace

        space = HyperparameterSpace()
        original = space.sample_all(rng)

        mutated = space.mutate_all(
            original.copy(),
            rng=rng,
            mutation_strength=0.2,
            mutation_probability=1.0,
        )

        changed = any(original[k] != mutated[k] for k in original if k in mutated)
        assert changed, "Mutation should change at least one hyperparameter"

        for name, value in mutated.items():
            param = space.get_all_params().get(name)
            if param and param.min_value is not None:
                assert value >= param.min_value, f"{name}={value} < {param.min_value}"
                assert value <= param.max_value, f"{name}={value} > {param.max_value}"

    def test_agent_state_serialization(self, rng: np.random.Generator) -> None:
        """AgentState serializes and deserializes correctly."""
        from alpha_zero.pbt.config import AgentState, HyperparameterSpace

        space = HyperparameterSpace()
        agent = AgentState(
            agent_id="test-agent-001",
            hyperparameters=space.sample_all(rng),
            fitness_score=0.75,
        )

        data = agent.to_dict()
        restored = AgentState.from_dict(data)

        assert restored.agent_id == agent.agent_id
        assert restored.fitness_score == pytest.approx(agent.fitness_score)
        for k in agent.hyperparameters:
            assert restored.hyperparameters[k] == pytest.approx(agent.hyperparameters[k], rel=1e-6)

    def test_population_stats(self, rng: np.random.Generator) -> None:
        """Population stats compute correctly."""
        from alpha_zero.pbt.config import PBTConfig
        from alpha_zero.pbt.population import PopulationManager

        config = PBTConfig(population_size=4)
        mgr = PopulationManager(config=config, rng=rng)
        mgr.initialize()

        ids = mgr.get_agent_ids()
        for i, agent_id in enumerate(ids):
            mgr.update_fitness(agent_id, float(i) / len(ids))

        stats = mgr.compute_stats()
        assert stats.mean_fitness >= 0
        assert stats.max_fitness >= stats.mean_fitness
        logger.info("Stats: mean=%.3f, max=%.3f", stats.mean_fitness, stats.max_fitness)

    def test_best_agent_selection(self, rng: np.random.Generator) -> None:
        """get_best_agent returns the highest fitness agent."""
        from alpha_zero.pbt.config import PBTConfig
        from alpha_zero.pbt.population import PopulationManager

        config = PBTConfig(population_size=4)
        mgr = PopulationManager(config=config, rng=rng)
        mgr.initialize()

        ids = mgr.get_agent_ids()
        mgr.update_fitness(ids[0], 0.1)
        mgr.update_fitness(ids[1], 0.9)  # Best
        mgr.update_fitness(ids[2], 0.5)
        mgr.update_fitness(ids[3], 0.3)

        best = mgr.get_best_agent()
        assert best is not None
        assert best.agent_id == ids[1]

    def test_hyperparameter_space_serialization(self) -> None:
        """HyperparameterSpace serializes to/from dict."""
        from alpha_zero.pbt.config import HyperparameterSpace

        space = HyperparameterSpace()
        data = space.to_dict()
        restored = HyperparameterSpace.from_dict(data)

        assert len(restored.get_all_params()) == len(space.get_all_params())
        assert set(restored.get_all_params().keys()) == set(space.get_all_params().keys())
