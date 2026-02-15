"""Unit tests for the strategic factory functions.

Tests component creation factories including:
- create_environment for registry-backed environment creation
- create_scalarizer for multi-objective scalarizers
- create_strategic_planner for the main planner
- create_strategic_network for neural networks
- create_multi_objective_mcts for MCTS instances
- create_scenario_engine for scenario management
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.strategic.factory import (
    create_environment,
    create_multi_objective_mcts,
    create_scenario_engine,
    create_scalarizer,
    create_strategic_network,
    create_strategic_planner,
)


# ---------------------------------------------------------------------------
# TestCreateEnvironment
# ---------------------------------------------------------------------------


class TestCreateEnvironment:
    """Tests for create_environment factory."""

    def test_create_resource_allocation(self) -> None:
        """Test creating a resource allocation environment via factory."""
        env = create_environment("resource_allocation")
        assert env is not None
        assert hasattr(env, "num_objectives")
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

    def test_create_unknown_raises(self) -> None:
        """Test that creating an unknown environment raises KeyError."""
        with pytest.raises(KeyError, match="not registered"):
            create_environment("totally_fake_env")


# ---------------------------------------------------------------------------
# TestCreateScalarizer
# ---------------------------------------------------------------------------


class TestCreateScalarizer:
    """Tests for create_scalarizer factory."""

    def test_weighted_sum(self) -> None:
        """Test creating a weighted sum scalarizer."""
        from alpha_zero.strategic.objectives import (
            ObjectiveSpec,
            WeightedSumScalarizer,
        )

        objectives = [
            ObjectiveSpec(name="efficiency", weight=0.6),
            ObjectiveSpec(name="equity", weight=0.4),
        ]
        scalarizer = create_scalarizer("weighted_sum", objectives)

        assert isinstance(scalarizer, WeightedSumScalarizer)
        assert scalarizer.num_objectives == 2

        # Verify it produces a scalar output
        result = scalarizer.scalarize(np.array([0.8, 0.6]))
        assert isinstance(result, float)

    def test_chebyshev(self) -> None:
        """Test creating a Chebyshev scalarizer."""
        from alpha_zero.strategic.objectives import (
            ChebyshevScalarizer,
            ObjectiveSpec,
        )

        objectives = [
            ObjectiveSpec(name="a", weight=0.5),
            ObjectiveSpec(name="b", weight=0.5),
        ]
        ideal = [1.0, 1.0]
        scalarizer = create_scalarizer("chebyshev", objectives, ideal_point=ideal)

        assert isinstance(scalarizer, ChebyshevScalarizer)
        np.testing.assert_allclose(scalarizer.ideal_point, ideal)

        result = scalarizer.scalarize(np.array([0.8, 0.6]))
        assert isinstance(result, float)

    def test_epsilon_constraint(self) -> None:
        """Test creating an epsilon-constraint scalarizer."""
        from alpha_zero.strategic.objectives import (
            EpsilonConstraintScalarizer,
            ObjectiveSpec,
        )

        objectives = [
            ObjectiveSpec(name="primary", weight=1.0),
            ObjectiveSpec(name="secondary", weight=0.5, constraint_value=0.3),
        ]
        scalarizer = create_scalarizer(
            "epsilon_constraint",
            objectives,
            primary_index=0,
            constraint_penalty=5.0,
        )

        assert isinstance(scalarizer, EpsilonConstraintScalarizer)
        assert scalarizer.primary_index == 0
        assert scalarizer.constraint_penalty == pytest.approx(5.0)

        result = scalarizer.scalarize(np.array([0.8, 0.5]))
        assert isinstance(result, float)

    def test_unknown_raises(self) -> None:
        """Test that an unknown scalarizer method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scalarization method"):
            create_scalarizer("nonexistent_method")


# ---------------------------------------------------------------------------
# TestCreateStrategicPlanner
# ---------------------------------------------------------------------------


class TestCreateStrategicPlanner:
    """Tests for create_strategic_planner factory."""

    def test_create_default(self) -> None:
        """Test creating a planner with default configuration."""
        from alpha_zero.strategic.planner import StrategicPlanner

        planner = create_strategic_planner()

        assert isinstance(planner, StrategicPlanner)


# ---------------------------------------------------------------------------
# TestCreateStrategicNetwork
# ---------------------------------------------------------------------------


class TestCreateStrategicNetwork:
    """Tests for create_strategic_network factory."""

    def test_create_default(self) -> None:
        """Test creating a network with default configuration."""
        import torch
        from alpha_zero.strategic.network import StrategicNetwork

        network = create_strategic_network()

        assert isinstance(network, StrategicNetwork)
        # Default uses vector-only (no spatial)
        assert network.spatial_channels == 0

        # Verify forward pass works
        vector = torch.randn(1, 64)
        policy, value = network(None, vector)
        assert policy.shape == (1, 100)  # default action_space_size
        assert value.shape == (1,)  # single objective squeezed

    def test_create_with_config(self) -> None:
        """Test creating a network with explicit configuration."""
        import torch
        from alpha_zero.strategic.config import StrategicNetworkConfig
        from alpha_zero.strategic.network import StrategicNetwork

        config = StrategicNetworkConfig(
            spatial_channels=0,
            vector_dim=32,
            num_objectives=3,
            num_res_blocks=2,
            num_filters=32,
            value_head_hidden=64,
            policy_head_hidden=64,
        )

        network = create_strategic_network(
            config=config,
            action_space_size=50,
            device="cpu",
        )

        assert isinstance(network, StrategicNetwork)
        assert network.num_objectives == 3

        # Verify forward pass
        vector = torch.randn(2, 32)
        policy, value = network(None, vector)
        assert policy.shape == (2, 50)
        assert value.shape == (2, 3)  # multi-objective


# ---------------------------------------------------------------------------
# TestCreateMultiObjectiveMCTS
# ---------------------------------------------------------------------------


class TestCreateMultiObjectiveMCTS:
    """Tests for create_multi_objective_mcts factory."""

    def test_create_default(self) -> None:
        """Test creating MCTS with default configuration."""
        from alpha_zero.strategic.mcts.multi_objective_mcts import (
            MultiObjectiveMCTS,
        )

        mcts = create_multi_objective_mcts()

        assert isinstance(mcts, MultiObjectiveMCTS)


# ---------------------------------------------------------------------------
# TestCreateScenarioEngine
# ---------------------------------------------------------------------------


class TestCreateScenarioEngine:
    """Tests for create_scenario_engine factory."""

    def test_create_empty(self) -> None:
        """Test creating an empty scenario engine."""
        from alpha_zero.strategic.scenarios import ScenarioEngine

        engine = create_scenario_engine()

        assert isinstance(engine, ScenarioEngine)
        assert engine.num_scenarios == 0

    def test_create_with_scenarios(self) -> None:
        """Test creating an engine pre-populated with scenario dicts."""
        from alpha_zero.strategic.scenarios import ScenarioEngine

        scenarios = [
            {
                "name": "optimistic",
                "parameters": {"growth": 0.05},
                "probability": 0.4,
            },
            {
                "name": "pessimistic",
                "parameters": {"growth": 0.01},
                "probability": 0.6,
            },
        ]

        engine = create_scenario_engine(scenarios=scenarios)

        assert isinstance(engine, ScenarioEngine)
        assert engine.num_scenarios == 2

        s = engine.get_scenario("optimistic")
        assert s.probability == pytest.approx(0.4)
        assert s.parameters["growth"] == pytest.approx(0.05)
