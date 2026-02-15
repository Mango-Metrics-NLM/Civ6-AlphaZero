"""Factory functions for creating Strategic Planning SDK components.

Provides a unified creation interface for all SDK components,
following the factory pattern used throughout the codebase.

Usage:
    from alpha_zero.strategic.factory import (
        create_strategic_planner,
        create_environment,
        create_strategic_network,
    )

    planner = create_strategic_planner(config)
    env = create_environment("resource_allocation", seed=42)
    network = create_strategic_network(config.network)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from alpha_zero.strategic.config import StrategicConfig, StrategicNetworkConfig
    from alpha_zero.strategic.mcts.multi_objective_mcts import (
        MultiObjectiveMCTS,
        MultiObjectiveMCTSConfig,
    )
    from alpha_zero.strategic.network import StrategicNetwork
    from alpha_zero.strategic.objectives import BaseScalarizer
    from alpha_zero.strategic.planner import StrategicPlanner
    from alpha_zero.strategic.scenarios import ScenarioEngine


def create_strategic_planner(
    config: StrategicConfig | None = None,
) -> StrategicPlanner:
    """Create a StrategicPlanner instance.

    Args:
        config: Strategic planning configuration. If None, uses defaults.

    Returns:
        Configured StrategicPlanner instance.
    """
    from alpha_zero.strategic.config import StrategicConfig
    from alpha_zero.strategic.planner import StrategicPlanner

    if config is None:
        config = StrategicConfig()

    logger.debug("Creating strategic planner with mode=%s", config.mode)
    return StrategicPlanner(config)


def create_strategic_network(
    config: StrategicNetworkConfig | None = None,
    action_space_size: int = 100,
    device: str = "cpu",
) -> StrategicNetwork:
    """Create a StrategicNetwork instance.

    Args:
        config: Network configuration. If None, uses defaults.
        action_space_size: Size of the action space.
        device: Device to place the network on.

    Returns:
        Configured StrategicNetwork on the specified device.
    """
    from alpha_zero.strategic.config import StrategicNetworkConfig
    from alpha_zero.strategic.network import StrategicNetwork

    if config is None:
        config = StrategicNetworkConfig()

    network = StrategicNetwork(
        spatial_channels=config.spatial_channels,
        spatial_size=config.spatial_size,
        vector_dim=config.vector_dim,
        action_space_size=action_space_size,
        num_objectives=config.num_objectives,
        num_res_blocks=config.num_res_blocks,
        num_filters=config.num_filters,
        value_head_hidden=config.value_head_hidden,
        policy_head_hidden=config.policy_head_hidden,
        dropout=config.dropout,
        use_attention=config.use_attention,
    )

    import torch

    network = network.to(torch.device(device))
    logger.debug(
        "Created strategic network: %d params, device=%s",
        sum(p.numel() for p in network.parameters()),
        device,
    )
    return network  # type: ignore[no-any-return]


def create_multi_objective_mcts(
    config: MultiObjectiveMCTSConfig | None = None,
    network: Any = None,
    scalarizer: BaseScalarizer | None = None,
    num_objectives: int = 1,
    device: str = "cpu",
) -> MultiObjectiveMCTS:
    """Create a MultiObjectiveMCTS instance.

    Args:
        config: MCTS configuration. If None, uses defaults.
        network: Optional neural network for policy/value estimation.
        scalarizer: Optional scalarizer for multi-objective values.
        num_objectives: Number of optimization objectives.
        device: Device for network inference.

    Returns:
        Configured MultiObjectiveMCTS instance.
    """
    from alpha_zero.strategic.mcts.multi_objective_mcts import (
        MultiObjectiveMCTS,
        MultiObjectiveMCTSConfig,
    )

    if config is None:
        config = MultiObjectiveMCTSConfig()

    logger.debug(
        "Creating MCTS: simulations=%d, objectives=%d",
        config.num_simulations,
        num_objectives,
    )
    return MultiObjectiveMCTS(
        config=config,
        network=network,
        scalarizer=scalarizer,
        num_objectives=num_objectives,
        device=device,
    )


def create_environment(
    name: str,
    config: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Create a strategic environment by name.

    Uses the EnvironmentRegistry to look up and instantiate environments.

    Args:
        name: Registered environment name.
        config: Optional environment-specific config object.
        **kwargs: Additional arguments passed to the constructor.

    Returns:
        Environment instance implementing StrategicEnvironment protocol.

    Raises:
        KeyError: If environment name is not registered.
    """
    from alpha_zero.strategic.registry import EnvironmentRegistry

    if config is not None:
        kwargs["config"] = config

    logger.debug("Creating environment: %s", name)
    return EnvironmentRegistry.create(name, **kwargs)


def create_scenario_engine(
    scenarios: list[dict[str, Any]] | None = None,
) -> ScenarioEngine:
    """Create a ScenarioEngine instance.

    Args:
        scenarios: Optional list of scenario dictionaries. Each dict
            should have "name", "parameters", and optionally "probability".

    Returns:
        Configured ScenarioEngine instance.
    """
    from alpha_zero.strategic.scenarios import Scenario, ScenarioEngine

    engine = ScenarioEngine()

    if scenarios:
        for scenario_data in scenarios:
            scenario = Scenario.from_dict(scenario_data)
            engine.add_scenario(scenario)
        logger.debug("Created scenario engine with %d scenarios", engine.num_scenarios)
    else:
        logger.debug("Created empty scenario engine")

    return engine


def create_scalarizer(
    method: str = "weighted_sum",
    objectives: list[Any] | None = None,
    **kwargs: Any,
) -> BaseScalarizer:
    """Create a scalarizer by method name.

    Args:
        method: Scalarization method name.
        objectives: List of ObjectiveSpec instances.
        **kwargs: Additional scalarizer-specific arguments.

    Returns:
        Configured scalarizer instance.

    Raises:
        ValueError: If method is unknown.
    """
    from alpha_zero.strategic.objectives import (
        ChebyshevScalarizer,
        EpsilonConstraintScalarizer,
        ObjectiveSpec,
        WeightedSumScalarizer,
    )

    if objectives is None:
        objectives = [ObjectiveSpec(name="default")]

    if method == "weighted_sum":
        return WeightedSumScalarizer(objectives)
    elif method == "chebyshev":
        ideal_point = kwargs.get("ideal_point")
        if ideal_point is not None:
            ideal_point = np.asarray(ideal_point)
        return ChebyshevScalarizer(objectives, ideal_point=ideal_point)
    elif method == "epsilon_constraint":
        return EpsilonConstraintScalarizer(
            objectives,
            primary_index=kwargs.get("primary_index", 0),
            constraint_penalty=kwargs.get("constraint_penalty", 10.0),
        )
    else:
        raise ValueError(
            f"Unknown scalarization method: '{method}'. "
            f"Available: weighted_sum, chebyshev, epsilon_constraint"
        )


__all__ = [
    "create_environment",
    "create_multi_objective_mcts",
    "create_scalarizer",
    "create_scenario_engine",
    "create_strategic_network",
    "create_strategic_planner",
]
