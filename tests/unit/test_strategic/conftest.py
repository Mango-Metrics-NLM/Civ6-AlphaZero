"""Shared fixtures for Strategic Planning SDK tests.

Provides reusable fixtures for objectives, environments, configuration,
and neural network instances used across the strategic test suite.
"""

from __future__ import annotations

import pytest

from alpha_zero.strategic.config import StrategicConfig
from alpha_zero.strategic.envs.climate_negotiation import ClimateNegotiationEnv
from alpha_zero.strategic.envs.pandemic_response import PandemicResponseEnv
from alpha_zero.strategic.envs.resource_allocation import ResourceAllocationEnv
from alpha_zero.strategic.envs.supply_chain import SupplyChainEnv
from alpha_zero.strategic.network import StrategicNetwork
from alpha_zero.strategic.objectives import ObjectiveSpec


@pytest.fixture
def objectives_2d() -> list[ObjectiveSpec]:
    """Return a list of two objective specs for 2D Pareto front tests.

    Objectives:
        - efficiency (maximize, weight=0.6)
        - equity (maximize, weight=0.4)
    """
    return [
        ObjectiveSpec(name="efficiency", direction="maximize", weight=0.6),
        ObjectiveSpec(name="equity", direction="maximize", weight=0.4),
    ]


@pytest.fixture
def objectives_3d() -> list[ObjectiveSpec]:
    """Return a list of three objective specs for 3D multi-objective tests.

    Objectives:
        - efficiency (maximize, weight=0.5)
        - equity (maximize, weight=0.3)
        - cost (minimize, weight=0.2)
    """
    return [
        ObjectiveSpec(name="efficiency", direction="maximize", weight=0.5),
        ObjectiveSpec(name="equity", direction="maximize", weight=0.3),
        ObjectiveSpec(name="cost", direction="minimize", weight=0.2),
    ]


@pytest.fixture
def resource_env() -> ResourceAllocationEnv:
    """Create a ResourceAllocationEnv with default config and seed=42."""
    from alpha_zero.strategic.envs.resource_allocation import ResourceAllocationConfig

    config = ResourceAllocationConfig(seed=42)
    return ResourceAllocationEnv(config)


@pytest.fixture
def climate_env() -> ClimateNegotiationEnv:
    """Create a ClimateNegotiationEnv with default config and seed=42."""
    from alpha_zero.strategic.envs.climate_negotiation import ClimateConfig

    config = ClimateConfig(seed=42)
    return ClimateNegotiationEnv(config)


@pytest.fixture
def pandemic_env() -> PandemicResponseEnv:
    """Create a PandemicResponseEnv with default config and seed=42."""
    from alpha_zero.strategic.envs.pandemic_response import PandemicConfig

    config = PandemicConfig(seed=42)
    return PandemicResponseEnv(config)


@pytest.fixture
def supply_chain_env() -> SupplyChainEnv:
    """Create a SupplyChainEnv with default config and seed=42."""
    from alpha_zero.strategic.envs.supply_chain import SupplyChainConfig

    config = SupplyChainConfig(seed=42)
    return SupplyChainEnv(config)


@pytest.fixture
def strategic_config_fast() -> StrategicConfig:
    """Return a StrategicConfig loaded from the 'fast' preset."""
    return StrategicConfig.from_preset("fast")


@pytest.fixture
def small_network() -> StrategicNetwork:
    """Create a small StrategicNetwork for fast testing.

    Configuration:
        - vector_dim=16
        - action_space_size=60
        - num_objectives=3
        - num_res_blocks=1
        - num_filters=16
        - No spatial input
    """
    return StrategicNetwork(
        vector_dim=16,
        action_space_size=60,
        num_objectives=3,
        num_res_blocks=1,
        num_filters=16,
    )
