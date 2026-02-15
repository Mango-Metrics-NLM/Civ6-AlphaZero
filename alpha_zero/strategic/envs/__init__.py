"""Reference environments for the Strategic Planning SDK.

Each environment implements the StrategicEnvironment protocol and
demonstrates a real-world strategic planning domain.
"""

from alpha_zero.strategic.envs.climate_negotiation import ClimateNegotiationEnv
from alpha_zero.strategic.envs.pandemic_response import PandemicResponseEnv
from alpha_zero.strategic.envs.resource_allocation import ResourceAllocationEnv
from alpha_zero.strategic.envs.supply_chain import SupplyChainEnv

__all__ = [
    "ClimateNegotiationEnv",
    "PandemicResponseEnv",
    "ResourceAllocationEnv",
    "SupplyChainEnv",
]
