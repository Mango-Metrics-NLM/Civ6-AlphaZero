"""Strategic Planning SDK - Domain-agnostic strategic planning intelligence.

This module transforms the AlphaZero game AI architecture into a generalizable
SDK for real-world strategic planning problems. It provides multi-objective
MCTS-based planning, explainable decision traces, and domain-agnostic
environment protocols that domain experts can implement.

Usage:
    from alpha_zero.strategic import (
        StrategicPlanner,
        StrategicConfig,
        StrategicEnvironment,
        ObjectiveSpec,
        ParetoFrontier,
        DecisionTrace,
        EnvironmentRegistry,
    )

    # Create a planner
    config = StrategicConfig.from_preset("balanced")
    planner = StrategicPlanner(config)

    # Plan with a registered environment
    result = planner.plan(
        environment=EnvironmentRegistry.create("resource_allocation"),
        objectives=[
            ObjectiveSpec(name="efficiency", direction="maximize"),
            ObjectiveSpec(name="equity", direction="maximize"),
        ],
    )

    # Examine decision traces
    for trace in result.traces:
        print(trace.summary())
"""

from __future__ import annotations

from alpha_zero.strategic.config import (
    ExplainabilityConfig,
    ObjectiveConfig,
    PlanningConfig,
    ScenarioConfig,
    StrategicConfig,
    StrategicNetworkConfig,
)
from alpha_zero.strategic.factory import (
    create_environment,
    create_multi_objective_mcts,
    create_scenario_engine,
    create_strategic_network,
    create_strategic_planner,
)
from alpha_zero.strategic.mcts.pareto import ParetoFrontier
from alpha_zero.strategic.objectives import (
    ChebyshevScalarizer,
    EpsilonConstraintScalarizer,
    ObjectiveSpec,
    WeightedSumScalarizer,
)
from alpha_zero.strategic.planner import PlanningResult, StrategicPlanner
from alpha_zero.strategic.protocols import (
    ObjectiveFunction,
    ScenarioProtocol,
    StrategicEnvironment,
)
from alpha_zero.strategic.registry import EnvironmentRegistry
from alpha_zero.strategic.scenarios import ScenarioEngine
from alpha_zero.strategic.trace.decision_trace import DecisionPoint, DecisionTrace
from alpha_zero.strategic.trace.explainability import (
    ComparisonExplainer,
    SensitivityAnalyzer,
    TraceExplainer,
    TradeoffAnalyzer,
)

__all__ = [
    "ChebyshevScalarizer",
    "ComparisonExplainer",
    "DecisionPoint",
    "DecisionTrace",
    "EnvironmentRegistry",
    "EpsilonConstraintScalarizer",
    "ExplainabilityConfig",
    "ObjectiveConfig",
    "ObjectiveFunction",
    "ObjectiveSpec",
    "ParetoFrontier",
    "PlanningConfig",
    "PlanningResult",
    "ScenarioConfig",
    "ScenarioEngine",
    "ScenarioProtocol",
    "SensitivityAnalyzer",
    "StrategicConfig",
    "StrategicEnvironment",
    "StrategicNetworkConfig",
    "StrategicPlanner",
    "TraceExplainer",
    "TradeoffAnalyzer",
    "WeightedSumScalarizer",
    "create_environment",
    "create_multi_objective_mcts",
    "create_scenario_engine",
    "create_strategic_network",
    "create_strategic_planner",
]
