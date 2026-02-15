# Strategic Planning SDK - Implementation Plan

## Executive Summary

Transform the Civ6-AlphaZero game AI into a generalizable **Strategic Planning SDK** that enables domain experts to apply MCTS + deep neural network planning to real-world problems (climate policy, pandemic response, supply chain resilience, energy grid optimization, urban infrastructure planning).

The SDK follows the AlphaFold playbook: solve a computational grand challenge (multi-objective sequential decision-making under uncertainty), release as open infrastructure, and let domain experts find applications.

---

## Architecture Overview

```
alpha_zero/strategic/
├── __init__.py                    # Public SDK API
├── protocols.py                   # Core protocol definitions
├── config.py                      # SDK configuration system
├── objectives.py                  # Multi-objective optimization
├── planner.py                     # StrategicPlanner (main entry point)
├── network.py                     # Domain-agnostic neural network
├── mcts/
│   ├── __init__.py
│   ├── multi_objective_mcts.py    # Multi-objective MCTS engine
│   └── pareto.py                  # Pareto frontier utilities
├── trace/
│   ├── __init__.py
│   ├── decision_trace.py          # Decision trace recording
│   └── explainability.py          # Explanation generation
├── envs/
│   ├── __init__.py
│   ├── resource_allocation.py     # Generic resource allocation env
│   ├── climate_negotiation.py     # Climate policy mini-game
│   ├── pandemic_response.py       # Pandemic resource allocation env
│   └── supply_chain.py            # Supply chain disruption env
├── scenarios.py                   # Scenario engine for uncertainty
├── registry.py                    # Plugin registry for environments
├── factory.py                     # Component factories
└── __main__.py                    # CLI entry point

tests/unit/test_strategic/
├── __init__.py
├── conftest.py                    # Shared fixtures
├── test_protocols.py              # Protocol compliance tests
├── test_config.py                 # Configuration tests
├── test_objectives.py             # Multi-objective optimization tests
├── test_multi_objective_mcts.py   # MCTS engine tests
├── test_pareto.py                 # Pareto utilities tests
├── test_network.py                # Strategic network tests
├── test_decision_trace.py         # Decision trace tests
├── test_explainability.py         # Explainability tests
├── test_planner.py                # Strategic planner tests
├── test_resource_allocation.py    # Resource allocation env tests
├── test_climate_negotiation.py    # Climate env tests
├── test_pandemic_response.py      # Pandemic env tests
├── test_supply_chain.py           # Supply chain env tests
├── test_scenarios.py              # Scenario engine tests
├── test_registry.py               # Plugin registry tests
└── test_factory.py                # Factory tests

tests/integration/test_strategic/
├── __init__.py
├── test_planning_pipeline.py      # End-to-end planning pipeline
└── test_multi_env_planning.py     # Cross-environment planning
```

---

## Phase 1: Core SDK Protocols and Abstractions

### 1.1 `alpha_zero/strategic/protocols.py`

Define the foundational protocols that all SDK components use:

```python
@runtime_checkable
class StrategicEnvironment(Protocol):
    """Domain-agnostic environment for strategic planning.

    Extends EnvironmentProtocol with multi-objective support,
    partial observability, agent identification, and scenario awareness.
    """
    @property
    def num_objectives(self) -> int: ...
    @property
    def objective_names(self) -> list[str]: ...
    @property
    def num_agents(self) -> int: ...
    @property
    def current_agent(self) -> int: ...
    @property
    def state_spec(self) -> StateSpec: ...
    @property
    def action_spec(self) -> ActionSpec: ...

    def reset(self, seed=None, scenario=None) -> StrategicObservation: ...
    def step(self, action) -> StrategicStepResult: ...
    def get_valid_actions(self) -> np.ndarray: ...
    def clone(self) -> StrategicEnvironment: ...
    def get_info(self) -> dict: ...

@runtime_checkable
class ObjectiveFunction(Protocol):
    """Defines what to optimize - potentially multiple competing objectives."""
    @property
    def num_objectives(self) -> int: ...
    @property
    def names(self) -> list[str]: ...
    def evaluate(self, state, action, next_state, info) -> np.ndarray: ...
    def scalarize(self, objective_values, weights) -> float: ...

@runtime_checkable
class ScenarioProtocol(Protocol):
    """Represents an uncertainty scenario."""
    @property
    def name(self) -> str: ...
    @property
    def parameters(self) -> dict: ...
    @property
    def probability(self) -> float: ...
```

### 1.2 `alpha_zero/strategic/objectives.py`

Multi-objective optimization support with scalarization strategies:

- `ObjectiveSpec` dataclass (name, direction, weight, bounds)
- `WeightedSumScalarizer` - linear combination
- `ChebyshevScalarizer` - minimax fairness
- `EpsilonConstraintScalarizer` - optimize one, constrain rest
- `ParetoFrontier` - track non-dominated solutions

### 1.3 `alpha_zero/strategic/config.py`

SDK configuration system extending the existing pattern:

- `StrategicConfig` - master config (YAML, env vars, presets)
- `PlanningConfig` - MCTS parameters for planning mode
- `ObjectiveConfig` - objective weights and constraints
- `ScenarioConfig` - scenario parameters
- `ExplainabilityConfig` - trace and explanation settings
- Environment variable prefix: `STRATEGIC_SDK_` (alongside existing `CIV6AZ_`)

---

## Phase 2: Multi-Objective MCTS Engine

### 2.1 `alpha_zero/strategic/mcts/multi_objective_mcts.py`

Extends the existing `Civ6MCTS` architecture with multi-objective support:

- **Multi-objective node values**: Store vector values instead of scalar
- **Scalarized UCB selection**: UCB with configurable scalarization
- **Pareto-aware backpropagation**: Track Pareto frontiers at nodes
- **Scenario-aware search**: Weight simulations across scenarios
- **Action masking**: Inherited from existing MCTS
- **Temperature-based selection**: Inherited from existing MCTS

Key design: wraps existing MCTS variants (Civ6MCTS, GumbelMCTS) rather than
replacing them. When `num_objectives == 1`, degrades to standard MCTS.

### 2.2 `alpha_zero/strategic/mcts/pareto.py`

Pareto frontier utilities:

- `ParetoFrontier` class for tracking non-dominated solutions
- `dominates()` comparison function
- `hypervolume_indicator()` for quality measurement
- `crowding_distance()` for diversity maintenance

---

## Phase 3: Domain-Agnostic Strategic Network

### 3.1 `alpha_zero/strategic/network.py`

Extends existing network architecture with:

- **Configurable input dimensions**: No hard-coded channel counts
- **Multi-head value output**: One head per objective (not just scalar)
- **Optional spatial encoder**: For environments with spatial structure
- **Vector-only mode**: For non-spatial environments (e.g., supply chain)
- **Hierarchical policy head**: Support for multi-level action spaces

```python
class StrategicNetwork(nn.Module):
    def __init__(self, config: StrategicNetworkConfig):
        # Builds spatial encoder IF spatial_channels > 0
        # Builds vector encoder always
        # Builds N value heads (one per objective)
        # Builds policy head with configurable action space

    def forward(self, spatial, vector, action_mask=None):
        # Returns (policy_logits, value_vector)
        # value_vector shape: (B, num_objectives)

    def predict(self, spatial, vector, action_mask=None):
        # Returns (policy_probs, value_vector)
```

Backward compatible: when `num_objectives == 1`, returns scalar value identical to existing networks.

---

## Phase 4: Decision Trace and Explainability System

### 4.1 `alpha_zero/strategic/trace/decision_trace.py`

Records the full decision-making process:

```python
@dataclass
class DecisionPoint:
    step: int
    agent_id: int
    state_summary: dict
    action_taken: int
    action_name: str
    alternatives_considered: list[ActionAlternative]
    objective_values: np.ndarray
    mcts_visit_counts: np.ndarray
    value_estimate: np.ndarray  # Per-objective
    confidence: float
    reasoning: str  # Optional LLM-generated

@dataclass
class DecisionTrace:
    environment_name: str
    scenario_name: str
    objective_names: list[str]
    decision_points: list[DecisionPoint]
    final_outcomes: np.ndarray
    pareto_rank: int

    def to_json(self) -> str: ...
    def to_dataframe(self) -> Any: ...
    def summary(self) -> str: ...
```

### 4.2 `alpha_zero/strategic/trace/explainability.py`

Generates human-readable explanations:

- `TraceExplainer` - converts decision traces to natural language
- `TradeoffAnalyzer` - analyzes objective tradeoffs across strategies
- `SensitivityAnalyzer` - shows how scenario changes affect outcomes
- `ComparisonExplainer` - compares two strategies head-to-head

---

## Phase 5: Strategic Planner Orchestrator

### 5.1 `alpha_zero/strategic/planner.py`

Main entry point for the SDK:

```python
class StrategicPlanner:
    def __init__(self, config: StrategicConfig): ...

    def plan(self, environment, objectives, scenarios=None) -> PlanningResult:
        """Run strategic planning on the given environment.

        Returns ranked strategies with decision traces.
        """

    def train(self, environment, objectives, num_iterations) -> TrainingResult:
        """Train a policy network via self-play for the given environment."""

    def evaluate(self, strategy, environment, scenarios) -> EvaluationResult:
        """Evaluate a strategy across multiple scenarios."""

    def compare(self, strategies, environment) -> ComparisonResult:
        """Compare multiple strategies head-to-head."""
```

### 5.2 `alpha_zero/strategic/scenarios.py`

Uncertainty handling:

- `ScenarioEngine` - manages scenario sets
- `RobustnessEvaluator` - finds strategies robust across scenarios
- `SensitivityAnalysis` - identifies which scenarios matter most

---

## Phase 6: Reference Environments

### 6.1 `alpha_zero/strategic/envs/resource_allocation.py`

Generic resource allocation environment:
- N resource types, M regions, T time steps
- Each region has demand, capacity, and priority
- Actions: allocate resources across regions
- Objectives: minimize shortage, maximize equity, minimize cost
- Configurable via YAML

### 6.2 `alpha_zero/strategic/envs/climate_negotiation.py`

Simplified climate negotiation game:
- N nations with different emission levels, GDP, vulnerability
- Actions: set carbon price, invest in tech, join/leave coalition
- Objectives: minimize warming, maximize GDP, ensure equity
- Scenarios: different climate sensitivities, tech breakthroughs

### 6.3 `alpha_zero/strategic/envs/pandemic_response.py`

Pandemic resource allocation:
- Grid of regions with population, infection rates
- Resource types: vaccines, PPE, ventilators, hospital beds
- Actions: allocate resources, set restrictions, deploy testing
- Objectives: minimize deaths, minimize economic impact, equity
- Scenarios: different R0 values, variant emergence

### 6.4 `alpha_zero/strategic/envs/supply_chain.py`

Supply chain disruption response:
- Network of suppliers, manufacturers, distributors
- Disruption events (port closure, tariffs, demand spike)
- Actions: reroute, stockpile, activate backup suppliers
- Objectives: minimize cost, maximize fill rate, reduce lead time
- Scenarios: different disruption types and durations

---

## Phase 7: SDK Configuration and Factory System

### 7.1 `alpha_zero/strategic/registry.py`

Plugin registry for extensibility:

```python
class EnvironmentRegistry:
    _environments: dict[str, type[StrategicEnvironment]] = {}

    @classmethod
    def register(cls, name, env_class): ...

    @classmethod
    def create(cls, name, config) -> StrategicEnvironment: ...

    @classmethod
    def list_available(cls) -> list[str]: ...
```

### 7.2 `alpha_zero/strategic/factory.py`

Factory functions for all SDK components:

- `create_strategic_planner(config)`
- `create_strategic_network(config)`
- `create_multi_objective_mcts(config, network)`
- `create_environment(name, config)`
- `create_scenario_engine(scenarios)`

### 7.3 `alpha_zero/strategic/__main__.py`

CLI entry point:

```bash
python -m alpha_zero.strategic --help
python -m alpha_zero.strategic plan --env resource_allocation --config config.yaml
python -m alpha_zero.strategic train --env climate --preset fast --dry_run
python -m alpha_zero.strategic evaluate --strategy saved_strategy.json --scenarios scenarios.yaml
python -m alpha_zero.strategic list-envs
```

---

## Phase 8: Full Test Suite

### Test Coverage Requirements
- >= 80% coverage on all new modules
- Unit tests for every public method
- Integration tests for cross-module flows
- Parametrized tests for multiple configurations
- Mock-based tests for external dependencies

### Test Structure
Following existing patterns:
- Class-based organization with `pytest.fixture`
- Docstrings on all test methods
- `@pytest.mark.parametrize` for variant testing
- `@pytest.mark.asyncio` for async tests
- Proper mocking via `unittest.mock`
- Seed-based reproducibility

---

## Phase 9: SDK Entry Point and Packaging

### 9.1 `alpha_zero/strategic/__init__.py`

Public API surface:

```python
from alpha_zero.strategic.protocols import (
    StrategicEnvironment,
    ObjectiveFunction,
    ScenarioProtocol,
)
from alpha_zero.strategic.planner import StrategicPlanner
from alpha_zero.strategic.config import StrategicConfig
from alpha_zero.strategic.objectives import ObjectiveSpec, ParetoFrontier
from alpha_zero.strategic.trace import DecisionTrace, TraceExplainer
from alpha_zero.strategic.registry import EnvironmentRegistry
```

### 9.2 Config file: `configs/strategic_sdk.yaml`

Default configuration for the SDK.

### 9.3 Update `pyproject.toml`

Add `[project.optional-dependencies]` for `strategic` extras.

---

## Implementation Order

1. **protocols.py** + **config.py** + **objectives.py** (foundations)
2. **mcts/pareto.py** + **mcts/multi_objective_mcts.py** (search engine)
3. **network.py** (neural network)
4. **trace/decision_trace.py** + **trace/explainability.py** (explainability)
5. **scenarios.py** + **planner.py** (orchestration)
6. **envs/** (reference environments)
7. **registry.py** + **factory.py** + **__main__.py** (packaging)
8. **tests/** (full test suite alongside each phase)
9. **configs/** + **pyproject.toml** updates (packaging)

---

## Key Design Principles

1. **Backward compatible**: Existing Civ6 training continues to work unchanged
2. **No hard-coded values**: All parameters configurable via YAML/env vars
3. **Protocol-based**: Structural typing enables any compatible implementation
4. **Graceful degradation**: Optional features degrade gracefully when deps missing
5. **Factory pattern**: All components created through factories
6. **Composition over inheritance**: Build complex behaviors from simple parts
7. **Test-driven**: Tests written alongside implementation
8. **Logged and debuggable**: Structured logging at appropriate levels
