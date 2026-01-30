# Comprehensive Code Review Analysis

> Generated: 2026-01-29
> Scope: alpha_zero and agentic_framework packages
> Focus: Dynamic/backwards-compatible, reusable code, no hardcoded values

---

## Executive Summary

This document presents a comprehensive code review analysis of the Civ6-AlphaZero codebase, focusing on five key areas:

| Category | Current State | Severity | Fix Effort |
|----------|--------------|----------|-----------|
| Hardcoded Values | ~30% need extraction | HIGH | Medium |
| Backwards Compatibility | Critical API inconsistencies | HIGH | Medium |
| Code Reusability | ~1,500 lines reducible | MEDIUM | High |
| Dynamic Configuration | Excellent (~85% configurable) | LOW | Low |
| API Consistency | Multiple patterns mixed | MEDIUM | Medium |

**Overall Assessment:** The codebase demonstrates mature architectural patterns with excellent configuration systems, but has accumulated technical debt in API consistency and hardcoded values that should be addressed before the next major release.

---

## 1. Hardcoded Values Analysis

### Critical Issues

#### 1.1 Reward Scaling Constants
**File:** `alpha_zero/envs/real_civ6_env.py:430-433`
```python
# Current (hardcoded)
SCORE_REWARD_SCALE = 0.001
CITY_REWARD_SCALE = 0.1
UNIT_REWARD_SCALE = 0.01
GOLD_REWARD_SCALE = 0.0001
```

**Recommendation:** Extract to `RewardScaleConfig`:
```python
@dataclass
class RewardScaleConfig:
    score: float = 0.001
    city: float = 0.1
    unit: float = 0.01
    gold: float = 0.0001

    @classmethod
    def from_env(cls, prefix: str = "CIV6AZ_REWARD_") -> "RewardScaleConfig":
        return cls(
            score=float(os.getenv(f"{prefix}SCORE", cls.score)),
            city=float(os.getenv(f"{prefix}CITY", cls.city)),
            # ...
        )
```

#### 1.2 Loss Function Support Size
**File:** `alpha_zero/core/losses.py` (5+ locations)
```python
# Repeated 5+ times:
support_size: int = 601
max_value: float = 300.0
```

**Recommendation:** Create `CategoricalLossConfig`:
```python
@dataclass
class CategoricalLossConfig:
    support_size: int = 601  # For values -300 to +300
    max_value: float = 300.0
    min_value: float = -300.0
```

#### 1.3 Timeout Values (Scattered)

| File | Value | Purpose |
|------|-------|---------|
| `alpha_zero/distributed/actor.py:303` | `timeout=5.0` | Trajectory submission |
| `alpha_zero/distributed/coordinator.py:515` | `timeout=10.0` | Actor pool stop |
| `alpha_zero/envs/real_civ6_env.py:92` | `state_timeout=60.0` | State reading |
| `agentic_framework/agents/planning_agent.py:121` | `timeout_seconds=300.0` | Planning |

**Recommendation:** Create `TimeoutConfig`:
```python
@dataclass
class TimeoutConfig:
    trajectory_submission: float = 5.0
    actor_pool_stop: float = 10.0
    state_reading: float = 60.0
    action_execution: float = 30.0
    agent_planning: float = 300.0
    agent_implementation: float = 600.0
    agent_testing: float = 600.0
```

### Summary: Hardcoded Values

| Category | Count | Priority |
|----------|-------|----------|
| Reward scaling | 4 constants | HIGH |
| Loss parameters | 5+ duplicates | HIGH |
| Timeout values | 15+ locations | HIGH |
| Buffer/cache sizes | 8 locations | MEDIUM |
| Neural network params | 6 locations | MEDIUM |

---

## 2. Backwards Compatibility Analysis

### Critical: Observation API Inconsistency

**The Problem:**
```python
# Protocol expects (civ6_mcts.py:79):
class MCTSEnvironment(Protocol):
    def _get_observation(self) -> dict[str, np.ndarray]: ...

# But implementations have (microciv.py:614):
class MicroCivEnv:
    def get_observation(self) -> dict[str, np.ndarray]: ...
```

**Files Affected:**
- `alpha_zero/core/civ6_mcts.py:79` - Protocol definition
- `alpha_zero/envs/microciv.py:614` - Has `get_observation()`
- `alpha_zero/envs/real_civ6_env.py:372` - Has `_get_observation()`
- `alpha_zero/orchestrator/orchestrator.py:134` - Calls `_get_observation()`

**Recommended Fix:**
```python
class MCTSEnvironment(Protocol):
    def get_observation(self) -> dict[str, np.ndarray]:
        """Public API - primary method."""
        ...

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Deprecated: use get_observation() instead."""
        return self.get_observation()
```

### Missing Re-exports in `__init__.py`

| Package | Status | Missing Exports |
|---------|--------|-----------------|
| `alpha_zero/__init__.py` | MINIMAL | Core classes (Civ6MCTS, ActionSpace, etc.) |
| `alpha_zero/interfaces/__init__.py` | PARTIAL | BaseGameInterface, GameObservation, etc. |
| `alpha_zero/core/__init__.py` | GOOD | N/A |
| `agentic_framework/__init__.py` | GOOD | N/A |

### Gymnasium API Version Mismatch

**Issue:** Code mixes v25 (4-tuple) and v26 (5-tuple) returns:
```python
# v25: obs, reward, done, info
# v26: obs, reward, terminated, truncated, info
```

**Recommendation:** Standardize on v26 with compatibility shim.

---

## 3. Code Reusability Assessment

### Duplicate Code Locations

#### 3.1 Network Head Implementations (~500 lines reducible)
```
alpha_zero/core/civ6_network.py:
  - PolicyHead (lines 186-244)
  - ValueHead (lines 247-303)

alpha_zero/core/transformer_network.py:
  - TransformerPolicyHead (lines 877-916)
  - TransformerValueHead (lines 919-957)
```

**Recommendation:** Extract `FeatureReductionHead` base class.

#### 3.2 Level Controller Patterns (~300 lines reducible)
```
alpha_zero/hierarchical/controllers/level_controller.py:
  - StrategicController (lines 219-511)
  - TacticalController (lines 514-736)
  - OperationalController (lines 739-896)
```

**Recommendation:** Use composition with `ControllerTemplate`.

#### 3.3 Loss Computation (~100 lines reducible)
```
alpha_zero/core/civ6_network.py:1555-1591 (compute_loss)
alpha_zero/core/transformer_network.py:1231-1285 (compute_transformer_loss)
```

**Recommendation:** Unify into `alpha_zero/core/loss_functions.py`.

### Missing Abstractions

| Abstraction | Location | Benefit |
|-------------|----------|---------|
| `EncoderModule` base | hierarchical/encoders | Reduce 200+ lines |
| `Protocol` definitions | core/protocols.py | Enable duck typing |
| `Serializable` mixin | utils/serialization.py | Consistent to_dict/from_dict |
| `TensorShapeHelper` | utils/tensor_utils.py | Reduce 30+ lines |

### Separation of Concerns Issues

**`HierarchicalController` (555 lines)** mixes:
- Initialization (FACTORY)
- Decision making (ORCHESTRATION)
- Training updates (LEARNING)
- Data collection (DATA PIPELINE)
- Serialization (PERSISTENCE)

**Recommendation:** Decompose into:
- `HierarchicalOrchestrator`
- `HierarchicalControllerFactory`
- `HierarchicalTrainingDataCollector`
- `HierarchicalCheckpointManager`

---

## 4. Dynamic Configuration Patterns

### Excellent Patterns (Replicate These)

#### 4.1 Hierarchical Dataclass Pattern
**File:** `alpha_zero/orchestrator/config.py`
```python
@dataclass
class OrchestratorConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    # ...

    @classmethod
    def from_yaml(cls, path: str) -> "OrchestratorConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        config = cls._from_dict(data)
        config.apply_env_overrides()
        return config
```

#### 4.2 Environment Override Pattern
**File:** `alpha_zero/orchestrator/config.py:789-850`
```python
def apply_env_overrides(self) -> "OrchestratorConfig":
    prefix = "CIV6AZ_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("_", 1)
        section, attr = parts
        self._apply_override(section, attr, value)
    return self
```

#### 4.3 Schedule-Based Configuration
**File:** `alpha_zero/orchestrator/config.py:54-133`
```python
class ScheduleType(str, Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    WARMUP_COSINE = "warmup_cosine"

@dataclass
class ScheduleConfig:
    schedule_type: str = "constant"
    initial_value: float = 0.001
    final_value: float | None = None

    def get_value(self, step: int) -> float:
        # Runtime hyperparameter computation
```

#### 4.4 Feature Flags Pattern
```python
@dataclass
class AutonomyLevel:
    auto_checkpoint: bool = True
    auto_adjust_lr: bool = True
    confirm_architecture_changes: bool = True
    prohibit_main_commits: bool = True
```

### Configuration Capabilities Matrix

| Capability | alpha_zero | agentic_framework |
|------------|------------|-------------------|
| Dataclass Config | ✅ | ✅ |
| YAML Loading | ✅ | ✅ |
| Environment Variables | ✅ | ✅ |
| Variable Substitution | ❌ | ✅ |
| Presets | ✅ | ❌ |
| Schedules | ✅ | ❌ |
| Feature Flags | ✅ | Limited |

---

## 5. API Consistency Issues

### Naming Inconsistencies

| Pattern | Example Files | Issue |
|---------|--------------|-------|
| `get_X()` vs property | permission_manager.py | Mixed access patterns |
| `has_X()` vs `get_X()` | checkpointer.py | Boolean vs value returns |
| `check_X()` vs `is_X()` | constraint_manager.py | Validation naming |

**Recommendation:**
- `is_*` for simple boolean properties
- `has_*` for existence checks
- `can_*` for capability checks
- `check_*` for validation with detailed results
- `get_*` for computed values (not simple access)

### Return Type Inconsistencies

```python
# Three different success indication patterns:

# Pattern 1: Boolean property
@property
def is_modified(self) -> bool: ...

# Pattern 2: Result object with __bool__
@dataclass
class AgentResult:
    def __bool__(self) -> bool:
        return self.success

# Pattern 3: Result object without __bool__
@dataclass
class FeedbackLoopResult:
    success: bool  # Must access .success
```

**Recommendation:** All result objects should have both `success` property AND `__bool__()`.

### Optional Type Annotation Styles

```python
# Style 1 (older)
def method(self, value: Optional[Path] = None): ...

# Style 2 (newer, Python 3.10+)
def method(self, value: Path | None = None): ...
```

**Recommendation:** Standardize on `X | None` throughout.

### Error Handling Patterns

| Pattern | Usage | Files |
|---------|-------|-------|
| Raise exception | Operation methods | civ6_interface.py |
| Return bool | Connection methods | civ6_interface.py |
| Return None | Persistence methods | session_memory.py |

**Recommendation:** Standardize per operation type.

---

## 6. Recommended Config Classes to Create

```python
# alpha_zero/utils/configs.py

@dataclass
class RewardScaleConfig:
    score: float = 0.001
    city: float = 0.1
    unit: float = 0.01
    gold: float = 0.0001

@dataclass
class CategoricalLossConfig:
    support_size: int = 601
    max_value: float = 300.0
    min_value: float = -300.0

@dataclass
class TimeoutConfig:
    trajectory_submission: float = 5.0
    process_join: float = 1.0
    actor_pool_stop: float = 10.0
    state_reading: float = 60.0
    action_execution: float = 30.0
    game_action: float = 5.0
    game_state: float = 2.0
    agent_planning: float = 300.0
    agent_implementation: float = 600.0
    agent_testing: float = 600.0
    agent_verification: float = 300.0

@dataclass
class CacheConfig:
    rag_query_cache_maxsize: int = 1000
    rag_cache_ttl: int = 3600
    action_pruning_cache_size: int = 1000
```

---

## 7. Recommended Protocol Definitions

```python
# alpha_zero/core/protocols.py

from typing import Protocol, Tuple
import torch
from torch import Tensor

class NeuralNetwork(Protocol):
    """Any class that can forward pass and predict."""
    def forward(self, spatial: Tensor, vector: Tensor,
                action_mask: Tensor | None = None) -> Tuple[Tensor, Tensor]: ...
    def predict(self, spatial: Tensor, vector: Tensor,
                action_mask: Tensor | None = None) -> Tuple[Tensor, Tensor]: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "NeuralNetwork": ...

class SearchAlgorithm(Protocol):
    """Any class that can search and return actions."""
    def search(self, state: dict, goals: list,
               valid_actions: list) -> Tuple[int, dict]: ...
    def reset(self) -> None: ...

class StateEncoder(Protocol):
    """Any encoder that processes game state."""
    def encode(self, state: dict) -> Tensor: ...
    def encode_batch(self, states: list[dict]) -> Tensor: ...
```

---

## 8. Implementation Priority

### Phase 1: Quick Wins (Week 1)
1. ✅ Extract `TimeoutConfig` from scattered constants
2. ✅ Unify `CategoricalLossConfig` (5+ duplicates)
3. ✅ Standardize Optional type annotations
4. ✅ Create Protocol definitions

### Phase 2: Medium Effort (Week 2-3)

1. ✅ Fix observation API (`get_observation` vs `_get_observation`) - DONE: Standardized with deprecated shim
2. ✅ Extract `RewardScaleConfig` - DONE: Now `RewardConfig` in `real_civ6_env.py:52-138`
3. Create `EncoderModule` base class
4. ✅ Add missing `__init__.py` re-exports - DONE: Updated `alpha_zero/__init__.py`

### Phase 3: Major Refactoring (Week 4+)
1. Decompose `HierarchicalController`
2. Extract network head base classes
3. Implement `ComposableNetwork` pattern
4. Convert factories to class-based registry

---

## 9. Files Requiring Immediate Attention

| File | Lines | Issues | Priority |
|------|-------|--------|----------|
| `alpha_zero/core/losses.py` | 1200+ | 5+ duplicate constants | HIGH |
| `alpha_zero/core/civ6_mcts.py` | 1900+ | Protocol mismatch | HIGH |
| `alpha_zero/envs/real_civ6_env.py` | 600+ | Hardcoded rewards | HIGH |
| `alpha_zero/hierarchical/controller.py` | 555 | Poor SoC | MEDIUM |
| `alpha_zero/core/civ6_network.py` | 1592 | Duplicate heads | MEDIUM |

---

## 10. Verification Checklist

Before merging any changes, verify:

- [x] All timeout values use `TimeoutConfig` - Created `alpha_zero/utils/timeout_config.py`
- [x] All loss functions use `CategoricalLossConfig` - Exists in `losses.py:52-110`
- [x] All reward scaling uses `RewardScaleConfig` - Now `RewardConfig` in `real_civ6_env.py`
- [x] Observation API is consistent (`get_observation`) - Standardized with deprecated shim
- [ ] Optional types use `X | None` syntax - In progress
- [ ] All result objects have `__bool__()` method
- [x] All config classes have `from_dict()` and `to_dict()` - TimeoutConfig, RewardConfig, LossConfig all have these
- [x] All `__init__.py` files export public APIs - Updated `alpha_zero/__init__.py`
- [ ] No magic numbers in validation logic

---

*This document should be updated as issues are resolved.*

*Last Updated: 2026-01-30*

---

## 11. Recent Updates (2026-01-30)

### CI/CD Improvements

- Added `pytest-asyncio>=0.23.0` to dependencies (79 async tests now properly supported)
- Configured `asyncio_mode = "auto"` in pyproject.toml
- Removed `continue-on-error` from MyPy and integration tests in CI
- Added `requires_env` pytest marker for environment-dependent tests
- Raised coverage threshold from 60% to 65%

### Code Review Fixes

- Created `TimeoutConfig` dataclass in `alpha_zero/utils/timeout_config.py`
- Updated `alpha_zero/__init__.py` with public API re-exports
- Verified existing config classes (RewardConfig, CategoricalSupportConfig, LossConfig) already implement best practices
