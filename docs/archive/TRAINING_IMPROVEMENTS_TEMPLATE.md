# Training Improvements Template - Civ6 AlphaZero

> **Design Principle:** This template treats prompting as constraint programming for implementing training improvements. It defines the feasible region, objective function, and search parameters for optimizing AlphaZero training.

---

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am building: A modular training improvement system for Civ6 AlphaZero that implements
environment tuning, enhanced reward signals, curriculum learning, and scalable training configurations.
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
- [ ] All tests pass: `pytest tests/ -v` exits with code 0
- [ ] Environment tuning reduces average game length by 30%+
- [ ] Reward signals provide 5x more frequent feedback than terminal rewards alone
- [ ] Curriculum learning shows monotonic difficulty progression
- [ ] Training supports configurable batch sizes from 32 to 2048
- [ ] All configurations are externalized (no hardcoded values)
- [ ] Logging captures all training metrics with debug capability
- [ ] Coverage >= 80% on new modules
```

### 1.3 Problem Description

**Paragraph 1 - Core Problem:**
AlphaZero training for complex games like Civ6 suffers from sparse rewards and long episode lengths.
This system implements four key improvements: (1) Environment tuning to reduce max turns and adjust
game mechanics for faster termination, (2) Stronger reward signals with intermediate rewards and
additional win conditions, (3) Curriculum learning that starts with simpler scenarios before full
games, (4) Scalable training configurations for larger batch sizes and more iterations.

**Paragraph 2 - Data Flows:**
Configuration flows from YAML files through the ConfigManager to all training components.
Environment tuning parameters control MicroCiv game mechanics. Reward signals are computed
by the enhanced RewardShaper based on game state deltas. Curriculum learning stages progress
based on performance metrics tracked by the DiagnosticsTracker. Training configuration
dynamically adjusts batch sizes and iteration counts based on available resources.

**Paragraph 3 - Failure Modes:**
Critical invariants: (1) Curriculum stages must be monotonically increasing in difficulty,
(2) Reward shaping must preserve optimal policy (potential-based), (3) Environment tuning
must not break game rules or victory conditions, (4) Configuration must support both minimal
testing and full-scale training. Failure modes include: curriculum regression causing
catastrophic forgetting, reward hacking due to poorly designed intermediate rewards,
environment changes that make the game unsolvable, and memory exhaustion from oversized batches.

---

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Failure)

```
- All values configurable via YAML or environment variables
- Potential-based reward shaping only (preserves optimal policy)
- Curriculum stages must define clear transition criteria
- Environment modifications must maintain Gymnasium compatibility
- Training must support CPU and GPU execution
- All new code must have type hints
- Tests required for all public interfaces
```

### 2.2 Soft Constraints (Preferences)

```
- Configuration: Dataclasses with validation
- Defaults: Reasonable values for development (fast) and production (thorough)
- Logging: Structured with JSON support
- Metrics: Tracked via MetricsTracker with moving averages
- Documentation: Docstrings on all public methods
```

### 2.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Add new configuration parameters
- Create new utility modules
- Modify existing reward shaping
- Add new environment wrapper classes
- Implement new curriculum stage types
- Adjust default hyperparameters
```

---

## SECTION 3: IMPLEMENTATION MODULES

### 3.1 Environment Tuning Module

**File:** `alpha_zero/utils/environment_tuning.py`

**Purpose:** Configurable game mechanics for faster training

**Configuration Parameters:**
- `max_turns`: Reduced turn limit (default: 50-100 instead of 200)
- `early_termination_enabled`: Enable early game ending conditions
- `early_termination_threshold`: Advantage threshold for early termination
- `simplified_combat`: Faster combat resolution
- `accelerated_production`: Reduced unit production time
- `fog_of_war_enabled`: Toggle visibility system

**Key Classes:**
- `EnvironmentTuningConfig`: Configuration dataclass
- `TunedEnvironmentWrapper`: Gymnasium wrapper applying tuning

### 3.2 Enhanced Reward Signals Module

**File:** `alpha_zero/utils/enhanced_rewards.py`

**Purpose:** Stronger intermediate rewards and additional win conditions

**Configuration Parameters:**
- `intermediate_reward_scale`: Multiplier for intermediate rewards
- `territory_bonus_enabled`: Reward for territory control
- `unit_preservation_bonus`: Reward for keeping units alive
- `aggression_bonus`: Reward for attacking/engaging
- `objective_completion_bonus`: Reward for completing mini-objectives
- `win_conditions`: List of active victory types

**Key Classes:**
- `EnhancedRewardsConfig`: Configuration dataclass
- `EnhancedRewardShaper`: Extended reward computation
- `WinConditionEvaluator`: Multi-objective victory checking

### 3.3 Curriculum Learning Module

**File:** `alpha_zero/utils/curriculum_learning.py`

**Purpose:** Progressive difficulty training

**Configuration Parameters:**
- `stages`: List of curriculum stages with parameters
- `promotion_threshold`: Win rate required to advance
- `demotion_threshold`: Win rate causing demotion
- `min_games_per_stage`: Minimum evaluation games
- `difficulty_dimensions`: Which parameters vary by stage

**Key Classes:**
- `CurriculumStage`: Single stage configuration
- `CurriculumConfig`: Overall curriculum configuration
- `CurriculumManager`: Stage progression logic
- `AdaptiveDifficultyController`: Dynamic difficulty adjustment

### 3.4 Training Configuration Module

**File:** `alpha_zero/utils/training_config_enhanced.py`

**Purpose:** Scalable training parameters

**Configuration Parameters:**
- `batch_size_schedule`: Schedule for batch size progression
- `learning_rate_schedule`: Schedule for learning rate decay
- `num_iterations`: Total training iterations
- `games_per_iteration`: Self-play games per iteration
- `replay_buffer_size`: Maximum replay buffer capacity
- `gradient_accumulation_steps`: For larger effective batch sizes

**Key Classes:**
- `EnhancedTrainingConfig`: Extended training configuration
- `ScheduleConfig`: Parameter scheduling configuration
- `ResourceManager`: Memory and compute management

---

## SECTION 4: TESTING REQUIREMENTS

### 4.1 Unit Tests Required

```
tests/unit/test_environment_tuning.py
- test_tuning_config_defaults
- test_tuning_config_from_yaml
- test_early_termination_trigger
- test_accelerated_production
- test_simplified_combat
- test_max_turns_reduction

tests/unit/test_enhanced_rewards.py
- test_enhanced_config_defaults
- test_intermediate_rewards
- test_territory_bonus_calculation
- test_unit_preservation_bonus
- test_aggression_bonus
- test_win_condition_evaluation
- test_potential_based_invariant

tests/unit/test_curriculum_learning.py
- test_curriculum_config_validation
- test_stage_progression
- test_stage_demotion
- test_difficulty_dimensions
- test_adaptive_adjustment
- test_metrics_tracking

tests/unit/test_training_config_enhanced.py
- test_batch_size_schedule
- test_learning_rate_schedule
- test_gradient_accumulation
- test_resource_management
- test_config_serialization
```

### 4.2 Integration Tests Required

```
tests/integration/test_training_improvements.py
- test_tuned_environment_training
- test_enhanced_rewards_training
- test_curriculum_progression
- test_full_training_pipeline
- test_configuration_persistence
```

---

## SECTION 5: FEEDBACK LOOP

### 5.1 Verification Commands

```bash
# 1. Lint new modules
ruff check alpha_zero/utils/environment_tuning.py alpha_zero/utils/enhanced_rewards.py \
    alpha_zero/utils/curriculum_learning.py alpha_zero/utils/training_config_enhanced.py

# 2. Type check
mypy alpha_zero/utils/ --ignore-missing-imports

# 3. Unit tests
pytest tests/unit/test_environment_tuning.py tests/unit/test_enhanced_rewards.py \
    tests/unit/test_curriculum_learning.py tests/unit/test_training_config_enhanced.py -v

# 4. Integration tests
pytest tests/integration/test_training_improvements.py -v

# 5. Full test suite
pytest tests/ -v --cov=alpha_zero

# 6. Training smoke test with improvements
python -m alpha_zero.training_civ6 --dry_run --use_tuned_env --use_enhanced_rewards --curriculum
```

---

## SECTION 6: CONFIGURATION SCHEMA

### 6.1 YAML Configuration Example

```yaml
# configs/training_improvements.yaml

environment_tuning:
  enabled: true
  max_turns: 100
  early_termination:
    enabled: true
    advantage_threshold: 3  # cities or units
  production_multiplier: 2.0
  fog_of_war_enabled: true

enhanced_rewards:
  enabled: true
  scale_factor: 1.0
  intermediate:
    territory_control: 0.02
    unit_preservation: 0.01
    city_proximity: 0.005
    exploration: 0.001
  victory_bonuses:
    domination: 1.0
    elimination: 1.0
    score: 0.5

curriculum:
  enabled: true
  stages:
    - name: "beginner"
      board_size: 8
      max_turns: 30
      num_opponents: 1
      promotion_threshold: 0.7
    - name: "intermediate"
      board_size: 12
      max_turns: 60
      num_opponents: 1
      promotion_threshold: 0.6
    - name: "advanced"
      board_size: 16
      max_turns: 100
      num_opponents: 1
      promotion_threshold: 0.55
  min_games_per_stage: 50
  allow_demotion: false

training:
  batch_size: 256
  batch_size_schedule:
    - iteration: 0
      batch_size: 64
    - iteration: 1000
      batch_size: 128
    - iteration: 5000
      batch_size: 256
  learning_rate: 0.001
  learning_rate_schedule:
    - iteration: 0
      lr: 0.001
    - iteration: 10000
      lr: 0.0001
    - iteration: 50000
      lr: 0.00001
  num_iterations: 100000
  games_per_iteration: 100
  replay_buffer_size: 500000
  gradient_accumulation_steps: 1

logging:
  level: INFO
  metrics_window: 100
  checkpoint_interval: 1000
  evaluation_interval: 500
```

---

## SECTION 7: IMPLEMENTATION CHECKLIST

```
Phase 1: Configuration Infrastructure
□ Create enhanced configuration dataclasses
□ Add YAML loading with validation
□ Implement environment variable overrides
□ Write configuration tests

Phase 2: Environment Tuning
□ Implement EnvironmentTuningConfig
□ Create TunedEnvironmentWrapper
□ Add early termination logic
□ Implement accelerated mechanics
□ Write environment tuning tests

Phase 3: Enhanced Rewards
□ Implement EnhancedRewardsConfig
□ Create EnhancedRewardShaper
□ Add intermediate reward calculations
□ Implement win condition evaluator
□ Write reward shaping tests

Phase 4: Curriculum Learning
□ Implement CurriculumConfig and CurriculumStage
□ Create CurriculumManager
□ Add stage progression logic
□ Implement difficulty controller
□ Write curriculum tests

Phase 5: Training Integration
□ Update training pipeline to use improvements
□ Add CLI arguments for new features
□ Implement schedule-based parameter adjustment
□ Write integration tests

Phase 6: Documentation & Polish
□ Update CLAUDE.md with new commands
□ Add usage examples
□ Verify all tests pass
□ Generate coverage report
```

---

*Template Version: 1.0*
*Last Updated: 2026-01-19*
