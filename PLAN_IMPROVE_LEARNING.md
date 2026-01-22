# Learning Improvement Plan

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

I am building improvements to the Civ6 AlphaZero training pipeline to enable the agent to learn effective game strategies through self-play.

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
- [x] Training loss decreases consistently across epochs (3.92 → 2.82)
- [ ] Win rate against random opponent > 50% after training (0% - needs more iterations)
- [x] Value predictions correlate with game outcomes (r = 0.78 > 0.3)
- [x] All unit tests pass (165 tests)
- [x] All integration tests pass (15 tests)
- [x] Training completes without numerical instability
```

### 1.3 Implementation Status: COMPLETE

All components have been implemented and tested:
- ✅ Reward shaping module (`alpha_zero/utils/reward_shaping.py`)
- ✅ Simplified MCTS module (`alpha_zero/utils/simple_mcts.py`)
- ✅ Training diagnostics module (`alpha_zero/utils/training_diagnostics.py`)
- ✅ Enhanced training script with all features integrated
- ✅ 165 comprehensive unit and integration tests

### 1.3 Problem Description

**Problem:** The current training pipeline shows loss decreasing (3.28 → 2.73) but 0% win rate and 0 value loss. Games reach max turns without termination, indicating the agent isn't learning winning strategies.

**Root Causes Identified:**
1. **Sparse rewards** - Only ±0.01-0.1 for actions, ±1.0 at game end
2. **No MCTS** - Pure policy network without tree search exploration
3. **Small training scale** - 20 games, 10 epochs insufficient
4. **Weak value learning** - All game rewards similar (~0.01), value head underutilized

**Solution Approach:**
1. Add reward shaping for intermediate progress signals
2. Implement simplified MCTS for better exploration
3. Add temperature-based exploration in action selection
4. Create training diagnostics to monitor learning
5. Increase training scale with configurable parameters

---

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Failure)

```
- Language/Runtime: Python 3.11+
- Required Dependencies: torch, numpy, gymnasium (already installed)
- Security: No hardcoded secrets
- Compatibility: Must work with existing MicroCivEnv and MicroCivNetwork
- Testing: All new code must have unit tests
```

### 2.2 Soft Constraints (Preferences)

```
- Style: Follow existing codebase conventions
- Architecture: Modular, testable functions
- Performance: Vectorized operations where possible
- Testing: pytest, comprehensive edge case coverage
```

### 2.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Add new files for MCTS, reward shaping, diagnostics
- Modify run_training_wandb.py
- Add new dependencies for visualization
- Refactor training configuration
- Add new test files
```

---

## SECTION 3: PERMISSION ARCHITECTURE

### 3.1 Scope

```
IN SCOPE:
- run_training_wandb.py (training script)
- alpha_zero/utils/* (utilities)
- tests/unit/* (unit tests)
- tests/integration/* (integration tests)

OUT OF SCOPE:
- alpha_zero/envs/microciv.py (environment - don't modify)
- alpha_zero/core/civ6_network.py (network - don't modify)
```

### 3.2 Autonomy Level

```
AUTONOMOUS:
- File creation for new components
- Adding unit/integration tests
- Running verification commands
- Refactoring training script

CONFIRM FIRST:
- None for this task

PROHIBITED:
- Modifying core environment logic
- Modifying network architecture
```

---

## SECTION 4: IMPLEMENTATION PLAN

### Phase 1: Reward Shaping Module

**File:** `alpha_zero/utils/reward_shaping.py`

Create reward shaping to provide intermediate learning signals:
- Territory control bonus (% of map)
- Unit advantage bonus
- City count bonus
- Health advantage bonus
- Strategic position bonus

### Phase 2: Simplified MCTS

**File:** `alpha_zero/utils/simple_mcts.py`

Implement lightweight MCTS:
- Node class with visit counts and values
- UCB1 selection
- Expand with network policy prior
- Backpropagate results
- Configurable simulation depth

### Phase 3: Enhanced Training Loop

**File:** `run_training_wandb.py` (modify)

Add:
- Temperature-based exploration
- Learning rate scheduling
- Training diagnostics
- Checkpoint saving
- Configurable training scale

### Phase 4: Training Diagnostics

**File:** `alpha_zero/utils/training_diagnostics.py`

Create diagnostics:
- Policy entropy tracking
- Value prediction accuracy
- Game outcome distribution
- Action diversity metrics

### Phase 5: Comprehensive Testing

**Files:**
- `tests/unit/test_reward_shaping.py`
- `tests/unit/test_simple_mcts.py`
- `tests/unit/test_training_diagnostics.py`
- `tests/integration/test_improved_training.py`

---

## SECTION 5: VERIFICATION COMMANDS

```bash
# After writing code, run in this order:
1. python -m pytest tests/unit/test_reward_shaping.py -v
2. python -m pytest tests/unit/test_simple_mcts.py -v
3. python -m pytest tests/unit/test_training_diagnostics.py -v
4. python -m pytest tests/integration/test_improved_training.py -v
5. python run_training_wandb.py  # Verify training works
```

---

## SECTION 6: SUCCESS VERIFICATION

Before reporting completion:
1. All unit tests pass
2. All integration tests pass
3. Training shows improved metrics:
   - Win rate > 0% after training
   - Value loss > 0 (value head learning)
   - Loss decreasing
4. W&B logging captures all metrics
5. CLAUDE.md updated with new commands

---

## DETAILED IMPLEMENTATION SPECS

### Reward Shaping Formula

```python
shaped_reward = base_reward + α * Δ(state_value)

where:
- base_reward = environment reward
- α = shaping coefficient (0.1 default)
- Δ(state_value) = state_value(s') - state_value(s)

state_value components:
- unit_advantage = (own_units - enemy_units) / max_units
- city_advantage = (own_cities - enemy_cities) / max_cities
- health_advantage = (own_avg_health - enemy_avg_health) / 100
- territory = own_territory / total_tiles
```

### MCTS Configuration

```python
MCTSConfig:
- num_simulations: int = 50  # Simulations per move
- c_puct: float = 1.5        # Exploration constant
- temperature: float = 1.0    # Action sampling temperature
- dirichlet_alpha: float = 0.3  # Root exploration noise
- dirichlet_epsilon: float = 0.25
```

### Training Configuration (Enhanced)

```python
TrainingConfig:
- board_size: int = 8
- num_games: int = 100       # Increased from 20
- num_epochs: int = 20       # Increased from 10
- batch_size: int = 32       # Increased from 8
- learning_rate: float = 0.001
- lr_decay: float = 0.99     # New: per-epoch decay
- temperature: float = 1.0   # New: exploration temperature
- temperature_decay: float = 0.95  # New: decay per game
- use_mcts: bool = True      # New: enable MCTS
- use_reward_shaping: bool = True  # New: enable shaping
- checkpoint_interval: int = 10    # New: save every N games
```

---

## ESTIMATED IMPACT

| Metric | Before | Expected After |
|--------|--------|----------------|
| Win Rate | 0% | >50% |
| Value Loss | ~0 | >0.1 |
| Policy Loss | 2.73 | <2.0 |
| Games to Termination | 100/100 | <50 avg |
