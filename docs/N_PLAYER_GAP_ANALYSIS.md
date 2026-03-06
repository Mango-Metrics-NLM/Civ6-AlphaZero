# N-Player MicroCiv Gap Analysis

**Date**: 2026-03-06
**Sprint**: Sprint 6 - Phase 1
**Version**: 1.4.0
**Status**: ANALYSIS COMPLETE ✅

---

## Executive Summary

MicroCiv environment supports 2-6 players (`num_players` parameter), but core training infrastructure has **critical gaps** preventing true N-player gameplay. This document analyzes three gaps identified in the roadmap and provides implementation guidance for Phase 2.

### Gap Status Overview

| Gap | Component | Status | Impact | Fix Complexity |
|-----|-----------|--------|--------|----------------|
| **G7** | Turn Rotation | ✅ RESOLVED | None | N/A (already N-aware) |
| **G8** | Encoder Channels | ⚠️ PARTIAL | Cannot distinguish opponents in 3+ player games | 20-80 LOC |
| **G9** | Value Vector | ❌ MISSING | Cannot represent non-zero-sum outcomes | 50-200 LOC |

**Blocker Assessment**: G8 and G9 are **hard blockers** for N-player training. Without these fixes, 4-player games will:
- Train on corrupted state representations (all opponents look identical)
- Use incorrect value targets (zero-sum assumption breaks down)
- Fail to learn meaningful policies (encoder confusion + value mismatch)

**Roadmap Impact**: Phase 2 must address G8 and G9 before M1 completion (N-player MicroCiv fully functional).

---

## G7: Turn Rotation (✅ RESOLVED)

### Current State

**Location**: `alpha_zero/envs/microciv.py:628`

```python
def _next_turn(self) -> None:
    """Advance to next player's turn."""
    self.current_player = (self.current_player + 1) % self.num_players
    self.turn_number += 1
```

### Analysis

The turn rotation logic is **already N-aware**:
- Uses modulo arithmetic `% self.num_players` for wraparound
- Works correctly for any number of players (2-6)
- No hardcoded assumptions about player count

### Verification

Tested in Sprint 6 Phase 1 exploration:
- 2-player: Player 0 → Player 1 → Player 0 (cycle)
- 4-player: Player 0 → Player 1 → Player 2 → Player 3 → Player 0 (cycle)
- 6-player: Player 0 → ... → Player 5 → Player 0 (cycle)

### Conclusion

**G7 is RESOLVED** ✅ - No implementation changes needed. Turn rotation is fully N-player compatible.

---

## G8: Encoder Channel Allocation (⚠️ PARTIAL)

### Problem Statement

**Location**: `alpha_zero/envs/civ6_encoder.py:322,377`

**Current Architecture** (13 channels):
```
Channels 0-2:   Terrain (plains, grassland, water)
Channels 3-5:   Own units (warrior, settler, archer)
Channels 6-8:   Enemy units (warrior, settler, archer)  ← PROBLEM
Channels 9-10:  Own cities, Enemy cities                ← PROBLEM
Channel 11:     Fog of war
Channel 12:     Bias channel (all ones)
```

**Issue**: "Enemy units" (channels 6-8) aggregates ALL opponents into a single channel group. In a 4-player game:
- Own units (Player 0) → Channels 3-5 ✅
- Player 1 units → Channels 6-8 ❌ (conflated with Player 2 and Player 3)
- Player 2 units → Channels 6-8 ❌ (conflated with Player 1 and Player 3)
- Player 3 units → Channels 6-8 ❌ (conflated with Player 1 and Player 2)

The network **cannot distinguish** between different opponents. This breaks the Markov assumption and corrupts policy/value learning.

### Impact

**Severity**: HIGH (training corruption)

**Scenarios Affected**:
- **3+ player games**: All opponents look identical to the network
- **Policy learning**: Cannot learn opponent-specific strategies (e.g., "attack Player 2 first")
- **Value prediction**: Cannot distinguish game states where different opponents are winning

**Example Failure Case**:
```
Game State A: Player 1 has 10 units, Player 2 has 1 unit
Game State B: Player 1 has 1 unit, Player 2 has 10 units

Current Encoder: Both states produce IDENTICAL channels 6-8 (aggregate 11 enemy units)
→ Network cannot tell the difference → Wrong policy/value predictions
```

### Current Code (Simplified)

**File**: `alpha_zero/envs/civ6_encoder.py:377-380`

```python
def _encode_spatial(self, state, current_player):
    spatial = np.zeros((13, self.board_size, self.board_size), dtype=np.float32)

    for unit in state.units:
        if unit.player == current_player:
            # Own units → channels 3-5
            channel_offset = 3
        else:
            # ALL enemy units → channels 6-8 (PROBLEM: no per-player distinction)
            channel_offset = 6

        type_idx = unit.type.value  # 0=warrior, 1=settler, 2=archer
        x, y = unit.position
        spatial[channel_offset + type_idx, y, x] = unit.health / 100.0
```

**Problem Line**: `else: channel_offset = 6` treats all opponents identically.

### Proposed Fix: 28-Channel Encoder

**Design Decision** (user confirmed, Sprint 6 Phase 1 planning): **Fixed 28 channels** supporting up to 6 players max.

**New Channel Layout** (28 channels):
```
Channels 0-2:   Terrain (plains, grassland, water)
Channels 3-5:   Own units (warrior, settler, archer)
Channels 6-8:   Player 1 units (relative to current player, rotated perspective)
Channels 9-11:  Player 2 units
Channels 12-14: Player 3 units
Channels 15-17: Player 4 units
Channels 18-20: Player 5 units
Channel 21:     Own cities
Channels 22-26: Opponent cities (P1-P5)
Channel 27:     Fog/bias
```

**Key Insight**: Use **relative player indexing** with rotated perspective:
- Current player always occupies channels 3-5 (ego-centric view)
- Opponent 1 (next in turn order) → channels 6-8
- Opponent 2 → channels 9-11
- ... and so on

**Example** (4-player game, current player = Player 2):
```
Channels 3-5:   Player 2 units (self)
Channels 6-8:   Player 3 units (next in turn order)
Channels 9-11:  Player 0 units (wraparound)
Channels 12-14: Player 1 units
Channels 15-20: Zeros (unused, only 4 players)
```

**Backward Compatibility** (2-player games):
- Channels 0-12: Same as old 13-channel encoder (terrain + own + opponent 1 + cities + fog)
- Channels 13-27: Zeros (unused)
- Old models can be migrated by padding channels 13-27 with zeros

### Implementation Plan

**Phase 2 Task 2.1**: Extend encoder to 28 channels (Sprint 6)

**Files to Modify**:
1. `alpha_zero/envs/civ6_encoder.py` (~40 LOC)
   - Update `TOTAL_CHANNELS = 13` → `28`
   - Refactor `_encode_spatial()` to use relative player indexing
   - Add `num_players` parameter to `__init__()`

2. `alpha_zero/cloud/gcp/config.py` (~2 LOC)
   - Update `NetworkArchitectureConfig.spatial_channels = 13` → `28`

3. `tests/unit/test_encoder.py` (~60 LOC)
   - Add test cases for 2-player, 4-player, 6-player
   - Verify backward compat (2-player uses channels 0-12, zeros in 13-27)

4. `scripts/migrate_models_13to28_channels.py` (~80 LOC, NEW)
   - Utility to pad existing models: copy channels 0-12, zero-fill 13-27
   - Allows fine-tuning existing models instead of retraining from scratch

**Expected LOC**: 182 LOC total

**Testing**:
```bash
pytest tests/unit/test_encoder.py::test_encoder_2player -v
pytest tests/unit/test_encoder.py::test_encoder_4player -v
pytest tests/unit/test_encoder.py::test_encoder_6player -v
```

### Risks & Mitigations

**Risk 1**: Network retraining (28 channels invalidates existing 13-channel models)
- **Mitigation**: Use migration script, then 10-iteration fine-tuning (~$0.50 cost)
- **Cost**: v218 model has decent strength (ELO ~1500), worth preserving via migration

**Risk 2**: Training time increase (more channels = larger model)
- **Mitigation**: Monitor training speed; if >20% slower, reduce `embed_dim` (64→48)
- **Fallback**: If training too slow, use smaller architecture (6 layers → 4 layers)

---

## G9: Value Vector (❌ MISSING)

### Problem Statement

**Locations**:
- `alpha_zero/core/civ6_network.py` (value head output layer)
- `alpha_zero/cloud/gcp/training_worker.py:437,444` (forward pass + loss computation)

**Current Architecture**: Scalar value head
```python
self.value_head = nn.Sequential(
    nn.Linear(embed_dim, value_head_dim),  # e.g., 64 → 64
    nn.ReLU(),
    nn.Linear(value_head_dim, 1),  # ← PROBLEM: scalar output
    nn.Tanh(),  # value ∈ [-1, 1]
)
```

**Issue**: Scalar value assumes **zero-sum 2-player** games:
- Winner gets `value = +1`
- Loser gets `value = -1`
- Sum of values = 0 (zero-sum property)

In N-player games (N ≥ 3), this breaks down:
- **Example** (4-player game): Player 0 wins, Players 1-3 lose
  - Current: All players get value ∈ [-1, +1] (no way to represent "Player 2 came in 2nd place")
  - Desired: Per-player outcomes, e.g., `[+1.0, +0.5, -0.5, -1.0]` (winner, 2nd, 3rd, 4th)

### Impact

**Severity**: HIGH (training corruption)

**Scenarios Affected**:
- **3+ player games**: Value targets are undefined (what is the "value" for Player 1 if Player 3 wins?)
- **Coalition dynamics**: Cannot represent "Player 0 and Player 1 alliance vs Player 2 and Player 3"
- **Elimination scenarios**: Cannot represent "Player 2 eliminated early, Player 3 eliminated late"

**Example Failure Case**:
```
4-player game:
- Player 0 (current player): 100 HP, 5 units, winning
- Player 1: 10 HP, 1 unit, losing
- Player 2: 50 HP, 3 units, competitive
- Player 3: 80 HP, 4 units, competitive

Current scalar value: Network predicts single value ∈ [-1, +1]
→ Cannot represent "Player 0 is winning, but Player 3 is close"
→ Training targets are ambiguous (what should value be for Player 0's perspective?)
```

### Current Code (Simplified)

**File**: `alpha_zero/cloud/gcp/training_worker.py:444`

```python
# Forward pass
policy_logits, value = model(spatial, vector, action_mask)
# value shape: (batch_size,) ← scalar per sample

# Value loss
value_loss = F.mse_loss(value.squeeze(), target_values)
# target_values: (batch_size,) with winner=1, loser=-1
```

**Problem**: `target_values` is a single scalar per trajectory. For 4-player games, we need 4 values (one per player).

### Proposed Fix: 6-Dimensional Value Vector

**Design Decision** (user confirmed, Sprint 6 Phase 1 planning): **Per-player value vector `[6]`** (pad unused players with 0).

**New Value Head**:
```python
self.value_head = nn.Sequential(
    nn.Linear(embed_dim, value_head_dim),  # e.g., 64 → 64
    nn.ReLU(),
    nn.Linear(value_head_dim, 6),  # ← 6 neurons (max 6 players)
    nn.Tanh(),  # Each value ∈ [-1, 1]
)

# Output shape: (batch_size, 6)
# For 4-player game: use indices 0-3, ignore 4-5
```

**Training Changes**:
```python
# Store current_player in trajectory (NEW)
trajectory = {
    "spatial": spatial_obs,
    "vector": vector_obs,
    "action_mask": action_mask,
    "policy": mcts_policy,
    "value": game_outcome,  # Still scalar (winner=1, loser=-1)
    "current_player": env.current_player,  # ← NEW FIELD
}

# Training worker: extract per-player value
value = model(spatial, vector, action_mask)  # shape: (batch_size, 6)
current_player_indices = current_players  # shape: (batch_size,) with values 0-5

# Index into value vector to get current player's value
value_for_player = value[torch.arange(batch_size), current_player_indices]
value_loss = F.mse_loss(value_for_player, target_values)
```

**Key Insight**: During training, we only supervise the **current player's value** (the one who made the decision). The network learns to predict all 6 values, but we only compute loss on the relevant index.

### Implementation Plan

**Phase 2 Task 2.2**: Extend value head to 6-dim output (Sprint 6)

**Files to Modify**:
1. `alpha_zero/core/civ6_network.py` (~20 LOC)
   - Change value head output layer: `nn.Linear(value_head_dim, 1)` → `nn.Linear(value_head_dim, 6)`

2. `alpha_zero/cloud/gcp/training_worker.py` (~30 LOC)
   - Update forward pass: `value.squeeze()` → `value[batch_indices, current_player_indices]`
   - Update dataset loading: Add `current_players` tensor to `TensorDataset`

3. `alpha_zero/cloud/gcp/selfplay_worker.py` (~10 LOC)
   - Store `current_player` in trajectory dict

4. `tests/unit/test_network.py` (~40 LOC)
   - Verify output shape: `(batch_size, 6)`

5. `tests/unit/test_training_loss.py` (~60 LOC, NEW)
   - Verify value loss computation for 4-player game
   - Test indexing logic with different `current_player` values

**Expected LOC**: 160 LOC total

**Testing**:
```bash
pytest tests/unit/test_network.py::test_value_head_6dim -v
pytest tests/unit/test_training_loss.py::test_4player_value_loss -v
pytest tests/integration/test_nplayer_pipeline.py -v
```

### Risks & Mitigations

**Risk 1**: Backward compatibility (existing models output scalar, new models output vector)
- **Mitigation**: Migration script replicates scalar → all 6 dims (broadcast)
- **Example**: Old model outputs `value=0.8` → New model gets `[0.8, 0.8, 0.8, 0.8, 0.8, 0.8]`

**Risk 2**: Target assignment complexity (how to assign values in elimination scenarios?)
- **Mitigation**: Use final outcome only (winner/losers), ignore intermediate eliminations for now
- **Example**: Winner gets `+1`, all losers get `-1` (simplified, can refine later)

**Risk 3**: MCTS integration (GumbelMCTS currently uses scalar value for tree search)
- **Mitigation**: Phase 3 task to extend MCTS to handle per-player values
- **Short-term workaround**: Use `value[current_player]` for MCTS backup (treat as scalar)

---

## Implementation Order

**Phase 2** must address G8 and G9 in sequence (not parallel, due to dependency on checkpoint format).

### Task 2.1: Encoder Extension (28 channels) - FIRST
**Duration**: 1 week
**Deliverables**:
- 28-channel encoder with per-player unit/city tracking
- Migration script for 13→28 channel padding
- Unit tests for 2-player, 4-player, 6-player
- Backward compatibility verification

**Blocking**: Must complete before Task 2.2 (value head) because network architecture changes affect checkpoint format.

### Task 2.2: Value Head Extension (6-dim output) - SECOND
**Duration**: 1 week
**Deliverables**:
- 6-dimensional value head output
- Per-player value loss computation
- Trajectory storage of `current_player`
- Integration tests for 4-player pipeline

**Dependency**: Requires Task 2.1 checkpoint format (28-channel models) to be established.

### Task 2.3: Model Migration - THIRD
**Duration**: 3-5 days
**Deliverables**:
- Migrate v218 model: 13→28 channels + scalar→6-dim value
- 10-iteration fine-tuning run (~$0.50 cost)
- ELO evaluation vs v218 baseline (expect within 50 points)

**Dependency**: Requires both Task 2.1 and Task 2.2 to be complete.

---

## Testing Strategy

### Unit Tests (Per-Gap)

**G8 (Encoder)**:
```python
def test_encoder_4player_distinguishes_opponents():
    """Verify 4-player game produces distinct opponent channels."""
    encoder = MicroCivEncoder(board_size=8, num_players=4)
    state = create_4player_state(...)

    spatial = encoder.encode(state, current_player=0)

    # Own units (Player 0) → channels 3-5
    assert spatial[3:6, ...].sum() > 0

    # Opponent 1 (Player 1) → channels 6-8
    assert spatial[6:9, ...].sum() > 0

    # Opponent 2 (Player 2) → channels 9-11
    assert spatial[9:12, ...].sum() > 0

    # Opponent 3 (Player 3) → channels 12-14
    assert spatial[12:15, ...].sum() > 0
```

**G9 (Value Head)**:
```python
def test_value_head_6dim_output():
    """Verify value head outputs 6-dimensional vector."""
    network = Civ6TransformerNetwork(...)
    batch_size = 8

    spatial = torch.randn(batch_size, 28, 8, 8)
    vector = torch.randn(batch_size, 64)
    action_mask = torch.ones(batch_size, 376)

    policy_logits, value = network(spatial, vector, action_mask)

    assert value.shape == (batch_size, 6)
```

### Integration Tests (End-to-End)

**4-Player Pipeline**:
```python
def test_4player_selfplay_training_cycle():
    """Verify 4-player game can be played, encoded, and trained."""
    env = MicroCivEnv(num_players=4, board_size=8)
    network = Civ6TransformerNetwork(spatial_channels=28, ...)
    mcts = GumbelMCTS(network, config=...)

    # Play 1 game
    trajectories = play_game(env, mcts)

    # Verify trajectory structure
    assert all("current_player" in t for t in trajectories)
    assert all(t["spatial"].shape == (28, 8, 8) for t in trajectories)

    # Train on trajectories
    model, losses = train_on_trajectories(network, trajectories)

    # Verify training ran without errors
    assert losses["policy_loss"] < 10.0
    assert losses["value_loss"] < 10.0
```

### Backward Compatibility Tests

**2-Player Behavior**:
```python
def test_2player_backward_compat():
    """Verify 2-player games produce same channels 0-12 as old encoder."""
    old_encoder = MicroCivEncoder_v13(board_size=8)
    new_encoder = MicroCivEncoder_v28(board_size=8, num_players=2)

    state = create_2player_state(...)

    old_spatial = old_encoder.encode(state, current_player=0)  # shape: (13, 8, 8)
    new_spatial = new_encoder.encode(state, current_player=0)  # shape: (28, 8, 8)

    # First 13 channels should match
    assert torch.allclose(old_spatial, new_spatial[:13, :, :])

    # Channels 13-27 should be zeros
    assert new_spatial[13:, :, :].sum() == 0.0
```

---

## Cost & Timeline Estimates

### Phase 2 (Encoder + Value Head Extensions)

**Task 2.1: Encoder (28 channels)**
- **Duration**: 1 week (5-7 days)
- **LOC**: 182 LOC
- **Cost**: $0 (local development + unit tests)

**Task 2.2: Value Head (6-dim)**
- **Duration**: 1 week (5-7 days)
- **LOC**: 160 LOC
- **Cost**: $0 (local development + unit tests)

**Task 2.3: Model Migration**
- **Duration**: 3-5 days
- **LOC**: 30 LOC (extend migration script)
- **Cost**: $0.50 (10-iteration fine-tuning on T4 GPU)

**Total Phase 2**: 2-3 weeks, 372 LOC, $0.50 cost

### Phase 3 (Cloud Deployment & Validation)

**Task 3.1: 20-iteration cloud benchmark**
- **Duration**: 1 day (mostly automated)
- **Cost**: $3-5 (20 iterations, 4 workers × 50 games, T4 spot)
- **Purpose**: Validate batch self-play + N-player encoder/value in production

**Task 3.2: ELO evaluation**
- **Duration**: 2 hours (GPU tournament)
- **Cost**: $0.50 (100 games, T4 GPU)
- **Purpose**: Verify migrated models are competitive with v218 baseline

**Total Phase 3**: 1-2 days, $3.50-5.50 cost

### Sprint 6 Total

**Duration**: 3-5 weeks
**LOC**: 372 LOC (code changes) + 400 LOC (tests) = 772 LOC
**Cost**: $4-6
**Outcome**: N-player MicroCiv fully functional (M1 complete)

---

## Success Criteria

### Phase 2 Complete When:
- ✅ Encoder outputs 28 channels with per-player unit tracking
- ✅ Value head outputs 6-dimensional vector
- ✅ All unit tests pass (2-player, 4-player, 6-player)
- ✅ Backward compatibility verified (2-player matches old encoder)
- ✅ Model migration script works (v218 → 28-channel + 6-dim value)

### M1 (N-Player MicroCiv) Complete When:
- ✅ 4-player selfplay generates valid trajectories
- ✅ Training converges on 4-player data (loss decreasing)
- ✅ Migrated models play competitive 4-player games (ELO within 50 points of baseline)
- ✅ No crashes or errors in 20-iteration cloud run

---

## Next Steps

**Immediate** (Phase 1 complete):
- ✅ M0 validation complete (loss converging, training works)
- ✅ Batch self-play enabled by default (3-5x speedup)
- ✅ N-player gap analysis documented (this document)

**Next Week** (Phase 2 Task 2.1):
- Extend encoder to 28 channels
- Write unit tests for 4-player and 6-player
- Verify backward compatibility (2-player)
- Create model migration script (13→28 channel padding)

**Following Week** (Phase 2 Task 2.2):
- Extend value head to 6-dim output
- Update training worker to use per-player value indexing
- Add `current_player` to trajectory storage
- Write integration tests for 4-player pipeline

**Week 3** (Phase 2 Task 2.3):
- Migrate v218 model (13→28 channels, scalar→6-dim value)
- Run 10-iteration fine-tuning
- Evaluate ELO vs v218 baseline

**Week 4** (Phase 3):
- Run 20-iteration production benchmark with batch enabled
- Analyze W&B metrics (throughput, cost, convergence)
- Document results and update roadmap

---

## References

### Related Documents
- `docs/M0_VALIDATION_RESULTS.md` - M0 convergence validation (Sprint 6 Phase 1)
- `docs/ROADMAP.md` - Full project roadmap (M0-M9 milestones)
- `C:\Users\iansh\.claude\plans\wobbly-hatching-hopcroft.md` - Sprint 6 implementation plan

### Key Files (Current Codebase)
- `alpha_zero/envs/microciv.py:628` - Turn rotation (G7)
- `alpha_zero/envs/civ6_encoder.py:322,377` - Encoder channel allocation (G8)
- `alpha_zero/core/civ6_network.py` - Value head architecture (G9)
- `alpha_zero/cloud/gcp/training_worker.py:437,444` - Value loss computation (G9)
- `alpha_zero/cloud/gcp/config.py` - Network architecture config (spatial_channels)

### Test Files (Existing)
- `tests/unit/test_encoder.py` - Encoder unit tests (extend for 4-player, 6-player)
- `tests/unit/test_network.py` - Network unit tests (add value_head_6dim test)
- `tests/integration/test_convergence_validation.py` - Convergence tests (M0 validation)

---

**Document Status**: COMPLETE ✅
**Approved**: Sprint 6 Phase 1 (2026-03-06)
**Next Review**: Phase 2 Task 2.1 completion (encoder extension)

---

**Authored by**: Claude Code (Sonnet 4.5)
**Sprint**: Sprint 6 - Phase 1
**Purpose**: Authoritative reference for N-player encoder and value head extensions
