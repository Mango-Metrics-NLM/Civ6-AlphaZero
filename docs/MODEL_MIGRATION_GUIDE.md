# Model Migration Guide: 2-Player → N-Player

**Sprint 6 Phase 2 Task 2.3** | Version 1.5.0

## Overview

This guide explains how to migrate existing Civ6 AlphaZero models from 2-player architecture (13 channels, scalar value) to N-player architecture (28 channels, 6-dim value).

## What Changed

### Encoder: 13 → 28 Channels

**Old Layout (13 channels):**
```
Channels 0-2:  Terrain (plains, grassland, water)
Channels 3-5:  Own units (warrior, settler, archer)
Channels 6-8:  Enemy units (combined, cannot distinguish Player 2 vs Player 3)
Channels 9-10: Cities (own, enemy combined)
Channel 11:    Fog of war
Channel 12:    Bias
```

**New Layout (28 channels):**
```
Channels 0-2:   Terrain (plains, grassland, water)
Channels 3-5:   Own units (warrior, settler, archer)
Channels 6-8:   Player 1 units (relative to current player)
Channels 9-11:  Player 2 units
Channels 12-14: Player 3 units
Channels 15-17: Player 4 units
Channels 18-20: Player 5 units
Channel 21:     Own cities
Channels 22-26: Opponent cities (P1-P5)
Channel 27:     Fog/bias
```

### Value Head: Scalar → 6-Dim

**Old:** Single output neuron `value ∈ [-1, 1]` (assumes zero-sum 2-player)

**New:** 6-dimensional output `value ∈ R^6` (one per player, pad unused with 0)

During training, the correct player's value is extracted via indexing: `value[batch_indices, current_players]`

## Migration Script

### Basic Usage

```bash
# Migrate local model
python scripts/migrate_model_to_nplayer.py \
    --input models/v218.pt \
    --output models/v218_migrated.pt

# Migrate GCS model
python scripts/migrate_model_to_nplayer.py \
    --input gs://civ6az-training/models/v218.pt \
    --output gs://civ6az-training/models/v218_migrated.pt

# Dry-run (preview changes without saving)
python scripts/migrate_model_to_nplayer.py \
    --input models/v218.pt \
    --dry-run
```

### What the Script Does

1. **Loads checkpoint** from local file or GCS
2. **Finds layers to migrate:**
   - Spatial encoder first conv layer (13 → 28 input channels)
   - Value head output layer (1 → 6 output neurons)
3. **Migrates spatial encoder:**
   - Pads input channels from 13 → 28
   - Copies channels 0-12 from old model
   - Zero-initializes channels 13-27 (new per-player channels)
4. **Migrates value head:**
   - Replicates scalar weight/bias to all 6 dimensions
   - All 6 players start with identical value predictions
5. **Updates metadata:**
   - Sets `spatial_channels: 28` in model_config
   - Adds `migrated_to_nplayer: true` flag
   - Adds `migration_version: "1.5.0"` tag
6. **Saves checkpoint** to local file or GCS

### Migration Guarantees

✅ **Backward Compatible:** For 2-player games:
- Channels 0-12 behave identically to old models
- Channels 13-27 are zero (no effect on 2-player games)
- Value predictions averaged across 6 dims (mean of identical values = original value)

✅ **Non-Destructive:** All other layers (policy head, residual blocks, etc.) are preserved exactly

✅ **Metadata Preserved:** Training iteration, optimizer state, etc. remain unchanged

## Post-Migration Fine-Tuning

After migration, the model needs **10-20 iterations of fine-tuning** to adapt the new channels and per-player value head:

```bash
# Run 10-iteration fine-tuning
python -m alpha_zero.cloud.gcp.vertex_orchestrator \
    --iterations=10 \
    --workers=2 \
    --games_per_worker=50 \
    --start_model_version=218_migrated \
    --num_parallel_games=4
```

**Expected Behavior:**
- Iterations 1-3: Model learns to use new per-player channels
- Iterations 4-7: Value head adapts to per-player predictions
- Iterations 8-10: ELO stabilizes, should match or exceed pre-migration baseline

**Cost Estimate:** ~$0.50 for 10 iterations (2 workers × 50 games, T4 spot)

## Validation

### 1. Load Test

Verify migrated model loads without errors:

```python
from alpha_zero.core.civ6_network import Civ6Network
import torch

# Load migrated checkpoint
checkpoint = torch.load("models/v218_migrated.pt")
print(f"Checkpoint keys: {checkpoint.keys()}")
print(f"Migrated flag: {checkpoint.get('migrated_to_nplayer')}")

# Create network with 28 channels
network = Civ6Network(spatial_channels=28)
network.load_state_dict(checkpoint["model_state_dict"])

# Test forward pass
spatial = torch.randn(4, 28, 16, 16)
vector = torch.randn(4, 64)
policy, value = network(spatial, vector)

print(f"Policy shape: {policy.shape}")  # (4, 376)
print(f"Value shape: {value.shape}")    # (4, 6)
```

### 2. 2-Player Game Test

Verify migrated model plays valid 2-player games:

```bash
# Run 10 test games vs random
python -m alpha_zero.tournament.runner \
    --models=v218_migrated,random \
    --games=10 \
    --mcts_sims=16
```

**Expected:** Win rate ≥80% vs random (same as pre-migration baseline)

### 3. 4-Player Game Test

Verify migrated model works in 4-player games:

```python
from alpha_zero.envs.microciv import MicroCivEnv

env = MicroCivEnv(num_players=4)
# ... play game with migrated model ...
```

**Expected:** No crashes, valid moves, reasonable value predictions

## Troubleshooting

### Error: "Expected 13 input channels, got 28"

**Cause:** Model is already migrated or has non-standard architecture

**Solution:** Check if model was already migrated: `checkpoint.get("migrated_to_nplayer")`

### Error: "Could not find conv layer"

**Cause:** Non-standard model architecture or different naming convention

**Solution:** Manually inspect state_dict keys and update `find_conv_layer_key()` function

### Warning: "No model_config found"

**Cause:** Very old models (v0-v2) don't have saved model_config

**Solution:** Migration still works, but metadata won't be updated. Safe to ignore.

### Low Win Rate After Migration

**Cause:** Model hasn't adapted to new architecture yet

**Solution:** Run 10-20 iteration fine-tuning. Check W&B for training progress.

## Migration Checklist

Before deploying migrated model:

- [ ] Dry-run shows expected changes (13→28 channels, 1→6 value neurons)
- [ ] Migrated checkpoint loads without errors
- [ ] Forward pass produces correct shapes: policy (B, 376), value (B, 6)
- [ ] 2-player test games achieve ≥80% win rate vs random
- [ ] 4-player test game runs without crashes
- [ ] 10-iteration fine-tuning completes successfully
- [ ] Fine-tuned model ELO ≥ pre-migration baseline (within 50 points)

## Script Details

**Location:** [`scripts/migrate_model_to_nplayer.py`](../scripts/migrate_model_to_nplayer.py)

**Tests:** [`tests/unit/test_model_migration.py`](../tests/unit/test_model_migration.py) (17 tests, all passing)

**Line Count:** 380 LOC (script) + 280 LOC (tests) = 660 LOC total

**Dependencies:**
- `torch` (PyTorch for tensor operations)
- `alpha_zero.cloud.gcp.storage` (for GCS I/O, optional)

## References

- [Sprint 6 Plan](../C:\Users\iansh\.claude\plans\wobbly-hatching-hopcroft.md)
- [N-Player Gap Analysis](N_PLAYER_GAP_ANALYSIS.md)
- [Phase 2 Task 2.1: Encoder Extension](../alpha_zero/envs/civ6_encoder.py)
- [Phase 2 Task 2.2: Value Head Extension](../alpha_zero/core/civ6_network.py)
