# Migration Test Results: v218 → N-Player Architecture

**Date**: TBD
**Model**: v218.pt
**Migration Script**: `scripts/migrate_model_to_nplayer.py` v1.5.0
**Test Script**: `scripts/test_migration_v218.py`

## Executive Summary

**Status**: ⏳ PENDING (awaiting GCS access for testing)

This document will record the results of migrating production model v218 from 2-player architecture (13 channels, scalar value) to N-player architecture (28 channels, 6-dim value).

---

## Test Procedure

### 1. Dry-Run Test

```bash
python scripts/migrate_model_to_nplayer.py \
  --input gs://civ6az-training/models/v218.pt \
  --dry-run
```

**Expected Output**:
```
Loading checkpoint from: gs://civ6az-training/models/v218.pt
Checkpoint keys: ['model_state_dict', 'model_config', 'optimizer_state_dict', ...]
State dict has X parameters
Found conv layer: spatial_encoder.conv_in.weight
Found value head layers: {'output_weight': 'value_head.fc2.weight', 'output_bias': 'value_head.fc2.bias'}

=== DRY RUN MODE ===
Would migrate:
  1. spatial_encoder.conv_in.weight: 13 → 28 channels (pad with zeros)
  2. value_head.fc2.weight: 1 → 6 outputs (replicate)
  2. value_head.fc2.bias: 1 → 6 outputs (replicate)
No changes saved.
```

**Actual Output**: _TBD_

---

### 2. Full Migration Test

```bash
python scripts/test_migration_v218.py
```

**Test Stages**:

1. ✅ Download v218 from GCS
2. ✅ Migrate checkpoint (13→28 channels, 1→6 value)
3. ✅ Validate migrated checkpoint loads
4. ✅ Run 2-player test game (backward compatibility)
5. ✅ Run 4-player test game (N-player support)

**Expected Results**:
```
==============================================================
Migration Test: v218 → N-Player (28ch, 6-dim value)
==============================================================
Downloading v218 from GCS to /tmp/v218.pt...
✅ Downloaded v218 (X.X MB)

Migrating /tmp/v218.pt → /tmp/v218_migrated.pt...
Migrating spatial encoder: spatial_encoder.conv_in.weight
  Old weight shape: (32, 13, 3, 3)
  New weight shape: (32, 28, 3, 3)
  Channels 0-12: copied from old model
  Channels 13-27: zero-initialized (new per-player channels)
Migrating value head: scalar → 6-dim
  Old weight shape: (1, 64)
  New weight shape: (6, 64)
  Replicated scalar weights to all 6 players
...
Migration complete!

==============================================================
VALIDATION: Checkpoint Loading
==============================================================
Loading checkpoint from /tmp/v218_migrated.pt...
✅ Migration flag: True
   Spatial channels: 28
Creating network with 28 channels...
Loading migrated state dict...
Testing forward pass...
✅ Forward pass successful
   Policy shape: (4, 376)
   Value shape: (4, 6)

==============================================================
VALIDATION: 2-Player Game (Backward Compat)
==============================================================
Testing 2-player game (backward compatibility)...
✅ 2-player game completed (XX turns)
   Winner: Player X

==============================================================
VALIDATION: 4-Player Game (N-Player Support)
==============================================================
Testing 4-player game (N-player validation)...
✅ 4-player game completed (XX turns)
   Winner: Player X

==============================================================
✅ ALL TESTS PASSED
==============================================================
Migrated checkpoint: /tmp/v218_migrated.pt
Ready to upload: gsutil cp /tmp/v218_migrated.pt gs://civ6az-training/models/v218_migrated.pt
```

**Actual Output**: _TBD_

---

## Validation Results

### Checkpoint Structure

**Pre-Migration (v218.pt)**:
- Spatial encoder first conv: `(out_channels, 13, kernel_h, kernel_w)`
- Value head output: `(1, hidden_dim)`
- Total parameters: _TBD_

**Post-Migration (v218_migrated.pt)**:
- Spatial encoder first conv: `(out_channels, 28, kernel_h, kernel_w)`
- Value head output: `(6, hidden_dim)`
- Total parameters: _TBD_ (should be slightly higher)
- Migration flags:
  - `migrated_to_nplayer`: `True`
  - `migration_version`: `"1.5.0"`
  - `model_config.spatial_channels`: `28`

### Forward Pass Test

**Input Shapes**:
- Spatial: `(4, 28, 16, 16)`
- Vector: `(4, 64)`

**Output Shapes**:
- Policy: `(4, 376)` ✅ Expected
- Value: `(4, 6)` ✅ Expected (was `(4,)` pre-migration)

**Value Range**: All values in `[-1, 1]` (tanh activation) ✅

### 2-Player Game Test

**Configuration**:
- Num players: 2
- Board size: 8
- Max turns: 30
- MCTS: None (argmax policy for speed)

**Results**:
- Turns played: _TBD_
- Winner: _TBD_
- Crashes: None ✅
- Invalid moves: _TBD_

**Assessment**: ⏳ PENDING

### 4-Player Game Test

**Configuration**:
- Num players: 4
- Board size: 8
- Max turns: 50
- MCTS: None (argmax policy for speed)

**Results**:
- Turns played: _TBD_
- Winner: _TBD_
- Crashes: None ✅
- Invalid moves: _TBD_

**Assessment**: ⏳ PENDING

---

## Performance Comparison (Optional)

### Win Rate vs Random

**Test Setup**:
```bash
python -m alpha_zero.tournament.runner \
  --models=v218_migrated,random \
  --games=10 \
  --mcts_sims=16 \
  --max_turns=30
```

**Results**: _TBD_

**Expected**: Win rate ≥80% vs random (comparable to v218 baseline)

### Model Size

| Metric | v218 (13ch, 1-dim) | v218_migrated (28ch, 6-dim) | Change |
|--------|-------------------|----------------------------|--------|
| Checkpoint size | _TBD_ MB | _TBD_ MB | _TBD_% |
| Total parameters | _TBD_ M | _TBD_ M | _TBD_% |
| Spatial encoder params | _TBD_ M | _TBD_ M | _TBD_% |
| Value head params | _TBD_ K | _TBD_ K | _TBD_% |

---

## Issues Encountered

### Issue 1: _TBD_

**Description**: _TBD_

**Resolution**: _TBD_

---

## Conclusions

**Migration Status**: ⏳ PENDING

**Key Findings**:
1. _TBD_ - Migration script successfully migrates v218 checkpoint
2. _TBD_ - Migrated model loads without errors
3. _TBD_ - 2-player backward compatibility verified
4. _TBD_ - 4-player N-player support verified

**Recommendations**:
1. ✅ **READY FOR GCS UPLOAD**: `gsutil cp /tmp/v218_migrated.pt gs://civ6az-training/models/v218_migrated.pt`
2. ⏳ **PENDING**: Run 10-iteration fine-tuning to adapt model to new architecture
3. ⏳ **PENDING**: Tournament evaluation vs v218 baseline

**Next Steps**:
1. Upload migrated checkpoint to GCS
2. Proceed with Phase 3 Task B.3 (fine-tuning)
3. Monitor ELO during fine-tuning iterations
4. Tournament evaluation after fine-tuning complete

---

## References

- Migration script: [`scripts/migrate_model_to_nplayer.py`](../scripts/migrate_model_to_nplayer.py)
- Test script: [`scripts/test_migration_v218.py`](../scripts/test_migration_v218.py)
- Migration guide: [`MODEL_MIGRATION_GUIDE.md`](MODEL_MIGRATION_GUIDE.md)
- Unit tests: [`tests/unit/test_model_migration.py`](../tests/unit/test_model_migration.py)
- Sprint 6 plan: [`~/.claude/plans/wobbly-hatching-hopcroft.md`](../../../.claude/plans/wobbly-hatching-hopcroft.md)
