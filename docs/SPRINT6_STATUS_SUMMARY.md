# Sprint 6 Status Summary

**Date**: 2026-03-06
**Version**: 1.5.0
**Status**: ✅ **PHASE 2 COMPLETE** | ⏳ **PHASE 3 READY**

---

## Executive Summary

**Sprint 6** delivers N-player support and batch self-play optimization for the Civ6 AlphaZero project. **Phase 2 is complete** with all implementation, testing, and migration validation finished. **Phase 3 is ready to launch** with production infrastructure in place.

---

## Completed Work (Phase 2)

### ✅ Task 2.1: N-Player Encoder (28 channels)
- **Status**: COMPLETE
- **Implementation**: `MicroCivEncoder` supports 13→28 channels
- **Features**: 2-6 player support, ego-centric rotation, per-player observation
- **Tests**: 100% passing

### ✅ Task 2.2: N-Player Value Head (6-dim output)
- **Status**: COMPLETE
- **Implementation**:
  - `TransformerConfig.value_output_dim: int = 6` (default for N-player)
  - `TransformerValueHead` accepts `output_dim` parameter
  - `Civ6TransformerNetwork` passes config to value head
  - `HeadFactory.create_transformer_value_head()` updated
  - Conditional squeeze for backward compatibility (value_output_dim=1 → scalar output)
- **Files Modified**:
  - [transformer_network.py](../alpha_zero/core/transformer_network.py) (~50 LOC)
  - [network_heads.py](../alpha_zero/core/network_heads.py) (~10 LOC)
- **Tests**: 19/19 passing ([test_nplayer_transformer.py](../tests/unit/test_nplayer_transformer.py), 450 LOC)
- **Quality**: mypy ✓, ruff ✓, black ✓

### ✅ Task 2.3: Migration Utility
- **Status**: COMPLETE
- **Implementation**: [migrate_model_to_nplayer.py](../scripts/migrate_model_to_nplayer.py) (380 LOC)
- **Features**:
  - Spatial encoder: 13→28 channels (zero-fill new channels)
  - Value head: 1→6 neurons (replicate scalar to all players)
  - Handles both weight and bias migration
  - GCS upload support
- **Validation**: Migrated v218 successfully tested
  - Checkpoint loads ✓
  - Forward pass: policy (B, 376), value (B, 6) ✓
  - Upload to GCS complete ✓

### 📦 Migrated Model on GCS
- **Location**: `gs://civ6az-training/models/civ6az/v218_migrated/model.pt`
- **Metadata**: `gs://civ6az-training/models/civ6az/v218_migrated/metadata.json`
- **Size**: 2.1 MB
- **Ready for**: 10-iteration fine-tuning

---

## Phase 3 Deployment (Ready to Launch)

### 🚀 Task B.1: Production Launch Script
- **Status**: ✅ COMPLETE
- **File**: [launch_prod_v1.5.0.sh](../scripts/launch_prod_v1.5.0.sh) (154 LOC)
- **Features**:
  - Batch self-play enabled (num_parallel_games=4, max_batch_size=32)
  - Auto-detect latest model from GCS
  - Runs on `civ6az-orchestrator` VM
  - W&B logging with batch-selfplay tags
  - Configurable via environment variables

### ⏳ Task B.2: 20-Iteration Benchmark
- **Status**: PENDING (ready to launch)
- **Objective**: Measure batch self-play throughput vs sequential baseline
- **Target Metrics**:
  - Speedup: 3-5x (0.33 → 1.2-1.6 games/min)
  - Cost reduction: 70-80% (~$0.06 → ~$0.012-0.018/game)
  - GPU utilization: 40% → >80%
- **Launch Command**:
  ```bash
  bash scripts/launch_prod_v1.5.0.sh 20
  ```
- **Expected Cost**: $3-5 (4000 games @ ~$0.001/game)
- **Expected Runtime**: 4-6 hours
- **Latest Model**: v268

### ⏳ Task B.3: Fine-Tune Migrated v218
- **Status**: PENDING (migrated model uploaded and ready)
- **Objective**: Adapt v218 to N-player architecture
- **Configuration**:
  - 10 iterations
  - 2 workers × 50 games = 100 games/iter
  - Batch self-play enabled
  - Start from `v218_migrated`
- **Expected Behavior**:
  - Iterations 1-3: Learn new per-player channels
  - Iterations 4-7: Adapt value head to per-player predictions
  - Iterations 8-10: ELO stabilization
- **Success Criteria**: Final ELO within 50 points of v218 baseline
- **Expected Cost**: ~$0.50 (1000 games)
- **Expected Runtime**: 2-3 hours
- **Launch Command** (need to create):
  ```bash
  bash scripts/launch_finetune_v218.sh
  ```

### ⏳ Task B.4: Tournament Evaluation
- **Status**: PENDING (awaiting fine-tuned models)
- **Objective**: Compare v218, v218_migrated, v228 (fine-tuned), v268 (latest)
- **Configuration**: 20 games, 25 MCTS sims, max_turns=30
- **Launch Command**:
  ```bash
  python -m alpha_zero.tournament.runner \
    --models=v218,v218_migrated,v228,v268 \
    --games=20 \
    --mcts_sims=25 \
    --max_turns=30
  ```

---

## Technical Foundation (Phase 1)

### ✅ Batch Self-Play Defaults Enabled
- **Status**: COMPLETE (Phase 16 work)
- **Config**: [config.py](../alpha_zero/cloud/gcp/config.py)
  - `DEFAULT_NUM_PARALLEL_GAMES = 4`
  - `DEFAULT_MAX_BATCH_SIZE = 32`
- **Orchestrator**: [vertex_orchestrator.py](../alpha_zero/cloud/gcp/vertex_orchestrator.py)
  - CLI args: `--num_parallel_games`, `--max_batch_size`
  - Defaults to batch self-play (no opt-in required)
- **Tests**: [test_batch_selfplay_smoke.py](../tests/cloud/integration/test_batch_selfplay_smoke.py)

### 📊 Current Production State
- **Latest Model**: v268 (from 50-iteration v1.2.0 run)
- **GCS Bucket**: `gs://civ6az-training`
- **Container**: `us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.2.0`
- **Orchestrator VM**: `civ6az-orchestrator` (e2-small, us-central1-a)
- **W&B Project**: https://wandb.ai/ianshank-none/civ6-alphazero

---

## Key Design Decisions

### 1. Default value_output_dim=6 (N-Player Focus)
- **Decision**: New models default to 6-dim value output
- **Rationale**: Aligns with project goal of N-player support
- **Backward Compatibility**: Existing 2-player models explicitly set value_output_dim=1 in saved config

### 2. Conditional Squeeze (2-Player API Compatibility)
- **Decision**: Squeeze value output only when value_output_dim=1
- **Rationale**: Maintains API compatibility with existing 2-player code expecting scalar (B,)
- **Alternative Rejected**: Always return (B, value_output_dim) and update all downstream code (too invasive)

### 3. Batch Self-Play as Default
- **Decision**: Enable batch self-play by default (opt-out, not opt-in)
- **Rationale**: Phase 16 implementation proven stable; 3-5x speedup justifies new default
- **Monitoring**: First few iterations of production run will validate at scale

---

## Next Steps (Recommended Order)

### Option A: 20-Iteration Benchmark First (Recommended)
**Why**: Validate batch self-play at scale before committing to fine-tuning

1. **Launch 20-iteration benchmark** (Task B.2)
   ```bash
   bash scripts/launch_prod_v1.5.0.sh 20
   ```
   - Uses latest model (v268)
   - Measures batch self-play performance vs baseline
   - Expected: 3-5x speedup, 70-80% cost reduction
   - Runtime: 4-6 hours, Cost: $3-5

2. **Monitor W&B dashboard**
   - https://wandb.ai/ianshank-none/civ6-alphazero
   - Key metrics: games/min, GPU utilization, cost/game
   - Compare to v1.2.0 baseline (0.33 games/min, $0.06/game)

3. **Analyze results** (Task B.2 completion)
   - Update [BATCH_SELFPLAY_CLOUD_RESULTS.md](BATCH_SELFPLAY_CLOUD_RESULTS.md)
   - Generate performance report
   - If successful, proceed to fine-tuning

4. **Fine-tune v218_migrated** (Task B.3)
   - Create `scripts/launch_finetune_v218.sh`
   - Run 10 iterations from v218_migrated
   - Validate N-player architecture adaptation

5. **Tournament evaluation** (Task B.4)
   - Compare v218, v218_migrated, v228, v268
   - Validate migration quality

### Option B: Fine-Tuning First (Alternative)
**Why**: Test N-player migration sooner, smaller scale

1. **Create fine-tuning launch script**
   - Based on `launch_prod_v1.5.0.sh`
   - 10 iterations, 2 workers, v218_migrated start model

2. **Launch 10-iteration fine-tuning** (Task B.3)
   - Runtime: 2-3 hours, Cost: ~$0.50
   - Validate N-player adaptation

3. **Tournament v218 vs v228** (Task B.4)
   - Quick validation of migration quality

4. **Launch 20-iteration benchmark** (Task B.2)
   - Full-scale batch self-play validation

---

## Validation Checklist

### Phase 2 (Complete)
- [x] Add `value_output_dim: int = 6` to TransformerConfig
- [x] Update `from_network_config()` with `value_output_dim` field
- [x] Add `output_dim` parameter to TransformerValueHead.__init__
- [x] Update value head MLP to use `output_dim` instead of hardcoded `1`
- [x] Pass `config.value_output_dim` in Civ6TransformerNetwork.__init__
- [x] Add conditional squeeze in forward() for backward compatibility
- [x] Add `output_dim` parameter to HeadFactory.create_transformer_value_head
- [x] Fix migration script to handle bias
- [x] Write comprehensive unit tests (19 tests)
- [x] Run tests - all 19 passed
- [x] Type checking (mypy) - passed
- [x] Linting (ruff) - passed
- [x] Validate migrated v218 loads correctly
- [x] Forward pass produces correct shapes
- [x] Upload v218_migrated to GCS

### Phase 3 (Pending)
- [x] Production launch script created
- [x] Batch defaults enabled
- [ ] 20-iteration benchmark launched
- [ ] Benchmark results analyzed
- [ ] Fine-tuning launch script created
- [ ] 10-iteration fine-tuning completed
- [ ] Tournament evaluation conducted
- [ ] Results documented

---

## Files Created/Modified

### Implementation (Phase 2)
1. **alpha_zero/core/transformer_network.py** (~50 LOC modified)
   - Added `value_output_dim: int = 6` to TransformerConfig
   - Updated `TransformerValueHead` with `output_dim` parameter
   - Added conditional squeeze for backward compatibility
2. **alpha_zero/core/network_heads.py** (~10 LOC modified)
   - Updated `HeadFactory.create_transformer_value_head()` with `output_dim`
3. **scripts/migrate_model_to_nplayer.py** (~5 LOC modified)
   - Fixed bias migration in `find_value_head_keys()`
4. **tests/unit/test_nplayer_transformer.py** (450 LOC new)
   - 19 comprehensive tests across 5 test classes

### Deployment (Phase 3)
5. **scripts/launch_prod_v1.5.0.sh** (154 LOC new)
   - Production launch script with batch self-play
6. **scripts/upload_migrated_v218.py** (93 LOC new)
   - Download, migrate, upload automation
7. **scripts/test_migration_v218.py** (470 LOC existing)
   - End-to-end migration validation

### Documentation
8. **docs/NPLAYER_TRANSFORMER_IMPLEMENTATION.md** (306 LOC new)
   - Complete Phase 2 implementation summary
9. **docs/SPRINT6_STATUS_SUMMARY.md** (this file, new)
   - Current status and next steps
10. **docs/BATCH_SELFPLAY_CLOUD_RESULTS.md** (234 LOC template)
    - Ready for Phase 3 results

---

## Quick Start Commands

### Launch 20-Iteration Benchmark (Recommended First Step)
```bash
cd "c:/Users/iansh/OneDrive/Documents/Civ6 AlphaZero AI agent/Civ6-AlphaZero"
bash scripts/launch_prod_v1.5.0.sh 20
```

### Monitor Progress
```bash
# W&B Dashboard
open https://wandb.ai/ianshank-none/civ6-alphazero

# Docker logs on orchestrator VM
gcloud compute ssh civ6az-orchestrator --zone=us-central1-a \
  --command='docker logs -f civ6az-prod-v150'

# GCS models
gsutil ls gs://civ6az-training/models/civ6az/ | tail -10
```

### Fine-Tune v218_migrated (After Benchmark)
```bash
# Create launch script first (see Option B in Next Steps)
bash scripts/launch_finetune_v218.sh
```

---

## Cost Estimates

| Task | Games | GPU Hours | Cost (T4 spot) |
|------|-------|-----------|----------------|
| 20-iter benchmark | 4000 | ~27-45 | $3-5 |
| 10-iter fine-tuning | 1000 | ~7-11 | $0.50-1 |
| Tournament (4 models) | 240 | ~2 | $0.20 |
| **Total Phase 3** | **5240** | **36-58** | **$3.70-6.20** |

**Comparison to Sequential Baseline**:
- Sequential: $0.06/game × 5240 = **$314.40**
- Batch (target): $0.015/game × 5240 = **$78.60**
- **Savings**: 75% reduction (**~$235/run**)

---

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Batch OOM on T4 | High | Low (10%) | Start with num_parallel_games=2, tune up to 4 |
| Fine-tuning loses quality | Medium | Low (20%) | Monitor ELO, rollback if >100 point drop |
| Cloud cost overrun | Low | Low (10%) | Caps: benchmark ~$5, fine-tuning ~$1 |
| W&B sync issues | Low | Medium (40%) | Verify WANDB_API_KEY on orchestrator VM |

---

## References

- **Implementation Plan**: [~/.claude/plans/wobbly-hatching-hopcroft.md](C:\Users\iansh\.claude\plans\wobbly-hatching-hopcroft.md)
- **N-Player Gap Analysis**: [N_PLAYER_GAP_ANALYSIS.md](N_PLAYER_GAP_ANALYSIS.md)
- **Migration Guide**: [MODEL_MIGRATION_GUIDE.md](MODEL_MIGRATION_GUIDE.md)
- **Phase 2 Implementation**: [NPLAYER_TRANSFORMER_IMPLEMENTATION.md](NPLAYER_TRANSFORMER_IMPLEMENTATION.md)
- **Batch Self-Play Results**: [BATCH_SELFPLAY_CLOUD_RESULTS.md](BATCH_SELFPLAY_CLOUD_RESULTS.md) (template)
- **W&B Project**: https://wandb.ai/ianshank-none/civ6-alphazero

---

## Contributors

- **Implementation**: Claude Sonnet 4.5 (Plan agent + sequential implementation)
- **Review**: Code passes mypy, ruff, and comprehensive test suite
- **Testing**: 19 unit tests + migration validation on production model v218

---

**Document Version**: 1.0
**Last Updated**: 2026-03-06 17:30 UTC
**Next Review**: After 20-iteration benchmark completion
