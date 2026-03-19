# Batch Self-Play Cloud Deployment Results

**Date**: TBD
**Version**: 1.5.0
**Sprint**: 6 Phase 3
**W&B Run**: TBD

## Executive Summary

**Status**: ⏳ PENDING (awaiting cloud deployment)

This document records the results of deploying batch self-play to production, measuring throughput improvements and cost savings compared to sequential self-play baseline.

**Key Metrics** (Target vs Actual):
- **Speedup**: 3-5x (target) vs _TBD_ (actual)
- **Cost per game**: ≤$0.015 (target) vs _TBD_ (actual)
- **GPU utilization**: >80% (target) vs _TBD_ (actual)

---

## Deployment Configuration

### Batch Self-Play Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_parallel_games` | 4 | Parallel games in batch |
| `max_batch_size` | 32 | Max inference batch size |
| MCTS type | gumbel | Fast MCTS variant |
| MCTS simulations | 16 | Per-move simulations |
| Max turns | 30 | Game length cap |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Iterations | 20 |
| Workers | 4 |
| Games per worker | 50 |
| **Total games** | **4000** (20 × 4 × 50) |
| Evaluation games | 30 |
| Win threshold | 0.55 |

### Infrastructure

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA T4 (16 GB VRAM) |
| Machine type | n1-standard-4 (4 vCPU, 15 GB RAM) |
| Spot instance | Yes (~$0.11/hour) |
| Region | us-central1 |
| Container | `us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.2.0` |

---

## Results

### Throughput Analysis

#### Wall-Clock Time

| Metric | Sequential (baseline) | Batch (v1.5.0) | Improvement |
|--------|----------------------|----------------|-------------|
| Time per iteration | _TBD_ min | _TBD_ min | _TBD_x |
| Total runtime (20 iters) | _TBD_ hours | _TBD_ hours | _TBD_x |
| Games per minute | ~0.33 | _TBD_ | _TBD_x |

**Assessment**: ⏳ PENDING

#### GPU Utilization

| Phase | GPU Util (%) | Memory (GB) | Notes |
|-------|-------------|-------------|-------|
| Self-play | _TBD_ | _TBD_ | Batched inference |
| Training | _TBD_ | _TBD_ | Standard training |
| Evaluation | _TBD_ | _TBD_ | Arena games |

**Assessment**: ⏳ PENDING

### Cost Analysis

#### Per-Game Cost

| Metric | Sequential | Batch | Savings |
|--------|-----------|-------|---------|
| GPU hours per 1000 games | _TBD_ | _TBD_ | _TBD_% |
| Cost per 1000 games | _TBD_ | _TBD_ | _TBD_% |
| **Cost per game** | **~$0.06** | **_TBD_** | **_TBD_%** |

**Target**: ≤$0.015/game (75% reduction)
**Actual**: _TBD_

#### Total Run Cost

| Component | Cost |
|-----------|------|
| Self-play (20 iters × 4 workers) | _TBD_ |
| Training (20 iters) | _TBD_ |
| Evaluation (20 × 30 games) | _TBD_ |
| **Total** | **_TBD_** |

**Budget**: $3-5
**Actual**: _TBD_

### Training Metrics

#### Model Promotions

| Iteration | Old Model | New Model | Win Rate | Promoted |
|-----------|-----------|-----------|----------|----------|
| _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Total Promotions**: _TBD_ (target: 1-2)

#### Loss Trends

| Metric | Initial | Final | Change |
|--------|---------|-------|--------|
| Policy loss | _TBD_ | _TBD_ | _TBD_ |
| Value loss | _TBD_ | _TBD_ | _TBD_ |
| Total loss | _TBD_ | _TBD_ | _TBD_ |

**Assessment**: ⏳ PENDING

---

## Performance Benchmarks

### Batch Size Scaling (Optional)

| Batch Size | Games/Min | GPU Util | Memory | Notes |
|-----------|-----------|----------|--------|-------|
| 1 (sequential) | 0.33 | ~40% | ~2 GB | Baseline |
| 2 | _TBD_ | _TBD_ | _TBD_ | |
| 4 | _TBD_ | _TBD_ | _TBD_ | **Production** |
| 8 | _TBD_ | _TBD_ | _TBD_ | If tested |

### Inference Latency

| Operation | Sequential | Batch (n=4) | Speedup |
|-----------|-----------|-------------|---------|
| Network forward | _TBD_ ms | _TBD_ ms | _TBD_x |
| MCTS search | _TBD_ ms | _TBD_ ms | _TBD_x |
| Full game | _TBD_ sec | _TBD_ sec | _TBD_x |

---

## Issues Encountered

### Issue 1: _TBD_

**Description**: _TBD_

**Impact**: _TBD_

**Resolution**: _TBD_

---

## W&B Dashboard

**Run URL**: https://wandb.ai/ianshank-none/civ6-alphazero/runs/_TBD_

**Key Plots**:
- Training loss (policy, value, total)
- Win rate progression
- Games per minute
- GPU utilization
- Cost per game

**Charts**: _TBD_ (embed screenshots or links)

---

## Comparison: Sequential vs Batch

### Performance Summary

| Metric | Sequential (v1.2.0) | Batch (v1.5.0) | Improvement |
|--------|-------------------|----------------|-------------|
| **Throughput** | 0.33 games/min | _TBD_ | **_TBD_x** |
| **Cost per game** | $0.06 | _TBD_ | **_TBD_%** |
| **GPU utilization** | 40% | _TBD_ | **_TBD_ pp** |
| **Model quality** | Baseline | _TBD_ | **_TBD_** |

### ROI Analysis

**Investment**:
- Development: ~4 weeks (Phase 16 + Sprint 6)
- Testing: ~2 days
- Deployment: ~1 hour

**Returns** (per 1000 games):
- Time saved: _TBD_ hours → **_TBD_x faster**
- Cost saved: _TBD_ → **$_TBD_ savings**

**Payback Period**: _TBD_ training runs (at 20 iterations/run)

---

## Conclusions

**Deployment Status**: ⏳ PENDING

**Key Findings**:
1. ⏳ PENDING - Batch self-play achieves _TBD_x speedup vs sequential
2. ⏳ PENDING - Cost per game reduced by _TBD_%
3. ⏳ PENDING - No quality degradation vs sequential baseline
4. ⏳ PENDING - GPU utilization improved from 40% to _TBD_%

**Recommendations**:
1. ⏳ PENDING - **Enable by default**: Batch self-play proven in production
2. ⏳ PENDING - **Optimal batch size**: _TBD_ games (balance throughput vs memory)
3. ⏳ PENDING - **Future optimization**: Test batch_size=8 if T4 memory allows

**Production Readiness**: ⏳ PENDING

**Next Steps**:
1. Monitor first 5 iterations closely for OOM errors or training issues
2. Compare model quality at iteration 20 vs v1.2.0 baseline
3. If successful, make batch self-play the default for all future runs
4. Document lessons learned and update training guide

---

## References

- Launch script: [`scripts/launch_prod_v1.5.0.sh`](../scripts/launch_prod_v1.5.0.sh)
- Orchestrator code: [`alpha_zero/cloud/gcp/vertex_orchestrator.py`](../alpha_zero/cloud/gcp/vertex_orchestrator.py)
- Batch MCTS implementation: [`alpha_zero/core/batch_mcts.py`](../alpha_zero/core/batch_mcts.py)
- Vectorized environment: [`alpha_zero/envs/vectorized_env.py`](../alpha_zero/envs/vectorized_env.py)
- Sprint 6 plan: [`~/.claude/plans/wobbly-hatching-hopcroft.md`](../../../.claude/plans/wobbly-hatching-hopcroft.md)
- W&B project: https://wandb.ai/ianshank-none/civ6-alphazero
