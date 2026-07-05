## Phase 19: MLflow Monitoring Layer (COMPLETE — 2026-05-21)

### Delivered
- `alpha_zero/monitoring/` package: MLflowLogger, MLflowMonitoringConfig, AlphaZeroMLflowEvaluator
- `alpha_zero/utils/env_loader.py`: zero-dependency .env loading
- 5 evaluation/analysis scripts in scripts/
- 81 new tests (217 total passing)
- SQLite MLflow migration tool
- P0 bug fixes: EvaluationPipeline API, EvaluationResult type access

### Next Steps (Phase 20)
1. **MLflow Dashboard Integration** — Wire analyse_training_run.py output into MLflow custom charts
2. **Automated Regression Suite** — Schedule post_run_evaluation.py as CI step on model checkpoint
3. **Coverage to 90%** — Add integration tests for monitoring ↔ evaluation pipeline
4. **ELO Dashboard** — Log ELO time-series to MLflow; create heatmap of model matchup results
5. **GCS Model Registry** — Replace manual version tracking with MLflow Model Registry
6. **Training Alert System** — Use log_evaluation_result() to trigger Slack/email alerts on regression

---

# Civ6 AlphaZero — Next Steps (Post v1.3.0)

## Current State

The project is a **153K-line codebase** (309 Python files, 7,380+ tests) that has progressed
far beyond its original AlphaZero-for-MicroCiv scope. The system now includes:

- **Core AlphaZero**: MicroCiv env, ResNet + Transformer networks, 4 MCTS variants (Simple, Civ6, Gumbel, Batch)
- **MuZero**: Learned dynamics model, MuZero trainer, MuZero vs AlphaZero benchmark
- **Multiplayer (N-player)**: Max-N/Paranoid/Best-Reply MCTS, DUCT for simultaneous actions, opponent modeling, N-player value heads, turn management, scoring
- **Cloud Infrastructure**: Vertex AI deployment, GCS storage, Docker containers, Spot VM preemption, cost estimation
- **Optimization**: BatchGumbelMCTS, VectorizedMicroCivEnv, hex crystalline encoder
- **PBT**: Population-based training with mutation operators, fitness evaluation, W&B tracking (10 files)
- **Tournament**: ELO-based tournament system, round-robin evaluation (9 files)
- **Web Interface**: FastAPI + WebSocket game UI (12 files)
- **Civ6 Bridge**: Lua mod file IPC for real Civ6 connection (7 files)
- **Hierarchical Planning**: Goal decomposition, sub-goal MCTS (19 files)
- **RAG/LLM**: Strategy advisor, knowledge base, ChromaDB integration

### Production History

| Run | Version | Iterations | Result |
|-----|---------|-----------|--------|
| `dashing-firebrand-1393` | v1.2.0 | 50 | **0 model promotions** (bug: trained from scratch each iter) |
| — | v1.3.0 | — | Fix deployed: `--model_version` loads base weights. **Not yet validated.** |

### Critical Issue: Training Convergence Unproven

**v1.3.0** fixed the training-from-scratch bug, but the fix has not been run in production.
The v1.2.0 run wasted ~$28.55 producing models that could never beat the champion because
each iteration started from random weights instead of fine-tuning the incumbent.

---

## Completed Phases (Summary)

| Phase | Description | Version | Key Artifact |
|-------|-------------|---------|------------|
| 1–7 | Core AlphaZero through cloud deployment | v0.8.0 | Full Vertex AI pipeline |
| 8–10 | Safeguards, MuZero, Tournaments | v0.8.0 | `alpha_zero/tournament/`, MuZero network |
| 11 | Code Quality & Tech Debt | v0.9.0 | −14,600 LOC dead code |
| 13 | Multi-Iteration Training | v0.10.0 | 3-phase cloud pipeline |
| 14 | Learning Signal Validation | v0.11.0 | 20-iter run, 2 promotions, loss 1.21→0.63 |
| 15 | Value Head Stabilization | v0.12.0 | Per-head LR multipliers |
| 16 | Batch Inference & Vectorized Self-Play | v0.13.0 | BatchGumbelMCTS, VectorizedMicroCivEnv |
| 17a | Hex Crystalline Encoder | v0.14.1 | HexSpatialEncoder, EncoderRegistry |
| 17b | Training Readiness & Production Launch | v1.2.0 | action_mask, W&B groups, prod run |
| 18 | Training Worker Fine-Tuning Fix | v1.3.0 | `--model_version` loads incumbent weights |
| — | Multiplayer Module | — | N-player MCTS, DUCT, turn manager (18 files) |
| — | PBT Module | — | Population-based training (10 files) |
| — | Web Interface | — | FastAPI + WebSocket (12 files) |
| — | Civ6 Bridge | — | Lua mod IPC (7 files) |
| — | Hierarchical Planning | — | Goal decomposition (19 files) |
| — | DUCT / Simultaneous Actions | — | SimultaneousMCTS, ActionResolver |

---

## Immediate Priority: Validate Training Convergence

**This must succeed before any other work makes sense.**

### What to Validate

1. **Weight continuity**: Each iteration fine-tunes the champion, not random init
2. **Loss decrease**: Policy loss trends downward across iterations
3. **Model promotions**: At least some trained models beat the incumbent (>55% win rate)
4. **Win rate improvement**: Network improves against random player

### How to Validate

```bash
# Local validation (no cloud needed)
python scripts/validate_v130_training.py --iterations=5 --games=10

# Cloud validation (requires GCP)
python -m infra.vertex_launcher --mode=iteration --iteration=0 --wait
```

### Success Criteria

- Local validation script exits 0
- Policy loss at iteration 5 < iteration 1
- Win rate vs random at iteration 5 > iteration 1
- At least 1 model promotion in a cloud run

---

## Near-Term Roadmap

### Phase 19: Batch Self-Play Cloud Integration

**Goal**: Wire `BatchGumbelMCTS` + `VectorizedMicroCivEnv` into cloud containers for
5-10x throughput improvement.

**Status**: Local implementation complete. Cloud wiring needed.

**Tasks**:
1. Add `--num_parallel_games` and `--max_batch_size` to `selfplay_worker.py`
2. Wire through `vertex_orchestrator.py` → `vertex_jobs.py`
3. Benchmark T4 GPU memory with batched self-play
4. Run 50-iteration batch training to measure wall-clock speedup

**Prerequisite**: Training convergence validated (Priority 0).

**Key Files**:
- `alpha_zero/cloud/gcp/selfplay_worker.py`
- `alpha_zero/core/batch_mcts.py` (exists, 580 lines)
- `alpha_zero/envs/vectorized_env.py` (exists, 350 lines)

---

### Phase 20: N-Player MicroCiv Training — G9 only (G7/G8 DONE, corrected 2026-07-05)

**Goal**: Enable 3-8 player training using the already-implemented multiplayer module.

**Status (corrected)**: G7 and G8 are **already done** — this section was stale.
`envs/microciv.py` parameterizes `num_players` (2–6) with N-player turn rotation, victory
and start placement; `envs/civ6_encoder.py` allocates dynamic channels `3 + 4·N + 2`. Only
**G9** (per-player value threading in the training pipeline) remains. See
`docs/NEXT_STEPS_ANALYSIS_2026-07.md`.

**Tasks**:
1. ~~G7: N-player MicroCiv~~ — **DONE** (`microciv.py`, `num_players` 2–6)
2. ~~G8: N-player encoder~~ — **DONE** (`civ6_encoder.py`, dynamic channels)
3. **G9: Training pipeline** — `TrainingExample.value` → per-player values,
   integrate `compute_nplayer_value_loss()` (already exists in `multiplayer/network_heads.py`)
4. Train 3-4 player MicroCiv agents

**Key Files**:
- `alpha_zero/envs/microciv.py` — hardcoded `1 - self.current_player`
- `alpha_zero/envs/civ6_encoder.py` — hardcoded 13 channels
- `alpha_zero/multiplayer/` — already complete

---

### Phase 21: DDP Test Fixes

**Goal**: Fix 12 xfailed DDP tests to increase confidence in distributed training.

**Tasks**:
1. Audit xfailed tests — categorize root causes
2. Rewrite using `torch.multiprocessing.spawn`
3. Mock DDP init where full process groups unavailable
4. Target: 0 xfails

---

## Phase 21: DDP Test Rewrites & Remaining Type Safety (origin/main)

**Goal**: Fix remaining 12 DDP xfailed tests and bring remaining ~98 mypy errors to zero.

> **Sprint 5 delivered**: 251 -> ~98 mypy errors (61% reduction) across 58 files.
> Zero-error packages: `envs/`, `evaluation/`, `web/`, `hierarchical/`, `pbt/`, `training_worker.py`, `training_civ6.py`.

**Tasks**:
1. Rewrite DDP tests to use `torch.multiprocessing.spawn` instead of subprocess
2. Add `disallow_untyped_defs` for remaining packages (`distributed`, `human_interface`, `rag`, `cloud` residual)
3. Target: 0 xfails, 0 mypy errors

---

## Medium-Term Roadmap

### Phase 22: MuZero PBT Training Run

**Goal**: Run MuZero with PBT on Vertex AI, compare against AlphaZero.

**Status**: MuZero network + trainer + PBT module all exist.

**Tasks**:
1. Run PBT with `configs/muzero_pbt.yaml`
2. Compare AlphaZero vs MuZero (ELO, loss, cost)
3. Measure search efficiency gain (no env cloning)

---

### Phase 23: Fog of War / Partial Observability (Gap G10)

**Goal**: Handle information asymmetry in multiplayer.

**Approach**: Start with determinization — sample consistent world states, run MCTS on each.

**Effort**: New module, ~500 lines + tests.

---

### Phase 24: Real Civ6 Integration (Gap G12)

**Goal**: Connect trained model to actual Civilization VI via Lua mod bridge.

**Status**: `alpha_zero/bridge/` module exists (7 files) with file-based IPC.

**Tasks**:
1. Validate Lua mod with current Civ6 version
2. Scale model to full 84-channel encoder
3. Implement full hierarchical action space

---

## Long-Term Vision

| Feature | Module Status | Effort |
|---------|---------------|--------|
| Population diversity training | PBT exists | Medium |
| Diplomatic reasoning (LLM) | RAG exists, diplomacy module basic | High |
| Human vs AI (polished) | Web UI exists (alpha) | Medium |
| Cross-game transfer learning | Not started | Very High |

---

## Priority Order

```
Validate v1.3.0 convergence (CRITICAL)
    ↓
Phase 19: Batch self-play in cloud
    ↓
Phase 20: N-player MicroCiv (G7/G8/G9)
    ↓
Phase 21: DDP test fixes ─────────────→ Phase 22: MuZero PBT
                                            ↓
                                       Phase 23: Fog of war
                                            ↓
                                       Phase 24: Full Civ6
```

---

*Updated: 2026-03-01 — partially superseded 2026-07-05 by `docs/NEXT_STEPS_ANALYSIS_2026-07.md`*
*Version at authoring: v1.3.0. Repo is now v1.7.0 (cloud deploy, full-Civ6 local gameplay, LLM*
*eval harness, chess/LM-Studio direction). Some priorities below are stale — see the analysis doc.*
