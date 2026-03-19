# Civ6 AlphaZero — Next Steps (Post-Phase 0)

All infrastructure, optimization, and training-readiness phases (1–17b) are complete. The system
runs multi-iteration training on Vertex AI with selfplay, training, evaluation, post-run tournament,
batch inference, and vectorized self-play. Docker **v1.2.0** is deployed on Artifact Registry.
**218+ model versions** (v0–v218+) exist in GCS.

**Current state**: The 50-iteration production run (`dashing-firebrand-1393`) is **COMPLETE**
(0 promotions — caused by the training-from-scratch bug, now fixed in v1.5.1). Phase 0 bug fix is
**COMPLETE**. Next priority is Phase 18 (Batch Self-Play Cloud Integration) with the fixed
training pipeline. W&B run: [ukxjr0uz](https://wandb.ai/ianshank-none/civ6-alphazero/runs/ukxjr0uz).

---

## Phase 0: Training-from-Scratch Bug Fix — COMPLETE (v1.5.1, 2026-03-18)

**Root cause**: `training_worker.py` always created fresh random-weight models, never loading the
champion from GCS. The orchestrator never forwarded `--start_model_version` to the training
container. Result: 50 iterations of production training with 0 promotions and $28.55 wasted.

**Fix**: `_load_champion_weights()` helper loads GCS champion weights before `model.to(device)`
and before optimizer construction. `build_training_extra_args()` now forwards
`--start_model_version={N}` when `model_version > 0`. 14 regression tests added in
`tests/unit/test_training_worker_resume.py`.

---

## Completed Phases (Summary)

| Phase | Description | Version | Key Result |
|-------|-------------|---------|------------|
| 1–7 | Core AlphaZero through Production PBT | v0.8.0 | Full cloud training pipeline |
| 8–10 | Safeguards, MuZero, Tournaments | v0.8.0 | Automated interventions + tournament system |
| 11 | Code Quality & Tech Debt | v0.9.0 | −14,600 LOC dead code, 16 xfail fixes |
| 13 | Multi-Iteration Training | v0.10.0 | 3-phase cloud pipeline validated |
| 14 | Learning Signal Validation | v0.11.0 | 20-iter run ($2.44), 2 promotions, policy loss 1.21→0.63 |
| 15 | Value Head Stabilization | v0.12.0 | Per-head LR multipliers (value=0.1×) |
| 16 | Batch Inference & Vectorized Self-Play | v0.13.0 | BatchGumbelMCTS, VectorizedMicroCivEnv, cloud orchestrator |
| 17a | Hex Crystalline Encoder & A/B Comparison | v0.14.1 | HexSpatialEncoder, EncoderRegistry, A/B runs |
| 17b | Training Readiness | **v1.2.0** | action_mask, W&B groups, gumbel default, eval retry, prod run |
| 0 | Training-from-Scratch Bug Fix | **v1.5.1** | Champion weights loaded before training; 14 regression tests |

---

## Phase 18: Batch Self-Play Cloud Integration

**Goal**: Enable `--num_parallel_games` and `--max_batch_size` in cloud training for 5–10×
throughput via `BatchGumbelMCTS` + `VectorizedMicroCivEnv`.

**Prerequisite**: Phase 0 training-from-scratch bug fix is now COMPLETE — training pipeline is
ready for a new production run.

### Tasks

1. **Rebuild container** with batch self-play enabled by default (`--num_parallel_games=4`)
2. **Benchmark cloud throughput**: Compare batched vs baseline games/min on T4 GPU
3. **Tune batch parameters**: Find optimal `num_parallel_games` for T4 memory (16 GB)
4. **Run 50-iteration batch training run**: Measure wall-clock speedup and cost reduction

### Estimated Cost

- Similar per-game cost but ~3–5× faster wall-clock → faster feedback loops

---

## Phase 19: MuZero Training Run

**Goal**: Validate the learned dynamics model on MicroCiv and compare against AlphaZero.

### Tasks

1. Run MuZero PBT with `configs/muzero_pbt.yaml` (population_size=6)
2. Compare AlphaZero vs MuZero on identical hyperparameters (ELO, policy loss, cost)
3. Measure MuZero's advantage in search efficiency (no env cloning needed)

### Estimated Cost

- 6 agents × 10 generations × ~$0.20/iteration ≈ $12

---

## Phase 20: Population-Based Training on Vertex AI

**Goal**: Run PBT with population_size=8 to automatically tune hyperparameters.

**Prerequisite**: Phase 18 batch self-play validates throughput improvements.

### Tasks

1. Launch PBT with `configs/pbt_prod.yaml`
2. Monitor W&B `pbt/elo_*`, `pbt/fitness_*`, `pbt/mutation_*` metrics
3. Validate baseline benchmarks at every 5th generation
4. Use dynamic evaluation strategy switch after generation 11

### Estimated Cost

- 8 agents × 10 generations × ~$0.20/iteration ≈ $16

---

## Phase 21: DDP Test Rewrites & Remaining Type Safety

**Goal**: Fix remaining 12 DDP xfailed tests and bring remaining ~98 mypy errors to zero.

> **Sprint 5 delivered**: 251→~98 mypy errors (61% reduction) across 58 files.
> Zero-error packages: `envs/`, `evaluation/`, `web/`, `hierarchical/`, `pbt/`, `training_worker.py`, `training_civ6.py`.

### Tasks

1. Rewrite DDP tests to use `torch.multiprocessing.spawn` instead of subprocess
2. Add `disallow_untyped_defs` for remaining packages (`distributed`, `human_interface`, `rag`, `cloud` residual)
3. Target: 0 xfails, 0 mypy errors

---

## Phase 22: Human vs AI Interface

**Goal**: Web-based UI for playing against the trained agent.

### Tasks

1. FastAPI backend serving the trained model
2. WebSocket game state streaming
3. React/HTMX frontend with MicroCiv board visualization
4. Move history and agent confidence display

---

## Phase 23: Full Civ6 Integration

**Goal**: Play against the real Civilization VI game.

### Tasks

1. Validate Lua mod with current Civ6 version
2. Scale model to 16×16 board with 84 spatial channels
3. Implement full 483K hierarchical action space
4. Bridge latency optimization

---

## Priority Order

```
Phase 0 DONE ──→ Phase 18 (Batch Cloud) ──→ Phase 19 (MuZero) ──→ Phase 20 (PBT)
                                         ╰──→ Phase 21 (DDP/Types) ──→ Phase 22 (UI) ──→ Phase 23 (Civ6)
```

**Immediate focus**: Phase 18 — Batch Self-Play Cloud Integration. Training pipeline is now
correct (Phase 0 bug fix complete). Next step is a new production run with champion model
fine-tuning working as intended.
