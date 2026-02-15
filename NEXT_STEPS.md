# Civ6 AlphaZero — Next Steps (Post-Phase 16)

All infrastructure and optimization phases (1-16) are complete. The system runs multi-iteration training on Vertex AI with selfplay, training, evaluation, post-run tournament, batch inference, and vectorized self-play. Docker v0.13.0 is deployed on GCR. 80 model versions (v0-v079) exist in GCS.

**Current state**: A 50-iteration training run (`indulgent-date-703`) is active on a Compute Engine VM orchestrator with value head stabilization (0.1x LR), warmup-cosine scheduling, and 25 evaluation games per iteration.

---

## Completed Phases (Summary)

| Phase | Description | Version | Key Result |
|-------|-------------|---------|------------|
| 1-7 | Core AlphaZero through Production PBT | v0.8.0 | Full cloud training pipeline |
| 8-10 | Safeguards, MuZero, Tournaments | v0.8.0 | Automated interventions + tournament system |
| 11 | Code Quality & Tech Debt | v0.9.0 | -14,600 LOC dead code, 16 xfail fixes |
| 13 | Multi-Iteration Training | v0.10.0 | 3-phase cloud pipeline validated |
| 14 | Learning Signal Validation | v0.11.0 | 20-iter run, $2.44, 2 promotions, policy loss 1.21→0.63 |
| 15 | Value Head Stabilization | v0.12.0 | Per-head LR multipliers (value=0.1x) |
| 16 | Batch Inference & Vectorized Self-Play | v0.13.0 | BatchGumbelMCTS, VectorizedMicroCivEnv, cloud orchestrator mode |

---

## Active: Monitor 50-Iteration Run (indulgent-date-703)

**Goal**: Validate value head stabilization and scaled training with Phase 16 infrastructure.

### Metrics to Watch

- `orchestrator/eval_win_rate` — should trend upward, target 3+ promotions
- `train/learning_rate_value_head` — confirm 10x lower than encoder, same schedule shape
- `train/value_correlation` — target >0.5 for 10+ consecutive iterations (was 0.07-0.91 in Phase 14)
- `tournament/elo_*` — post-run ELO spread target >100 points (was 66 in Phase 14)

### Success Criteria

- 3+ model promotions over 50 iterations
- Value correlation stabilization >0.5
- Post-run tournament ELO spread >100 points
- Total cost <$15

---

## Phase 17: Batch Self-Play Cloud Integration

**Goal**: Enable `--num_parallel_games` and `--max_batch_size` in cloud training for 5-10x throughput.

**Prerequisite**: Phase 16 code is merged (PR #50) and 50-iteration run validates stability.

### Tasks

1. **Rebuild container** with batch self-play enabled by default
2. **Benchmark cloud throughput**: Compare `--num_parallel_games=4 --max_batch_size=8` vs baseline
3. **Tune batch parameters**: Find optimal num_parallel_games for T4 GPU memory
4. **Run 50-iteration batch training run**: Measure wall-clock speedup and cost reduction

### Estimated Cost

- Similar per-iteration but ~3-5x fewer wall-clock minutes = faster feedback loops

---

## Phase 18: MuZero Training Run

**Goal**: Validate learned dynamics model on MicroCiv.

### Tasks

1. Run MuZero PBT with `configs/muzero_pbt.yaml` (population_size=6)
2. Compare AlphaZero vs MuZero training modes on identical hyperparameters
3. Measure MuZero's advantage in search efficiency (no env cloning needed)

### Estimated Cost

- 6 agents x 10 generations x ~$0.20/iteration = ~$12

---

## Phase 19: Population-Based Training on Vertex AI

**Goal**: Run PBT with population_size=8 to automatically tune hyperparameters.

**Prerequisite**: 50-iteration run shows meaningful model improvement.

### Tasks

1. Launch PBT with `configs/pbt_prod.yaml`
2. Monitor W&B `pbt/elo_*`, `pbt/fitness_*`, `pbt/mutation_*` metrics
3. Validate baseline benchmarks at every 5th generation
4. Use dynamic evaluation strategy switch after generation 11

### Estimated Cost

- 8 agents x 10 generations x ~$0.20/iteration = ~$16

---

## Phase 20: DDP Test Rewrites & Full Type Safety

**Goal**: Fix remaining 12 DDP xfailed tests and achieve full mypy strict mode.

### Tasks

1. Rewrite DDP tests to use `torch.multiprocessing.spawn` instead of subprocess
2. Add `disallow_untyped_defs` for remaining packages (cloud, evaluation, rag)
3. Target: 0 xfails, 0 mypy errors

---

## Phase 21: Human vs AI Interface

**Goal**: Web-based UI for playing against the trained agent.

### Tasks

1. FastAPI backend serving the trained model
2. WebSocket game state streaming
3. React/HTMX frontend with MicroCiv board visualization
4. Move history and agent confidence display

---

## Phase 22: Full Civ6 Integration

**Goal**: Play against the real Civilization VI game.

### Tasks

1. Validate Lua mod with current Civ6 version
2. Scale model to 16x16 board with 84 spatial channels
3. Implement full 483K hierarchical action space
4. Bridge latency optimization

---

## Priority Order

```
Monitor 50-iter ──→ Phase 17 (Batch Cloud) ──→ Phase 18 (MuZero) ──→ Phase 19 (PBT)
                                             ╰──→ Phase 20 (DDP/Types) ──→ Phase 21 (UI) ──→ Phase 22 (Civ6)
```

**Immediate focus**: Monitor `indulgent-date-703` and prepare batch self-play cloud integration.
