# Civ6 AlphaZero — Next Steps (Post-Phase 13)

All infrastructure phases (1-13) are complete. The system can run multi-iteration training on Vertex AI with selfplay, training, evaluation, and post-run tournament. Docker v0.10.2 is deployed on GCR. 7 model versions (v0-v6) exist in GCS.

**Current state**: Models train but don't yet show meaningful improvement over random play (~25% win rate in evaluation). The priority is now making the agent actually learn.

---

## Phase 14: Extended Training & Learning Signal Validation

**Goal**: Run 20-50 iterations and verify that later model versions beat earlier ones.

**Why this matters**: The pipeline works end-to-end, but 3-iteration runs are too short to observe learning. We need to confirm the training signal propagates correctly through the selfplay → training → evaluation loop.

### Tasks

1. **Long training run** (20-50 iterations)
   ```bash
   python -m alpha_zero.cloud.gcp.vertex_orchestrator \
       --iterations=20 --workers=2 --games_per_worker=25 \
       --mcts_type=gumbel --mcts_simulations=16 --board_size=8 \
       --epochs=5 --max_turns=20 --evaluation_games=10 \
       --post_run_tournament --tournament_games=6 --tournament_mcts_sims=16
   ```
   Estimated cost: ~$2-4 (20 iterations x ~$0.10-0.20/iteration)

2. **Monitor W&B metrics**
   - `orchestrator/eval_win_rate` — should trend upward over iterations
   - `training/loss` — should decrease
   - `tournament/elo_*` — later versions should have higher ELO

3. **Diagnose if learning stalls**
   - Check policy entropy (too high = random, too low = collapsed)
   - Check value accuracy (predicted vs actual game outcomes)
   - Verify trajectory quality (are selfplay games diverse?)
   - Consider increasing games_per_worker or epochs

4. **Hyperparameter tuning** based on W&B data
   - Learning rate (currently default from TransformerConfig)
   - MCTS simulations (8-32 range)
   - Training epochs per iteration (3-10)
   - Temperature schedule for selfplay exploration

### Success Criteria
- Evaluation win rate > 55% for at least 3 consecutive iterations
- Post-run tournament shows monotonic ELO increase for later versions
- Training loss decreases over iterations

---

## Phase 15: Population-Based Training on Vertex AI

**Goal**: Run PBT with population_size=8 to automatically tune hyperparameters.

**Prerequisite**: Phase 14 shows that single-agent training works (models improve).

### Tasks

1. Launch PBT with `configs/pbt_prod.yaml`
2. Monitor W&B `pbt/elo_*`, `pbt/fitness_*`, `pbt/mutation_*` metrics
3. Validate baseline benchmarks at every 5th generation
4. Use dynamic evaluation strategy switch after generation 11

### Estimated Cost
- 8 agents x 10 generations x ~$0.20/iteration = ~$16

---

## Phase 16: Inference Optimization & Vectorized Self-Play

**Goal**: 10x+ self-play throughput via batch MCTS inference and vectorized environments.

### Tasks

1. **Batch MCTS inference**: Accumulate leaf nodes from multiple MCTS trees and evaluate in a single GPU batch
2. **Vectorized environments**: Run N games simultaneously using vectorized MicroCivEnv
3. **CUDA graph optimization**: Capture and replay GPU execution graphs for fixed-size batches
4. **Benchmark**: Target ~50 games/min vs current ~4.4 games/min (GumbelMCTS, 8 sims)

---

## Phase 17: MuZero Training Run

**Goal**: Validate learned dynamics model on MicroCiv.

### Tasks

1. Run MuZero PBT with `configs/muzero_pbt.yaml` (population_size=6)
2. Compare AlphaZero vs MuZero training modes on identical hyperparameters
3. Measure MuZero's advantage in search efficiency (no env cloning needed)

---

## Phase 18: DDP Test Rewrites & Full Type Safety

**Goal**: Fix remaining 12 DDP xfailed tests and achieve full mypy strict mode.

### Tasks

1. Rewrite DDP tests to use `torch.multiprocessing.spawn` instead of subprocess
2. Add `disallow_untyped_defs` for remaining packages (cloud, evaluation, rag)
3. Target: 0 xfails, 0 mypy errors

---

## Phase 19: Human vs AI Interface

**Goal**: Web-based UI for playing against the trained agent.

### Tasks

1. FastAPI backend serving the trained model
2. WebSocket game state streaming
3. React/HTMX frontend with MicroCiv board visualization
4. Move history and agent confidence display

---

## Phase 20: Full Civ6 Integration

**Goal**: Play against the real Civilization VI game.

### Tasks

1. Validate Lua mod with current Civ6 version
2. Scale model to 16x16 board with 84 spatial channels
3. Implement full 483K hierarchical action space
4. Bridge latency optimization

---

## Priority Order

```
Phase 14 (Learn!) ──→ Phase 15 (PBT) ──→ Phase 16 (Speed) ──→ Phase 17 (MuZero)
                                      ╰──→ Phase 18 (DDP/Types) ──→ Phase 19 (UI) ──→ Phase 20 (Civ6)
```

**Immediate focus**: Phase 14 — make the agent learn. Everything else depends on this.
