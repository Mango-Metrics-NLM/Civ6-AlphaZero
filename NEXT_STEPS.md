# Civ6 AlphaZero — Next Steps (Post-Phase 17b)

All infrastructure, optimization, and training-readiness phases (1–17b) are complete. The system
runs multi-iteration training on Vertex AI with selfplay, training, evaluation, post-run tournament,
batch inference, and vectorized self-play. Docker **v1.2.0** is deployed on Artifact Registry.
**218+ model versions** (v0–v218+) exist in GCS.

**Current state**: A 50-iteration production run (`dashing-firebrand-1393`) is **ACTIVE** on
`civ6az-orchestrator` VM (e2-small, us-central1-a). 4 workers × 50 games = 200 games/iteration,
GumbelMCTS 16 sims, max_turns=30. W&B run:
[ukxjr0uz](https://wandb.ai/ianshank-none/civ6-alphazero/runs/ukxjr0uz).

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

---

## Active: Monitor 50-Iteration Production Run (`dashing-firebrand-1393`)

**Goal**: Validate training readiness improvements (action_mask, W&B groups, gumbel default) at
scale and generate a strong model for tournament evaluation.

### Metrics to Watch

- `orchestrator/eval_win_rate` — target ≥ 3 promotions over 50 iterations
- `train/policy_loss` — should trend downward (was 1.21→0.63 in Phase 14)
- `train/value_correlation` — target >0.5 for 10+ consecutive iterations
- `eval/win_rate` per W&B group — confirm action_mask improves promotion rate
- `tournament/elo_*` — post-run ELO spread target >100 points

### Success Criteria

- 3+ model promotions over 50 iterations
- Policy loss <0.5 by iteration 50
- Value correlation stabilization >0.5
- Post-run tournament ELO spread >100 points
- Total cost <$15

### Monitor Commands (run on `civ6az-orchestrator` VM)

```bash
docker logs -f civ6az-prod-v120 2>&1 | tail -50
```

---

## Phase 18: Batch Self-Play Cloud Integration

**Goal**: Enable `--num_parallel_games` and `--max_batch_size` in cloud training for 5–10×
throughput via `BatchGumbelMCTS` + `VectorizedMicroCivEnv`.

**Prerequisite**: Production run `dashing-firebrand-1393` completes successfully.

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

## Phase 21: DDP Test Rewrites & Full Type Safety

**Goal**: Fix remaining 12 DDP xfailed tests and achieve full mypy strict mode.

### Tasks

1. Rewrite DDP tests to use `torch.multiprocessing.spawn` instead of subprocess
2. Add `disallow_untyped_defs` for remaining packages (cloud, evaluation, rag)
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
Monitor prod run ──→ Phase 18 (Batch Cloud) ──→ Phase 19 (MuZero) ──→ Phase 20 (PBT)
                                              ╰──→ Phase 21 (DDP/Types) ──→ Phase 22 (UI) ──→ Phase 23 (Civ6)
```

**Immediate focus**: Monitor `dashing-firebrand-1393` production run (50 iterations on Vertex AI).
