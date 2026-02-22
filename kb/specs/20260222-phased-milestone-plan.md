# Phased Milestone Plan — Civ6 AlphaZero Next Steps

**Date:** 2026-02-22
**Status:** Planning
**Scope:** Post-Phase 17 roadmap covering production validation, multiplayer training, cloud scaling, and Civ6 integration

---

## Current State Summary

| Metric | Value |
|--------|-------|
| Completed Phases | 1–17 (all complete) |
| Core LOC | ~62,000 |
| Tests | 5,100+ (all passing) |
| Docker | v1.2.0 on Artifact Registry |
| GCS Models | 218+ versions (v0–v218+) |
| Active Run | `dashing-firebrand-1393` (50-iter, 4 workers × 50 games) |
| Modules | 22 subpackages under `alpha_zero/` |

### What's Implemented but Not Yet Integrated

| Module | Status | Gap |
|--------|--------|-----|
| `alpha_zero/multiplayer/` | 11 files, 175 tests passing | Not wired into training pipeline (G7–G9 open) |
| `alpha_zero/pbt/` | 8 files, core logic complete | Vertex AI orchestrator not deployed (P5) |
| `alpha_zero/human_interface/` | 6 files (controller, input, replay, visualizer) | Needs end-to-end testing |
| DDP Learner | Implemented | 12 xfail tests (deferred) |
| MuZero Bridge | Fully integrated | No cloud training run yet (P3) |
| Civ6 Lua Bridge | 245 bridge tests, 50 integration tests | No end-to-end game test (P4) |

---

## Milestone 18: Production Run Validation & Analysis

**Priority:** P0 (Immediate)
**Depends on:** Phase 17 production run completing
**Estimated effort:** Small

### Objective
Validate that the 50-iteration production run (`dashing-firebrand-1393`) produced meaningful learning signal and stable training dynamics.

### Deliverables

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 18.1 | Collect run metrics from W&B | Policy loss trend, value correlation, ELO spread exported |
| 18.2 | Post-run tournament | Round-robin of all 50 models (stride=5), ELO spread > 100 pts |
| 18.3 | Value head stability report | `value_correlation > 0.5` for 10+ consecutive iterations |
| 18.4 | Model promotion analysis | 3+ promotions over 50 iterations documented |
| 18.5 | Write analysis report | `kb/specs/20260222-phase18-analysis.md` with findings and recommendations |

### Key Questions to Answer
- Did GumbelMCTS + action masking resolve Phase 14's value instability?
- Is the current architecture (embed=64, layers=6) sufficient or should we scale up?
- What's the effective ELO ceiling with MicroCiv 16×16?

---

## Milestone 19: Batch Self-Play Cloud Integration

**Priority:** P1
**Depends on:** Milestone 18 (confirms training pipeline is healthy)
**Estimated effort:** Medium

### Objective
Enable `--num_parallel_games` in cloud self-play workers for 3–10× throughput improvement, leveraging the batch MCTS and vectorized environments from Phase 16.

### Deliverables

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 19.1 | Wire `VectorizedSelfPlay` into `selfplay_worker.py` | `--num_parallel_games N` flag accepted |
| 19.2 | Batch inference in cloud worker | Single GPU serves N parallel games via batch MCTS |
| 19.3 | Trajectory batching for GCS upload | N games flush trajectories concurrently without corruption |
| 19.4 | Preemption handling for batch workers | SIGTERM flushes all N in-progress trajectories |
| 19.5 | Benchmark: throughput vs single-game | Measure games/hour at N=1,4,8,16 on T4 and L4 |
| 19.6 | Update cloud configs and Makefile | New `--num_parallel_games` in orchestrator YAML |

### Architecture Notes
- `alpha_zero/optimizations/` already has `VectorizedEnv` and `BatchMCTS`
- Cloud worker needs to wrap these with GCS trajectory flushing
- Memory budget: T4 (16 GB) limits N to ~8; L4 (24 GB) supports ~16

---

## Milestone 20: Multiplayer Training Pipeline (Gaps G7–G9)

**Priority:** P1
**Depends on:** None (can start in parallel with M19)
**Estimated effort:** Large

### Objective
Extend the training pipeline from 2-player to N-player (2–8), integrating the existing `alpha_zero/multiplayer/` module into MicroCiv, the state encoder, and the training loop.

### Sub-milestones

#### 20a: MicroCiv N-Player Support (G7)

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 20a.1 | Extend `MicroCivEnv` for N players | `MicroCivEnv(num_players=4)` works |
| 20a.2 | N-player unit placement and turns | Each player has own units, cities, turn order |
| 20a.3 | N-player victory conditions | Elimination, score-based ranking, configurable |
| 20a.4 | Integration with `TurnManager` | Sequential and dynamic turn modes |
| 20a.5 | Tests | 50+ new tests for N-player MicroCiv |

#### 20b: State Encoder N-Player Channels (G8)

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 20b.1 | Dynamic channel allocation | 2p=13ch, 3p=17ch, 4p=21ch, 8p=37ch |
| 20b.2 | Player-relative encoding | Current player always channel group 0 |
| 20b.3 | Network input adaptation | `MicroCivNetwork` accepts variable channel count |
| 20b.4 | Backward compatibility | 2-player mode produces identical encoding to current |
| 20b.5 | Tests | Encoder tests for 2, 3, 4, 8 player configurations |

#### 20c: Training Pipeline N-Player Integration (G9)

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 20c.1 | N-player value targets in replay buffer | Value dicts instead of scalars stored |
| 20c.2 | N-player value head in training loss | Per-player MSE loss, properly masked |
| 20c.3 | N-player self-play actor | Player rotation, multi-policy support |
| 20c.4 | Scoring integration | `linear_ranking` / `winner_take_all` configurable |
| 20c.5 | 4-player MicroCiv training smoke test | `--num_players 4 --dry_run` completes |
| 20c.6 | Integration tests | End-to-end 4-player training for 5 iterations |

---

## Milestone 21: PBT on Vertex AI

**Priority:** P2
**Depends on:** Milestone 19 (batch self-play for faster PBT eval cycles)
**Estimated effort:** Medium

### Objective
Deploy Population-Based Training to Vertex AI, using the existing `alpha_zero/pbt/` module with cloud orchestration for automatic hyperparameter optimization.

### Deliverables

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 21.1 | `PBTOrchestrator` Vertex AI integration | Launches N training jobs per generation |
| 21.2 | Cross-agent ELO evaluation on cloud | All-vs-all tournament after each generation |
| 21.3 | Checkpoint copy for exploit step | Bottom agents load top agent weights from GCS |
| 21.4 | PBT W&B dashboard | Population fitness, hyperparameter distributions, lineage |
| 21.5 | CLI entry point | `python -m alpha_zero.cloud.gcp.pbt_orchestrator --config configs/pbt_config.yaml` |
| 21.6 | Budget guard | Max cost per generation, auto-pause if exceeded |
| 21.7 | Dry run smoke test | `--dry_run` validates config without launching jobs |

### Configuration (from Phase 6 spec)
```yaml
population_size: 4
generations: 10
games_per_eval: 20
mutation_rate: 0.2
hyperparameters:
  learning_rate: {min: 1e-5, max: 1e-2, init: log_uniform}
  mcts_c_puct: {min: 0.1, max: 5.0, init: uniform}
```

---

## Milestone 22: MuZero Cloud Training Run

**Priority:** P2
**Depends on:** Milestone 19 (batch self-play), Milestone 21 (PBT optional)
**Estimated effort:** Medium

### Objective
Execute a MuZero training run on Vertex AI to validate the learned dynamics model and compare against AlphaZero baseline.

### Deliverables

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 22.1 | MuZero cloud config | `configs/muzero_cloud.yaml` with dynamics model settings |
| 22.2 | MuZero self-play worker | Dynamics model inference in cloud worker |
| 22.3 | 20-iteration MuZero run | Completes on Vertex AI with metrics in W&B |
| 22.4 | AlphaZero vs MuZero tournament | Head-to-head ELO comparison (100+ games) |
| 22.5 | Analysis report | Dynamics model accuracy, value prediction quality, training cost comparison |

---

## Milestone 23: DDP Multi-GPU Training

**Priority:** P3
**Depends on:** Milestone 19
**Estimated effort:** Medium

### Objective
Fix the 12 deferred DDP xfail tests and enable multi-GPU distributed training for faster iteration on larger models.

### Deliverables

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 23.1 | Fix 12 DDP xfail tests | All pass without xfail marker |
| 23.2 | Multi-GPU gradient sync | Verified on 2× and 4× GPU configs |
| 23.3 | DDP cloud worker profile | `train_multi_gpu` machine profile in `CloudConfig` |
| 23.4 | Scaling efficiency benchmark | Linear speedup within 80% on 2× GPU |
| 23.5 | Integration test | DDP training for 5 iterations on 2 GPUs |

---

## Milestone 24: Partial Observability & Fog of War (G10)

**Priority:** P3
**Depends on:** Milestone 20 (N-player training)
**Estimated effort:** Large

### Objective
Implement information-set MCTS or determinized search for fog-of-war play, critical for realistic Civ6 multiplayer.

### Deliverables

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 24.1 | Determinized MCTS implementation | Sample N possible world states, search each |
| 24.2 | Observation model | Predict hidden state from visible information |
| 24.3 | Fog-of-war MicroCiv mode | `MicroCivEnv(fog_of_war=True)` hides enemy units |
| 24.4 | Training with partial observability | Agents learn to scout and handle uncertainty |
| 24.5 | Comparison: full vs partial observability | Win rate and ELO impact measured |

### Architecture Decision Needed
- **Information Set MCTS** — theoretically sound, expensive (sample many worlds)
- **Determinization** — simpler, known biases (strategy fusion)
- **Recommendation:** Start with determinization (3–5 samples), iterate if quality insufficient

---

## Milestone 25: Real Civ6 End-to-End Testing (P4)

**Priority:** P3
**Depends on:** None (can start anytime, independent)
**Estimated effort:** Medium

### Objective
Validate the Python bridge + Lua mod pipeline against an actual Civilization VI game instance.

### Deliverables

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 25.1 | Civ6 test environment setup | Documented setup with game install + mod loading |
| 25.2 | Bridge handshake test | Python ↔ Lua communication established |
| 25.3 | State extraction test | Full game state extracted from running Civ6 game |
| 25.4 | Action execution test | AI issues move/build/attack commands that execute in-game |
| 25.5 | 10-turn integration test | AI plays 10 turns in Civ6 without crashes |
| 25.6 | Setup documentation | Step-by-step guide for contributors |

---

## Milestone 26: Human vs AI Web Interface

**Priority:** P4
**Depends on:** Milestone 20 (multiplayer for N-player support)
**Estimated effort:** Medium

### Objective
Polish the web interface for human players to challenge trained AI models, with real-time game visualization.

### Deliverables

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 26.1 | Model selection UI | User picks from GCS model versions |
| 26.2 | Real-time game rendering | Board state updates live via WebSocket |
| 26.3 | Action input interface | Human selects unit, action type, target |
| 26.4 | Game replay viewer | Step through completed games |
| 26.5 | AI thinking visualization | Show MCTS visit counts and value estimates |
| 26.6 | End-to-end test | Human plays full game against AI in browser |

---

## Milestone 27: Simultaneous Actions & Diplomacy (G11, G13)

**Priority:** P4
**Depends on:** Milestone 20, Milestone 24
**Estimated effort:** Large

### Deliverables

| # | Task | Acceptance Criteria |
|---|------|-------------------|
| 27.1 | Simultaneous action resolution | Decoupled UCT or regret matching |
| 27.2 | Diplomatic action space | Propose/accept/reject peace, alliance, trade |
| 27.3 | Coalition formation model | LLM-assisted or learned diplomacy |
| 27.4 | Dynamic turn mode training | Train agents that handle mode switches |

---

## Dependency Graph

```
M18 (Run Analysis)
 │
 ├──→ M19 (Batch Self-Play) ──→ M21 (PBT Vertex) ──→ M22 (MuZero Run)
 │                            │
 │                            └──→ M23 (Multi-GPU DDP)
 │
 └──→ M20 (Multiplayer Training) ──→ M24 (Fog of War) ──→ M27 (Diplomacy)
                                  │
                                  └──→ M26 (Human vs AI)

M25 (Civ6 E2E Testing) — Independent, can start anytime
```

---

## Parallel Work Streams

### Stream A: Production Scaling (M18 → M19 → M21 → M22)
Focus: Maximize training throughput and validate production pipeline.

### Stream B: Multiplayer (M20 → M24 → M27)
Focus: Extend from 2-player to N-player with realistic game mechanics.

### Stream C: Integration & Polish (M25, M26)
Focus: Real-game testing and human-facing interfaces.

### Stream D: Infrastructure (M23)
Focus: Multi-GPU scaling for larger models.

---

## Role Assignments (KB Workflow)

Each milestone follows the Planning → DEV-SQE → Pre-PR pipeline:

| Milestone | Planning | DEV-SQE | Pre-PR |
|-----------|----------|---------|--------|
| M18 | Lock analysis criteria, define metrics thresholds | Run tournament, collect metrics, write report | Validate report, PR with analysis |
| M19 | Design batch worker architecture, memory budgets | Implement vectorized cloud worker, benchmark | Integration test, PR with benchmarks |
| M20 | Lock N-player API contracts, encoder channel spec | Implement G7/G8/G9, extend MicroCiv | Training smoke test, 4-player validation |
| M21 | Lock PBT cloud config, budget limits | Wire PBTOrchestrator to Vertex AI | Dry run validation, cost projection |
| M22 | Lock MuZero config, comparison metrics | Run MuZero on Vertex, collect results | Analysis report, model comparison |
| M23 | Lock DDP test requirements, GPU profiles | Fix xfail tests, benchmark scaling | Multi-GPU integration test |
| M24 | Lock ISMCTS vs determinization decision | Implement fog-of-war search, train agents | Observability comparison report |
| M25 | Lock Civ6 test environment requirements | Set up bridge test, run 10-turn game | Documentation, reproducibility check |
| M26 | Lock web UI requirements, model selection flow | Implement UI, WebSocket updates | End-to-end browser test |
| M27 | Lock diplomacy action space, coalition model | Implement simultaneous actions, diplomacy | Integration test with N-player games |

---

## Success Criteria (Global)

| Criterion | Target |
|-----------|--------|
| All tests pass | `pytest tests/ -v` exits 0 |
| Coverage | >= 80% on new modules |
| Type checking | `mypy alpha_zero/` clean |
| Lint | `ruff check alpha_zero/` clean |
| N-player training | 4-player MicroCiv agents with positive learning signal |
| Cloud PBT | At least 1 successful PBT run on Vertex AI |
| MuZero comparison | Head-to-head ELO measured vs AlphaZero |
| Production ELO | > 100 point spread across 50 models |

---

## Recommended Execution Order

1. **M18** — Analyze production run (blocks everything else with data)
2. **M19 + M20** — Start in parallel (batch self-play + multiplayer training)
3. **M25** — Start Civ6 E2E testing (independent, can run anytime)
4. **M21** — PBT on Vertex AI (after batch self-play proves throughput gains)
5. **M22 + M23** — MuZero run + DDP fixes (after PBT infrastructure ready)
6. **M24** — Fog of war (after multiplayer training baseline established)
7. **M26** — Human vs AI (after multiplayer + web interface polish)
8. **M27** — Diplomacy (final milestone, highest complexity)

---

*Generated: 2026-02-22*
*Source: KB specs, README roadmap, CLAUDE.md Section 13, multiplayer gap analysis*
