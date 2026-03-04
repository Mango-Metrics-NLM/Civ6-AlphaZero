# Civ6 AlphaZero — Project Plan

> **Active Sprint**: Sprint 2 — Scaling & Optimization
> **Branch**: `feature/sprint2-scaling-optimization`

## Current State

| Area | Status | Detail |
|------|--------|--------|
| Training | ✅ Complete | 49 iters, 219 model versions, v1.2.0 |
| Cloud Build | ✅ Passing | 9/9 steps, v15-stabilized + v1.2.0 |
| Architecture fix | ✅ Merged | cli.py + upload_initial_model.py aligned |
| Lint | ⚠️ 28 E501 | Line-too-long in alpha_zero/ |
| Secrets | ⚠️ Risk | WANDB_API_KEY in plain text in job specs |
| GCP_INFRA.md | ✅ Created | Full infra documentation |

## Sprint 1 — COMPLETE ✅ (Phases 0–7)

| Milestone | Status | Detail |
|-----------|--------|--------|
| M1: Code Quality | ✅ | 35 lint fixes, stale file cleanup |
| M2: DDP Stabilization | ✅ | DDPLearner init, 32 tests pass |
| M3: Cloud Workers | ✅ | Imports + timeouts verified |
| M4: Pre-Build | ✅ | 4/4 gates, 509 cloud tests |
| M5: Full Suite | ✅ | 3,795+ tests, 3 drift bugs fixed |
| M6: Container Build | ✅ | Build 32c5bbcc, 5m7s, 9/9 |
| M7: GCP Auth | ✅ | ADC + env vars + dry-run |
| M8: Training | ✅ | 49 iters, v219, T4 budget |
| M9: Validation | ✅ | GCP_INFRA.md, eval v219 vs v218 |

## Sprint 2 — IN PROGRESS (2 weeks)

### Phase 1: PR Prep & Code Quality (S) — Week 1

- Fix 28 E501 lint violations
- Hardcoded secret scan + remediation
- Pre-PR checklist (diff, tests, docs)

### Phase 2: Multi-Iteration Orchestrator M10 (L) — Week 1

- `vertex_orchestrator.run_iterations(start, count)`
- Evaluation gate (55% win rate)
- GCS-based preemption recovery
- `cli orchestrate` subcommand

### Phase 3: PBT Hyperparameter Tuning M11 (XL) — Week 2

- `alpha_zero/pbt/` module (config + tuner)
- 4-8 agent population, exploit/explore cycle
- Tune: LR, MCTS sims, temperature, c_puct

### Phase 4: Production Monitoring (M) — Week 2

- Cloud Monitoring alerting
- Cost ceiling ($50/day)
- Auto-restart patterns

### Phase 5: Secret Manager Migration (S) — Week 2

- WANDB_API_KEY → Secret Manager
- Remove plain-text keys from job specs

## Sprint 3 — PLANNED (future)

- 100-iteration scaling run
- CivRealm/Unciv game bridge
- Multi-GPU DDP training
- Model serving endpoint

## Docs

- [ADR: Scaling](docs/architecture/ADR-scaling-optimization.md)
- [PRD: M10 Orchestrator](docs/prd/prd-multi-iteration-orchestrator.md)
- [PRD: M11 PBT](docs/prd/prd-pbt-hyperparameter-tuning.md)
- [GCP Infra](GCP_INFRA.md)
