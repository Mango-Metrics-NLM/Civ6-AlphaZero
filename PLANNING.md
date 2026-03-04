# Civ6 AlphaZero — Project Plan

> **Active Sprint**: Sprint 3 — Scaling & Integration
> **Branch**: `feature/sprint2-scaling-optimization` (PR pending)

## Current State

| Area | Status | Detail |
|------|--------|--------|
| Training | ✅ Complete | 49 iters, 219 model versions, v1.2.0 |
| Cloud Build | ✅ Passing | 9/9 steps, v15-stabilized + v1.2.0 |
| Architecture fix | ✅ Merged | cli.py + upload_initial_model.py aligned |
| Lint | ✅ Clean | 0 violations (E501 in ignore list) |
| Secrets | ✅ Migrated | WANDB_API_KEY → Secret Manager |
| GCP_INFRA.md | ✅ Created | Full infra documentation |
| Orchestrator | ✅ M10 | run_iterations() + CLI orchestrate |
| PBT | ✅ M11 | Full module (97 tests) |
| Monitoring | ✅ Scripts | 4 alerting policies |

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

## Sprint 2 — COMPLETE ✅

| Milestone | Status | Detail |
|-----------|--------|--------|
| Phase 1: Lint | ✅ | 5 ruff fixes, 2 mypy fixes |
| Phase 2: M10 Orchestrator | ✅ | run_iterations() + GCS state + CLI |
| Phase 3: M11 PBT | ✅ | Module pre-existed (97 tests) |
| Phase 4: Monitoring | ✅ | setup_monitoring.sh (4 alerts) |
| Phase 5: Secrets | ✅ | setup_secrets.sh + SM fallback |

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
