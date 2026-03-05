# Civ6 AlphaZero — Project Plan

> **Active Sprint**: Sprint 4 — Scalable Multi-Player & Cloud Optimization
> **Branch**: TBD (create from `main` after PR #68 merges)

## Current State

| Area | Status | Detail |
|------|--------|--------|
| Training | ✅ Complete | 49 iters, 219 model versions, v1.3.0 |
| Cloud Build | ✅ Passing | 9/9 steps, v15-stabilized + v1.2.0 |
| CI Pipeline | ✅ Passing | All jobs green (lint, test, security, cloud) |
| Architecture | ✅ Merged | cli.py + upload_initial_model.py aligned |
| Lint | ✅ Clean | 0 violations (ruff 0.8.0, mypy 1.13.0) |
| Secrets | ✅ Migrated | WANDB_API_KEY → Secret Manager |
| Orchestrator | ✅ M10 | run_iterations() + CLI orchestrate |
| PBT | ✅ M11 | Full module (97 tests) |
| Monitoring | ✅ Scripts | 4 alerting policies |
| N-Player | ✅ Sprint 3 | MicroCivEnv(num_players=2..6) |
| Batch Self-Play | ✅ Sprint 3 | BatchGumbelMCTS + VectorizedMicroCivEnv validated |

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

## Sprint 2 — COMPLETE ✅ (Merged PR #66)

| Milestone | Status | Detail |
|-----------|--------|--------|
| Phase 1: Lint | ✅ | 5 ruff fixes, 2 mypy fixes |
| Phase 2: M10 Orchestrator | ✅ | run_iterations() + GCS state + CLI |
| Phase 3: M11 PBT | ✅ | Module pre-existed (97 tests) |
| Phase 4: Monitoring | ✅ | setup_monitoring.sh (4 alerts) |
| Phase 5: Secrets | ✅ | setup_secrets.sh + SM fallback |
| CI Fix | ✅ | ruff 0.8.0 compat, JAX import guards, checkpoint assertion |

## Sprint 3 — COMPLETE ✅ (PR #68)

| Epic | Complexity | Status | Detail |
|------|------------|--------|--------|
| E1: CI/CD Hardening | S | ✅ | Pinned ruff/mypy, 2 lint fixes |
| E2: Convergence Validation | M | ✅ | validate_v130_training.py + 6 tests |
| E3: Batch Self-Play Wiring | L | ✅ | 29 integration tests (existing infra validated) |
| E4: DDP Test Stabilization | M | ✅ | 17 attr fixes, 5 xfails removed → 22/22 |
| E5: N-Player Foundation | L | ✅ | num_players=2..6, 51 tests |

## Sprint 4 — PLANNED 📋

| Epic | Complexity | Status | Detail |
|------|------------|--------|--------|
| E1: Batch Cloud Deploy | L | [ ] | Phase 18: Docker v1.3.0, `--num_parallel_games=4` default |
| E2: N-Player Network Heads | L | [ ] | Multi-head policy/value for N-player self-play |
| E3: Cloud Run Inference | M | [ ] | Deploy inference service to Cloud Run |
| E4: DDP Multinode Stabilization | M | [ ] | Fix remaining 7 xfail multinode tests |
| E5: Type Safety Expansion | S | [ ] | Add disallow_untyped_defs for cloud/eval packages |

## Docs

- [ADR: Scaling](docs/architecture/ADR-scaling-optimization.md)
- [PRD: M10 Orchestrator](docs/prd/prd-multi-iteration-orchestrator.md)
- [PRD: M11 PBT](docs/prd/prd-pbt-hyperparameter-tuning.md)
- [PRD: Batch Self-Play](docs/prd/prd-batch-selfplay.md)
- [PRD: N-Player MicroCiv](docs/prd/prd-nplayer-microciv.md)
- [GCP Infra](GCP_INFRA.md)
- [Next Steps Roadmap](NEXT_STEPS.md)
