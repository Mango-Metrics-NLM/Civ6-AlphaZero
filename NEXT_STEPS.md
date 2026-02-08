# Civ6 AlphaZero — Comprehensive Next Steps Plan

Post-merge roadmap covering quality hardening, CI/CD maturation, cloud validation, training execution, and evaluation — leveraging all available agent skills, tools, and MCPs.

---

## Phase 1: Test Coverage Hardening (65% → 80%)

**Goal**: Raise `--cov-fail-under` from 65 to 80 using the `python-testing-patterns` skill.

**Rationale**: User rule requires 80% coverage across unit, integration, functional, e2e, user journey, security, and sanity tests. Current CI enforces only 65%.

### Changes

- **`pyproject.toml`** — Add `pytest-benchmark`, `pytest-xdist` to dev deps
- **`ci.yml`** — Raise `--cov-fail-under` to 80, add cloud unit + security jobs
- **New `tests/unit/test_pbt.py`** — PBT population lifecycle tests
- **New `tests/unit/test_tournament.py`** — ELO/matchmaking tests
- **New `tests/unit/test_hierarchical_mcts.py`** — 3-level hierarchy tests
- **New `tests/security/test_security_sanity.py`** — Credential leak checks
- **New `tests/sanity/test_import_sanity.py`** — Module importability checks

---

## Phase 2: CI/CD Pipeline Enhancements

**Goal**: Mature CI with parallel jobs, cloud test isolation, and release automation (`changelog-automation` skill).

### Changes

- **`ci.yml`** — Add `cloud-tests`, `security` jobs; add Python 3.12 to matrix
- **New `.github/workflows/release.yml`** — Auto changelog + GitHub Release on `v*.*.*` tags
- **`CHANGELOG.md`** — Add `[Unreleased]` section for PRs #29, #31, #34

---

## Phase 3: GCP E2E Cloud Validation

**Goal**: Validate GCS storage + Vertex AI jobs against live GCP. Uses `ml-pipeline-workflow` skill.

### Changes

- **New `tests/e2e/test_gcs_live.py`** — Real GCS upload/download roundtrip
- **New `tests/e2e/test_vertex_submission.py`** — Dry-run job spec validation
- **`alpha_zero/cloud/gcp/cli.py`** — Add `validate` subcommand for pre-flight checks
- **New `.agent/workflows/gcp-validate.md`** — GCP auth + quota validation workflow

---

## Phase 4: Local Training Smoke Test

**Goal**: Validate self-play → training → eval locally. Uses `systematic-debugging` + `ml-pipeline-workflow` skills.

### Changes

- **New `tests/e2e/test_local_training_smoke.py`** — End-to-end dry-run validation
- **New `.agent/workflows/local-smoke-test.md`** — Step-by-step local smoke test workflow
- **`alpha_zero/training_civ6.py`** — Add `--max-iters` flag for bounded testing

---

## Phase 5: Distributed Training Dry-Run

**Goal**: Verify multi-worker coordination, trajectory merging, and preemption recovery.

### Changes

- **New `tests/integration/test_distributed_coordination.py`** — 4-worker simulation with mock GCS
- **New `tests/integration/test_preemption_recovery.py`** — Spot VM preemption simulation
- **`alpha_zero/distributed/`** — Add health-check heartbeat and structured logging

---

## Phase 6: PBT Optimization & Evaluation League

**Goal**: Launch PBT with automated hyperparameter mutation and ELO evaluation. Uses `ml-pipeline-workflow` skill.

### Changes

- **New `configs/pbt_config.yaml`** — Population size, mutation rates, search space
- **`alpha_zero/pbt/`** — Wire PBT to Vertex AI lifecycle + W&B tracking
- **New `tests/integration/test_pbt_lifecycle.py`** — Full PBT cycle simulation
- **New `.agent/workflows/pbt-launch.md`** — Step-by-step PBT launch workflow

---

## Skills & MCP Integration Matrix

| Skill / MCP | Phases | Usage |
|---|---|---|
| `python-testing-patterns` | 1, 3-6 | Fixtures, parameterization, AAA pattern in all test files |
| `ml-pipeline-workflow` | 3, 4, 6 | Pipeline stage validation, data contracts |
| `changelog-automation` | 2 | Conventional commits, auto release notes |
| `systematic-debugging` | 4 | Root-cause-first debugging in smoke tests |
| **GitKraken MCP** | All | Branch management, PR creation, pushes |
| **GitHub CI** | 1-2 | Enhanced pipeline with coverage/security jobs |

---

## Execution Priority

```
Phase 1 (Test Coverage) ──→ Phase 2 (CI/CD) ──→ Phase 3 (GCP E2E) ──→ Phase 5 (Distributed) ──→ Phase 6 (PBT)
                                             ╰──→ Phase 4 (Local Smoke) ─╯
```

**Estimated**: Phases 1-2 (1-2 days) → Phases 3-4 (2-3 days) → Phase 5 (3-5 days) → Phase 6 (1-2 weeks)
