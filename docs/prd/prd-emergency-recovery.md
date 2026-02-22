# PRD: Container Rebuild & Deployment + Orchestrator Fix

**Epics**: 1 (Container Rebuild) + 2 (Orchestrator Fix)
**Priority**: P0 — Production down
**Milestone**: 1 — Emergency Recovery

---

## User Story

**As a** ML engineer running MuZero training on Vertex AI,
**I want** the container image to support the `--mcts_type=muzero` argument,
**So that** the orchestrator can launch MuZero self-play workers without argparse failures.

---

## Background

The orchestrator passes `--mcts_type=muzero` to self-play workers when `--algorithm=muzero` is specified. The local source code already supports this (`selfplay_worker.py` line 81: `choices=["civ6", "gumbel", "batch_gumbel", "muzero"]`), but the deployed container image (`muzero-training:v13-compat`) was built before this change. Additionally, the orchestrator's own `--mcts_type` choices (line 104 of `vertex_orchestrator.py`) are limited to `["civ6", "gumbel"]` — missing `muzero` and `batch_gumbel`.

---

## Acceptance Criteria

### AC-1: Container supports muzero MCTS type

**Given** a newly built container image `v1.0.0`,
**When** the container is invoked with `--mcts_type=muzero`,
**Then** the self-play worker starts without argparse errors.

### AC-2: Container pushed to Artifact Registry

**Given** the Artifact Registry repo exists,
**When** `build_and_push.py --tag v1.0.0` completes,
**Then** the image is available at `us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.0.0`.

### AC-3: Orchestrator accepts muzero choice

**Given** the orchestrator CLI,
**When** `--mcts_type=muzero` is passed,
**Then** the argument is accepted without argparse errors.

### AC-4: Orchestrator passes muzero to workers

**Given** `--algorithm=muzero` is specified,
**When** `build_selfplay_extra_args` is called,
**Then** the extra args list contains `--mcts_type=muzero`.

### AC-5: Unit tests pass

**Given** the updated codebase,
**When** `pytest tests/cloud/unit/` is run,
**Then** all tests pass, including new tests for muzero MCTS type handling.

---

## Out of Scope

- MuZero MCTS algorithm implementation changes (already working)
- Container dependency upgrades beyond the image rebuild
- GPU type or machine type changes

---

## Success Metrics

| Metric | Target |
|:---|:---|
| Container starts with `--mcts_type=muzero` | No argparse errors |
| Orchestrator CLI accepts `muzero` | Pass |
| Self-play smoke test (1 iter, 2 games) | JOB_STATE_SUCCEEDED |
| Unit test suite | 100% pass |

---

## Open Questions

1. **Artifact Registry provisioning**: Has `setup_gcp.ps1` been run? The repo `civ6az` may not exist yet. The currently working repo is `muzero-world-model`.
2. **Container registry target**: Should we push to the existing `muzero-world-model` repo or the new `civ6az` repo?
3. **Dockerfile**: Which Dockerfile — `Dockerfile.vertex` or `Dockerfile`?
