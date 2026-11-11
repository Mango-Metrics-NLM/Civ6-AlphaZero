# PRD: Container Version Validation Gate

**Epic**: 5 — Container Version Validation Gate
**Priority**: P1
**Milestone**: 2 — Pipeline Hardening

---

## User Story

**As a** ML engineer deploying training jobs,
**I want** the orchestrator to validate that the container image supports the required CLI flags before launching,
**So that** stale-container failures are caught at orchestrator startup, not inside Vertex AI workers.

---

## Background

The root cause of the current incident was a mismatch between orchestrator expectations and container capabilities. The orchestrator sent `--mcts_type=muzero` to a container that didn't support it. A pre-flight validation step would have caught this before any cloud resources were consumed.

---

## Acceptance Criteria

### AC-1: Container reports build version

**Given** the container image,
**When** invoked with `--version`,
**Then** it prints a structured version string (e.g., `alphazero-training v1.0.0 (commit: abc123)`).

### AC-2: Orchestrator validates container

**Given** the orchestrator startup sequence,
**When** the container image URI is resolved,
**Then** the orchestrator checks the container's `--version` output or image labels.

### AC-3: Stale container rejected

**Given** a container that doesn't support `--mcts_type=muzero`,
**When** the orchestrator is invoked with `--algorithm=muzero`,
**Then** it logs an error and exits before launching any Vertex AI jobs.

### AC-4: Skip validation flag

**Given** the orchestrator CLI,
**When** `--skip_container_validation` is passed,
**Then** the pre-flight check is skipped (for CI or known-good environments).

---

## Out of Scope

- Container auto-rebuild on validation failure
- Version compatibility matrix management
- Docker image scanning for vulnerabilities

---

## Success Metrics

| Metric | Target |
|:---|:---|
| Stale container caught before job launch | 100% |
| Validation overhead | <5 seconds |
| False positive rate | 0% |
