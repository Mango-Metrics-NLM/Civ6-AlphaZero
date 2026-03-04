# PRD: Multi-Iteration Training Orchestrator (M10)

**Epic**: M10 — Multi-Iteration Loop
**Complexity**: L (Large)
**Sprint**: 2

## User Story

**As a** ML researcher,
**I want** to run 50+ AlphaZero iterations automatically with evaluation gates,
**So that** I can train a strong game-playing agent without manual intervention.

## Acceptance Criteria

### AC1: Automated Loop

**Given** a configured GCP environment and initial model v0,
**When** I run `python -m alpha_zero.cloud.gcp.cli orchestrate --start-iter 50 --end-iter 100 --tier budget`,
**Then** the system runs self-play → training → evaluation for each iteration automatically.

### AC2: Evaluation Gate

**Given** a newly trained model vN,
**When** it plays 30 games against model v(N-1),
**Then** if win rate < 55%, the model is rejected and v(N-1) is reused for the next iteration.

### AC3: Preemption Recovery

**Given** a Spot VM preemption during iteration 75,
**When** the orchestrator is restarted,
**Then** it resumes from iteration 75 (not iteration 1) using GCS state markers.

### AC4: No Hardcoded Values

**Given** the orchestrator code,
**When** reviewed for hardcoded project IDs, bucket names, or API keys,
**Then** zero hardcoded values are found — all are sourced from env vars or config files.

## Out of Scope

- Multi-GPU training (stay on single T4)
- CivRealm/Unciv integration
- Model deployment to production serving endpoint

## Success Metrics

- 50 iterations complete without manual intervention
- < 5% iteration failure rate due to infrastructure
- Model v269 shows measurable Elo improvement over v219
