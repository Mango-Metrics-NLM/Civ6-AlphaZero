# PRD: Pipeline Hardening — Evaluation Worker, Backoff, Validation

## User Stories

### US-1: Arena MuZero Compatibility

**As a** training engineer,
**I want** the evaluation worker to correctly run MuZero model comparisons,
**So that** I can measure model improvement across training iterations.

#### Acceptance Criteria

- **Given** `--mcts_type=muzero` is passed to the evaluation worker
- **When** the arena runs head-to-head games between v(N) and v(N-1)
- **Then** the arena extracts observation and action_mask from the env and passes them to `MuZeroMCTS.search()`
- **And** the evaluation completes with a valid `ArenaResult`

### US-2: Exponential Backoff

**As a** cost-conscious operator,
**I want** orchestrator retries to use exponential backoff with jitter,
**So that** transient failures don't incur flat-rate retry costs and API hammering.

#### Acceptance Criteria

- **Given** a self-play or training job fails transiently
- **When** the orchestrator retries
- **Then** the delay between retries increases exponentially (base 30s, max 300s)
- **And** each delay includes random jitter (±25%)
- **And** the delay values are logged for observability

### US-3: Container Version Validation

**As a** training engineer,
**I want** the orchestrator to verify the container image exists before dispatching jobs,
**So that** I don't waste ~5 min + costs waiting for jobs to fail on missing images.

#### Acceptance Criteria

- **Given** the orchestrator starts and `--skip_image_validation` is not set
- **When** the container image URI is checked against the registry
- **Then** if the image exists, the orchestrator logs the digest and proceeds
- **And** if the image does not exist, the orchestrator aborts with a clear error
- **And** if the registry is unreachable, the orchestrator logs a warning and proceeds

## Out of Scope

- Artifact Registry migration (deferred — currently using GCR)
- Secret Manager integration (deferred)
- Automated rollback on evaluation failure (evaluation is informational, not blocking)

## Success Metrics

| Metric | Target |
|---|---|
| Evaluation worker smoke test | All 3 phases pass (self-play ✅, training ✅, eval ✅) |
| Test coverage for new code | ≥80% |
| Retry delay on 3rd attempt | ≥120s (vs current flat 30s) |
| Missing image detection | <5s (vs current ~5min job fail) |
