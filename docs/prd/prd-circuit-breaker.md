# PRD: Circuit Breaker & Retry Hardening

**Epic**: 4 — Circuit Breaker & Retry Hardening
**Priority**: P1
**Milestone**: 2 — Pipeline Hardening

---

## User Story

**As a** ML engineer running long training campaigns on Vertex AI,
**I want** the orchestrator to automatically stop retrying after consecutive failures,
**So that** I don't burn cloud compute costs on a known-broken configuration.

---

## Background

The current orchestrator retry logic (up to `max_retries=2` per iteration) does not track *consecutive* iteration failures. When a systemic issue occurs (e.g., stale container, missing model), the orchestrator retries each iteration independently, advancing through iterations even when every single one fails. In the current incident, this produced 50+ failed jobs over ~2 hours.

---

## Acceptance Criteria

### AC-1: Consecutive failure tracking

**Given** the orchestrator main loop,
**When** N consecutive iterations fail (all retry attempts exhausted),
**Then** the orchestrator logs a CRITICAL message and exits cleanly.

### AC-2: Configurable threshold

**Given** the orchestrator CLI,
**When** `--max_consecutive_failures=5` is specified,
**Then** the circuit breaker triggers after 5 consecutive iteration failures.

### AC-3: Cost tracking

**Given** an active training run,
**When** each iteration completes (success or failure),
**Then** the estimated cost is logged and the cumulative cost is tracked.

### AC-4: Exponential backoff

**Given** a failed iteration,
**When** the orchestrator retries,
**Then** it waits with exponential backoff + jitter (30s, 60s, 120s).

### AC-5: Reset on success

**Given** the consecutive failure counter is at N,
**When** an iteration succeeds,
**Then** the counter resets to 0.

---

## Out of Scope

- Email/SMS alerting (covered by Epic 6)
- Auto-remediation (e.g., auto-rebuilding containers)
- Budget enforcement via GCP Billing APIs

---

## Success Metrics

| Metric | Target |
|:---|:---|
| Circuit breaker triggers before >$5 wasted | Yes |
| Backoff prevents API throttling | No rate limit errors |
| Unit test coverage for circuit breaker | ≥90% |

---

## Open Questions

1. **Default threshold**: Should `--max_consecutive_failures` default to 3 or 5?
2. **Partial failures**: If 1 of 2 workers succeeds, does the iteration count as failed for circuit breaker purposes?
