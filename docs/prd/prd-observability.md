# PRD: Failure Alerting & W&B Dashboard

**Epic**: 6 — Failure Alerting & W&B Dashboard
**Priority**: P2
**Milestone**: 3 — Observability

---

## User Story

**As a** ML engineer running multi-day training campaigns,
**I want** to receive alerts when training failures occur and see training health at a glance,
**So that** I can intervene early instead of discovering failures hours later.

---

## Background

The current incident went undetected for ~2 hours and 50+ failed jobs. W&B is already integrated for metrics logging, but failure events are not surfaced as alerts and there is no dashboard for pipeline health.

---

## Acceptance Criteria

### AC-1: Structured failure logging

**Given** a failed Vertex AI job,
**When** the orchestrator handles the failure,
**Then** it logs a structured event to W&B with: error type, iteration, job name, estimated cost, and timestamp.

### AC-2: W&B alert on consecutive failures

**Given** the W&B project `civ6-alphazero`,
**When** 3 consecutive iterations fail,
**Then** a W&B alert is triggered (visible in the W&B UI and via email if configured).

### AC-3: Training health dashboard

**Given** an active training run,
**When** the user opens the W&B project,
**Then** they see a dashboard with: iterations/hour, cost/iteration, failure rate, and model version timeline.

### AC-4: Iteration cost panel

**Given** completed iterations,
**When** the dashboard is viewed,
**Then** each iteration shows estimated cost (self-play + training + evaluation).

---

## Out of Scope

- PagerDuty/Slack integration
- GCP Cloud Monitoring alerts (uses W&B instead)
- Historical cost reconciliation with GCP Billing

---

## Success Metrics

| Metric | Target |
|:---|:---|
| Alert latency (failure → notification) | <5 minutes |
| Dashboard load time | <3 seconds |
| False alert rate | 0% |

---

## Open Questions

1. **W&B Alerts API**: Does the W&B Python SDK support programmatic alert creation, or is manual setup required?
2. **Email delivery**: Is W&B email alerting configured for the maintainer (@ianshank)?
