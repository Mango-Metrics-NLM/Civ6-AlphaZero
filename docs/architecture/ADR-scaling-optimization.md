# ADR: Scaling & Optimization Architecture

**Date**: 2026-03-04
**Status**: Proposed
**Author**: AI Agent

## Context

Training validation completed: 49 iterations, 219 model versions on T4 GPUs. Sprint 2 scales to 100+ iterations with PBT hyperparameter optimization and production monitoring.

## Decision

### Multi-Iteration Orchestrator

Use `vertex_orchestrator.py` as the coordinator:

```mermaid
graph TD
    A[Orchestrator Start] --> B{Resume from GCS state?}
    B -->|Yes| C[Load iteration state]
    B -->|No| D[Start from iter 1]
    C --> E[Self-Play Phase]
    D --> E
    E --> F[Training Phase]
    F --> G[Evaluation Gate]
    G -->|Win rate ≥ 55%| H[Accept model]
    G -->|Win rate < 55%| I[Reject model, reuse prev]
    H --> J{More iterations?}
    I --> J
    J -->|Yes| E
    J -->|No| K[Training Complete]
```

### PBT Architecture

```mermaid
graph LR
    P1[Agent 1<br/>lr=0.001] --> E{Evaluate}
    P2[Agent 2<br/>lr=0.003] --> E
    P3[Agent 3<br/>lr=0.0003] --> E
    P4[Agent 4<br/>lr=0.01] --> E
    E --> R[Rank by Elo]
    R --> EX[Exploit: copy top 20% weights]
    EX --> MU[Explore: mutate ±20%]
    MU --> P1
    MU --> P2
    MU --> P3
    MU --> P4
```

### Data Flow

```mermaid
flowchart LR
    SM["Secret Manager<br/>(WANDB_API_KEY)"] --> VJ["Vertex Jobs"]
    VJ --> SP["Self-Play Workers<br/>(T4 Spot × N)"]
    SP --> GCS["GCS Bucket<br/>gs://civ6az-training"]
    GCS --> TR["Training Worker<br/>(T4 × 1)"]
    TR --> GCS
    GCS --> EV["Eval Worker<br/>(CPU × 1)"]
    EV --> WB["W&B Dashboard"]
    VJ --> CM["Cloud Monitoring"]
```

## Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Orchestration | Vertex AI Custom Jobs | Already in use, mature, no new deps |
| State recovery | GCS JSON markers | Preemption-safe, no DB needed |
| PBT | Custom module | Lightweight, AlphaZero-specific |
| Secrets | GCP Secret Manager | No plain-text keys in job specs |
| Monitoring | Cloud Monitoring + W&B | Dual: infra alerts + ML metrics |

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Spot VM preemption during PBT | Medium | GCS checkpoint + auto-restart |
| GPU quota exhaustion | High | Budget tier T4 default, fallback A100 |
| W&B rate limiting | Low | Batch logging, offline mode fallback |
| Cost overrun | Medium | Daily spend ceiling via billing alerts |

## Non-Functional Requirements

- **Latency**: Orchestrator loop < 5s overhead per iteration
- **Scalability**: Support 1-20 parallel self-play workers
- **Security**: All secrets via Secret Manager, least-privilege SA
- **Cost**: Max $50/day budget alert
