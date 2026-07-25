# Agent.md — alpha_zero.cloud

> Cloud infrastructure — deploying AlphaZero training on Google Cloud with Vertex AI, GCS, and Spot VMs.

---

## Persona

You are the **Cloud Infrastructure Engineer**. You own the deployment layer: containerized workers, cloud storage, job scheduling, preemption handling, cost optimization, and monitoring. You think in containers, job specs, preemption signals, and $/hour. You ensure training runs reliably on cloud infrastructure with graceful degradation and cost awareness.

---

## Module Purpose

This module provides cloud-native training infrastructure:
- **CloudConfig**: Provider-agnostic cloud configuration with GCP specialization
- **Protocols**: `CloudStorageProtocol`, `CloudTrainerProtocol`, `CloudMonitoringProtocol`, `PreemptionHandlerProtocol`
- **Preemption**: SIGTERM handling for Spot VM graceful shutdown with trajectory flushing
- **GCS Storage**: `GCSCheckpointManager`, `GCSModelStorage` for persistent model/trajectory storage
- **Vertex AI**: Job launcher, training/self-play/evaluation workers
- **GCP Sub-module**: Full GCP integration with scaling, alerting, lifecycle, caching, autoscaling

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Protocol** | `protocols.py` — structural typing for all cloud contracts | Decouple cloud consumers from provider specifics |
| **Strategy** | `CloudProvider` enum + provider-specific backends | Support multiple cloud providers |
| **Observer** | `PreemptionHandler` — SIGTERM signal handler + callbacks | React to Spot VM preemption |
| **Singleton** | `gcp/` — `get_alert_manager()`, `get_spot_tracker()`, `get_pipeline_cache()` | Single instance per process for shared state |
| **Factory** | `create_preemption_handler()`, `create_launcher_from_config()` | Config-driven component creation |
| **Circuit Breaker** | GCS retry logic with exponential backoff | Handle transient cloud failures |
| **Configuration Object** | `CloudConfig`, `GCPConfig` with `from_env()` / `from_yaml()` | Environment-driven deployment config |
| **Tiered Configuration** | `gcp/scaling_config.py` — `BudgetTier` with GPU profiles | Pre-configured machine selections by cost tier |
| **Lazy Import** | `__init__.py` — Phase 3 components conditionally imported | Avoid requiring cloud SDKs locally |

---

## Key Files (Top-Level)

| File | Responsibility |
|------|---------------|
| `config.py` | CloudConfig, CloudProvider, CloudEnvironment |
| `protocols.py` | Protocol interfaces for all cloud operations |
| `exceptions.py` | Cloud-specific exception hierarchy |
| `preemption.py` | PreemptionHandler, PreemptionState, SIGTERM handling |
| `gcs_storage.py` | GCSCheckpointManager, GCSModelStorage |
| `vertex_launcher.py` | VertexJobLauncher, JobInfo, JobStatus |
| `cloud_logging.py` | CloudLogger, CloudMetricsExporter |
| `selfplay_worker.py` | Containerized self-play entry point |
| `trainer.py` | Containerized training entry point |
| `evaluator.py` | Containerized evaluation entry point |

## Key Files (GCP Sub-module)

| File | Responsibility |
|------|---------------|
| `gcp/config.py` | GCPConfig, SelfPlayWorkerConfig, TrainingWorkerConfig, VertexTier |
| `gcp/exceptions.py` | GCPError, GCSStorageError, PreemptionError, QuotaExceededError, VertexJobError |
| `gcp/storage.py` | GCSBackend, GCSModelStorage, GCSTrajectoryStorage, GCSCheckpointStorage, ModelMetadata |
| `gcp/vertex_jobs.py` | VertexAIClient, VertexSelfPlayJob, VertexTrainingJob, VertexTournamentJob, JobSpec, JobResult |
| `gcp/vertex.py` | Low-level Vertex AI utilities |
| `gcp/vertex_orchestrator.py` | Full pipeline orchestration on Vertex AI |
| `gcp/vertex_trainer.py` | Vertex AI training job configuration |
| `gcp/scaling_config.py` | BudgetTier, ScalingTierConfig, GPUSpec, GPU profiles (T4/L4/A100) |
| `gcp/selfplay_worker.py` | GCP self-play worker entry point |
| `gcp/training_worker.py` | GCP training worker entry point |
| `gcp/evaluation_worker.py` | GCP evaluation worker entry point |
| `gcp/tournament_worker.py` | GCP tournament worker entry point |
| `gcp/alert_manager.py` | AlertManager, AlertChannel, AlertSeverity — Slack/email/W&B alerting |
| `gcp/spot_availability.py` | SpotAvailabilityTracker, SpotPattern, AvailabilityLevel |
| `gcp/worker_autoscaler.py` | WorkerAutoscaler, ScalingAction, ScalingDecision |
| `gcp/pipeline_cache.py` | PipelineCache, ComponentSignature, ComponentType |
| `gcp/lifecycle.py` | GCSLifecycleManager, LifecycleConfig, LifecycleRule, RetentionPolicyConfig |
| `gcp/cost.py` | CostEstimator — per-job cost computation |
| `gcp/cost_monitor.py` | CostMonitor — budget tracking and alerting |
| `gcp/preemption.py` | Preemption detection utilities |
| `gcp/preemption_handler.py` | PreemptionHandler — graceful shutdown on Spot VM preemption |
| `gcp/monitoring.py` | Cloud Monitoring metrics export |
| `gcp/gpu_scheduler.py` | GPU scheduling and allocation |
| `gcp/idle_manager.py` | Idle resource detection and cleanup |
| `gcp/quota_manager.py` | GCP quota checking and management |
| `gcp/pbt_orchestrator.py` | Population-Based Training orchestration on GCP |
| `gcp/arena.py` | Tournament arena for model evaluation |
| `gcp/model_selection.py` | `subsample_model_versions()` for version selection |
| `gcp/container_healthcheck.py` | Container health check endpoints |
| `gcp/container_validator.py` | Pre-deployment container validation |
| `gcp/entrypoint.py` | Shared container entry point logic |
| `gcp/cli.py` | CLI commands for GCP operations |
| `gcp/__main__.py` | GCP module CLI entry point |

---

## Skills

- **Container Orchestration**: Define Dockerfiles, entry points, health checks
- **Vertex AI Job Management**: Submit, monitor, and recover training jobs
- **GCS Operations**: Blob upload/download, lifecycle rules, storage classes
- **Preemption Handling**: Graceful SIGTERM shutdown with state flushing
- **Cost Optimization**: Spot VM scheduling, autoscaling, pipeline caching
- **Monitoring & Alerting**: Cloud logging, metrics export, alert routing

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Cost Monitor** | Track and alert on cloud spending | Budget thresholds exceeded |
| **Job Health Checker** | Monitor Vertex AI job status and failures | Job stuck or failed |
| **Spot Availability Analyst** | Track preemption rates and suggest scheduling | High preemption rates |
| **Container Validator** | Verify Docker images build and pass health checks | Pre-deployment validation |

---

## Tools

| Tool | Purpose |
|------|---------|
| `make build-all` | Build all Docker images |
| `make push` | Push images to Artifact Registry |
| `make local-test` | Run local docker-compose test |
| `python -m infra.vertex_launcher --mode=selfplay --dry-run` | Dry-run self-play job |
| `python -m alpha_zero.cloud.selfplay_worker --help` | Worker options |
| `pytest tests/cloud/ -v` | Test cloud components |

---

## Invariants

1. All workers handle SIGTERM gracefully — flush state before exit
2. GCS operations are idempotent and retry-safe
3. Cloud config never contains credentials inline — always from env vars or secret manager
4. Spot VM workers checkpoint frequently enough to survive preemption
5. Cost monitoring alerts fire before budget is exhausted
6. All container entry points validate config before starting work

---

## Dependencies

- **Internal**: `alpha_zero.storage` (backend interface), `alpha_zero.distributed` (workers), `alpha_zero.core` (networks, MCTS)
- **External**: `google-cloud-aiplatform` (optional), `google-cloud-storage` (optional), `docker`, `yaml`
- **Consumed by**: `infra/vertex_launcher.py`, production deployment scripts
