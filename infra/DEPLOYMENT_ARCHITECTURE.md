# Civ6 AlphaZero Gameplay Deployment Architecture

## Infrastructure Diagram (ASCII)

```
                           INTERNET
                              |
                    +-------------------+
                    | Cloud CDN / LB    |  <-- Global HTTPS Load Balancer
                    +-------------------+
                      /       |        \
                     /        |         \
            +-------+  +----------+  +-----------+
            |  CDN  |  |Cloud Run |  | Cloud Run |
            |Static |  |Auth Svc  |  |Matchmaker |
            |Assets |  |(256MB)   |  |(256MB)    |
            +-------+  +----------+  +-----------+
               |              |            |
               |        +-----+-----+     |
     React SPA |        | Firebase  |     |
     (GCS)     |        | Auth      |     |
               |        +-----------+     |
               |                          |
               +----------+    +----------+
                           |    |
                    +-------------------+
                    |  Serverless VPC   |
                    |  Connector        |
                    +-------------------+
                           |
              +------------+-------------+
              |                          |
    +------------------+     +-----------------------+
    | Cloud Run        |     | SpacetimeDB           |
    | AI Inference     |     | (GCE e2-standard-4    |
    | (GPU: L4)        |     |  + persistent SSD)    |
    | min=0, max=10    |     |                       |
    | concurrency=5    |     | WebSocket :3000       |
    +------------------+     | gRPC      :50000      |
              |              +-----------------------+
              |                          |
    +------------------+     +-----------------------+
    | GCS Bucket       |     | GCS Bucket            |
    | civ6az-models    |     | civ6az-replays        |
    | (model weights)  |     | (game replays)        |
    +------------------+     +-----------------------+
              |
    +------------------+
    | Artifact Registry|
    | civ6az           |
    | (container imgs) |
    +------------------+
```

## Component Architecture

### 1. SpacetimeDB Server (Game State + Real-Time Sync)

**Service**: GCE Instance (e2-standard-4) with SpacetimeDB binary
**Why GCE over Cloud Run**: SpacetimeDB requires persistent WebSocket connections and local
state. Cloud Run's request-based model with 60-min timeout cap does not suit long-lived game
sessions that may span hours. GCE provides stable, addressable compute with persistent disk.

- **Machine**: e2-standard-4 (4 vCPU, 16 GB RAM)
- **Disk**: 50 GB SSD persistent disk (game state + SpacetimeDB WAL)
- **Networking**: Static internal IP, exposed via Internal Load Balancer for Cloud Run access;
  external access via Global HTTPS LB with WebSocket support
- **Scaling**: Vertical scaling (upgrade machine type) or horizontal with multiple instances
  behind the LB, each handling a subset of game rooms
- **Backup**: Hourly snapshots of persistent disk, daily export to GCS

### 2. AI Inference Service (PyTorch + GumbelMCTS)

**Service**: Cloud Run (GPU) with L4 GPU
**Why Cloud Run GPU**: Provides scale-to-zero (no cost when idle), automatic scaling,
and L4 GPU access. The Civ6TransformerNetwork (embed_dim=64, 6 layers) is small enough
to load in <2s on an L4. GumbelMCTS with 16-25 simulations completes in <500ms per move.

- **Container**: Custom image based on Dockerfile.vertex, extended with a gRPC/HTTP
  inference server
- **GPU**: NVIDIA L4 (24 GB VRAM -- far exceeds the ~50 MB model)
- **Min instances**: 0 (scale to zero when no games)
- **Max instances**: 10 (each handles ~5 concurrent MCTS searches = 50 concurrent inferences)
- **Concurrency**: 5 requests per instance (MCTS is CPU+GPU intensive)
- **Timeout**: 300s (covers long MCTS searches)
- **Model loading**: Download latest model from GCS on cold start, cache in memory
- **Latency budget**: ~200ms model load (cached) + ~500ms MCTS (16 sims) + ~100ms overhead = <1s

### 3. Web Frontend (React SPA)

**Service**: GCS Static Website + Cloud CDN
**Why GCS+CDN**: Cheapest possible hosting for static assets. React SPA connects to
SpacetimeDB via WebSocket and to the inference service via the HTTPS LB.

- **Bucket**: `civ6az-frontend` with website configuration
- **CDN**: Cloud CDN with 1-hour TTL on static assets, no-cache on index.html
- **Custom domain**: Optional, via Cloud DNS + managed SSL certificate
- **Cost**: Essentially free at this scale (<$1/month for CDN + storage)

### 4. Authentication Service

**Service**: Firebase Authentication (managed) + thin Cloud Run proxy
**Why Firebase Auth**: Free tier supports 50K MAU. Handles OAuth (Google, GitHub, email/password)
without custom auth infrastructure. The Cloud Run proxy validates Firebase JWT tokens and
translates them to SpacetimeDB identity tokens.

- **Cloud Run**: 256 MB, 1 vCPU, min=0, max=5
- **Concurrency**: 80 (auth is lightweight)
- **Integration**: Firebase Admin SDK validates JWTs; issues SpacetimeDB connection tokens

### 5. Matchmaking Service

**Service**: Cloud Run + Firestore (serverless document DB)
**Why Firestore**: Real-time listeners for lobby state changes, serverless scaling,
generous free tier (1 GB storage, 50K reads/day, 20K writes/day).

- **Cloud Run**: 256 MB, 1 vCPU, min=0, max=5
- **Concurrency**: 80
- **Queue**: Firestore collection for matchmaking queue; Cloud Run polls/matches players
- **Matching**: ELO-based matching with expanding search windows (30s increments)
- **Game creation**: On match, creates SpacetimeDB game room, returns connection info

### 6. Replay Storage

**Service**: GCS Bucket with lifecycle policies
**Why GCS**: Cheapest storage for large replay files. Lifecycle policies auto-transition
old replays to Nearline (30 days) then Coldline (90 days).

- **Bucket**: `civ6az-replays`
- **Structure**: `replays/{game_id}/replay.json` + `replays/{game_id}/metadata.json`
- **Lifecycle**: Standard -> Nearline (30d) -> Coldline (90d) -> Delete (365d)
- **Access**: Signed URLs generated by the auth service for replay viewing

## GCP Service Selection Summary

| Component | GCP Service | Rationale |
|:---|:---|:---|
| SpacetimeDB | GCE (e2-standard-4) | Persistent WebSocket, local state, long-lived |
| AI Inference | Cloud Run (GPU, L4) | Scale-to-zero, auto-scaling, GPU support |
| Frontend | GCS + Cloud CDN | Static SPA, cheapest option |
| Auth | Firebase Auth + Cloud Run | Free tier, managed OAuth |
| Matchmaking | Cloud Run + Firestore | Serverless, real-time listeners |
| Replay Storage | GCS (lifecycle) | Cheap bulk storage with tiering |
| Load Balancer | Global HTTPS LB | SSL termination, WebSocket support, CDN |
| Container Registry | Artifact Registry | Existing `civ6az` repo |
| CI/CD | Cloud Build | Existing pipeline, extends current cloudbuild.yaml |
| Monitoring | Cloud Monitoring + Logging | Native GCP integration |
| Secrets | Secret Manager | Existing setup for WANDB/API keys |

## Networking and Security Architecture

### VPC Design

```
VPC: civ6az-gameplay-vpc (10.0.0.0/16)
  |
  +-- Subnet: services (10.0.1.0/24, us-central1)
  |     - SpacetimeDB GCE instance
  |     - Internal Load Balancer
  |
  +-- Subnet: serverless (10.0.2.0/28, us-central1)
  |     - Serverless VPC Connector (Cloud Run -> VPC)
  |
  +-- Cloud NAT (outbound internet for GCE)
  +-- Cloud Router
```

### Firewall Rules

| Rule | Direction | Source | Target | Ports | Purpose |
|:---|:---|:---|:---|:---|:---|
| allow-lb-health | Ingress | 130.211.0.0/22, 35.191.0.0/16 | SpacetimeDB | 3000,50000 | LB health checks |
| allow-serverless | Ingress | 10.0.2.0/28 | SpacetimeDB | 3000,50000 | Cloud Run -> SpacetimeDB |
| allow-iap-ssh | Ingress | 35.235.240.0/20 | SpacetimeDB | 22 | IAP tunneled SSH |
| deny-all-ingress | Ingress | 0.0.0.0/0 | all | all | Default deny |

### Security Layers

1. **Edge**: Cloud Armor WAF on Global HTTPS LB (OWASP top 10 rules)
2. **Transport**: TLS 1.3 everywhere (managed certificates)
3. **Identity**: Firebase Auth JWT tokens, validated at every service boundary
4. **Network**: VPC with private GCE, Serverless VPC Connector for Cloud Run
5. **Secrets**: Secret Manager for API keys, no plaintext in env vars
6. **IAM**: Least-privilege service accounts per component
7. **Container**: Non-root user in containers (already implemented in Dockerfile.vertex)

### Service Accounts

| SA Name | Roles | Used By |
|:---|:---|:---|
| `civ6az-inference@` | storage.objectViewer, run.invoker | AI Inference Cloud Run |
| `civ6az-spacetimedb@` | storage.objectAdmin, compute.instanceAdmin.v1 | SpacetimeDB GCE |
| `civ6az-auth@` | firebaseauth.admin, run.invoker, secretmanager.secretAccessor | Auth Service |
| `civ6az-matchmaker@` | datastore.user, run.invoker | Matchmaking Service |
| `civ6az-vertex@` | (existing) aiplatform.user, storage.objectAdmin | Training pipeline |

## CI/CD Pipeline Design

### Pipeline Stages

```
GitHub Push (main or release/*)
       |
       v
  Cloud Build Trigger
       |
  +----+----+
  |         |
  v         v
[Lint]   [Unit Tests]     (parallel)
  |         |
  +----+----+
       |
       v
  [Build Containers]      (3 images: inference, auth, matchmaker)
       |
  +----+----+----+
  |    |    |    |
  v    v    v    v
[Smoke Tests per image]   (parallel --dry_run tests)
       |
       v
  [Push to Artifact Registry]
       |
       v
  [Deploy to Staging]     (Cloud Run --tag=staging)
       |
       v
  [Integration Tests]     (hit staging endpoints)
       |
       v
  [Promote to Prod]       (Cloud Run traffic split: 0% -> 100%)
```

### Model Update Pipeline (separate trigger)

```
GCS Model Upload (gs://civ6az-training/models/civ6az/v*/model.pt)
       |
       v
  Cloud Function (GCS trigger)
       |
       v
  [Validate model loads]  (temporary Cloud Run job)
       |
       v
  [Update inference service env var MODEL_VERSION=vNNN]
       |
       v
  [Cloud Run new revision deployed]
       |
       v
  [Canary: 10% traffic for 5 min]
       |
       v
  [Monitor latency + error rate]
       |
       v
  [Full rollout or auto-rollback]
```

## Monitoring and Observability

### Metrics

| Metric | Source | Alert Threshold |
|:---|:---|:---|
| AI move latency (p50, p95, p99) | Cloud Run custom metric | p99 > 2s |
| SpacetimeDB WebSocket connections | Custom metric (Prometheus) | > 2000 (capacity) |
| Active games | Firestore counter | > 1000 (scaling trigger) |
| Model inference errors | Cloud Run error rate | > 1% over 5 min |
| GPU utilization | Cloud Run GPU metrics | < 10% for 30 min (scale down) |
| Cold start latency | Cloud Run startup metric | > 5s |
| Auth token validation failures | Cloud Logging | > 50/min (attack?) |
| GCS egress | Storage metrics | > 100 GB/day (cost alert) |

### Dashboards

1. **Operations**: Request rate, error rate, latency (RED) per service
2. **Game Health**: Active games, moves per second, average game duration
3. **AI Performance**: MCTS search time, model inference time, GPU utilization
4. **Cost**: Daily spend by service, projected monthly cost
5. **Security**: Auth failures, WAF blocks, unusual traffic patterns

### Logging

- **Structured JSON**: All services log structured JSON to Cloud Logging
- **Trace context**: Distributed tracing via Cloud Trace (auto-instrumented in Cloud Run)
- **Log sink**: Critical logs -> BigQuery for long-term analysis
- **Retention**: Cloud Logging 30 days, BigQuery 1 year

### Alerting

- **PagerDuty/Slack**: Critical alerts (service down, error spike)
- **Email**: Warning alerts (high latency, cost threshold)
- **Auto-remediation**: Cloud Run auto-restart on crash, GCE auto-heal MIG

## Disaster Recovery and Backup

### RTO/RPO Targets

| Component | RPO | RTO | Strategy |
|:---|:---|:---|:---|
| SpacetimeDB state | 1 hour | 15 min | Hourly disk snapshots, standby instance |
| Model weights | 0 (GCS durable) | 5 min | GCS multi-region replication |
| Replays | 0 (GCS durable) | 5 min | GCS lifecycle, cross-region copy |
| Firestore (matchmaking) | 0 (managed) | 0 | Automatic, multi-region |
| Firebase Auth | 0 (managed) | 0 | Automatic, global |

### Backup Strategy

1. **SpacetimeDB**: Scheduled disk snapshot every hour via Cloud Scheduler -> Cloud Function
2. **Model weights**: Already in GCS (11 nines durability); copy to secondary bucket weekly
3. **Replays**: GCS with versioning enabled
4. **Configuration**: All IaC in Git; Terraform state in GCS backend with versioning

### Recovery Procedures

1. **SpacetimeDB failure**: Auto-heal MIG replaces instance from latest snapshot (<15 min)
2. **Cloud Run failure**: Automatic rollback to previous revision (immediate)
3. **Model corruption**: Revert MODEL_VERSION env var to previous version (1 min)
4. **Full region failure**: Manual failover to us-east1 (pre-provisioned Terraform, ~30 min)

## Cost Breakdown

### Steady State: 100 Concurrent Games

| Component | Specification | Hours/mo | Unit Cost | Monthly |
|:---|:---|:---|:---|:---|
| SpacetimeDB (GCE) | e2-standard-4, 50GB SSD | 730 | $0.134/hr + $8.50/mo disk | $106 |
| AI Inference (Cloud Run GPU) | L4, min=0 | ~200 active hrs | $0.88/hr (L4) | ~$15* |
| Frontend (GCS+CDN) | <1 GB stored, <10 GB egress | -- | $0.02/GB + $0.08/GB | <$1 |
| Auth (Cloud Run) | 256 MB | ~50 active hrs | $0.00002400/vCPU-s | <$2 |
| Matchmaking (Cloud Run) | 256 MB | ~50 active hrs | $0.00002400/vCPU-s | <$2 |
| Firestore | <1 GB, <100K ops/day | -- | Free tier | $0 |
| Firebase Auth | <50K MAU | -- | Free tier | $0 |
| GCS Replays | ~10 GB/mo | -- | $0.02/GB | <$1 |
| Cloud Logging | ~5 GB/mo | -- | Free first 50GB | $0 |
| Load Balancer | 1 forwarding rule | 730 | $0.025/hr | $18 |
| Cloud Armor | Standard tier | -- | $5/policy + $0.75/M req | ~$6 |
| **Total** | | | | **~$151** |

*Cloud Run GPU billing: billed only when processing requests. At 100 concurrent games with
~10 moves/min total, that is ~600 inferences/hour at ~0.8s each = ~480 GPU-seconds/hour.
Scale-to-zero means no cost during idle periods (nights, low traffic).

### Peak: 1000 Concurrent Games

| Component | Change | Additional Cost |
|:---|:---|:---|
| SpacetimeDB | Upgrade to e2-standard-8 | +$98/mo |
| AI Inference | Max 10 instances, ~1500 active hrs | +$45/mo (burst only) |
| Load Balancer | Same | $0 |
| Firestore | Exceeds free tier slightly | +$5/mo |
| **Total at peak** | | **~$199** |

### Cost Optimization Strategies

1. **Scale-to-zero inference**: Cloud Run GPU scales to 0 when no games active (saves ~60% vs always-on)
2. **Committed use discounts**: 1-year CUD on GCE for SpacetimeDB saves 37%
3. **Spot instances for training**: Already implemented (saves 60-91%)
4. **CDN caching**: Reduces origin fetches for static assets
5. **Firestore free tier**: Covers matchmaking at indie scale
6. **Firebase free tier**: Covers auth at indie scale
7. **Right-size inference**: The model is tiny (~50 MB); L4 is overkill but cheapest GPU on Cloud Run

## Terraform/IaC Structure

```
infra/terraform/
  |-- main.tf                      # Root module, wires everything together
  |-- variables.tf                 # Input variables
  |-- outputs.tf                   # Output values (URLs, IPs)
  |-- terraform.tf                 # Provider and backend config
  |-- environments/
  |     |-- prod/
  |     |     |-- terraform.tfvars # Production values
  |     |     |-- backend.tf       # GCS backend config
  |     |-- staging/
  |           |-- terraform.tfvars # Staging values
  |           |-- backend.tf       # GCS backend config
  |-- modules/
        |-- networking/            # VPC, subnets, firewall, NAT, LB
        |-- spacetimedb/           # GCE instance, disk, health check
        |-- ai-inference/          # Cloud Run GPU service
        |-- frontend/              # GCS bucket, CDN, DNS
        |-- auth/                  # Firebase setup, Cloud Run proxy
        |-- matchmaking/           # Cloud Run service, Firestore
        |-- replay-storage/        # GCS bucket with lifecycle
        |-- monitoring/            # Dashboards, alerts, uptime checks
        |-- ci-cd/                 # Cloud Build triggers
```
