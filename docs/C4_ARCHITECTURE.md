# C4 Architecture — Civ6 AlphaZero

Architecture documentation using the C4 model (Levels 1–3) with Mermaid diagrams.
This document reflects the production GCP cloud pipeline as of Phase 17 / Phase 0 bug fix.

---

## Level 1: System Context

Shows how external actors and systems interact with the Civ6 AlphaZero training platform.

```mermaid
C4Context
  title System Context — Civ6 AlphaZero

  Person(ml_engineer, "ML Engineer", "Monitors training runs, triggers production jobs, inspects W&B dashboards, and analyses model promotions.")
  Person_Ext(future_player, "Future Human Player", "Will challenge trained models via a web UI (planned).")

  System(civ6az, "Civ6 AlphaZero Platform", "AlphaZero self-play RL pipeline: orchestrates self-play, training, evaluation, and tournament rounds to produce progressively stronger Civ6 agents.")

  System_Ext(vertex_ai, "GCP Vertex AI", "Managed ML platform. Runs custom training jobs for self-play, training, evaluation, and tournament workers on T4 GPUs.")
  System_Ext(gcs, "Google Cloud Storage", "Stores versioned model weights, self-play trajectories, and mid-training checkpoints in the `civ6az-training` bucket.")
  System_Ext(wandb, "Weights & Biases", "Experiment tracking: loss curves, win rates, ELO standings, cost metrics, and W&B Groups per run.")
  System_Ext(artifact_registry, "GCP Artifact Registry", "Hosts Docker container images (e.g. `civ6az-training:v1.2.0`) used by every Vertex AI job.")

  Rel(ml_engineer, civ6az, "Launches runs, monitors progress, promotes models", "SSH / gcloud / W&B UI")
  Rel(civ6az, vertex_ai, "Submits custom training jobs (self-play, train, eval, tournament)", "Vertex AI Python SDK")
  Rel(civ6az, gcs, "Reads/writes models, trajectories, checkpoints", "GCS Python client")
  Rel(civ6az, wandb, "Logs metrics, run groups, ELO tables", "wandb SDK")
  Rel(civ6az, artifact_registry, "Pulls container image for each job", "Docker / ORAS")
  Rel(future_player, civ6az, "Will play against trained agents", "Web UI (planned)")

  UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

**Key points:**

- The ML Engineer is the primary operator: they start production runs via SSH on the orchestrator VM (`civ6az-orchestrator`) and monitor results in W&B.
- All compute-intensive work runs on Vertex AI custom jobs, not on the orchestrator VM itself.
- GCS is the single source of truth for model state. The orchestrator auto-detects the latest version at startup.
- W&B Groups tie all phases of a single production run together under one dashboard.

---

## Level 2: Container Diagram

Shows the major deployable units (containers) that compose the platform and how they communicate.

```mermaid
C4Container
  title Container Diagram — Civ6 AlphaZero Platform

  Person(ml_engineer, "ML Engineer", "Launches and monitors runs")

  System_Boundary(platform, "Civ6 AlphaZero Platform") {

    Container(orchestrator, "Orchestrator", "Python process on GCE e2-small VM", "Coordinates the self-play → training → evaluation loop. Submits Vertex AI jobs, polls for completion, decides promotions, and launches post-run tournament. Entry point: vertex_orchestrator.py")

    Container(selfplay_worker, "Self-Play Workers (×N)", "Vertex AI Custom Job — Docker container, T4 GPU / n1-standard-8", "Runs GumbelMCTS or BatchGumbelMCTS to generate game trajectories. Writes trajectory files to GCS. Supports vectorised parallel games via VectorizedMicroCivEnv. Entry point: selfplay_worker.py")

    Container(training_worker, "Training Worker", "Vertex AI Custom Job — Docker container, T4 GPU / n1-standard-8", "Loads champion weights from GCS (Phase 0 fix: before optimizer init), trains on trajectory TensorDataset with action-masked cross-entropy, saves new versioned model to GCS. Entry point: training_worker.py")

    Container(eval_worker, "Evaluation Worker", "Vertex AI Custom Job — Docker container, T4 GPU / n1-standard-8", "Runs arena games between new model and current champion. Promotes new model if win_rate >= 0.55. Supports max_retries=2 with exponential backoff. Entry point: evaluation_worker.py")

    Container(tournament_worker, "Tournament Worker", "Vertex AI Custom Job (post-run) or local CPU on VM", "Round-robin ELO tournament across all saved model versions. Produces ELO standings and W&B dashboard. Entry point: tournament_worker.py")

    ContainerDb(gcs_models, "GCS Model Storage", "Google Cloud Storage", "Versioned model weights + metadata (v0–v268+). Managed by GCSModelStorage. Auto-detected latest version at orchestrator startup.")

    ContainerDb(gcs_trajectories, "GCS Trajectory Storage", "Google Cloud Storage", "Self-play trajectory files (spatial, vector, action_mask, policy, value tensors). Managed by GCSTrajectoryStorage.")

    ContainerDb(gcs_checkpoints, "GCS Checkpoint Storage", "Google Cloud Storage", "Mid-training optimizer state for preemption recovery. Managed by GCSCheckpointStorage.")

    Container(container_image, "Docker Container Image", "GCP Artifact Registry — PyTorch 2.5 + CUDA 12.4", "Single image used by all Vertex AI workers. Contains alpha_zero package, requirements-container.txt deps. Tag: civ6az-training:v1.2.0")
  }

  System_Ext(vertex_ai, "GCP Vertex AI", "Managed ML job runner")
  System_Ext(wandb, "Weights & Biases", "Experiment tracking")

  Rel(ml_engineer, orchestrator, "SSH — starts detached run via setsid/disown", "gcloud compute ssh")
  Rel(orchestrator, vertex_ai, "Submits and polls custom training jobs", "Vertex AI Python SDK")
  Rel(vertex_ai, selfplay_worker, "Runs job with container image", "Docker pull + exec")
  Rel(vertex_ai, training_worker, "Runs job with container image", "Docker pull + exec")
  Rel(vertex_ai, eval_worker, "Runs job with container image", "Docker pull + exec")
  Rel(vertex_ai, tournament_worker, "Runs job with container image", "Docker pull + exec")

  Rel(selfplay_worker, gcs_trajectories, "Writes trajectory files per iteration/worker", "GCS client")
  Rel(selfplay_worker, gcs_models, "Reads current champion weights", "GCS client")

  Rel(training_worker, gcs_trajectories, "Reads trajectories for current iteration", "GCS client")
  Rel(training_worker, gcs_models, "Reads champion weights; writes new versioned model", "GCS client")
  Rel(training_worker, gcs_checkpoints, "Saves/restores mid-training optimizer state", "GCS client")

  Rel(eval_worker, gcs_models, "Reads new candidate + champion model weights", "GCS client")
  Rel(eval_worker, wandb, "Logs win_rate, draw_rate, games/min at each iteration", "wandb SDK")

  Rel(tournament_worker, gcs_models, "Reads all versioned models (stride-selectable)", "GCS client")
  Rel(tournament_worker, wandb, "Logs ELO standings, head-to-head matrix", "wandb SDK")

  Rel(orchestrator, wandb, "Logs iteration metrics, cost, phase timing, final summary", "wandb SDK")
  Rel(container_image, vertex_ai, "Pulled by all worker jobs", "Artifact Registry")

  UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="2")
```

**Key data flows:**

1. **Self-play phase**: Orchestrator submits N parallel Vertex AI self-play jobs. Each loads the current champion model, generates games with GumbelMCTS, and writes trajectory files to GCS.
2. **Training phase**: A single training job reads all new trajectories, loads champion weights before optimizer init (Phase 0 fix), trains for one epoch, and writes a new versioned model to GCS.
3. **Evaluation phase**: An evaluation job loads both the new model and the champion, plays arena games, and promotes the new model if it exceeds the win threshold. Failed evaluations are retried up to 2 times with backoff.
4. **Post-run tournament**: After all iterations, a tournament job loads every model version, runs a round-robin, and publishes ELO standings to W&B.

---

## Level 3: Component Diagram — Training Worker

Shows the internal components of `training_worker.py` and how they interact with each other and with external storage/monitoring.

```mermaid
C4Component
  title Component Diagram — Training Worker

  Container_Ext(gcs_models, "GCS Model Storage", "Google Cloud Storage", "Versioned model weights + metadata")
  Container_Ext(gcs_trajectories, "GCS Trajectory Storage", "Google Cloud Storage", "Self-play trajectory files")
  Container_Ext(gcs_checkpoints, "GCS Checkpoint Storage", "Google Cloud Storage", "Mid-training optimizer state")
  Container_Ext(wandb, "Weights & Biases", "Experiment tracking")

  Container_Boundary(training_worker, "Training Worker (training_worker.py)") {

    Component(parse_args, "parse_args()", "Python / argparse", "Parses CLI args from Vertex AI job spec: --iteration, --learning_rate, --mcts_type, --start_model_version, --fresh_start, --wandb_project, --wandb_group, --wandb_run_name, and all network/training hyperparams.")

    Component(load_champion, "_load_champion_weights()", "Python / GCSModelStorage", "Loads champion model weights from GCS BEFORE constructing the optimizer (Phase 0 critical fix). Prevents training from scratch by ensuring the network starts from the current best weights. Falls back to latest version if no explicit version specified.")

    Component(network, "Civ6TransformerNetwork", "PyTorch — transformer_network.py", "Dual-head transformer network. Spatial input: 13-channel 8x8 board via PatchEmbedEncoder or HexCrystallineEncoder. Vector input: 64-dim game state. Outputs: policy logits (376 actions) + scalar value. Architecture: embed_dim=64, num_layers=6.")

    Component(load_trajectories, "load_trajectories()", "PyTorch / GCSTrajectoryStorage", "Downloads trajectory files for the current iteration from GCS. Unpacks 5-tensor TensorDataset: (spatial, vector, action_mask, policies, values). Constructs DataLoader with configurable batch size and shuffling.")

    Component(train_epoch, "train_epoch()", "PyTorch training loop", "Runs one full training epoch. Policy loss: action-masked cross-entropy (target *= mask; target /= target.sum(-1).clamp(1e-8)). Value loss: MSE. Optimizer: AdamW with per-head LR groups (encoder, policy_head, value_head) and warmup_cosine schedule. Logs batch metrics at configurable interval.")

    Component(wandb_logger, "WandbLogger", "wandb SDK — wandb_logger.py", "Initialises W&B run with project/group/run_name from env vars. Logs per-batch and per-epoch metrics: policy_loss, value_loss, total_loss, learning_rate, gradient_norm. Finishes run on exit.")

    Component(save_model, "save_final_model()", "Python / GCSModelStorage", "Serialises model state_dict + model_config (dataclasses.asdict with Enum→value conversion). Uploads to GCS as the next versioned model. Metadata includes: iteration, training_loss, policy_loss, value_loss, timestamp, hyperparams.")

    Component(checkpoint_mgr, "GCSCheckpointStorage", "Python / GCS client", "Saves mid-training optimizer + scheduler state after each epoch for preemption recovery on Spot VMs. Deletes stale checkpoints after successful training completion.")
  }

  Rel(parse_args, load_champion, "Passes model version, network config, device", "function call")
  Rel(parse_args, load_trajectories, "Passes iteration, batch_size, GCS bucket", "function call")
  Rel(parse_args, train_epoch, "Passes LR, warmup_steps, value_lr_multiplier", "function call")
  Rel(parse_args, wandb_logger, "Passes project, group, run_name, tags", "function call")

  Rel(load_champion, gcs_models, "Downloads champion weights (auto-detects latest version)", "GCS get_blob")
  Rel(load_champion, network, "Loads state_dict into network BEFORE optimizer init", "model.load_state_dict()")

  Rel(load_trajectories, gcs_trajectories, "Downloads trajectory files for current iteration", "GCS list + get_blob")
  Rel(load_trajectories, train_epoch, "Provides TensorDataset DataLoader", "PyTorch DataLoader")

  Rel(train_epoch, network, "Forward pass: network(spatial, vector, action_mask)", "PyTorch forward")
  Rel(train_epoch, wandb_logger, "Logs batch-level losses and LR", "logger.log()")
  Rel(train_epoch, checkpoint_mgr, "Saves optimizer state after each epoch", "checkpoint_mgr.save()")

  Rel(checkpoint_mgr, gcs_checkpoints, "Uploads/downloads optimizer state", "GCS put/get_blob")

  Rel(save_model, network, "Reads final state_dict + model_config", "model.state_dict()")
  Rel(save_model, gcs_models, "Uploads new versioned model weights + metadata", "GCS put_blob")
  Rel(save_model, wandb_logger, "Logs final training summary metrics", "logger.log()")

  UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

**Component responsibilities:**

| Component | File | Key responsibility |
|-----------|------|--------------------|
| `parse_args()` | `training_worker.py` | CLI arg parsing; new fields use `getattr(args, field, default)` for backwards compat |
| `_load_champion_weights()` | `training_worker.py` | Phase 0 fix: load weights **before** `torch.optim.AdamW(...)` so optimizer starts from champion state |
| `Civ6TransformerNetwork` | `transformer_network.py` | 3-input forward: `(spatial, vector, action_mask)` → `(policy_logits, value)` |
| `load_trajectories()` | `training_worker.py` | 5-tensor TensorDataset; action_mask included since Phase 17 |
| `train_epoch()` | `training_worker.py` | Masked cross-entropy policy loss; per-head AdamW param groups; warmup_cosine LR |
| `WandbLogger` | `utils/wandb_logger.py` | W&B init/log/finish; reads `WANDB_GROUP`/`WANDB_RUN_NAME` env vars set by orchestrator |
| `save_final_model()` | `training_worker.py` | Enum→value conversion required before `dataclasses.asdict()` for JSON serialisation |
| `GCSCheckpointStorage` | `storage.py` | Uses `backend.upload_blob`/`download_blob`/`delete_blob` (not higher-level wrappers) |

---

## Architecture Decision Record

| Decision | Rationale |
|----------|-----------|
| Single Docker image for all workers | Simplifies versioning; avoids image drift between self-play and training |
| Orchestrator runs on e2-small VM (not Vertex AI) | Orchestrator is blocked in container mode (`entrypoint.py --mode=orchestrator` exits 1); VM avoids Vertex AI idle costs |
| `_load_champion_weights()` called before optimizer init (Phase 0) | AdamW initialises momentum buffers from current param values; loading weights after init means optimizer state diverges from the loaded weights, causing effective training from scratch |
| action_mask in trajectory + training | Prevents policy from assigning probability to illegal actions; ensures masked cross-entropy target sums to 1 over legal actions only |
| GumbelMCTS as default MCTS type | 13× faster than Civ6MCTS (no Dirichlet noise required); enables 200 games/iteration within Vertex AI job time limits |
| W&B Groups set by orchestrator env var | Ties all self-play/training/eval phases in one production run under a single W&B group for easy comparison |
| Auto-detect latest model from GCS | No `--start_model_version` needed at orchestrator startup; prevents off-by-one errors when resuming interrupted runs |
| Per-head LR multipliers (value_lr_multiplier=0.1) | Value head receives 10× lower LR to stabilise value correlation, which was highly variable (0.07–0.91) in early runs |
