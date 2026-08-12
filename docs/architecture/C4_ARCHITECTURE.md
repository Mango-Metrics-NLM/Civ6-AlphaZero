# C4 Architecture Diagrams — Civ6-AlphaZero

> **Standard:** [C4 Model](https://c4model.com/) — four levels of abstraction from system context down to code/data flow.
> Each diagram is a Mermaid flowchart. Render in GitHub, VS Code with the Mermaid extension, or at [mermaid.live](https://mermaid.live).

---

## Level 1 — System Context Diagram

> **Scope:** Civ6-AlphaZero as a black box. Shows the people and external systems that interact with it.

```mermaid
flowchart TB
    classDef person fill:#08427B,color:#fff,stroke:#073B6F
    classDef system fill:#1168BD,color:#fff,stroke:#0E5EA8
    classDef external fill:#999,color:#fff,stroke:#6B6B6B

    User["👤 Researcher / Engineer\n(runs training, inspects metrics)"]:::person

    subgraph CIV6AZ["Civ6-AlphaZero System"]
        direction TB
        Core["AlphaZero Training System\n(self-play, MCTS, neural network training,\nevaluation, monitoring)"]:::system
    end

    GCS["☁️ Google Cloud Storage\n(model checkpoints, replay buffers,\ntrajectories, ELO snapshots)"]:::external
    VertexAI["☁️ Google Vertex AI\n(distributed self-play workers,\ntraining jobs, evaluation jobs)"]:::external
    MLflowServer["📊 MLflow Server\n(experiment tracking,\nSQLite backend: mlflow.db)"]:::external
    Civ6Game["🎮 Civilization VI\n(Lua mod bridge via file IPC;\nproduces game states + actions)"]:::external
    WandB["📈 Weights & Biases\n(training loss curves,\nhyperparameter sweeps)"]:::external

    User -->|"launches training runs,\nviews dashboards"| Core
    Core -->|"reads/writes model checkpoints\n& replay buffers"| GCS
    Core -->|"submits Vertex AI jobs"| VertexAI
    VertexAI -->|"worker output via GCS"| GCS
    Core -->|"logs metrics, params, artifacts"| MLflowServer
    Core -->|"receives game states,\nsends actions"| Civ6Game
    Core -->|"logs loss / ELO metrics"| WandB
    User -->|"browses MLflow UI"| MLflowServer
```

**Legend:**
| Colour | Meaning |
|--------|---------|
| Dark blue | Person (actor) |
| Mid blue | Civ6-AlphaZero system boundary |
| Grey | External system / service |

---

## Level 2 — Container Diagram

> **Scope:** Inside the Civ6-AlphaZero boundary. Shows the major deployable units (containers) and how they communicate.

```mermaid
flowchart TB
    classDef worker fill:#1168BD,color:#fff,stroke:#0E5EA8
    classDef storage fill:#067D26,color:#fff,stroke:#055E1C
    classDef new fill:#D04A02,color:#fff,stroke:#A33A01
    classDef infra fill:#555,color:#fff,stroke:#333

    GCS["☁️ GCS Storage\n(checkpoints, replays,\nELO snapshots)"]:::storage
    MLiteDB["🗄️ MLflow SQLite DB\n(mlflow.db)\n[NEW]"]:::new

    subgraph VertexAI["Vertex AI (Cloud)"]
        SPWorker["Self-Play Workers\nalpha_zero/cloud/gcp/selfplay_worker.py\n(BatchGumbelMCTS + VectorizedMicroCivEnv)"]:::worker
        TrainWorker["Training Worker\nalpha_zero/cloud/gcp/training_worker.py\n(gradient updates, champion promotion)"]:::worker
        TournWorker["Tournament Worker\nalpha_zero/cloud/gcp/tournament_worker.py\n(round-robin ELO evaluation)"]:::worker
        EvalWorker["Evaluation Worker\nalpha_zero/cloud/gcp/evaluation_worker.py\n(win-rate vs random / heuristic)"]:::worker
        VOrch["Vertex Orchestrator\nalpha_zero/cloud/gcp/vertex_orchestrator.py\n(job sequencing + preemption recovery)"]:::infra
    end

    subgraph Local["Local / CI"]
        MonLayer["MLflow Monitoring Layer\nalpha_zero/monitoring/\nMLflowLogger · AlphaZeroMLflowEvaluator\n[NEW]"]:::new
        WebUI["Web UI\nalpha_zero/web/\n(FastAPI + WebSocket game UI)"]:::worker
        Bridge["Civ6 Bridge\nalpha_zero/bridge/\n(Lua mod file IPC)"]:::worker
    end

    VOrch -->|"launch jobs"| SPWorker
    VOrch -->|"launch jobs"| TrainWorker
    VOrch -->|"launch jobs"| TournWorker
    VOrch -->|"launch jobs"| EvalWorker

    SPWorker -->|"write trajectories"| GCS
    TrainWorker -->|"read trajectories,\nwrite checkpoints"| GCS
    TournWorker -->|"read checkpoints,\nwrite ELO"| GCS
    EvalWorker -->|"read checkpoints"| GCS

    EvalWorker -->|"log evaluation metrics"| MonLayer
    TournWorker -->|"log ELO metrics"| MonLayer
    TrainWorker -->|"log training loss"| MonLayer
    MonLayer -->|"persist experiments"| MLiteDB

    Bridge -->|"file IPC"| WebUI
```

**Container responsibilities:**

| Container | Technology | Hosts |
|-----------|-----------|-------|
| Self-Play Workers | Python / PyTorch / Docker (Vertex) | BatchGumbelMCTS + VectorizedMicroCivEnv |
| Training Worker | Python / PyTorch / Docker (Vertex) | SGD updates, weight promotion |
| Vertex Orchestrator | Python / google-cloud-aiplatform | Job sequencing, preemption |
| Tournament Worker | Python / ELO library | Round-robin, ELO snapshots |
| Evaluation Worker | Python | Win-rate vs random/heuristic |
| MLflow Monitoring Layer | Python / MLflow / SQLite | Experiment tracking **(NEW)** |
| Web UI | FastAPI + WebSocket | Human-vs-AI browser interface |
| Civ6 Bridge | Python / Lua mod | File-based IPC to real Civ6 |

---

## Level 3 — Component Diagram (MLflow Monitoring Layer)

> **Scope:** Inside `alpha_zero/monitoring/` and the scripts that drive it. Focuses on the Phase 19 additions.

```mermaid
flowchart LR
    classDef comp fill:#1168BD,color:#fff,stroke:#0E5EA8
    classDef script fill:#444,color:#fff,stroke:#222
    classDef ext fill:#999,color:#fff,stroke:#6B6B6B
    classDef util fill:#067D26,color:#fff,stroke:#055E1C

    subgraph Monitoring["alpha_zero/monitoring/"]
        MLflowLogger["MLflowLogger\nmlflow_logger.py\n• start_run() / end_run()\n• log_evaluation_result() → 14 metrics\n• FINISHED / FAILED lifecycle"]:::comp
        MLflowConfig["MLflowMonitoringConfig\nconfig.py\n• tracking_uri, experiment_name\n• SQLite backend settings"]:::comp
        AZEvaluator["AlphaZeroMLflowEvaluator\nmlflow_evaluator.py\n• policy_entropy()\n• value_mae()\n• top1_accuracy()"]:::comp
    end

    subgraph Scripts["scripts/"]
        AnalyseScript["analyse_training_run.py\n(reads GCS runs → MLflow)"]:::script
        PostRunScript["post_run_evaluation.py\n(ELO + baseline eval)"]:::script
        BaselineScript["run_baseline_eval.py\n(random / heuristic eval)"]:::script
        MigrateScript["migrate_mlflow_to_sqlite.py\n(mlruns/ → mlflow.db)"]:::script
    end

    subgraph Utils["alpha_zero/utils/"]
        EnvLoader["env_loader.py\nZero-dependency .env loader\nCIV6AZ_ convention"]:::util
    end

    EvalPipeline["EvaluationPipeline\n(existing)"]:::ext
    Civ6MCTS["Civ6MCTS\n(existing)"]:::ext
    MLflowServer["MLflow Server\n(sqlite:///mlflow.db)"]:::ext
    GCSStorage["GCS Storage\n(model checkpoints)"]:::ext
    DotEnvFile[".env / OS environment"]:::ext

    MLflowConfig --> MLflowLogger
    AZEvaluator --> MLflowLogger
    MLflowLogger -->|"mlflow.log_metrics()"| MLflowServer

    AnalyseScript -->|"reads runs"| GCSStorage
    AnalyseScript --> MLflowLogger
    PostRunScript --> MLflowLogger
    PostRunScript --> EvalPipeline
    PostRunScript --> Civ6MCTS
    BaselineScript --> EvalPipeline
    BaselineScript --> GCSStorage
    MigrateScript -->|"migrate"| MLflowServer

    EnvLoader -->|"loads"| DotEnvFile
    MLflowConfig -->|"reads via"| EnvLoader
```

**14 metrics logged by `log_evaluation_result()`:**

| Category | Metrics |
|----------|---------|
| Counts (4) | `n_games`, `n_wins`, `n_losses`, `n_draws` |
| Normalised rates (3) | `win_rate` (0-1), `loss_rate`, `draw_rate` |
| Game stats (7) | `avg_game_length`, `max_game_length`, `min_game_length`, `avg_score_diff`, `total_turns`, `games_per_second`, `eval_duration_s` |

---

## Data Flow Diagram — 50-Iteration Training Run

> **Scope:** End-to-end data flow for a full 50-iteration GCS-backed training run (v042–v091), including how evaluation results reach MLflow.

```mermaid
flowchart TD
    classDef gcs fill:#067D26,color:#fff,stroke:#055E1C
    classDef worker fill:#1168BD,color:#fff,stroke:#0E5EA8
    classDef mlflow fill:#D04A02,color:#fff,stroke:#A33A01
    classDef eval fill:#555,color:#fff,stroke:#333

    GCSCheckpoints["☁️ GCS\nModel Checkpoints\nv042 → v091\n(50 iterations)"]:::gcs

    subgraph SelfPlayLoop["Self-Play Loop (per iteration)"]
        SP["Self-Play Workers\n(BatchGumbelMCTS)\n→ trajectories to GCS"]:::worker
        Train["Training Worker\n(SGD, 1000 steps)\n→ new checkpoint to GCS"]:::worker
        SP --> Train
    end

    GCSCheckpoints -->|"load champion weights"| SP
    Train -->|"write new checkpoint"| GCSCheckpoints

    subgraph Tournament["Tournament (post-run)"]
        TW["Tournament Worker\nround-robin\nv042 vs v043 vs … vs v091"]:::worker
        ELO["ELO Rankings\n(champion = v091)"]:::eval
        TW --> ELO
    end

    GCSCheckpoints -->|"read all checkpoints"| TW

    subgraph Evaluation["Evaluation (per checkpoint)"]
        EP["EvaluationPipeline\nevaluate_vs_random()\nn_games=400"]:::eval
        WRR["WinRateResult\n• wins, losses, draws\n• win_rate (0-100 scale)"]:::eval
        EP --> WRR
    end

    GCSCheckpoints -->|"load checkpoint"| EP

    subgraph MLflowLogging["MLflow Logging"]
        LER["log_evaluation_result()\n14 metrics per run"]:::mlflow
        MLflowDB["MLflow SQLite DB\n(mlflow.db)\nexperiment per checkpoint"]:::mlflow
        LER --> MLflowDB
    end

    WRR -->|"normalise to 0-1"| LER
    ELO -->|"log ELO scores"| LER

    User["👤 Researcher\nmlflow ui --backend-store-uri sqlite:///mlflow.db"]
    MLflowDB -->|"visualise"| User
```

**Key invariants:**
- `WinRateResult.win_rate` is on a **0–100 scale**; `log_evaluation_result()` normalises to **0–1** before storing in MLflow.
- Each MLflow run corresponds to one model checkpoint (one iteration).
- `resolve_run_id()` auto-detects the latest champion from GCS — no hardcoded UUIDs.

---

## Checkpoint Validation & Hub Publishing

```mermaid
flowchart TD
    classDef storage fill:#067D26,color:#fff,stroke:#055E1C
    classDef logic fill:#1168BD,color:#fff,stroke:#0E5EA8
    classDef external fill:#999,color:#fff,stroke:#6B6B6B

    GCS["☁️ GCS Checkpoints"]:::storage
    HF["🤗 Hugging Face Hub"]:::external

    subgraph LoadPath["Worker Load Path (evaluation_worker / selfplay_worker)"]
        Load["load_state_dict(strict=False)"]:::logic
        Classify["classify_encoder_keys()"]:::logic
        Validate["validate_encoder_state()"]:::logic
        Warn["Log mismatch warning / error"]:::logic
        
        Load --> Classify
        Classify --> Validate
        Validate -->|"mismatch detected"| Warn
    end

    subgraph ExportPath["Export & Publish (upload_encoder_to_hub.py)"]
        Extract["Extract encoder sub-module"]:::logic
        Verify["Verify safe-tensors only"]:::logic
        ModelCard["Assemble Model Card README"]:::logic
        Publish["HfApi.upload_folder()"]:::logic
        
        Extract --> Verify
        Verify --> ModelCard
        ModelCard --> Publish
    end

    GCS -->|"checkpoint file"| Load
    GCS -->|"checkpoint file"| Extract
    Publish -->|"safetensors & config"| HF
```

---

*Generated: 2026-08-11*
*Architecture version: Phase 21 (Hexcrystal Encoder, SpaceTimeDB, & Spaces Integrations)*
