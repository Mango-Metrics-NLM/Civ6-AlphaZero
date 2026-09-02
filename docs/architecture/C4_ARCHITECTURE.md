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
    Civ6Game["🎮 Civilization VI\n(Lua mod bridge: file IPC or\nheadless FireTuner socket)"]:::external
    HFHub["🤗 Hugging Face Hub\n(encoder / transformer / MuZero\nmodel repos, ELO dataset, Spaces)"]:::external

    User -->|"launches training runs,\nviews dashboards"| Core
    Core -->|"reads/writes model checkpoints\n& replay buffers"| GCS
    Core -->|"submits Vertex AI jobs"| VertexAI
    VertexAI -->|"worker output via GCS"| GCS
    Core -->|"logs metrics, params, artifacts"| MLflowServer
    Core -->|"receives game states,\nsends actions"| Civ6Game
    Core -->|"publishes checkpoints\n& datasets"| HFHub
    User -->|"browses MLflow UI"| MLflowServer
```

> **Weights & Biases is no longer an external system here.** It was retired on
> 2026-07-13; MLflow is the sole tracker. Only two read-only historical-run
> readers (`scripts/compare_training_runs.py`, `scripts/search_efficiency_report.py`)
> still reach W&B-hosted history, and they are documented as deprecated.

**Legend:**
| Colour | Meaning |
|--------|---------|
| Dark blue | Person (actor) |
| Mid blue | Civ6-AlphaZero system boundary |
| Grey | External system / service |

---

## Level 2 — Container Diagram

> **Scope:** Inside the Civ6-AlphaZero boundary. Shows the major deployable units (containers) and how they communicate.

**There is exactly one training image.** `cloudbuild.yaml` builds only
`dockerfiles/Dockerfile.vertex`, whose `alpha_zero.cloud.gcp.entrypoint` routes
`--mode` to one of four workers in the same image. The four per-role images
(`Dockerfile.base` / `.selfplay` / `.trainer` / `.evaluator`), `docker-compose.yml`
and the LEGACY `alpha_zero/cloud/{selfplay_worker,trainer,evaluator}.py` they
entrypointed were deleted on 2026-09-02 — nothing built or deployed them. The
second image, `dockerfiles/Dockerfile.inference`, is built separately by
`infra/cloudbuild-gameplay.yaml` and serves `alpha_zero/serving/`.

```mermaid
flowchart TB
    classDef worker fill:#1168BD,color:#fff,stroke:#0E5EA8
    classDef storage fill:#067D26,color:#fff,stroke:#055E1C
    classDef new fill:#D04A02,color:#fff,stroke:#A33A01
    classDef infra fill:#555,color:#fff,stroke:#333

    GCS["☁️ GCS Storage\n(checkpoints, replays,\nELO snapshots)"]:::storage
    MLiteDB["🗄️ MLflow SQLite DB\n(mlflow.db)"]:::new

    subgraph VertexAI["Vertex AI — one image: dockerfiles/Dockerfile.vertex"]
        Entry["Container entrypoint\nalpha_zero/cloud/gcp/entrypoint.py\n--mode routes to one worker\n(orchestrator mode is refused here)"]:::infra
        SPWorker["--mode=selfplay\ncloud/gcp/selfplay_worker.py\n(Gumbel / BatchGumbel MCTS + MicroCiv)"]:::worker
        TrainWorker["--mode=training\ncloud/gcp/training_worker.py\n(gradient updates, champion promotion)"]:::worker
        TournWorker["--mode=tournament\ncloud/gcp/tournament_worker.py\n(round-robin ELO evaluation)"]:::worker
        EvalWorker["--mode=evaluation\ncloud/gcp/evaluation_worker.py\n(arena.py scoring vs incumbent)"]:::worker
    end

    subgraph Local["Local / CI"]
        VOrch["Vertex Orchestrator\ncloud/gcp/vertex_orchestrator.py\n(job sequencing + preemption recovery)\nruns LOCALLY, not in the image"]:::infra
        MonLayer["MLflow Monitoring Layer\nalpha_zero/monitoring/\nMLflowLogger · AlphaZeroMLflowEvaluator"]:::new
        WebUI["Web UI\nalpha_zero/web/\n(FastAPI + WebSocket game UI)"]:::worker
        Bridge["Civ6 Bridge\nalpha_zero/bridge/\n(file IPC or FireTuner socket)"]:::worker
    end

    VOrch -->|"submit Vertex job"| Entry
    Entry --> SPWorker
    Entry --> TrainWorker
    Entry --> TournWorker
    Entry --> EvalWorker

    SPWorker -->|"write trajectories"| GCS
    TrainWorker -->|"read trajectories,\nwrite checkpoints"| GCS
    TournWorker -->|"read checkpoints,\nwrite ELO"| GCS
    EvalWorker -->|"read checkpoints"| GCS

    EvalWorker -->|"log evaluation metrics"| MonLayer
    TournWorker -->|"log ELO metrics"| MonLayer
    TrainWorker -->|"log training loss"| MonLayer
    MonLayer -->|"persist experiments"| MLiteDB

    Bridge -->|"game state / actions"| WebUI
```

**Container responsibilities:**

| Unit | `--mode` | Technology | Hosts |
|------|----------|-----------|-------|
| Self-Play Worker | `selfplay` | Python / PyTorch / Docker (Vertex) | Gumbel or BatchGumbel MCTS over MicroCiv |
| Training Worker | `training` | Python / PyTorch / Docker (Vertex) | SGD updates, weight promotion |
| Evaluation Worker | `evaluation` | Python (Vertex) | Arena scoring vs the incumbent |
| Tournament Worker | `tournament` | Python (Vertex) | Round-robin, ELO snapshots |
| Vertex Orchestrator | — (local) | Python / google-cloud-aiplatform | Job sequencing, preemption recovery |
| MLflow Monitoring Layer | — (local) | Python / MLflow / SQLite | Experiment tracking (sole tracker) |
| Web UI | — (local) | FastAPI + WebSocket | Human-vs-AI browser interface |
| Civ6 Bridge | — (local) | Python / Lua mod | File IPC or headless FireTuner socket |

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
        MLflowConfig["MLflowConfig\nconfig.py\n(MonitoringConfig = back-compat alias)\n• tracking_uri, experiment_name\n• SQLite backend settings"]:::comp
        AZEvaluator["AlphaZeroMLflowEvaluator\nmlflow_evaluator.py\ncompute_metrics() →\npolicy_entropy · value_mae\npolicy_top1_acc · action_diversity"]:::comp
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

## Level 3 — Component Diagram (MCTS Search Family)

`alpha_zero/core/civ6_mcts.py` held five search variants, four protocols and three
configuration dataclasses in 2,443 lines until 2026-08-30. It is now
`alpha_zero/core/mcts/`, one module per variant, with `civ6_mcts.py` retained as a
re-export shim so every historical import path resolves to the same objects.

The layering below is **enforced, not aspirational**: `tests/unit/test_mcts_split_backcompat.py`
parses module-scope imports and fails if `gumbel` or `muzero` ever import `base`, or if
`muzero` imports the `gumbel` search module. That is why the shared Gumbel-Top-k helpers
live in their own leaf module — `MuZeroMCTS` calls them, and putting them in `gumbel.py`
would have coupled the MuZero variant to the Gumbel variant.

```mermaid
flowchart TD
    classDef leaf fill:#e8f4ea,stroke:#2d6a4f,color:#1b4332
    classDef variant fill:#e7eefc,stroke:#1d4ed8,color:#1e3a8a
    classDef shim fill:#fdf3e3,stroke:#b45309,color:#7c2d12
    classDef ext fill:#f3f4f6,stroke:#6b7280,color:#374151,stroke-dasharray: 4 3

    subgraph Leaves["Leaves — no sibling imports"]
        Protocols["mcts_protocols.py<br/>MCTSNetwork · MCTSEnvironment<br/>ActionSpace · MuZeroNetworkProtocol"]:::leaf
        GumbelOps["gumbel_ops.py<br/>gumbel_sample_actions<br/>compute_completed_q"]:::leaf
        Base["base.py<br/>MCTSConfig · MCTSNode<br/>_env_current_player"]:::leaf
    end

    subgraph Variants["Search variants"]
        Puct["puct.py<br/>Civ6MCTS"]:::variant
        Simple["simple.py<br/>SimpleMCTS"]:::variant
        Gumbel["gumbel.py<br/>GumbelMCTS · GumbelMCTSNode<br/>GumbelMCTSConfig"]:::variant
        MuZero["muzero.py<br/>MuZeroMCTS · MuZeroNode<br/>MuZeroMCTSConfig"]:::variant
    end

    Factory["mcts_factory.py<br/>create_mcts() · _MCTS_REGISTRY<br/><i>unchanged by the split</i>"]:::ext
    Shim["civ6_mcts.py<br/>re-export shim, explicit __all__"]:::shim

    Base --> Puct
    Base --> Simple
    Protocols --> Puct
    Protocols --> Gumbel
    Protocols --> MuZero
    GumbelOps --> Gumbel
    GumbelOps --> MuZero

    Puct --> Shim
    Simple --> Shim
    Gumbel --> Shim
    MuZero --> Shim
    Base --> Shim
    Protocols --> Shim
    GumbelOps --> Shim

    Shim -.->|"lazy import inside create_mcts()"| Factory
```

Line counts below were re-measured on 2026-09-02 (`wc -l alpha_zero/core/mcts/*.py`);
they grew with the 2026-08-30 change that gave every search its own RNG.

| Module | Lines | Contents | Depends on |
|---|---:|---|---|
| `mcts_protocols.py` | 161 | `MCTSNetwork`, `MuZeroNetworkProtocol`, `ActionSpace`, `MCTSEnvironment` | — |
| `gumbel_ops.py` | 112 | `gumbel_sample_actions`, `compute_completed_q` | — |
| `base.py` | 223 | `MCTSConfig`, `MCTSNode`, `_env_current_player` | — |
| `puct.py` | 401 | `Civ6MCTS` | `base`, `mcts_protocols` |
| `simple.py` | 199 | `SimpleMCTS` | `base`, `mcts_protocols` |
| `gumbel.py` | 572 | `GumbelMCTS`, `GumbelMCTSNode`, `GumbelMCTSConfig` | `gumbel_ops`, `mcts_protocols` |
| `muzero.py` | 888 | `MuZeroMCTS`, `MuZeroNode`, `MuZeroMCTSConfig` | `gumbel_ops`, `mcts_protocols` |

**Extension seam.** New variants are added through `create_mcts(...)`'s `_MCTS_REGISTRY`
(charter Invariant 1); callers never branch on a concrete MCTS type. The split left that
factory untouched — its lazy import resolves through the shim — so the diff outside the
moved code is the shim and the documentation.

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

*Last revised: 2026-09-02 — Tier A dead-code removal (44 modules / 21,903 lines; the
four per-role Dockerfiles and `docker-compose.yml` with them), W&B removed from the
context diagram, one-image container model made explicit.*
*Architecture version: Phase 21 (Hexcrystal Encoder, SpaceTimeDB, & Spaces Integrations) + MCTS package decomposition*
