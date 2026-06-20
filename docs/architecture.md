# C4 Architecture Documentation - Civ6 AlphaZero AI

> Complete architectural visualization using C4 model and Mermaid diagrams.
> Reflects all 6 implementation phases including GCP cloud infrastructure.

> **Last reconciled: 2026-06-20 (v1.7.0).** For the authoritative, up-to-date list of
> `alpha_zero/` submodules and their responsibilities, see
> [`MODULE_REFERENCE.md`](MODULE_REFERENCE.md). The phase table and diagrams below are a
> historical/architectural narrative and may lag the latest module additions.

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Core AlphaZero (MCTS + Neural Network) | ✅ Complete |
| Phase 2 | Hierarchical Reinforcement Learning | ✅ Complete |
| Phase 3 | Training Improvements (Curriculum, Rewards) | ✅ Complete |
| Phase 4 | Unified Training Orchestrator | ✅ Complete |
| Phase 5a | RAG-LLM Integration | ✅ Complete |
| Phase 5b | GCP/Vertex AI Cloud Infrastructure | ✅ Complete |
| Phase 5c | Cost Optimization & Resource Management | ✅ Complete |
| Phase 6 | ~~Agentic Framework~~ (removed in Phase 11) | ❌ Removed |
| Phase 7 | Production PBT Scaling & Validation | ✅ Complete |
| Phase 8 | Training Safeguards & Automated Intervention | ✅ Complete |
| Phase 9 | MuZero Integration Bridge | ✅ Complete |
| Phase 10 | Tournament & Matchmaking System | ✅ Complete |
| Phase 11 | Code Quality & Tech Debt Reduction | ✅ Complete |
| Phase 12-Prep | Docker & Vertex AI Smoke Test | ✅ Complete |
| Phase 13 | Multi-Iteration Training & Post-Run Evaluation | ✅ Complete |
| Phase 14 | Learning Signal Validation (20-iteration run) | ✅ Complete |
| Phase 15 | Scaled Training & Value Head Stabilization | ✅ Complete |

**Codebase Statistics:** ~55,000 LOC core, 3,678+ tests, Docker v0.12.0

---

## Level 1: System Context Diagram

```mermaid
C4Context
    title System Context - Civ6 AlphaZero AI Framework

    Person(researcher, "Researcher/Developer", "Configures training, analyzes results, evaluates AI performance")
    Person(player, "Human Player", "Plays against or observes the AI agent")

    System(civ6ai, "Civ6 AlphaZero AI", "AlphaZero-based AI for Civilization VI using hierarchical RL, MCTS, and neural networks")

    System_Ext(civ6, "Civilization VI", "Firaxis game via Lua mod bridge")
    System_Ext(unciv, "Unciv", "Open-source Civ clone via JSON interface")
    System_Ext(wandb, "Weights & Biases", "Experiment tracking and metrics")
    System_Ext(vertex, "Vertex AI", "GCP managed ML platform for distributed training")
    System_Ext(gcs, "Google Cloud Storage", "Stores models, trajectories, and evaluation results")
    System_Ext(llm, "LLM APIs", "OpenAI/Anthropic for strategy advice")
    System_Ext(vectordb, "Vector Databases", "ChromaDB/FAISS for RAG knowledge")

    Rel(researcher, civ6ai, "Trains, configures, monitors")
    Rel(player, civ6ai, "Plays against, observes")
    Rel(civ6ai, civ6, "Game state, actions via Lua IPC")
    Rel(civ6ai, unciv, "Game state via JSON")
    Rel(civ6ai, wandb, "Logs metrics, artifacts")
    Rel(civ6ai, vertex, "Submits training jobs")
    Rel(civ6ai, gcs, "Stores/loads models and data")
    Rel(civ6ai, llm, "Strategy queries")
    Rel(civ6ai, vectordb, "Knowledge retrieval")
```

---

## Level 2: Container Diagram

```mermaid
C4Container
    title Container Diagram - Civ6 AlphaZero System

    Person(user, "User/Researcher")

    System_Boundary(c1, "Civ6 AlphaZero System") {
        Container(orchestrator, "Training Orchestrator", "Python", "Unified training coordination, hyperparameter scheduling")
        Container(selfplay, "Self-Play Engine", "Python/multiprocessing", "Generates training data via MCTS games")
        Container(learner, "Neural Network Trainer", "PyTorch", "Optimizes policy and value networks")
        Container(evaluator, "Agent Evaluator", "Python", "Tournament-based performance measurement")
        
        Container(mcts, "MCTS Engine", "Python/NumPy", "Monte Carlo Tree Search with Gumbel/PUCT")
        Container(network, "Neural Networks", "PyTorch", "ResNet, Transformer, Hierarchical architectures")
        Container(factory, "MCTS Factory", "Python", "Creates MCTS variants by name")
        Container(buffer, "Replay Buffer", "Python", "Prioritized experience storage")
        
        Container(hierarchical, "Hierarchical RL", "Python", "3-level decision hierarchy with goal conditioning")
        Container(mechanics, "Game Mechanics", "Python", "11 Civ6 subsystems simulation")
        Container(rag, "RAG Pipeline", "Python/async", "Knowledge retrieval and LLM integration")
        
        Container(microciv, "MicroCiv Environment", "Python/Gymnasium", "Simplified Civ6 for fast training")
        Container(bridge, "Civ6 Bridge", "Python/Lua", "Real game file-based IPC")
        
        Container(cloud, "GCP Cloud Layer", "Python/GCS/Vertex", "Distributed training infrastructure")
        Container(pbt, "Population-Based Training", "Python", "Hyperparameter evolution")
        Container(agentic, "Agentic Framework", "Python", "Claude Code development workflows")
    }

    System_Ext(wandb, "W&B Cloud")
    System_Ext(vertex, "Vertex AI", "GCP managed ML platform")
    System_Ext(gcs, "Google Cloud Storage", "Cloud object storage")

    Rel(user, orchestrator, "Configures, monitors")
    Rel(orchestrator, selfplay, "Triggers games")
    Rel(orchestrator, learner, "Triggers training")
    Rel(orchestrator, evaluator, "Triggers evaluation")
    Rel(orchestrator, pbt, "Hyperparameter optimization")
    
    Rel(selfplay, mcts, "Tree search")
    Rel(selfplay, buffer, "Stores experience")
    Rel(mcts, network, "Policy/value queries")
    Rel(mcts, hierarchical, "Multi-level search")
    
    Rel(learner, network, "Updates weights")
    Rel(learner, buffer, "Samples batches")
    
    Rel(selfplay, microciv, "Training games")
    Rel(selfplay, bridge, "Real Civ6 games")
    Rel(microciv, mechanics, "Game rules")
    
    Rel(selfplay, factory, "Creates MCTS instances")
    Rel(evaluator, factory, "Creates MCTS instances")
    
    Rel(orchestrator, rag, "Strategy context")
    Rel(orchestrator, cloud, "Distributed execution")
    Rel(cloud, vertex, "Submits jobs")
    Rel(cloud, gcs, "Storage operations")
    Rel(orchestrator, wandb, "Metrics logging")
```

## Cloud Training Pipeline

```mermaid
sequenceDiagram
    participant U as User / CLI
    participant O as Vertex Orchestrator
    participant SP as Self-Play Workers (N)
    participant GCS as Google Cloud Storage
    participant TR as Training Worker
    participant EV as Evaluation Worker
    participant WB as Weights & Biases

    U->>O: python -m vertex_orchestrator --iterations=K
    O->>WB: Init orchestrator W&B run
    loop For each iteration
        O->>SP: Launch N self-play jobs (Vertex AI)
        SP->>SP: MCTS self-play games
        SP->>GCS: Save trajectories
        SP-->>O: Jobs complete

        O->>TR: Launch training job (Vertex AI)
        TR->>GCS: Load trajectories
        TR->>TR: Train neural network
        TR->>GCS: Save new model (v_new) + model_config
        TR-->>O: Job complete

        O->>EV: Launch evaluation job (Vertex AI)
        EV->>GCS: Load model v_new and v_old
        EV->>EV: Arena: v_new vs v_old
        EV->>GCS: Save results JSON
        EV-->>O: Job complete

        O->>O: Accept/reject model (win_rate > threshold?)
        O->>WB: Log iteration metrics + cost
    end

    opt Post-Run Tournament (--post_run_tournament)
        O->>GCS: Load all model versions
        O->>O: Round-robin arena (local CPU)
        O->>O: Compute ELO ratings
        O->>GCS: Save tournament results JSON
        O->>WB: Log ELO rankings + match results
    end

    O->>WB: Log final summary
    O-->>U: Pipeline complete
```

---

## Level 3: Component Diagram - Core AI Layer

```mermaid
C4Component
    title Component Diagram - Core AI Components

    Container_Boundary(core, "Core AI Layer") {
        Component(civ6network, "Civ6Network", "PyTorch nn.Module", "84-channel ResNet with attention, dual policy/value heads")
        Component(transformer, "TransformerNetwork", "PyTorch", "ViT-based with RoPE, MoE, SwiGLU activation")
        Component(microcivnet, "MicroCivNetwork", "PyTorch", "13-channel simplified network for fast training")
        Component(factory, "NetworkFactory", "Python", "Dynamic network instantiation from config")
        
        Component(civ6mcts, "Civ6MCTS", "Python/NumPy", "PUCT-based tree search with virtual loss")
        Component(gumbel, "GumbelMCTS", "Python", "Gumbel AlphaZero variant for improved exploration")
        Component(node, "MCTSNode", "Python dataclass", "Tree node with prior, visits, value_sum")
        
        Component(actions, "Civ6Actions", "Python", "Hierarchical 4-level action space (50k+ actions)")
        Component(pruning, "ActionPruner", "Python", "Feasibility, strategic, resource-based pruning")
        Component(macro, "MacroActions", "Python", "Multi-turn action abstractions")
    }

    Rel(factory, civ6network, "Creates")
    Rel(factory, transformer, "Creates")
    Rel(factory, microcivnet, "Creates")
    
    Rel(civ6mcts, node, "Manages tree")
    Rel(gumbel, node, "Manages tree")
    Rel(civ6mcts, civ6network, "Evaluates states")
    Rel(civ6mcts, pruning, "Reduces action space")
    Rel(pruning, actions, "Filters valid actions")
    Rel(macro, actions, "Decomposes to primitives")
```

---

## Level 3: Component Diagram - Hierarchical RL Layer

```mermaid
C4Component
    title Component Diagram - Hierarchical RL (Phase 2)

    Container_Boundary(hier, "Hierarchical RL Layer") {
        Component(controller, "HierarchicalController", "Python", "Coordinates 3-level decision hierarchy")
        
        Component(strategic, "StrategicLevel", "Python", "30-100 turn horizon, victory/expansion goals")
        Component(tactical, "TacticalLevel", "Python", "5-15 turn horizon, regional control")
        Component(operational, "OperationalLevel", "Python", "1-3 turn horizon, unit actions")
        
        Component(goals, "GoalManager", "Python", "Goal stack, propagation, achievement tracking")
        Component(encoders, "LevelEncoders", "PyTorch", "Per-level state feature extraction")
        Component(hiernet, "HierarchicalNetwork", "PyTorch", "Multi-level policy with cross-level context")
        Component(hiermcts, "HierarchicalMCTS", "Python", "Option-based temporal abstraction")
    }

    Rel(controller, strategic, "Sets long-term goals")
    Rel(strategic, tactical, "Subgoal delegation")
    Rel(tactical, operational, "Action commands")
    Rel(operational, tactical, "Feedback")
    Rel(tactical, strategic, "Goal completion")
    
    Rel(controller, goals, "Manages goal stack")
    Rel(hiernet, encoders, "Level-specific encoding")
    Rel(hiermcts, hiernet, "Multi-level policy queries")
```

---

## Level 3: Component Diagram - GCP Cloud Infrastructure

```mermaid
C4Component
    title Component Diagram - GCP Cloud Infrastructure (Phase 5b/5c)

    Container_Boundary(cloud, "GCP Cloud Layer") {
        Component(gcpconfig, "GCPConfig", "Python dataclass", "Environment-based configuration")
        Component(vertexclient, "VertexAIClient", "Python/aiplatform", "Job submission abstraction")
        Component(vertexjobs, "VertexJobs", "Python", "SelfPlay and Training job wrappers")

        Component(gcsstorage, "GCSStorage", "Python/GCS", "Model, trajectory, checkpoint backends")
        Component(modelstorage, "GCSModelStorage", "Python", "Versioned model weights with latest.json")
        Component(trajstorage, "GCSTrajectoryStorage", "Python", "Sharded, LZMA-compressed trajectories")

        Component(selfplayworker, "SelfPlayWorker", "Python", "Stateless parallel game generation")
        Component(trainingworker, "TrainingWorker", "Python", "Centralized learner with checkpointing")
        Component(evalworker, "EvaluationWorker", "Python", "Model arena with Civ6TransformerNetwork")

        Component(arena, "Arena", "Python", "Shared arena logic for eval + tournament")
        Component(netconfig, "NetworkArchitectureConfig", "Python dataclass", "Single source of truth for architecture")
        Component(orchestrator, "VertexOrchestrator", "Python", "Multi-iteration pipeline + W&B + post-run tournament")

        Component(costmonitor, "CostMonitor", "Python", "Budget tracking via Billing API")
        Component(gpuscheduler, "GPUScheduler", "Python", "Dynamic T4/L4/A100 selection")
        Component(idlemanager, "IdleManager", "Python", "Inactive worker termination")
        Component(quotamanager, "QuotaManager", "Python", "Pre-submission quota validation")
        Component(preemption, "PreemptionHandler", "Python", "Spot VM recovery with manifests")
    }

    System_Ext(vertexai, "Vertex AI")
    System_Ext(gcs, "Cloud Storage")
    System_Ext(billing, "Cloud Billing")

    Rel(vertexclient, vertexai, "CustomJob submission")
    Rel(gcsstorage, gcs, "Blob operations")
    Rel(costmonitor, billing, "Budget queries")

    Rel(orchestrator, vertexjobs, "Launches worker jobs")
    Rel(orchestrator, arena, "Post-run tournament")
    Rel(vertexjobs, selfplayworker, "Launches")
    Rel(vertexjobs, trainingworker, "Launches")
    Rel(evalworker, arena, "Arena games")
    Rel(selfplayworker, trajstorage, "Pushes shards")
    Rel(trainingworker, modelstorage, "Promotes versions")
    Rel(netconfig, selfplayworker, "Architecture defaults")
    Rel(netconfig, trainingworker, "Architecture defaults")
    Rel(netconfig, evalworker, "Architecture defaults")

    Rel(gpuscheduler, vertexjobs, "GPU tier selection")
    Rel(idlemanager, vertexjobs, "Terminates idle")
    Rel(preemption, gcsstorage, "Recovery from manifests")
```

---

## Level 3: Component Diagram - Game Mechanics Layer

```mermaid
C4Component
    title Component Diagram - Game Mechanics (11 Modules)

    Container_Boundary(mech, "Game Mechanics Layer") {
        Component(diplomacy, "DiplomacyManager", "Python", "Relations (-100 to 100), war/peace, alliances")
        Component(trade, "TradeManager", "Python", "Routes, yields, trader units")
        Component(espionage, "EspionageManager", "Python", "Spies, missions, counter-intel")
        Component(religion, "ReligionManager", "Python", "Faith, beliefs, religious units")
        Component(government, "GovernmentManager", "Python", "Government types, policy cards")
        Component(greatpeople, "GreatPeopleManager", "Python", "Points, recruitment, abilities")
        Component(promotions, "PromotionManager", "Python", "Unit XP, promotion trees")
        Component(eras, "EraManager", "Python", "Era score, ages (Dark/Golden/Heroic)")
        Component(climate, "ClimateManager", "Python", "CO2, sea level, disasters")
        Component(congress, "WorldCongressManager", "Python", "Resolutions, voting, emergencies")
        Component(culture, "CultureManager", "Python", "Tourism, cultural victory progress")
    }

    Component(microciv, "MicroCivEnv", "Gymnasium", "Core game simulation")
    
    Rel(microciv, diplomacy, "Diplomatic actions")
    Rel(microciv, trade, "Trade route management")
    Rel(microciv, espionage, "Spy operations")
    Rel(microciv, religion, "Faith mechanics")
    Rel(microciv, government, "Policy changes")
    Rel(microciv, greatpeople, "Great person actions")
    Rel(microciv, promotions, "Unit upgrades")
    Rel(microciv, eras, "Era progression")
    Rel(microciv, climate, "Disaster events")
    Rel(microciv, congress, "Voting sessions")
    Rel(microciv, culture, "Tourism tracking")
```

---

## Level 3: Component Diagram - RAG & LLM Integration

```mermaid
C4Component
    title Component Diagram - RAG Pipeline (Phase 5a)

    Container_Boundary(rag, "RAG Integration Layer") {
        Component(pipeline, "RAGPipeline", "Python/async", "Main orchestration for ingest/query")
        Component(ragconfig, "RAGConfig", "Python dataclass", "Vector store and embedding settings")
        
        Component(memorystore, "MemoryVectorStore", "Python", "In-memory vector storage")
        Component(chromastore, "ChromaVectorStore", "Python/ChromaDB", "Persistent vector database")
        Component(faissstore, "FAISSVectorStore", "Python/FAISS", "High-performance similarity search")
        
        Component(localembed, "LocalEmbeddings", "Python", "Lightweight embedding generation")
        Component(openaiembed, "OpenAIEmbeddings", "Python", "OpenAI text-embedding-3 models")
        Component(docproc, "DocumentProcessor", "Python", "Chunking, metadata extraction")
        
        Component(agentworkflow, "RAGAgentWorkflow", "Python/async", "Hook-based retrieval flow")
        Component(stratadvisor, "StrategyAdvisor", "Python", "LLM-based strategy guidance")
        Component(ragintegration, "RAGIntegration", "Python", "Orchestrator-level context injection")
    }

    System_Ext(llmapi, "LLM APIs (OpenAI/Anthropic)")

    Rel(pipeline, memorystore, "Store/retrieve")
    Rel(pipeline, chromastore, "Store/retrieve")
    Rel(pipeline, faissstore, "Store/retrieve")
    Rel(pipeline, localembed, "Generate embeddings")
    Rel(pipeline, openaiembed, "Generate embeddings")
    Rel(pipeline, docproc, "Process documents")
    
    Rel(agentworkflow, pipeline, "Query context")
    Rel(stratadvisor, llmapi, "Strategy queries")
    Rel(ragintegration, stratadvisor, "Injects advice")
```

---

## Level 3: Component Diagram - Training Orchestrator

```mermaid
C4Component
    title Component Diagram - Training Orchestrator (Phase 4)

    Container_Boundary(orch, "Training Orchestrator") {
        Component(orchestrator, "TrainingOrchestrator", "Python", "Main training loop coordinator")
        Component(orchconfig, "OrchestratorConfig", "Python/YAML", "Presets: fast, balanced, production")
        Component(checkpointer, "CheckpointManager", "Python", "Model and state persistence")
        Component(scheduler, "HyperparameterScheduler", "Python", "Learning rate, temperature annealing")
        
        Component(curriculum, "CurriculumManager", "Python", "Multi-stage difficulty progression")
        Component(rewards, "RewardShaper", "Python", "Potential-based shaping F(s,s')=γΦ(s')-Φ(s)")
        Component(envtuner, "EnvironmentTuner", "Python", "Early termination, production acceleration")
        
        Component(pbtcoord, "PBTCoordinator", "Python", "Population-based hyperparameter tuning")
        Component(population, "PopulationManager", "Python", "Agent lifecycle, fitness tracking")
        Component(mutations, "MutationOperators", "Python", "Gaussian, polynomial, adaptive mutations")
    }

    Rel(orchestrator, orchconfig, "Loads configuration")
    Rel(orchestrator, checkpointer, "Saves/restores state")
    Rel(orchestrator, scheduler, "Anneals hyperparameters")
    Rel(orchestrator, curriculum, "Manages difficulty")
    Rel(orchestrator, rewards, "Shapes rewards")
    Rel(curriculum, envtuner, "Adjusts environment")
    
    Rel(orchestrator, pbtcoord, "Triggers evolution")
    Rel(pbtcoord, population, "Manages agents")
    Rel(pbtcoord, mutations, "Applies perturbations")
```

---

## Level 3: Component Diagram - Agentic Framework (Removed)

> The agentic framework (~14,600 LOC) was removed in Phase 11 as dead code. See CHANGELOG.md for details.

---

## Level 3: Component Diagram - Training Diagnostics & Value Head Stabilization (Phases 14-15)

```mermaid
C4Component
    title Component Diagram - Training Diagnostics & Value Head LR (Phases 14-15)

    Container_Boundary(diag, "Training Diagnostics (Phase 14)") {
        Component(diagnostics, "TrainingDiagnostics", "Python", "Real-time training health monitoring")
        Component(gradnorms, "Gradient Norms", "Python", "Per-layer gradient magnitude tracking")
        Component(lossdecomp, "Loss Decomposition", "Python", "Policy/value loss trends and ratios")
        Component(valuecorr, "Value Correlation", "Python", "Predicted vs actual value correlation")
        Component(policystats, "Policy Statistics", "Python", "Entropy, top-k confidence, diversity")
        Component(stalldtect, "Stall Detection", "Python", "Loss plateau and learning stall alerts")
    }

    Container_Boundary(vlr, "Value Head LR Stabilization (Phase 15)") {
        Component(paramgroups, "Optimizer Param Groups", "PyTorch", "3 groups: encoder, policy_head, value_head")
        Component(lrmult, "LR Multipliers", "Python", "Per-head LR scaling (value=0.1x, policy=1.0x)")
        Component(baselrratio, "base_lr_ratio", "Python", "Ratio metadata preserved through scheduler")
        Component(perheadlog, "Per-Head LR Logging", "W&B", "learning_rate_encoder, _policy_head, _value_head")
    }

    Component(scheduler, "LR Scheduler", "Python", "warmup_cosine / constant / step_decay")
    Component(wandb, "W&B Logger", "Python", "train/* metrics namespace")
    Component(trainworker, "Training Worker", "Python", "train_epoch() loop")

    Rel(trainworker, diagnostics, "Computes per-batch diagnostics")
    Rel(diagnostics, gradnorms, "Measures")
    Rel(diagnostics, lossdecomp, "Decomposes")
    Rel(diagnostics, valuecorr, "Correlates")
    Rel(diagnostics, policystats, "Analyzes")
    Rel(diagnostics, stalldtect, "Monitors")
    Rel(diagnostics, wandb, "Logs diagnostic metrics")

    Rel(trainworker, paramgroups, "Creates optimizer with 3 groups")
    Rel(paramgroups, lrmult, "Applies multipliers")
    Rel(lrmult, baselrratio, "Stores ratio")
    Rel(scheduler, baselrratio, "Reads ratio to scale LR")
    Rel(perheadlog, wandb, "Logs per-head LRs")
```

---

## Deployment Diagram

```mermaid
C4Deployment
    title Deployment Architecture

    Deployment_Node(local, "Local Development", "Windows/Linux/macOS") {
        Deployment_Node(python, "Python 3.11+") {
            Container(localorch, "Training Orchestrator", "Single-process or multiprocessing")
            Container(localmcts, "MCTS Engine", "CPU-based tree search")
        }
        Deployment_Node(gpu, "GPU (CUDA/MPS)") {
            Container(localnet, "Neural Networks", "PyTorch training/inference")
        }
    }

    Deployment_Node(gcp, "Google Cloud Platform") {
        Deployment_Node(vertexai, "Vertex AI") {
            Container(selfplay, "Self-Play Workers", "Spot VMs (T4/L4), stateless")
            Container(learner, "Training Worker", "On-demand VM (A100)")
        }
        Deployment_Node(storage, "Cloud Storage") {
            ContainerDb(gcs, "GCS Bucket", "Models, trajectories, checkpoints")
        }
        Deployment_Node(monitoring, "Cloud Monitoring") {
            Container(billing, "Cost Monitor", "Budget alerts")
            Container(metrics, "Metrics Export", "Training progress")
        }
    }

    Deployment_Node(external, "External Services") {
        Container(wandbcloud, "W&B Cloud", "Experiment tracking")
        Container(llmcloud, "LLM APIs", "OpenAI/Anthropic")
    }

    Rel(localorch, selfplay, "Launches via Vertex API")
    Rel(selfplay, gcs, "Pushes trajectories")
    Rel(learner, gcs, "Reads trajectories, saves models")
    Rel(localorch, wandbcloud, "Logs metrics")
    Rel(localorch, llmcloud, "Strategy queries")
```

---

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant C as Curriculum
    participant S as Self-Play
    participant M as MCTS
    participant H as HierarchicalRL
    participant N as Network
    participant R as RewardShaper
    participant B as Buffer
    participant T as Trainer
    participant G as GCS
    participant W as W&B

    O->>C: get_current_stage()
    C-->>O: stage_config
    
    O->>S: generate_games(config)
    loop Each Game
        S->>H: get_goals()
        H-->>S: hierarchical_goals
        S->>M: search(state, goals)
        M->>N: evaluate(state)
        N-->>M: policy, value
        M-->>S: action
        S->>R: shape_reward(state, next_state)
        R-->>S: shaped_reward
        S->>B: store(state, policy, reward)
    end
    S-->>O: game_stats
    O->>W: log_selfplay(stats)
    
    O->>T: train(buffer)
    loop Each Epoch
        T->>B: sample_batch()
        B-->>T: batch
        T->>N: forward(batch)
        N-->>T: predictions
        T->>N: backward(loss)
        T->>W: log_training(metrics)
    end
    T->>G: save_checkpoint()
    T-->>O: training_stats
    
    O->>C: update_progress(win_rate)
    alt Stage Advancement
        C->>O: advance_stage()
    end
```

---

## Technology Stack

| Layer | Component | Technology | Purpose |
|-------|-----------|------------|---------|
| **Core AI** | Neural Network | PyTorch 2.0+ | Model training/inference |
| **Core AI** | MCTS | NumPy | Tree search algorithms |
| **Core AI** | Transformer | PyTorch | ViT with RoPE, MoE |
| **Environment** | MicroCiv | Gymnasium | Training simulation |
| **Environment** | Civ6 Bridge | Lua/Python IPC | Real game integration |
| **Training** | Orchestrator | Python | Training coordination |
| **Training** | Distributed | multiprocessing | Parallel self-play |
| **Training** | PBT | Python | Hyperparameter evolution |
| **Cloud** | Compute | Vertex AI | Distributed training |
| **Cloud** | Storage | GCS | Artifacts, checkpoints |
| **Cloud** | Container | Docker + GCR | Containerized workers (v0.12.0) |
| **Cloud** | Optimization | Python | Cost/quota management |
| **Knowledge** | RAG | ChromaDB/FAISS | Vector search |
| **Knowledge** | LLM | OpenAI/Anthropic | Strategy advisor |
| **Logging** | Metrics | W&B, TensorBoard | Experiment tracking |
| **Testing** | Framework | pytest | 3,678+ tests, >80% coverage |
| **Linting** | Quality | ruff, mypy | Static analysis |

---

## Key Architectural Decisions

| Decision | Rationale | Trade-offs |
|----------|-----------|------------|
| **Hierarchical action space** | Reduces branching from 10^6 to 10^3 | Encoding complexity |
| **3-level decision hierarchy** | Temporal abstraction matches game scales | Coordination overhead |
| **Curriculum learning** | Faster convergence, better generalization | Requires environment abstraction |
| **GCP Spot VMs** | 60-91% cost reduction for self-play | Preemption handling complexity |
| **Protocol-driven cloud** | Backend-agnostic design for testability | Abstraction layers |
| **LZMA trajectory compression** | Minimizes GCS egress costs | CPU overhead |
| **RAG integration** | Knowledge-enhanced decisions | Query latency |
| **NetworkArchitectureConfig** | Single source of truth for model dimensions | Must stay in sync with TransformerConfig |
| **Post-run tournament local** | Avoids extra Vertex AI job cost for evaluation | Limited by local CPU speed |
| **model_config persistence** | Enables legacy model loading with correct architecture | Enum→string conversion needed |
| **Per-head LR multipliers** | Value head trains at lower LR to reduce instability | Requires optimizer param groups |
| **base_lr_ratio in param groups** | LR scheduler preserves per-head ratios automatically | Extra metadata in optimizer state |
| **Training diagnostics** | Real-time gradient health, loss trends, value correlation | Minimal overhead per batch |
