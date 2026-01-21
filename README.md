# Civ6 AlphaZero

A high-performance implementation of the AlphaZero algorithm tailored for a simplified Civilization VI environment (MicroCiv). This project leverages Monte Carlo Tree Search (MCTS) combined with deep residual neural networks to master complex strategic decision-making.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Key Features

- **AlphaZero Core**: Full implementation of the AlphaZero self-play reinforcement learning loop
- **Hierarchical MCTS**: Advanced tree search supporting complex, multi-category action spaces
- **Residual Network**: Dual-headed (Policy & Value) architecture for joint estimation
- **Curriculum Learning**: Multi-stage difficulty progression for accelerated training
- **Reward Shaping**: Potential-based shaping that preserves optimal policy
- **Environment Tuning**: Configurable game mechanics for faster iteration
- **W&B Integration**: Full experiment tracking with Weights & Biases
- **MicroCiv Environment**: Gymnasium-compliant simulation of Civ6 core mechanics
- **RAG Integration**: Retrieval-Augmented Generation for knowledge-enhanced decision making

## Architecture (C4 Model)

### Level 1: System Context Diagram

Shows how external actors interact with the Civ6 AlphaZero system.

```mermaid
graph TB
    subgraph External
        Researcher[/"Researcher/Developer"/]
        WandB[("Weights & Biases<br/>Experiment Tracking")]
        GPU[("GPU Cluster<br/>Compute Resources")]
    end

    subgraph "Civ6 AlphaZero System"
        System["Civ6 AlphaZero<br/>Training System"]
    end

    Researcher -->|"Configures training<br/>Analyzes results"| System
    System -->|"Logs metrics<br/>Saves artifacts"| WandB
    System -->|"Runs training<br/>Neural network ops"| GPU
    System -->|"Returns trained<br/>agent models"| Researcher

    style System fill:#1168bd,stroke:#0b4884,color:#ffffff
    style Researcher fill:#08427b,stroke:#052e56,color:#ffffff
    style WandB fill:#999999,stroke:#666666,color:#ffffff
    style GPU fill:#999999,stroke:#666666,color:#ffffff
```

### Level 2: Container Diagram

High-level software containers that compose the system.

```mermaid
graph TB
    subgraph "Civ6 AlphaZero System"
        subgraph "Training Pipeline"
            Orchestrator["Training Orchestrator<br/><i>run_enhanced_training.py</i>"]
            SelfPlay["Self-Play Engine<br/><i>Generates training data</i>"]
            Trainer["Neural Network Trainer<br/><i>Optimizes policy & value</i>"]
            Evaluator["Agent Evaluator<br/><i>Measures performance</i>"]
        end

        subgraph "Core AI Components"
            MCTS["MCTS Engine<br/><i>civ6_mcts.py</i>"]
            Network["Neural Network<br/><i>civ6_network.py</i>"]
            Buffer["Replay Buffer<br/><i>Experience storage</i>"]
        end

        subgraph "Training Enhancements"
            Curriculum["Curriculum Manager<br/><i>Staged difficulty</i>"]
            Rewards["Reward Shaper<br/><i>Potential-based shaping</i>"]
            EnvTuner["Environment Tuner<br/><i>Game mechanics</i>"]
        end

        subgraph "Environment"
            MicroCiv["MicroCiv Environment<br/><i>Simplified Civ6</i>"]
        end

        subgraph "Observability"
            WandbLogger["W&B Logger<br/><i>Experiment tracking</i>"]
        end
    end

    Orchestrator --> SelfPlay
    Orchestrator --> Trainer
    Orchestrator --> Evaluator
    Orchestrator --> Curriculum
    Orchestrator --> WandbLogger

    SelfPlay --> MCTS
    SelfPlay --> Buffer
    SelfPlay --> Rewards

    MCTS --> Network
    Trainer --> Network
    Trainer --> Buffer

    Evaluator --> MCTS
    Curriculum --> EnvTuner
    EnvTuner --> MicroCiv

    style Orchestrator fill:#1168bd,stroke:#0b4884,color:#ffffff
    style Network fill:#438dd5,stroke:#2e6295,color:#ffffff
    style MCTS fill:#438dd5,stroke:#2e6295,color:#ffffff
    style MicroCiv fill:#85bbf0,stroke:#5d82a8,color:#000000
```

### Level 3: Component Diagram - Neural Network

Internal structure of the neural network component.

```mermaid
graph TB
    subgraph "Civ6Network"
        subgraph "Encoders"
            SpatialEnc["Spatial Encoder<br/><i>Conv2D layers</i><br/>Board state processing"]
            VectorEnc["Vector Encoder<br/><i>MLP layers</i><br/>Global features"]
        end

        subgraph "Residual Tower"
            ResBlock1["Residual Block 1"]
            ResBlock2["Residual Block 2"]
            ResBlockN["Residual Block N"]
        end

        subgraph "Output Heads"
            PolicyHead["Policy Head<br/><i>Action probabilities</i>"]
            ValueHead["Value Head<br/><i>Position evaluation</i>"]
        end

        subgraph "Action Decoder"
            UnitPolicy["Unit Actions<br/><i>Move, Attack, etc.</i>"]
            TechPolicy["Tech Research<br/><i>Technology selection</i>"]
            CityPolicy["City Production<br/><i>Building/Unit queue</i>"]
        end
    end

    Input["Game State<br/>Observation"] --> SpatialEnc
    Input --> VectorEnc

    SpatialEnc --> ResBlock1
    VectorEnc --> ResBlock1
    ResBlock1 --> ResBlock2
    ResBlock2 --> ResBlockN

    ResBlockN --> PolicyHead
    ResBlockN --> ValueHead

    PolicyHead --> UnitPolicy
    PolicyHead --> TechPolicy
    PolicyHead --> CityPolicy

    ValueHead --> Output["Value Estimate<br/>[-1, 1]"]

    style Input fill:#85bbf0,stroke:#5d82a8,color:#000000
    style Output fill:#85bbf0,stroke:#5d82a8,color:#000000
    style PolicyHead fill:#438dd5,stroke:#2e6295,color:#ffffff
    style ValueHead fill:#438dd5,stroke:#2e6295,color:#ffffff
```

### Level 3: Component Diagram - MCTS Engine

Internal structure of the Monte Carlo Tree Search component.

```mermaid
graph TB
    subgraph "Civ6MCTS"
        subgraph "Tree Operations"
            Selection["Selection<br/><i>PUCT algorithm</i>"]
            Expansion["Expansion<br/><i>Add child nodes</i>"]
            Evaluation["Evaluation<br/><i>Neural network query</i>"]
            Backprop["Backpropagation<br/><i>Update statistics</i>"]
        end

        subgraph "Node Management"
            RootNode["Root Node<br/><i>Current game state</i>"]
            TreeCache["Tree Cache<br/><i>Subtree reuse</i>"]
            NodePool["Node Pool<br/><i>Memory management</i>"]
        end

        subgraph "Action Processing"
            ActionMask["Action Masking<br/><i>Legal moves only</i>"]
            Hierarchical["Hierarchical Decoder<br/><i>Multi-category actions</i>"]
            Temperature["Temperature Scaling<br/><i>Exploration control</i>"]
        end
    end

    GameState["Current State"] --> RootNode
    RootNode --> Selection
    Selection -->|"Leaf node"| Expansion
    Expansion --> Evaluation
    Evaluation -->|"Policy, Value"| Backprop
    Backprop --> Selection

    Selection --> ActionMask
    ActionMask --> Hierarchical
    Hierarchical --> Temperature
    Temperature --> FinalAction["Selected Action"]

    TreeCache --> RootNode
    NodePool --> Expansion

    style GameState fill:#85bbf0,stroke:#5d82a8,color:#000000
    style FinalAction fill:#85bbf0,stroke:#5d82a8,color:#000000
    style Selection fill:#438dd5,stroke:#2e6295,color:#ffffff
    style Backprop fill:#438dd5,stroke:#2e6295,color:#ffffff
```

### Level 3: Component Diagram - Training Enhancements

Internal structure of the Phase 3 training improvements.

```mermaid
graph TB
    subgraph "Curriculum Learning"
        StageConfig["Stage Configuration<br/><i>Board size, turns, difficulty</i>"]
        ProgressTracker["Progress Tracker<br/><i>Win rate monitoring</i>"]
        StageManager["Stage Manager<br/><i>Promotion/demotion logic</i>"]
        DifficultyCtrl["Adaptive Controller<br/><i>Fine-grained adjustment</i>"]
    end

    subgraph "Reward Shaping"
        PotentialCalc["Potential Calculator<br/><i>State value estimation</i>"]
        WinCondEval["Win Condition Evaluator<br/><i>Multi-factor scoring</i>"]
        ShapingFunc["Shaping Function<br/><i>F(s,s') = γΦ(s') - Φ(s)</i>"]
    end

    subgraph "Environment Tuning"
        EarlyTerm["Early Termination<br/><i>Decisive advantage detection</i>"]
        ProdAccel["Production Accelerator<br/><i>Faster game progression</i>"]
        EnvWrapper["Environment Wrapper<br/><i>Gymnasium compatible</i>"]
    end

    StageConfig --> StageManager
    ProgressTracker --> StageManager
    StageManager --> DifficultyCtrl
    DifficultyCtrl --> EnvWrapper

    WinCondEval --> PotentialCalc
    PotentialCalc --> ShapingFunc

    EarlyTerm --> EnvWrapper
    ProdAccel --> EnvWrapper

    EnvWrapper --> Output["Tuned Environment"]
    ShapingFunc --> RewardOut["Shaped Reward"]

    style Output fill:#85bbf0,stroke:#5d82a8,color:#000000
    style RewardOut fill:#85bbf0,stroke:#5d82a8,color:#000000
    style StageManager fill:#438dd5,stroke:#2e6295,color:#ffffff
    style ShapingFunc fill:#438dd5,stroke:#2e6295,color:#ffffff
```

### Level 4: Code Diagram - Training Flow

Sequence of operations during a training iteration.

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant C as Curriculum
    participant S as SelfPlay
    participant M as MCTS
    participant N as Network
    participant R as RewardShaper
    participant B as Buffer
    participant T as Trainer
    participant W as WandbLogger

    O->>C: get_current_stage()
    C-->>O: stage_config

    O->>S: generate_games(stage_config)
    loop For each game
        S->>M: search(state)
        M->>N: evaluate(state)
        N-->>M: policy, value
        M-->>S: action
        S->>R: shape_reward(state, next_state)
        R-->>S: shaped_reward
        S->>B: store(state, policy, reward)
    end
    S-->>O: game_stats

    O->>W: log_selfplay(game_stats)

    O->>T: train(buffer)
    loop For each epoch
        T->>B: sample_batch()
        B-->>T: batch
        T->>N: forward(batch)
        N-->>T: predictions
        T->>N: backward(loss)
        T->>W: log_training_step(metrics)
    end
    T-->>O: training_stats

    O->>C: update_progress(win_rate)
    C-->>O: stage_changed?

    O->>W: log_evaluation(results)
```

## Installation

### Prerequisites

- Python 3.11 or higher
- CUDA-capable GPU (recommended) or Apple Silicon (MPS)
- 8GB+ RAM

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Mango-Metrics-NLM/Civ6-AlphaZero.git
cd Civ6-AlphaZero

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: Install W&B for experiment tracking
pip install wandb
wandb login
```

### Development Setup

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests to verify installation
pytest tests/unit -v
```

## Usage

### Unified Training Orchestrator (Recommended)

The new `TrainingOrchestrator` provides a single entry point for all training:

```bash
# Quick training run (development/testing)
python -m alpha_zero.orchestrator --preset fast

# Balanced training (recommended for initial experiments)
python -m alpha_zero.orchestrator --preset balanced

# Production training
python -m alpha_zero.orchestrator --preset production

# Dry run (validate configuration without training)
python -m alpha_zero.orchestrator --preset fast --dry_run

# Resume from checkpoint
python -m alpha_zero.orchestrator --preset balanced --resume

# Custom YAML configuration
python -m alpha_zero.orchestrator --config configs/orchestrator.yaml
```

### Orchestrator Presets

| Preset | Iterations | Games/Iter | Board | Simulations | Use Case |
| ------ | ---------- | ---------- | ----- | ----------- | -------- |
| `fast` | 10 | 5 | 6x6 | 10 | Quick testing |
| `balanced` | 100 | 20 | 8x8 | 50 | Experiments |
| `production` | 1000 | 100 | 16x16 | 200 | Full training |

### Legacy Training Scripts

The previous training entry points are still available:

```bash
# Enhanced training script
python run_enhanced_training.py --preset balanced --device cuda

# With W&B tracking
python run_enhanced_training.py --preset balanced --wandb --wandb-project "civ6-alphazero"
```

### Programmatic Usage

```python
from alpha_zero.orchestrator import TrainingOrchestrator, OrchestratorConfig

# Load configuration from YAML
config = OrchestratorConfig.from_yaml("configs/orchestrator.yaml")

# Or use a preset
config = OrchestratorConfig.from_preset("balanced")

# Create and run orchestrator
orchestrator = TrainingOrchestrator(config)
orchestrator.run()

# Access training state
print(f"Total games: {orchestrator.state.total_games}")
print(f"Best win rate: {orchestrator.state.best_win_rate:.2%}")
```

## Configuration

### Training Presets

| Preset | Iterations | Games/Iter | Board Size | Use Case |
|--------|-----------|------------|------------|----------|
| `fast` | 2 | 2 | 6x6 | Development, debugging |
| `balanced` | 10 | 20 | 8x8 | Initial experiments |
| `full` | 50 | 100 | 10x10 | Production training |

### Curriculum Stages

| Stage | Board Size | Max Turns | Win Rate to Advance |
|-------|-----------|-----------|---------------------|
| Beginner | 6x6 | 30 | 60% |
| Intermediate | 8x8 | 50 | 65% |
| Advanced | 10x10 | 100 | 70% |

### Environment Variables

Create a `.env` file based on `.env.template`:

```bash
# Weights & Biases
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=civ6-alphazero
WANDB_ENTITY=your_team_name

# Training
CUDA_VISIBLE_DEVICES=0
```

## RAG Integration

The RAG (Retrieval-Augmented Generation) system enables knowledge-enhanced agent decision making by retrieving relevant context from a vector database.

### Installation

```bash
# Install RAG dependencies
pip install -e ".[rag]"

# Or install all dependencies including RAG
pip install -e ".[all]"
```

### Quick Start

```python
import asyncio
from alpha_zero.rag import RAGPipeline, RAGConfig

async def main():
    # Create pipeline with default configuration
    pipeline = await RAGPipeline.create()

    # Ingest strategy documents
    await pipeline.ingest_text(
        "Early game strategy: Focus on expanding territory and building settlers.",
        source="strategy_guide.txt",
        metadata={"category": "early_game"}
    )

    # Query for relevant context
    result = await pipeline.query(
        "How should I play the early game?",
        top_k=5
    )

    print(f"Context: {result.context}")
    print(f"Retrieved {len(result.documents)} documents")

    await pipeline.close()

asyncio.run(main())
```

### Configuration

RAG can be configured via environment variables or YAML:

```bash
# Environment variables
export RAG_VECTOR_STORE_TYPE=chroma
export RAG_EMBEDDING_MODEL_TYPE=openai
export OPENAI_API_KEY=your_key_here
```

```python
from alpha_zero.rag import RAGConfig, VectorStoreType, EmbeddingModelType

config = RAGConfig()
config.vector_store.store_type = VectorStoreType.CHROMA
config.embedding.model_type = EmbeddingModelType.OPENAI

pipeline = await RAGPipeline.create(config)
```

### Agent Workflow Integration

```python
from alpha_zero.rag.agents import RAGAgentWorkflow
from alpha_zero.rag.stores import MemoryVectorStore
from alpha_zero.rag.processors import LocalEmbeddings

# Create workflow
workflow = RAGAgentWorkflow(
    vector_store=MemoryVectorStore(dimension=384),
    embedding_generator=LocalEmbeddings(dimension=384),
)
await workflow.initialize()

# Register hooks for custom processing
async def pre_retrieve_hook(context):
    print(f"Searching for: {context.query}")

workflow.register_hook("pre_retrieve", pre_retrieve_hook)

# Execute workflow
result = await workflow.execute("What is the best opening strategy?")
print(result.context)
```

## Testing

The project maintains comprehensive test coverage:

```bash
# Run all tests
pytest

# Unit tests only
pytest tests/unit -v

# Integration tests only
pytest tests/integration -v

# With coverage report
pytest --cov=alpha_zero --cov-report=html

# Specific test file
pytest tests/unit/test_curriculum_learning.py -v

# Run tests matching pattern
pytest -k "curriculum" -v
```

### Test Categories

- **Unit Tests** (`tests/unit/`): Test individual components in isolation
- **Integration Tests** (`tests/integration/`): Test component interactions
- **Performance Tests**: Benchmark critical paths

## Project Structure

```
Civ6-AlphaZero/
├── alpha_zero/
│   ├── orchestrator/         # Unified training orchestrator (Phase 4)
│   │   ├── __init__.py       # Module exports
│   │   ├── __main__.py       # CLI entry point
│   │   ├── config.py         # OrchestratorConfig
│   │   ├── orchestrator.py   # TrainingOrchestrator
│   │   ├── scheduler.py      # HyperparameterScheduler
│   │   └── checkpointer.py   # CheckpointManager
│   ├── core/                 # Core AlphaZero components
│   │   ├── civ6_network.py   # Neural network architecture
│   │   ├── civ6_mcts.py      # MCTS implementation
│   │   └── civ6_actions.py   # Action space handling
│   ├── distributed/          # Distributed training
│   │   ├── actor.py          # Self-play actors
│   │   ├── learner.py        # Training workers
│   │   └── coordinator.py    # Orchestration
│   ├── envs/                 # Game environments
│   │   ├── microciv.py       # MicroCiv environment
│   │   └── civ6_encoder.py   # State encoding
│   ├── rag/                  # RAG Integration
│   │   ├── config/           # Configuration management
│   │   │   └── settings.py   # RAG configuration classes
│   │   ├── stores/           # Vector store implementations
│   │   │   ├── base.py       # Abstract interfaces
│   │   │   ├── memory_store.py
│   │   │   ├── chroma_store.py
│   │   │   └── faiss_store.py
│   │   ├── processors/       # Document processing
│   │   │   ├── text_processor.py
│   │   │   └── embeddings.py
│   │   ├── agents/           # Agent workflow
│   │   │   ├── workflow.py   # RAG workflow orchestration
│   │   │   ├── context_builder.py
│   │   │   └── query_enhancer.py
│   │   ├── utils/            # RAG utilities
│   │   │   ├── logging.py    # Structured logging
│   │   │   ├── metrics.py    # Performance metrics
│   │   │   └── cache.py      # Caching implementations
│   │   └── pipeline.py       # Main RAG pipeline
│   └── utils/                # Utilities & enhancements
│       ├── factories.py      # Component factory functions
│       ├── curriculum_learning.py
│       ├── enhanced_rewards.py
│       ├── environment_tuning.py
│       ├── training_config_enhanced.py
│       └── wandb_logger.py
├── configs/                  # Configuration files
│   ├── orchestrator.yaml     # Master configuration
│   ├── orchestrator_fast.yaml
│   └── orchestrator_production.yaml
├── tests/
│   ├── unit/                 # Unit tests
│   │   └── test_rag/         # RAG unit tests
│   └── integration/          # Integration tests
│       └── test_rag/         # RAG integration tests
├── scripts/                  # Utility scripts
├── run_enhanced_training.py  # Legacy training entry point
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Technologies

| Category | Technologies |
|----------|-------------|
| Core | Python 3.11+ |
| Deep Learning | PyTorch 2.0+ |
| Environment | Gymnasium |
| Experiment Tracking | Weights & Biases |
| RAG / Vector DB | ChromaDB, FAISS, Pinecone |
| Embeddings | OpenAI, Sentence Transformers |
| Testing | Pytest, pytest-asyncio |
| Visualization | Mermaid.js, Matplotlib |
| Code Quality | MyPy, Ruff, Black |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DeepMind's AlphaZero paper for the foundational algorithm
- The PyTorch team for the deep learning framework
- Firaxis Games for the inspiration from Civilization VI
