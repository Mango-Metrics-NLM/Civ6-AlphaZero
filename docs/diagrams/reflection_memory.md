# Patent Case 2: LATS-Style Reflection Memory with Game-State Embedding

## Architecture Diagram

```mermaid
flowchart TB
    subgraph LEARNING["Phase 1: Learning from Failures (Post-Game)"]
        direction TB
        FT["Failed Game\nTrajectory"]

        FT --> GSD["GameStateDescriber\n- State-to-Text\n- Action Narration\n- Outcome Description"]

        GSD --> LLM["LLM Critique Generator\n(OpenAI / Anthropic)\n- Identify Key Mistakes\n- Suggest Alternatives\n- Extract Strategic Lessons"]

        LLM --> GC["GeneratedCritique\n- mistake_description\n- alternative_actions\n- strategic_lesson\n- severity_score"]

        FT --> SE["StateEncoder\n- Spatial Features\n- Vector Features\n- Linear Projection"]

        SE --> EMB["State Embedding\n(Fixed-Dim Vector)"]

        GC --> RM[("ReflectionMemory\n(Vector Store)\n- Embeddings\n- Critiques\n- Metadata")]
        EMB --> RM
    end

    subgraph INFERENCE["Phase 2: Experience Reuse (During MCTS)"]
        direction TB
        CS["Current Game State\n(Live Observation)"]

        CS --> SE2["StateEncoder\n(Same Encoder)"]

        SE2 --> QV["Query Vector"]

        QV --> SS["Similarity Search\n- Cosine Distance\n- Top-K Retrieval\n- Score Thresholding"]

        SS --> SFP["Similar Failure\nPatterns"]

        SFP --> PA["Prior Adjustment\n- Penalty on Failed\n  Action Types\n- Boost Alternatives\n- Confidence Scaling"]

        CS --> NN["Neural Network\n- Policy Head\n- Value Head"]

        NN --> |"Raw Policy\nPriors"| PA

        PA --> |"Adjusted\nPriors"| MCTS["MCTS Search\n(Failure-Aware)"]
    end

    RM --> |"Reflection\nIndex"| SS

    style LEARNING fill:#2d1a00,stroke:#ff9f4a,stroke-width:2px,color:#fff
    style INFERENCE fill:#001a2d,stroke:#4a9fff,stroke-width:2px,color:#fff
```

## Critique Generation Flow

```mermaid
sequenceDiagram
    participant Game as Game Engine
    participant Traj as Trajectory Buffer
    participant Desc as GameStateDescriber
    participant LLM as LLM Client
    participant Enc as StateEncoder
    participant Mem as ReflectionMemory

    Game->>Traj: Game ends (Loss/Poor Score)
    Note over Traj: Stores states,<br/>actions, rewards

    Traj->>Desc: Failed Trajectory
    Desc->>Desc: Convert to narrative:<br/>"Turn 45: Moved warrior to (3,7)<br/>but enemy cavalry was at (4,7)..."

    Desc->>LLM: Prompt with game narrative
    LLM->>LLM: Analyze mistakes:<br/>1. Failed to scout<br/>2. Overextended army<br/>3. Ignored defense

    LLM-->>Mem: GeneratedCritique

    Traj->>Enc: Key game states
    Enc->>Enc: Project spatial + vector<br/>features to embedding

    Enc-->>Mem: State embeddings

    Mem->>Mem: Store (embedding, critique)<br/>pairs in vector DB
```

## Inference-Time Retrieval Flow

```mermaid
sequenceDiagram
    participant MCTS as MCTS Search
    participant Enc as StateEncoder
    participant Mem as ReflectionMemory
    participant NN as Neural Network

    MCTS->>Enc: Current game state
    Enc->>Enc: Encode to vector

    Enc->>Mem: Query with state vector
    Mem->>Mem: Find similar past failures<br/>(cosine similarity)

    Mem-->>MCTS: Top-K Reflections:<br/>"Last time in similar position,<br/>attacking without defense lost"

    MCTS->>NN: Get raw policy priors
    NN-->>MCTS: action_probs[]

    Note over MCTS: Adjust priors:<br/>- Penalize actions matching<br/>  past mistakes<br/>- Boost suggested alternatives

    MCTS->>MCTS: Continue search with<br/>experience-adjusted priors
```

## Component Class Diagram

```mermaid
classDiagram
    class ReflectionMemory {
        +config: ReflectionConfig
        +vector_store: VectorStore
        +state_encoder: StateEncoder
        +add_reflection(trajectory, critique)
        +find_similar_reflections(embedding, k) list~ReflectionEntry~
        +save(path) / load(path)
    }

    class ReflectionGenerator {
        +config: ReflectionGeneratorConfig
        +llm_client: LLMClient
        +state_describer: GameStateDescriber
        +generate_critique(trajectory) GeneratedCritique
    }

    class StateEncoder {
        <<protocol>>
        +encode(observation) ndarray
        +embedding_dim() int
    }

    class DefaultStateEncoder {
        +spatial_channels: int
        +vector_dim: int
        +projection: Linear
        +encode(observation) ndarray
    }

    class GameStateDescriber {
        <<protocol>>
        +describe_state(state) str
        +describe_action(action) str
        +describe_trajectory(trajectory) str
    }

    class GeneratedCritique {
        +mistakes: list~str~
        +alternatives: list~str~
        +strategic_lessons: list~str~
        +severity: float
        +raw_text: str
    }

    class ReflectionEntry {
        +embedding: ndarray
        +critique: GeneratedCritique
        +trajectory_id: str
        +similarity_score: float
    }

    class GameTrajectory {
        +states: list~dict~
        +actions: list~int~
        +rewards: list~float~
        +outcome: str
    }

    ReflectionGenerator --> GeneratedCritique
    ReflectionGenerator --> GameStateDescriber
    ReflectionMemory --> StateEncoder
    ReflectionMemory --> ReflectionEntry
    DefaultStateEncoder ..|> StateEncoder
    ReflectionMemory ..> ReflectionGenerator : "receives critiques"
    GameTrajectory --> ReflectionGenerator : "input"
    GameTrajectory --> ReflectionMemory : "input"
```

## Innovation Comparison

```mermaid
flowchart LR
    subgraph LATS_2023["LATS (2023)\nLLM Agent Domain"]
        direction TB
        L1["LLM generates\nreasoning trace"]
        L1 --> L2["Self-critique\nvia LLM"]
        L2 --> L3["Retry with\nimproved prompt"]
    end

    subgraph ALPHAZERO["Standard AlphaZero\nGame AI Domain"]
        direction TB
        A1["Self-play\ngenerates data"]
        A1 --> A2["Train neural\nnetwork"]
        A2 --> A3["No failure\nanalysis"]
    end

    subgraph NOVEL["THIS SYSTEM\nCombines Both Domains"]
        direction TB
        N1["Self-play generates\nfailed trajectories"]
        N1 --> N2["LLM critiques\ngame failures"]
        N2 --> N3["Store as vector\nembeddings"]
        N3 --> N4["Retrieve during MCTS\nto adjust priors"]
    end

    LATS_2023 -.->|"Transfer:\nLLM critique"| NOVEL
    ALPHAZERO -.->|"Transfer:\nMCTS priors"| NOVEL

    style LATS_2023 fill:#3d3d1a,stroke:#ffff4a,stroke-width:2px,color:#fff
    style ALPHAZERO fill:#3d1a1a,stroke:#ff4a4a,stroke-width:2px,color:#fff
    style NOVEL fill:#1a3d1a,stroke:#4aff4a,stroke-width:2px,color:#fff
```

## Data Flow Summary

| Step | Component | Input | Output | Novel? |
|------|-----------|-------|--------|--------|
| 1 | GameStateDescriber | Failed trajectory | Text narrative | Moderate |
| 2 | LLM Critique Generator | Game narrative | Structured critique | **YES** |
| 3 | StateEncoder | Game observation | Embedding vector | Moderate |
| 4 | ReflectionMemory.add | Embedding + critique | Stored reflection | **YES** |
| 5 | ReflectionMemory.find | Current state embedding | Similar failures | **YES** |
| 6 | Prior Adjustment | NN priors + failure data | Modified priors | **YES** |

> **Key Novelty:** The transfer of LATS-style verbal self-critique from the LLM domain to neural-network-guided MCTS game agents. The combination of (a) LLM-generated game critiques, (b) game-state embedding similarity search, and (c) MCTS prior adjustment based on retrieved failure patterns is unprecedented.
