# Patent Case 1: RAG-Enhanced Monte Carlo Tree Search

## Architecture Diagram

```mermaid
flowchart TB
    subgraph KNOWLEDGE_INGESTION["Phase 1: Knowledge Ingestion"]
        direction TB
        SG["Strategy Guides\n(Text Documents)"]
        GW["Game Wikis\n(HTML/PDF)"]
        HD["Historical Game Data\n(Replays)"]

        SG --> TP["TextDocumentProcessor\n- Chunking\n- Cleaning\n- Metadata Extraction"]
        GW --> TP
        HD --> TP

        TP --> EG["EmbeddingGenerator\n- SentenceTransformer\n- OpenAI Embeddings"]

        EG --> VS[("VectorStore\n- FAISS\n- Chroma\n- In-Memory")]
    end

    subgraph MCTS_SEARCH["Phase 2: RAG-Augmented MCTS Search"]
        direction TB
        GS["Current Game State\n(Observation Tensors)"]

        GS --> QE["Query Encoder\n- State-to-Text\n- Feature Extraction"]

        QE --> SIM["Similarity Search\n- Top-K Retrieval\n- Metadata Filtering\n- Query Enhancement"]

        SIM --> |"Retrieved\nDocuments"| KA["Knowledge Aggregator\n- Relevance Scoring\n- Context Window\n- Deduplication"]

        KA --> PA["Prior Augmentation\n- Policy Bias Injection\n- Value Adjustment\n- Confidence Weighting"]

        GS --> NN["Neural Network\n(TransformerConfig)\n- Policy Head\n- Value Head"]

        NN --> |"Raw Priors"| PA

        PA --> |"Augmented\nPriors"| MCTS["Civ6MCTS\n- Tree Expansion\n- Backup\n- Action Selection"]
    end

    VS --> |"Embedding\nIndex"| SIM

    subgraph OUTPUT["Search Output"]
        MCTS --> AP["Action Probabilities\n(Visit Counts)"]
        MCTS --> RV["Root Value\nEstimate"]
    end

    style KNOWLEDGE_INGESTION fill:#1a2744,stroke:#4a9eff,stroke-width:2px,color:#fff
    style MCTS_SEARCH fill:#1a3324,stroke:#4aff7f,stroke-width:2px,color:#fff
    style OUTPUT fill:#2d1a44,stroke:#9f4aff,stroke-width:2px,color:#fff
```

## Key Innovation Points

```mermaid
flowchart LR
    subgraph PRIOR_ART["Prior Art (AlphaZero/MuZero)"]
        direction TB
        SP["Self-Play\nOnly"] --> PNN["Neural Network\nPriors"]
        PNN --> SMCTS["MCTS\nSearch"]
    end

    subgraph NOVEL["Novel: This System"]
        direction TB
        SP2["Self-Play"] --> PNN2["Neural Network\nPriors"]
        EK["External\nKnowledge"] --> RAG["RAG\nPipeline"]
        RAG --> AUG["Augmented\nPriors"]
        PNN2 --> AUG
        AUG --> SMCTS2["MCTS\nSearch"]
    end

    PRIOR_ART -.->|"INNOVATION\nGAP"| NOVEL

    style PRIOR_ART fill:#3d1a1a,stroke:#ff4a4a,stroke-width:2px,color:#fff
    style NOVEL fill:#1a3d1a,stroke:#4aff4a,stroke-width:2px,color:#fff
```

## Component Detail

```mermaid
classDiagram
    class RAGPipeline {
        +config: RAGConfig
        +vector_store: VectorStore
        +embedding_generator: EmbeddingGenerator
        +processor: TextDocumentProcessor
        +ingest_text(text, source, metadata) IngestionResult
        +query(query, top_k, filter_metadata) QueryResult
    }

    class VectorStore {
        <<interface>>
        +add(embeddings, documents, metadata)
        +query(embedding, top_k) list
        +delete(ids)
    }

    class EmbeddingGenerator {
        <<interface>>
        +generate(texts) ndarray
        +dimension() int
    }

    class TextDocumentProcessor {
        +chunk_size: int
        +overlap: int
        +process(text) list~Chunk~
    }

    class QueryResult {
        +documents: list~str~
        +scores: list~float~
        +metadata: list~dict~
        +query_time_ms: float
    }

    class Civ6MCTS {
        +config: MCTSConfig
        +network: MCTSNetwork
        +search(state, num_simulations) tuple
    }

    RAGPipeline --> VectorStore
    RAGPipeline --> EmbeddingGenerator
    RAGPipeline --> TextDocumentProcessor
    RAGPipeline --> QueryResult
    Civ6MCTS ..> RAGPipeline : "retrieves knowledge\nduring search"
```

## Data Flow Summary

| Step | Component | Input | Output | Novel? |
|------|-----------|-------|--------|--------|
| 1 | TextDocumentProcessor | Raw strategy docs | Chunked text segments | No |
| 2 | EmbeddingGenerator | Text chunks | Vector embeddings | No |
| 3 | VectorStore | Embeddings + metadata | Indexed knowledge base | No |
| 4 | Query Encoder | Game state tensor | Text query | **YES** |
| 5 | Similarity Search | Query embedding | Top-K relevant docs | No |
| 6 | Prior Augmentation | NN priors + retrieved docs | Augmented priors | **YES** |
| 7 | Civ6MCTS | Augmented priors | Action selection | **YES** |

> **Key Novelty:** Steps 4, 6, and 7 represent the novel contribution -- converting game state observations into knowledge queries, fusing retrieved knowledge with neural network priors, and using the augmented priors within MCTS search. No prior AlphaZero variant incorporates external retrieval during search.
