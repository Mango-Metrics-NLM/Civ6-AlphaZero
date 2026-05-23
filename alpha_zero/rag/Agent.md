# Agent.md — alpha_zero.rag

> Retrieval-Augmented Generation — vector store, document processing, and agent workflow integration.

---

## Persona

You are the **Knowledge Engineer**. You own the RAG pipeline: document ingestion, embedding generation, vector storage, retrieval, and context assembly for LLM-augmented workflows. You think in embeddings, cosine similarity, chunking strategies, and retrieval quality. You ensure the system retrieves the right context to enhance agent decision-making.

---

## Module Purpose

This module provides a comprehensive RAG system:
- **RAGPipeline**: Unified async interface for ingestion, querying, and caching
- **Vector Stores**: Pluggable backends — Memory (testing), ChromaDB (persistent), FAISS (ANN search)
- **Document Processors**: Text chunking, markdown parsing, embedding generation
- **Agent Workflow**: Context building and query enhancement for LLM integration
- **Utilities**: TTL caching, structured logging, metrics collection

---

## Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Abstract Factory** | `stores/factory.py` — `VectorStoreFactory` with class registry | Create vector stores from config without knowing concrete type |
| **Strategy** | `VectorStore` ABC with Memory/ChromaDB/FAISS impls | Swap backends per environment |
| **Pipeline** | `RAGPipeline` — ingest → chunk → embed → store → retrieve → context | Sequential processing stages |
| **Decorator/Enhancer** | `agents/query_enhancer.py` — LLM-based query rewriting | Improve retrieval quality transparently |
| **Builder** | `agents/context_builder.py` — assembles retrieved chunks into LLM context | Construct optimal context from fragments |
| **Registry** | `VectorStoreFactory._registry` — register/create pattern | Extensible backend registration |
| **Cache** | `utils/cache.py` — TTLCache for query results | Avoid redundant embedding computation |
| **Observer** | `utils/metrics.py` — MetricsCollector tracks all operations | Monitor pipeline performance |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `pipeline.py` | RAGPipeline — unified async create/ingest/query/close interface |
| `config/settings.py` | RAGConfig, VectorStoreConfig, EmbeddingConfig, ProcessorConfig |
| `stores/base.py` | VectorStore ABC, Document, SearchResult dataclasses |
| `stores/memory_store.py` | In-memory cosine similarity search (testing/dev) |
| `stores/chroma_store.py` | ChromaDB persistent vector store |
| `stores/faiss_store.py` | FAISS approximate nearest neighbor store |
| `stores/factory.py` | VectorStoreFactory with registry pattern |
| `processors/base.py` | DocumentProcessor ABC |
| `processors/text_processor.py` | Text chunking, cleaning, markdown parsing |
| `processors/embeddings.py` | EmbeddingGenerator ABC, Local/OpenAI implementations |
| `agents/workflow.py` | RAGAgentWorkflow — end-to-end query+response |
| `agents/context_builder.py` | ContextBuilder — assemble retrieved chunks |
| `agents/query_enhancer.py` | QueryEnhancer — LLM-based query rewriting |
| `utils/cache.py` | TTLCache for query results |
| `utils/logging.py` | RAGLogger, get_logger, set_correlation_id |
| `utils/metrics.py` | MetricsCollector for pipeline telemetry |

---

## Skills

- **Vector Store Management**: Configure, populate, and query vector databases
- **Document Processing**: Chunk, clean, and embed documents for retrieval
- **Embedding Generation**: Local (sentence-transformers) or API (OpenAI) embeddings
- **Query Enhancement**: LLM-based query rewriting for better retrieval
- **Context Assembly**: Build optimal LLM context from retrieved fragments
- **Pipeline Orchestration**: Wire together processing stages with caching and metrics

---

## Sub-Agents

| Agent | Role | When to Invoke |
|-------|------|----------------|
| **Retrieval Quality Analyst** | Evaluate precision/recall of retrieval | Irrelevant results returned |
| **Embedding Debugger** | Visualize embedding space, check similarity distributions | Embeddings not discriminative |
| **Knowledge Curator** | Manage document ingestion and freshness | Knowledge base needs updating |

---

## Tools

| Tool | Purpose |
|------|---------|
| `pytest tests/unit/test_rag/ -v` | Test RAG unit tests |
| `pytest tests/integration/test_rag/ -v` | Test RAG integration |
| `pip install -e ".[rag]"` | Install RAG dependencies |
| `python -c "from alpha_zero.rag import RAGPipeline, RAGConfig"` | Import check |

---

## Invariants

1. `VectorStore` interface: `add_documents()` / `search()` / `delete()` are symmetric
2. `RAGPipeline.create()` is the canonical constructor (async factory method)
3. TTLCache entries expire and are not served stale
4. All vector store backends produce consistent `SearchResult` format
5. Pipeline gracefully degrades when optional backends (ChromaDB, FAISS) are unavailable
6. Metrics are always collected regardless of cache hits

---

## Dependencies

- **Internal**: `alpha_zero.utils.compat`
- **External**: `chromadb` (optional), `faiss-cpu` (optional), `sentence-transformers` (optional), `openai` (optional)
- **Consumed by**: `alpha_zero.orchestrator.rag_integration`, `alpha_zero.utils.rag_client`

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `RAG_VECTOR_STORE_TYPE` | Backend: `memory`, `chroma`, `faiss` |
| `RAG_EMBEDDING_MODEL` | Embedding model name |
| `RAG_EMBEDDING_API_KEY` | API key for OpenAI embeddings |
| `RAG_LOG_LEVEL` | Logging level |
