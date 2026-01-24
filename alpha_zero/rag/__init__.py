"""RAG (Retrieval-Augmented Generation) integration for AlphaZero agent workflow.

This module provides a comprehensive RAG system with:
- Pluggable vector store backends (ChromaDB, FAISS, Pinecone)
- Document processing and embedding generation
- Agent workflow integration with context enhancement
- Structured logging and debugging utilities

Example usage:
    from alpha_zero.rag import RAGPipeline, RAGConfig

    config = RAGConfig.from_env()
    pipeline = await RAGPipeline.create(config)

    # Add documents
    await pipeline.add_documents(documents)

    # Query with context enhancement
    results = await pipeline.query("What strategy should I use?", top_k=5)
"""

from alpha_zero.rag.agents.workflow import RAGAgentWorkflow, WorkflowContext
from alpha_zero.rag.config.settings import (
    EmbeddingConfig,
    LoggingConfig,
    ProcessorConfig,
    RAGConfig,
    VectorStoreConfig,
)
from alpha_zero.rag.pipeline import RAGPipeline
from alpha_zero.rag.processors.base import DocumentProcessor
from alpha_zero.rag.stores.base import Document, SearchResult, VectorStore

__all__ = [
    # Configuration
    "RAGConfig",
    "VectorStoreConfig",
    "EmbeddingConfig",
    "ProcessorConfig",
    "LoggingConfig",
    # Pipeline
    "RAGPipeline",
    # Stores
    "VectorStore",
    "Document",
    "SearchResult",
    # Processors
    "DocumentProcessor",
    # Agents
    "RAGAgentWorkflow",
    "WorkflowContext",
]
