"""Vector store implementations for RAG system."""

from alpha_zero.rag.stores.base import (
    VectorStore,
    Document,
    SearchResult,
    DocumentMetadata,
)
from alpha_zero.rag.stores.factory import VectorStoreFactory, create_vector_store
from alpha_zero.rag.stores.memory_store import MemoryVectorStore
from alpha_zero.rag.stores.chroma_store import ChromaVectorStore
from alpha_zero.rag.stores.faiss_store import FAISSVectorStore

__all__ = [
    # Base
    "VectorStore",
    "Document",
    "SearchResult",
    "DocumentMetadata",
    # Factory
    "VectorStoreFactory",
    "create_vector_store",
    # Implementations
    "MemoryVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
]
