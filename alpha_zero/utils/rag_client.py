"""RAG Client for Civ6 AlphaZero.

This module provides a wrapper around ChromaDB and SentenceTransformers
to enable Retrieval-Augmented Generation (RAG) capabilities.

Usage:
    from alpha_zero.utils.rag_client import RAGClient

    client = RAGClient()
    client.add_documents(["Civ6 relies on district adjacency.", "Archers have range 2."])
    results = client.query("How do archers work?")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings  # noqa: F401
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG Client."""
    persist_directory: str = "data/knowledge_db"
    collection_name: str = "civ6_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 3

    @classmethod
    def from_env(cls) -> RAGConfig:
        """Create configuration from environment variables."""
        return cls(
            persist_directory=os.getenv("RAG_PERSIST_DIR", "data/knowledge_db"),
            collection_name=os.getenv("RAG_COLLECTION_NAME", "civ6_knowledge"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            top_k=int(os.getenv("RAG_TOP_K", "3")),
        )

class RAGClient:
    """Client for RAG operations using ChromaDB."""

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig.from_env()

        if not CHROMA_AVAILABLE:
            logger.warning("chromadb not installed. RAG features disabled.")
            return

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not installed. using default chroma embedding.")

        self._init_db()

    def _init_db(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure persist directory exists
            Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(path=self.config.persist_directory)

            # Use sentence-transformers if available, otherwise default
            embedding_func = None
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.config.embedding_model
                )

            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=embedding_func
            )
            logger.info(f"RAG initialized at {self.config.persist_directory}, collection: {self.config.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize RAG client: {e}")
            self.client = None
            self.collection = None

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None
    ) -> bool:
        """Add documents to the knowledge base.

        Args:
            documents: List of text strings to add.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of unique IDs. If None, auto-generated.

        Returns:
            True if successful, False otherwise.
        """
        if not self.collection:
            logger.warning("RAG collection not initialized.")
            return False

        try:
            count = len(documents)
            if ids is None:
                # Simple auto-ID generation based on existing count could be better
                # but for now rely on caller or simple hashing if strictly needed.
                # Chroma requires IDs. Let's generate simple string IDs based on current time/idx
                import time
                base_id = int(time.time() * 1000)
                ids = [f"doc_{base_id}_{i}" for i in range(count)]

            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {count} documents to RAG.")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def query(self, query_text: str, n_results: int | None = None) -> list[str]:
        """Query the knowledge base.

        Args:
            query_text: The question or topic to search for.
            n_results: Number of results. Defaults to config.top_k.

        Returns:
            List of matching document strings.
        """
        if not self.collection:
            return []

        try:
            k = n_results or self.config.top_k
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k
            )

            # results['documents'] is list of list of str
            if results and results['documents']:
                return list(results['documents'][0])
            return []

        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            return []

    def count(self) -> int:
        """Get number of documents in collection."""
        if not self.collection:
            return 0
        return int(self.collection.count())
