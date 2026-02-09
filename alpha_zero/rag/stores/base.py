"""Base classes and protocols for vector stores.

This module defines the abstract interface that all vector store implementations
must follow, enabling pluggable backends for different use cases.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np


@dataclass
class DocumentMetadata:
    """Metadata associated with a document.

    Attributes:
        source: Source identifier (file path, URL, etc.)
        created_at: Document creation timestamp
        updated_at: Last update timestamp
        chunk_index: Index if document is part of a chunked source
        total_chunks: Total number of chunks from source
        custom: Additional custom metadata
    """

    source: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None
    chunk_index: int = 0
    total_chunks: int = 1
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            **self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentMetadata:
        """Create from dictionary."""
        custom = {
            k: v
            for k, v in data.items()
            if k not in ("source", "created_at", "updated_at", "chunk_index", "total_chunks")
        }
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            source=data.get("source", ""),
            created_at=created_at or datetime.utcnow(),
            updated_at=updated_at,
            chunk_index=data.get("chunk_index", 0),
            total_chunks=data.get("total_chunks", 1),
            custom=custom,
        )


@dataclass
class Document:
    """A document to be stored in the vector store.

    Attributes:
        id: Unique document identifier
        content: Text content of the document
        embedding: Vector embedding (populated after embedding generation)
        metadata: Associated metadata
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: np.ndarray | None = None
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    def __post_init__(self) -> None:
        """Validate and convert embedding if needed."""
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Create from dictionary."""
        embedding = data.get("embedding")
        if embedding is not None:
            embedding = np.array(embedding, dtype=np.float32)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data.get("content", ""),
            embedding=embedding,
            metadata=DocumentMetadata.from_dict(data.get("metadata", {})),
        )


@dataclass
class SearchResult:
    """Result from a vector similarity search.

    Attributes:
        document: The matched document
        score: Similarity score (higher is more similar for cosine)
        distance: Distance metric value
        rank: Rank in result set (1-indexed)
    """

    document: Document
    score: float
    distance: float = 0.0
    rank: int = 1

    @property
    def relevance(self) -> float:
        """Normalized relevance score between 0 and 1."""
        # For cosine similarity, score is already in [-1, 1], normalize to [0, 1]
        return (self.score + 1) / 2


T = TypeVar("T", bound="VectorStore")


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Protocol defining the vector store interface."""

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the store."""
        ...

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        ...

    async def delete(self, ids: list[str]) -> int:
        """Delete documents by ID."""
        ...

    async def get(self, ids: list[str]) -> list[Document | None]:
        """Get documents by ID."""
        ...


class VectorStore(ABC):
    """Abstract base class for vector store implementations.

    All vector store backends must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, collection_name: str = "documents", dimension: int = 1536) -> None:
        """Initialize vector store.

        Args:
            collection_name: Name of the collection/index
            dimension: Embedding dimension
        """
        self.collection_name = collection_name
        self.dimension = dimension
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if the store is initialized.

        Returns:
            True if initialized, False otherwise
        """
        return self._is_initialized

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector store connection/resources.

        Must be called before any other operations.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close connections and release resources."""

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the store.

        Args:
            documents: List of documents with embeddings

        Returns:
            List of document IDs that were added

        Raises:
            ValueError: If documents lack embeddings
        """

    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Maximum number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of search results ordered by relevance
        """

    @abstractmethod
    async def delete(self, ids: list[str]) -> int:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """

    @abstractmethod
    async def get(self, ids: list[str]) -> list[Document | None]:
        """Get documents by ID.

        Args:
            ids: List of document IDs

        Returns:
            List of documents (None for missing IDs)
        """

    @abstractmethod
    async def count(self) -> int:
        """Get total number of documents in the store.

        Returns:
            Document count
        """

    async def update_documents(self, documents: list[Document]) -> int:
        """Update existing documents.

        Default implementation deletes and re-adds.

        Args:
            documents: Documents to update

        Returns:
            Number of documents updated
        """
        ids = [doc.id for doc in documents]
        await self.delete(ids)
        await self.add_documents(documents)
        return len(documents)

    async def search_by_text(
        self,
        query_text: str,
        embedding_fn: Any,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search using text query with embedding function.

        Args:
            query_text: Text query
            embedding_fn: Function to generate embedding from text
            top_k: Maximum number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of search results
        """
        query_embedding = await embedding_fn(query_text)
        return await self.search(query_embedding, top_k, filter_metadata)

    async def __aenter__(self: T) -> T:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    def _validate_documents(self, documents: list[Document]) -> None:
        """Validate documents before adding.

        Args:
            documents: Documents to validate

        Raises:
            ValueError: If validation fails
        """
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                raise ValueError(f"Document {i} (id={doc.id}) has no embedding")
            if len(doc.embedding) != self.dimension:
                raise ValueError(
                    f"Document {i} (id={doc.id}) has wrong dimension: "
                    f"expected {self.dimension}, got {len(doc.embedding)}",
                )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity score
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Euclidean distance
        """
        return float(np.linalg.norm(a - b))
