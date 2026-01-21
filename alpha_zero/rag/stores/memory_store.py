"""In-memory vector store for testing and development.

This implementation stores all documents in memory using NumPy arrays for
similarity computation. Useful for testing and small-scale applications.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alpha_zero.rag.stores.base import Document, SearchResult, VectorStore


@dataclass
class MemoryStoreState:
    """Internal state for memory vector store."""

    documents: dict[str, Document] = field(default_factory=dict)
    embeddings: np.ndarray | None = None
    ids: list[str] = field(default_factory=list)

    def rebuild_embeddings(self) -> None:
        """Rebuild the embeddings matrix from documents."""
        if not self.documents:
            self.embeddings = None
            self.ids = []
            return

        self.ids = list(self.documents.keys())
        embeddings_list = [
            self.documents[doc_id].embedding
            for doc_id in self.ids
            if self.documents[doc_id].embedding is not None
        ]

        if embeddings_list:
            self.embeddings = np.vstack(embeddings_list)
        else:
            self.embeddings = None


class MemoryVectorStore(VectorStore):
    """In-memory vector store implementation.

    Stores documents in a dictionary with a NumPy array for embeddings.
    Supports cosine similarity search with optional metadata filtering.

    Example:
        async with MemoryVectorStore(dimension=1536) as store:
            await store.add_documents(documents)
            results = await store.search(query_embedding, top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "memory_store",
        dimension: int = 1536,
        distance_metric: str = "cosine",
    ):
        """Initialize memory vector store.

        Args:
            collection_name: Name of the collection
            dimension: Embedding dimension
            distance_metric: Distance metric ("cosine" or "euclidean")
        """
        super().__init__(collection_name, dimension)
        self.distance_metric = distance_metric
        self._state = MemoryStoreState()
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the store (no-op for memory store)."""
        self._is_initialized = True

    async def close(self) -> None:
        """Close the store (clears memory)."""
        async with self._lock:
            self._state = MemoryStoreState()
        self._is_initialized = False

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the store.

        Args:
            documents: List of documents with embeddings

        Returns:
            List of document IDs that were added
        """
        self._validate_documents(documents)

        async with self._lock:
            added_ids = []
            for doc in documents:
                self._state.documents[doc.id] = doc
                added_ids.append(doc.id)

            self._state.rebuild_embeddings()

        return added_ids

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
        async with self._lock:
            if self._state.embeddings is None or len(self._state.ids) == 0:
                return []

            # Apply metadata filter if provided
            if filter_metadata:
                filtered_indices = self._apply_metadata_filter(filter_metadata)
                if not filtered_indices:
                    return []
                embeddings = self._state.embeddings[filtered_indices]
                ids = [self._state.ids[i] for i in filtered_indices]
            else:
                embeddings = self._state.embeddings
                ids = self._state.ids

            # Compute similarities
            if self.distance_metric == "cosine":
                scores = self._compute_cosine_similarities(query_embedding, embeddings)
                # Higher score is better for cosine similarity
                sorted_indices = np.argsort(scores)[::-1][:top_k]
            else:
                distances = self._compute_euclidean_distances(query_embedding, embeddings)
                # Lower distance is better for euclidean
                sorted_indices = np.argsort(distances)[:top_k]
                scores = 1 / (1 + distances)  # Convert to similarity

            # Build results
            results = []
            for rank, idx in enumerate(sorted_indices, start=1):
                doc_id = ids[idx]
                doc = self._state.documents[doc_id]
                score = float(scores[idx])

                results.append(
                    SearchResult(
                        document=doc,
                        score=score,
                        distance=1 - score if self.distance_metric == "cosine" else float(distances[idx]),
                        rank=rank,
                    )
                )

            return results

    async def delete(self, ids: list[str]) -> int:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        async with self._lock:
            deleted = 0
            for doc_id in ids:
                if doc_id in self._state.documents:
                    del self._state.documents[doc_id]
                    deleted += 1

            if deleted > 0:
                self._state.rebuild_embeddings()

            return deleted

    async def get(self, ids: list[str]) -> list[Document | None]:
        """Get documents by ID.

        Args:
            ids: List of document IDs

        Returns:
            List of documents (None for missing IDs)
        """
        async with self._lock:
            return [self._state.documents.get(doc_id) for doc_id in ids]

    async def count(self) -> int:
        """Get total number of documents.

        Returns:
            Document count
        """
        async with self._lock:
            return len(self._state.documents)

    async def clear(self) -> None:
        """Clear all documents from the store."""
        async with self._lock:
            self._state = MemoryStoreState()

    def _apply_metadata_filter(
        self, filter_metadata: dict[str, Any]
    ) -> list[int]:
        """Apply metadata filter and return matching indices.

        Args:
            filter_metadata: Metadata key-value pairs to match

        Returns:
            List of indices that match the filter
        """
        matching_indices = []
        for idx, doc_id in enumerate(self._state.ids):
            doc = self._state.documents[doc_id]
            metadata_dict = doc.metadata.to_dict()

            matches = all(
                metadata_dict.get(key) == value
                for key, value in filter_metadata.items()
            )

            if matches:
                matching_indices.append(idx)

        return matching_indices

    def _compute_cosine_similarities(
        self, query: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarities between query and all embeddings.

        Args:
            query: Query vector
            embeddings: Matrix of embeddings (N x D)

        Returns:
            Array of similarity scores
        """
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(len(embeddings))

        query_normalized = query / query_norm
        embeddings_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norms = np.where(embeddings_norms == 0, 1, embeddings_norms)
        embeddings_normalized = embeddings / embeddings_norms

        return np.dot(embeddings_normalized, query_normalized)

    def _compute_euclidean_distances(
        self, query: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute Euclidean distances between query and all embeddings.

        Args:
            query: Query vector
            embeddings: Matrix of embeddings (N x D)

        Returns:
            Array of distances
        """
        return np.linalg.norm(embeddings - query, axis=1)
