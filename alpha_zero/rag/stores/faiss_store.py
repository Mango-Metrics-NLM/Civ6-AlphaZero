"""FAISS vector store implementation.

FAISS (Facebook AI Similarity Search) provides efficient similarity search
for dense vectors with support for:
- Various index types (Flat, IVF, HNSW)
- GPU acceleration
- Billion-scale datasets
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import faiss

from alpha_zero.rag.stores.base import Document, SearchResult, VectorStore


class FAISSVectorStore(VectorStore):
    """FAISS vector store implementation.

    Provides a wrapper around FAISS for high-performance vector similarity search.
    Supports persistent storage with separate index and metadata files.

    Example:
        async with FAISSVectorStore(dimension=1536, persist_directory=Path("./faiss_db")) as store:
            await store.add_documents(documents)
            results = await store.search(query_embedding, top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "documents",
        dimension: int = 1536,
        persist_directory: Path | None = None,
        index_type: str = "Flat",
        distance_metric: str = "cosine",
        nlist: int = 100,
    ) -> None:
        """Initialize FAISS vector store.

        Args:
            collection_name: Name of the collection
            dimension: Embedding dimension
            persist_directory: Directory for persistent storage
            index_type: FAISS index type ("Flat", "IVF", "HNSW")
            distance_metric: Distance metric ("cosine", "l2", "ip")
            nlist: Number of clusters for IVF index
        """
        super().__init__(collection_name, dimension)
        self.persist_directory = persist_directory
        self.index_type = index_type
        self.distance_metric = distance_metric
        self.nlist = nlist

        self._index: faiss.Index | None = None
        self._id_to_idx: dict[str, int] = {}
        self._idx_to_id: dict[int, str] = {}
        self._documents: dict[str, Document] = {}
        self._next_idx = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "FAISS is not installed. Install with: pip install faiss-cpu",
            ) from e

        self._loop = asyncio.get_event_loop()

        # Try to load existing index
        if self.persist_directory and self._index_file.exists():
            await self._load_index()
        else:
            # Create new index
            await self._create_index()

        self._is_initialized = True

    async def _create_index(self) -> None:
        """Create a new FAISS index."""
        import faiss

        def _create() -> Any:
            if self.distance_metric == "cosine":
                # For cosine similarity, use IP with normalized vectors
                if self.index_type == "Flat":
                    index = faiss.IndexFlatIP(self.dimension)
                elif self.index_type == "IVF":
                    quantizer = faiss.IndexFlatIP(self.dimension)
                    index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
                else:
                    index = faiss.IndexFlatIP(self.dimension)
            else:
                # L2 distance
                if self.index_type == "Flat":
                    index = faiss.IndexFlatL2(self.dimension)
                elif self.index_type == "IVF":
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
                else:
                    index = faiss.IndexFlatL2(self.dimension)

            return index

        assert self._loop is not None
        self._index = await self._loop.run_in_executor(None, _create)

    async def _load_index(self) -> None:
        """Load existing index from disk."""
        import faiss

        def _load() -> Any:
            index = faiss.read_index(str(self._index_file))

            # Load metadata
            with open(self._metadata_file) as f:
                metadata = json.load(f)

            return index, metadata

        assert self._loop is not None
        self._index, metadata = await self._loop.run_in_executor(None, _load)

        self._id_to_idx = metadata.get("id_to_idx", {})
        self._idx_to_id = {int(k): v for k, v in metadata.get("idx_to_id", {}).items()}
        self._next_idx = metadata.get("next_idx", 0)

        # Load documents
        for doc_data in metadata.get("documents", []):
            doc = Document.from_dict(doc_data)
            self._documents[doc.id] = doc

    async def _save_index(self) -> None:
        """Save index to disk."""
        if not self.persist_directory:
            return

        import faiss

        assert self._loop is not None
        assert self._index is not None
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        index = self._index  # Capture for closure

        def _save() -> None:
            faiss.write_index(index, str(self._index_file))

            # Save metadata
            metadata = {
                "id_to_idx": self._id_to_idx,
                "idx_to_id": {str(k): v for k, v in self._idx_to_id.items()},
                "next_idx": self._next_idx,
                "documents": [doc.to_dict() for doc in self._documents.values()],
            }

            with open(self._metadata_file, "w") as f:
                json.dump(metadata, f)

        await self._loop.run_in_executor(None, _save)

    @property
    def _index_file(self) -> Path:
        """Get path to index file."""
        assert self.persist_directory is not None
        return self.persist_directory / f"{self.collection_name}.index"

    @property
    def _metadata_file(self) -> Path:
        """Get path to metadata file."""
        assert self.persist_directory is not None
        return self.persist_directory / f"{self.collection_name}.json"

    async def close(self) -> None:
        """Close and optionally save the index."""
        if self._is_initialized and self.persist_directory:
            await self._save_index()
        self._index = None
        self._is_initialized = False

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to FAISS index.

        Args:
            documents: List of documents with embeddings

        Returns:
            List of document IDs that were added
        """
        if not self._is_initialized:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        self._validate_documents(documents)
        assert self._loop is not None
        assert self._index is not None
        index = self._index  # Capture for closures

        async with self._lock:
            # Prepare embeddings (_validate_documents guarantees embeddings are set)
            embeddings: np.ndarray = np.vstack(
                [emb for doc in documents if (emb := doc.embedding) is not None]
            ).astype(np.float32)

            # Normalize for cosine similarity
            if self.distance_metric == "cosine":
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                embeddings = embeddings / norms

            # Train index if IVF and not trained
            if self.index_type == "IVF" and not index.is_trained:

                def _train() -> None:
                    index.train(embeddings)

                await self._loop.run_in_executor(None, _train)

            # Add to index
            def _add() -> None:
                index.add(embeddings)

            await self._loop.run_in_executor(None, _add)

            # Update mappings
            added_ids = []
            for doc in documents:
                self._id_to_idx[doc.id] = self._next_idx
                self._idx_to_id[self._next_idx] = doc.id
                self._documents[doc.id] = doc
                added_ids.append(doc.id)
                self._next_idx += 1

            # Save if persistent
            if self.persist_directory:
                await self._save_index()

            return added_ids

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search FAISS index for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Maximum number of results
            filter_metadata: Optional metadata filters (applied post-retrieval)

        Returns:
            List of search results
        """
        if not self._is_initialized:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        assert self._loop is not None
        assert self._index is not None
        index = self._index  # Capture for closure

        if index.ntotal == 0:
            return []

        async with self._lock:
            # Prepare query
            query = query_embedding.reshape(1, -1).astype(np.float32)

            # Normalize for cosine similarity
            if self.distance_metric == "cosine":
                norm = np.linalg.norm(query)
                if norm > 0:
                    query = query / norm

            # Search with extra results for filtering
            search_k = top_k * 3 if filter_metadata else top_k

            def _search() -> tuple[np.ndarray, np.ndarray]:
                return index.search(query, min(search_k, index.ntotal))  # type: ignore[no-any-return]

            distances, indices = await self._loop.run_in_executor(None, _search)

            # Build results
            results: list[SearchResult] = []
            for dist, idx in zip(distances[0], indices[0], strict=True):
                if idx < 0:  # FAISS returns -1 for invalid results
                    continue

                doc_id = self._idx_to_id.get(int(idx))
                if doc_id is None:
                    continue

                doc = self._documents.get(doc_id)
                if doc is None:
                    continue

                # Apply metadata filter
                if filter_metadata:
                    metadata_dict = doc.metadata.to_dict()
                    if not all(metadata_dict.get(k) == v for k, v in filter_metadata.items()):
                        continue

                # Convert distance to score
                if self.distance_metric == "cosine":
                    score = float(dist)  # IP returns similarity directly
                else:
                    score = 1 / (1 + float(dist))

                results.append(
                    SearchResult(
                        document=doc,
                        score=score,
                        distance=float(dist)
                        if self.distance_metric != "cosine"
                        else 1 - float(dist),
                        rank=len(results) + 1,
                    ),
                )

                if len(results) >= top_k:
                    break

            return results

    async def delete(self, ids: list[str]) -> int:
        """Delete documents by ID (logical deletion).

        Important: FAISS does not support native vector deletion. This method
        performs a logical delete by removing documents from the metadata
        mappings, but the vectors remain in the FAISS index.

        Implications:
            - Memory: The index size (index.ntotal) does not decrease
            - Search: Deleted vectors are filtered out during search, but may
              cause fewer than top_k results if many deletions have occurred
            - Rebuild: For production use with frequent deletions, periodically
              rebuild the index to reclaim space and maintain performance

        To rebuild the index, export all remaining documents and create a new
        FAISSVectorStore instance.

        Args:
            ids: List of document IDs to delete

        Returns:
            Number of documents logically deleted
        """
        if not self._is_initialized:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        async with self._lock:
            deleted = 0
            for doc_id in ids:
                if doc_id in self._documents:
                    # Remove from mappings
                    idx = self._id_to_idx.pop(doc_id, None)
                    if idx is not None:
                        self._idx_to_id.pop(idx, None)
                    del self._documents[doc_id]
                    deleted += 1

            # Save if persistent
            if deleted > 0 and self.persist_directory:
                await self._save_index()

            return deleted

    async def get(self, ids: list[str]) -> list[Document | None]:
        """Get documents by ID.

        Args:
            ids: List of document IDs

        Returns:
            List of documents (None for missing IDs)
        """
        if not self._is_initialized:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        async with self._lock:
            return [self._documents.get(doc_id) for doc_id in ids]

    async def count(self) -> int:
        """Get total number of documents.

        Returns:
            Document count
        """
        if not self._is_initialized:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        async with self._lock:
            return len(self._documents)

    @classmethod
    def from_config(cls, config: Any) -> FAISSVectorStore:
        """Create store from VectorStoreConfig.

        Args:
            config: VectorStoreConfig instance

        Returns:
            Configured FAISSVectorStore
        """
        return cls(
            collection_name=config.collection_name,
            dimension=config.dimension,
            persist_directory=config.persist_directory,
            distance_metric=config.distance_metric,
        )
