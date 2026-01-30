"""ChromaDB vector store implementation.

ChromaDB is an open-source embedding database that provides:
- Persistent and ephemeral storage
- Built-in embedding functions
- Metadata filtering
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from alpha_zero.rag.stores.base import Document, DocumentMetadata, SearchResult, VectorStore


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store implementation.

    Provides a wrapper around ChromaDB for vector similarity search.
    Supports both persistent and in-memory storage.

    Example:
        config = VectorStoreConfig(
            store_type=VectorStoreType.CHROMA,
            persist_directory=Path("./chroma_db")
        )
        async with ChromaVectorStore.from_config(config) as store:
            await store.add_documents(documents)
    """

    def __init__(
        self,
        collection_name: str = "documents",
        dimension: int = 1536,
        persist_directory: Path | None = None,
        distance_metric: str = "cosine",
    ):
        """Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection
            dimension: Embedding dimension
            persist_directory: Directory for persistent storage (None for in-memory)
            distance_metric: Distance metric ("cosine", "l2", "ip")
        """
        super().__init__(collection_name, dimension)
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric
        self._client: Any = None
        self._collection: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install chromadb",
            ) from e

        self._loop = asyncio.get_event_loop()

        # Create client in thread pool to avoid blocking
        def _create_client():
            if self.persist_directory:
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                return chromadb.PersistentClient(
                    path=str(self.persist_directory),
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                return chromadb.EphemeralClient(
                    settings=Settings(anonymized_telemetry=False),
                )

        assert self._loop is not None
        self._client = await self._loop.run_in_executor(None, _create_client)

        # Get or create collection
        distance_fn = {
            "cosine": "cosine",
            "l2": "l2",
            "euclidean": "l2",
            "ip": "ip",
            "inner_product": "ip",
        }.get(self.distance_metric, "cosine")

        def _get_collection():
            return self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": distance_fn},
            )

        assert self._loop is not None
        self._collection = await self._loop.run_in_executor(None, _get_collection)
        self._is_initialized = True

    async def close(self) -> None:
        """Close ChromaDB client."""
        if self._client is not None:
            # ChromaDB handles cleanup automatically
            self._client = None
            self._collection = None
        self._is_initialized = False

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to ChromaDB.

        Args:
            documents: List of documents with embeddings

        Returns:
            List of document IDs that were added
        """
        if not self._is_initialized:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        self._validate_documents(documents)

        ids = [doc.id for doc in documents]
        # Embeddings are validated by _validate_documents above
        embeddings = [doc.embedding.tolist() for doc in documents]  # type: ignore[union-attr]
        contents = [doc.content for doc in documents]
        metadatas = [self._serialize_metadata(doc.metadata.to_dict()) for doc in documents]

        def _add():
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas,
            )

        assert self._loop is not None
        await self._loop.run_in_executor(None, _add)
        return ids

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search ChromaDB for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Maximum number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of search results
        """
        if not self._is_initialized:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        where_filter = None
        if filter_metadata:
            where_filter = self._build_where_filter(filter_metadata)

        def _query():
            return self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_filter,
                include=["embeddings", "documents", "metadatas", "distances"],
            )

        assert self._loop is not None
        result = await self._loop.run_in_executor(None, _query)

        # Parse results
        search_results = []
        if result["ids"] and result["ids"][0]:
            ids = result["ids"][0]
            documents = result["documents"][0] if result["documents"] else [None] * len(ids)
            embeddings = result["embeddings"][0] if result["embeddings"] else [None] * len(ids)
            metadatas = result["metadatas"][0] if result["metadatas"] else [{}] * len(ids)
            distances = result["distances"][0] if result["distances"] else [0.0] * len(ids)

            for rank, (doc_id, content, embedding, metadata, distance) in enumerate(
                zip(ids, documents, embeddings, metadatas, distances, strict=True),
                start=1,
            ):
                # Convert distance to similarity score
                if self.distance_metric == "cosine":
                    score = 1 - distance  # Chroma returns cosine distance
                else:
                    score = 1 / (1 + distance)

                doc = Document(
                    id=doc_id,
                    content=content or "",
                    embedding=np.array(embedding, dtype=np.float32) if embedding else None,
                    metadata=DocumentMetadata.from_dict(metadata),
                )

                search_results.append(
                    SearchResult(
                        document=doc,
                        score=score,
                        distance=distance,
                        rank=rank,
                    ),
                )

        return search_results

    async def delete(self, ids: list[str]) -> int:
        """Delete documents from ChromaDB.

        Args:
            ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        if not self._is_initialized:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        # Get count before deletion
        existing = await self.get(ids)
        count_before = sum(1 for doc in existing if doc is not None)

        def _delete():
            self._collection.delete(ids=ids)

        assert self._loop is not None
        await self._loop.run_in_executor(None, _delete)
        return count_before

    async def get(self, ids: list[str]) -> list[Document | None]:
        """Get documents from ChromaDB by ID.

        Args:
            ids: List of document IDs

        Returns:
            List of documents (None for missing IDs)
        """
        if not self._is_initialized:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        def _get():
            return self._collection.get(
                ids=ids,
                include=["embeddings", "documents", "metadatas"],
            )

        assert self._loop is not None
        result = await self._loop.run_in_executor(None, _get)

        # Build ID to result mapping
        id_to_result = {}
        if result["ids"]:
            for i, doc_id in enumerate(result["ids"]):
                embedding = result["embeddings"][i] if result["embeddings"] else None
                content = result["documents"][i] if result["documents"] else ""
                metadata = result["metadatas"][i] if result["metadatas"] else {}

                id_to_result[doc_id] = Document(
                    id=doc_id,
                    content=content or "",
                    embedding=np.array(embedding, dtype=np.float32) if embedding else None,
                    metadata=DocumentMetadata.from_dict(metadata),
                )

        return [id_to_result.get(doc_id) for doc_id in ids]

    async def count(self) -> int:
        """Get total number of documents in ChromaDB.

        Returns:
            Document count
        """
        if not self._is_initialized:
            raise RuntimeError("Store not initialized. Call initialize() first.")

        def _count() -> int:
            return self._collection.count()  # type: ignore[no-any-return]

        assert self._loop is not None
        return await self._loop.run_in_executor(None, _count)

    def _serialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Serialize metadata for ChromaDB (only allows str, int, float, bool).

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Serialized metadata with compatible types
        """
        serialized = {}
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        return serialized

    def _build_where_filter(
        self,
        filter_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Build ChromaDB where filter from metadata.

        Args:
            filter_metadata: Metadata key-value pairs

        Returns:
            ChromaDB where filter
        """
        conditions = []
        for key, value in filter_metadata.items():
            conditions.append({key: {"$eq": value}})

        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    @classmethod
    def from_config(cls, config) -> ChromaVectorStore:
        """Create store from VectorStoreConfig.

        Args:
            config: VectorStoreConfig instance

        Returns:
            Configured ChromaVectorStore
        """
        return cls(
            collection_name=config.collection_name,
            dimension=config.dimension,
            persist_directory=config.persist_directory,
            distance_metric=config.distance_metric,
        )
