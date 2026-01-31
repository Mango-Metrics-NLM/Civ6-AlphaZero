"""Unit tests for vector store implementations."""

import numpy as np
import pytest

from alpha_zero.rag.config.settings import VectorStoreConfig, VectorStoreType
from alpha_zero.rag.stores.base import Document, DocumentMetadata, SearchResult
from alpha_zero.rag.stores.factory import VectorStoreFactory, create_vector_store
from alpha_zero.rag.stores.memory_store import MemoryVectorStore


class TestDocument:
    """Tests for Document dataclass."""

    def test_create_document(self):
        """Test creating a document."""
        doc = Document(
            id="test-1",
            content="Test content",
        )

        assert doc.id == "test-1"
        assert doc.content == "Test content"
        assert doc.embedding is None
        assert isinstance(doc.metadata, DocumentMetadata)

    def test_document_with_embedding(self):
        """Test document with embedding."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        doc = Document(
            id="test-1",
            content="Test content",
            embedding=embedding,
        )

        assert doc.embedding is not None
        assert len(doc.embedding) == 3
        np.testing.assert_array_almost_equal(doc.embedding, embedding)

    def test_document_to_dict(self):
        """Test document serialization."""
        doc = Document(
            id="test-1",
            content="Test content",
            embedding=np.array([0.1, 0.2]),
        )
        doc.metadata.source = "test_source"

        data = doc.to_dict()

        assert data["id"] == "test-1"
        assert data["content"] == "Test content"
        assert data["embedding"] == [0.1, 0.2]
        assert data["metadata"]["source"] == "test_source"

    def test_document_from_dict(self):
        """Test document deserialization."""
        data = {
            "id": "test-1",
            "content": "Test content",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"source": "test_source"},
        }

        doc = Document.from_dict(data)

        assert doc.id == "test-1"
        assert doc.content == "Test content"
        assert len(doc.embedding) == 3
        assert doc.metadata.source == "test_source"


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self):
        """Test creating a search result."""
        doc = Document(id="test-1", content="Test")
        result = SearchResult(
            document=doc,
            score=0.85,
            distance=0.15,
            rank=1,
        )

        assert result.document == doc
        assert result.score == 0.85
        assert result.distance == 0.15
        assert result.rank == 1

    def test_relevance_score(self):
        """Test relevance score calculation."""
        doc = Document(id="test-1", content="Test")

        # Score of 1.0 (max cosine similarity)
        result = SearchResult(document=doc, score=1.0, rank=1)
        assert result.relevance == 1.0

        # Score of 0.0 (orthogonal)
        result = SearchResult(document=doc, score=0.0, rank=1)
        assert result.relevance == 0.5

        # Score of -1.0 (opposite)
        result = SearchResult(document=doc, score=-1.0, rank=1)
        assert result.relevance == 0.0


class TestMemoryVectorStore:
    """Tests for MemoryVectorStore."""

    @pytest.fixture
    def store(self):
        """Create a memory vector store for testing."""
        return MemoryVectorStore(collection_name="test", dimension=4)

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="doc-1",
                content="First document about cats",
                embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
            Document(
                id="doc-2",
                content="Second document about dogs",
                embedding=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            ),
            Document(
                id="doc-3",
                content="Third document about birds",
                embedding=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            ),
        ]

    @pytest.mark.asyncio
    async def test_initialize_and_close(self, store):
        """Test store initialization and closing."""
        await store.initialize()
        assert store._is_initialized

        await store.close()
        assert not store._is_initialized

    @pytest.mark.asyncio
    async def test_add_documents(self, store, sample_documents):
        """Test adding documents to the store."""
        async with store:
            ids = await store.add_documents(sample_documents)

            assert len(ids) == 3
            assert "doc-1" in ids
            assert "doc-2" in ids
            assert "doc-3" in ids

    @pytest.mark.asyncio
    async def test_add_documents_validates_embeddings(self, store):
        """Test that adding documents without embeddings fails."""
        async with store:
            doc = Document(id="no-embedding", content="Test")

            with pytest.raises(ValueError, match="no embedding"):
                await store.add_documents([doc])

    @pytest.mark.asyncio
    async def test_add_documents_validates_dimension(self, store):
        """Test that adding documents with wrong dimension fails."""
        async with store:
            doc = Document(
                id="wrong-dim",
                content="Test",
                embedding=np.array([1.0, 2.0]),  # Dimension 2 instead of 4
            )

            with pytest.raises(ValueError, match="wrong dimension"):
                await store.add_documents([doc])

    @pytest.mark.asyncio
    async def test_search(self, store, sample_documents):
        """Test searching for documents."""
        async with store:
            await store.add_documents(sample_documents)

            # Search for something similar to doc-1
            query = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
            results = await store.search(query, top_k=2)

            assert len(results) == 2
            assert results[0].document.id == "doc-1"  # Most similar
            assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_search_empty_store(self, store):
        """Test searching an empty store."""
        async with store:
            query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            results = await store.search(query, top_k=5)

            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, store):
        """Test searching with metadata filter."""
        async with store:
            docs = [
                Document(
                    id="doc-1",
                    content="Content 1",
                    embedding=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                ),
                Document(
                    id="doc-2",
                    content="Content 2",
                    embedding=np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
                ),
            ]
            docs[0].metadata.source = "source_a"
            docs[1].metadata.source = "source_b"

            await store.add_documents(docs)

            query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            results = await store.search(query, top_k=5, filter_metadata={"source": "source_b"})

            assert len(results) == 1
            assert results[0].document.id == "doc-2"

    @pytest.mark.asyncio
    async def test_delete(self, store, sample_documents):
        """Test deleting documents."""
        async with store:
            await store.add_documents(sample_documents)
            assert await store.count() == 3

            deleted = await store.delete(["doc-1", "doc-2"])

            assert deleted == 2
            assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store, sample_documents):
        """Test deleting nonexistent documents."""
        async with store:
            await store.add_documents(sample_documents)

            deleted = await store.delete(["nonexistent-id"])

            assert deleted == 0
            assert await store.count() == 3

    @pytest.mark.asyncio
    async def test_get(self, store, sample_documents):
        """Test getting documents by ID."""
        async with store:
            await store.add_documents(sample_documents)

            docs = await store.get(["doc-1", "doc-3", "nonexistent"])

            assert len(docs) == 3
            assert docs[0].id == "doc-1"
            assert docs[1].id == "doc-3"
            assert docs[2] is None

    @pytest.mark.asyncio
    async def test_count(self, store, sample_documents):
        """Test document counting."""
        async with store:
            assert await store.count() == 0

            await store.add_documents(sample_documents[:1])
            assert await store.count() == 1

            await store.add_documents(sample_documents[1:])
            assert await store.count() == 3

    @pytest.mark.asyncio
    async def test_clear(self, store, sample_documents):
        """Test clearing the store."""
        async with store:
            await store.add_documents(sample_documents)
            assert await store.count() == 3

            await store.clear()
            assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_euclidean_distance(self):
        """Test store with euclidean distance metric."""
        store = MemoryVectorStore(
            collection_name="test",
            dimension=3,
            distance_metric="euclidean",
        )

        async with store:
            docs = [
                Document(
                    id="doc-1",
                    content="Near origin",
                    embedding=np.array([0.1, 0.1, 0.1], dtype=np.float32),
                ),
                Document(
                    id="doc-2",
                    content="Far from origin",
                    embedding=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                ),
            ]
            await store.add_documents(docs)

            query = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            results = await store.search(query, top_k=2)

            # doc-1 should be closer (smaller distance)
            assert results[0].document.id == "doc-1"


class TestVectorStoreFactory:
    """Tests for VectorStoreFactory."""

    def test_create_memory_store(self):
        """Test creating memory store from config."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.MEMORY,
            collection_name="test_memory",
            dimension=512,
        )

        store = VectorStoreFactory.create(config)

        assert isinstance(store, MemoryVectorStore)
        assert store.collection_name == "test_memory"
        assert store.dimension == 512

    def test_create_vector_store_convenience(self):
        """Test convenience function for creating stores."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.MEMORY,
            collection_name="convenience_test",
        )

        store = create_vector_store(config)

        assert isinstance(store, MemoryVectorStore)

    def test_unsupported_store_type(self):
        """Test error for unsupported store type."""
        config = VectorStoreConfig()
        # Manually set an invalid type
        config.store_type = "invalid_type"  # type: ignore

        with pytest.raises((ValueError, KeyError)):
            VectorStoreFactory.create(config)
