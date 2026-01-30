"""Unit tests for RAG pipeline."""

import pytest
import pytest_asyncio

from alpha_zero.rag.config.settings import EmbeddingModelType, RAGConfig, VectorStoreType
from alpha_zero.rag.pipeline import IngestionResult, QueryResult, RAGPipeline


class TestRAGPipeline:
    """Tests for RAGPipeline."""

    @pytest_asyncio.fixture
    async def pipeline(self):
        """Create a test pipeline."""
        pipeline = await RAGPipeline.create_for_testing()
        yield pipeline
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_create_pipeline(self):
        """Test creating a pipeline."""
        pipeline = await RAGPipeline.create_for_testing()

        assert pipeline._is_initialized
        assert pipeline.vector_store is not None
        assert pipeline.embedding_generator is not None
        assert pipeline.processor is not None

        await pipeline.close()

    @pytest.mark.asyncio
    async def test_ingest_text(self, pipeline):
        """Test ingesting text content."""
        text = "This is a test document. It contains some information about testing."

        result = await pipeline.ingest_text(text, source="test.txt")

        assert result.success
        assert result.num_chunks >= 1
        assert len(result.document_ids) >= 1
        assert result.total_time_ms >= 0

    @pytest.mark.asyncio
    async def test_ingest_text_with_metadata(self, pipeline):
        """Test ingesting text with custom metadata."""
        text = "Document with metadata"
        metadata = {"category": "test", "priority": 1}

        result = await pipeline.ingest_text(text, source="meta.txt", metadata=metadata)

        assert result.success

        # Verify metadata is stored
        docs = await pipeline.get_documents(result.document_ids)
        for doc in docs:
            if doc:
                assert doc.metadata.custom.get("category") == "test"

    @pytest.mark.asyncio
    async def test_query(self, pipeline):
        """Test querying the pipeline."""
        # First ingest some documents
        await pipeline.ingest_text(
            "Cats are wonderful pets. They are independent and playful.", source="cats.txt"
        )
        await pipeline.ingest_text(
            "Dogs are loyal companions. They love to play fetch.", source="dogs.txt"
        )

        # Query
        result = await pipeline.query("Tell me about cats", top_k=2)

        assert isinstance(result, QueryResult)
        assert result.query == "Tell me about cats"
        assert len(result.documents) >= 1
        assert len(result.context) > 0
        assert result.total_time_ms >= 0

    @pytest.mark.asyncio
    async def test_query_empty_store(self, pipeline):
        """Test querying an empty store."""
        result = await pipeline.query("Test query", top_k=5)

        assert len(result.documents) == 0
        assert result.context == ""

    @pytest.mark.asyncio
    async def test_query_with_filter(self, pipeline):
        """Test querying with metadata filter."""
        await pipeline.ingest_text(
            "Document A content", source="doc_a.txt", metadata={"category": "A"}
        )
        await pipeline.ingest_text(
            "Document B content", source="doc_b.txt", metadata={"category": "B"}
        )

        result = await pipeline.query(
            "document content", top_k=5, filter_metadata={"category": "A"}
        )

        # Should only return documents with category A
        assert len(result.documents) >= 1

    @pytest.mark.asyncio
    async def test_query_caching(self):
        """Test that queries are cached."""
        # Create pipeline with caching explicitly enabled
        config = RAGConfig(enable_caching=True, cache_ttl=300)
        config.vector_store.store_type = VectorStoreType.MEMORY
        config.embedding.model_type = EmbeddingModelType.LOCAL

        pipeline = await RAGPipeline.create(config, use_local_embeddings=True)
        try:
            await pipeline.ingest_text("Test content for caching", source="cache.txt")

            # First query
            result1 = await pipeline.query("test cache query", top_k=2)
            assert not result1.cached

            # Second identical query should be cached
            result2 = await pipeline.query("test cache query", top_k=2)
            assert result2.cached
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_query_cache_disabled(self, pipeline):
        """Test querying with cache disabled."""
        await pipeline.ingest_text("Test content", source="test.txt")

        result1 = await pipeline.query("test query", use_cache=False)
        result2 = await pipeline.query("test query", use_cache=False)

        assert not result1.cached
        assert not result2.cached

    @pytest.mark.asyncio
    async def test_delete_documents(self, pipeline):
        """Test deleting documents."""
        result = await pipeline.ingest_text("Content to delete", source="delete.txt")
        doc_ids = result.document_ids

        count_before = await pipeline.count_documents()
        deleted = await pipeline.delete_documents(doc_ids)
        count_after = await pipeline.count_documents()

        assert deleted == len(doc_ids)
        assert count_after < count_before

    @pytest.mark.asyncio
    async def test_get_documents(self, pipeline):
        """Test getting documents by ID."""
        result = await pipeline.ingest_text("Test content", source="get.txt")
        doc_ids = result.document_ids

        docs = await pipeline.get_documents(doc_ids)

        assert len(docs) == len(doc_ids)
        for doc in docs:
            assert doc is not None
            assert doc.content is not None

    @pytest.mark.asyncio
    async def test_count_documents(self, pipeline):
        """Test counting documents."""
        initial_count = await pipeline.count_documents()

        await pipeline.ingest_text("First document", source="first.txt")
        await pipeline.ingest_text("Second document", source="second.txt")

        final_count = await pipeline.count_documents()

        assert final_count >= initial_count + 2

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test clearing the query cache."""
        # Create pipeline with caching explicitly enabled
        config = RAGConfig(enable_caching=True, cache_ttl=300)
        config.vector_store.store_type = VectorStoreType.MEMORY
        config.embedding.model_type = EmbeddingModelType.LOCAL

        pipeline = await RAGPipeline.create(config, use_local_embeddings=True)
        try:
            await pipeline.ingest_text("Cache test content", source="cache.txt")

            await pipeline.query("cache test")
            assert pipeline._query_cache and pipeline._query_cache.size() > 0

            pipeline.clear_cache()

            assert pipeline._query_cache.size() == 0
        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_get_metrics(self, pipeline):
        """Test getting pipeline metrics."""
        await pipeline.ingest_text("Metrics test", source="metrics.txt")
        await pipeline.query("test query")

        metrics = pipeline.get_metrics()

        assert "timers" in metrics
        assert "counters" in metrics
        assert "gauges" in metrics
        assert "uptime_seconds" in metrics

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using pipeline as context manager."""
        config = RAGConfig()
        config.vector_store.store_type = VectorStoreType.MEMORY
        config.embedding.model_type = EmbeddingModelType.LOCAL

        async with await RAGPipeline.create(config, use_local_embeddings=True) as pipeline:
            assert pipeline._is_initialized
            await pipeline.ingest_text("Context manager test", source="ctx.txt")

        # Should be closed after context
        assert not pipeline._is_initialized

    @pytest.mark.asyncio
    async def test_search_simple(self, pipeline):
        """Test simple search interface."""
        await pipeline.ingest_text("Information about machine learning and AI", source="ml.txt")

        results = await pipeline.search("machine learning", top_k=5)

        assert len(results) >= 1
        assert all(hasattr(r, "score") for r in results)
        assert all(hasattr(r, "document") for r in results)


class TestIngestionResult:
    """Tests for IngestionResult."""

    def test_success_with_documents(self):
        """Test successful ingestion result."""
        result = IngestionResult(
            document_ids=["doc-1", "doc-2"],
            num_chunks=2,
            total_chars=100,
            processing_time_ms=10.0,
            embedding_time_ms=20.0,
            storage_time_ms=5.0,
        )

        assert result.success
        assert result.total_time_ms == 35.0

    def test_failure_with_errors(self):
        """Test failed ingestion result."""
        result = IngestionResult(
            document_ids=[],
            errors=["File not found"],
        )

        assert not result.success

    def test_failure_with_no_documents(self):
        """Test failure when no documents ingested."""
        result = IngestionResult(document_ids=[])

        assert not result.success


class TestQueryResult:
    """Tests for QueryResult."""

    def test_query_result_properties(self):
        """Test query result properties."""
        from alpha_zero.rag.stores.base import Document

        docs = [Document(id="1", content="Test")]

        result = QueryResult(
            query="test query",
            context="Built context",
            documents=docs,
            scores=[0.9],
            total_time_ms=15.5,
            cached=False,
        )

        assert result.query == "test query"
        assert result.context == "Built context"
        assert len(result.documents) == 1
        assert result.scores == [0.9]
        assert result.total_time_ms == 15.5
        assert not result.cached
