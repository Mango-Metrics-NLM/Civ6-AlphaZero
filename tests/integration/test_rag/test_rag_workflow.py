"""Integration tests for RAG agent workflow."""

import numpy as np
import pytest
import pytest_asyncio

from alpha_zero.rag.config.settings import RAGConfig, VectorStoreType, EmbeddingModelType
from alpha_zero.rag.pipeline import RAGPipeline
from alpha_zero.rag.agents.workflow import (
    RAGAgentWorkflow,
    WorkflowContext,
    WorkflowState,
)
from alpha_zero.rag.agents.context_builder import ContextBuilder, ContextStrategy
from alpha_zero.rag.agents.query_enhancer import QueryEnhancer
from alpha_zero.rag.stores.memory_store import MemoryVectorStore
from alpha_zero.rag.processors.embeddings import LocalEmbeddings


class TestRAGAgentWorkflowIntegration:
    """Integration tests for RAG agent workflow."""

    @pytest_asyncio.fixture
    async def workflow(self):
        """Create a test workflow."""
        store = MemoryVectorStore(collection_name="test", dimension=128)
        embeddings = LocalEmbeddings(dimension=128)

        workflow = RAGAgentWorkflow(
            vector_store=store,
            embedding_generator=embeddings,
        )
        await workflow.initialize()

        yield workflow

        await workflow.close()

    @pytest.mark.asyncio
    async def test_full_workflow_execution(self, workflow):
        """Test complete workflow from ingestion to query."""
        # Add documents
        await workflow.add_documents(
            texts=[
                "The early game in Civ6 requires establishing a strong foundation. Focus on building scouts and settlers.",
                "Mid-game strategy involves expanding your empire and building military units for defense.",
                "In the late game, prioritize science and culture to achieve victory conditions.",
            ],
            sources=["early.txt", "mid.txt", "late.txt"],
        )

        # Execute workflow
        result = await workflow.execute(
            query="How should I play the early game?",
            top_k=3,
        )

        assert result.success
        assert result.context.workflow_id is not None
        assert result.context.state == WorkflowState.COMPLETED
        assert len(result.context.steps) >= 4  # enhance, embed, retrieve, build
        assert len(result.sources) >= 1
        assert result.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_workflow_with_hooks(self, workflow):
        """Test workflow with registered hooks."""
        hook_calls = []

        async def pre_retrieve_hook(context: WorkflowContext):
            hook_calls.append(("pre_retrieve", context.query))

        async def post_retrieve_hook(context: WorkflowContext):
            hook_calls.append(("post_retrieve", len(context.retrieved_documents)))

        workflow.register_hook("pre_retrieve", pre_retrieve_hook)
        workflow.register_hook("post_retrieve", post_retrieve_hook)

        await workflow.add_documents(
            texts=["Test document for hooks"],
            sources=["hooks.txt"],
        )

        await workflow.execute("test hooks", top_k=1)

        assert len(hook_calls) == 2
        assert hook_calls[0][0] == "pre_retrieve"
        assert hook_calls[1][0] == "post_retrieve"

    @pytest.mark.asyncio
    async def test_workflow_context_tracking(self, workflow):
        """Test that workflow context is properly tracked."""
        await workflow.add_documents(
            texts=["Context tracking test document"],
            sources=["context.txt"],
        )

        result = await workflow.execute("context test", top_k=1)

        context = result.context
        assert context.query == "context test"
        assert context.enhanced_query is not None
        assert "query_embedding" in context.metadata

        # Verify steps have timing information
        for step in context.steps:
            assert step.started_at is not None
            assert step.completed_at is not None
            assert step.success

    @pytest.mark.asyncio
    async def test_workflow_simple_query(self, workflow):
        """Test simple query interface."""
        await workflow.add_documents(
            texts=["Simple query test content about programming"],
            sources=["simple.txt"],
        )

        results = await workflow.query("programming", top_k=5)

        assert len(results) >= 1
        # Note: With local/random embeddings, scores can be negative (cosine similarity range: -1 to 1)
        assert isinstance(results[0].score, float)
        assert results[0].document.content is not None


class TestContextBuilderIntegration:
    """Integration tests for context building."""

    @pytest.fixture
    def builder(self):
        """Create a context builder."""
        return ContextBuilder(
            strategy=ContextStrategy.HYBRID,
            max_chars=1000,
            min_score=0.0,
            deduplicate=True,
        )

    def test_build_context_from_results(self, builder):
        """Test building context from search results."""
        from alpha_zero.rag.stores.base import Document, SearchResult

        results = [
            SearchResult(
                document=Document(id="1", content="First relevant result"),
                score=0.95,
                rank=1,
            ),
            SearchResult(
                document=Document(id="2", content="Second relevant result"),
                score=0.85,
                rank=2,
            ),
            SearchResult(
                document=Document(id="3", content="Third relevant result"),
                score=0.75,
                rank=3,
            ),
        ]
        results[0].document.metadata.source = "source1.txt"
        results[1].document.metadata.source = "source2.txt"
        results[2].document.metadata.source = "source3.txt"

        context = builder.build(results)

        assert len(context) > 0
        assert "First relevant result" in context
        assert "source1.txt" in context

    def test_context_respects_max_chars(self, builder):
        """Test that context respects character limit."""
        from alpha_zero.rag.stores.base import Document, SearchResult

        # Create results with long content
        results = [
            SearchResult(
                document=Document(id=str(i), content="A" * 500),
                score=0.9 - i * 0.1,
                rank=i + 1,
            )
            for i in range(5)
        ]

        context = builder.build(results)

        # Context should be limited (with some overhead for formatting)
        assert len(context) <= 1500  # Some buffer for formatting

    def test_context_deduplication(self, builder):
        """Test that duplicate sources are deduplicated."""
        from alpha_zero.rag.stores.base import Document, SearchResult

        results = [
            SearchResult(
                document=Document(id="1", content="Content from source A"),
                score=0.9,
                rank=1,
            ),
            SearchResult(
                document=Document(id="2", content="Different content from source A"),
                score=0.8,
                rank=2,
            ),
        ]
        results[0].document.metadata.source = "source_a.txt"
        results[1].document.metadata.source = "source_a.txt"

        context = builder.build(results)

        # Only one result from source_a should be included
        assert context.count("[1]") == 1


class TestQueryEnhancerIntegration:
    """Integration tests for query enhancement."""

    @pytest.fixture
    def enhancer(self):
        """Create a query enhancer."""
        return QueryEnhancer()

    def test_expand_query_with_synonyms(self, enhancer):
        """Test query expansion with synonyms."""
        expansion = enhancer.expand("How to win the early game?")

        assert expansion.original == "How to win the early game?"
        assert len(expansion.keywords) > 0
        assert "win" in expansion.keywords or "early" in expansion.keywords or "game" in expansion.keywords

    def test_extract_keywords(self, enhancer):
        """Test keyword extraction."""
        keywords = enhancer.extract_keywords(
            "What is the best strategy for defending against attacks?"
        )

        assert "strategy" in keywords
        assert "defending" in keywords
        assert "attacks" in keywords
        # Stop words should be filtered
        assert "the" not in keywords
        assert "is" not in keywords

    def test_decompose_complex_query(self, enhancer):
        """Test decomposing complex queries."""
        query = "How do I build cities and train military units?"

        sub_queries = enhancer.decompose(query)

        assert len(sub_queries) >= 1
        assert query in sub_queries

    def test_normalize_query(self, enhancer):
        """Test query normalization."""
        query = "  What   IS  the   BEST  strategy?  "

        normalized = enhancer.normalize(query)

        assert normalized == "what is the best strategy?"


class TestEndToEndRAGPipeline:
    """End-to-end integration tests for the RAG pipeline."""

    @pytest.mark.asyncio
    async def test_full_rag_flow(self):
        """Test complete RAG flow from config to query."""
        # Create pipeline with test configuration
        pipeline = await RAGPipeline.create_for_testing()

        try:
            # Ingest multiple documents
            doc1 = await pipeline.ingest_text(
                "Machine learning is a subset of artificial intelligence. "
                "It allows computers to learn from data without explicit programming.",
                source="ml_intro.txt",
                metadata={"topic": "machine_learning"},
            )
            assert doc1.success

            doc2 = await pipeline.ingest_text(
                "Deep learning uses neural networks with multiple layers. "
                "It has revolutionized image and speech recognition.",
                source="deep_learning.txt",
                metadata={"topic": "deep_learning"},
            )
            assert doc2.success

            doc3 = await pipeline.ingest_text(
                "Natural language processing enables computers to understand human language. "
                "Applications include translation and sentiment analysis.",
                source="nlp.txt",
                metadata={"topic": "nlp"},
            )
            assert doc3.success

            # Verify documents are stored
            count = await pipeline.count_documents()
            assert count >= 3

            # Query the pipeline
            result = await pipeline.query(
                "What is machine learning?",
                top_k=3,
            )

            assert len(result.documents) >= 1
            assert len(result.context) > 0
            # The most relevant document should be about ML
            assert any("machine learning" in doc.content.lower() for doc in result.documents)

            # Test filtered query
            filtered_result = await pipeline.query(
                "neural networks",
                top_k=5,
                filter_metadata={"topic": "deep_learning"},
            )
            # Should only return deep learning documents
            if filtered_result.documents:
                for doc in filtered_result.documents:
                    assert doc.metadata.custom.get("topic") == "deep_learning"

            # Test metrics
            metrics = pipeline.get_metrics()
            assert metrics["counters"]["documents_ingested"]["value"] >= 3
            assert metrics["counters"]["queries_processed"]["value"] >= 1

        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_rag_with_file_ingestion(self, tmp_path):
        """Test RAG with file-based ingestion."""
        pipeline = await RAGPipeline.create_for_testing()

        try:
            # Create test files
            (tmp_path / "doc1.txt").write_text("Content about Python programming language.")
            (tmp_path / "doc2.txt").write_text("Content about JavaScript and web development.")

            # Ingest files
            result1 = await pipeline.ingest_file(tmp_path / "doc1.txt")
            result2 = await pipeline.ingest_file(tmp_path / "doc2.txt")

            assert result1.success
            assert result2.success

            # Query
            result = await pipeline.query("Python programming", top_k=5)
            assert len(result.documents) >= 1

        finally:
            await pipeline.close()

    @pytest.mark.asyncio
    async def test_rag_caching_behavior(self):
        """Test RAG caching behavior."""
        pipeline = await RAGPipeline.create_for_testing()
        # Enable caching for this test
        from alpha_zero.rag.utils.cache import TTLCache
        pipeline._query_cache = TTLCache(ttl=300, maxsize=100)

        try:
            await pipeline.ingest_text("Caching test document", source="cache.txt")

            # First query - not cached
            result1 = await pipeline.query("caching test")
            assert not result1.cached

            # Second query - should be cached
            result2 = await pipeline.query("caching test")
            assert result2.cached

            # Clear cache
            pipeline.clear_cache()

            # Third query - not cached again
            result3 = await pipeline.query("caching test")
            assert not result3.cached

        finally:
            await pipeline.close()
