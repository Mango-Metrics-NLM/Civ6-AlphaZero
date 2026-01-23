"""Main RAG pipeline orchestrating all components.

This module provides the unified interface for RAG operations:
- Document ingestion with processing and embedding
- Query execution with context enhancement
- Caching and metrics collection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from alpha_zero.rag.agents.context_builder import ContextBuilder, ContextStrategy
from alpha_zero.rag.agents.query_enhancer import QueryEnhancer
from alpha_zero.rag.config.settings import (
    EmbeddingModelType,
    RAGConfig,
    VectorStoreType,
)
from alpha_zero.rag.processors.embeddings import (
    EmbeddingGenerator,
    LocalEmbeddings,
    create_embedding_generator,
)
from alpha_zero.rag.processors.text_processor import TextDocumentProcessor
from alpha_zero.rag.stores.base import Document, SearchResult, VectorStore
from alpha_zero.rag.stores.factory import create_vector_store
from alpha_zero.rag.utils.cache import TTLCache, cache_key
from alpha_zero.rag.utils.logging import RAGLogger, get_logger, set_correlation_id
from alpha_zero.rag.utils.metrics import MetricsCollector


@dataclass
class IngestionResult:
    """Result of document ingestion.

    Attributes:
        document_ids: IDs of ingested documents
        num_chunks: Number of chunks created
        total_chars: Total characters processed
        processing_time_ms: Time for processing
        embedding_time_ms: Time for embedding generation
        storage_time_ms: Time for storage
        errors: Any errors encountered
    """

    document_ids: list[str] = field(default_factory=list)
    num_chunks: int = 0
    total_chars: int = 0
    processing_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    storage_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if ingestion was successful."""
        return len(self.errors) == 0 and len(self.document_ids) > 0

    @property
    def total_time_ms(self) -> float:
        """Get total processing time."""
        return self.processing_time_ms + self.embedding_time_ms + self.storage_time_ms


@dataclass
class QueryResult:
    """Result of a RAG query.

    Attributes:
        query: Original query
        context: Built context string
        documents: Retrieved documents
        scores: Relevance scores
        total_time_ms: Total query time
        cached: Whether result was from cache
    """

    query: str
    context: str = ""
    documents: list[Document] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    total_time_ms: float = 0.0
    cached: bool = False


class RAGPipeline:
    """Unified RAG pipeline for document ingestion and querying.

    Orchestrates all RAG components:
    - Document processing and chunking
    - Embedding generation
    - Vector storage
    - Query enhancement and context building
    - Caching and metrics

    Example:
        config = RAGConfig.from_env()
        pipeline = await RAGPipeline.create(config)

        # Ingest documents
        result = await pipeline.ingest_text(
            "Strategy guide content...",
            source="strategy_guide.txt"
        )

        # Query
        result = await pipeline.query("How to win early game?", top_k=5)
        print(result.context)
    """

    def __init__(
        self,
        config: RAGConfig,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        processor: TextDocumentProcessor,
        logger: RAGLogger | None = None,
    ):
        """Initialize RAG pipeline.

        Args:
            config: RAG configuration
            vector_store: Vector store instance
            embedding_generator: Embedding generator instance
            processor: Document processor instance
            logger: Optional logger instance
        """
        self.config = config
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.processor = processor
        self.logger = logger or get_logger("rag.pipeline", config.logging)

        # Components
        self.context_builder = ContextBuilder(
            strategy=ContextStrategy.HYBRID,
            max_chars=4000,
        )
        self.query_enhancer = QueryEnhancer()
        self.metrics = MetricsCollector()

        # Caching
        self._query_cache: TTLCache[str, QueryResult] | None = None
        if config.enable_caching:
            self._query_cache = TTLCache(ttl=config.cache_ttl, maxsize=1000)

        self._is_initialized = False

    @classmethod
    async def create(
        cls,
        config: RAGConfig | None = None,
        use_local_embeddings: bool = False,
    ) -> RAGPipeline:
        """Factory method to create and initialize a RAG pipeline.

        Args:
            config: RAG configuration (uses defaults if None)
            use_local_embeddings: Use local embeddings for testing

        Returns:
            Initialized RAGPipeline instance
        """
        config = config or RAGConfig()

        # Create vector store
        vector_store = create_vector_store(config.vector_store)

        # Create embedding generator
        if use_local_embeddings:
            embedding_generator = LocalEmbeddings(dimension=config.vector_store.dimension)
        else:
            embedding_generator = create_embedding_generator(config.embedding)

        # Create processor
        processor = TextDocumentProcessor(config.processor)

        # Create pipeline
        pipeline = cls(
            config=config,
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            processor=processor,
        )

        # Initialize
        await pipeline.initialize()

        return pipeline

    @classmethod
    async def create_for_testing(cls) -> RAGPipeline:
        """Create a pipeline configured for testing.

        Uses in-memory storage and local embeddings.

        Returns:
            Initialized RAGPipeline for testing
        """
        config = RAGConfig(
            enable_caching=False,
        )
        config.vector_store.store_type = VectorStoreType.MEMORY
        config.embedding.model_type = EmbeddingModelType.LOCAL

        return await cls.create(config, use_local_embeddings=True)

    async def initialize(self) -> None:
        """Initialize pipeline components."""
        self.logger.info("Initializing RAG pipeline")

        await self.vector_store.initialize()
        self._is_initialized = True

        self.logger.info("RAG pipeline initialized")

    async def close(self) -> None:
        """Close pipeline and release resources."""
        self.logger.info("Closing RAG pipeline")

        await self.vector_store.close()
        if self._query_cache:
            self._query_cache.clear()

        self._is_initialized = False

    async def ingest_text(
        self,
        text: str,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest text content into the RAG system.

        Args:
            text: Text content to ingest
            source: Source identifier
            metadata: Optional metadata

        Returns:
            IngestionResult with ingestion details
        """
        self._check_initialized()
        correlation_id = set_correlation_id()

        self.logger.info(
            "Starting text ingestion",
            source=source,
            text_length=len(text),
            correlation_id=correlation_id,
        )

        result = IngestionResult()

        # Step 1: Process text into chunks
        processing_timer = self.metrics.timer("processing")
        with processing_timer:
            processing_result = await self.processor.process(text, source)
        result.processing_time_ms = processing_timer.duration_ms or 0

        if not processing_result.success:
            result.errors = processing_result.errors
            return result

        result.num_chunks = processing_result.num_chunks
        result.total_chars = processing_result.total_chars

        # Step 2: Convert to documents
        documents = processing_result.to_documents()

        # Add custom metadata
        if metadata:
            for doc in documents:
                doc.metadata.custom.update(metadata)

        # Step 3: Generate embeddings
        embedding_timer = self.metrics.timer("embedding")
        with embedding_timer:
            documents = await self.embedding_generator.embed_documents(documents)
        result.embedding_time_ms = embedding_timer.duration_ms or 0

        # Step 4: Store in vector store
        storage_timer = self.metrics.timer("storage")
        with storage_timer:
            document_ids = await self.vector_store.add_documents(documents)
        result.storage_time_ms = storage_timer.duration_ms or 0

        result.document_ids = document_ids

        self.metrics.counter("documents_ingested").inc(len(document_ids))
        self.metrics.counter("chunks_created").inc(result.num_chunks)

        self.logger.info(
            "Text ingestion completed",
            num_chunks=result.num_chunks,
            total_time_ms=result.total_time_ms,
        )

        return result

    async def ingest_file(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest a file into the RAG system.

        Args:
            file_path: Path to file
            metadata: Optional metadata

        Returns:
            IngestionResult with ingestion details
        """
        path = Path(file_path)
        self.logger.info("Ingesting file", path=str(path))

        # Process file
        processing_result = await self.processor.process_file(str(path))

        if not processing_result.success:
            return IngestionResult(errors=processing_result.errors)

        # Continue with ingestion
        documents = processing_result.to_documents()

        if metadata:
            for doc in documents:
                doc.metadata.custom.update(metadata)

        documents = await self.embedding_generator.embed_documents(documents)
        document_ids = await self.vector_store.add_documents(documents)

        return IngestionResult(
            document_ids=document_ids,
            num_chunks=processing_result.num_chunks,
            total_chars=processing_result.total_chars,
        )

    async def query(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        use_cache: bool = True,
        enhance_query: bool = True,
    ) -> QueryResult:
        """Query the RAG system.

        Args:
            query: Query text
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filters
            use_cache: Whether to use query cache
            enhance_query: Whether to enhance query

        Returns:
            QueryResult with context and documents
        """
        self._check_initialized()
        correlation_id = set_correlation_id()

        start_time = datetime.utcnow()

        self.logger.debug(
            "Processing query",
            query=query[:100],
            top_k=top_k,
            correlation_id=correlation_id,
        )

        # Check cache
        if use_cache and self._query_cache:
            cache_k = cache_key(query, top_k, filter_metadata or {})
            cached_result = self._query_cache.get(cache_k)
            if cached_result:
                cached_result.cached = True
                self.metrics.counter("cache_hits").inc()
                return cached_result
            self.metrics.counter("cache_misses").inc()

        # Enhance query if enabled
        effective_query = query
        if enhance_query:
            expansion = self.query_enhancer.expand(query)
            if expansion.expanded:
                effective_query = expansion.expanded[0]

        # Generate query embedding
        with self.metrics.timer("query_embedding"):
            query_embedding = await self.embedding_generator.embed_text(effective_query)

        # Search vector store
        with self.metrics.timer("vector_search"):
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )

        # Build context
        context = self.context_builder.build(search_results)

        # Calculate total time
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = QueryResult(
            query=query,
            context=context,
            documents=[r.document for r in search_results],
            scores=[r.score for r in search_results],
            total_time_ms=total_time,
            cached=False,
        )

        # Cache result
        if use_cache and self._query_cache:
            cache_k = cache_key(query, top_k, filter_metadata or {})
            self._query_cache.set(cache_k, result)

        self.metrics.counter("queries_processed").inc()

        self.logger.debug(
            "Query completed",
            num_results=len(search_results),
            total_time_ms=total_time,
        )

        return result

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Simple search interface returning raw results.

        Args:
            query: Query text
            top_k: Number of results
            filter_metadata: Optional filters

        Returns:
            List of SearchResult objects
        """
        self._check_initialized()

        query_embedding = await self.embedding_generator.embed_text(query)
        return await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

    async def delete_documents(self, ids: list[str]) -> int:
        """Delete documents from the RAG system.

        Args:
            ids: Document IDs to delete

        Returns:
            Number of documents deleted
        """
        self._check_initialized()

        deleted = await self.vector_store.delete(ids)
        self.metrics.counter("documents_deleted").inc(deleted)

        # Clear cache since data changed
        if self._query_cache:
            self._query_cache.clear()

        return deleted

    async def get_documents(self, ids: list[str]) -> list[Document | None]:
        """Get documents by ID.

        Args:
            ids: Document IDs

        Returns:
            List of documents (None for missing)
        """
        self._check_initialized()
        return await self.vector_store.get(ids)

    async def count_documents(self) -> int:
        """Get total document count.

        Returns:
            Number of documents
        """
        self._check_initialized()
        return await self.vector_store.count()

    def get_metrics(self) -> dict[str, Any]:
        """Get pipeline metrics.

        Returns:
            Metrics summary
        """
        return self.metrics.get_summary()

    def clear_cache(self) -> None:
        """Clear the query cache."""
        if self._query_cache:
            self._query_cache.clear()

    def _check_initialized(self) -> None:
        """Check if pipeline is initialized."""
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

    async def __aenter__(self) -> RAGPipeline:
        """Async context manager entry."""
        if not self._is_initialized:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
