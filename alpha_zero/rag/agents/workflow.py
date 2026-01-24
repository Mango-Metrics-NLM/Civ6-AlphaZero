"""RAG-enhanced agent workflow integration.

This module provides the main workflow orchestration for RAG-enhanced agents:
- Workflow state management
- Step-by-step execution with context enhancement
- Integration with vector stores and embedding generators
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from alpha_zero.rag.config.settings import RAGConfig
from alpha_zero.rag.processors.embeddings import EmbeddingGenerator
from alpha_zero.rag.stores.base import Document, SearchResult, VectorStore


class WorkflowState(str, Enum):
    """State of the workflow execution."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowStep:
    """A single step in the workflow.

    Attributes:
        name: Step name
        description: Step description
        action: Action to execute
        input_data: Input data for the step
        output_data: Output data from the step
        started_at: Step start time
        completed_at: Step completion time
        error: Error if step failed
    """

    name: str
    description: str = ""
    action: str = ""
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        """Get step duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def success(self) -> bool:
        """Check if step completed successfully."""
        return self.error is None and self.completed_at is not None


@dataclass
class WorkflowContext:
    """Context for RAG-enhanced workflow execution.

    Attributes:
        workflow_id: Unique workflow identifier
        query: Original user query
        enhanced_query: Query after enhancement
        retrieved_documents: Documents retrieved from RAG
        context_text: Combined context text
        metadata: Additional metadata
        steps: Execution steps
        state: Current workflow state
    """

    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    enhanced_query: str = ""
    retrieved_documents: list[SearchResult] = field(default_factory=list)
    context_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[WorkflowStep] = field(default_factory=list)
    state: WorkflowState = WorkflowState.IDLE
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        self.steps.append(step)

    def get_current_step(self) -> WorkflowStep | None:
        """Get the current (last) step."""
        return self.steps[-1] if self.steps else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "query": self.query,
            "enhanced_query": self.enhanced_query,
            "num_documents": len(self.retrieved_documents),
            "context_length": len(self.context_text),
            "num_steps": len(self.steps),
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class WorkflowResult:
    """Result of workflow execution.

    Attributes:
        context: Workflow context
        response: Generated response
        sources: Source documents used
        total_time_ms: Total execution time
        success: Whether execution succeeded
        error: Error message if failed
    """

    context: WorkflowContext
    response: str = ""
    sources: list[Document] = field(default_factory=list)
    total_time_ms: float = 0.0
    success: bool = True
    error: str | None = None


T = TypeVar("T")


class RAGAgentWorkflow:
    """RAG-enhanced agent workflow.

    Provides a structured workflow for:
    1. Query enhancement
    2. Context retrieval from vector store
    3. Context building and ranking
    4. Response generation

    Example:
        workflow = RAGAgentWorkflow(vector_store, embedding_generator)
        await workflow.initialize()

        result = await workflow.execute(
            query="What strategy should I use for early game?",
            top_k=5
        )

        print(result.response)
        print(f"Used {len(result.sources)} sources")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        config: RAGConfig | None = None,
    ):
        """Initialize RAG agent workflow.

        Args:
            vector_store: Vector store for document retrieval
            embedding_generator: Embedding generator for queries
            config: RAG configuration
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.config = config or RAGConfig()
        self._is_initialized = False
        self._hooks: dict[str, list[Callable]] = {
            "pre_query": [],
            "post_query": [],
            "pre_retrieve": [],
            "post_retrieve": [],
            "pre_generate": [],
            "post_generate": [],
        }

    async def initialize(self) -> None:
        """Initialize the workflow components."""
        if not self.vector_store.is_initialized:
            await self.vector_store.initialize()
        self._is_initialized = True

    async def close(self) -> None:
        """Close workflow and release resources."""
        await self.vector_store.close()
        self._is_initialized = False

    def register_hook(
        self,
        event: str,
        callback: Callable,
    ) -> None:
        """Register a hook for workflow events.

        Args:
            event: Event name (pre_query, post_query, etc.)
            callback: Callback function
        """
        if event in self._hooks:
            self._hooks[event].append(callback)

    async def _run_hooks(self, event: str, context: WorkflowContext) -> None:
        """Run hooks for an event.

        Args:
            event: Event name
            context: Workflow context
        """
        for hook in self._hooks.get(event, []):
            if asyncio.iscoroutinefunction(hook):
                await hook(context)
            else:
                hook(context)

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
        include_sources: bool = True,
    ) -> WorkflowResult:
        """Execute the RAG workflow.

        Args:
            query: User query
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filters
            include_sources: Whether to include source documents

        Returns:
            WorkflowResult with context and response
        """
        if not self._is_initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")

        start_time = time.perf_counter()
        context = WorkflowContext(query=query)
        context.state = WorkflowState.RUNNING

        try:
            # Step 1: Query Enhancement
            await self._step_enhance_query(context)

            # Step 2: Generate Query Embedding
            await self._step_generate_embedding(context)

            # Step 3: Retrieve Documents
            await self._step_retrieve_documents(context, top_k, filter_metadata)

            # Step 4: Build Context
            await self._step_build_context(context)

            # Complete workflow
            context.state = WorkflowState.COMPLETED

            # Calculate total time
            total_time = (time.perf_counter() - start_time) * 1000

            return WorkflowResult(
                context=context,
                response=context.context_text,
                sources=[r.document for r in context.retrieved_documents] if include_sources else [],
                total_time_ms=total_time,
                success=True,
            )

        except Exception as e:
            context.state = WorkflowState.FAILED
            total_time = (time.perf_counter() - start_time) * 1000

            return WorkflowResult(
                context=context,
                total_time_ms=total_time,
                success=False,
                error=str(e),
            )

    async def _step_enhance_query(self, context: WorkflowContext) -> None:
        """Enhance the user query.

        Args:
            context: Workflow context
        """
        step = WorkflowStep(
            name="enhance_query",
            description="Enhance user query for better retrieval",
            input_data={"query": context.query},
        )
        step.started_at = datetime.utcnow()
        context.add_step(step)

        await self._run_hooks("pre_query", context)

        # Simple enhancement: keep original query
        # Can be extended with query expansion, spell correction, etc.
        context.enhanced_query = context.query

        step.output_data = {"enhanced_query": context.enhanced_query}
        step.completed_at = datetime.utcnow()

        await self._run_hooks("post_query", context)

    async def _step_generate_embedding(self, context: WorkflowContext) -> None:
        """Generate embedding for the query.

        Args:
            context: Workflow context
        """
        step = WorkflowStep(
            name="generate_embedding",
            description="Generate query embedding",
            input_data={"query": context.enhanced_query},
        )
        step.started_at = datetime.utcnow()
        context.add_step(step)

        embedding = await self.embedding_generator.embed_text(context.enhanced_query)
        context.metadata["query_embedding"] = embedding

        step.output_data = {"embedding_dimension": len(embedding)}
        step.completed_at = datetime.utcnow()

    async def _step_retrieve_documents(
        self,
        context: WorkflowContext,
        top_k: int,
        filter_metadata: dict[str, Any] | None,
    ) -> None:
        """Retrieve relevant documents.

        Args:
            context: Workflow context
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filters
        """
        step = WorkflowStep(
            name="retrieve_documents",
            description="Retrieve relevant documents from vector store",
            input_data={"top_k": top_k, "filter_metadata": filter_metadata},
        )
        step.started_at = datetime.utcnow()
        context.add_step(step)

        await self._run_hooks("pre_retrieve", context)

        query_embedding = context.metadata["query_embedding"]
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )
        context.retrieved_documents = results

        step.output_data = {
            "num_documents": len(results),
            "scores": [r.score for r in results],
        }
        step.completed_at = datetime.utcnow()

        await self._run_hooks("post_retrieve", context)

    async def _step_build_context(self, context: WorkflowContext) -> None:
        """Build context text from retrieved documents.

        Args:
            context: Workflow context
        """
        step = WorkflowStep(
            name="build_context",
            description="Build context from retrieved documents",
            input_data={"num_documents": len(context.retrieved_documents)},
        )
        step.started_at = datetime.utcnow()
        context.add_step(step)

        # Build context text from documents
        context_parts = []
        for i, result in enumerate(context.retrieved_documents, start=1):
            doc = result.document
            source = doc.metadata.source or "Unknown"
            context_parts.append(
                f"[Source {i}: {source} (relevance: {result.score:.3f})]\n{doc.content}",
            )

        context.context_text = "\n\n---\n\n".join(context_parts)

        step.output_data = {"context_length": len(context.context_text)}
        step.completed_at = datetime.utcnow()

    async def add_documents(
        self,
        texts: list[str],
        sources: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            texts: Document texts
            sources: Optional source identifiers
            metadata: Optional metadata for each document

        Returns:
            List of document IDs
        """
        if not self._is_initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")

        # Create documents
        documents = []
        for i, text in enumerate(texts):
            doc = Document(content=text)
            if sources and i < len(sources):
                doc.metadata.source = sources[i]
            if metadata and i < len(metadata):
                doc.metadata.custom = metadata[i]
            documents.append(doc)

        # Generate embeddings
        documents = await self.embedding_generator.embed_documents(documents)

        # Add to store
        return await self.vector_store.add_documents(documents)

    async def query(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Simple query interface for document retrieval.

        Args:
            query: Query text
            top_k: Number of results
            filter_metadata: Optional metadata filters

        Returns:
            List of search results
        """
        if not self._is_initialized:
            raise RuntimeError("Workflow not initialized. Call initialize() first.")

        embedding = await self.embedding_generator.embed_text(query)
        return await self.vector_store.search(embedding, top_k, filter_metadata)

    async def __aenter__(self) -> RAGAgentWorkflow:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
