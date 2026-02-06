"""RAG (Retrieval-Augmented Generation) integration for the training orchestrator.

This module provides a bridge between the RAG system and the training orchestrator,
enabling:
- Automatic initialization of ChromaDB on startup
- Knowledge base ingestion from document directories
- Strategy retrieval for game state analysis
- Training insight storage and retrieval
- Session-based context management

Usage:
    from alpha_zero.orchestrator.rag_integration import RAGIntegration
    from alpha_zero.orchestrator.config import RAGConfig

    config = RAGConfig(enabled=True, persist_directory="data/knowledge_db")
    rag = RAGIntegration(config)
    rag.initialize()

    # Query for strategies
    strategies = rag.query_strategy("early game expansion")

    # Store training insights
    rag.store_training_insight("Found that aggressive expansion works well")

    # Cleanup
    rag.close()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alpha_zero.orchestrator.config import RAGConfig

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from a RAG query."""

    content: str
    metadata: dict[str, Any]
    score: float = 0.0
    source: str = ""


@dataclass
class TrainingInsight:
    """A training insight to store in the knowledge base."""

    content: str
    category: str = "training"
    iteration: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


class RAGIntegration:
    """Integration layer between RAG system and training orchestrator.

    Manages the lifecycle of RAG components and provides a simplified interface
    for querying game strategies and storing training insights.

    Attributes:
        config: RAG configuration.
        initialized: Whether the RAG system has been initialized.
    """

    def __init__(self, config: RAGConfig) -> None:
        """Initialize RAG integration.

        Args:
            config: RAG configuration.
        """
        self.config = config
        self._rag_client: Any = None
        self._knowledge_base: Any = None
        self._initialized = False
        self._session_id: str = ""

    @property
    def initialized(self) -> bool:
        """Check if RAG system is initialized."""
        return self._initialized

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        return self._session_id

    def initialize(self) -> bool:
        """Initialize RAG system with ChromaDB.

        Returns:
            True if initialization successful, False otherwise.
        """
        if not self.config.enabled:
            logger.info("RAG integration disabled")
            return False

        if self._initialized:
            logger.info("RAG integration already initialized")
            return True

        try:
            from alpha_zero.utils.factories import create_knowledge_base, create_rag_client

            # Create RAG client
            self._rag_client = create_rag_client(self.config)
            if self._rag_client is None:
                logger.warning("Failed to create RAG client")
                return False

            # Create knowledge base
            self._knowledge_base = create_knowledge_base(
                self._rag_client,
                self.config,
            )

            # Generate session ID
            self._session_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

            # Auto-ingest knowledge if configured
            if self.config.auto_ingest:
                self._ingest_knowledge_base()

            self._initialized = True
            logger.info(f"RAG integration initialized (session={self._session_id})")
            return True

        except ImportError as e:
            logger.warning(f"RAG dependencies not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize RAG integration: {e}")
            return False

    def _ingest_knowledge_base(self) -> int:
        """Ingest documents from knowledge directory.

        Returns:
            Number of documents ingested.
        """
        if not self._knowledge_base:
            return 0

        knowledge_dir = Path(self.config.knowledge_dir)
        if not knowledge_dir.exists():
            logger.info(f"Knowledge directory not found: {knowledge_dir}")
            return 0

        ingested = 0
        for file_path in knowledge_dir.glob("**/*"):
            if file_path.is_file() and file_path.suffix in [".md", ".txt", ".json"]:
                try:
                    self._knowledge_base.add_document(
                        str(file_path),
                        metadata={"source": str(file_path), "type": "knowledge"},
                    )
                    ingested += 1
                except Exception as e:
                    logger.warning(f"Failed to ingest {file_path}: {e}")

        logger.info(f"Ingested {ingested} documents from {knowledge_dir}")
        return ingested

    def query_strategy(
        self,
        query: str,
        game_context: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[QueryResult]:
        """Query for relevant strategies.

        Args:
            query: Strategy query string.
            game_context: Optional game state context to include.
            top_k: Number of results to return. Defaults to config.top_k.

        Returns:
            List of relevant strategy results.
        """
        if not self._initialized or not self._rag_client:
            return []

        k = top_k or self.config.top_k

        try:
            # Enhance query with game context
            enhanced_query = query
            if game_context:
                context_str = ", ".join(f"{k}={v}" for k, v in game_context.items())
                enhanced_query = f"{query} (context: {context_str})"

            # Query RAG client
            results = self._rag_client.query(enhanced_query, top_k=k)

            return [
                QueryResult(
                    content=r.get("content", r.get("text", "")),
                    metadata=r.get("metadata", {}),
                    score=r.get("score", 0.0),
                    source=r.get("metadata", {}).get("source", ""),
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Strategy query failed: {e}")
            return []

    def query_similar_states(
        self,
        state_description: str,
        top_k: int | None = None,
    ) -> list[QueryResult]:
        """Find similar game states from training history.

        Args:
            state_description: Description of current game state.
            top_k: Number of results to return.

        Returns:
            List of similar states from history.
        """
        if not self._initialized or not self._rag_client:
            return []

        k = top_k or self.config.top_k

        try:
            # Query with state category filter
            results = self._rag_client.query(
                state_description,
                top_k=k,
                filter_metadata={"category": "game_state"},
            )

            return [
                QueryResult(
                    content=r.get("content", r.get("text", "")),
                    metadata=r.get("metadata", {}),
                    score=r.get("score", 0.0),
                    source=r.get("metadata", {}).get("source", ""),
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Similar state query failed: {e}")
            return []

    def store_training_insight(
        self,
        insight: str | TrainingInsight,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a training insight in the knowledge base.

        Args:
            insight: Insight content or TrainingInsight object.
            metadata: Additional metadata.

        Returns:
            True if stored successfully.
        """
        if not self._initialized or not self._rag_client:
            return False

        if not self.config.store_insights:
            return False

        try:
            if isinstance(insight, str):
                insight = TrainingInsight(content=insight)

            # Prepare metadata
            doc_metadata = {
                "category": insight.category,
                "iteration": insight.iteration,
                "session_id": self._session_id,
                "timestamp": insight.timestamp,
                **insight.metrics,
                **(metadata or {}),
            }

            # Store in RAG client
            self._rag_client.add(
                text=insight.content,
                metadata=doc_metadata,
            )

            logger.debug(f"Stored training insight: {insight.content[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to store insight: {e}")
            return False

    def store_game_summary(
        self,
        summary: str,
        outcome: str,
        metrics: dict[str, float],
        iteration: int,
    ) -> bool:
        """Store a game summary for future retrieval.

        Args:
            summary: Game summary text.
            outcome: Game outcome (win/loss/draw).
            metrics: Game metrics.
            iteration: Training iteration.

        Returns:
            True if stored successfully.
        """
        if not self._initialized or not self._rag_client:
            return False

        try:
            metadata = {
                "category": "game_summary",
                "outcome": outcome,
                "iteration": iteration,
                "session_id": self._session_id,
                "timestamp": datetime.now(UTC).isoformat(),
                **metrics,
            }

            self._rag_client.add(
                text=summary,
                metadata=metadata,
            )

            logger.debug(f"Stored game summary (iteration={iteration}, outcome={outcome})")
            return True

        except Exception as e:
            logger.error(f"Failed to store game summary: {e}")
            return False

    def get_session_insights(self) -> list[QueryResult]:
        """Get all insights from current session.

        Returns:
            List of insights from this session.
        """
        if not self._initialized or not self._rag_client:
            return []

        try:
            # Query for this session's insights
            results = self._rag_client.query(
                "",  # Empty query to get all
                top_k=100,
                filter_metadata={"session_id": self._session_id},
            )

            return [
                QueryResult(
                    content=r.get("content", r.get("text", "")),
                    metadata=r.get("metadata", {}),
                    score=r.get("score", 0.0),
                    source=r.get("metadata", {}).get("source", ""),
                )
                for r in results
            ]
        except Exception as e:
            logger.error(f"Failed to get session insights: {e}")
            return []

    def get_statistics(self) -> dict[str, Any]:
        """Get RAG system statistics.

        Returns:
            Dictionary with statistics.
        """
        if not self._initialized or not self._rag_client:
            return {"initialized": False}

        try:
            count = self._rag_client.count() if hasattr(self._rag_client, "count") else 0
            return {
                "initialized": True,
                "session_id": self._session_id,
                "collection_name": self.config.collection_name,
                "document_count": count,
                "persist_directory": self.config.persist_directory,
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"initialized": True, "error": str(e)}

    def close(self) -> None:
        """Clean shutdown of RAG system."""
        if not self._initialized:
            return

        try:
            # Persist any pending writes if applicable
            if hasattr(self._rag_client, "persist"):
                self._rag_client.persist()

            logger.info(f"RAG integration closed (session={self._session_id})")

        except Exception as e:
            logger.error(f"Error during RAG shutdown: {e}")

        finally:
            self._initialized = False
            self._rag_client = None
            self._knowledge_base = None

    def __enter__(self) -> RAGIntegration:
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RAGIntegration("
            f"initialized={self._initialized}, "
            f"session={self._session_id or 'N/A'}, "
            f"collection={self.config.collection_name})"
        )
