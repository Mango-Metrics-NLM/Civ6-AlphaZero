"""Unit tests for the RAG integration module.

Tests RAG integration functionality including:
- Initialization and lifecycle
- Strategy queries
- Training insight storage
- Session management
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch


@dataclass
class MockRAGConfig:
    """Mock RAG configuration."""

    enabled: bool = True
    persist_directory: str = "data/test_knowledge_db"
    collection_name: str = "test_collection"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    auto_ingest: bool = False
    knowledge_dir: str = "data/knowledge"
    store_insights: bool = True


class TestRAGIntegrationInit:
    """Tests for RAGIntegration initialization."""

    def test_init_with_config(self):
        """Test initialization with config."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)

        assert rag.config == config
        assert not rag.initialized
        assert rag.session_id == ""

    def test_not_initialized_by_default(self):
        """Test RAG is not initialized on construction."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)

        assert not rag.initialized


class TestRAGIntegrationInitialize:
    """Tests for initialize() method."""

    def test_returns_false_when_disabled(self):
        """Test initialize returns False when RAG is disabled."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig(enabled=False)
        rag = RAGIntegration(config)

        result = rag.initialize()

        assert result is False
        assert not rag.initialized

    def test_returns_true_when_already_initialized(self):
        """Test initialize returns True when already initialized."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._session_id = "test_session"

        result = rag.initialize()

        assert result is True

    @patch("alpha_zero.utils.factories.create_rag_client")
    @patch("alpha_zero.utils.factories.create_knowledge_base")
    def test_initializes_with_mock_clients(self, mock_kb, mock_rag):
        """Test initialize creates clients."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        mock_rag.return_value = MagicMock()
        mock_kb.return_value = MagicMock()

        config = MockRAGConfig(auto_ingest=False)
        rag = RAGIntegration(config)

        result = rag.initialize()

        assert result is True
        assert rag.initialized
        assert rag.session_id != ""
        mock_rag.assert_called_once()

    @patch("alpha_zero.utils.factories.create_rag_client")
    def test_returns_false_when_client_creation_fails(self, mock_rag):
        """Test initialize returns False when client creation fails."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        mock_rag.return_value = None

        config = MockRAGConfig()
        rag = RAGIntegration(config)

        result = rag.initialize()

        assert result is False
        assert not rag.initialized


class TestQueryStrategy:
    """Tests for query_strategy method."""

    def test_returns_empty_when_not_initialized(self):
        """Test returns empty list when not initialized."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)

        results = rag.query_strategy("test query")

        assert results == []

    def test_returns_results_from_client(self):
        """Test returns results from RAG client."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)
        rag._initialized = True

        # Mock RAG client
        mock_client = MagicMock()
        mock_client.query.return_value = [
            {"content": "Strategy 1", "metadata": {"source": "file1"}, "score": 0.9},
            {"content": "Strategy 2", "metadata": {"source": "file2"}, "score": 0.8},
        ]
        rag._rag_client = mock_client

        results = rag.query_strategy("expansion strategy")

        assert len(results) == 2
        assert results[0].content == "Strategy 1"
        assert results[0].score == 0.9

    def test_includes_game_context_in_query(self):
        """Test includes game context in enhanced query."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)
        rag._initialized = True

        mock_client = MagicMock()
        mock_client.query.return_value = []
        rag._rag_client = mock_client

        context = {"turn": 50, "units": 5}
        rag.query_strategy("attack", game_context=context)

        # Verify query was called with enhanced context
        call_args = mock_client.query.call_args[0][0]
        assert "context:" in call_args.lower()


class TestQuerySimilarStates:
    """Tests for query_similar_states method."""

    def test_returns_empty_when_not_initialized(self):
        """Test returns empty list when not initialized."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)

        results = rag.query_similar_states("early game")

        assert results == []

    def test_queries_with_game_state_filter(self):
        """Test queries with game_state category filter."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)
        rag._initialized = True

        mock_client = MagicMock()
        mock_client.query.return_value = []
        rag._rag_client = mock_client

        rag.query_similar_states("early expansion")

        # Verify filter was applied
        call_kwargs = mock_client.query.call_args[1]
        assert "filter_metadata" in call_kwargs


class TestStoreTrainingInsight:
    """Tests for store_training_insight method."""

    def test_returns_false_when_not_initialized(self):
        """Test returns False when not initialized."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)

        result = rag.store_training_insight("Test insight")

        assert result is False

    def test_returns_false_when_store_insights_disabled(self):
        """Test returns False when store_insights is disabled."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig(store_insights=False)
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()

        result = rag.store_training_insight("Test insight")

        assert result is False

    def test_stores_string_insight(self):
        """Test stores string insight."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._session_id = "test_session"

        mock_client = MagicMock()
        rag._rag_client = mock_client

        result = rag.store_training_insight("Test insight")

        assert result is True
        mock_client.add.assert_called_once()

    def test_stores_training_insight_object(self):
        """Test stores TrainingInsight object."""
        from alpha_zero.orchestrator.rag_integration import (
            RAGIntegration,
            TrainingInsight,
        )

        config = MockRAGConfig()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._session_id = "test_session"

        mock_client = MagicMock()
        rag._rag_client = mock_client

        insight = TrainingInsight(
            content="Win rate improved",
            category="training",
            iteration=100,
            metrics={"win_rate": 0.65},
        )

        result = rag.store_training_insight(insight)

        assert result is True


class TestStoreGameSummary:
    """Tests for store_game_summary method."""

    def test_returns_false_when_not_initialized(self):
        """Test returns False when not initialized."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)

        result = rag.store_game_summary(
            summary="Game summary",
            outcome="win",
            metrics={"score": 100},
            iteration=50,
        )

        assert result is False

    def test_stores_game_summary(self):
        """Test stores game summary with metadata."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._session_id = "test_session"

        mock_client = MagicMock()
        rag._rag_client = mock_client

        result = rag.store_game_summary(
            summary="Aggressive expansion led to victory",
            outcome="win",
            metrics={"score": 150, "turns": 75},
            iteration=100,
        )

        assert result is True
        mock_client.add.assert_called_once()


class TestGetStatistics:
    """Tests for get_statistics method."""

    def test_returns_not_initialized_when_not_init(self):
        """Test returns initialized=False when not initialized."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)

        stats = rag.get_statistics()

        assert stats["initialized"] is False

    def test_returns_statistics_when_initialized(self):
        """Test returns full statistics when initialized."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._session_id = "test_session"

        mock_client = MagicMock()
        mock_client.count.return_value = 100
        rag._rag_client = mock_client

        stats = rag.get_statistics()

        assert stats["initialized"] is True
        assert stats["session_id"] == "test_session"
        assert stats["document_count"] == 100


class TestClose:
    """Tests for close method."""

    def test_closes_without_error_when_not_initialized(self):
        """Test close works when not initialized."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)

        # Should not raise
        rag.close()

    def test_persists_and_cleans_up(self):
        """Test close persists data and cleans up."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._session_id = "test_session"

        mock_client = MagicMock()
        rag._rag_client = mock_client

        rag.close()

        assert not rag.initialized
        assert rag._rag_client is None


class TestContextManager:
    """Tests for context manager protocol."""

    @patch("alpha_zero.utils.factories.create_rag_client")
    @patch("alpha_zero.utils.factories.create_knowledge_base")
    def test_context_manager_initializes_and_closes(self, mock_kb, mock_rag):
        """Test context manager calls initialize and close."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        mock_rag.return_value = MagicMock()
        mock_kb.return_value = MagicMock()

        config = MockRAGConfig()

        with RAGIntegration(config) as rag:
            assert rag.initialized

        # Should be closed after context
        assert not rag.initialized


class TestRepr:
    """Tests for __repr__ method."""

    def test_repr_contains_info(self):
        """Test __repr__ contains useful information."""
        from alpha_zero.orchestrator.rag_integration import RAGIntegration

        config = MockRAGConfig()
        rag = RAGIntegration(config)

        repr_str = repr(rag)

        assert "RAGIntegration" in repr_str
        assert "initialized=" in repr_str


class TestTrainingInsight:
    """Tests for TrainingInsight dataclass."""

    def test_default_timestamp(self):
        """Test TrainingInsight gets default timestamp."""
        from alpha_zero.orchestrator.rag_integration import TrainingInsight

        insight = TrainingInsight(content="Test")

        assert insight.timestamp != ""
        assert insight.category == "training"

    def test_custom_values(self):
        """Test TrainingInsight with custom values."""
        from alpha_zero.orchestrator.rag_integration import TrainingInsight

        insight = TrainingInsight(
            content="Win rate improved",
            category="performance",
            iteration=100,
            metrics={"win_rate": 0.7},
        )

        assert insight.content == "Win rate improved"
        assert insight.category == "performance"
        assert insight.iteration == 100


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_default_values(self):
        """Test QueryResult default values."""
        from alpha_zero.orchestrator.rag_integration import QueryResult

        result = QueryResult(content="Test", metadata={})

        assert result.content == "Test"
        assert result.score == 0.0
        assert result.source == ""
