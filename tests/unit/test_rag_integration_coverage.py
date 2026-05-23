"""Coverage tests for alpha_zero.orchestrator.rag_integration module."""

from unittest.mock import MagicMock, patch

from alpha_zero.orchestrator.rag_integration import (
    QueryResult,
    RAGIntegration,
    TrainingInsight,
)


def _make_config(enabled=True, store_insights=True, auto_ingest=False):
    config = MagicMock()
    config.enabled = enabled
    config.store_insights = store_insights
    config.auto_ingest = auto_ingest
    config.top_k = 5
    config.collection_name = "test_collection"
    config.persist_directory = "/tmp/test_rag"
    config.knowledge_dir = "/tmp/knowledge"
    return config


class TestQueryResult:
    def test_defaults(self):
        qr = QueryResult(content="test", metadata={})
        assert qr.score == 0.0
        assert qr.source == ""

    def test_with_values(self):
        qr = QueryResult(content="strategy", metadata={"a": 1}, score=0.9, source="guide.md")
        assert qr.content == "strategy"
        assert qr.score == 0.9


class TestTrainingInsight:
    def test_defaults(self):
        ti = TrainingInsight(content="test insight")
        assert ti.category == "training"
        assert ti.iteration == 0
        assert ti.metrics == {}
        assert ti.timestamp != ""

    def test_custom(self):
        ti = TrainingInsight(content="test", category="game", iteration=5, metrics={"wr": 0.6})
        assert ti.category == "game"
        assert ti.iteration == 5


class TestRAGIntegration:
    def test_init(self):
        config = _make_config()
        rag = RAGIntegration(config)
        assert rag.initialized is False
        assert rag.session_id == ""

    def test_initialize_disabled(self):
        config = _make_config(enabled=False)
        rag = RAGIntegration(config)
        assert rag.initialize() is False

    def test_initialize_already_initialized(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        assert rag.initialize() is True

    def test_initialize_success(self):
        config = _make_config()
        rag = RAGIntegration(config)
        mock_client = MagicMock()
        mock_kb = MagicMock()

        with (
            patch("alpha_zero.utils.factories.create_rag_client", return_value=mock_client),
            patch("alpha_zero.utils.factories.create_knowledge_base", return_value=mock_kb),
        ):
            result = rag.initialize()

        assert result is True
        assert rag.initialized is True
        assert rag.session_id != ""

    def test_initialize_no_client(self):
        config = _make_config()
        rag = RAGIntegration(config)
        with patch("alpha_zero.utils.factories.create_rag_client", return_value=None):
            result = rag.initialize()
        assert result is False

    def test_initialize_import_error(self):
        config = _make_config()
        rag = RAGIntegration(config)
        with patch(
            "alpha_zero.utils.factories.create_rag_client",
            side_effect=ImportError("no chromadb"),
        ):
            result = rag.initialize()
        assert result is False

    def test_initialize_exception(self):
        config = _make_config()
        rag = RAGIntegration(config)
        with patch(
            "alpha_zero.utils.factories.create_rag_client",
            side_effect=RuntimeError("error"),
        ):
            result = rag.initialize()
        assert result is False

    def test_initialize_with_auto_ingest(self, tmp_path):
        config = _make_config(auto_ingest=True)
        config.knowledge_dir = str(tmp_path / "knowledge")

        # Create knowledge dir with a file
        kdir = tmp_path / "knowledge"
        kdir.mkdir()
        (kdir / "test.md").write_text("content")

        rag = RAGIntegration(config)
        mock_client = MagicMock()
        mock_kb = MagicMock()
        with (
            patch("alpha_zero.utils.factories.create_rag_client", return_value=mock_client),
            patch("alpha_zero.utils.factories.create_knowledge_base", return_value=mock_kb),
        ):
            rag.initialize()

        mock_kb.add_document.assert_called_once()

    def test_ingest_no_knowledge_base(self):
        config = _make_config()
        rag = RAGIntegration(config)
        assert rag._ingest_knowledge_base() == 0

    def test_ingest_missing_dir(self):
        config = _make_config()
        config.knowledge_dir = "/nonexistent/path"
        rag = RAGIntegration(config)
        rag._knowledge_base = MagicMock()
        assert rag._ingest_knowledge_base() == 0

    def test_query_strategy_not_initialized(self):
        config = _make_config()
        rag = RAGIntegration(config)
        assert rag.query_strategy("test") == []

    def test_query_strategy_success(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._rag_client.query.return_value = [
            {"content": "Expand early", "metadata": {"source": "guide"}, "score": 0.9}
        ]
        results = rag.query_strategy("early game")
        assert len(results) == 1
        assert results[0].content == "Expand early"
        assert results[0].score == 0.9

    def test_query_strategy_with_context(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._rag_client.query.return_value = []
        rag.query_strategy("test", game_context={"turn": 10, "era": "ancient"})
        call_args = rag._rag_client.query.call_args
        assert "context:" in call_args[0][0]

    def test_query_strategy_exception(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._rag_client.query.side_effect = RuntimeError("error")
        assert rag.query_strategy("test") == []

    def test_query_similar_states_not_initialized(self):
        config = _make_config()
        rag = RAGIntegration(config)
        assert rag.query_similar_states("test") == []

    def test_query_similar_states_success(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._rag_client.query.return_value = [
            {"text": "state info", "metadata": {"category": "game_state"}, "score": 0.8}
        ]
        results = rag.query_similar_states("warrior near enemy")
        assert len(results) == 1

    def test_query_similar_states_exception(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._rag_client.query.side_effect = RuntimeError("error")
        assert rag.query_similar_states("test") == []

    def test_store_training_insight_not_initialized(self):
        config = _make_config()
        rag = RAGIntegration(config)
        assert rag.store_training_insight("test") is False

    def test_store_training_insight_disabled(self):
        config = _make_config(store_insights=False)
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        assert rag.store_training_insight("test") is False

    def test_store_training_insight_string(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._session_id = "test_session"
        assert rag.store_training_insight("good insight") is True
        rag._rag_client.add.assert_called_once()

    def test_store_training_insight_object(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        insight = TrainingInsight(content="test", category="game", iteration=3)
        assert rag.store_training_insight(insight, metadata={"extra": "data"}) is True

    def test_store_training_insight_exception(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._rag_client.add.side_effect = RuntimeError("error")
        assert rag.store_training_insight("test") is False

    def test_store_game_summary_not_initialized(self):
        config = _make_config()
        rag = RAGIntegration(config)
        assert rag.store_game_summary("summary", "win", {}, 1) is False

    def test_store_game_summary_success(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        assert rag.store_game_summary("summary", "win", {"wr": 0.8}, 5) is True

    def test_store_game_summary_exception(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._rag_client.add.side_effect = RuntimeError("error")
        assert rag.store_game_summary("summary", "loss", {}, 1) is False

    def test_get_session_insights_not_initialized(self):
        config = _make_config()
        rag = RAGIntegration(config)
        assert rag.get_session_insights() == []

    def test_get_session_insights_success(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._session_id = "test"
        rag._rag_client.query.return_value = [{"content": "insight1", "metadata": {}, "score": 0.5}]
        results = rag.get_session_insights()
        assert len(results) == 1

    def test_get_session_insights_exception(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._rag_client.query.side_effect = RuntimeError("error")
        assert rag.get_session_insights() == []

    def test_get_statistics_not_initialized(self):
        config = _make_config()
        rag = RAGIntegration(config)
        stats = rag.get_statistics()
        assert stats["initialized"] is False

    def test_get_statistics_success(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._rag_client.count.return_value = 42
        rag._session_id = "test"
        stats = rag.get_statistics()
        assert stats["initialized"] is True
        assert stats["document_count"] == 42

    def test_get_statistics_exception(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._rag_client.count.side_effect = RuntimeError("error")
        stats = rag.get_statistics()
        assert "error" in stats

    def test_close_not_initialized(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag.close()  # Should not raise

    def test_close_with_persist(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        mock_client = MagicMock()
        rag._rag_client = mock_client
        rag._session_id = "test"
        rag.close()
        # close() sets _rag_client = None in finally, so check mock directly
        mock_client.persist.assert_called_once()
        assert rag.initialized is False

    def test_close_exception(self):
        config = _make_config()
        rag = RAGIntegration(config)
        rag._initialized = True
        rag._rag_client = MagicMock()
        rag._rag_client.persist.side_effect = RuntimeError("error")
        rag.close()  # Should not raise
        assert rag.initialized is False

    def test_context_manager(self):
        config = _make_config(enabled=False)
        with RAGIntegration(config) as rag:
            assert rag.initialized is False
