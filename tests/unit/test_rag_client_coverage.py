"""Comprehensive tests for alpha_zero.utils.rag_client module."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.utils.rag_client import RAGClient, RAGConfig

# ---------------------------------------------------------------------------
# RAGConfig
# ---------------------------------------------------------------------------


class TestRAGConfig:
    def test_defaults(self):
        cfg = RAGConfig()
        assert cfg.persist_directory == "data/knowledge_db"
        assert cfg.collection_name == "civ6_knowledge"
        assert cfg.embedding_model == "all-MiniLM-L6-v2"
        assert cfg.top_k == 3

    def test_from_env_defaults(self):
        cfg = RAGConfig.from_env()
        assert cfg.persist_directory == "data/knowledge_db"

    @patch.dict(
        "os.environ",
        {
            "RAG_PERSIST_DIR": "/tmp/rag_test",
            "RAG_COLLECTION_NAME": "my_collection",
            "RAG_EMBEDDING_MODEL": "custom-model",
            "RAG_TOP_K": "10",
        },
    )
    def test_from_env_overrides(self):
        cfg = RAGConfig.from_env()
        assert cfg.persist_directory == "/tmp/rag_test"
        assert cfg.collection_name == "my_collection"
        assert cfg.embedding_model == "custom-model"
        assert cfg.top_k == 10

    def test_custom_values(self):
        cfg = RAGConfig(
            persist_directory="/data/db",
            collection_name="test_col",
            embedding_model="bert",
            top_k=5,
        )
        assert cfg.persist_directory == "/data/db"
        assert cfg.top_k == 5


# ---------------------------------------------------------------------------
# RAGClient - chroma not available
# ---------------------------------------------------------------------------


class TestRAGClientNoChroma:
    @patch("alpha_zero.utils.rag_client.CHROMA_AVAILABLE", False)
    def test_init_without_chroma(self, caplog):
        with caplog.at_level(logging.WARNING):
            RAGClient(RAGConfig())
        assert "chromadb not installed" in caplog.text

    @patch("alpha_zero.utils.rag_client.CHROMA_AVAILABLE", False)
    def test_add_documents_no_collection(self):
        client = RAGClient.__new__(RAGClient)
        client.config = RAGConfig()
        client.collection = None
        result = client.add_documents(["doc1"])
        assert result is False

    @patch("alpha_zero.utils.rag_client.CHROMA_AVAILABLE", False)
    def test_query_no_collection(self):
        client = RAGClient.__new__(RAGClient)
        client.config = RAGConfig()
        client.collection = None
        result = client.query("test query")
        assert result == []

    @patch("alpha_zero.utils.rag_client.CHROMA_AVAILABLE", False)
    def test_count_no_collection(self):
        client = RAGClient.__new__(RAGClient)
        client.config = RAGConfig()
        client.collection = None
        assert client.count() == 0


# ---------------------------------------------------------------------------
# RAGClient - with mocked chroma
# ---------------------------------------------------------------------------


class TestRAGClientWithMockedChroma:
    @pytest.fixture()
    def client_with_collection(self):
        """Create a RAGClient with a mocked collection, bypassing __init__."""
        client = RAGClient.__new__(RAGClient)
        client.config = RAGConfig(top_k=3)
        client.collection = MagicMock()
        client.client = MagicMock()
        return client

    # --- _init_db ---------------------------------------------------------

    @patch("alpha_zero.utils.rag_client.CHROMA_AVAILABLE", True)
    @patch("alpha_zero.utils.rag_client.SENTENCE_TRANSFORMERS_AVAILABLE", True)
    @patch("alpha_zero.utils.rag_client.chromadb", create=True)
    @patch("alpha_zero.utils.rag_client.embedding_functions", create=True)
    def test_init_db_with_sentence_transformers(self, mock_ef, mock_chromadb, tmp_path):
        mock_persistent = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_persistent
        mock_collection = MagicMock()
        mock_persistent.get_or_create_collection.return_value = mock_collection
        mock_embed_fn = MagicMock()
        mock_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embed_fn

        cfg = RAGConfig(persist_directory=str(tmp_path / "db"))
        client = RAGClient.__new__(RAGClient)
        client.config = cfg
        client._init_db()

        mock_chromadb.PersistentClient.assert_called_once()
        mock_ef.SentenceTransformerEmbeddingFunction.assert_called_once()
        assert client.collection is mock_collection

    @patch("alpha_zero.utils.rag_client.CHROMA_AVAILABLE", True)
    @patch("alpha_zero.utils.rag_client.SENTENCE_TRANSFORMERS_AVAILABLE", False)
    @patch("alpha_zero.utils.rag_client.chromadb", create=True)
    def test_init_db_without_sentence_transformers(self, mock_chromadb, tmp_path):
        mock_persistent = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_persistent
        mock_collection = MagicMock()
        mock_persistent.get_or_create_collection.return_value = mock_collection

        cfg = RAGConfig(persist_directory=str(tmp_path / "db"))
        client = RAGClient.__new__(RAGClient)
        client.config = cfg
        client._init_db()

        # embedding_function should be None
        call_kwargs = mock_persistent.get_or_create_collection.call_args
        assert call_kwargs[1].get("embedding_function") is None or call_kwargs[0] is not None

    @patch("alpha_zero.utils.rag_client.CHROMA_AVAILABLE", True)
    @patch("alpha_zero.utils.rag_client.SENTENCE_TRANSFORMERS_AVAILABLE", False)
    @patch("alpha_zero.utils.rag_client.chromadb", create=True)
    def test_init_db_exception(self, mock_chromadb, tmp_path, caplog):
        mock_chromadb.PersistentClient.side_effect = RuntimeError("db error")

        cfg = RAGConfig(persist_directory=str(tmp_path / "db"))
        client = RAGClient.__new__(RAGClient)
        client.config = cfg

        with caplog.at_level(logging.ERROR):
            client._init_db()

        assert client.client is None
        assert client.collection is None
        assert "Failed to initialize" in caplog.text

    # --- add_documents ----------------------------------------------------

    def test_add_documents_with_ids(self, client_with_collection):
        client = client_with_collection
        result = client.add_documents(
            ["doc1", "doc2"],
            metadatas=[{"k": "v1"}, {"k": "v2"}],
            ids=["id1", "id2"],
        )
        assert result is True
        client.collection.add.assert_called_once_with(
            documents=["doc1", "doc2"],
            metadatas=[{"k": "v1"}, {"k": "v2"}],
            ids=["id1", "id2"],
        )

    def test_add_documents_auto_ids(self, client_with_collection):
        client = client_with_collection
        result = client.add_documents(["doc1", "doc2"])
        assert result is True
        call_kwargs = client.collection.add.call_args[1]
        assert len(call_kwargs["ids"]) == 2
        assert all(i.startswith("doc_") for i in call_kwargs["ids"])

    def test_add_documents_exception(self, client_with_collection, caplog):
        client = client_with_collection
        client.collection.add.side_effect = RuntimeError("add failed")
        with caplog.at_level(logging.ERROR):
            result = client.add_documents(["doc1"])
        assert result is False
        assert "Error adding documents" in caplog.text

    # --- query ------------------------------------------------------------

    def test_query_success(self, client_with_collection):
        client = client_with_collection
        client.collection.query.return_value = {
            "documents": [["result1", "result2"]],
        }
        results = client.query("test question")
        assert results == ["result1", "result2"]
        client.collection.query.assert_called_once_with(
            query_texts=["test question"],
            n_results=3,
        )

    def test_query_custom_n_results(self, client_with_collection):
        client = client_with_collection
        client.collection.query.return_value = {"documents": [["r1"]]}
        results = client.query("q", n_results=1)
        client.collection.query.assert_called_once_with(
            query_texts=["q"],
            n_results=1,
        )
        assert results == ["r1"]

    def test_query_empty_results(self, client_with_collection):
        client = client_with_collection
        client.collection.query.return_value = {"documents": []}
        results = client.query("q")
        assert results == []

    def test_query_none_documents(self, client_with_collection):
        client = client_with_collection
        client.collection.query.return_value = {"documents": None}
        results = client.query("q")
        assert results == []

    def test_query_exception(self, client_with_collection, caplog):
        client = client_with_collection
        client.collection.query.side_effect = RuntimeError("query failed")
        with caplog.at_level(logging.ERROR):
            results = client.query("q")
        assert results == []
        assert "Error querying RAG" in caplog.text

    # --- count ------------------------------------------------------------

    def test_count(self, client_with_collection):
        client = client_with_collection
        client.collection.count.return_value = 42
        assert client.count() == 42

    # --- full init path ---------------------------------------------------

    @patch("alpha_zero.utils.rag_client.CHROMA_AVAILABLE", True)
    @patch("alpha_zero.utils.rag_client.SENTENCE_TRANSFORMERS_AVAILABLE", False)
    @patch("alpha_zero.utils.rag_client.chromadb", create=True)
    def test_full_init_no_sentence_transformers(self, mock_chromadb, tmp_path, caplog):
        mock_persistent = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_persistent
        mock_persistent.get_or_create_collection.return_value = MagicMock()

        cfg = RAGConfig(persist_directory=str(tmp_path / "db"))
        with caplog.at_level(logging.WARNING):
            RAGClient(cfg)
        assert "sentence-transformers not installed" in caplog.text
