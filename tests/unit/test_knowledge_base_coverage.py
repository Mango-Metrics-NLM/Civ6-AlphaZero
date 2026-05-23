"""Comprehensive tests for alpha_zero.utils.knowledge_base module."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.utils.knowledge_base import KnowledgeBase


@pytest.fixture()
def mock_rag_client():
    client = MagicMock()
    client.add_documents.return_value = True
    return client


@pytest.fixture()
def kb(mock_rag_client):
    return KnowledgeBase(mock_rag_client)


# ---------------------------------------------------------------------------
# KnowledgeBase.__init__
# ---------------------------------------------------------------------------


class TestKnowledgeBaseInit:
    def test_stores_rag_client(self, mock_rag_client):
        kb = KnowledgeBase(mock_rag_client)
        assert kb.rag_client is mock_rag_client


# ---------------------------------------------------------------------------
# KnowledgeBase._chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_single_paragraph_under_limit(self, kb):
        text = "Hello world"
        chunks = list(kb._chunk_text(text, chunk_size=1000))
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_multiple_paragraphs_under_limit(self, kb):
        text = "First paragraph.\n\nSecond paragraph."
        chunks = list(kb._chunk_text(text, chunk_size=1000))
        assert len(chunks) == 1
        assert "First paragraph." in chunks[0]
        assert "Second paragraph." in chunks[0]

    def test_splits_when_exceeding_chunk_size(self, kb):
        para1 = "A" * 600
        para2 = "B" * 600
        text = f"{para1}\n\n{para2}"
        chunks = list(kb._chunk_text(text, chunk_size=700))
        assert len(chunks) == 2
        assert chunks[0] == para1
        assert chunks[1] == para2

    def test_empty_text(self, kb):
        chunks = list(kb._chunk_text("", chunk_size=100))
        assert len(chunks) == 0

    def test_only_whitespace_paragraphs(self, kb):
        text = "\n\n   \n\n   \n\n"
        chunks = list(kb._chunk_text(text, chunk_size=100))
        assert len(chunks) == 0

    def test_large_single_paragraph(self, kb):
        # A single paragraph exceeding chunk_size — should still be yielded
        big = "X" * 5000
        chunks = list(kb._chunk_text(big, chunk_size=1000))
        assert len(chunks) == 1
        assert chunks[0] == big

    def test_many_small_paragraphs_grouped(self, kb):
        paragraphs = ["word"] * 20
        text = "\n\n".join(paragraphs)
        chunks = list(kb._chunk_text(text, chunk_size=50))
        # All paragraphs are 4 chars; 50/4 ~ 12 fit per chunk
        assert len(chunks) >= 1
        total_words = sum(c.count("word") for c in chunks)
        assert total_words == 20

    def test_custom_chunk_size(self, kb):
        text = "A" * 100 + "\n\n" + "B" * 100
        chunks = list(kb._chunk_text(text, chunk_size=150))
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# KnowledgeBase.ingest_directory
# ---------------------------------------------------------------------------


class TestIngestDirectory:
    def test_directory_not_found(self, kb, tmp_path):
        result = kb.ingest_directory(tmp_path / "nonexistent")
        assert result == 0
        kb.rag_client.add_documents.assert_not_called()

    def test_empty_directory(self, kb, tmp_path):
        result = kb.ingest_directory(tmp_path)
        assert result == 0
        kb.rag_client.add_documents.assert_not_called()

    def test_ingest_single_file(self, kb, tmp_path):
        md = tmp_path / "test.md"
        md.write_text("Hello world.\n\nSecond paragraph.")
        result = kb.ingest_directory(tmp_path)
        assert result >= 1
        kb.rag_client.add_documents.assert_called_once()
        docs, metas, ids = kb.rag_client.add_documents.call_args[0]
        assert len(docs) == result
        assert all(isinstance(m, dict) for m in metas)
        assert all(isinstance(i, str) for i in ids)

    def test_ingest_multiple_files(self, kb, tmp_path):
        for i in range(3):
            (tmp_path / f"file{i}.md").write_text(f"Content {i}")
        result = kb.ingest_directory(tmp_path)
        assert result == 3
        kb.rag_client.add_documents.assert_called_once()

    def test_ingest_nested_directory(self, kb, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("Nested content")
        result = kb.ingest_directory(tmp_path, pattern="**/*.md")
        assert result >= 1

    def test_custom_pattern(self, kb, tmp_path):
        (tmp_path / "file.txt").write_text("txt content")
        (tmp_path / "file.md").write_text("md content")
        result = kb.ingest_directory(tmp_path, pattern="*.txt")
        assert result >= 1
        # Only .txt should be ingested
        docs = kb.rag_client.add_documents.call_args[0][0]
        assert any("txt content" in d for d in docs)

    def test_ingest_handles_read_error(self, kb, tmp_path, caplog):
        md = tmp_path / "bad.md"
        md.write_text("good content")
        # Create a file that will cause an error
        bad = tmp_path / "err.md"
        bad.write_text("some content")
        # Patch Path.read_text to fail for err.md only
        original_read = Path.read_text

        def patched_read(self_path, *args, **kwargs):
            if "err.md" in str(self_path):
                raise OSError("permission denied")
            return original_read(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", patched_read):
            result = kb.ingest_directory(tmp_path)
        # Should still ingest the good file
        assert result >= 1

    def test_ingest_rag_client_failure(self, kb, tmp_path, caplog):
        (tmp_path / "file.md").write_text("Content here")
        kb.rag_client.add_documents.return_value = False
        with caplog.at_level(logging.ERROR):
            result = kb.ingest_directory(tmp_path)
        assert result >= 1  # chunks counted regardless of add failure

    def test_metadata_contains_source_and_chunk_index(self, kb, tmp_path):
        (tmp_path / "doc.md").write_text("Para one.\n\nPara two.")
        kb.ingest_directory(tmp_path)
        _, metas, ids = kb.rag_client.add_documents.call_args[0]
        for m in metas:
            assert "source" in m
            assert "chunk_index" in m
        # IDs should contain the filename
        for doc_id in ids:
            assert "doc.md" in doc_id

    def test_string_path_accepted(self, kb, tmp_path):
        (tmp_path / "file.md").write_text("Content")
        result = kb.ingest_directory(str(tmp_path))
        assert result >= 1


# ---------------------------------------------------------------------------
# KnowledgeBase.clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_does_not_raise(self, kb):
        # clear() is a no-op currently, just verify it doesn't error
        kb.clear()
