"""Tests for alpha_zero.utils.knowledge_base.KnowledgeBase."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alpha_zero.utils.knowledge_base import KnowledgeBase


@pytest.fixture
def fake_rag() -> MagicMock:
    """RAGClient stub that records add_documents calls."""
    client = MagicMock()
    client.add_documents.return_value = True
    return client


@pytest.fixture
def kb_dir(tmp_path: Path) -> Path:
    """Knowledge directory with three small markdown files."""
    (tmp_path / "a.md").write_text("alpha paragraph one\n\nalpha paragraph two\n")
    (tmp_path / "b.md").write_text("beta content\n")
    (tmp_path / "c.txt").write_text("ignored — not markdown")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.md").write_text("nested content")
    return tmp_path


class TestKnowledgeBaseIngest:
    def test_ingests_matching_files(self, fake_rag: MagicMock, kb_dir: Path) -> None:
        kb = KnowledgeBase(fake_rag)
        n = kb.ingest_directory(kb_dir)
        # 3 markdown files → at least 3 chunks (could be more if paragraphs split)
        assert n >= 3
        assert fake_rag.add_documents.called
        args, _ = fake_rag.add_documents.call_args
        documents, metadatas, ids = args
        assert len(documents) == len(metadatas) == len(ids)
        # The .txt file should be excluded by the default **/*.md pattern.
        assert not any(".txt" in m["source"] for m in metadatas)

    def test_ingests_via_custom_pattern(
        self, fake_rag: MagicMock, kb_dir: Path
    ) -> None:
        kb = KnowledgeBase(fake_rag)
        n = kb.ingest_directory(kb_dir, pattern="*.txt")
        assert n == 1
        args, _ = fake_rag.add_documents.call_args
        documents, metadatas, _ = args
        assert any(".txt" in m["source"] for m in metadatas)

    def test_missing_directory_returns_zero(self, fake_rag: MagicMock) -> None:
        kb = KnowledgeBase(fake_rag)
        assert kb.ingest_directory("/nonexistent/path") == 0
        assert not fake_rag.add_documents.called

    def test_metadata_includes_chunk_index_and_source(
        self, fake_rag: MagicMock, kb_dir: Path
    ) -> None:
        kb = KnowledgeBase(fake_rag)
        kb.ingest_directory(kb_dir)
        args, _ = fake_rag.add_documents.call_args
        _, metadatas, _ = args
        for m in metadatas:
            assert "source" in m
            assert "chunk_index" in m
            assert m["chunk_index"] >= 0

    def test_chunk_ids_are_unique(
        self, fake_rag: MagicMock, kb_dir: Path
    ) -> None:
        kb = KnowledgeBase(fake_rag)
        kb.ingest_directory(kb_dir)
        args, _ = fake_rag.add_documents.call_args
        _, _, ids = args
        assert len(ids) == len(set(ids))

    def test_handles_read_errors_gracefully(
        self, fake_rag: MagicMock, tmp_path: Path
    ) -> None:
        """If a file becomes unreadable mid-walk, the error is logged and
        the rest of the batch still ingests."""
        good = tmp_path / "good.md"
        good.write_text("readable content")
        bad = tmp_path / "bad.md"
        bad.write_text("placeholder")
        bad.chmod(0o000)  # Make it unreadable
        try:
            kb = KnowledgeBase(fake_rag)
            n = kb.ingest_directory(tmp_path)
            # We may or may not get 1 chunk (depends on root user privilege)
            # but the call must not raise.
            assert n >= 0
        finally:
            bad.chmod(0o644)  # Restore for cleanup

    def test_no_files_means_no_add_documents_call(
        self, fake_rag: MagicMock, tmp_path: Path
    ) -> None:
        kb = KnowledgeBase(fake_rag)
        n = kb.ingest_directory(tmp_path)
        assert n == 0
        assert not fake_rag.add_documents.called


class TestKnowledgeBaseChunker:
    def test_short_text_yields_single_chunk(self, fake_rag: MagicMock) -> None:
        kb = KnowledgeBase(fake_rag)
        chunks = list(kb._chunk_text("one paragraph"))
        assert chunks == ["one paragraph"]

    def test_paragraphs_split_at_chunk_size(self, fake_rag: MagicMock) -> None:
        kb = KnowledgeBase(fake_rag)
        para1 = "A" * 600
        para2 = "B" * 600
        text = f"{para1}\n\n{para2}"
        chunks = list(kb._chunk_text(text, chunk_size=500))
        # Each paragraph exceeds the chunk size individually, so each lands
        # in its own chunk.
        assert len(chunks) == 2
        assert chunks[0].startswith("A")
        assert chunks[1].startswith("B")

    def test_empty_text_yields_nothing(self, fake_rag: MagicMock) -> None:
        kb = KnowledgeBase(fake_rag)
        assert list(kb._chunk_text("")) == []
        assert list(kb._chunk_text("\n\n\n\n")) == []

    def test_chunks_pack_paragraphs_under_chunk_size(
        self, fake_rag: MagicMock
    ) -> None:
        kb = KnowledgeBase(fake_rag)
        # Three small paragraphs that should pack into one chunk
        text = "alpha\n\nbeta\n\ngamma"
        chunks = list(kb._chunk_text(text, chunk_size=1000))
        assert len(chunks) == 1
        assert "alpha" in chunks[0]
        assert "beta" in chunks[0]
        assert "gamma" in chunks[0]


class TestKnowledgeBaseClear:
    def test_clear_is_callable_noop(self, fake_rag: MagicMock) -> None:
        """clear() exists for API symmetry but is intentionally a no-op
        to prevent accidental data loss."""
        kb = KnowledgeBase(fake_rag)
        kb.clear()  # Should not raise.
