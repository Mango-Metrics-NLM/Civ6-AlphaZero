"""Knowledge Base Manager for RAG Ingestion.

This module handles loading, chunking, and ingesting documents into the RAG system.

Usage:
    from alpha_zero.utils.rag_client import RAGClient
    from alpha_zero.utils.knowledge_base import KnowledgeBase

    rag = RAGClient()
    kb = KnowledgeBase(rag)
    kb.ingest_directory("data/knowledge")
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from alpha_zero.utils.rag_client import RAGClient

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Manages document ingestion for the RAG system."""

    def __init__(self, rag_client: RAGClient) -> None:
        self.rag_client = rag_client

    def ingest_directory(self, directory: str | Path, pattern: str = "**/*.md") -> int:
        """Ingest all matching files from a directory.

        Args:
            directory: Root directory to scan.
            pattern: Glob pattern for files (default: markdown files).

        Returns:
            Number of chunks ingested.
        """
        path = Path(directory)
        if not path.exists():
            logger.warning(f"Directory not found: {path}")
            return 0

        logger.info(f"Scanning {path} for {pattern}...")

        total_chunks = 0
        documents = []
        metadatas = []
        ids = []

        for file_path in path.glob(pattern):
            try:
                content = file_path.read_text(encoding="utf-8")
                chunks = list(self._chunk_text(content))

                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append(
                        {
                            "source": str(file_path),
                            "chunk_index": i,
                        }
                    )
                    ids.append(f"{file_path.name}_{i}")

                total_chunks += len(chunks)
                logger.debug(f"Processed {file_path.name}: {len(chunks)} chunks")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        if documents:
            success = self.rag_client.add_documents(documents, metadatas, ids)
            if success:
                logger.info(f"Successfully ingested {len(documents)} chunks from {directory}")
            else:
                logger.error("Failed to ingest documents into RAG client")

        return total_chunks

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> Iterator[str]:
        """Split text into chunks.

        Currently implements a simple splitter based on double newlines (paragraphs)
        and then grouping them to fit chunk_size.

        Args:
            text: Source text.
            chunk_size: Approximate size of chunk.
            overlap: (Not heavily used in this simple version yet).

        Yields:
            Text chunks.
        """
        # Split by headers or paragraphs
        paragraphs = text.split("\n\n")

        current_chunk: list[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            # If adding this paragraph exceeds chunk size, yield current chunk
            if current_len + para_len > chunk_size and current_chunk:
                yield "\n\n".join(current_chunk)
                current_chunk = []
                current_len = 0

            # If paragraph itself is huge, strictly we should split it,
            # but for now we just accept it (or specific handling could be added)
            current_chunk.append(para)
            current_len += para_len

        if current_chunk:
            yield "\n\n".join(current_chunk)

    def clear(self) -> None:
        """Clear the backend database (if supported)."""
        # Chroma's delete collection or reset logic if needed
        # For now, not exposed to avoid accidental data loss via this API
