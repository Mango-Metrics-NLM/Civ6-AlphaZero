"""Base classes for document processing.

This module defines the abstract interface for document processors and
data structures for processing results.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from alpha_zero.rag.stores.base import Document, DocumentMetadata


@dataclass
class TextChunk:
    """A chunk of text extracted from a document.

    Attributes:
        content: The text content of the chunk
        start_char: Starting character position in original document
        end_char: Ending character position in original document
        chunk_index: Index of this chunk within the source
        total_chunks: Total number of chunks from the source
        metadata: Additional metadata for this chunk
    """

    content: str
    start_char: int = 0
    end_char: int = 0
    chunk_index: int = 0
    total_chunks: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Return length of content."""
        return len(self.content)

    def to_document(self, source: str, doc_id: str | None = None) -> Document:
        """Convert chunk to a Document for storage.

        Args:
            source: Source identifier for the document
            doc_id: Optional document ID

        Returns:
            Document instance
        """
        return Document(
            id=doc_id or f"{source}_{self.chunk_index}",
            content=self.content,
            metadata=DocumentMetadata(
                source=source,
                created_at=datetime.now(UTC),
                chunk_index=self.chunk_index,
                total_chunks=self.total_chunks,
                custom={
                    "start_char": self.start_char,
                    "end_char": self.end_char,
                    **self.metadata,
                },
            ),
        )


@dataclass
class ProcessingResult:
    """Result of document processing.

    Attributes:
        chunks: List of text chunks extracted
        source: Source identifier
        total_chars: Total characters in original document
        processing_time_ms: Time taken to process in milliseconds
        errors: Any errors encountered during processing
    """

    chunks: list[TextChunk]
    source: str = ""
    total_chars: int = 0
    processing_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if processing was successful."""
        return len(self.errors) == 0

    @property
    def num_chunks(self) -> int:
        """Get number of chunks produced."""
        return len(self.chunks)

    def to_documents(self) -> list[Document]:
        """Convert all chunks to Documents.

        Returns:
            List of Document instances
        """
        return [chunk.to_document(self.source) for chunk in self.chunks]


class DocumentProcessor(ABC):
    """Abstract base class for document processors.

    Document processors handle the conversion of raw text or files into
    chunks suitable for embedding and storage.
    """

    @abstractmethod
    async def process(self, content: str, source: str = "") -> ProcessingResult:
        """Process text content into chunks.

        Args:
            content: Raw text content to process
            source: Source identifier for the content

        Returns:
            ProcessingResult with extracted chunks
        """

    @abstractmethod
    async def process_file(self, file_path: str) -> ProcessingResult:
        """Process a file into chunks.

        Args:
            file_path: Path to the file to process

        Returns:
            ProcessingResult with extracted chunks
        """

    async def process_batch(
        self,
        contents: list[tuple[str, str]],
    ) -> list[ProcessingResult]:
        """Process multiple documents concurrently.

        Args:
            contents: List of (content, source) tuples

        Returns:
            List of ProcessingResults
        """
        tasks = [self.process(content, source) for content, source in contents]
        return await asyncio.gather(*tasks)


class TextSplitter(ABC):
    """Abstract base class for text splitting strategies."""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
