"""Text document processor with configurable chunking strategies.

This module provides text processing capabilities including:
- Recursive text splitting with configurable separators
- Chunk overlap for context preservation
- Metadata extraction and propagation
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from alpha_zero.rag.config.settings import ProcessorConfig
from alpha_zero.rag.processors.base import (
    DocumentProcessor,
    ProcessingResult,
    TextChunk,
    TextSplitter,
)


class RecursiveTextSplitter(TextSplitter):
    """Recursive text splitter that tries different separators.

    Attempts to split on paragraph breaks first, then sentences,
    then words, to preserve semantic coherence.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
        keep_separator: bool = True,
    ):
        """Initialize recursive text splitter.

        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to try (in order)
            keep_separator: Whether to keep separators in output
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.keep_separator = keep_separator

    def split(self, text: str) -> list[str]:
        """Split text into chunks using recursive strategy.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using the given separators.

        Args:
            text: Text to split
            separators: Remaining separators to try

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Find the first separator that exists in the text
        separator = ""
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # Split on the separator
        if separator:
            splits = text.split(separator)
        else:
            # No separator found, split by character
            return self._split_by_character(text)

        # Merge splits to respect chunk size
        chunks = []
        current_chunk = []
        current_length = 0

        for i, split in enumerate(splits):
            split_with_sep = split + separator if self.keep_separator and i < len(splits) - 1 else split
            split_length = len(split_with_sep)

            if current_length + split_length <= self.chunk_size:
                current_chunk.append(split_with_sep)
                current_length += split_length
            else:
                # Current chunk is full
                if current_chunk:
                    chunk_text = "".join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)

                # Check if split itself is too large
                if split_length > self.chunk_size:
                    # Recursively split with remaining separators
                    remaining_seps = separators[separators.index(separator) + 1:] if separator in separators else [""]
                    sub_chunks = self._split_recursive(split, remaining_seps)
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_length = 0
                else:
                    current_chunk = [split_with_sep]
                    current_length = split_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = "".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        # Apply overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _split_by_character(self, text: str) -> list[str]:
        """Split text by character when no separator works.

        Args:
            text: Text to split

        Returns:
            List of chunks split by character limit
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap between chunks.

        Args:
            chunks: List of chunks

        Returns:
            Chunks with overlap applied
        """
        if not chunks:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]

            # Get overlap from end of previous chunk
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) > self.chunk_overlap else prev_chunk

            # Prepend overlap to current chunk
            merged = overlap_text + " " + curr_chunk
            result.append(merged.strip())

        return result


class TextDocumentProcessor(DocumentProcessor):
    """Document processor for text content.

    Handles text processing with configurable chunking and metadata extraction.

    Example:
        config = ProcessorConfig(chunk_size=1000, chunk_overlap=200)
        processor = TextDocumentProcessor(config)
        result = await processor.process(text, source="document.txt")
    """

    def __init__(self, config: ProcessorConfig | None = None):
        """Initialize text document processor.

        Args:
            config: Processing configuration
        """
        self.config = config or ProcessorConfig()
        self.splitter = RecursiveTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
        )

    async def process(self, content: str, source: str = "") -> ProcessingResult:
        """Process text content into chunks.

        Args:
            content: Raw text content to process
            source: Source identifier for the content

        Returns:
            ProcessingResult with extracted chunks
        """
        start_time = time.perf_counter()
        errors = []

        try:
            # Clean whitespace if configured
            if self.config.clean_whitespace:
                content = self._clean_whitespace(content)

            # Split into chunks
            chunk_texts = self.splitter.split(content)

            # Create TextChunk objects with position tracking
            chunks = []
            current_pos = 0

            for i, chunk_text in enumerate(chunk_texts):
                # Find position in original content
                start_pos = content.find(chunk_text, current_pos)
                if start_pos == -1:
                    start_pos = current_pos
                end_pos = start_pos + len(chunk_text)

                chunk = TextChunk(
                    content=chunk_text,
                    start_char=start_pos,
                    end_char=end_pos,
                    chunk_index=i,
                    total_chunks=len(chunk_texts),
                )
                chunks.append(chunk)
                current_pos = end_pos

        except Exception as e:
            errors.append(f"Processing error: {str(e)}")
            chunks = []

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return ProcessingResult(
            chunks=chunks,
            source=source,
            total_chars=len(content),
            processing_time_ms=elapsed_ms,
            errors=errors,
        )

    async def process_file(self, file_path: str) -> ProcessingResult:
        """Process a file into chunks.

        Args:
            file_path: Path to the file to process

        Returns:
            ProcessingResult with extracted chunks
        """
        path = Path(file_path)

        if not path.exists():
            return ProcessingResult(
                chunks=[],
                source=str(path),
                errors=[f"File not found: {file_path}"],
            )

        try:
            content = path.read_text(encoding="utf-8")
            return await self.process(content, source=str(path))
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ["latin-1", "cp1252"]:
                try:
                    content = path.read_text(encoding=encoding)
                    return await self.process(content, source=str(path))
                except UnicodeDecodeError:
                    continue

            return ProcessingResult(
                chunks=[],
                source=str(path),
                errors=[f"Unable to decode file: {file_path}"],
            )
        except Exception as e:
            return ProcessingResult(
                chunks=[],
                source=str(path),
                errors=[f"Error reading file: {str(e)}"],
            )

    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace in text.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Replace multiple whitespace with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

    @classmethod
    def from_config(cls, config: ProcessorConfig) -> TextDocumentProcessor:
        """Create processor from configuration.

        Args:
            config: ProcessorConfig instance

        Returns:
            Configured TextDocumentProcessor
        """
        return cls(config)
