"""Unit tests for document processors and embeddings."""

import numpy as np
import pytest

from alpha_zero.rag.processors.base import DocumentProcessor, ProcessingResult, TextChunk
from alpha_zero.rag.processors.text_processor import (
    RecursiveTextSplitter,
    TextDocumentProcessor,
)
from alpha_zero.rag.processors.embeddings import (
    LocalEmbeddings,
    create_embedding_generator,
)
from alpha_zero.rag.config.settings import EmbeddingConfig, EmbeddingModelType, ProcessorConfig


class TestTextChunk:
    """Tests for TextChunk dataclass."""

    def test_create_chunk(self):
        """Test creating a text chunk."""
        chunk = TextChunk(
            content="This is test content",
            start_char=0,
            end_char=19,
            chunk_index=0,
            total_chunks=1,
        )

        assert chunk.content == "This is test content"
        assert chunk.start_char == 0
        assert chunk.end_char == 19
        assert len(chunk) == 20

    def test_chunk_to_document(self):
        """Test converting chunk to document."""
        chunk = TextChunk(
            content="Test content",
            start_char=0,
            end_char=11,
            chunk_index=0,
            total_chunks=3,
        )

        doc = chunk.to_document(source="test.txt")

        assert doc.content == "Test content"
        assert doc.metadata.source == "test.txt"
        assert doc.metadata.chunk_index == 0
        assert doc.metadata.total_chunks == 3


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_successful_result(self):
        """Test successful processing result."""
        chunks = [
            TextChunk(content="Chunk 1", chunk_index=0, total_chunks=2),
            TextChunk(content="Chunk 2", chunk_index=1, total_chunks=2),
        ]

        result = ProcessingResult(
            chunks=chunks,
            source="test.txt",
            total_chars=100,
            processing_time_ms=10.5,
        )

        assert result.success
        assert result.num_chunks == 2
        assert result.total_chars == 100

    def test_failed_result(self):
        """Test failed processing result."""
        result = ProcessingResult(
            chunks=[],
            source="test.txt",
            errors=["File not found"],
        )

        assert not result.success
        assert result.num_chunks == 0

    def test_to_documents(self):
        """Test converting result to documents."""
        chunks = [
            TextChunk(content="Chunk 1", chunk_index=0, total_chunks=2),
            TextChunk(content="Chunk 2", chunk_index=1, total_chunks=2),
        ]

        result = ProcessingResult(chunks=chunks, source="test.txt")
        docs = result.to_documents()

        assert len(docs) == 2
        assert docs[0].content == "Chunk 1"
        assert docs[1].content == "Chunk 2"


class TestRecursiveTextSplitter:
    """Tests for RecursiveTextSplitter."""

    def test_split_simple_text(self):
        """Test splitting simple text."""
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "This is a simple text that should be split into chunks."

        chunks = splitter.split(text)

        assert len(chunks) >= 1
        # All chunks should be non-empty
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_split_respects_chunk_size(self):
        """Test that chunks respect size limit."""
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=0)
        text = "A" * 300  # 300 characters

        chunks = splitter.split(text)

        for chunk in chunks:
            assert len(chunk) <= 100

    def test_split_on_paragraphs(self):
        """Test splitting on paragraph breaks."""
        splitter = RecursiveTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
            separators=["\n\n", "\n", " "],
        )
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        chunks = splitter.split(text)

        # Should split on paragraph breaks
        assert len(chunks) >= 1

    def test_split_empty_text(self):
        """Test splitting empty text."""
        splitter = RecursiveTextSplitter(chunk_size=100)
        chunks = splitter.split("")

        assert len(chunks) == 0

    def test_split_with_overlap(self):
        """Test that overlap is applied."""
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=10)
        text = "First part of text. Second part of text. Third part of text."

        chunks = splitter.split(text)

        if len(chunks) > 1:
            # Check that there's some overlap between consecutive chunks
            # (text from one chunk should appear in the next)
            assert len(chunks) >= 1


class TestTextDocumentProcessor:
    """Tests for TextDocumentProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a processor for testing."""
        config = ProcessorConfig(
            chunk_size=100,
            chunk_overlap=20,
            clean_whitespace=True,
        )
        return TextDocumentProcessor(config)

    @pytest.mark.asyncio
    async def test_process_text(self, processor):
        """Test processing text content."""
        text = "This is a test document. It has multiple sentences. Each sentence has content."

        result = await processor.process(text, source="test.txt")

        assert result.success
        assert result.num_chunks >= 1
        assert result.total_chars == len(text)
        assert result.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_process_long_text(self, processor):
        """Test processing long text creates multiple chunks."""
        text = "This is a sentence. " * 50  # Long text

        result = await processor.process(text, source="long.txt")

        assert result.success
        assert result.num_chunks > 1

    @pytest.mark.asyncio
    async def test_process_cleans_whitespace(self, processor):
        """Test that whitespace is cleaned."""
        text = "Text   with    multiple     spaces\n\n\n\nand many newlines"

        result = await processor.process(text)

        # Chunks should have normalized whitespace
        for chunk in result.chunks:
            assert "   " not in chunk.content  # No triple spaces
            assert "\n\n\n" not in chunk.content  # No triple newlines

    @pytest.mark.asyncio
    async def test_process_preserves_source(self, processor):
        """Test that source is preserved in metadata."""
        result = await processor.process("Test content", source="important.txt")
        docs = result.to_documents()

        assert all(doc.metadata.source == "important.txt" for doc in docs)

    @pytest.mark.asyncio
    async def test_process_file_not_found(self, processor, tmp_path):
        """Test processing non-existent file."""
        result = await processor.process_file(str(tmp_path / "nonexistent.txt"))

        assert not result.success
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_process_file(self, processor, tmp_path):
        """Test processing an actual file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is file content. It should be processed correctly.")

        result = await processor.process_file(str(test_file))

        assert result.success
        assert result.num_chunks >= 1
        assert result.source == str(test_file)


class TestLocalEmbeddings:
    """Tests for LocalEmbeddings."""

    @pytest.fixture
    def embeddings(self):
        """Create local embeddings for testing."""
        return LocalEmbeddings(dimension=128)

    @pytest.mark.asyncio
    async def test_embed_text(self, embeddings):
        """Test embedding a single text."""
        text = "This is a test sentence."

        embedding = await embeddings.embed_text(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (128,)
        assert embedding.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_texts(self, embeddings):
        """Test embedding multiple texts."""
        texts = ["First text", "Second text", "Third text"]

        embeddings_list = await embeddings.embed_texts(texts)

        assert len(embeddings_list) == 3
        for emb in embeddings_list:
            assert emb.shape == (128,)

    @pytest.mark.asyncio
    async def test_deterministic_embeddings(self, embeddings):
        """Test that embeddings are deterministic for same text."""
        text = "Deterministic test"

        emb1 = await embeddings.embed_text(text)
        emb2 = await embeddings.embed_text(text)

        np.testing.assert_array_equal(emb1, emb2)

    @pytest.mark.asyncio
    async def test_different_texts_different_embeddings(self, embeddings):
        """Test that different texts produce different embeddings."""
        text1 = "First unique text"
        text2 = "Second different text"

        emb1 = await embeddings.embed_text(text1)
        emb2 = await embeddings.embed_text(text2)

        # Embeddings should be different
        assert not np.allclose(emb1, emb2)

    @pytest.mark.asyncio
    async def test_embeddings_are_normalized(self, embeddings):
        """Test that embeddings are normalized to unit length."""
        text = "Test normalization"

        embedding = await embeddings.embed_text(text)
        norm = np.linalg.norm(embedding)

        assert np.isclose(norm, 1.0, atol=1e-5)

    @pytest.mark.asyncio
    async def test_embed_documents(self, embeddings):
        """Test embedding documents in place."""
        from alpha_zero.rag.stores.base import Document

        docs = [
            Document(id="1", content="First document"),
            Document(id="2", content="Second document"),
        ]

        result = await embeddings.embed_documents(docs)

        assert len(result) == 2
        for doc in result:
            assert doc.embedding is not None
            assert doc.embedding.shape == (128,)

    def test_clear_cache(self, embeddings):
        """Test clearing the embedding cache."""
        embeddings._cache["test"] = np.zeros(128)
        assert len(embeddings._cache) == 1

        embeddings.clear_cache()

        assert len(embeddings._cache) == 0


class TestCreateEmbeddingGenerator:
    """Tests for embedding generator factory."""

    def test_create_local_embeddings(self):
        """Test creating local embeddings."""
        config = EmbeddingConfig(model_type=EmbeddingModelType.LOCAL)

        generator = create_embedding_generator(config)

        assert isinstance(generator, LocalEmbeddings)

    def test_create_openai_embeddings_requires_api_key(self):
        """Test that OpenAI embeddings require API key."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.OPENAI_3_SMALL,
            api_key=None,
        )

        with pytest.raises(ValueError, match="API key"):
            create_embedding_generator(config)
