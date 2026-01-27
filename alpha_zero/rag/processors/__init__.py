"""Document processing and embedding generation for RAG system."""

from alpha_zero.rag.processors.base import (
    DocumentProcessor,
    ProcessingResult,
    TextChunk,
)
from alpha_zero.rag.processors.embeddings import (
    EmbeddingGenerator,
    LocalEmbeddings,
    OpenAIEmbeddings,
    create_embedding_generator,
)
from alpha_zero.rag.processors.text_processor import (
    RecursiveTextSplitter,
    TextDocumentProcessor,
)

__all__ = [
    # Base
    "DocumentProcessor",
    "TextChunk",
    "ProcessingResult",
    # Text Processing
    "TextDocumentProcessor",
    "RecursiveTextSplitter",
    # Embeddings
    "EmbeddingGenerator",
    "OpenAIEmbeddings",
    "LocalEmbeddings",
    "create_embedding_generator",
]
