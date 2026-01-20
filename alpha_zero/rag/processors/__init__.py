"""Document processing and embedding generation for RAG system."""

from alpha_zero.rag.processors.base import (
    DocumentProcessor,
    TextChunk,
    ProcessingResult,
)
from alpha_zero.rag.processors.text_processor import (
    TextDocumentProcessor,
    RecursiveTextSplitter,
)
from alpha_zero.rag.processors.embeddings import (
    EmbeddingGenerator,
    OpenAIEmbeddings,
    LocalEmbeddings,
    create_embedding_generator,
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
