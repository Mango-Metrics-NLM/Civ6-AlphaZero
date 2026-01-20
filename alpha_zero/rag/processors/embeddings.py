"""Embedding generation for RAG system.

This module provides embedding generators for converting text to vectors:
- OpenAI embeddings (text-embedding-3-small, text-embedding-ada-002)
- Local embeddings for testing
- Batch processing with retry logic
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alpha_zero.rag.config.settings import EmbeddingConfig, EmbeddingModelType
from alpha_zero.rag.stores.base import Document


@dataclass
class EmbeddingResult:
    """Result of embedding generation.

    Attributes:
        embeddings: Generated embedding vectors
        model: Model used for generation
        total_tokens: Total tokens processed
        processing_time_ms: Time taken in milliseconds
    """

    embeddings: list[np.ndarray]
    model: str = ""
    total_tokens: int = 0
    processing_time_ms: float = 0.0


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators."""

    def __init__(self, dimension: int = 1536):
        """Initialize embedding generator.

        Args:
            dimension: Output embedding dimension
        """
        self.dimension = dimension

    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    async def embed_documents(self, documents: list[Document]) -> list[Document]:
        """Generate embeddings for documents in place.

        Args:
            documents: Documents to embed

        Returns:
            Documents with embeddings populated
        """
        texts = [doc.content for doc in documents]
        embeddings = await self.embed_texts(texts)

        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        return documents


class OpenAIEmbeddings(EmbeddingGenerator):
    """OpenAI embedding generator.

    Supports text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002.

    Example:
        config = EmbeddingConfig(model_type=EmbeddingModelType.OPENAI_3_SMALL)
        embeddings = OpenAIEmbeddings.from_config(config)
        vector = await embeddings.embed_text("Hello world")
    """

    MODEL_DIMENSIONS = {
        EmbeddingModelType.OPENAI_ADA: 1536,
        EmbeddingModelType.OPENAI_3_SMALL: 1536,
        EmbeddingModelType.OPENAI_3_LARGE: 3072,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        """Initialize OpenAI embeddings.

        Args:
            api_key: OpenAI API key
            model: Model name
            batch_size: Maximum batch size for API calls
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        # Determine dimension based on model
        dimension = 1536
        for model_type, dim in self.MODEL_DIMENSIONS.items():
            if model_type.value == model:
                dimension = dim
                break

        super().__init__(dimension)
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None

    async def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)
            except ImportError as e:
                raise ImportError(
                    "OpenAI package not installed. Install with: pip install openai"
                ) from e
        return self._client

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        client = await self._get_client()
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._embed_batch_with_retry(client, batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch_with_retry(
        self, client, texts: list[str]
    ) -> list[np.ndarray]:
        """Embed a batch of texts with retry logic.

        Args:
            client: OpenAI client
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await client.embeddings.create(
                    model=self.model,
                    input=texts,
                )

                embeddings = []
                for item in response.data:
                    embedding = np.array(item.embedding, dtype=np.float32)
                    embeddings.append(embedding)

                return embeddings

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                continue

        raise RuntimeError(f"Failed to generate embeddings after {self.max_retries} attempts: {last_error}")

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> OpenAIEmbeddings:
        """Create from configuration.

        Args:
            config: EmbeddingConfig instance

        Returns:
            Configured OpenAIEmbeddings
        """
        if not config.api_key:
            raise ValueError("API key required for OpenAI embeddings")

        return cls(
            api_key=config.api_key,
            model=config.model_type.value,
            batch_size=config.batch_size,
            max_retries=config.max_retries,
            timeout=config.timeout,
        )


class LocalEmbeddings(EmbeddingGenerator):
    """Local embedding generator for testing.

    Generates deterministic embeddings based on text hashing.
    Useful for testing without API calls.

    Example:
        embeddings = LocalEmbeddings(dimension=1536)
        vector = await embeddings.embed_text("Hello world")
    """

    def __init__(self, dimension: int = 1536, seed: int = 42):
        """Initialize local embeddings.

        Args:
            dimension: Embedding dimension
            seed: Random seed for deterministic output
        """
        super().__init__(dimension)
        self.seed = seed
        self._cache: dict[str, np.ndarray] = {}

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (deterministic based on text hash)
        """
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate deterministic embedding from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], "big")

        rng = np.random.Generator(np.random.PCG64(seed))
        embedding = rng.standard_normal(self.dimension).astype(np.float32)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self._cache[cache_key] = embedding
        return embedding

    async def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return [await self.embed_text(text) for text in texts]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()


class SentenceTransformerEmbeddings(EmbeddingGenerator):
    """Sentence Transformer embedding generator.

    Uses Hugging Face sentence-transformers for local embedding generation.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """Initialize Sentence Transformer embeddings.

        Args:
            model_name: Name of the model to use
            device: Device for inference ("cpu" or "cuda")
            batch_size: Batch size for encoding
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e

        self._model = SentenceTransformer(model_name, device=device)
        dimension = self._model.get_sentence_embedding_dimension()
        super().__init__(dimension)

        self.model_name = model_name
        self.batch_size = batch_size
        self._loop = None

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

        def _encode():
            return self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        embeddings = await self._loop.run_in_executor(None, _encode)
        return [emb.astype(np.float32) for emb in embeddings]


def create_embedding_generator(config: EmbeddingConfig) -> EmbeddingGenerator:
    """Factory function to create an embedding generator.

    Args:
        config: EmbeddingConfig instance

    Returns:
        Configured EmbeddingGenerator

    Raises:
        ValueError: If model type is not supported
    """
    if config.model_type in (
        EmbeddingModelType.OPENAI_ADA,
        EmbeddingModelType.OPENAI_3_SMALL,
        EmbeddingModelType.OPENAI_3_LARGE,
    ):
        return OpenAIEmbeddings.from_config(config)
    elif config.model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS:
        return SentenceTransformerEmbeddings(
            model_name=config.model_name or "all-MiniLM-L6-v2",
            batch_size=config.batch_size,
        )
    elif config.model_type == EmbeddingModelType.LOCAL:
        return LocalEmbeddings()
    else:
        raise ValueError(f"Unsupported embedding model type: {config.model_type}")
