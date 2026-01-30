"""Configuration settings for RAG system using Pydantic models.

This module provides type-safe, validated configuration for all RAG components.
Configuration can be loaded from environment variables, YAML files, or constructed
programmatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import yaml


class VectorStoreType(str, Enum):
    """Supported vector store backends."""

    CHROMA = "chroma"
    FAISS = "faiss"
    PINECONE = "pinecone"
    MEMORY = "memory"  # For testing


class EmbeddingModelType(str, Enum):
    """Supported embedding models."""

    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    LOCAL = "local"  # For testing without API


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


T = TypeVar("T")


def _get_env(key: str, default: T, type_: type[T] = str) -> T:  # type: ignore[assignment]
    """Get environment variable with type conversion."""
    value = os.getenv(key)
    if value is None:
        return default
    if type_ is bool:
        return value.lower() in ("true", "1", "yes")  # type: ignore[return-value]
    return type_(value)  # type: ignore[return-value, call-arg]


@dataclass
class VectorStoreConfig:
    """Configuration for vector store backend.

    Attributes:
        store_type: Type of vector store to use
        collection_name: Name of the collection/index
        persist_directory: Directory for persistent storage (Chroma, FAISS)
        host: Host for remote stores (Pinecone)
        api_key: API key for remote stores
        dimension: Embedding dimension
        distance_metric: Distance metric for similarity search
    """

    store_type: VectorStoreType = VectorStoreType.CHROMA
    collection_name: str = "rag_documents"
    persist_directory: Path | None = None
    host: str | None = None
    api_key: str | None = None
    dimension: int = 1536
    distance_metric: str = "cosine"

    @classmethod
    def from_env(cls, prefix: str = "RAG_VECTOR_STORE") -> VectorStoreConfig:
        """Create configuration from environment variables.

        Args:
            prefix: Environment variable prefix

        Returns:
            Configured VectorStoreConfig instance
        """
        store_type_str = os.getenv(f"{prefix}_TYPE", "chroma")
        try:
            store_type = VectorStoreType(store_type_str.lower())
        except ValueError:
            store_type = VectorStoreType.CHROMA

        persist_dir = os.getenv(f"{prefix}_PERSIST_DIR")

        return cls(
            store_type=store_type,
            collection_name=_get_env(f"{prefix}_COLLECTION", "rag_documents"),
            persist_directory=Path(persist_dir) if persist_dir else None,
            host=os.getenv(f"{prefix}_HOST"),
            api_key=os.getenv(f"{prefix}_API_KEY"),
            dimension=_get_env(f"{prefix}_DIMENSION", 1536, int),
            distance_metric=_get_env(f"{prefix}_DISTANCE_METRIC", "cosine"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "store_type": self.store_type.value,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory) if self.persist_directory else None,
            "host": self.host,
            "dimension": self.dimension,
            "distance_metric": self.distance_metric,
        }


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model.

    Attributes:
        model_type: Type of embedding model
        model_name: Specific model name/path
        api_key: API key for remote models
        batch_size: Batch size for embedding generation
        max_retries: Maximum retries for API calls
        timeout: Timeout in seconds for API calls
    """

    model_type: EmbeddingModelType = EmbeddingModelType.OPENAI_3_SMALL
    model_name: str | None = None
    api_key: str | None = None
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 30.0

    @classmethod
    def from_env(cls, prefix: str = "RAG_EMBEDDING") -> EmbeddingConfig:
        """Create configuration from environment variables."""
        model_type_str = os.getenv(f"{prefix}_MODEL", "text-embedding-3-small")
        try:
            model_type = EmbeddingModelType(model_type_str)
        except ValueError:
            model_type = EmbeddingModelType.OPENAI_3_SMALL

        return cls(
            model_type=model_type,
            model_name=os.getenv(f"{prefix}_MODEL_NAME"),
            api_key=os.getenv(f"{prefix}_API_KEY") or os.getenv("OPENAI_API_KEY"),
            batch_size=_get_env(f"{prefix}_BATCH_SIZE", 100, int),
            max_retries=_get_env(f"{prefix}_MAX_RETRIES", 3, int),
            timeout=_get_env(f"{prefix}_TIMEOUT", 30.0, float),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }


@dataclass
class ProcessorConfig:
    """Configuration for document processor.

    Attributes:
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks
        separators: Text separators for chunking
        include_metadata: Whether to include metadata in chunks
        clean_whitespace: Whether to normalize whitespace
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])
    include_metadata: bool = True
    clean_whitespace: bool = True

    @classmethod
    def from_env(cls, prefix: str = "RAG_PROCESSOR") -> ProcessorConfig:
        """Create configuration from environment variables."""
        separators_str = os.getenv(f"{prefix}_SEPARATORS")
        separators = separators_str.split(",") if separators_str else None

        return cls(
            chunk_size=_get_env(f"{prefix}_CHUNK_SIZE", 1000, int),
            chunk_overlap=_get_env(f"{prefix}_CHUNK_OVERLAP", 200, int),
            separators=separators or ["\n\n", "\n", ". ", " "],
            include_metadata=_get_env(f"{prefix}_INCLUDE_METADATA", True, bool),
            clean_whitespace=_get_env(f"{prefix}_CLEAN_WHITESPACE", True, bool),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": self.separators,
            "include_metadata": self.include_metadata,
            "clean_whitespace": self.clean_whitespace,
        }


@dataclass
class LoggingConfig:
    """Configuration for logging and debugging.

    Attributes:
        level: Log level
        enable_tracing: Enable distributed tracing
        correlation_id_header: Header name for correlation ID
        log_format: Log format string
        enable_metrics: Enable metrics collection
    """

    level: LogLevel = LogLevel.INFO
    enable_tracing: bool = False
    correlation_id_header: str = "X-Correlation-ID"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s"
    enable_metrics: bool = True

    @classmethod
    def from_env(cls, prefix: str = "RAG_LOG") -> LoggingConfig:
        """Create configuration from environment variables."""
        level_str = os.getenv(f"{prefix}_LEVEL", "INFO").upper()
        try:
            level = LogLevel(level_str)
        except ValueError:
            level = LogLevel.INFO

        return cls(
            level=level,
            enable_tracing=_get_env(f"{prefix}_ENABLE_TRACING", False, bool),
            correlation_id_header=_get_env(
                f"{prefix}_CORRELATION_HEADER",
                "X-Correlation-ID",
            ),
            enable_metrics=_get_env(f"{prefix}_ENABLE_METRICS", True, bool),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "enable_tracing": self.enable_tracing,
            "correlation_id_header": self.correlation_id_header,
            "log_format": self.log_format,
            "enable_metrics": self.enable_metrics,
        }


@dataclass
class RAGConfig:
    """Main configuration for RAG system.

    Combines all component configurations into a single unified configuration.

    Attributes:
        vector_store: Vector store configuration
        embedding: Embedding model configuration
        processor: Document processor configuration
        logging: Logging configuration
        cache_ttl: Cache time-to-live in seconds
        enable_caching: Whether to enable query caching
    """

    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache_ttl: int = 3600
    enable_caching: bool = True

    @classmethod
    def from_env(cls) -> RAGConfig:
        """Create configuration from environment variables.

        Returns:
            Fully configured RAGConfig instance
        """
        return cls(
            vector_store=VectorStoreConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            processor=ProcessorConfig.from_env(),
            logging=LoggingConfig.from_env(),
            cache_ttl=_get_env("RAG_CACHE_TTL", 3600, int),
            enable_caching=_get_env("RAG_ENABLE_CACHING", True, bool),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RAGConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Configured RAGConfig instance
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Configured RAGConfig instance
        """
        vector_store_data = data.get("vector_store", {})
        if "store_type" in vector_store_data:
            vector_store_data["store_type"] = VectorStoreType(
                vector_store_data["store_type"],
            )
        if "persist_directory" in vector_store_data and vector_store_data["persist_directory"]:
            vector_store_data["persist_directory"] = Path(
                vector_store_data["persist_directory"],
            )

        embedding_data = data.get("embedding", {})
        if "model_type" in embedding_data:
            embedding_data["model_type"] = EmbeddingModelType(
                embedding_data["model_type"],
            )

        logging_data = data.get("logging", {})
        if "level" in logging_data:
            logging_data["level"] = LogLevel(logging_data["level"])

        return cls(
            vector_store=VectorStoreConfig(**vector_store_data),
            embedding=EmbeddingConfig(**embedding_data),
            processor=ProcessorConfig(**data.get("processor", {})),
            logging=LoggingConfig(**logging_data),
            cache_ttl=data.get("cache_ttl", 3600),
            enable_caching=data.get("enable_caching", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "vector_store": self.vector_store.to_dict(),
            "embedding": self.embedding.to_dict(),
            "processor": self.processor.to_dict(),
            "logging": self.logging.to_dict(),
            "cache_ttl": self.cache_ttl,
            "enable_caching": self.enable_caching,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Output path for YAML file
        """
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        # Vector store validation
        if self.vector_store.store_type == VectorStoreType.PINECONE:
            if not self.vector_store.api_key:
                issues.append("Pinecone requires API key")
            if not self.vector_store.host:
                issues.append("Pinecone requires host configuration")

        # Embedding validation
        if self.embedding.model_type in (
            EmbeddingModelType.OPENAI_ADA,
            EmbeddingModelType.OPENAI_3_SMALL,
            EmbeddingModelType.OPENAI_3_LARGE,
        ):
            if not self.embedding.api_key:
                issues.append("OpenAI embedding models require API key")

        # Processor validation
        if self.processor.chunk_overlap >= self.processor.chunk_size:
            issues.append("chunk_overlap must be less than chunk_size")

        return issues
