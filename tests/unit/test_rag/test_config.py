"""Unit tests for RAG configuration module."""

from pathlib import Path

from alpha_zero.rag.config.settings import (
    EmbeddingConfig,
    EmbeddingModelType,
    LoggingConfig,
    LogLevel,
    ProcessorConfig,
    RAGConfig,
    VectorStoreConfig,
    VectorStoreType,
)


class TestVectorStoreConfig:
    """Tests for VectorStoreConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VectorStoreConfig()

        assert config.store_type == VectorStoreType.CHROMA
        assert config.collection_name == "rag_documents"
        assert config.dimension == 1536
        assert config.distance_metric == "cosine"
        assert config.persist_directory is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.FAISS,
            collection_name="custom_collection",
            dimension=768,
            distance_metric="l2",
            persist_directory=Path("/tmp/test"),
        )

        assert config.store_type == VectorStoreType.FAISS
        assert config.collection_name == "custom_collection"
        assert config.dimension == 768
        assert config.distance_metric == "l2"

    def test_from_env(self, monkeypatch):
        """Test loading configuration from environment."""
        monkeypatch.setenv("RAG_VECTOR_STORE_TYPE", "faiss")
        monkeypatch.setenv("RAG_VECTOR_STORE_COLLECTION", "test_collection")
        monkeypatch.setenv("RAG_VECTOR_STORE_DIMENSION", "512")

        config = VectorStoreConfig.from_env()

        assert config.store_type == VectorStoreType.FAISS
        assert config.collection_name == "test_collection"
        assert config.dimension == 512

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = VectorStoreConfig(
            store_type=VectorStoreType.MEMORY,
            collection_name="test",
        )
        data = config.to_dict()

        assert data["store_type"] == "memory"
        assert data["collection_name"] == "test"


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.model_type == EmbeddingModelType.OPENAI_3_SMALL
        assert config.batch_size == 100
        assert config.max_retries == 3
        assert config.timeout == 30.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.OPENAI_3_LARGE,
            api_key="test-key",
            batch_size=50,
            max_retries=5,
        )

        assert config.model_type == EmbeddingModelType.OPENAI_3_LARGE
        assert config.api_key == "test-key"
        assert config.batch_size == 50
        assert config.max_retries == 5

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("RAG_EMBEDDING_MODEL", "text-embedding-ada-002")
        monkeypatch.setenv("RAG_EMBEDDING_API_KEY", "sk-test")
        monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "200")

        config = EmbeddingConfig.from_env()

        assert config.model_type == EmbeddingModelType.OPENAI_ADA
        assert config.api_key == "sk-test"
        assert config.batch_size == 200


class TestProcessorConfig:
    """Tests for ProcessorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ProcessorConfig()

        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert len(config.separators) == 4
        assert config.include_metadata is True
        assert config.clean_whitespace is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ProcessorConfig(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n", " "],
        )

        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.separators == ["\n", " "]

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = ProcessorConfig(chunk_size=800)
        data = config.to_dict()

        assert data["chunk_size"] == 800
        assert "chunk_overlap" in data
        assert "separators" in data


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoggingConfig()

        assert config.level == LogLevel.INFO
        assert config.enable_tracing is False
        assert config.enable_metrics is True

    def test_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("RAG_LOG_ENABLE_TRACING", "true")

        config = LoggingConfig.from_env()

        assert config.level == LogLevel.DEBUG
        assert config.enable_tracing is True


class TestRAGConfig:
    """Tests for RAGConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RAGConfig()

        assert isinstance(config.vector_store, VectorStoreConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.processor, ProcessorConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.cache_ttl == 3600
        assert config.enable_caching is True

    def test_from_env(self, monkeypatch):
        """Test loading full config from environment."""
        monkeypatch.setenv("RAG_VECTOR_STORE_TYPE", "memory")
        monkeypatch.setenv("RAG_CACHE_TTL", "7200")
        monkeypatch.setenv("RAG_ENABLE_CACHING", "false")

        config = RAGConfig.from_env()

        assert config.vector_store.store_type == VectorStoreType.MEMORY
        assert config.cache_ttl == 7200
        assert config.enable_caching is False

    def test_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        yaml_content = """
vector_store:
  store_type: memory
  collection_name: yaml_test
  dimension: 768
embedding:
  model_type: local
processor:
  chunk_size: 500
cache_ttl: 1800
enable_caching: true
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = RAGConfig.from_yaml(yaml_file)

        assert config.vector_store.store_type == VectorStoreType.MEMORY
        assert config.vector_store.collection_name == "yaml_test"
        assert config.vector_store.dimension == 768
        assert config.processor.chunk_size == 500
        assert config.cache_ttl == 1800

    def test_to_yaml(self, tmp_path):
        """Test saving configuration to YAML file."""
        config = RAGConfig(
            cache_ttl=1800,
        )
        config.vector_store.store_type = VectorStoreType.MEMORY

        yaml_file = tmp_path / "output.yaml"
        config.to_yaml(yaml_file)

        assert yaml_file.exists()

        # Load and verify
        loaded = RAGConfig.from_yaml(yaml_file)
        assert loaded.cache_ttl == 1800
        assert loaded.vector_store.store_type == VectorStoreType.MEMORY

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = RAGConfig()
        config.vector_store.store_type = VectorStoreType.MEMORY
        config.embedding.model_type = EmbeddingModelType.LOCAL

        issues = config.validate()
        assert len(issues) == 0

    def test_validate_invalid_pinecone(self):
        """Test validation catches missing Pinecone credentials."""
        config = RAGConfig()
        config.vector_store.store_type = VectorStoreType.PINECONE
        config.vector_store.api_key = None

        issues = config.validate()
        assert len(issues) > 0
        assert any("API key" in issue for issue in issues)

    def test_validate_invalid_overlap(self):
        """Test validation catches invalid chunk overlap."""
        config = RAGConfig()
        config.processor.chunk_size = 500
        config.processor.chunk_overlap = 600  # Greater than chunk_size

        issues = config.validate()
        assert len(issues) > 0
        assert any("overlap" in issue.lower() for issue in issues)

    def test_to_dict_and_back(self):
        """Test round-trip serialization."""
        config = RAGConfig(
            cache_ttl=1200,
            enable_caching=False,
        )
        config.vector_store.store_type = VectorStoreType.FAISS
        config.embedding.model_type = EmbeddingModelType.LOCAL

        data = config.to_dict()
        restored = RAGConfig.from_dict(data)

        assert restored.cache_ttl == 1200
        assert restored.enable_caching is False
        assert restored.vector_store.store_type == VectorStoreType.FAISS
