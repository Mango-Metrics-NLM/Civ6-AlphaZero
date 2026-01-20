"""Factory for creating vector store instances.

This module provides a factory pattern for creating vector store instances
based on configuration, enabling runtime backend selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alpha_zero.rag.config.settings import VectorStoreConfig, VectorStoreType
from alpha_zero.rag.stores.base import VectorStore

if TYPE_CHECKING:
    pass


class VectorStoreFactory:
    """Factory for creating vector store instances.

    Supports registration of custom store implementations and provides
    a unified interface for store creation.

    Example:
        factory = VectorStoreFactory()
        factory.register("custom", CustomVectorStore)
        store = factory.create(config)
    """

    _registry: dict[VectorStoreType, type[VectorStore]] = {}

    @classmethod
    def register(
        cls,
        store_type: VectorStoreType,
        store_class: type[VectorStore],
    ) -> None:
        """Register a vector store implementation.

        Args:
            store_type: Type identifier for the store
            store_class: Store class to register
        """
        cls._registry[store_type] = store_class

    @classmethod
    def create(cls, config: VectorStoreConfig) -> VectorStore:
        """Create a vector store instance from configuration.

        Args:
            config: Vector store configuration

        Returns:
            Configured vector store instance

        Raises:
            ValueError: If store type is not supported
        """
        # Lazy import to avoid circular dependencies
        from alpha_zero.rag.stores.chroma_store import ChromaVectorStore
        from alpha_zero.rag.stores.faiss_store import FAISSVectorStore
        from alpha_zero.rag.stores.memory_store import MemoryVectorStore

        # Register default implementations
        if VectorStoreType.MEMORY not in cls._registry:
            cls._registry[VectorStoreType.MEMORY] = MemoryVectorStore
        if VectorStoreType.CHROMA not in cls._registry:
            cls._registry[VectorStoreType.CHROMA] = ChromaVectorStore
        if VectorStoreType.FAISS not in cls._registry:
            cls._registry[VectorStoreType.FAISS] = FAISSVectorStore

        store_class = cls._registry.get(config.store_type)

        if store_class is None:
            available = [t.value for t in cls._registry.keys()]
            raise ValueError(
                f"Unsupported store type: {config.store_type}. "
                f"Available types: {available}"
            )

        # Create instance with appropriate parameters
        if config.store_type == VectorStoreType.MEMORY:
            return MemoryVectorStore(
                collection_name=config.collection_name,
                dimension=config.dimension,
                distance_metric=config.distance_metric,
            )
        elif config.store_type == VectorStoreType.CHROMA:
            return ChromaVectorStore(
                collection_name=config.collection_name,
                dimension=config.dimension,
                persist_directory=config.persist_directory,
                distance_metric=config.distance_metric,
            )
        elif config.store_type == VectorStoreType.FAISS:
            return FAISSVectorStore(
                collection_name=config.collection_name,
                dimension=config.dimension,
                persist_directory=config.persist_directory,
                distance_metric=config.distance_metric,
            )
        else:
            # Use generic from_config if available
            if hasattr(store_class, "from_config"):
                return store_class.from_config(config)
            return store_class(
                collection_name=config.collection_name,
                dimension=config.dimension,
            )

    @classmethod
    def available_types(cls) -> list[VectorStoreType]:
        """Get list of available store types.

        Returns:
            List of registered store types
        """
        return list(cls._registry.keys())


def create_vector_store(config: VectorStoreConfig) -> VectorStore:
    """Convenience function to create a vector store.

    Args:
        config: Vector store configuration

    Returns:
        Configured vector store instance
    """
    return VectorStoreFactory.create(config)
