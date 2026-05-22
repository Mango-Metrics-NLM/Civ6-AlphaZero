"""LATS-style reflection memory for learning from failures.

This module provides infrastructure to store failed game trajectories with
LLM-generated critiques, enabling retrieval of relevant failure patterns
during MCTS evaluation to avoid repeating mistakes.

Usage:
    from alpha_zero.core.reflection_memory import (
        ReflectionConfig,
        ReflectionMemory,
    )

    # Create reflection memory
    config = ReflectionConfig()
    memory = ReflectionMemory(config)
    await memory.initialize()

    # Add a failed game
    await memory.add_failed_game(trajectory, critique="Lost due to weak defense")

    # Query similar failures during MCTS
    similar = await memory.query_similar_failures(current_state)
    penalty = memory.get_value_adjustment_sync(state, action)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from alpha_zero.distributed.shared_storage import GameTrajectory
from alpha_zero.rag.stores.base import Document, DocumentMetadata
from alpha_zero.utils.compat import UTC

if TYPE_CHECKING:
    from alpha_zero.rag.stores.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class ReflectionConfig:
    """Configuration for reflection memory system."""

    # Storage backend
    storage_backend: str = "memory"  # "memory", "chroma", "faiss"
    persist_directory: str = "data/reflections"
    collection_name: str = "game_reflections"

    # Embedding settings
    embedding_dimension: int = 384  # Default for sentence-transformers
    use_observation_embedding: bool = True  # Use raw observation as embedding

    # Retrieval settings
    similarity_threshold: float = 0.7
    max_retrievals: int = 5

    # Action warning settings
    warning_penalty: float = 0.1  # Penalty applied to MCTS value
    max_warnings_per_action: int = 3

    # State encoding
    spatial_flatten: bool = True  # Flatten spatial features for embedding
    include_vector: bool = True  # Include vector features in embedding

    # Memory limits
    max_entries: int = 10000  # Maximum stored reflections
    prune_threshold: float = 0.5  # Prune entries with relevance below this

    @classmethod
    def from_env(cls) -> ReflectionConfig:
        """Load configuration from environment variables."""
        return cls(
            storage_backend=os.getenv("CIV6AZ_REFLECTION_BACKEND", "memory"),
            persist_directory=os.getenv("CIV6AZ_REFLECTION_PERSIST_DIR", "data/reflections"),
            collection_name=os.getenv("CIV6AZ_REFLECTION_COLLECTION", "game_reflections"),
            embedding_dimension=int(os.getenv("CIV6AZ_REFLECTION_EMBED_DIM", "384")),
            similarity_threshold=float(os.getenv("CIV6AZ_REFLECTION_SIMILARITY_THRESHOLD", "0.7")),
            max_retrievals=int(os.getenv("CIV6AZ_REFLECTION_MAX_RETRIEVALS", "5")),
            warning_penalty=float(os.getenv("CIV6AZ_REFLECTION_WARNING_PENALTY", "0.1")),
            max_entries=int(os.getenv("CIV6AZ_REFLECTION_MAX_ENTRIES", "10000")),
        )


@dataclass
class ReflectionEntry:
    """A stored reflection about a failed game."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trajectory_summary: str = ""  # Compressed game state sequence
    failure_point_idx: int = 0  # Index where critical mistake occurred
    critique: str = ""  # LLM-generated verbal critique
    action_taken: int = -1  # The bad action
    better_alternatives: list[int] = field(default_factory=list)
    game_outcome: float = 0.0  # -1 for loss, 0 for draw
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    # State information for similarity search
    state_hash: str = ""  # Hash of the failure state for deduplication
    state_embedding: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "trajectory_summary": self.trajectory_summary,
            "failure_point_idx": self.failure_point_idx,
            "critique": self.critique,
            "action_taken": self.action_taken,
            "better_alternatives": self.better_alternatives,
            "game_outcome": self.game_outcome,
            "timestamp": self.timestamp,
            "state_hash": self.state_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReflectionEntry:
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            trajectory_summary=data.get("trajectory_summary", ""),
            failure_point_idx=data.get("failure_point_idx", 0),
            critique=data.get("critique", ""),
            action_taken=data.get("action_taken", -1),
            better_alternatives=data.get("better_alternatives", []),
            game_outcome=data.get("game_outcome", 0.0),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            state_hash=data.get("state_hash", ""),
            metadata=data.get("metadata", {}),
        )


class StateEncoder(Protocol):
    """Protocol for encoding game states for similarity search."""

    def encode_state(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Encode a single observation into an embedding vector.

        Args:
            obs: Observation dictionary with 'spatial' and 'vector' keys.

        Returns:
            Embedding vector of shape (embedding_dimension,).
        """
        ...

    def encode_trajectory(self, trajectory: GameTrajectory) -> np.ndarray:
        """Encode a full trajectory into an embedding.

        Args:
            trajectory: Game trajectory with training examples.

        Returns:
            Embedding vector representing the trajectory.
        """
        ...


class DefaultStateEncoder:
    """Default encoder using observation vector and spatial features.

    Creates embeddings by combining flattened spatial features with
    the vector observation, then projecting to the target dimension.
    """

    def __init__(self, config: ReflectionConfig) -> None:
        """Initialize encoder.

        Args:
            config: Reflection configuration.
        """
        self.config = config
        self._projection_matrix: np.ndarray | None = None

    def encode_state(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Encode observation to embedding vector.

        Args:
            obs: Observation with 'spatial' and 'vector' keys.

        Returns:
            Embedding vector.
        """
        features = []

        # Add vector features
        if self.config.include_vector and "vector" in obs:
            features.append(obs["vector"].flatten())

        # Add spatial features (flattened or pooled)
        if self.config.spatial_flatten and "spatial" in obs:
            spatial = obs["spatial"]
            # Use spatial statistics instead of full flatten to reduce dimension
            spatial_mean = spatial.mean(axis=(1, 2))  # Mean per channel
            spatial_max = spatial.max(axis=(1, 2))  # Max per channel
            features.append(spatial_mean)
            features.append(spatial_max)

        if not features:
            # Fallback: return zero vector
            return np.zeros(self.config.embedding_dimension, dtype=np.float32)

        # Concatenate all features
        combined = np.concatenate(features).astype(np.float32)

        # Project to target dimension
        return self._project_to_dimension(combined)

    def _project_to_dimension(self, features: np.ndarray) -> np.ndarray:
        """Project features to target embedding dimension.

        Args:
            features: Raw feature vector.

        Returns:
            Projected vector of target dimension (L2 normalized).
        """
        target_dim = self.config.embedding_dimension

        if len(features) == target_dim:
            # Normalize and return
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            return features

        if len(features) < target_dim:
            # Pad with zeros
            result = np.zeros(target_dim, dtype=np.float32)
            result[: len(features)] = features
            # Normalize
            norm = np.linalg.norm(result)
            if norm > 0:
                result = result / norm
            return result

        # Random projection for dimensionality reduction
        if self._projection_matrix is None or self._projection_matrix.shape[0] != len(features):
            # Create deterministic random projection
            rng = np.random.default_rng(42)
            self._projection_matrix = rng.standard_normal((len(features), target_dim)).astype(
                np.float32
            )
            # Normalize columns
            norms = np.linalg.norm(self._projection_matrix, axis=0, keepdims=True)
            self._projection_matrix /= np.maximum(norms, 1e-8)

        projected = features @ self._projection_matrix

        # L2 normalize
        norm = np.linalg.norm(projected)
        if norm > 0:
            projected /= norm

        return projected.astype(np.float32)

    def encode_trajectory(self, trajectory: GameTrajectory) -> np.ndarray:
        """Encode trajectory using mean of state embeddings.

        Args:
            trajectory: Game trajectory.

        Returns:
            Mean embedding of all states.
        """
        if not trajectory.examples:
            return np.zeros(self.config.embedding_dimension, dtype=np.float32)

        embeddings = []
        for example in trajectory.examples:
            obs = {
                "spatial": example.spatial,
                "vector": example.vector,
            }
            embeddings.append(self.encode_state(obs))

        return np.mean(embeddings, axis=0).astype(np.float32)  # type: ignore[no-any-return]


class ReflectionMemory:
    """LATS-style reflection memory for learning from failures.

    Stores failed game trajectories with LLM-generated critiques,
    enabling retrieval of relevant failure patterns during MCTS
    evaluation to avoid repeating mistakes.
    """

    def __init__(
        self,
        config: ReflectionConfig | None = None,
        vector_store: VectorStore | None = None,
        state_encoder: StateEncoder | None = None,
    ) -> None:
        """Initialize reflection memory.

        Args:
            config: Reflection configuration. Uses defaults if None.
            vector_store: Custom vector store (for testing/dependency injection).
            state_encoder: Custom state encoder (for testing).
        """
        self.config = config or ReflectionConfig()
        self._logger = logging.getLogger(__name__)

        # Lazy initialization for optional dependencies
        self._vector_store = vector_store
        self._encoder = state_encoder or DefaultStateEncoder(self.config)
        self._initialized = False

        # Local cache for fast lookups
        self._entry_cache: dict[str, ReflectionEntry] = {}
        self._action_warnings_cache: dict[str, list[ReflectionEntry]] = {}

        # Statistics
        self._stats = {
            "total_entries": 0,
            "unique_patterns": 0,
            "queries": 0,
            "cache_hits": 0,
        }

    @property
    def initialized(self) -> bool:
        """Check if memory is initialized."""
        return self._initialized

    @property
    def size(self) -> int:
        """Get number of stored reflections."""
        return self._stats["total_entries"]

    @property
    def unique_patterns(self) -> int:
        """Get number of unique failure patterns."""
        return self._stats["unique_patterns"]

    async def initialize(self) -> bool:
        """Initialize storage backend.

        Returns:
            True if initialization succeeded.
        """
        if self._initialized:
            return True

        try:
            if self._vector_store is None:
                self._vector_store = self._create_vector_store()

            await self._vector_store.initialize()
            self._initialized = True
            self._logger.info(
                f"ReflectionMemory initialized with {self.config.storage_backend} backend"
            )
            return True

        except Exception as e:
            self._logger.warning(f"Failed to initialize ReflectionMemory: {e}")
            # Graceful degradation - memory still usable without persistence
            self._initialized = True
            return False

    def _create_vector_store(self) -> VectorStore:
        """Create vector store based on config."""
        from alpha_zero.rag.config.settings import VectorStoreConfig, VectorStoreType
        from alpha_zero.rag.stores.factory import create_vector_store

        store_type_map = {
            "memory": VectorStoreType.MEMORY,
            "chroma": VectorStoreType.CHROMA,
            "faiss": VectorStoreType.FAISS,
        }

        store_type = store_type_map.get(self.config.storage_backend, VectorStoreType.MEMORY)

        store_config = VectorStoreConfig(
            store_type=store_type,
            collection_name=self.config.collection_name,
            dimension=self.config.embedding_dimension,
            persist_directory=Path(self.config.persist_directory)
            if self.config.persist_directory
            else None,
        )

        return create_vector_store(store_config)

    async def add_failed_game(
        self,
        trajectory: GameTrajectory,
        critique: str,
        failure_point_idx: int | None = None,
        action_taken: int | None = None,
        better_alternatives: list[int] | None = None,
    ) -> str:
        """Add a failed game with its critique.

        Args:
            trajectory: The failed game trajectory.
            critique: LLM-generated verbal critique.
            failure_point_idx: Index of the critical mistake state.
            action_taken: The bad action that was taken.
            better_alternatives: Suggested better actions.

        Returns:
            Entry ID of the stored reflection.
        """
        if not self._initialized:
            await self.initialize()

        # Determine failure point (last state if not specified)
        if failure_point_idx is None:
            failure_point_idx = len(trajectory.examples) - 1 if trajectory.examples else 0

        # Get the failure state
        failure_state = None
        if trajectory.examples and failure_point_idx < len(trajectory.examples):
            example = trajectory.examples[failure_point_idx]
            failure_state = {
                "spatial": example.spatial,
                "vector": example.vector,
            }

        # Create embedding
        if failure_state:
            embedding = self._encoder.encode_state(failure_state)
            state_hash = self._compute_state_hash(failure_state)
        else:
            embedding = self._encoder.encode_trajectory(trajectory)
            state_hash = hashlib.md5(critique.encode(), usedforsecurity=False).hexdigest()

        # Check for duplicate
        if state_hash in self._entry_cache:
            existing = self._entry_cache[state_hash]
            # Update critique if new one is longer
            if len(critique) > len(existing.critique):
                existing.critique = critique
            return existing.id

        # Create entry
        entry = ReflectionEntry(
            trajectory_summary=self._summarize_trajectory(trajectory),
            failure_point_idx=failure_point_idx,
            critique=critique,
            action_taken=action_taken or -1,
            better_alternatives=better_alternatives or [],
            game_outcome=-1.0 if trajectory.winner is not None else 0.0,
            state_hash=state_hash,
            state_embedding=embedding,
            metadata={
                "game_length": trajectory.game_length,
                "winner": trajectory.winner,
            },
        )

        # Store in vector store
        if self._vector_store is not None:
            doc = Document(
                id=entry.id,
                content=entry.critique,
                embedding=embedding,
                metadata=DocumentMetadata(
                    source="reflection",
                    custom=entry.to_dict(),
                ),
            )
            await self._vector_store.add_documents([doc])

        # Update cache
        self._entry_cache[state_hash] = entry
        self._stats["total_entries"] += 1
        self._stats["unique_patterns"] = len(self._entry_cache)

        # Clear action warnings cache
        self._action_warnings_cache.clear()

        self._logger.debug(f"Added reflection: {entry.id[:8]}... - {critique[:50]}...")
        return entry.id

    def _compute_state_hash(self, state: dict[str, np.ndarray]) -> str:
        """Compute hash of game state for deduplication."""
        hasher = hashlib.md5(usedforsecurity=False)

        if "vector" in state:
            hasher.update(state["vector"].tobytes())

        if "spatial" in state:
            # Use spatial statistics for hash to handle slight variations
            spatial = state["spatial"]
            stats = np.array(
                [
                    spatial.mean(),
                    spatial.std(),
                    spatial.max(),
                    spatial.min(),
                ]
            )
            hasher.update(np.round(stats, 2).tobytes())

        return hasher.hexdigest()

    def _summarize_trajectory(self, trajectory: GameTrajectory) -> str:
        """Create a brief summary of the game trajectory."""
        return (
            f"Game {trajectory.game_id}: {trajectory.game_length} turns, winner={trajectory.winner}"
        )

    async def query_similar_failures(
        self,
        current_state: dict[str, np.ndarray],
        top_k: int | None = None,
    ) -> list[ReflectionEntry]:
        """Find similar past failures for current state.

        Args:
            current_state: Current game observation.
            top_k: Maximum results to return.

        Returns:
            List of similar failure entries.
        """
        if not self._initialized or self._vector_store is None:
            return []

        top_k = top_k or self.config.max_retrievals
        self._stats["queries"] += 1

        # Encode current state
        query_embedding = self._encoder.encode_state(current_state)

        # Search vector store
        try:
            results = await self._vector_store.search(
                query_embedding,
                top_k=top_k,
            )
        except Exception as e:
            self._logger.warning(f"Vector search failed: {e}")
            return []

        # Filter by similarity threshold and convert to entries
        entries = []
        for result in results:
            if result.score >= self.config.similarity_threshold:
                entry_data = result.document.metadata.custom
                entry = ReflectionEntry.from_dict(entry_data)
                entries.append(entry)

        return entries

    async def get_action_warnings(
        self,
        state: dict[str, np.ndarray],
        candidate_actions: list[int],
    ) -> dict[int, list[str]]:
        """Get warnings for candidate actions based on past failures.

        Args:
            state: Current game state.
            candidate_actions: Actions being considered.

        Returns:
            Dictionary mapping action -> list of warning messages.
        """
        similar_failures = await self.query_similar_failures(state)

        warnings: dict[int, list[str]] = {}

        for entry in similar_failures:
            if entry.action_taken in candidate_actions:
                action = entry.action_taken
                if action not in warnings:
                    warnings[action] = []

                if len(warnings[action]) < self.config.max_warnings_per_action:
                    warnings[action].append(entry.critique)

        return warnings

    def get_value_adjustment_sync(
        self,
        state: dict[str, np.ndarray],
        action: int,
    ) -> float:
        """Get value penalty for an action based on reflection memory (sync version).

        This is a synchronous wrapper for use in MCTS evaluation.

        Args:
            state: Current game state.
            action: Action being evaluated.

        Returns:
            Negative value penalty if action is risky, 0 otherwise.
        """
        # Use cached results if available
        state_hash = self._compute_state_hash(state)
        cache_key = f"{state_hash}_{action}"

        if cache_key in self._action_warnings_cache:
            cached_entries = self._action_warnings_cache[cache_key]
            if cached_entries:
                return -self.config.warning_penalty * len(cached_entries)
            return 0.0

        # Search through local cache (fast path)
        matching_entries = []
        query_embedding = self._encoder.encode_state(state)

        for entry in self._entry_cache.values():
            if entry.action_taken == action and entry.state_embedding is not None:
                # Compute similarity
                similarity = self._cosine_similarity(query_embedding, entry.state_embedding)
                if similarity >= self.config.similarity_threshold:
                    matching_entries.append(entry)

        # Cache result
        self._action_warnings_cache[cache_key] = matching_entries

        if matching_entries:
            penalty = -self.config.warning_penalty * min(
                len(matching_entries), self.config.max_warnings_per_action
            )
            return penalty

        return 0.0

    async def get_value_adjustment(
        self,
        state: dict[str, np.ndarray],
        action: int,
    ) -> float:
        """Async version of get_value_adjustment_sync."""
        return self.get_value_adjustment_sync(state, action)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def export_reflections(self, path: str) -> int:
        """Export all reflections to JSON file.

        Args:
            path: Output file path.

        Returns:
            Number of entries exported.
        """
        entries = [entry.to_dict() for entry in self._entry_cache.values()]

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w") as f:
            json.dump({"reflections": entries, "count": len(entries)}, f, indent=2)

        self._logger.info(f"Exported {len(entries)} reflections to {path}")
        return len(entries)

    async def import_reflections(self, path: str) -> int:
        """Import reflections from JSON file.

        Args:
            path: Input file path.

        Returns:
            Number of entries imported.
        """
        if not os.path.exists(path):
            self._logger.warning(f"Import file not found: {path}")
            return 0

        with open(path) as f:
            data = json.load(f)

        entries = data.get("reflections", [])
        imported = 0

        for entry_data in entries:
            entry = ReflectionEntry.from_dict(entry_data)

            if entry.state_hash not in self._entry_cache:
                self._entry_cache[entry.state_hash] = entry
                imported += 1

        self._stats["total_entries"] = len(self._entry_cache)
        self._stats["unique_patterns"] = len(self._entry_cache)

        self._logger.info(f"Imported {imported} reflections from {path}")
        return imported

    async def close(self) -> None:
        """Close resources."""
        if self._vector_store is not None:
            await self._vector_store.close()
        self._initialized = False

    def get_statistics(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            **self._stats,
            "cache_size": len(self._entry_cache),
            "initialized": self._initialized,
        }

    def clear(self) -> None:
        """Clear all entries from memory."""
        self._entry_cache.clear()
        self._action_warnings_cache.clear()
        self._stats = {
            "total_entries": 0,
            "unique_patterns": 0,
            "queries": 0,
            "cache_hits": 0,
        }
        self._logger.info("ReflectionMemory cleared")


def create_reflection_memory(
    config: ReflectionConfig | None = None,
) -> ReflectionMemory:
    """Create and return a ReflectionMemory instance.

    Args:
        config: Configuration. Uses defaults if None.

    Returns:
        Uninitialized ReflectionMemory (call initialize() before use).
    """
    return ReflectionMemory(config or ReflectionConfig.from_env())
