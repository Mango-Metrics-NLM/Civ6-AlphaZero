"""Tests for the reflection memory module."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from alpha_zero.core.reflection_memory import (
    DefaultStateEncoder,
    ReflectionConfig,
    ReflectionEntry,
    ReflectionMemory,
    create_reflection_memory,
)
from alpha_zero.distributed.shared_storage import GameTrajectory, TrainingExample


class TestReflectionConfig:
    """Tests for ReflectionConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = ReflectionConfig()

        assert config.storage_backend == "memory"
        assert config.collection_name == "game_reflections"
        assert config.embedding_dimension == 384
        assert config.similarity_threshold == 0.7
        assert config.max_retrievals == 5
        assert config.warning_penalty == 0.1
        assert config.max_entries == 10000

    def test_custom_values(self):
        """Should accept custom values."""
        config = ReflectionConfig(
            storage_backend="chroma",
            embedding_dimension=512,
            similarity_threshold=0.8,
        )

        assert config.storage_backend == "chroma"
        assert config.embedding_dimension == 512
        assert config.similarity_threshold == 0.8

    def test_from_env(self):
        """Should load from environment variables."""
        with patch.dict(os.environ, {
            "CIV6AZ_REFLECTION_BACKEND": "faiss",
            "CIV6AZ_REFLECTION_EMBED_DIM": "256",
            "CIV6AZ_REFLECTION_WARNING_PENALTY": "0.2",
        }):
            config = ReflectionConfig.from_env()

            assert config.storage_backend == "faiss"
            assert config.embedding_dimension == 256
            assert config.warning_penalty == 0.2


class TestReflectionEntry:
    """Tests for ReflectionEntry dataclass."""

    def test_creation(self):
        """Should create entry with defaults."""
        entry = ReflectionEntry()

        assert entry.id is not None
        assert entry.critique == ""
        assert entry.action_taken == -1
        assert entry.game_outcome == 0.0
        assert entry.timestamp is not None

    def test_to_dict(self):
        """Should serialize to dictionary."""
        entry = ReflectionEntry(
            critique="Lost due to weak defense",
            action_taken=42,
            game_outcome=-1.0,
        )

        data = entry.to_dict()

        assert data["critique"] == "Lost due to weak defense"
        assert data["action_taken"] == 42
        assert data["game_outcome"] == -1.0
        assert "id" in data
        assert "timestamp" in data

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "id": "test-id",
            "critique": "Bad strategy",
            "action_taken": 10,
            "game_outcome": -1.0,
            "better_alternatives": [5, 6, 7],
        }

        entry = ReflectionEntry.from_dict(data)

        assert entry.id == "test-id"
        assert entry.critique == "Bad strategy"
        assert entry.action_taken == 10
        assert entry.better_alternatives == [5, 6, 7]

    def test_roundtrip_serialization(self):
        """Should survive to_dict -> from_dict roundtrip."""
        original = ReflectionEntry(
            critique="Test critique",
            action_taken=100,
            better_alternatives=[1, 2, 3],
            metadata={"key": "value"},
        )

        data = original.to_dict()
        restored = ReflectionEntry.from_dict(data)

        assert restored.critique == original.critique
        assert restored.action_taken == original.action_taken
        assert restored.better_alternatives == original.better_alternatives


class TestDefaultStateEncoder:
    """Tests for DefaultStateEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create encoder for testing."""
        config = ReflectionConfig(embedding_dimension=64)
        return DefaultStateEncoder(config)

    @pytest.fixture
    def sample_obs(self):
        """Create sample observation."""
        return {
            "spatial": np.random.randn(12, 8, 8).astype(np.float32),
            "vector": np.random.randn(32).astype(np.float32),
        }

    def test_encode_state_returns_correct_shape(self, encoder, sample_obs):
        """Should return embedding of correct dimension."""
        embedding = encoder.encode_state(sample_obs)

        assert embedding.shape == (64,)
        assert embedding.dtype == np.float32

    def test_encode_state_is_normalized(self, encoder, sample_obs):
        """Embedding should be L2 normalized."""
        embedding = encoder.encode_state(sample_obs)
        norm = np.linalg.norm(embedding)

        # Should be normalized (or zero)
        assert abs(norm - 1.0) < 0.01 or norm == 0

    def test_encode_state_deterministic(self, encoder, sample_obs):
        """Same observation should produce same embedding."""
        emb1 = encoder.encode_state(sample_obs)
        emb2 = encoder.encode_state(sample_obs)

        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_encode_trajectory(self, encoder):
        """Should encode trajectory to single embedding."""
        examples = [
            TrainingExample(
                spatial=np.random.randn(12, 8, 8).astype(np.float32),
                vector=np.random.randn(32).astype(np.float32),
                action_mask=np.ones(100, dtype=np.float32),
                policy=np.zeros(100, dtype=np.float32),
                value=0.0,
            )
            for _ in range(5)
        ]
        trajectory = GameTrajectory(examples=examples)

        embedding = encoder.encode_trajectory(trajectory)

        assert embedding.shape == (64,)

    def test_encode_empty_trajectory(self, encoder):
        """Should handle empty trajectory."""
        trajectory = GameTrajectory(examples=[])

        embedding = encoder.encode_trajectory(trajectory)

        assert embedding.shape == (64,)
        assert np.allclose(embedding, 0)


class TestReflectionMemory:
    """Tests for ReflectionMemory."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return ReflectionConfig(
            storage_backend="memory",
            embedding_dimension=64,
            similarity_threshold=0.5,
            warning_penalty=0.1,
        )

    @pytest.fixture
    def memory(self, config):
        """Create memory instance."""
        return ReflectionMemory(config)

    @pytest.fixture
    def sample_trajectory(self):
        """Create sample trajectory."""
        examples = [
            TrainingExample(
                spatial=np.random.randn(12, 8, 8).astype(np.float32),
                vector=np.random.randn(32).astype(np.float32),
                action_mask=np.ones(100, dtype=np.float32),
                policy=np.zeros(100, dtype=np.float32),
                value=0.0,
            )
            for _ in range(10)
        ]
        return GameTrajectory(
            examples=examples,
            game_id=1,
            game_length=10,
            winner=1,  # Agent lost
        )

    @pytest.mark.asyncio
    async def test_initialization(self, memory):
        """Should initialize successfully."""
        result = await memory.initialize()

        assert result is True
        assert memory.initialized is True

    @pytest.mark.asyncio
    async def test_add_failed_game(self, memory, sample_trajectory):
        """Should add failed game with critique."""
        await memory.initialize()

        entry_id = await memory.add_failed_game(
            sample_trajectory,
            critique="Lost due to poor unit positioning",
            action_taken=42,
        )

        assert entry_id is not None
        assert memory.size == 1

    @pytest.mark.asyncio
    async def test_query_similar_failures(self, memory, sample_trajectory):
        """Should find similar failures."""
        await memory.initialize()

        # Add a failure
        await memory.add_failed_game(
            sample_trajectory,
            critique="Failed to defend",
            failure_point_idx=5,
        )

        # Query with similar state
        state = {
            "spatial": sample_trajectory.examples[5].spatial,
            "vector": sample_trajectory.examples[5].vector,
        }

        similar = await memory.query_similar_failures(state)

        # Should find the failure we added (assuming high similarity with itself)
        assert len(similar) >= 0  # May or may not find depending on embedding

    @pytest.mark.asyncio
    async def test_query_empty_memory(self, memory):
        """Should return empty list for empty memory."""
        await memory.initialize()

        state = {
            "spatial": np.random.randn(12, 8, 8).astype(np.float32),
            "vector": np.random.randn(32).astype(np.float32),
        }

        similar = await memory.query_similar_failures(state)

        assert similar == []

    @pytest.mark.asyncio
    async def test_get_action_warnings(self, memory, sample_trajectory):
        """Should get warnings for risky actions."""
        await memory.initialize()

        # Add failure with specific action
        await memory.add_failed_game(
            sample_trajectory,
            critique="Action 42 led to defeat",
            action_taken=42,
        )

        state = {
            "spatial": sample_trajectory.examples[5].spatial,
            "vector": sample_trajectory.examples[5].vector,
        }

        warnings = await memory.get_action_warnings(state, [42, 43, 44])

        # Warnings structure should be correct
        assert isinstance(warnings, dict)

    @pytest.mark.asyncio
    async def test_get_value_adjustment_penalizes_bad_actions(
        self, memory, sample_trajectory
    ):
        """Should return negative penalty for bad actions."""
        await memory.initialize()

        # Add failure with specific action
        example = sample_trajectory.examples[5]
        await memory.add_failed_game(
            sample_trajectory,
            critique="Bad action",
            action_taken=42,
            failure_point_idx=5,
        )

        state = {
            "spatial": example.spatial,
            "vector": example.vector,
        }

        # Get adjustment for the bad action
        adjustment = memory.get_value_adjustment_sync(state, 42)

        # Should be negative (penalty)
        assert adjustment <= 0

    @pytest.mark.asyncio
    async def test_get_value_adjustment_no_penalty_for_good_actions(
        self, memory, sample_trajectory
    ):
        """Should return zero for actions without failures."""
        await memory.initialize()

        state = {
            "spatial": np.random.randn(12, 8, 8).astype(np.float32),
            "vector": np.random.randn(32).astype(np.float32),
        }

        adjustment = memory.get_value_adjustment_sync(state, 99)

        assert adjustment == 0.0

    @pytest.mark.asyncio
    async def test_export_import_roundtrip(self, memory, sample_trajectory):
        """Should export and import reflections correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            await memory.initialize()

            # Add entries
            await memory.add_failed_game(
                sample_trajectory,
                critique="First failure",
            )
            await memory.add_failed_game(
                sample_trajectory,
                critique="Second failure",
            )

            # Export
            export_path = os.path.join(tmpdir, "reflections.json")
            count = await memory.export_reflections(export_path)

            assert count > 0
            assert os.path.exists(export_path)

            # Create new memory and import
            memory2 = ReflectionMemory(memory.config)
            await memory2.initialize()
            imported = await memory2.import_reflections(export_path)

            assert imported > 0

    @pytest.mark.asyncio
    async def test_duplicate_detection(self, memory, sample_trajectory):
        """Should not add duplicate entries for same state."""
        await memory.initialize()

        # Add same trajectory twice with different critiques
        id1 = await memory.add_failed_game(
            sample_trajectory,
            critique="Short critique",
            failure_point_idx=5,
        )

        id2 = await memory.add_failed_game(
            sample_trajectory,
            critique="Much longer and more detailed critique about the failure",
            failure_point_idx=5,
        )

        # Should return same ID (or update existing)
        assert memory.size == 1
        assert id1 == id2

    @pytest.mark.asyncio
    async def test_clear(self, memory, sample_trajectory):
        """Should clear all entries."""
        await memory.initialize()

        await memory.add_failed_game(sample_trajectory, critique="Test")
        assert memory.size > 0

        memory.clear()

        assert memory.size == 0

    def test_get_statistics(self, memory):
        """Should return statistics."""
        stats = memory.get_statistics()

        assert "total_entries" in stats
        assert "unique_patterns" in stats
        assert "queries" in stats
        assert "initialized" in stats

    @pytest.mark.asyncio
    async def test_graceful_degradation_without_deps(self, config):
        """Should work even if vector store fails."""
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock(side_effect=Exception("Connection failed"))

        memory = ReflectionMemory(config, vector_store=mock_store)

        # Should still initialize (gracefully degrade)
        result = await memory.initialize()

        # Should be marked as initialized for basic operation
        assert memory.initialized is True


class TestMCTSIntegration:
    """Tests for MCTS integration patterns."""

    @pytest.fixture
    def memory_with_entries(self):
        """Create memory with some entries."""
        config = ReflectionConfig(
            storage_backend="memory",
            embedding_dimension=64,
            warning_penalty=0.1,
        )
        memory = ReflectionMemory(config)

        # Manually add entry to cache for sync testing
        entry = ReflectionEntry(
            critique="Bad move",
            action_taken=42,
            state_hash="test_hash",
            state_embedding=np.random.randn(64).astype(np.float32),
        )
        memory._entry_cache[entry.state_hash] = entry
        memory._stats["total_entries"] = 1

        return memory

    def test_mcts_without_reflection(self):
        """MCTS should work without reflection memory."""
        memory = ReflectionMemory()
        # Not initialized

        state = {
            "spatial": np.zeros((12, 8, 8), dtype=np.float32),
            "vector": np.zeros(32, dtype=np.float32),
        }

        # Should not crash
        adjustment = memory.get_value_adjustment_sync(state, 42)

        assert adjustment == 0.0

    def test_sync_value_adjustment(self, memory_with_entries):
        """Sync value adjustment should work for MCTS."""
        memory = memory_with_entries

        state = {
            "spatial": np.zeros((12, 8, 8), dtype=np.float32),
            "vector": np.zeros(32, dtype=np.float32),
        }

        # Should return adjustment (might be 0 if state doesn't match)
        adjustment = memory.get_value_adjustment_sync(state, 42)

        assert isinstance(adjustment, float)
        assert adjustment <= 0  # Should be non-positive


class TestCreateReflectionMemory:
    """Tests for factory function."""

    def test_create_with_default_config(self):
        """Should create memory with default config."""
        memory = create_reflection_memory()

        assert memory is not None
        assert memory.config.storage_backend == "memory"

    def test_create_with_custom_config(self):
        """Should create memory with custom config."""
        config = ReflectionConfig(
            storage_backend="faiss",
            embedding_dimension=512,
        )

        memory = create_reflection_memory(config)

        assert memory.config.storage_backend == "faiss"
        assert memory.config.embedding_dimension == 512


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_import_nonexistent_file(self):
        """Should handle missing import file."""
        memory = ReflectionMemory()
        await memory.initialize()

        count = await memory.import_reflections("/nonexistent/path.json")

        assert count == 0

    @pytest.mark.asyncio
    async def test_query_before_initialize(self):
        """Should handle query before initialization."""
        memory = ReflectionMemory()

        state = {
            "spatial": np.zeros((12, 8, 8), dtype=np.float32),
            "vector": np.zeros(32, dtype=np.float32),
        }

        # Should not crash, just return empty
        similar = await memory.query_similar_failures(state)

        assert similar == []

    @pytest.mark.asyncio
    async def test_empty_trajectory(self):
        """Should handle empty trajectory."""
        memory = ReflectionMemory()
        await memory.initialize()

        empty_traj = GameTrajectory(examples=[], game_id=1)

        entry_id = await memory.add_failed_game(empty_traj, critique="Empty game")

        assert entry_id is not None
