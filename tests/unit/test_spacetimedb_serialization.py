"""Unit tests for SpacetimeDB serialization utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alpha_zero.spacetimedb.exceptions import SpacetimeDBSerializationError
from alpha_zero.spacetimedb.serialization import (
    deserialize_checkpoint,
    deserialize_config,
    deserialize_metadata,
    deserialize_state_dict,
    deserialize_trajectory,
    estimate_blob_size_mb,
    serialize_checkpoint,
    serialize_config,
    serialize_metadata,
    serialize_state_dict,
    serialize_trajectory,
)


class TestStateDictSerialization:
    """Tests for state dict serialization round-trips."""

    @pytest.fixture
    def sample_state_dict(self):
        """Create a sample model state dict."""
        return {
            "layer1.weight": torch.randn(64, 32),
            "layer1.bias": torch.randn(64),
            "layer2.weight": torch.randn(10, 64),
            "layer2.bias": torch.randn(10),
        }

    def test_roundtrip_compressed(self, sample_state_dict):
        """Compressed round-trip preserves values."""
        data = serialize_state_dict(sample_state_dict, compress=True)
        result = deserialize_state_dict(data, compressed=True)

        assert set(result.keys()) == set(sample_state_dict.keys())
        for key in sample_state_dict:
            assert torch.allclose(result[key], sample_state_dict[key])

    def test_roundtrip_uncompressed(self, sample_state_dict):
        """Uncompressed round-trip preserves values."""
        data = serialize_state_dict(sample_state_dict, compress=False)
        result = deserialize_state_dict(data, compressed=False)

        for key in sample_state_dict:
            assert torch.allclose(result[key], sample_state_dict[key])

    def test_compressed_smaller(self, sample_state_dict):
        """Compressed data is smaller than uncompressed."""
        compressed = serialize_state_dict(sample_state_dict, compress=True)
        uncompressed = serialize_state_dict(sample_state_dict, compress=False)
        assert len(compressed) < len(uncompressed)

    def test_empty_state_dict(self):
        """Empty state dict round-trips correctly."""
        data = serialize_state_dict({}, compress=False)
        result = deserialize_state_dict(data, compressed=False)
        assert result == {}

    @pytest.mark.parametrize(
        "shape",
        [
            (1,),
            (10, 10),
            (3, 16, 16),
            (64, 13, 8, 8),
        ],
    )
    def test_various_tensor_shapes(self, shape):
        """Various tensor shapes serialize correctly."""
        state_dict = {"tensor": torch.randn(*shape)}
        data = serialize_state_dict(state_dict, compress=True)
        result = deserialize_state_dict(data, compressed=True)
        assert torch.allclose(result["tensor"], state_dict["tensor"])

    def test_map_location(self, sample_state_dict):
        """map_location parameter is respected."""
        data = serialize_state_dict(sample_state_dict, compress=False)
        result = deserialize_state_dict(data, compressed=False, map_location="cpu")
        assert result["layer1.weight"].device == torch.device("cpu")


class TestTrajectorySerialization:
    """Tests for trajectory serialization."""

    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample GameTrajectory."""
        from alpha_zero.distributed.shared_storage import (
            GameTrajectory,
            TrainingExample,
        )

        examples = [
            TrainingExample(
                spatial=np.random.randn(13, 16, 16).astype(np.float32),
                vector=np.random.randn(64).astype(np.float32),
                action_mask=np.ones(356, dtype=np.float32),
                policy=np.random.dirichlet(np.ones(356)).astype(np.float32),
                value=0.5,
            )
            for _ in range(5)
        ]
        return GameTrajectory(
            examples=examples,
            game_id=42,
            actor_id=1,
            game_length=5,
            winner=0,
        )

    def test_roundtrip_uncompressed(self, sample_trajectory):
        """Uncompressed trajectory round-trip preserves data."""
        data = serialize_trajectory(sample_trajectory, compress=False)
        result = deserialize_trajectory(data, compressed=False)

        assert len(result.examples) == len(sample_trajectory.examples)
        assert result.game_id == sample_trajectory.game_id
        assert result.actor_id == sample_trajectory.actor_id

    def test_roundtrip_compressed(self, sample_trajectory):
        """Compressed trajectory round-trip preserves data."""
        data = serialize_trajectory(sample_trajectory, compress=True)
        result = deserialize_trajectory(data, compressed=True)

        assert len(result.examples) == len(sample_trajectory.examples)
        assert result.game_id == sample_trajectory.game_id

    def test_example_values_preserved(self, sample_trajectory):
        """Individual training example values are preserved."""
        data = serialize_trajectory(sample_trajectory, compress=False)
        result = deserialize_trajectory(data, compressed=False)

        for orig, restored in zip(
            sample_trajectory.examples, result.examples, strict=True
        ):
            np.testing.assert_array_almost_equal(orig.spatial, restored.spatial)
            np.testing.assert_array_almost_equal(orig.vector, restored.vector)
            np.testing.assert_array_almost_equal(orig.policy, restored.policy)
            assert abs(orig.value - restored.value) < 1e-6


class TestCheckpointSerialization:
    """Tests for checkpoint serialization."""

    @pytest.fixture
    def sample_checkpoint(self):
        """Create a sample checkpoint."""
        return {
            "model_state_dict": {
                "layer.weight": torch.randn(32, 16),
                "layer.bias": torch.randn(32),
            },
            "optimizer_state_dict": {"step": 100},
            "iteration": 50,
            "training_step": 5000,
        }

    def test_roundtrip_compressed(self, sample_checkpoint):
        """Checkpoint round-trip with compression."""
        data = serialize_checkpoint(sample_checkpoint, compress=True)
        result = deserialize_checkpoint(data, compressed=True)

        assert result["iteration"] == 50
        assert result["training_step"] == 5000
        assert torch.allclose(
            result["model_state_dict"]["layer.weight"],
            sample_checkpoint["model_state_dict"]["layer.weight"],
        )

    def test_roundtrip_uncompressed(self, sample_checkpoint):
        """Checkpoint round-trip without compression."""
        data = serialize_checkpoint(sample_checkpoint, compress=False)
        result = deserialize_checkpoint(data, compressed=False)
        assert result["iteration"] == 50


class TestMetadataSerialization:
    """Tests for metadata JSON serialization."""

    def test_dict_roundtrip(self):
        """Dictionary metadata round-trips."""
        meta = {"version": 5, "win_rate": 0.62, "elo": 1500.0}
        json_str = serialize_metadata(meta)
        result = deserialize_metadata(json_str)
        assert result == meta

    def test_dataclass_roundtrip(self):
        """Dataclass metadata serializes via asdict."""
        from alpha_zero.storage.models import ModelMetadata

        meta = ModelMetadata(iteration=10, version=5, elo_rating=1500.0)
        json_str = serialize_metadata(meta)
        result = deserialize_metadata(json_str)
        assert result["iteration"] == 10
        assert result["version"] == 5
        assert result["elo_rating"] == 1500.0

    def test_invalid_json_raises(self):
        """Invalid JSON raises deserialization error."""
        with pytest.raises(SpacetimeDBSerializationError):
            deserialize_metadata("not valid json {{{")


class TestConfigSerialization:
    """Tests for config JSON serialization."""

    def test_roundtrip(self):
        """Config round-trips through JSON."""
        config = {
            "training": {"batch_size": 256, "learning_rate": 0.001},
            "mcts": {"num_simulations": 800},
        }
        json_str = serialize_config(config)
        result = deserialize_config(json_str)
        assert result == config

    def test_empty_config(self):
        """Empty config serializes correctly."""
        json_str = serialize_config({})
        result = deserialize_config(json_str)
        assert result == {}


class TestBlobSizeEstimation:
    """Tests for blob size estimation."""

    def test_known_size(self):
        """Known-size data returns correct MB estimate."""
        data = b"\x00" * (1024 * 1024)  # 1 MB
        assert abs(estimate_blob_size_mb(data) - 1.0) < 0.001

    def test_empty_data(self):
        """Empty data returns 0."""
        assert estimate_blob_size_mb(b"") == 0.0


class TestSerializationErrors:
    """Tests for error handling."""

    def test_deserialize_corrupt_state_dict(self):
        """Corrupt data raises serialization error."""
        with pytest.raises(SpacetimeDBSerializationError):
            deserialize_state_dict(b"corrupt data", compressed=False)

    def test_deserialize_corrupt_compressed(self):
        """Corrupt compressed data raises serialization error."""
        with pytest.raises(SpacetimeDBSerializationError):
            deserialize_state_dict(b"corrupt compressed", compressed=True)

    def test_deserialize_corrupt_checkpoint(self):
        """Corrupt checkpoint data raises serialization error."""
        with pytest.raises(SpacetimeDBSerializationError):
            deserialize_checkpoint(b"corrupt", compressed=False)
