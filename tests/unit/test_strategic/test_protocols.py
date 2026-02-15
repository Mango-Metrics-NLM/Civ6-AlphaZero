"""Tests for Strategic Planning SDK protocol definitions and data classes.

Validates the core data structures (StateSpec, ActionSpec,
StrategicObservation, StrategicStepResult) that underpin all
SDK components.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.strategic.protocols import (
    ActionSpec,
    StateSpec,
    StrategicObservation,
    StrategicStepResult,
)


class TestStateSpec:
    """Test suite for StateSpec dataclass."""

    def test_defaults(self) -> None:
        """StateSpec has sensible defaults for a non-spatial environment."""
        spec = StateSpec()

        assert spec.spatial_shape is None
        assert spec.vector_dim == 64
        assert spec.action_space_size == 100
        assert spec.spatial_channel_names == []
        assert spec.vector_feature_names == []

    def test_has_spatial(self) -> None:
        """has_spatial returns True when spatial_shape has 3 dimensions."""
        spec = StateSpec(spatial_shape=(8, 16, 16))

        assert spec.has_spatial is True

    def test_no_spatial(self) -> None:
        """has_spatial returns False when spatial_shape is None or wrong rank."""
        spec_none = StateSpec(spatial_shape=None)
        assert spec_none.has_spatial is False

        spec_2d = StateSpec(spatial_shape=(16, 16))
        assert spec_2d.has_spatial is False

    def test_spatial_channels(self) -> None:
        """spatial_channels returns first dimension of spatial_shape."""
        spec_spatial = StateSpec(spatial_shape=(13, 16, 16))
        assert spec_spatial.spatial_channels == 13

        spec_no_spatial = StateSpec(spatial_shape=None)
        assert spec_no_spatial.spatial_channels == 0

    def test_spatial_size(self) -> None:
        """spatial_size returns (H, W) tuple from spatial_shape."""
        spec_spatial = StateSpec(spatial_shape=(8, 32, 24))
        assert spec_spatial.spatial_size == (32, 24)

        spec_no_spatial = StateSpec(spatial_shape=None)
        assert spec_no_spatial.spatial_size == (0, 0)

    def test_to_from_dict(self) -> None:
        """StateSpec round-trips through dictionary serialization."""
        original = StateSpec(
            spatial_shape=(13, 16, 16),
            vector_dim=128,
            action_space_size=500,
            spatial_channel_names=["terrain", "units"],
            vector_feature_names=["gold", "science"],
        )

        data = original.to_dict()
        restored = StateSpec.from_dict(data)

        assert restored.spatial_shape == original.spatial_shape
        assert restored.vector_dim == original.vector_dim
        assert restored.action_space_size == original.action_space_size
        assert restored.spatial_channel_names == original.spatial_channel_names
        assert restored.vector_feature_names == original.vector_feature_names

    def test_to_from_dict_no_spatial(self) -> None:
        """StateSpec round-trips when spatial_shape is None."""
        original = StateSpec(spatial_shape=None, vector_dim=32)

        data = original.to_dict()
        assert data["spatial_shape"] is None

        restored = StateSpec.from_dict(data)
        assert restored.spatial_shape is None
        assert restored.vector_dim == 32


class TestActionSpec:
    """Test suite for ActionSpec dataclass."""

    def test_defaults(self) -> None:
        """ActionSpec has sensible defaults."""
        spec = ActionSpec()

        assert spec.total_size == 100
        assert spec.hierarchy_levels == []
        assert spec.level_sizes == []
        assert spec.action_names == {}

    def test_is_hierarchical(self) -> None:
        """is_hierarchical returns True when more than one hierarchy level exists."""
        flat_spec = ActionSpec(hierarchy_levels=["move"])
        assert flat_spec.is_hierarchical is False

        hier_spec = ActionSpec(hierarchy_levels=["type", "target"])
        assert hier_spec.is_hierarchical is True

        empty_spec = ActionSpec(hierarchy_levels=[])
        assert empty_spec.is_hierarchical is False

    def test_get_action_name(self) -> None:
        """get_action_name returns named action or falls back to default."""
        spec = ActionSpec(
            total_size=5,
            action_names={0: "move_north", 1: "move_south", 3: "attack"},
        )

        assert spec.get_action_name(0) == "move_north"
        assert spec.get_action_name(1) == "move_south"
        assert spec.get_action_name(3) == "attack"
        # Unnamed action falls back to default format
        assert spec.get_action_name(2) == "action_2"
        assert spec.get_action_name(4) == "action_4"

    def test_to_from_dict(self) -> None:
        """ActionSpec round-trips through dictionary serialization."""
        original = ActionSpec(
            total_size=25,
            hierarchy_levels=["carbon_price", "tech_investment"],
            level_sizes=[5, 5],
            action_names={0: "low_low", 24: "high_high"},
        )

        data = original.to_dict()
        restored = ActionSpec.from_dict(data)

        assert restored.total_size == original.total_size
        assert restored.hierarchy_levels == original.hierarchy_levels
        assert restored.level_sizes == original.level_sizes
        # Keys are converted to int during from_dict
        assert restored.action_names[0] == "low_low"
        assert restored.action_names[24] == "high_high"

    def test_to_dict_converts_keys_to_str(self) -> None:
        """to_dict converts action_names keys to strings for JSON compatibility."""
        spec = ActionSpec(action_names={0: "a", 5: "b"})
        data = spec.to_dict()

        assert "0" in data["action_names"]
        assert "5" in data["action_names"]


class TestStrategicObservation:
    """Test suite for StrategicObservation dataclass."""

    def test_creation(self) -> None:
        """StrategicObservation can be created with required fields."""
        vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        action_mask = np.array([1.0, 0.0, 1.0], dtype=np.float32)

        obs = StrategicObservation(
            spatial=None,
            vector=vector,
            action_mask=action_mask,
            agent_id=1,
            step_number=5,
            done=False,
            info={"score": 42},
        )

        assert obs.spatial is None
        np.testing.assert_array_equal(obs.vector, vector)
        np.testing.assert_array_equal(obs.action_mask, action_mask)
        assert obs.agent_id == 1
        assert obs.step_number == 5
        assert obs.done is False
        assert obs.info == {"score": 42}

    def test_creation_with_spatial(self) -> None:
        """StrategicObservation works with spatial data."""
        spatial = np.random.randn(8, 16, 16).astype(np.float32)
        vector = np.zeros(10, dtype=np.float32)
        action_mask = np.ones(5, dtype=np.float32)

        obs = StrategicObservation(
            spatial=spatial,
            vector=vector,
            action_mask=action_mask,
        )

        assert obs.spatial is not None
        assert obs.spatial.shape == (8, 16, 16)
        assert obs.agent_id == 0  # default
        assert obs.step_number == 0  # default
        assert obs.done is False  # default
        assert obs.info == {}  # default

    def test_to_dict(self) -> None:
        """StrategicObservation serializes to dictionary correctly."""
        spatial = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        vector = np.array([0.5, 0.6], dtype=np.float32)
        action_mask = np.array([1.0, 0.0], dtype=np.float32)

        obs = StrategicObservation(
            spatial=spatial,
            vector=vector,
            action_mask=action_mask,
            agent_id=2,
            step_number=10,
            done=True,
            info={"key": "value"},
        )

        data = obs.to_dict()

        assert data["spatial"] is not None
        assert data["agent_id"] == 2
        assert data["step_number"] == 10
        assert data["done"] is True
        assert data["info"] == {"key": "value"}
        assert len(data["vector"]) == 2
        assert len(data["action_mask"]) == 2

    def test_to_dict_no_spatial(self) -> None:
        """to_dict sets spatial to None when no spatial data exists."""
        obs = StrategicObservation(
            spatial=None,
            vector=np.zeros(4, dtype=np.float32),
            action_mask=np.ones(3, dtype=np.float32),
        )

        data = obs.to_dict()
        assert data["spatial"] is None


class TestStrategicStepResult:
    """Test suite for StrategicStepResult dataclass."""

    def test_creation(self) -> None:
        """StrategicStepResult bundles observation, rewards, and flags."""
        obs = StrategicObservation(
            spatial=None,
            vector=np.zeros(5, dtype=np.float32),
            action_mask=np.ones(3, dtype=np.float32),
            done=True,
        )
        rewards = np.array([0.5, -0.3, 0.1], dtype=np.float64)

        result = StrategicStepResult(
            observation=obs,
            rewards=rewards,
            done=True,
            truncated=False,
            info={"step_detail": "some_info"},
        )

        assert result.observation is obs
        np.testing.assert_array_almost_equal(result.rewards, [0.5, -0.3, 0.1])
        assert result.done is True
        assert result.truncated is False
        assert result.info == {"step_detail": "some_info"}

    def test_creation_defaults(self) -> None:
        """StrategicStepResult defaults done=False, truncated=False, info={}."""
        obs = StrategicObservation(
            spatial=None,
            vector=np.zeros(3, dtype=np.float32),
            action_mask=np.ones(2, dtype=np.float32),
        )
        rewards = np.array([0.0, 0.0], dtype=np.float64)

        result = StrategicStepResult(
            observation=obs,
            rewards=rewards,
        )

        assert result.done is False
        assert result.truncated is False
        assert result.info == {}
