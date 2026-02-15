"""Comprehensive tests for src.core.types module.

Covers type aliases, enums, constants, and all dataclass containers.
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar

import numpy as np
import pytest

from src.core import types as types_module
from src.core.types import (
    # Constants
    DEFAULT_C_PUCT,
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_DIRICHLET_EPSILON,
    MCTS_INF,
    MIN_VISIT_COUNT,
    SDK_ENV_PREFIX,
    # Type aliases
    ActionMask,
    # Enums
    ActionSpaceType,
    BackboneType,
    DeviceType,
    # Data containers
    EvalResult,
    GameOutcome,
    GameTrajectory,
    HierarchicalAction,
    HierarchicalActionSpec,
    ObservationSpec,
    PolicyArray,
    ReplayBufferType,
    SearchResult,
    SelectionPolicyType,
    SpatialTensor,
    TrainingExample,
    ValueEstimate,
    VectorFeatures,
)

# =========================================================================
# Section 1: __all__ exports completeness
# =========================================================================


class TestAllExports:
    """Verify __all__ contains all public names and nothing extra."""

    EXPECTED_NAMES: ClassVar[set[str]] = {
        # Type aliases
        "PolicyArray",
        "ValueEstimate",
        "ActionMask",
        "SpatialTensor",
        "VectorFeatures",
        # Enums
        "DeviceType",
        "SelectionPolicyType",
        "BackboneType",
        "ActionSpaceType",
        "ReplayBufferType",
        "GameOutcome",
        # Constants
        "MCTS_INF",
        "DEFAULT_C_PUCT",
        "MIN_VISIT_COUNT",
        "DEFAULT_DIRICHLET_ALPHA",
        "DEFAULT_DIRICHLET_EPSILON",
        "SDK_ENV_PREFIX",
        # Data containers
        "ObservationSpec",
        "HierarchicalActionSpec",
        "HierarchicalAction",
        "TrainingExample",
        "GameTrajectory",
        "SearchResult",
        "EvalResult",
    }

    def test_all_contains_expected_names(self) -> None:
        actual = set(types_module.__all__)
        missing = self.EXPECTED_NAMES - actual
        assert not missing, f"Missing from __all__: {missing}"

    def test_all_has_no_extra_names(self) -> None:
        actual = set(types_module.__all__)
        extra = actual - self.EXPECTED_NAMES
        assert not extra, f"Unexpected names in __all__: {extra}"

    def test_all_names_are_importable(self) -> None:
        for name in types_module.__all__:
            assert hasattr(types_module, name), (
                f"{name} is listed in __all__ but not accessible on the module"
            )


# =========================================================================
# Section 2: Type aliases
# =========================================================================


class TestTypeAliases:
    """Verify type aliases reference the correct numpy dtypes."""

    def test_policy_array_is_ndarray_alias(self) -> None:
        _arr = np.zeros(5, dtype=np.float32)
        # PolicyArray is a type alias; we just verify it exists and the
        # annotation is usable at runtime.
        assert PolicyArray is not None

    def test_value_estimate_is_float(self) -> None:
        assert ValueEstimate is float

    def test_action_mask_is_ndarray_alias(self) -> None:
        assert ActionMask is not None

    def test_spatial_tensor_is_ndarray_alias(self) -> None:
        assert SpatialTensor is not None

    def test_vector_features_is_ndarray_alias(self) -> None:
        assert VectorFeatures is not None


# =========================================================================
# Section 3: Enums
# =========================================================================


class TestDeviceType:
    """DeviceType enum tests."""

    @pytest.mark.parametrize("value", ["cpu", "cuda", "mps", "auto"])
    def test_members_accessible_by_value(self, value: str) -> None:
        member = DeviceType(value)
        assert member.value == value

    def test_is_str_subclass(self) -> None:
        assert isinstance(DeviceType.cpu, str)

    def test_all_members(self) -> None:
        assert set(DeviceType) == {
            DeviceType.cpu,
            DeviceType.cuda,
            DeviceType.mps,
            DeviceType.auto,
        }

    def test_member_count(self) -> None:
        assert len(DeviceType) == 4

    def test_string_equality(self) -> None:
        assert DeviceType.cpu == "cpu"
        assert DeviceType.cuda == "cuda"

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            DeviceType("tpu")


class TestSelectionPolicyType:
    """SelectionPolicyType enum tests."""

    @pytest.mark.parametrize("value", ["puct", "gumbel", "alphazero"])
    def test_members_accessible_by_value(self, value: str) -> None:
        member = SelectionPolicyType(value)
        assert member.value == value

    def test_is_str_subclass(self) -> None:
        assert isinstance(SelectionPolicyType.puct, str)

    def test_member_count(self) -> None:
        assert len(SelectionPolicyType) == 3

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            SelectionPolicyType("random")


class TestBackboneType:
    """BackboneType enum tests."""

    @pytest.mark.parametrize("value", ["resnet", "transformer", "gnn"])
    def test_members_accessible_by_value(self, value: str) -> None:
        member = BackboneType(value)
        assert member.value == value

    def test_is_str_subclass(self) -> None:
        assert isinstance(BackboneType.resnet, str)

    def test_member_count(self) -> None:
        assert len(BackboneType) == 3


class TestActionSpaceType:
    """ActionSpaceType enum tests."""

    @pytest.mark.parametrize("value", ["flat", "hierarchical"])
    def test_members_accessible_by_value(self, value: str) -> None:
        member = ActionSpaceType(value)
        assert member.value == value

    def test_is_str_subclass(self) -> None:
        assert isinstance(ActionSpaceType.flat, str)

    def test_member_count(self) -> None:
        assert len(ActionSpaceType) == 2


class TestReplayBufferType:
    """ReplayBufferType enum tests."""

    @pytest.mark.parametrize("value", ["uniform", "prioritized"])
    def test_members_accessible_by_value(self, value: str) -> None:
        member = ReplayBufferType(value)
        assert member.value == value

    def test_is_str_subclass(self) -> None:
        assert isinstance(ReplayBufferType.uniform, str)

    def test_member_count(self) -> None:
        assert len(ReplayBufferType) == 2


class TestGameOutcome:
    """GameOutcome enum tests."""

    @pytest.mark.parametrize("value", ["win", "loss", "draw", "timeout"])
    def test_members_accessible_by_value(self, value: str) -> None:
        member = GameOutcome(value)
        assert member.value == value

    def test_is_str_subclass(self) -> None:
        assert isinstance(GameOutcome.win, str)

    def test_member_count(self) -> None:
        assert len(GameOutcome) == 4

    def test_string_equality(self) -> None:
        assert GameOutcome.win == "win"
        assert GameOutcome.loss == "loss"
        assert GameOutcome.draw == "draw"
        assert GameOutcome.timeout == "timeout"

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            GameOutcome("surrender")


# =========================================================================
# Section 4: Constants
# =========================================================================


class TestConstants:
    """Verify constants have correct types and values."""

    def test_mcts_inf_is_float(self) -> None:
        assert isinstance(MCTS_INF, float)

    def test_mcts_inf_is_positive_infinity(self) -> None:
        assert math.isinf(MCTS_INF)
        assert MCTS_INF > 0

    def test_default_c_puct_type_and_value(self) -> None:
        assert isinstance(DEFAULT_C_PUCT, float)
        assert DEFAULT_C_PUCT == 1.5

    def test_min_visit_count_type_and_value(self) -> None:
        assert isinstance(MIN_VISIT_COUNT, int)
        assert MIN_VISIT_COUNT == 1

    def test_default_dirichlet_alpha_type_and_value(self) -> None:
        assert isinstance(DEFAULT_DIRICHLET_ALPHA, float)
        assert DEFAULT_DIRICHLET_ALPHA == pytest.approx(0.3)

    def test_default_dirichlet_epsilon_type_and_value(self) -> None:
        assert isinstance(DEFAULT_DIRICHLET_EPSILON, float)
        assert DEFAULT_DIRICHLET_EPSILON == pytest.approx(0.25)

    def test_sdk_env_prefix_type_and_value(self) -> None:
        assert isinstance(SDK_ENV_PREFIX, str)
        assert SDK_ENV_PREFIX == "SPSDK_"


# =========================================================================
# Section 5: ObservationSpec
# =========================================================================


class TestObservationSpec:
    """Tests for ObservationSpec frozen dataclass."""

    def test_creation_basic(self) -> None:
        spec = ObservationSpec(spatial_shape=(13, 16, 16), vector_dim=64)
        assert spec.spatial_shape == (13, 16, 16)
        assert spec.vector_dim == 64
        assert spec.temporal_depth == 1

    def test_creation_with_temporal_depth(self) -> None:
        spec = ObservationSpec(spatial_shape=(3, 8, 8), vector_dim=32, temporal_depth=4)
        assert spec.temporal_depth == 4

    def test_frozen(self) -> None:
        spec = ObservationSpec(spatial_shape=(3, 8, 8), vector_dim=10)
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.vector_dim = 20  # type: ignore[misc]

    def test_has_slots(self) -> None:
        spec = ObservationSpec(spatial_shape=(3, 8, 8), vector_dim=10)
        assert hasattr(spec, "__slots__")

    # full_shape tests -------------------------------------------------

    def test_full_shape_3d_no_stacking(self) -> None:
        spec = ObservationSpec(spatial_shape=(13, 16, 16), vector_dim=64, temporal_depth=1)
        assert spec.full_shape == (13, 16, 16)

    def test_full_shape_3d_with_stacking(self) -> None:
        spec = ObservationSpec(spatial_shape=(3, 8, 8), vector_dim=32, temporal_depth=4)
        # channels * temporal_depth = 3 * 4 = 12
        assert spec.full_shape == (12, 8, 8)

    @pytest.mark.parametrize(
        "spatial_shape, temporal_depth, expected",
        [
            ((1, 4, 4), 1, (1, 4, 4)),
            ((1, 4, 4), 3, (3, 4, 4)),
            ((6, 20, 20), 2, (12, 20, 20)),
            ((10, 1, 1), 5, (50, 1, 1)),
        ],
        ids=["1ch-1t", "1ch-3t", "6ch-2t", "10ch-5t"],
    )
    def test_full_shape_parametrized(
        self,
        spatial_shape: tuple[int, ...],
        temporal_depth: int,
        expected: tuple[int, ...],
    ) -> None:
        spec = ObservationSpec(
            spatial_shape=spatial_shape, vector_dim=10, temporal_depth=temporal_depth
        )
        assert spec.full_shape == expected

    def test_full_shape_empty_spatial(self) -> None:
        """When spatial_shape is empty, full_shape falls back to (vector_dim * T,)."""
        spec = ObservationSpec(spatial_shape=(), vector_dim=64, temporal_depth=1)
        assert spec.full_shape == (64,)

    def test_full_shape_empty_spatial_with_temporal(self) -> None:
        spec = ObservationSpec(spatial_shape=(), vector_dim=32, temporal_depth=3)
        assert spec.full_shape == (96,)

    def test_full_shape_2d_spatial(self) -> None:
        """A 2D spatial_shape like (channels, width) -- only first element is scaled."""
        spec = ObservationSpec(spatial_shape=(4, 10), vector_dim=5, temporal_depth=2)
        assert spec.full_shape == (8, 10)

    def test_equality(self) -> None:
        a = ObservationSpec(spatial_shape=(3, 8, 8), vector_dim=10)
        b = ObservationSpec(spatial_shape=(3, 8, 8), vector_dim=10)
        assert a == b

    def test_inequality(self) -> None:
        a = ObservationSpec(spatial_shape=(3, 8, 8), vector_dim=10)
        b = ObservationSpec(spatial_shape=(3, 8, 8), vector_dim=20)
        assert a != b


# =========================================================================
# Section 6: HierarchicalActionSpec
# =========================================================================


class TestHierarchicalActionSpec:
    """Tests for HierarchicalActionSpec frozen dataclass with validation."""

    def test_valid_creation(self) -> None:
        spec = HierarchicalActionSpec(num_levels=3, level_sizes=(5, 10, 4))
        assert spec.num_levels == 3
        assert spec.level_sizes == (5, 10, 4)
        assert spec.category_names == ()
        assert spec.total_actions == 0

    def test_valid_creation_with_category_names(self) -> None:
        spec = HierarchicalActionSpec(
            num_levels=2,
            level_sizes=(3, 7),
            category_names=("type", "target"),
        )
        assert spec.category_names == ("type", "target")

    def test_valid_creation_with_total_actions(self) -> None:
        spec = HierarchicalActionSpec(
            num_levels=2,
            level_sizes=(5, 8),
            total_actions=40,
        )
        assert spec.total_actions == 40

    def test_frozen(self) -> None:
        spec = HierarchicalActionSpec(num_levels=1, level_sizes=(10,))
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.num_levels = 2  # type: ignore[misc]

    def test_has_slots(self) -> None:
        spec = HierarchicalActionSpec(num_levels=1, level_sizes=(10,))
        assert hasattr(spec, "__slots__")

    # Validation errors ------------------------------------------------

    def test_mismatched_level_sizes_raises(self) -> None:
        with pytest.raises(ValueError, match=r"level_sizes length.*must equal num_levels"):
            HierarchicalActionSpec(num_levels=3, level_sizes=(5, 10))

    def test_mismatched_level_sizes_too_many(self) -> None:
        with pytest.raises(ValueError, match=r"level_sizes length.*must equal num_levels"):
            HierarchicalActionSpec(num_levels=1, level_sizes=(5, 10))

    def test_mismatched_category_names_raises(self) -> None:
        with pytest.raises(ValueError, match=r"category_names length.*must equal num_levels"):
            HierarchicalActionSpec(
                num_levels=2,
                level_sizes=(5, 10),
                category_names=("only_one",),
            )

    def test_mismatched_category_names_too_many(self) -> None:
        with pytest.raises(ValueError, match=r"category_names length.*must equal num_levels"):
            HierarchicalActionSpec(
                num_levels=2,
                level_sizes=(5, 10),
                category_names=("a", "b", "c"),
            )

    def test_empty_category_names_does_not_raise(self) -> None:
        """An empty tuple for category_names is always valid (it's the default)."""
        spec = HierarchicalActionSpec(num_levels=2, level_sizes=(3, 4), category_names=())
        assert spec.category_names == ()

    def test_single_level(self) -> None:
        spec = HierarchicalActionSpec(num_levels=1, level_sizes=(100,))
        assert spec.num_levels == 1
        assert spec.level_sizes == (100,)


# =========================================================================
# Section 7: HierarchicalAction
# =========================================================================


class TestHierarchicalAction:
    """Tests for HierarchicalAction frozen dataclass."""

    def test_creation(self) -> None:
        action = HierarchicalAction(levels=(0, 3, 7))
        assert action.levels == (0, 3, 7)

    def test_frozen(self) -> None:
        action = HierarchicalAction(levels=(1, 2))
        with pytest.raises(dataclasses.FrozenInstanceError):
            action.levels = (3, 4)  # type: ignore[misc]

    def test_equality(self) -> None:
        a = HierarchicalAction(levels=(1, 2, 3))
        b = HierarchicalAction(levels=(1, 2, 3))
        assert a == b

    def test_inequality(self) -> None:
        a = HierarchicalAction(levels=(1, 2, 3))
        b = HierarchicalAction(levels=(1, 2, 4))
        assert a != b


# =========================================================================
# Section 8: TrainingExample
# =========================================================================


class TestTrainingExample:
    """Tests for mutable TrainingExample with serialization."""

    @pytest.fixture()
    def example(self) -> TrainingExample:
        return TrainingExample(
            spatial=np.random.randn(3, 8, 8).astype(np.float32),
            vector=np.random.randn(16).astype(np.float32),
            action_mask=np.array([True, False, True, True], dtype=np.bool_),
            policy=np.array([0.5, 0.0, 0.3, 0.2], dtype=np.float32),
            value=0.75,
        )

    def test_creation(self, example: TrainingExample) -> None:
        assert example.spatial.shape == (3, 8, 8)
        assert example.vector.shape == (16,)
        assert example.action_mask.shape == (4,)
        assert example.policy.shape == (4,)
        assert example.value == 0.75

    def test_mutable(self, example: TrainingExample) -> None:
        """TrainingExample is NOT frozen -- fields can be reassigned."""
        example.value = -0.5
        assert example.value == -0.5

    def test_has_slots(self, example: TrainingExample) -> None:
        assert hasattr(example, "__slots__")

    # Serialization round-trip -----------------------------------------

    def test_to_dict_keys(self, example: TrainingExample) -> None:
        d = example.to_dict()
        assert set(d.keys()) == {"spatial", "vector", "action_mask", "policy", "value"}

    def test_to_dict_value_is_plain_float(self, example: TrainingExample) -> None:
        d = example.to_dict()
        assert isinstance(d["value"], float)

    def test_to_dict_arrays_are_lists(self, example: TrainingExample) -> None:
        d = example.to_dict()
        assert isinstance(d["spatial"], list)
        assert isinstance(d["vector"], list)
        assert isinstance(d["action_mask"], list)
        assert isinstance(d["policy"], list)

    def test_round_trip_preserves_spatial(self, example: TrainingExample) -> None:
        reconstructed = TrainingExample.from_dict(example.to_dict())
        np.testing.assert_allclose(
            reconstructed.spatial, example.spatial, atol=1e-6
        )

    def test_round_trip_preserves_vector(self, example: TrainingExample) -> None:
        reconstructed = TrainingExample.from_dict(example.to_dict())
        np.testing.assert_allclose(
            reconstructed.vector, example.vector, atol=1e-6
        )

    def test_round_trip_preserves_action_mask(self, example: TrainingExample) -> None:
        reconstructed = TrainingExample.from_dict(example.to_dict())
        np.testing.assert_array_equal(reconstructed.action_mask, example.action_mask)

    def test_round_trip_preserves_policy(self, example: TrainingExample) -> None:
        reconstructed = TrainingExample.from_dict(example.to_dict())
        np.testing.assert_allclose(
            reconstructed.policy, example.policy, atol=1e-6
        )

    def test_round_trip_preserves_value(self, example: TrainingExample) -> None:
        reconstructed = TrainingExample.from_dict(example.to_dict())
        assert reconstructed.value == pytest.approx(example.value)

    def test_round_trip_preserves_dtypes(self, example: TrainingExample) -> None:
        reconstructed = TrainingExample.from_dict(example.to_dict())
        assert reconstructed.spatial.dtype == np.float32
        assert reconstructed.vector.dtype == np.float32
        assert reconstructed.action_mask.dtype == np.bool_
        assert reconstructed.policy.dtype == np.float32

    def test_round_trip_preserves_shapes(self, example: TrainingExample) -> None:
        reconstructed = TrainingExample.from_dict(example.to_dict())
        assert reconstructed.spatial.shape == example.spatial.shape
        assert reconstructed.vector.shape == example.vector.shape
        assert reconstructed.action_mask.shape == example.action_mask.shape
        assert reconstructed.policy.shape == example.policy.shape

    def test_from_dict_with_negative_value(self) -> None:
        data = {
            "spatial": [[[0.0]]],
            "vector": [1.0],
            "action_mask": [True],
            "policy": [1.0],
            "value": -1.0,
        }
        ex = TrainingExample.from_dict(data)
        assert ex.value == -1.0

    def test_from_dict_with_zero_value(self) -> None:
        data = {
            "spatial": [[[0.0]]],
            "vector": [0.0],
            "action_mask": [False],
            "policy": [1.0],
            "value": 0.0,
        }
        ex = TrainingExample.from_dict(data)
        assert ex.value == 0.0


# =========================================================================
# Section 9: GameTrajectory
# =========================================================================


class TestGameTrajectory:
    """Tests for mutable GameTrajectory with auto-UUID."""

    def _make_example(self) -> TrainingExample:
        return TrainingExample(
            spatial=np.zeros((1, 2, 2), dtype=np.float32),
            vector=np.zeros(4, dtype=np.float32),
            action_mask=np.ones(3, dtype=np.bool_),
            policy=np.array([0.33, 0.33, 0.34], dtype=np.float32),
            value=0.0,
        )

    def test_creation(self) -> None:
        ex = self._make_example()
        traj = GameTrajectory(examples=[ex], outcome=GameOutcome.win)
        assert len(traj.examples) == 1
        assert traj.outcome == GameOutcome.win

    def test_auto_episode_id_is_string(self) -> None:
        traj = GameTrajectory(examples=[], outcome=GameOutcome.loss)
        assert isinstance(traj.episode_id, str)
        assert len(traj.episode_id) == 32  # uuid4().hex is 32 hex chars

    def test_auto_episode_id_unique(self) -> None:
        ids = {
            GameTrajectory(examples=[], outcome=GameOutcome.draw).episode_id
            for _ in range(100)
        }
        assert len(ids) == 100, "Expected 100 unique episode IDs"

    def test_default_iteration_is_zero(self) -> None:
        traj = GameTrajectory(examples=[], outcome=GameOutcome.timeout)
        assert traj.iteration == 0

    def test_default_metadata_is_empty_dict(self) -> None:
        traj = GameTrajectory(examples=[], outcome=GameOutcome.win)
        assert traj.metadata == {}
        assert isinstance(traj.metadata, dict)

    def test_default_metadata_is_not_shared(self) -> None:
        """Each instance should get its own metadata dict (field default_factory)."""
        a = GameTrajectory(examples=[], outcome=GameOutcome.win)
        b = GameTrajectory(examples=[], outcome=GameOutcome.win)
        a.metadata["key"] = "value"
        assert "key" not in b.metadata

    def test_mutable(self) -> None:
        traj = GameTrajectory(examples=[], outcome=GameOutcome.win)
        traj.iteration = 42
        assert traj.iteration == 42

    def test_has_slots(self) -> None:
        traj = GameTrajectory(examples=[], outcome=GameOutcome.win)
        assert hasattr(traj, "__slots__")

    def test_custom_metadata(self) -> None:
        traj = GameTrajectory(
            examples=[],
            outcome=GameOutcome.win,
            metadata={"opponent": "random", "seed": 42},
        )
        assert traj.metadata["opponent"] == "random"
        assert traj.metadata["seed"] == 42

    def test_multiple_examples(self) -> None:
        examples = [self._make_example() for _ in range(5)]
        traj = GameTrajectory(examples=examples, outcome=GameOutcome.loss, iteration=7)
        assert len(traj.examples) == 5
        assert traj.iteration == 7


# =========================================================================
# Section 10: SearchResult
# =========================================================================


class TestSearchResult:
    """Tests for frozen SearchResult dataclass."""

    @pytest.fixture()
    def result(self) -> SearchResult:
        return SearchResult(
            action=2,
            policy=np.array([0.1, 0.2, 0.7], dtype=np.float32),
            root_value=0.85,
            visit_counts=np.array([10.0, 20.0, 70.0], dtype=np.float32),
            num_simulations=100,
        )

    def test_creation(self, result: SearchResult) -> None:
        assert result.action == 2
        assert result.root_value == pytest.approx(0.85)
        assert result.num_simulations == 100

    def test_frozen_action(self, result: SearchResult) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.action = 0  # type: ignore[misc]

    def test_frozen_root_value(self, result: SearchResult) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.root_value = 0.0  # type: ignore[misc]

    def test_frozen_num_simulations(self, result: SearchResult) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.num_simulations = 200  # type: ignore[misc]

    def test_frozen_policy(self, result: SearchResult) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.policy = np.zeros(3, dtype=np.float32)  # type: ignore[misc]

    def test_frozen_visit_counts(self, result: SearchResult) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.visit_counts = np.zeros(3, dtype=np.float32)  # type: ignore[misc]

    def test_has_slots(self, result: SearchResult) -> None:
        assert hasattr(result, "__slots__")

    def test_policy_shape(self, result: SearchResult) -> None:
        assert result.policy.shape == (3,)

    def test_visit_counts_shape(self, result: SearchResult) -> None:
        assert result.visit_counts.shape == (3,)


# =========================================================================
# Section 11: EvalResult
# =========================================================================


class TestEvalResult:
    """Tests for frozen EvalResult dataclass."""

    @pytest.fixture()
    def result(self) -> EvalResult:
        return EvalResult(
            win_rate=0.65,
            draw_rate=0.10,
            elo_rating=1500.0,
            num_games=400,
        )

    def test_creation(self, result: EvalResult) -> None:
        assert result.win_rate == pytest.approx(0.65)
        assert result.draw_rate == pytest.approx(0.10)
        assert result.elo_rating == pytest.approx(1500.0)
        assert result.num_games == 400

    def test_frozen_win_rate(self, result: EvalResult) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.win_rate = 0.0  # type: ignore[misc]

    def test_frozen_draw_rate(self, result: EvalResult) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.draw_rate = 0.0  # type: ignore[misc]

    def test_frozen_elo_rating(self, result: EvalResult) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.elo_rating = 0.0  # type: ignore[misc]

    def test_frozen_num_games(self, result: EvalResult) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.num_games = 0  # type: ignore[misc]

    def test_has_slots(self, result: EvalResult) -> None:
        assert hasattr(result, "__slots__")

    def test_equality(self) -> None:
        a = EvalResult(win_rate=0.5, draw_rate=0.1, elo_rating=1200.0, num_games=100)
        b = EvalResult(win_rate=0.5, draw_rate=0.1, elo_rating=1200.0, num_games=100)
        assert a == b

    def test_inequality(self) -> None:
        a = EvalResult(win_rate=0.5, draw_rate=0.1, elo_rating=1200.0, num_games=100)
        b = EvalResult(win_rate=0.6, draw_rate=0.1, elo_rating=1200.0, num_games=100)
        assert a != b
