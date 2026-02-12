"""Comprehensive tests for MicroCivAdapter in stub and non-stub modes.

Stub-mode tests use use_stub=True. Non-stub tests mock alpha_zero.envs.microciv
because gymnasium is not importable in this environment.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.adapters.civ6 import MicroCivAdapter
from src.core.constants import (
    DEFAULT_BOARD_SIZE,
    DEFAULT_MAX_STEPS,
    MICROCIV_SPATIAL_CHANNELS,
    MICROCIV_VECTOR_DIM,
    STUB_SPATIAL_CHANNELS,
    STUB_VECTOR_DIM,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def adapter() -> MicroCivAdapter:
    """Default stub-mode adapter with standard parameters."""
    return MicroCivAdapter(use_stub=True)


@pytest.fixture()
def small_adapter() -> MicroCivAdapter:
    """Small board adapter for faster tests."""
    return MicroCivAdapter(board_size=8, max_steps=10, use_stub=True)


# ---------------------------------------------------------------------------
# Constructor Tests
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for MicroCivAdapter.__init__."""

    def test_stub_mode_explicit(self) -> None:
        """Explicit use_stub=True creates adapter in stub mode."""
        adapter = MicroCivAdapter(use_stub=True)
        assert adapter._use_stub is True
        assert adapter._env is None

    def test_fallback_to_stub_when_import_fails(self) -> None:
        """When use_stub=False but import fails, adapter falls back to stub."""
        adapter = MicroCivAdapter(use_stub=False)
        # In this environment, alpha_zero.envs.microciv is not importable,
        # so it should have fallen back to stub mode.
        assert adapter._use_stub is True
        assert adapter._env is None

    def test_default_board_size(self) -> None:
        adapter = MicroCivAdapter(use_stub=True)
        assert adapter._board_size == DEFAULT_BOARD_SIZE

    def test_custom_board_size(self) -> None:
        adapter = MicroCivAdapter(board_size=32, use_stub=True)
        assert adapter._board_size == 32

    def test_default_max_steps(self) -> None:
        adapter = MicroCivAdapter(use_stub=True)
        assert adapter._max_steps == DEFAULT_MAX_STEPS

    def test_custom_max_steps(self) -> None:
        adapter = MicroCivAdapter(max_steps=50, use_stub=True)
        assert adapter._max_steps == 50

    def test_initial_state_flags(self) -> None:
        adapter = MicroCivAdapter(use_stub=True)
        assert adapter._step_count == 0
        assert adapter._terminated is False
        assert adapter._truncated is False

    def test_action_space_size_formula(self) -> None:
        """Stub action_space_size = 5 * board_size^2 + 1."""
        board = 10
        adapter = MicroCivAdapter(board_size=board, use_stub=True)
        expected = 5 * board * board + 1
        assert adapter.action_space_size == expected

    def test_stub_observation_spec_channels(self) -> None:
        """Stub uses 6 channels (not 13 like real MicroCiv)."""
        adapter = MicroCivAdapter(use_stub=True)
        assert adapter.observation_spec.spatial_shape[0] == STUB_SPATIAL_CHANNELS

    def test_stub_observation_spec_vector_dim(self) -> None:
        """Stub uses vector_dim=4 (not 64 like real MicroCiv)."""
        adapter = MicroCivAdapter(use_stub=True)
        assert adapter.observation_spec.vector_dim == STUB_VECTOR_DIM


# ---------------------------------------------------------------------------
# observation_spec property
# ---------------------------------------------------------------------------


class TestObservationSpec:
    """Tests for the observation_spec property."""

    def test_spatial_shape_matches_board(self) -> None:
        board = 12
        adapter = MicroCivAdapter(board_size=board, use_stub=True)
        spec = adapter.observation_spec
        assert spec.spatial_shape == (STUB_SPATIAL_CHANNELS, board, board)

    def test_vector_dim(self, adapter: MicroCivAdapter) -> None:
        assert adapter.observation_spec.vector_dim == STUB_VECTOR_DIM

    def test_spec_is_frozen(self, adapter: MicroCivAdapter) -> None:
        """ObservationSpec is a frozen dataclass; mutation should raise."""
        with pytest.raises(AttributeError):
            adapter.observation_spec.vector_dim = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# action_space_size property
# ---------------------------------------------------------------------------


class TestActionSpaceSize:
    """Tests for the action_space_size property."""

    def test_default_board(self, adapter: MicroCivAdapter) -> None:
        expected = 5 * DEFAULT_BOARD_SIZE * DEFAULT_BOARD_SIZE + 1
        assert adapter.action_space_size == expected

    @pytest.mark.parametrize("board", [4, 8, 16, 32])
    def test_various_board_sizes(self, board: int) -> None:
        adapter = MicroCivAdapter(board_size=board, use_stub=True)
        expected = 5 * board * board + 1
        assert adapter.action_space_size == expected


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    """Tests for the reset method in stub mode."""

    def test_returns_dict_with_expected_keys(self, adapter: MicroCivAdapter) -> None:
        obs = adapter.reset()
        assert isinstance(obs, dict)
        assert "spatial" in obs
        assert "vector" in obs

    def test_spatial_shape(self, adapter: MicroCivAdapter) -> None:
        obs = adapter.reset()
        expected = adapter.observation_spec.spatial_shape
        assert obs["spatial"].shape == expected

    def test_vector_shape(self, adapter: MicroCivAdapter) -> None:
        obs = adapter.reset()
        expected_dim = adapter.observation_spec.vector_dim
        assert obs["vector"].shape == (expected_dim,)

    def test_dtype_is_float32(self, adapter: MicroCivAdapter) -> None:
        obs = adapter.reset()
        assert obs["spatial"].dtype == np.float32
        assert obs["vector"].dtype == np.float32

    def test_stub_observation_is_zeros(self, adapter: MicroCivAdapter) -> None:
        obs = adapter.reset()
        np.testing.assert_array_equal(obs["spatial"], 0.0)
        np.testing.assert_array_equal(obs["vector"], 0.0)

    def test_reset_clears_step_count(self, small_adapter: MicroCivAdapter) -> None:
        small_adapter.reset()
        for _ in range(3):
            small_adapter.step(0)
        assert small_adapter._step_count == 3
        small_adapter.reset()
        assert small_adapter._step_count == 0

    def test_reset_clears_terminal_flags(
        self, small_adapter: MicroCivAdapter
    ) -> None:
        small_adapter.reset()
        # Drive to truncation
        for _ in range(small_adapter._max_steps):
            small_adapter.step(0)
        assert small_adapter.is_terminal() is True
        small_adapter.reset()
        assert small_adapter.is_terminal() is False
        assert small_adapter._terminated is False
        assert small_adapter._truncated is False

    def test_reset_with_seed(self, adapter: MicroCivAdapter) -> None:
        """reset(seed=...) should succeed (stub ignores seed)."""
        obs = adapter.reset(seed=42)
        assert isinstance(obs, dict)
        assert "spatial" in obs

    def test_reset_with_none_seed(self, adapter: MicroCivAdapter) -> None:
        obs = adapter.reset(seed=None)
        assert isinstance(obs, dict)


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


class TestStep:
    """Tests for the step method in stub mode."""

    def test_returns_five_tuple(self, adapter: MicroCivAdapter) -> None:
        adapter.reset()
        result = adapter.step(0)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_obs_structure(self, adapter: MicroCivAdapter) -> None:
        adapter.reset()
        obs, _, _, _, _ = adapter.step(0)
        assert "spatial" in obs
        assert "vector" in obs

    def test_obs_shapes(self, adapter: MicroCivAdapter) -> None:
        adapter.reset()
        obs, _, _, _, _ = adapter.step(0)
        assert obs["spatial"].shape == adapter.observation_spec.spatial_shape
        assert obs["vector"].shape == (adapter.observation_spec.vector_dim,)

    def test_reward_is_zero(self, adapter: MicroCivAdapter) -> None:
        adapter.reset()
        _, reward, _, _, _ = adapter.step(0)
        assert reward == 0.0

    def test_not_terminated_before_max_steps(
        self, small_adapter: MicroCivAdapter
    ) -> None:
        small_adapter.reset()
        _, _, terminated, truncated, _ = small_adapter.step(0)
        assert terminated is False
        assert truncated is False

    def test_truncation_at_max_steps(self, small_adapter: MicroCivAdapter) -> None:
        small_adapter.reset()
        max_steps = small_adapter._max_steps
        for i in range(max_steps):
            _, _, terminated, truncated, _info = small_adapter.step(0)
            if i < max_steps - 1:
                assert truncated is False, f"Truncated early at step {i + 1}"
        # After exactly max_steps, truncated should be True
        assert truncated is True
        assert terminated is False

    def test_info_contains_step_count(
        self, small_adapter: MicroCivAdapter
    ) -> None:
        small_adapter.reset()
        _, _, _, _, info = small_adapter.step(0)
        assert info["step"] == 1
        _, _, _, _, info = small_adapter.step(0)
        assert info["step"] == 2

    def test_step_increments_internal_counter(
        self, adapter: MicroCivAdapter
    ) -> None:
        adapter.reset()
        assert adapter._step_count == 0
        adapter.step(0)
        assert adapter._step_count == 1
        adapter.step(0)
        assert adapter._step_count == 2

    def test_step_after_truncation_stays_truncated(
        self, small_adapter: MicroCivAdapter
    ) -> None:
        """Steps beyond max_steps keep truncated=True."""
        small_adapter.reset()
        for _ in range(small_adapter._max_steps + 2):
            _, _, _, truncated, _ = small_adapter.step(0)
        assert truncated is True

    def test_reward_clamping_note(self, adapter: MicroCivAdapter) -> None:
        """In stub mode reward is always 0.0, which is within [-1, 1]."""
        adapter.reset()
        _, reward, _, _, _ = adapter.step(0)
        assert -1.0 <= reward <= 1.0


# ---------------------------------------------------------------------------
# get_valid_actions()
# ---------------------------------------------------------------------------


class TestGetValidActions:
    """Tests for get_valid_actions in stub mode."""

    def test_returns_all_ones(self, adapter: MicroCivAdapter) -> None:
        mask = adapter.get_valid_actions()
        assert np.all(mask)

    def test_shape_matches_action_space(self, adapter: MicroCivAdapter) -> None:
        mask = adapter.get_valid_actions()
        assert mask.shape == (adapter.action_space_size,)

    def test_dtype_is_bool(self, adapter: MicroCivAdapter) -> None:
        mask = adapter.get_valid_actions()
        assert mask.dtype == np.bool_

    @pytest.mark.parametrize("board", [4, 16])
    def test_mask_length_varies_with_board(self, board: int) -> None:
        adapter = MicroCivAdapter(board_size=board, use_stub=True)
        mask = adapter.get_valid_actions()
        expected_len = 5 * board * board + 1
        assert len(mask) == expected_len


# ---------------------------------------------------------------------------
# get_observation()
# ---------------------------------------------------------------------------


class TestGetObservation:
    """Tests for get_observation in stub mode."""

    def test_returns_dict(self, adapter: MicroCivAdapter) -> None:
        obs = adapter.get_observation()
        assert isinstance(obs, dict)
        assert set(obs.keys()) == {"spatial", "vector"}

    def test_shapes_match_spec(self, adapter: MicroCivAdapter) -> None:
        obs = adapter.get_observation()
        assert obs["spatial"].shape == adapter.observation_spec.spatial_shape
        assert obs["vector"].shape == (adapter.observation_spec.vector_dim,)

    def test_does_not_advance_step_count(
        self, adapter: MicroCivAdapter
    ) -> None:
        adapter.reset()
        adapter.step(0)
        count_before = adapter._step_count
        adapter.get_observation()
        assert adapter._step_count == count_before

    def test_dtype(self, adapter: MicroCivAdapter) -> None:
        obs = adapter.get_observation()
        assert obs["spatial"].dtype == np.float32
        assert obs["vector"].dtype == np.float32


# ---------------------------------------------------------------------------
# is_terminal()
# ---------------------------------------------------------------------------


class TestIsTerminal:
    """Tests for is_terminal in stub mode."""

    def test_not_terminal_after_reset(self, adapter: MicroCivAdapter) -> None:
        adapter.reset()
        assert adapter.is_terminal() is False

    def test_not_terminal_after_one_step(
        self, adapter: MicroCivAdapter
    ) -> None:
        adapter.reset()
        adapter.step(0)
        assert adapter.is_terminal() is False

    def test_terminal_after_truncation(
        self, small_adapter: MicroCivAdapter
    ) -> None:
        small_adapter.reset()
        for _ in range(small_adapter._max_steps):
            small_adapter.step(0)
        assert small_adapter.is_terminal() is True

    def test_terminal_reflects_terminated_flag(self) -> None:
        adapter = MicroCivAdapter(use_stub=True)
        adapter.reset()
        adapter._terminated = True
        assert adapter.is_terminal() is True

    def test_terminal_reflects_truncated_flag(self) -> None:
        adapter = MicroCivAdapter(use_stub=True)
        adapter.reset()
        adapter._truncated = True
        assert adapter.is_terminal() is True

    def test_not_terminal_when_both_flags_false(self) -> None:
        adapter = MicroCivAdapter(use_stub=True)
        adapter.reset()
        adapter._terminated = False
        adapter._truncated = False
        assert adapter.is_terminal() is False


# ---------------------------------------------------------------------------
# clone()
# ---------------------------------------------------------------------------


class TestClone:
    """Tests for the clone (deep copy) method."""

    def test_clone_returns_microcivadapter(
        self, adapter: MicroCivAdapter
    ) -> None:
        adapter.reset()
        cloned = adapter.clone()
        assert isinstance(cloned, MicroCivAdapter)

    def test_clone_is_independent(
        self, small_adapter: MicroCivAdapter
    ) -> None:
        small_adapter.reset()
        small_adapter.step(0)
        cloned = small_adapter.clone()
        # Advance original further
        small_adapter.step(0)
        small_adapter.step(0)
        assert small_adapter._step_count == 3
        assert cloned._step_count == 1

    def test_clone_preserves_config(self) -> None:
        adapter = MicroCivAdapter(board_size=12, max_steps=50, use_stub=True)
        adapter.reset()
        cloned = adapter.clone()
        assert cloned._board_size == 12
        assert cloned._max_steps == 50
        assert cloned._use_stub is True

    def test_clone_preserves_flags(
        self, small_adapter: MicroCivAdapter
    ) -> None:
        small_adapter.reset()
        for _ in range(small_adapter._max_steps):
            small_adapter.step(0)
        assert small_adapter._truncated is True
        cloned = small_adapter.clone()
        assert cloned._truncated is True
        assert cloned.is_terminal() is True

    def test_clone_observation_spec_matches(
        self, adapter: MicroCivAdapter
    ) -> None:
        cloned = adapter.clone()
        assert cloned.observation_spec == adapter.observation_spec

    def test_clone_action_space_matches(
        self, adapter: MicroCivAdapter
    ) -> None:
        cloned = adapter.clone()
        assert cloned.action_space_size == adapter.action_space_size


# ---------------------------------------------------------------------------
# get_symmetries()
# ---------------------------------------------------------------------------


class TestGetSymmetries:
    """Tests for get_symmetries (identity-only for MicroCiv)."""

    def test_returns_single_pair(self, adapter: MicroCivAdapter) -> None:
        obs = np.random.randn(
            *adapter.observation_spec.spatial_shape
        ).astype(np.float32)
        policy = np.random.randn(adapter.action_space_size).astype(np.float32)
        result = adapter.get_symmetries(obs, policy)
        assert len(result) == 1

    def test_returns_original_unchanged(
        self, adapter: MicroCivAdapter
    ) -> None:
        obs = np.random.randn(
            *adapter.observation_spec.spatial_shape
        ).astype(np.float32)
        policy = np.random.randn(adapter.action_space_size).astype(np.float32)
        result = adapter.get_symmetries(obs, policy)
        returned_obs, returned_policy = result[0]
        np.testing.assert_array_equal(returned_obs, obs)
        np.testing.assert_array_equal(returned_policy, policy)

    def test_returned_arrays_are_same_objects(
        self, adapter: MicroCivAdapter
    ) -> None:
        """get_symmetries returns the original arrays, not copies."""
        obs = np.zeros(adapter.observation_spec.spatial_shape, dtype=np.float32)
        policy = np.zeros(adapter.action_space_size, dtype=np.float32)
        result = adapter.get_symmetries(obs, policy)
        assert result[0][0] is obs
        assert result[0][1] is policy


# ---------------------------------------------------------------------------
# _translate_obs() - dict input
# ---------------------------------------------------------------------------


class TestTranslateObsDict:
    """Tests for _translate_obs with dictionary observations."""

    def test_dict_with_spatial_and_vector(
        self, adapter: MicroCivAdapter
    ) -> None:
        spec = adapter.observation_spec
        raw = {
            "spatial": np.ones(spec.spatial_shape, dtype=np.float32),
            "vector": np.ones(spec.vector_dim, dtype=np.float32),
        }
        result = adapter._translate_obs(raw)
        np.testing.assert_array_equal(result["spatial"], 1.0)
        np.testing.assert_array_equal(result["vector"], 1.0)

    def test_dict_with_board_and_global_fallback(
        self, adapter: MicroCivAdapter
    ) -> None:
        """Keys 'board' and 'global' are used when 'spatial'/'vector' missing."""
        spec = adapter.observation_spec
        raw = {
            "board": np.full(spec.spatial_shape, 2.0, dtype=np.float32),
            "global": np.full(spec.vector_dim, 3.0, dtype=np.float32),
        }
        result = adapter._translate_obs(raw)
        np.testing.assert_array_equal(result["spatial"], 2.0)
        np.testing.assert_array_equal(result["vector"], 3.0)

    def test_dict_with_missing_keys_falls_back_to_zeros(
        self, adapter: MicroCivAdapter
    ) -> None:
        """Dict with no recognized keys produces zero arrays."""
        result = adapter._translate_obs({"unrelated": 42})
        np.testing.assert_array_equal(result["spatial"], 0.0)
        np.testing.assert_array_equal(result["vector"], 0.0)

    def test_dict_spatial_preferred_over_board(
        self, adapter: MicroCivAdapter
    ) -> None:
        """When both 'spatial' and 'board' exist, 'spatial' is used."""
        spec = adapter.observation_spec
        raw = {
            "spatial": np.full(spec.spatial_shape, 5.0, dtype=np.float32),
            "board": np.full(spec.spatial_shape, 9.0, dtype=np.float32),
        }
        result = adapter._translate_obs(raw)
        np.testing.assert_array_equal(result["spatial"], 5.0)

    def test_dict_dtype_cast_to_float32(
        self, adapter: MicroCivAdapter
    ) -> None:
        spec = adapter.observation_spec
        raw = {
            "spatial": np.ones(spec.spatial_shape, dtype=np.float64),
            "vector": np.ones(spec.vector_dim, dtype=np.int32),
        }
        result = adapter._translate_obs(raw)
        assert result["spatial"].dtype == np.float32
        assert result["vector"].dtype == np.float32


# ---------------------------------------------------------------------------
# _translate_obs() - tuple input
# ---------------------------------------------------------------------------


class TestTranslateObsTuple:
    """Tests for _translate_obs with tuple observations."""

    def test_tuple_two_elements(self, adapter: MicroCivAdapter) -> None:
        spec = adapter.observation_spec
        spatial = np.ones(spec.spatial_shape, dtype=np.float32)
        vector = np.full(spec.vector_dim, 7.0, dtype=np.float32)
        result = adapter._translate_obs((spatial, vector))
        np.testing.assert_array_equal(result["spatial"], 1.0)
        np.testing.assert_array_equal(result["vector"], 7.0)

    def test_tuple_single_element(self, adapter: MicroCivAdapter) -> None:
        """Single-element tuple: vector should be zeros."""
        spec = adapter.observation_spec
        spatial = np.ones(spec.spatial_shape, dtype=np.float32)
        result = adapter._translate_obs((spatial,))
        np.testing.assert_array_equal(result["spatial"], 1.0)
        expected_vector = np.zeros(spec.vector_dim, dtype=np.float32)
        np.testing.assert_array_equal(result["vector"], expected_vector)

    def test_tuple_dtype_cast(self, adapter: MicroCivAdapter) -> None:
        spec = adapter.observation_spec
        spatial = np.ones(spec.spatial_shape, dtype=np.float64)
        vector = np.ones(spec.vector_dim, dtype=np.int64)
        result = adapter._translate_obs((spatial, vector))
        assert result["spatial"].dtype == np.float32
        assert result["vector"].dtype == np.float32


# ---------------------------------------------------------------------------
# _translate_obs() - unknown input
# ---------------------------------------------------------------------------


class TestTranslateObsUnknown:
    """Tests for _translate_obs with unrecognized input types."""

    @pytest.mark.parametrize("bad_input", [42, "string", 3.14, None, True])
    def test_unknown_type_returns_zeros(
        self, adapter: MicroCivAdapter, bad_input: object
    ) -> None:
        result = adapter._translate_obs(bad_input)
        spec = adapter.observation_spec
        assert result["spatial"].shape == spec.spatial_shape
        assert result["vector"].shape == (spec.vector_dim,)
        np.testing.assert_array_equal(result["spatial"], 0.0)
        np.testing.assert_array_equal(result["vector"], 0.0)

    def test_unknown_dtype(self, adapter: MicroCivAdapter) -> None:
        result = adapter._translate_obs(object())
        assert result["spatial"].dtype == np.float32
        assert result["vector"].dtype == np.float32


# ---------------------------------------------------------------------------
# _stub_observation()
# ---------------------------------------------------------------------------


class TestStubObservation:
    """Tests for the _stub_observation helper."""

    def test_returns_dict(self, adapter: MicroCivAdapter) -> None:
        obs = adapter._stub_observation()
        assert isinstance(obs, dict)

    def test_keys(self, adapter: MicroCivAdapter) -> None:
        obs = adapter._stub_observation()
        assert set(obs.keys()) == {"spatial", "vector"}

    def test_spatial_shape(self, adapter: MicroCivAdapter) -> None:
        obs = adapter._stub_observation()
        assert obs["spatial"].shape == adapter.observation_spec.spatial_shape

    def test_vector_shape(self, adapter: MicroCivAdapter) -> None:
        obs = adapter._stub_observation()
        assert obs["vector"].shape == (adapter.observation_spec.vector_dim,)

    def test_all_zeros(self, adapter: MicroCivAdapter) -> None:
        obs = adapter._stub_observation()
        np.testing.assert_array_equal(obs["spatial"], 0.0)
        np.testing.assert_array_equal(obs["vector"], 0.0)

    def test_dtype(self, adapter: MicroCivAdapter) -> None:
        obs = adapter._stub_observation()
        assert obs["spatial"].dtype == np.float32
        assert obs["vector"].dtype == np.float32

    @pytest.mark.parametrize("board", [4, 8, 16, 32])
    def test_shape_varies_with_board_size(self, board: int) -> None:
        adapter = MicroCivAdapter(board_size=board, use_stub=True)
        obs = adapter._stub_observation()
        assert obs["spatial"].shape == (STUB_SPATIAL_CHANNELS, board, board)


# ---------------------------------------------------------------------------
# Observation shape consistency
# ---------------------------------------------------------------------------


class TestObservationShapeConsistency:
    """Verify observation shapes are self-consistent across all methods."""

    def test_reset_matches_spec(self, adapter: MicroCivAdapter) -> None:
        obs = adapter.reset()
        spec = adapter.observation_spec
        assert obs["spatial"].shape == spec.spatial_shape
        assert obs["vector"].shape == (spec.vector_dim,)

    def test_step_matches_spec(self, adapter: MicroCivAdapter) -> None:
        adapter.reset()
        obs, _, _, _, _ = adapter.step(0)
        spec = adapter.observation_spec
        assert obs["spatial"].shape == spec.spatial_shape
        assert obs["vector"].shape == (spec.vector_dim,)

    def test_get_observation_matches_spec(
        self, adapter: MicroCivAdapter
    ) -> None:
        obs = adapter.get_observation()
        spec = adapter.observation_spec
        assert obs["spatial"].shape == spec.spatial_shape
        assert obs["vector"].shape == (spec.vector_dim,)

    def test_stub_observation_matches_spec(
        self, adapter: MicroCivAdapter
    ) -> None:
        obs = adapter._stub_observation()
        spec = adapter.observation_spec
        assert obs["spatial"].shape == spec.spatial_shape
        assert obs["vector"].shape == (spec.vector_dim,)

    @pytest.mark.parametrize("board", [4, 8, 16])
    def test_all_methods_consistent_for_board_size(self, board: int) -> None:
        adapter = MicroCivAdapter(board_size=board, use_stub=True)
        spec = adapter.observation_spec

        reset_obs = adapter.reset()
        step_obs, _, _, _, _ = adapter.step(0)
        get_obs = adapter.get_observation()
        stub_obs = adapter._stub_observation()

        for obs in [reset_obs, step_obs, get_obs, stub_obs]:
            assert obs["spatial"].shape == spec.spatial_shape
            assert obs["vector"].shape == (spec.vector_dim,)


# ---------------------------------------------------------------------------
# Full episode integration
# ---------------------------------------------------------------------------


class TestFullEpisode:
    """Integration-style tests running a complete stub episode."""

    def test_episode_to_truncation(self) -> None:
        max_steps = 5
        adapter = MicroCivAdapter(
            board_size=4, max_steps=max_steps, use_stub=True
        )
        adapter.reset()

        for i in range(max_steps):
            _obs, reward, terminated, truncated, info = adapter.step(0)
            assert reward == 0.0
            assert terminated is False
            assert info["step"] == i + 1

        # Should be truncated now
        assert truncated is True
        assert adapter.is_terminal() is True

    def test_reset_after_episode_allows_new_episode(self) -> None:
        adapter = MicroCivAdapter(board_size=4, max_steps=3, use_stub=True)

        # First episode
        adapter.reset()
        for _ in range(3):
            adapter.step(0)
        assert adapter.is_terminal() is True

        # Second episode
        adapter.reset()
        assert adapter.is_terminal() is False
        assert adapter._step_count == 0
        _obs, _reward, _terminated, truncated, info = adapter.step(0)
        assert info["step"] == 1
        assert truncated is False

    def test_valid_actions_unchanged_through_episode(self) -> None:
        adapter = MicroCivAdapter(board_size=4, max_steps=5, use_stub=True)
        adapter.reset()
        initial_mask = adapter.get_valid_actions()
        for _ in range(3):
            adapter.step(0)
        later_mask = adapter.get_valid_actions()
        np.testing.assert_array_equal(initial_mask, later_mask)


# ---------------------------------------------------------------------------
# Non-stub mode tests (mock MicroCivEnv)
# ---------------------------------------------------------------------------


def _make_mock_microciv(board_size: int = 16, action_n: int = 356) -> MagicMock:
    """Create a mock that looks like alpha_zero.envs.microciv.MicroCivEnv."""
    mock_env = MagicMock()
    mock_action_space = MagicMock()
    mock_action_space.n = action_n
    mock_env.action_space = mock_action_space
    mock_env.reset.return_value = (
        {"spatial": np.zeros((13, board_size, board_size), dtype=np.float32)},
        {},
    )
    mock_env.get_observation.return_value = {
        "spatial": np.ones((13, board_size, board_size), dtype=np.float32)
    }
    mock_env.get_valid_actions.return_value = np.ones(action_n, dtype=np.bool_)
    mock_env.step.return_value = (
        {"spatial": np.zeros((13, board_size, board_size), dtype=np.float32)},
        0.5,
        False,
        False,
        {"info_key": "info_val"},
    )
    return mock_env


class TestNonStubModeConstruction:
    """Tests for MicroCivAdapter using mocked real MicroCivEnv."""

    def test_non_stub_uses_real_env(self) -> None:
        """When MicroCivEnv is importable, adapter uses it (not stub)."""
        mock_env = _make_mock_microciv()
        mock_module = MagicMock()
        mock_module.MicroCivEnv.return_value = mock_env

        with patch.dict("sys.modules", {"alpha_zero.envs.microciv": mock_module}):
            adapter = MicroCivAdapter(use_stub=False)

        assert adapter._use_stub is False
        assert adapter._env is mock_env

    def test_non_stub_action_space_from_env(self) -> None:
        """Non-stub adapter reads action_space.n from the real env."""
        mock_env = _make_mock_microciv(action_n=500)
        mock_module = MagicMock()
        mock_module.MicroCivEnv.return_value = mock_env

        with patch.dict("sys.modules", {"alpha_zero.envs.microciv": mock_module}):
            adapter = MicroCivAdapter(use_stub=False)

        assert adapter.action_space_size == 500

    def test_non_stub_observation_spec_uses_microciv_channels(self) -> None:
        """Non-stub adapter uses MICROCIV_SPATIAL_CHANNELS, not STUB values."""
        mock_env = _make_mock_microciv(board_size=16)
        mock_module = MagicMock()
        mock_module.MicroCivEnv.return_value = mock_env

        with patch.dict("sys.modules", {"alpha_zero.envs.microciv": mock_module}):
            adapter = MicroCivAdapter(board_size=16, use_stub=False)

        spec = adapter.observation_spec
        assert spec.spatial_shape == (MICROCIV_SPATIAL_CHANNELS, 16, 16)
        assert spec.vector_dim == MICROCIV_VECTOR_DIM


class TestNonStubModeOperations:
    """Tests for non-stub adapter operations."""

    @pytest.fixture()
    def non_stub_adapter(self) -> MicroCivAdapter:
        """Create a non-stub adapter with mocked MicroCivEnv."""
        mock_env = _make_mock_microciv(board_size=8)
        mock_module = MagicMock()
        mock_module.MicroCivEnv.return_value = mock_env
        with patch.dict("sys.modules", {"alpha_zero.envs.microciv": mock_module}):
            adapter = MicroCivAdapter(board_size=8, use_stub=False)
        return adapter

    def test_reset_delegates_to_env(self, non_stub_adapter: MicroCivAdapter) -> None:
        """Reset calls _env.reset and returns translated observation."""
        obs = non_stub_adapter.reset(seed=42)
        non_stub_adapter._env.reset.assert_called_once_with(seed=42)
        assert "spatial" in obs
        assert "vector" in obs

    def test_step_delegates_to_env(self, non_stub_adapter: MicroCivAdapter) -> None:
        """Step calls _env.step and propagates return values."""
        obs, _reward, _terminated, _truncated, _info = non_stub_adapter.step(7)
        non_stub_adapter._env.step.assert_called_once_with(7)
        assert "spatial" in obs
        assert "vector" in obs

    def test_step_clamps_reward(self, non_stub_adapter: MicroCivAdapter) -> None:
        """Reward is clamped to [-1, 1]."""
        non_stub_adapter._env.step.return_value = (
            {"spatial": np.zeros((13, 8, 8), dtype=np.float32)},
            5.0,  # out of range
            False,
            False,
            {},
        )
        _obs, reward, _t, _tr, _info = non_stub_adapter.step(0)
        assert reward == 1.0

        non_stub_adapter._env.step.return_value = (
            {"spatial": np.zeros((13, 8, 8), dtype=np.float32)},
            -10.0,
            False,
            False,
            {},
        )
        _obs, reward, _t, _tr, _info = non_stub_adapter.step(0)
        assert reward == -1.0

    def test_get_valid_actions_delegates(self, non_stub_adapter: MicroCivAdapter) -> None:
        """get_valid_actions returns mask from real env as bool array."""
        mask = non_stub_adapter.get_valid_actions()
        non_stub_adapter._env.get_valid_actions.assert_called_once()
        assert mask.dtype == np.bool_

    def test_get_observation_delegates(self, non_stub_adapter: MicroCivAdapter) -> None:
        """get_observation returns translated obs from real env."""
        obs = non_stub_adapter.get_observation()
        non_stub_adapter._env.get_observation.assert_called_once()
        assert "spatial" in obs
        assert "vector" in obs

    def test_is_terminal_delegates(self, non_stub_adapter: MicroCivAdapter) -> None:
        """is_terminal checks terminated/truncated flags from step()."""
        non_stub_adapter._env.step.return_value = (
            {"spatial": np.zeros((13, 8, 8), dtype=np.float32)},
            0.0,
            True,
            False,
            {},
        )
        non_stub_adapter.step(0)
        assert non_stub_adapter.is_terminal() is True
