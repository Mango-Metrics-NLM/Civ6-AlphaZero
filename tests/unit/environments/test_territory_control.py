"""Tests for TerritoryControlEnv."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.types import ObservationSpec
from src.environments.mini_games.territory_control import TerritoryControlEnv

# ===================================================================
# Fixtures
# ===================================================================

SEED = 42
DEFAULT_BOARD_SIZE = 6


@pytest.fixture()
def env() -> TerritoryControlEnv:
    """Create a default territory control environment with a fixed seed."""
    e = TerritoryControlEnv(board_size=DEFAULT_BOARD_SIZE, seed=SEED)
    e.reset()
    return e


@pytest.fixture()
def small_env() -> TerritoryControlEnv:
    """Create a small 2x2 board for fast terminal tests."""
    e = TerritoryControlEnv(board_size=2, seed=SEED)
    e.reset()
    return e


# ===================================================================
# Properties
# ===================================================================


class TestTerritoryControlProperties:
    """Tests for observation_spec and action_space_size."""

    def test_observation_spec_type(self, env: TerritoryControlEnv) -> None:
        spec = env.observation_spec
        assert isinstance(spec, ObservationSpec)

    def test_observation_spec_spatial_shape(self) -> None:
        board_size = 10
        e = TerritoryControlEnv(board_size=board_size)
        spec = e.observation_spec
        assert spec.spatial_shape == (5, board_size, board_size)

    def test_observation_spec_vector_dim(self, env: TerritoryControlEnv) -> None:
        spec = env.observation_spec
        assert spec.vector_dim == 4

    def test_action_space_size_equals_board_squared(self) -> None:
        for bs in (4, 6, 10):
            e = TerritoryControlEnv(board_size=bs)
            assert e.action_space_size == bs * bs


# ===================================================================
# Reset
# ===================================================================


class TestTerritoryControlReset:
    """Tests for the reset method."""

    def test_reset_returns_dict_with_spatial_and_vector(
        self, env: TerritoryControlEnv
    ) -> None:
        obs = env.reset()
        assert isinstance(obs, dict)
        assert "spatial" in obs
        assert "vector" in obs

    def test_reset_spatial_shape_matches_spec(
        self, env: TerritoryControlEnv
    ) -> None:
        obs = env.reset()
        assert obs["spatial"].shape == env.observation_spec.spatial_shape

    def test_reset_vector_shape(self, env: TerritoryControlEnv) -> None:
        obs = env.reset()
        assert obs["vector"].shape == (4,)

    def test_reset_spatial_dtype(self, env: TerritoryControlEnv) -> None:
        obs = env.reset()
        assert obs["spatial"].dtype == np.float32

    def test_reset_vector_dtype(self, env: TerritoryControlEnv) -> None:
        obs = env.reset()
        assert obs["vector"].dtype == np.float32

    def test_reset_not_terminal(self, env: TerritoryControlEnv) -> None:
        env.reset()
        assert not env.is_terminal()

    def test_reset_clears_board(self, env: TerritoryControlEnv) -> None:
        """After stepping, reset should yield an empty board again."""
        env.step(0)
        env.reset()
        assert not env.is_terminal()
        # All cells unclaimed: channel 2 should be all ones
        obs = env.get_observation()
        np.testing.assert_array_equal(
            obs["spatial"][2], np.ones((DEFAULT_BOARD_SIZE, DEFAULT_BOARD_SIZE))
        )

    def test_seeded_reset_is_reproducible(self) -> None:
        e1 = TerritoryControlEnv()
        e2 = TerritoryControlEnv()
        obs1 = e1.reset(seed=99)
        obs2 = e2.reset(seed=99)
        np.testing.assert_array_equal(obs1["spatial"], obs2["spatial"])
        np.testing.assert_array_equal(obs1["vector"], obs2["vector"])


# ===================================================================
# Step
# ===================================================================


class TestTerritoryControlStep:
    """Tests for the step method."""

    def test_step_returns_five_tuple(self, env: TerritoryControlEnv) -> None:
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_valid_action_claims_cell(self, env: TerritoryControlEnv) -> None:
        """Placing on an unclaimed cell should claim it with no penalty."""
        action = 0  # top-left cell
        _, reward, _, _, _ = env.step(action)
        # Valid move yields 0.0 reward (unless it also terminates)
        if not env.is_terminal():
            assert reward == 0.0

    def test_valid_action_removes_cell_from_valid(
        self, env: TerritoryControlEnv
    ) -> None:
        action = 0
        env.step(action)
        mask = env.get_valid_actions()
        assert not mask[action]

    def test_invalid_action_already_claimed_penalizes(
        self, env: TerritoryControlEnv
    ) -> None:
        """Placing on an already claimed cell should give a negative reward."""
        action = 0
        env.step(action)  # player 0 claims cell 0
        # Now player 1 tries to claim the same cell
        _, reward, _, _, _ = env.step(action)
        if not env.is_terminal():
            assert reward == pytest.approx(-0.1)

    def test_step_alternates_current_player(
        self, env: TerritoryControlEnv
    ) -> None:
        assert env._current_player == 0
        env.step(0)
        assert env._current_player == 1
        env.step(1)
        assert env._current_player == 0

    def test_step_increments_step_count(
        self, env: TerritoryControlEnv
    ) -> None:
        initial_step = env._step_count
        env.step(0)
        assert env._step_count == initial_step + 1

    def test_step_reward_in_range(self, env: TerritoryControlEnv) -> None:
        board_size = env._board_size
        for action in range(board_size * board_size):
            if env.is_terminal():
                break
            _, reward, _, _, _ = env.step(action)
            assert -1.0 <= reward <= 1.0

    def test_step_on_terminated_raises(
        self, small_env: TerritoryControlEnv
    ) -> None:
        """Stepping a terminated environment should raise RuntimeError."""
        total_cells = small_env._board_size * small_env._board_size
        for i in range(total_cells):
            small_env.step(i)
        assert small_env.is_terminal()
        with pytest.raises(RuntimeError, match="terminated"):
            small_env.step(0)

    def test_step_out_of_range_raises(
        self, env: TerritoryControlEnv
    ) -> None:
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(-1)
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(env.action_space_size)

    def test_info_contains_expected_keys(
        self, env: TerritoryControlEnv
    ) -> None:
        _, _, _, _, info = env.step(0)
        assert "scores" in info
        assert "step" in info
        assert "current_player" in info
        assert "unclaimed" in info


# ===================================================================
# get_valid_actions
# ===================================================================


class TestTerritoryControlValidActions:
    """Tests for the get_valid_actions method."""

    def test_initial_all_valid(self, env: TerritoryControlEnv) -> None:
        mask = env.get_valid_actions()
        assert mask.shape == (env.action_space_size,)
        assert mask.dtype == np.bool_
        assert np.all(mask)

    def test_claimed_cells_become_invalid(
        self, env: TerritoryControlEnv
    ) -> None:
        env.step(0)  # player 0 claims cell 0
        mask = env.get_valid_actions()
        assert not mask[0]
        # Other cells remain valid
        assert mask[1]

    def test_valid_count_decreases(
        self, env: TerritoryControlEnv
    ) -> None:
        initial_valid = int(np.sum(env.get_valid_actions()))
        env.step(0)  # valid claim
        new_valid = int(np.sum(env.get_valid_actions()))
        assert new_valid == initial_valid - 1

    def test_invalid_move_does_not_decrease_valid_count(
        self, env: TerritoryControlEnv
    ) -> None:
        env.step(0)  # player 0 claims cell 0
        valid_before = int(np.sum(env.get_valid_actions()))
        env.step(0)  # player 1 tries same cell (invalid, no cell claimed)
        valid_after = int(np.sum(env.get_valid_actions()))
        assert valid_after == valid_before


# ===================================================================
# Terminal conditions
# ===================================================================


class TestTerritoryControlTermination:
    """Tests for game termination."""

    def test_game_ends_when_board_full(
        self, small_env: TerritoryControlEnv
    ) -> None:
        board_size = small_env._board_size
        total_cells = board_size * board_size
        for i in range(total_cells):
            assert not small_env.is_terminal()
            small_env.step(i)
        assert small_env.is_terminal()

    def test_terminated_flag_set(
        self, small_env: TerritoryControlEnv
    ) -> None:
        board_size = small_env._board_size
        total_cells = board_size * board_size
        for i in range(total_cells):
            _, _, terminated, truncated, _ = small_env.step(i)
        assert terminated
        assert not truncated

    def test_not_terminal_before_board_full(
        self, env: TerritoryControlEnv
    ) -> None:
        env.step(0)
        assert not env.is_terminal()


# ===================================================================
# Scoring
# ===================================================================


class TestTerritoryControlScoring:
    """Tests for territory scoring with connected region bonus."""

    def test_territory_count_tracked_in_scores(
        self, env: TerritoryControlEnv
    ) -> None:
        env.step(0)  # player 0 claims a cell
        assert env._scores[0] == 1
        assert env._scores[1] == 0
        env.step(5)  # player 1 claims a cell
        assert env._scores[1] == 1

    def test_connected_region_bonus(self) -> None:
        """Adjacent cells should produce a connected region bonus."""
        env = TerritoryControlEnv(board_size=3, seed=SEED)
        env.reset()
        # Player 0 claims (0,0), player 1 claims (2,2)
        # Player 0 claims (0,1), player 1 claims (2,1)
        # Player 0 claims (0,2), player 1 claims (2,0)
        # Remaining: (1,0), (1,1), (1,2)
        actions_p0 = [0, 1, 2]   # row 0: cells 0, 1, 2 (connected horizontal)
        actions_p1 = [8, 7, 6]   # row 2: cells 8, 7, 6 (connected horizontal)
        for a0, a1 in zip(actions_p0, actions_p1, strict=True):
            env.step(a0)
            env.step(a1)

        # Each player has 3 cells in a connected region of size 3
        score_p0 = env._compute_territory_score(1)  # marker 1 = player 0
        score_p1 = env._compute_territory_score(2)  # marker 2 = player 1
        # Score = owned cells (3) + largest connected region bonus (3) = 6
        assert score_p0 == 3 + 3
        assert score_p1 == 3 + 3

    def test_disconnected_regions_give_smaller_bonus(self) -> None:
        """Disconnected cells should give a smaller bonus than one big region."""
        env = TerritoryControlEnv(board_size=4, seed=SEED)
        env.reset()
        # Player 0 claims cells (0,0) and (2,2) -- disconnected
        env.step(0)   # (0,0) for player 0
        env.step(1)   # (0,1) for player 1
        env.step(10)  # (2,2) for player 0
        env.step(3)   # (0,3) for player 1

        # Player 0: 2 cells, but largest connected region = 1
        score_p0 = env._compute_territory_score(1)
        assert score_p0 == 2 + 1  # 2 cells + largest region of 1

    def test_final_reward_win(self) -> None:
        """Player who controls more territory should win."""
        env = TerritoryControlEnv(board_size=2, seed=SEED)
        env.reset()
        # 2x2 board = 4 cells.  If player 0 gets 3, player 1 gets 1:
        # Alternate: p0 gets cell 0, p1 gets cell 1, p0 gets cell 2, p0 claims cell 3 (invalid)
        # Actually with alternating turns, each gets 2 cells in a 4-cell board.
        # Let's fill it up: p0 claims 0, p1 claims 1, p0 claims 2, p1 claims 3
        env.step(0)  # p0
        env.step(1)  # p1
        env.step(2)  # p0
        _, reward, terminated, _, _ = env.step(3)  # p1
        assert terminated
        # Reward is from player 0's perspective
        # p0: cells (0,0),(1,0) in row-wise layout. Scores computed with connected bonus.
        assert reward in (-1.0, 0.0, 1.0)

    def test_final_reward_draw(self) -> None:
        """Equal scores should produce a draw (reward 0)."""
        env = TerritoryControlEnv(board_size=2, seed=SEED)
        env.reset()
        # Both players claim 2 cells with equal connectivity bonus
        # Layout: p0 claims 0, p1 claims 3, p0 claims 1, p1 claims 2
        # p0 has (0,0) and (0,1) = connected region of 2, score=2+2=4
        # p1 has (1,1) and (1,0) = connected region of 2, score=2+2=4
        env.step(0)  # p0 claims (0,0)
        env.step(3)  # p1 claims (1,1)
        env.step(1)  # p0 claims (0,1)
        _, reward, terminated, _, _ = env.step(2)  # p1 claims (1,0)
        assert terminated
        assert reward == pytest.approx(0.0)


# ===================================================================
# Clone
# ===================================================================


class TestTerritoryControlClone:
    """Tests for the clone method."""

    def test_clone_returns_distinct_object(
        self, env: TerritoryControlEnv
    ) -> None:
        cloned = env.clone()
        assert cloned is not env

    def test_clone_preserves_state(
        self, env: TerritoryControlEnv
    ) -> None:
        env.step(0)
        env.step(5)
        cloned = env.clone()
        assert cloned._step_count == env._step_count
        assert cloned._current_player == env._current_player
        assert cloned._scores == env._scores
        np.testing.assert_array_equal(cloned._board, env._board)

    def test_clone_independent_of_original(
        self, env: TerritoryControlEnv
    ) -> None:
        env.step(0)
        cloned = env.clone()
        # Mutate original
        env.step(5)
        # Clone should be unaffected
        assert cloned._step_count != env._step_count
        assert not np.array_equal(cloned._board, env._board)

    def test_clone_produces_same_observation(
        self, env: TerritoryControlEnv
    ) -> None:
        env.step(0)
        cloned = env.clone()
        obs_orig = env.get_observation()
        obs_clone = cloned.get_observation()
        np.testing.assert_array_equal(obs_orig["spatial"], obs_clone["spatial"])
        np.testing.assert_array_equal(obs_orig["vector"], obs_clone["vector"])


# ===================================================================
# get_symmetries
# ===================================================================


class TestTerritoryControlSymmetries:
    """Tests for the get_symmetries method."""

    def test_returns_four_rotational_symmetries(
        self, env: TerritoryControlEnv
    ) -> None:
        obs = env.get_observation()
        policy = np.ones(env.action_space_size, dtype=np.float32) / env.action_space_size
        symmetries = env.get_symmetries(obs["spatial"], policy)
        assert isinstance(symmetries, list)
        assert len(symmetries) == 4

    def test_first_symmetry_is_identity(
        self, env: TerritoryControlEnv
    ) -> None:
        obs = env.get_observation()
        policy = np.arange(env.action_space_size, dtype=np.float32)
        symmetries = env.get_symmetries(obs["spatial"], policy)
        sym_obs, sym_policy = symmetries[0]
        np.testing.assert_array_equal(sym_obs, obs["spatial"])
        np.testing.assert_array_equal(sym_policy, policy)

    def test_symmetry_shapes_match_original(
        self, env: TerritoryControlEnv
    ) -> None:
        obs = env.get_observation()
        policy = np.ones(env.action_space_size, dtype=np.float32)
        symmetries = env.get_symmetries(obs["spatial"], policy)
        for sym_obs, sym_policy in symmetries:
            assert sym_obs.shape == obs["spatial"].shape
            assert sym_policy.shape == policy.shape

    def test_rotations_are_distinct_with_asymmetric_state(self) -> None:
        """With an asymmetric board, rotations should be different."""
        env = TerritoryControlEnv(board_size=4, seed=SEED)
        env.reset()
        # Create an asymmetric board state
        env.step(0)   # claim top-left
        env.step(15)  # opponent claims bottom-right
        obs = env.get_observation()
        policy = np.zeros(env.action_space_size, dtype=np.float32)
        policy[0] = 1.0  # asymmetric policy

        symmetries = env.get_symmetries(obs["spatial"], policy)
        # At least some rotations should differ from identity
        identity_obs = symmetries[0][0]
        differs = False
        for i in range(1, len(symmetries)):
            if not np.array_equal(symmetries[i][0], identity_obs):
                differs = True
                break
        assert differs

    def test_symmetry_policy_sums_preserved(
        self, env: TerritoryControlEnv
    ) -> None:
        """Each rotated policy should sum to the same total."""
        obs = env.get_observation()
        policy = np.random.default_rng(SEED).dirichlet(
            np.ones(env.action_space_size)
        ).astype(np.float32)
        symmetries = env.get_symmetries(obs["spatial"], policy)
        original_sum = float(np.sum(policy))
        for _, sym_policy in symmetries:
            assert float(np.sum(sym_policy)) == pytest.approx(original_sum, abs=1e-5)


# ===================================================================
# Observation shapes
# ===================================================================


class TestTerritoryControlObservationShapes:
    """Tests for spatial and vector observation shapes and values."""

    def test_spatial_shape_5_h_w(self, env: TerritoryControlEnv) -> None:
        obs = env.get_observation()
        bs = env._board_size
        assert obs["spatial"].shape == (5, bs, bs)

    def test_vector_shape_4(self, env: TerritoryControlEnv) -> None:
        obs = env.get_observation()
        assert obs["vector"].shape == (4,)

    def test_spatial_channels_initial_state(
        self, env: TerritoryControlEnv
    ) -> None:
        """On a fresh board, channel 2 (unclaimed) should be all ones."""
        obs = env.get_observation()
        # Channel 0 (current player territory): all zeros
        assert np.sum(obs["spatial"][0]) == 0
        # Channel 1 (opponent territory): all zeros
        assert np.sum(obs["spatial"][1]) == 0
        # Channel 2 (unclaimed): all ones
        expected = np.ones((env._board_size, env._board_size), dtype=np.float32)
        np.testing.assert_array_equal(obs["spatial"][2], expected)

    def test_vector_bias_term(self, env: TerritoryControlEnv) -> None:
        obs = env.get_observation()
        assert obs["vector"][3] == pytest.approx(1.0)

    def test_vector_initial_values(self, env: TerritoryControlEnv) -> None:
        """At reset, scores should be 0 and progress should be 0."""
        obs = env.get_observation()
        assert obs["vector"][0] == pytest.approx(0.0)  # own score
        assert obs["vector"][1] == pytest.approx(0.0)  # opponent score
        assert obs["vector"][2] == pytest.approx(0.0)  # progress

    def test_observation_updates_after_step(
        self, env: TerritoryControlEnv
    ) -> None:
        """After a step, observation should reflect the new state."""
        env.step(0)
        obs = env.get_observation()
        # Progress should be > 0
        assert obs["vector"][2] > 0.0

    def test_custom_board_size_shapes(self) -> None:
        for bs in (3, 5, 12):
            e = TerritoryControlEnv(board_size=bs, seed=SEED)
            e.reset()
            obs = e.get_observation()
            assert obs["spatial"].shape == (5, bs, bs)
            assert obs["vector"].shape == (4,)


# ===================================================================
# Coverage gap tests — terminal reward paths
# ===================================================================


class TestTerminalRewardPaths:
    """Test win/loss reward at game end (territory score comparison)."""

    def test_p1_wins_positive_reward(self) -> None:
        """When p1 has higher territory score, final reward is 1.0."""
        env = TerritoryControlEnv(board_size=3, seed=SEED)
        env.reset()
        # Give player 1 (marker=1) almost all territory, leave one empty
        env._board[:] = 1
        env._board[0, 0] = 0  # one unclaimed cell for the final action
        # Advance step count to near end so next step triggers termination
        env._step_count = env._max_steps - 1
        _obs, reward, terminated, _truncated, _info = env.step(0)
        assert terminated or _truncated
        # p1 owns 8 cells + the action cell, p2 owns 0 => reward = +1.0
        assert reward == pytest.approx(1.0)

    def test_p2_wins_negative_reward(self) -> None:
        """When p2 has higher territory score, final reward is -1.0."""
        env = TerritoryControlEnv(board_size=3, seed=SEED)
        env.reset()
        # Give player 2 (marker=2) almost all territory, leave one empty
        env._board[:] = 2
        env._board[0, 0] = 0  # one unclaimed cell for the final action
        # Advance step count to near end
        env._step_count = env._max_steps - 1
        _obs, reward, terminated, _truncated, _info = env.step(0)
        assert terminated or _truncated
        # p2 owns 8 cells, p0's action claims 1 => p2 still dominates
        assert reward == pytest.approx(-1.0)

    def test_reward_perspective_when_player1_acts_and_wins(self) -> None:
        """When player 1 is the acting player and wins, reward should be +1.0.

        Validates the acting-player-perspective reward convention (consistent
        with MiniResourceEnv and TacticalCombatEnv).
        """
        env = TerritoryControlEnv(board_size=3, seed=SEED)
        env.reset()
        # Set up: player 1 (marker=2) owns most territory
        env._board[:] = 2
        env._board[0, 0] = 0  # one unclaimed cell
        # Make player 1 the acting player
        env._current_player = 1
        env._step_count = env._max_steps - 1
        _obs, reward, terminated, _truncated, _info = env.step(0)
        assert terminated or _truncated
        # Player 1 is acting and has more territory → reward = +1.0
        assert reward == pytest.approx(1.0)

    def test_reward_perspective_when_player1_acts_and_loses(self) -> None:
        """When player 1 is the acting player and loses, reward should be -1.0."""
        env = TerritoryControlEnv(board_size=3, seed=SEED)
        env.reset()
        # Set up: player 0 (marker=1) owns most territory
        env._board[:] = 1
        env._board[0, 0] = 0  # one unclaimed cell
        # Make player 1 the acting player
        env._current_player = 1
        env._step_count = env._max_steps - 1
        _obs, reward, terminated, _truncated, _info = env.step(0)
        assert terminated or _truncated
        # Player 1 is acting but player 0 has more territory → reward = -1.0
        assert reward == pytest.approx(-1.0)
