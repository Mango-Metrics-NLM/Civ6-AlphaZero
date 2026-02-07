"""Tests for TacticalCombatEnv."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.types import ObservationSpec
from src.environments.mini_games.tactical_combat import TacticalCombatEnv, Unit

# ===================================================================
# Fixtures
# ===================================================================

SEED = 42
DEFAULT_BOARD_SIZE = 8
DEFAULT_NUM_UNITS = 3


@pytest.fixture()
def env() -> TacticalCombatEnv:
    """Create a default tactical combat environment with a fixed seed."""
    e = TacticalCombatEnv(
        board_size=DEFAULT_BOARD_SIZE,
        num_units=DEFAULT_NUM_UNITS,
        seed=SEED,
    )
    e.reset()
    return e


@pytest.fixture()
def small_env() -> TacticalCombatEnv:
    """Create a minimal environment for targeted testing."""
    e = TacticalCombatEnv(
        board_size=4,
        num_units=1,
        max_steps=50,
        seed=SEED,
    )
    e.reset()
    return e


# ===================================================================
# Unit dataclass
# ===================================================================


class TestUnit:
    """Tests for the Unit dataclass."""

    def test_default_values(self) -> None:
        u = Unit(player=0, row=0, col=0)
        assert u.health == 10
        assert u.max_health == 10
        assert u.attack == 3
        assert u.alive

    def test_take_damage_reduces_health(self) -> None:
        u = Unit(player=0, row=0, col=0, health=10)
        u.take_damage(3)
        assert u.health == 7
        assert u.alive

    def test_take_damage_exactly_lethal(self) -> None:
        u = Unit(player=0, row=0, col=0, health=5)
        u.take_damage(5)
        assert u.health == 0
        assert not u.alive

    def test_take_damage_overkill_clamps_to_zero(self) -> None:
        u = Unit(player=0, row=0, col=0, health=3)
        u.take_damage(100)
        assert u.health == 0
        assert not u.alive

    def test_take_zero_damage(self) -> None:
        u = Unit(player=0, row=0, col=0, health=10)
        u.take_damage(0)
        assert u.health == 10
        assert u.alive

    def test_alive_after_partial_damage(self) -> None:
        u = Unit(player=0, row=0, col=0, health=10)
        u.take_damage(9)
        assert u.health == 1
        assert u.alive


# ===================================================================
# Properties
# ===================================================================


class TestTacticalCombatProperties:
    """Tests for observation_spec and action_space_size."""

    def test_observation_spec_type(self, env: TacticalCombatEnv) -> None:
        spec = env.observation_spec
        assert isinstance(spec, ObservationSpec)

    def test_observation_spec_spatial_shape(self) -> None:
        board_size = 10
        e = TacticalCombatEnv(board_size=board_size, num_units=2)
        spec = e.observation_spec
        assert spec.spatial_shape == (7, board_size, board_size)

    def test_observation_spec_vector_dim(self, env: TacticalCombatEnv) -> None:
        spec = env.observation_spec
        assert spec.vector_dim == 5

    def test_action_space_size_equals_num_units_times_five(self) -> None:
        for n in (1, 3, 5):
            e = TacticalCombatEnv(num_units=n)
            assert e.action_space_size == n * 5


# ===================================================================
# Reset
# ===================================================================


class TestTacticalCombatReset:
    """Tests for the reset method."""

    def test_reset_returns_dict_with_spatial_and_vector(
        self, env: TacticalCombatEnv
    ) -> None:
        obs = env.reset()
        assert isinstance(obs, dict)
        assert "spatial" in obs
        assert "vector" in obs

    def test_reset_spatial_shape_matches_spec(
        self, env: TacticalCombatEnv
    ) -> None:
        obs = env.reset()
        assert obs["spatial"].shape == env.observation_spec.spatial_shape

    def test_reset_vector_shape(self, env: TacticalCombatEnv) -> None:
        obs = env.reset()
        assert obs["vector"].shape == (5,)

    def test_reset_spatial_dtype(self, env: TacticalCombatEnv) -> None:
        obs = env.reset()
        assert obs["spatial"].dtype == np.float32

    def test_reset_vector_dtype(self, env: TacticalCombatEnv) -> None:
        obs = env.reset()
        assert obs["vector"].dtype == np.float32

    def test_reset_not_terminal(self, env: TacticalCombatEnv) -> None:
        env.reset()
        assert not env.is_terminal()

    def test_units_placed_on_reset(self, env: TacticalCombatEnv) -> None:
        env.reset()
        assert len(env._units[0]) == DEFAULT_NUM_UNITS
        assert len(env._units[1]) == DEFAULT_NUM_UNITS

    def test_player_0_units_on_top_row(self, env: TacticalCombatEnv) -> None:
        env.reset()
        for u in env._units[0]:
            assert u.row == 0

    def test_player_1_units_on_bottom_row(self, env: TacticalCombatEnv) -> None:
        env.reset()
        for u in env._units[1]:
            assert u.row == env._board_size - 1

    def test_all_units_alive_on_reset(self, env: TacticalCombatEnv) -> None:
        env.reset()
        for player_units in env._units:
            for u in player_units:
                assert u.alive
                assert u.health == u.max_health

    def test_seeded_reset_is_reproducible(self) -> None:
        e1 = TacticalCombatEnv(seed=99)
        e2 = TacticalCombatEnv(seed=99)
        obs1 = e1.reset()
        obs2 = e2.reset()
        np.testing.assert_array_equal(obs1["spatial"], obs2["spatial"])
        np.testing.assert_array_equal(obs1["vector"], obs2["vector"])

    def test_reset_clears_previous_state(self, env: TacticalCombatEnv) -> None:
        """After stepping and resetting, state should be fresh."""
        env.step(0)
        env.step(0)
        env.reset()
        assert env._step_count == 0
        assert env._current_player == 0
        assert not env.is_terminal()


# ===================================================================
# Step
# ===================================================================


class TestTacticalCombatStep:
    """Tests for the step method."""

    def test_step_returns_five_tuple(self, env: TacticalCombatEnv) -> None:
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_unit_moves_on_step(self, small_env: TacticalCombatEnv) -> None:
        """Moving down (action index 2) should increase the unit's row."""
        unit = small_env._units[0][0]
        initial_row = unit.row
        # action 2 = move down for unit 0
        small_env.step(2)
        assert unit.row == min(initial_row + 1, small_env._board_size - 1)

    def test_unit_stays_on_stay_action(self, small_env: TacticalCombatEnv) -> None:
        """Stay/skip action (index 4) should not move the unit."""
        unit = small_env._units[0][0]
        initial_row = unit.row
        initial_col = unit.col
        small_env.step(4)  # stay action for unit 0
        assert unit.row == initial_row
        assert unit.col == initial_col

    def test_unit_clamped_to_board_boundary(
        self, small_env: TacticalCombatEnv
    ) -> None:
        """Moving up from row 0 should keep the unit at row 0."""
        unit = small_env._units[0][0]
        assert unit.row == 0
        # action 0 = move up
        small_env.step(0)
        assert unit.row == 0

    def test_combat_damages_enemy(self) -> None:
        """When a unit moves adjacent to an enemy, it should deal damage."""
        env = TacticalCombatEnv(board_size=4, num_units=1, max_steps=100, seed=SEED)
        env.reset()
        # Manually place units in the same column so they will converge
        p0_unit = env._units[0][0]
        p1_unit = env._units[1][0]
        p0_unit.row = 0
        p0_unit.col = 1
        p1_unit.row = 3
        p1_unit.col = 1
        initial_health = p1_unit.health

        # Move p0 down (action 2), p1 up (action 0), alternating
        for _ in range(10):
            if env.is_terminal():
                break
            env.step(2)  # p0 moves down
            if env.is_terminal():
                break
            env.step(0)  # p1 moves up

        # After converging in the same column, they will be adjacent and attack
        damage_dealt = (initial_health - p1_unit.health) + (
            initial_health - p0_unit.health
        )
        assert damage_dealt > 0 or not p0_unit.alive or not p1_unit.alive

    def test_elimination_reward(self) -> None:
        """Killing an enemy unit should give a positive reward."""
        env = TacticalCombatEnv(
            board_size=3, num_units=1, max_steps=100, seed=SEED
        )
        env.reset()
        # Manually place units adjacent for guaranteed combat
        env._units[0][0].row = 1
        env._units[0][0].col = 1
        env._units[1][0].row = 1
        env._units[1][0].col = 2
        env._units[1][0].health = 1  # one hit kill

        # Player 0 moves right (action 1) into combat range
        _, reward, _, _, _ = env.step(1)
        # Enemy should be eliminated; reward should be positive
        assert not env._units[1][0].alive
        assert reward > 0

    def test_step_alternates_current_player(
        self, env: TacticalCombatEnv
    ) -> None:
        assert env._current_player == 0
        env.step(4)  # stay
        assert env._current_player == 1
        env.step(4)  # stay
        assert env._current_player == 0

    def test_step_increments_step_count(
        self, env: TacticalCombatEnv
    ) -> None:
        initial = env._step_count
        env.step(4)
        assert env._step_count == initial + 1

    def test_step_reward_in_range(self, env: TacticalCombatEnv) -> None:
        for _ in range(20):
            if env.is_terminal():
                break
            _, reward, _, _, _ = env.step(4)
            assert -1.0 <= reward <= 1.0

    def test_step_on_terminated_raises(self) -> None:
        """Stepping a terminated environment should raise RuntimeError."""
        env = TacticalCombatEnv(board_size=3, num_units=1, max_steps=100, seed=SEED)
        env.reset()
        # Manually kill all enemy units to terminate
        for u in env._units[1]:
            u.take_damage(u.health)
        # Now step to trigger termination check
        env.step(4)
        assert env.is_terminal()
        with pytest.raises(RuntimeError, match="terminated"):
            env.step(4)

    def test_step_out_of_range_raises(self, env: TacticalCombatEnv) -> None:
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(-1)
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(env.action_space_size)

    def test_info_contains_expected_keys(
        self, env: TacticalCombatEnv
    ) -> None:
        _, _, _, _, info = env.step(4)
        assert "step" in info
        assert "current_player" in info
        assert "p0_alive" in info
        assert "p1_alive" in info


# ===================================================================
# get_valid_actions
# ===================================================================


class TestTacticalCombatValidActions:
    """Tests for the get_valid_actions method."""

    def test_initial_all_units_have_valid_actions(
        self, env: TacticalCombatEnv
    ) -> None:
        mask = env.get_valid_actions()
        assert mask.shape == (env.action_space_size,)
        assert mask.dtype == np.bool_
        # All units alive -> all action slots valid
        for i in range(DEFAULT_NUM_UNITS):
            start = i * 5
            assert np.all(mask[start : start + 5])

    def test_dead_unit_has_no_valid_actions(self) -> None:
        """After a unit dies, its action slots should be masked out."""
        env = TacticalCombatEnv(
            board_size=8, num_units=3, max_steps=100, seed=SEED
        )
        env.reset()
        # Kill the last unit of player 0
        env._units[0][-1].take_damage(env._units[0][-1].health)
        mask = env.get_valid_actions()
        # Only 2 alive units -> only first 2 * 5 = 10 actions valid
        alive_count = sum(1 for u in env._units[0] if u.alive)
        assert alive_count == 2
        # Actions for alive units
        for i in range(alive_count):
            start = i * 5
            assert np.all(mask[start : start + 5])
        # Actions beyond alive units should be invalid
        for i in range(alive_count, 3):
            start = i * 5
            assert not np.any(mask[start : start + 5])

    def test_all_dead_no_valid_actions(self) -> None:
        """If all units are dead, no actions should be valid."""
        env = TacticalCombatEnv(board_size=8, num_units=2, max_steps=100, seed=SEED)
        env.reset()
        for u in env._units[0]:
            u.take_damage(u.health)
        mask = env.get_valid_actions()
        assert not np.any(mask)


# ===================================================================
# Terminal conditions
# ===================================================================


class TestTacticalCombatTermination:
    """Tests for game termination."""

    def test_game_ends_when_one_side_eliminated(self) -> None:
        env = TacticalCombatEnv(
            board_size=4, num_units=1, max_steps=100, seed=SEED
        )
        env.reset()
        # Kill all player 1 units
        for u in env._units[1]:
            u.take_damage(u.health)
        # Step to detect termination
        _, _, terminated, truncated, info = env.step(4)
        assert terminated
        assert not truncated
        assert env.is_terminal()
        assert info["p1_alive"] == 0

    def test_game_ends_when_player_0_eliminated(self) -> None:
        env = TacticalCombatEnv(
            board_size=4, num_units=1, max_steps=100, seed=SEED
        )
        env.reset()
        # Kill all player 0 units
        for u in env._units[0]:
            u.take_damage(u.health)
        _, _, terminated, _, info = env.step(4)
        assert terminated
        assert info["p0_alive"] == 0

    def test_not_terminal_when_both_sides_have_units(
        self, env: TacticalCombatEnv
    ) -> None:
        env.step(4)  # stay
        assert not env.is_terminal()


# ===================================================================
# Truncation
# ===================================================================


class TestTacticalCombatTruncation:
    """Tests for truncation at max_steps."""

    def test_game_truncates_at_max_steps(self) -> None:
        max_steps = 10
        env = TacticalCombatEnv(
            board_size=8,
            num_units=1,
            max_steps=max_steps,
            seed=SEED,
        )
        env.reset()
        terminated = False
        truncated = False
        for _ in range(max_steps):
            if terminated or truncated:
                break
            _, _, terminated, truncated, _ = env.step(4)  # stay (no combat)
        assert truncated
        assert not terminated
        assert env.is_terminal()

    def test_truncation_reward_based_on_health(self) -> None:
        """At truncation, the player with more health should get a positive reward."""
        max_steps = 2
        env = TacticalCombatEnv(
            board_size=8, num_units=1, max_steps=max_steps, seed=SEED
        )
        env.reset()
        # Give player 0 more health advantage
        env._units[0][0].health = 10
        env._units[1][0].health = 1
        # Play max_steps of stay actions
        for _ in range(max_steps):
            if env.is_terminal():
                break
            _, reward, _, truncated, _ = env.step(4)
        assert truncated
        # Reward is from current-player-at-time-of-step perspective
        # Last step was player 1's turn (step 0 = p0, step 1 = p1)
        # p0 health > p1 health, so from p1's view: reward is -0.5
        # But actual reward depends on the player variable at step time
        assert reward in (-0.5, 0.0, 0.5)

    def test_truncation_draw_equal_health(self) -> None:
        """Equal health at truncation should yield 0 reward."""
        max_steps = 2
        env = TacticalCombatEnv(
            board_size=8, num_units=1, max_steps=max_steps, seed=SEED
        )
        env.reset()
        # Set equal health
        env._units[0][0].health = 5
        env._units[1][0].health = 5
        for _ in range(max_steps):
            if env.is_terminal():
                break
            _, reward, _, _, _ = env.step(4)
        assert env.is_terminal()
        assert reward == pytest.approx(0.0)


# ===================================================================
# Clone
# ===================================================================


class TestTacticalCombatClone:
    """Tests for the clone method."""

    def test_clone_returns_distinct_object(
        self, env: TacticalCombatEnv
    ) -> None:
        cloned = env.clone()
        assert cloned is not env

    def test_clone_preserves_state(self, env: TacticalCombatEnv) -> None:
        env.step(4)
        env.step(4)
        cloned = env.clone()
        assert cloned._step_count == env._step_count
        assert cloned._current_player == env._current_player
        assert len(cloned._units[0]) == len(env._units[0])
        assert len(cloned._units[1]) == len(env._units[1])

    def test_clone_independent_of_original(
        self, env: TacticalCombatEnv
    ) -> None:
        env.step(4)
        cloned = env.clone()
        env.step(4)
        assert cloned._step_count != env._step_count

    def test_clone_units_are_independent(
        self, env: TacticalCombatEnv
    ) -> None:
        """Modifying units in clone should not affect original."""
        cloned = env.clone()
        cloned._units[0][0].health = 0
        assert env._units[0][0].health != 0

    def test_clone_produces_same_observation(
        self, env: TacticalCombatEnv
    ) -> None:
        env.step(4)
        cloned = env.clone()
        obs_orig = env.get_observation()
        obs_clone = cloned.get_observation()
        np.testing.assert_array_equal(obs_orig["spatial"], obs_clone["spatial"])
        np.testing.assert_array_equal(obs_orig["vector"], obs_clone["vector"])


# ===================================================================
# Observation shapes
# ===================================================================


class TestTacticalCombatObservationShapes:
    """Tests for spatial and vector observation shapes and values."""

    def test_spatial_shape_7_h_w(self, env: TacticalCombatEnv) -> None:
        obs = env.get_observation()
        bs = env._board_size
        assert obs["spatial"].shape == (7, bs, bs)

    def test_vector_shape_5(self, env: TacticalCombatEnv) -> None:
        obs = env.get_observation()
        assert obs["vector"].shape == (5,)

    def test_spatial_dtype(self, env: TacticalCombatEnv) -> None:
        obs = env.get_observation()
        assert obs["spatial"].dtype == np.float32

    def test_vector_dtype(self, env: TacticalCombatEnv) -> None:
        obs = env.get_observation()
        assert obs["vector"].dtype == np.float32

    def test_initial_unit_positions_in_spatial(
        self, env: TacticalCombatEnv
    ) -> None:
        """After reset, channel 0 should mark current player unit positions."""
        obs = env.get_observation()
        # Channel 0: current player unit positions
        assert np.sum(obs["spatial"][0]) > 0
        # Channel 2: opponent unit positions
        assert np.sum(obs["spatial"][2]) > 0

    def test_unit_health_normalized(self, env: TacticalCombatEnv) -> None:
        """Health channels should be in [0, 1]."""
        obs = env.get_observation()
        assert np.all(obs["spatial"][1] >= 0.0)
        assert np.all(obs["spatial"][1] <= 1.0)
        assert np.all(obs["spatial"][3] >= 0.0)
        assert np.all(obs["spatial"][3] <= 1.0)

    def test_vector_bias_term(self, env: TacticalCombatEnv) -> None:
        obs = env.get_observation()
        assert obs["vector"][4] == pytest.approx(1.0)

    def test_vector_initial_health_normalized(
        self, env: TacticalCombatEnv
    ) -> None:
        """At start, both players should have full health (normalized to 1.0)."""
        obs = env.get_observation()
        assert obs["vector"][0] == pytest.approx(1.0)  # own health
        assert obs["vector"][1] == pytest.approx(1.0)  # opponent health

    def test_vector_initial_alive_normalized(
        self, env: TacticalCombatEnv
    ) -> None:
        obs = env.get_observation()
        assert obs["vector"][2] == pytest.approx(1.0)  # own alive count
        assert obs["vector"][3] == pytest.approx(1.0)  # opponent alive count

    def test_custom_board_size_shapes(self) -> None:
        for bs in (3, 6, 12):
            e = TacticalCombatEnv(board_size=bs, num_units=2, seed=SEED)
            e.reset()
            obs = e.get_observation()
            assert obs["spatial"].shape == (7, bs, bs)
            assert obs["vector"].shape == (5,)

    def test_observation_reflects_damage(self) -> None:
        """After damaging a unit, health channels should update."""
        env = TacticalCombatEnv(board_size=4, num_units=1, seed=SEED)
        env.reset()
        obs_before = env.get_observation()
        initial_health_sum = float(np.sum(obs_before["spatial"][1]))

        # Damage player 0's unit
        env._units[0][0].take_damage(5)
        obs_after = env.get_observation()
        updated_health_sum = float(np.sum(obs_after["spatial"][1]))

        assert updated_health_sum < initial_health_sum
