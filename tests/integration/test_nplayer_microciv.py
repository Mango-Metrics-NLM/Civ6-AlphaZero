"""Integration tests for N-player MicroCiv foundation.

Validates the `num_players` parameterization in `MicroCivEnv`:
  - Backwards compatibility (default 2-player behavior)
  - N-player construction (3, 4, 6 players)
  - Turn cycling for N players
  - Victory conditions (last player standing)
  - Truncation tiebreak across N players
  - Encoder compatibility with N-player games

Run with:
    pytest tests/integration/test_nplayer_microciv.py -v --timeout=30
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.envs.microciv import MicroCivEnv


_BOARD_SIZE = 8
_MAX_TURNS = 30


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def env_2p():
    """Default 2-player environment."""
    return MicroCivEnv(board_size=_BOARD_SIZE, max_turns=_MAX_TURNS, seed=42)


@pytest.fixture(params=[3, 4, 6])
def env_np(request):
    """N-player environment (parametrized)."""
    return MicroCivEnv(
        board_size=_BOARD_SIZE,
        max_turns=_MAX_TURNS,
        seed=42,
        num_players=request.param,
    )


# ============================================================================
# Test: Backwards Compatibility
# ============================================================================


class TestBackwardsCompat:
    """Default MicroCivEnv should behave identically to pre-N-player version."""

    def test_default_is_2_players(self, env_2p):
        """Default num_players should be 2."""
        assert env_2p.num_players == 2

    def test_reset_produces_4_units(self, env_2p):
        """2 players × 2 units each = 4 starting units."""
        env_2p.reset()
        assert len(env_2p.units) == 4

    def test_players_are_0_and_1(self, env_2p):
        """Starting units belong to players 0 and 1."""
        env_2p.reset()
        players = {u.player for u in env_2p.units}
        assert players == {0, 1}

    def test_turn_cycles_between_0_and_1(self, env_2p):
        """Turn cycling should alternate between 0 and 1."""
        env_2p.reset()
        assert env_2p.current_player == 0
        env_2p._end_turn()
        assert env_2p.current_player == 1
        env_2p._end_turn()
        assert env_2p.current_player == 0

    def test_info_has_2_player_lists(self, env_2p):
        """Info dict should have lists of length 2."""
        env_2p.reset()
        info = env_2p.get_info()
        assert len(info["num_units"]) == 2
        assert len(info["num_cities"]) == 2

    def test_observation_shape(self, env_2p):
        """Observation shape unchanged for 2 players."""
        obs, _ = env_2p.reset()
        from alpha_zero.envs.civ6_encoder import MicroCivEncoder

        assert obs["spatial"].shape == (MicroCivEncoder.TOTAL_CHANNELS, _BOARD_SIZE, _BOARD_SIZE)
        assert obs["vector"].shape == (64,)


# ============================================================================
# Test: N-Player Construction
# ============================================================================


class TestNPlayerConstruction:
    """Tests for constructing N-player environments."""

    def test_num_players_stored(self, env_np):
        """num_players attribute should match constructor arg."""
        assert env_np.num_players in {3, 4, 6}

    def test_starting_units_count(self, env_np):
        """Each player gets 2 starting units (settler + warrior)."""
        env_np.reset()
        expected_units = env_np.num_players * 2
        assert len(env_np.units) == expected_units

    def test_all_players_have_units(self, env_np):
        """Every player should have exactly 2 starting units."""
        env_np.reset()
        for p in range(env_np.num_players):
            player_units = [u for u in env_np.units if u.player == p]
            assert len(player_units) == 2, f"Player {p} has {len(player_units)} units, expected 2"

    def test_unit_types_per_player(self, env_np):
        """Each player starts with 1 settler + 1 warrior."""
        env_np.reset()
        for p in range(env_np.num_players):
            player_units = [u for u in env_np.units if u.player == p]
            types = sorted(u.unit_type for u in player_units)
            assert types == ["settler", "warrior"], f"Player {p}: {types}"

    def test_info_lists_match_num_players(self, env_np):
        """Info dict lists should have length num_players."""
        env_np.reset()
        info = env_np.get_info()
        assert len(info["num_units"]) == env_np.num_players
        assert len(info["num_cities"]) == env_np.num_players
        assert info["num_players"] == env_np.num_players

    def test_invalid_num_players_raises(self):
        """Invalid num_players should raise ValueError.

        MicroCivEnv accepts ``2 <= num_players <= GAME.MAX_PLAYERS``; assert
        the boundary using the constant so this test stays correct if the
        cap moves.
        """
        from alpha_zero.utils.constants import GAME

        with pytest.raises(ValueError):
            MicroCivEnv(board_size=_BOARD_SIZE, num_players=1)
        with pytest.raises(ValueError):
            MicroCivEnv(board_size=_BOARD_SIZE, num_players=GAME.MAX_PLAYERS + 1)
        with pytest.raises(ValueError):
            MicroCivEnv(board_size=_BOARD_SIZE, num_players=0)


# ============================================================================
# Test: N-Player Turn Cycling
# ============================================================================


class TestNPlayerTurnCycling:
    """Tests for turn cycling with N players."""

    def test_turn_cycles_through_all_players(self, env_np):
        """current_player should cycle 0→1→...→(N-1)→0."""
        env_np.reset()
        n = env_np.num_players

        for cycle in range(2):  # 2 full cycles
            for expected_player in range(n):
                assert env_np.current_player == expected_player, (
                    f"Cycle {cycle}, expected player {expected_player}, "
                    f"got {env_np.current_player}"
                )
                env_np._end_turn()

    def test_turn_increments_after_last_player(self, env_np):
        """Turn counter increments only when cycling back to player 0."""
        env_np.reset()
        n = env_np.num_players
        initial_turn = env_np.turn

        # Cycle through all players
        for _ in range(n):
            env_np._end_turn()

        assert env_np.turn == initial_turn + 1
        assert env_np.current_player == 0

    def test_turn_does_not_increment_mid_round(self, env_np):
        """Turn should NOT increment before all players have acted."""
        env_np.reset()
        initial_turn = env_np.turn

        # Only cycle through N-1 players (not the last)
        for _ in range(env_np.num_players - 1):
            env_np._end_turn()

        assert env_np.turn == initial_turn  # Not yet incremented


# ============================================================================
# Test: N-Player Victory
# ============================================================================


class TestNPlayerVictory:
    """Tests for N-player victory conditions."""

    def test_last_player_standing_wins(self):
        """When all but one player is eliminated, that player wins."""
        env = MicroCivEnv(board_size=_BOARD_SIZE, max_turns=_MAX_TURNS, num_players=4, seed=42)
        env.reset()

        # Remove all units/cities for players 1, 2, 3
        env.units = [u for u in env.units if u.player == 0]
        env.cities = [c for c in env.cities if c.player == 0]

        assert env._check_victory() is True
        assert env.winner == 0

    def test_two_remaining_no_victory(self):
        """Game continues while 2+ players remain."""
        env = MicroCivEnv(board_size=_BOARD_SIZE, max_turns=_MAX_TURNS, num_players=4, seed=42)
        env.reset()

        # Remove only player 2 and 3's units
        env.units = [u for u in env.units if u.player in (0, 1)]

        assert env._check_victory() is False

    def test_all_eliminated_none_winner(self):
        """If all players eliminated, winner is None."""
        env = MicroCivEnv(board_size=_BOARD_SIZE, max_turns=_MAX_TURNS, num_players=3, seed=42)
        env.reset()

        env.units = []
        env.cities = []

        assert env._check_victory() is True
        assert env.winner is None


# ============================================================================
# Test: N-Player Truncation Tiebreak
# ============================================================================


class TestNPlayerTruncation:
    """Tests for truncation tiebreak with N players."""

    def test_most_cities_wins(self):
        """Player with most cities wins on truncation."""
        from alpha_zero.envs.microciv import City

        env = MicroCivEnv(board_size=_BOARD_SIZE, max_turns=5, num_players=3, seed=42)
        env.reset()
        env.turn = 5  # Force truncation

        # Give player 2 the most cities
        env.cities = [
            City(id=0, player=0, name="A", position=(1, 1)),
            City(id=1, player=2, name="B", position=(3, 3)),
            City(id=2, player=2, name="C", position=(5, 5)),
        ]

        obs, reward, terminated, truncated, info = env.step(0)
        assert env.winner == 2

    def test_tie_gives_none(self):
        """Equal city counts produce None winner."""
        from alpha_zero.envs.microciv import City

        env = MicroCivEnv(board_size=_BOARD_SIZE, max_turns=5, num_players=3, seed=42)
        env.reset()
        env.turn = 5

        env.cities = [
            City(id=0, player=0, name="A", position=(1, 1)),
            City(id=1, player=1, name="B", position=(3, 3)),
            City(id=2, player=2, name="C", position=(5, 5)),
        ]

        obs, reward, terminated, truncated, info = env.step(0)
        assert env.winner is None


# ============================================================================
# Test: Encoder Compatibility
# ============================================================================


class TestNPlayerEncoder:
    """Tests that encoder produces correct shapes for N-player games."""

    def test_spatial_shape_unchanged(self, env_np):
        """Spatial observation shape is independent of num_players."""
        obs, _ = env_np.reset()
        from alpha_zero.envs.civ6_encoder import MicroCivEncoder

        assert obs["spatial"].shape == (MicroCivEncoder.TOTAL_CHANNELS, _BOARD_SIZE, _BOARD_SIZE)

    def test_vector_shape_unchanged(self, env_np):
        """Vector observation shape is independent of num_players."""
        obs, _ = env_np.reset()
        assert obs["vector"].shape == (64,)

    def test_action_mask_shape_unchanged(self, env_np):
        """Action mask shape is independent of num_players."""
        obs, _ = env_np.reset()
        assert obs["action_mask"].shape == (env_np.action_space.n,)

    def test_encoder_distinguishes_own_vs_enemy(self, env_np):
        """Encoder should place current player's units in 'own' channels."""
        env_np.reset()
        obs = env_np.get_observation()
        spatial = obs["spatial"]

        # Own unit channels (3-5) should have some nonzero values
        own_unit_channels = spatial[3:6]
        assert np.any(own_unit_channels > 0), "No own units encoded"

    def test_game_state_has_correct_num_players(self, env_np):
        """Game state dict should report actual num_players."""
        env_np.reset()
        state = env_np._get_game_state()
        assert state["num_players"] == env_np.num_players
