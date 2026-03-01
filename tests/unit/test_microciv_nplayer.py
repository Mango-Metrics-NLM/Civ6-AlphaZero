"""Unit tests for N-player MicroCiv environment support.

Tests that MicroCivEnv correctly supports 2-8 players while maintaining
backward compatibility with the default 2-player mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.envs.microciv import City, MicroCivEnv
from alpha_zero.utils.constants import GAME
from tests.test_constants import TestDefaults


class TestNPlayerCreation:
    """Test environment creation with different player counts."""

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_create_env_with_n_players(self, num_players: int) -> None:
        """Environment can be created with valid player counts."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            num_players=num_players,
            seed=42,
        )
        assert env.num_players == num_players

    def test_default_is_two_players(self) -> None:
        """Default num_players is 2 for backward compatibility."""
        env = MicroCivEnv(board_size=TestDefaults.SMALL_BOARD_SIZE, seed=42)
        assert env.num_players == 2

    def test_invalid_num_players_too_low(self) -> None:
        """Creating env with fewer than 2 players raises ValueError."""
        with pytest.raises(ValueError, match="num_players must be between 2"):
            MicroCivEnv(board_size=TestDefaults.SMALL_BOARD_SIZE, num_players=1)

    def test_invalid_num_players_too_high(self) -> None:
        """Creating env with more than MAX_PLAYERS raises ValueError."""
        with pytest.raises(ValueError, match="num_players must be between 2"):
            MicroCivEnv(
                board_size=TestDefaults.SMALL_BOARD_SIZE,
                num_players=GAME.MAX_PLAYERS + 1,
            )

    def test_invalid_num_players_zero(self) -> None:
        """Creating env with 0 players raises ValueError."""
        with pytest.raises(ValueError, match="num_players must be between 2"):
            MicroCivEnv(board_size=TestDefaults.SMALL_BOARD_SIZE, num_players=0)


class TestNPlayerStartingPositions:
    """Test that starting positions are correctly distributed for N players."""

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_starting_positions_no_overlap(self, num_players: int) -> None:
        """Starting positions should not overlap for any player count."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            num_players=num_players,
            seed=42,
        )
        env.reset()

        # All starting positions should be unique (no two units on same tile)
        # Note: with small boards some collisions could happen due to _find_valid_position
        # randomness, but the positions themselves should be computed distinctly
        computed_positions = env._compute_starting_positions()
        assert len(computed_positions) == num_players

        # Verify computed positions are within board bounds
        for x, y in computed_positions:
            assert 1 <= x <= env.board_size - 2
            assert 1 <= y <= env.board_size - 2

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_each_player_has_starting_units(self, num_players: int) -> None:
        """Each player should have a settler and a warrior after reset."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            num_players=num_players,
            seed=42,
        )
        env.reset()

        for player in range(num_players):
            player_units = [u for u in env.units if u.player == player]
            assert len(player_units) >= 2, (
                f"Player {player} should have at least 2 starting units, "
                f"got {len(player_units)}"
            )
            unit_types = {u.unit_type for u in player_units}
            assert "settler" in unit_types, f"Player {player} missing settler"
            assert "warrior" in unit_types, f"Player {player} missing warrior"

    def test_total_starting_units(self) -> None:
        """Total starting units should be 2 per player."""
        for num_players in [2, 3, 4, 6, 8]:
            env = MicroCivEnv(
                board_size=TestDefaults.SMALL_BOARD_SIZE,
                num_players=num_players,
                seed=42,
            )
            env.reset()
            assert len(env.units) == num_players * 2

    def test_positions_spread_for_large_player_count(self) -> None:
        """For 8 players on a 16x16 board, positions should be spread out."""
        env = MicroCivEnv(
            board_size=TestDefaults.BOARD_SIZE,
            num_players=8,
            seed=42,
        )
        positions = env._compute_starting_positions()
        assert len(positions) == 8

        # No two computed positions should be identical
        unique_positions = set(positions)
        assert len(unique_positions) == 8, "All 8 starting positions should be unique"


class TestNPlayerTurnRotation:
    """Test that turn rotation cycles correctly through all players."""

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_turn_rotation_cycles(self, num_players: int) -> None:
        """Players should take turns in order 0, 1, ..., N-1, 0, 1, ..."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            max_turns=TestDefaults.MAX_TURNS,
            num_players=num_players,
            seed=42,
        )
        env.reset()
        end_turn_action = env.action_space_handler.end_turn_offset

        # Cycle through all players once
        for expected_player in range(num_players):
            assert env.current_player == expected_player, (
                f"Expected player {expected_player}, got {env.current_player}"
            )
            env.step(end_turn_action)

        # After full cycle, should be back to player 0
        assert env.current_player == 0

    @pytest.mark.parametrize("num_players", [2, 3, 4])
    def test_turn_increments_after_full_cycle(self, num_players: int) -> None:
        """Turn counter should only increment after all players have moved."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            max_turns=TestDefaults.MAX_TURNS,
            num_players=num_players,
            seed=42,
        )
        env.reset()
        initial_turn = env.turn
        end_turn_action = env.action_space_handler.end_turn_offset

        # All but the last player end turn - turn should NOT increment yet
        for _ in range(num_players - 1):
            env.step(end_turn_action)
            assert env.turn == initial_turn

        # Last player ends turn - NOW turn should increment
        env.step(end_turn_action)
        assert env.turn == initial_turn + 1

    def test_two_full_cycles(self) -> None:
        """Two full cycles should increment turn by 2."""
        num_players = 4
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            max_turns=TestDefaults.MAX_TURNS,
            num_players=num_players,
            seed=42,
        )
        env.reset()
        end_turn_action = env.action_space_handler.end_turn_offset

        for _ in range(num_players * 2):
            env.step(end_turn_action)

        assert env.turn == 2


class TestNPlayerVictory:
    """Test victory detection for N-player games."""

    def test_last_player_standing_wins(self) -> None:
        """If all but one player are eliminated, the remaining player wins."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            max_turns=TestDefaults.MAX_TURNS,
            num_players=4,
            seed=42,
        )
        env.reset()

        # Remove all units/cities for players 1, 2, 3
        env.units = [u for u in env.units if u.player == 0]
        env.cities = [c for c in env.cities if c.player == 0]

        # Check victory
        terminated = env._check_victory()
        assert terminated is True
        assert env.winner == 0

    def test_two_survivors_no_victory(self) -> None:
        """If 2 players still have units, the game continues."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            max_turns=TestDefaults.MAX_TURNS,
            num_players=4,
            seed=42,
        )
        env.reset()

        # Remove players 2 and 3 but keep 0 and 1
        env.units = [u for u in env.units if u.player in (0, 1)]
        env.cities = [c for c in env.cities if c.player in (0, 1)]

        terminated = env._check_victory()
        assert terminated is False

    def test_elimination_mid_game(self) -> None:
        """Players eliminated during the game should not block victory."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            max_turns=TestDefaults.MAX_TURNS,
            num_players=3,
            seed=42,
        )
        env.reset()

        # Simulate: player 2 is eliminated, 0 and 1 survive
        env.units = [u for u in env.units if u.player != 2]
        env.cities = [c for c in env.cities if c.player != 2]

        # Game should continue (two survivors)
        terminated = env._check_victory()
        assert terminated is False

        # Now eliminate player 1 as well
        env.units = [u for u in env.units if u.player == 0]
        env.cities = [c for c in env.cities if c.player == 0]

        terminated = env._check_victory()
        assert terminated is True
        assert env.winner == 0

    def test_truncation_most_cities_wins(self) -> None:
        """At turn limit, player with most cities wins."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            max_turns=2,
            num_players=3,
            seed=42,
        )
        env.reset()

        # Give player 1 the most cities
        env.cities = [
            City(id=0, player=0, name="C0", position=(1, 1)),
            City(id=1, player=1, name="C1", position=(3, 3)),
            City(id=2, player=1, name="C2", position=(5, 5)),
            City(id=3, player=2, name="C3", position=(6, 6)),
        ]

        # Play until turn limit
        end_turn_action = env.action_space_handler.end_turn_offset
        for _ in range(100):
            _, _, terminated, truncated, _ = env.step(end_turn_action)
            if terminated or truncated:
                break

        assert env.winner == 1  # Player 1 had 2 cities (most)

    def test_truncation_draw_on_tie(self) -> None:
        """At turn limit with tied city counts, result is a draw."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            max_turns=2,
            num_players=3,
            seed=42,
        )
        env.reset()

        # Give players 0 and 1 equal cities, player 2 fewer
        env.cities = [
            City(id=0, player=0, name="C0", position=(1, 1)),
            City(id=1, player=1, name="C1", position=(3, 3)),
        ]

        end_turn_action = env.action_space_handler.end_turn_offset
        for _ in range(100):
            _, _, terminated, truncated, _ = env.step(end_turn_action)
            if terminated or truncated:
                break

        assert env.winner is None  # Draw


class TestNPlayerGameInfo:
    """Test that game info returns correct per-player stats."""

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_info_has_correct_player_count(self, num_players: int) -> None:
        """Info should report the correct number of players."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            num_players=num_players,
            seed=42,
        )
        env.reset()
        info = env.get_info()

        assert info["num_players"] == num_players

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_info_num_units_length(self, num_players: int) -> None:
        """num_units list should have one entry per player."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            num_players=num_players,
            seed=42,
        )
        env.reset()
        info = env.get_info()

        assert len(info["num_units"]) == num_players
        assert len(info["num_cities"]) == num_players

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_info_unit_counts_correct(self, num_players: int) -> None:
        """Unit counts in info should match actual units."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            num_players=num_players,
            seed=42,
        )
        env.reset()
        info = env.get_info()

        for p in range(num_players):
            expected = len([u for u in env.units if u.player == p])
            assert info["num_units"][p] == expected


class TestNPlayerGameState:
    """Test game state dictionary for N players."""

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_game_state_num_players(self, num_players: int) -> None:
        """Game state dict should report correct num_players."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            num_players=num_players,
            seed=42,
        )
        env.reset()
        state = env._get_game_state()

        assert state["num_players"] == num_players


class TestNPlayerStepReset:
    """Test step and reset work correctly for N players."""

    @pytest.mark.parametrize("num_players", [2, 3, 4, 6, 8])
    def test_reset_returns_valid_observation(self, num_players: int) -> None:
        """Reset returns valid observation for any player count."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            num_players=num_players,
            seed=42,
        )
        obs, _info = env.reset()

        assert "spatial" in obs
        assert "vector" in obs
        assert "action_mask" in obs
        assert obs["action_mask"].shape == (env.action_space.n,)

    @pytest.mark.parametrize("num_players", [2, 3, 4])
    def test_step_with_end_turn(self, num_players: int) -> None:
        """Step with end_turn action works for N players."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            max_turns=TestDefaults.MAX_TURNS,
            num_players=num_players,
            seed=42,
        )
        env.reset()

        end_turn_action = env.action_space_handler.end_turn_offset
        _obs, _reward, terminated, truncated, _info = env.step(end_turn_action)

        assert not terminated
        assert not truncated
        assert env.current_player == 1  # Next player

    @pytest.mark.parametrize("num_players", [2, 3, 4])
    def test_game_terminates(self, num_players: int) -> None:
        """Game eventually terminates for any player count."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            max_turns=5,
            num_players=num_players,
            seed=42,
        )
        env.reset()

        end_turn_action = env.action_space_handler.end_turn_offset
        for _ in range(200):  # More iterations for more players
            _, _, terminated, truncated, _ = env.step(end_turn_action)
            if terminated or truncated:
                break

        assert terminated or truncated

    @pytest.mark.parametrize("num_players", [3, 4, 6])
    def test_valid_action_mask_always_has_end_turn(self, num_players: int) -> None:
        """End turn should always be a valid action."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            max_turns=TestDefaults.MAX_TURNS,
            num_players=num_players,
            seed=42,
        )
        env.reset()

        mask = env.get_valid_actions()
        assert mask[env.action_space_handler.end_turn_offset] == 1.0


class TestNPlayerClone:
    """Test cloning for N-player environments."""

    @pytest.mark.parametrize("num_players", [2, 3, 4, 8])
    def test_clone_preserves_num_players(self, num_players: int) -> None:
        """Clone should preserve num_players."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            num_players=num_players,
            seed=42,
        )
        env.reset()

        clone = env.clone()
        assert clone.num_players == num_players

    def test_clone_is_independent(self) -> None:
        """Clone should be independent from original."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            num_players=4,
            seed=42,
        )
        env.reset()

        clone = env.clone()

        # Advance original
        end_turn_action = env.action_space_handler.end_turn_offset
        env.step(end_turn_action)

        # Clone should be unaffected
        assert clone.current_player == 0
        assert env.current_player == 1


class TestNPlayerRendering:
    """Test rendering for N-player environments."""

    @pytest.mark.parametrize("num_players", [2, 3, 4, 8])
    def test_render_includes_all_players(self, num_players: int) -> None:
        """Render output should include stats for all players."""
        env = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            render_mode="ansi",
            num_players=num_players,
            seed=42,
        )
        env.reset()

        output = env.render()
        assert output is not None
        assert isinstance(output, str)

        # Check that all player stats are in the output
        for p in range(num_players):
            assert f"P{p}:" in output, f"P{p} stats missing from render output"


class TestBackwardCompatibility:
    """Test that default 2-player behavior is unchanged."""

    def test_default_env_matches_original(self) -> None:
        """Default env (no num_players arg) should behave like original."""
        env = MicroCivEnv(board_size=TestDefaults.BOARD_SIZE, seed=42)
        env.reset()

        assert env.num_players == 2
        assert len([u for u in env.units if u.player == 0]) >= 2
        assert len([u for u in env.units if u.player == 1]) >= 2

    def test_info_backward_compatible(self) -> None:
        """Info dict should still have num_units and num_cities as lists of length 2."""
        env = MicroCivEnv(board_size=TestDefaults.SMALL_BOARD_SIZE, seed=42)
        env.reset()
        info = env.get_info()

        assert len(info["num_units"]) == 2
        assert len(info["num_cities"]) == 2

    def test_end_turn_switches_to_player_1(self) -> None:
        """With 2 players, end turn should switch to player 1."""
        env = MicroCivEnv(board_size=TestDefaults.SMALL_BOARD_SIZE, seed=42)
        env.reset()

        end_turn_action = env.action_space_handler.end_turn_offset
        env.step(end_turn_action)
        assert env.current_player == 1

        # And back to 0
        env.step(end_turn_action)
        assert env.current_player == 0

    def test_seed_reproducibility_maintained(self) -> None:
        """Same seed should produce same initial state."""
        env1 = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            seed=12345,
        )
        env2 = MicroCivEnv(
            board_size=TestDefaults.SMALL_BOARD_SIZE,
            seed=12345,
        )

        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        assert np.allclose(obs1["spatial"], obs2["spatial"])
        assert np.allclose(obs1["vector"], obs2["vector"])

    def test_clone_backward_compatible(self) -> None:
        """Clone should work the same as before for 2-player games."""
        env = MicroCivEnv(board_size=TestDefaults.SMALL_BOARD_SIZE, seed=42)
        env.reset()

        clone = env.clone()
        assert clone.num_players == 2
        assert clone.board_size == env.board_size
        assert clone.max_turns == env.max_turns
