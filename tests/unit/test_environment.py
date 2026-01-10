"""Unit tests for MicroCiv environment."""

import numpy as np
import pytest

from alpha_zero.envs.microciv import MicroCivEnv


class TestMicroCivEnv:
    """Test suite for MicroCiv environment."""

    @pytest.fixture
    def env(self):
        return MicroCivEnv(board_size=16, max_turns=200, seed=42)

    def test_reset_returns_observation(self, env):
        """Reset returns valid observation."""
        obs, info = env.reset()

        assert "spatial" in obs
        assert "vector" in obs
        assert "action_mask" in obs

        assert obs["spatial"].shape == (13, 16, 16)
        assert obs["vector"].shape == (64,)
        assert obs["action_mask"].shape == (env.action_space.n,)

    def test_reset_creates_units(self, env):
        """Reset creates starting units for both players."""
        env.reset()

        p0_units = [u for u in env.units if u.player == 0]
        p1_units = [u for u in env.units if u.player == 1]

        # Each player should have settler + warrior
        assert len(p0_units) >= 2
        assert len(p1_units) >= 2

    def test_step_changes_state(self, env):
        """Taking an action changes game state."""
        obs1, _ = env.reset()

        # Take END_TURN action
        end_turn_idx = env.action_space_handler.end_turn_offset
        obs2, _, _, _, _ = env.step(end_turn_idx)

        # State should change (different player's turn)
        assert env.current_player == 1

    def test_valid_action_mask(self, env):
        """Action mask only allows valid actions."""
        env.reset()
        mask = env.get_valid_actions()

        # At least END_TURN should be valid
        assert mask[env.action_space_handler.end_turn_offset] == 1.0

        # Mask should be binary
        assert np.all((mask == 0) | (mask == 1))

    def test_game_terminates(self, env):
        """Game eventually terminates."""
        env.reset()

        for _ in range(1000):
            mask = env.get_valid_actions()
            valid = np.where(mask > 0)[0]
            action = np.random.choice(valid)
            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

        assert terminated or truncated

    def test_clone_creates_copy(self, env):
        """Clone creates independent copy."""
        env.reset()

        # Make some moves
        for _ in range(5):
            mask = env.get_valid_actions()
            valid = np.where(mask > 0)[0]
            action = np.random.choice(valid)
            env.step(action)

        # Clone
        clone = env.clone()

        # Modify original
        mask = env.get_valid_actions()
        valid = np.where(mask > 0)[0]
        action = np.random.choice(valid)
        env.step(action)

        # Clone should be unaffected
        assert clone.turn != env.turn or clone.current_player != env.current_player

    def test_unit_movement(self, env):
        """Units can move to adjacent tiles."""
        env.reset()

        # Find a unit with moves
        own_units = [u for u in env.units if u.player == 0 and u.moves_remaining > 0]
        if own_units:
            unit = own_units[0]
            original_pos = unit.position

            # Find valid move action
            mask = env.get_valid_actions()
            move_start = env.action_space_handler.move_offset

            for dir_idx in range(8):
                action_idx = move_start + dir_idx
                if mask[action_idx] == 1.0:
                    env.step(action_idx)
                    # Position should change
                    assert unit.position != original_pos or unit.moves_remaining < 2
                    break

    def test_combat_damage(self, env):
        """Combat deals damage to units."""
        # Create controlled state
        env.reset()
        env.units = []

        # Place units adjacent
        from alpha_zero.envs.microciv import Unit

        attacker = Unit(
            id=0, player=0, unit_type="warrior", position=(5, 5), health=100, strength=20
        )
        defender = Unit(
            id=1, player=1, unit_type="warrior", position=(6, 5), health=100, strength=20
        )
        env.units = [attacker, defender]
        env.current_player = 0

        # Attack
        attack_start = env.action_space_handler.attack_offset
        dir_idx = env.action_space_handler.directions.index((1, 0))
        action_idx = attack_start + 0 * 8 + dir_idx

        env.step(action_idx)

        # At least one unit should have taken damage
        total_health = attacker.health + defender.health
        assert total_health < 200

    def test_city_production(self, env):
        """Cities can produce units."""
        env.reset()

        # Add a city for player 0
        from alpha_zero.envs.microciv import City

        city = City(
            id=0,
            player=0,
            name="Test City",
            position=(5, 5),
            population=3,
            production=0,
            production_queue=None,
        )
        env.cities = [city]
        env.current_player = 0

        # Start production
        produce_start = env.action_space_handler.produce_offset
        action_idx = produce_start + 0  # warrior
        env.step(action_idx)

        assert city.production_queue == "warrior"

    def test_turn_increments(self, env):
        """Turn counter increments correctly."""
        env.reset()
        initial_turn = env.turn

        # End both players' turns
        env.step(env.action_space_handler.end_turn_offset)  # P0 end
        env.step(env.action_space_handler.end_turn_offset)  # P1 end

        assert env.turn == initial_turn + 1

    def test_render_ansi(self, env):
        """ANSI rendering produces output."""
        env = MicroCivEnv(board_size=8, render_mode="ansi")
        env.reset()
        output = env.render()

        assert output is not None
        assert isinstance(output, str)
        assert "Turn" in output

    def test_max_turns_truncation(self):
        """Game truncates at max turns."""
        env = MicroCivEnv(board_size=8, max_turns=10)
        env.reset()

        truncated = False
        for _ in range(100):
            action = env.action_space_handler.end_turn_offset
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        assert truncated or terminated

    def test_seed_reproducibility(self):
        """Same seed produces same initial state."""
        env1 = MicroCivEnv(seed=12345)
        env2 = MicroCivEnv(seed=12345)

        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        assert np.allclose(obs1["spatial"], obs2["spatial"])
        assert np.allclose(obs1["vector"], obs2["vector"])
