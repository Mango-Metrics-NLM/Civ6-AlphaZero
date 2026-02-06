"""Unit tests for action validation and mask enforcement.

Tests the bug fixes for the 1-turn game termination issue:
1. Action mask enforcement in MCTS select_action
2. Action validation in MicroCivActionSpace
3. Execute action success/failure returns
"""

import numpy as np
import pytest

from alpha_zero.core.civ6_actions import MicroCivActionSpace
from alpha_zero.core.civ6_mcts import SimpleMCTS
from alpha_zero.envs.microciv import MicroCivEnv


class TestActionMaskEnforcement:
    """Test that MCTS select_action respects action masks."""

    def test_select_action_enforces_mask_simple_mcts(self):
        """SimpleMCTS should only select valid actions when mask provided."""
        mcts = SimpleMCTS(num_simulations=10)

        # Create probs that favor invalid action
        action_probs = np.zeros(100, dtype=np.float32)
        action_probs[50] = 1.0  # All probability on action 50

        # But mask says only action 0 is valid
        action_mask = np.zeros(100, dtype=np.float32)
        action_mask[0] = 1.0

        # With mask, must select action 0
        action = mcts.select_action(action_probs, temperature=0.0, action_mask=action_mask)
        assert action == 0, f"Expected action 0 (only valid), got {action}"

    def test_select_action_enforces_mask_greedy(self):
        """Greedy selection should respect mask."""
        mcts = SimpleMCTS(num_simulations=10)

        action_probs = np.zeros(100, dtype=np.float32)
        action_probs[99] = 0.9
        action_probs[0] = 0.1

        # Only action 0 is valid
        action_mask = np.zeros(100, dtype=np.float32)
        action_mask[0] = 1.0

        action = mcts.select_action(action_probs, temperature=0.0, action_mask=action_mask)
        assert action == 0

    def test_select_action_enforces_mask_stochastic(self):
        """Stochastic selection should only sample from valid actions."""
        mcts = SimpleMCTS(num_simulations=10)

        action_probs = np.ones(100, dtype=np.float32) / 100  # Uniform

        # Only actions 10-15 are valid
        action_mask = np.zeros(100, dtype=np.float32)
        action_mask[10:16] = 1.0

        # Run many times to verify
        for _ in range(100):
            action = mcts.select_action(action_probs, temperature=1.0, action_mask=action_mask)
            assert 10 <= action <= 15, f"Selected invalid action {action}"

    def test_select_action_fallback_first_valid(self):
        """When all probs are 0, should fallback to first valid action."""
        mcts = SimpleMCTS(num_simulations=10)

        action_probs = np.zeros(100, dtype=np.float32)  # All zero

        action_mask = np.zeros(100, dtype=np.float32)
        action_mask[42] = 1.0  # Only action 42 is valid

        action = mcts.select_action(action_probs, temperature=0.0, action_mask=action_mask)
        assert action == 42

    def test_select_action_no_mask_backwards_compatible(self):
        """Without mask, should work as before."""
        mcts = SimpleMCTS(num_simulations=10)

        action_probs = np.zeros(100, dtype=np.float32)
        action_probs[50] = 1.0

        # No mask provided
        action = mcts.select_action(action_probs, temperature=0.0)
        assert action == 50

    def test_select_action_raises_no_valid_actions(self):
        """Should raise when mask has no valid actions."""
        mcts = SimpleMCTS(num_simulations=10)

        action_probs = np.ones(100, dtype=np.float32) / 100
        action_mask = np.zeros(100, dtype=np.float32)  # No valid actions

        with pytest.raises(ValueError, match="No valid actions"):
            mcts.select_action(action_probs, temperature=1.0, action_mask=action_mask)


class TestActionValidation:
    """Test action validation in MicroCivActionSpace."""

    @pytest.fixture
    def action_space(self):
        return MicroCivActionSpace(board_size=8, max_units=5, max_cities=3)

    def test_validate_end_turn_always_valid(self, action_space):
        """End turn action should always be valid."""
        game_state = {"current_player": 0, "units": [], "cities": []}

        is_valid, error = action_space.validate_action(action_space.end_turn_offset, game_state)
        assert is_valid
        assert error == ""

    def test_validate_move_invalid_unit_idx(self, action_space):
        """Should detect invalid unit indices."""
        game_state = {
            "current_player": 0,
            "units": [{"player": 0, "position": (0, 0)}],  # Only 1 unit
            "cities": [],
        }

        # Action for unit_idx=3 (doesn't exist)
        action_idx = action_space.move_offset + 3 * 8 + 0  # Move unit 3 north
        is_valid, error = action_space.validate_action(action_idx, game_state)

        assert not is_valid
        assert "unit_idx 3" in error

    def test_validate_move_valid_unit_idx(self, action_space):
        """Should accept valid unit indices."""
        game_state = {
            "current_player": 0,
            "units": [
                {"player": 0, "position": (0, 0)},
                {"player": 0, "position": (1, 1)},
            ],
            "cities": [],
        }

        # Action for unit_idx=1 (exists)
        action_idx = action_space.move_offset + 1 * 8 + 0
        is_valid, error = action_space.validate_action(action_idx, game_state)

        assert is_valid
        assert error == ""

    def test_validate_ignores_enemy_units(self, action_space):
        """Should only count current player's units."""
        game_state = {
            "current_player": 0,
            "units": [
                {"player": 1, "position": (0, 0)},  # Enemy unit
                {"player": 0, "position": (1, 1)},  # Own unit (idx 0)
            ],
            "cities": [],
        }

        # Action for unit_idx=1 (only 1 own unit exists)
        action_idx = action_space.move_offset + 1 * 8 + 0
        is_valid, error = action_space.validate_action(action_idx, game_state)

        assert not is_valid
        assert "unit_idx 1" in error

    def test_validate_produce_invalid_city_idx(self, action_space):
        """Should detect invalid city indices."""
        game_state = {
            "current_player": 0,
            "units": [],
            "cities": [{"player": 0, "position": (2, 2)}],  # Only 1 city
        }

        # Action for city_idx=2 (doesn't exist)
        action_idx = action_space.produce_offset + 2 * 3 + 0  # Produce warrior at city 2
        is_valid, error = action_space.validate_action(action_idx, game_state)

        assert not is_valid
        assert "city_idx 2" in error


class TestExecuteActionReturns:
    """Test that _execute_action returns success/failure info."""

    @pytest.fixture
    def env(self):
        env = MicroCivEnv(board_size=8, max_turns=50)
        env.reset()
        return env

    def test_execute_action_returns_tuple(self, env):
        """Should return (reward, success, error) tuple."""
        action_dict = {"type": "end_turn"}
        result = env._execute_action(action_dict)

        assert isinstance(result, tuple)
        assert len(result) == 3
        reward, success, error = result
        assert isinstance(reward, float)
        assert isinstance(success, bool)
        assert isinstance(error, str)

    def test_execute_action_end_turn_success(self, env):
        """End turn should always succeed."""
        action_dict = {"type": "end_turn"}
        _reward, success, error = env._execute_action(action_dict)

        assert success
        assert error == ""

    def test_execute_action_invalid_unit_idx(self, env):
        """Should fail for invalid unit index."""
        action_dict = {"type": "move", "unit_idx": 99, "direction": (1, 0)}
        _reward, success, error = env._execute_action(action_dict)

        assert not success
        assert "Invalid unit_idx" in error

    def test_execute_action_move_success(self, env):
        """Valid move should succeed."""
        # Find a unit that can move
        own_units = [u for u in env.units if u.player == env.current_player]
        if own_units and own_units[0].moves_remaining > 0:
            unit = own_units[0]
            # Try moving in a valid direction, using env's _is_valid_move for consistency
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                new_x = unit.position[0] + dx
                new_y = unit.position[1] + dy
                new_pos = (new_x, new_y)
                # Use env's validation to ensure we find a truly valid move
                if env._is_valid_move(new_pos):
                    action_dict = {"type": "move", "unit_idx": 0, "direction": (dx, dy)}
                    _reward, success, error = env._execute_action(action_dict)
                    assert success, f"Move should succeed: {error}"
                    return
        # If no valid move found (unit may be surrounded), test passes silently
        # The important thing is that valid moves succeed when available

    def test_step_includes_action_info(self, env):
        """Step should include action_success and action_error in info."""
        action = 0  # End turn action

        _obs, _reward, _terminated, _truncated, info = env.step(action)

        assert "action_success" in info
        assert "action_error" in info

    def test_execute_attack_empty_tile_fails(self, env):
        """Attack on tile with no defender should fail with descriptive error.

        This tests the fix for the attack validation bug where attacking an
        empty tile would return success=True with reward=0, making the
        action_success unreliable.
        """
        # Find a unit with moves remaining
        own_units = [u for u in env.units if u.player == env.current_player]
        if not own_units:
            pytest.skip("No units available for test")

        unit = own_units[0]
        if unit.moves_remaining <= 0:
            pytest.skip("Unit has no moves remaining")

        # Find a direction with no enemy unit
        enemy_positions = {u.position for u in env.units if u.player != env.current_player}

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            target_pos = (unit.position[0] + dx, unit.position[1] + dy)
            if target_pos not in enemy_positions:
                # This tile has no enemy - attack should fail
                action_dict = {"type": "attack", "unit_idx": 0, "direction": (dx, dy)}
                reward, success, error = env._execute_action(action_dict)

                assert not success, "Attack on empty tile should fail"
                assert "No enemy unit" in error, f"Expected 'No enemy unit' in error, got: {error}"
                assert reward == 0.0, "Attack on empty tile should have zero reward"
                return

        # If all adjacent tiles have enemies, skip (rare but possible)
        pytest.skip("All adjacent tiles have enemy units")


class TestIntegrationMCTSMasking:
    """Integration tests for MCTS with action masking."""

    def test_mcts_search_returns_masked_probs(self):
        """MCTS search should only assign probability to valid actions."""
        env = MicroCivEnv(board_size=8, max_turns=50)
        env.reset()

        mcts = SimpleMCTS(num_simulations=20)
        action_probs, _ = mcts.search(env)

        # Get action mask
        obs = env.get_observation()
        mask = obs["action_mask"]

        # Check that invalid actions have ~0 probability
        for i in range(len(action_probs)):
            if mask[i] == 0:
                assert action_probs[i] < 0.01, f"Invalid action {i} has prob {action_probs[i]}"

    def test_full_game_no_invalid_actions(self):
        """Full game should never select invalid actions."""
        env = MicroCivEnv(board_size=8, max_turns=20)
        mcts = SimpleMCTS(num_simulations=10)

        obs, _ = env.reset()
        invalid_count = 0

        for _ in range(100):  # Safety limit
            action_probs, _ = mcts.search(env)
            action = mcts.select_action(
                action_probs, temperature=1.0, action_mask=obs["action_mask"]
            )

            # Verify action is valid
            if obs["action_mask"][action] == 0:
                invalid_count += 1

            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        assert invalid_count == 0, f"Selected {invalid_count} invalid actions"
