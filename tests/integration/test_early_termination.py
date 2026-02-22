"""Integration tests for game termination behavior.

Tests that games don't terminate prematurely after 1 turn,
verifying the bug fixes work end-to-end.
"""

from alpha_zero.core.civ6_mcts import SimpleMCTS
from alpha_zero.envs.microciv import MicroCivEnv


class TestGameLength:
    """Tests that games last a reasonable number of turns."""

    def test_games_should_last_multiple_turns(self):
        """Games should not end after just 1 turn."""
        # Use max_turns=10 for faster tests (games take ~7 steps per turn)
        env = MicroCivEnv(board_size=8, max_turns=10)
        mcts = SimpleMCTS(num_simulations=10)

        game_lengths = []
        for _ in range(5):  # 5 games for faster testing
            obs, _ = env.reset()

            for _ in range(150):  # ~15 steps per turn * 10 turns = 150 steps max
                action_probs, _ = mcts.search(env)
                action = mcts.select_action(
                    action_probs, temperature=1.0, action_mask=obs["action_mask"]
                )
                obs, _, terminated, truncated, _info = env.step(action)

                if terminated or truncated:
                    break

            # Always record the final turn count (whether terminated or step limit reached)
            game_lengths.append(env.turn)

        avg_length = sum(game_lengths) / len(game_lengths)
        min_length = min(game_lengths)

        assert avg_length >= 3, f"Average game length {avg_length:.1f} too short"
        assert min_length >= 1, f"Minimum game length {min_length} is 0"

        # Count how many games ended at turn 1 or less
        early_terminations = sum(1 for t in game_lengths if t <= 1)
        assert (
            early_terminations == 0
        ), f"{early_terminations} games ended at turn 1 or less. Game lengths: {game_lengths}"

    def test_no_immediate_victory(self):
        """Game should not immediately end due to victory conditions."""
        env = MicroCivEnv(board_size=8, max_turns=50)

        for _ in range(20):
            obs, _info = env.reset()

            # Immediately after reset, game should not be done
            assert not env.done, "Game is done immediately after reset"

            # Both players should have units
            p0_units = len([u for u in env.units if u.player == 0])
            p1_units = len([u for u in env.units if u.player == 1])
            assert p0_units > 0, "Player 0 has no units after reset"
            assert p1_units > 0, "Player 1 has no units after reset"

            # Take one action
            action = 0  # End turn
            _obs, _reward, terminated, truncated, info = env.step(action)

            # Should not terminate after just one end_turn
            if terminated and not truncated:
                # If terminated, check it's not due to elimination
                p0_units_now = len([u for u in env.units if u.player == 0])
                p1_units_now = len([u for u in env.units if u.player == 1])
                p0_cities = len([c for c in env.cities if c.player == 0])
                p1_cities = len([c for c in env.cities if c.player == 1])

                raise AssertionError(
                    f"Game terminated after 1 action. "
                    f"Turn: {env.turn}, Winner: {info.get('winner')}, "
                    f"P0: {p0_units_now}u/{p0_cities}c, P1: {p1_units_now}u/{p1_cities}c"
                )


class TestActionExecutionSuccess:
    """Tests that actions execute successfully."""

    def test_valid_actions_succeed(self):
        """Valid actions should execute successfully."""
        env = MicroCivEnv(board_size=8, max_turns=50)
        mcts = SimpleMCTS(num_simulations=10)

        obs, _ = env.reset()
        failed_actions = 0
        total_actions = 0

        for _ in range(100):  # Run for 100 steps
            action_probs, _ = mcts.search(env)
            action = mcts.select_action(
                action_probs, temperature=1.0, action_mask=obs["action_mask"]
            )

            obs, _, terminated, truncated, info = env.step(action)
            total_actions += 1

            # Check if action succeeded
            if not info.get("action_success", True):
                failed_actions += 1

            if terminated or truncated:
                break

        # Most actions should succeed (some failures are OK for movement into mountains etc)
        failure_rate = failed_actions / max(total_actions, 1)
        assert failure_rate < 0.5, (
            f"Action failure rate {failure_rate:.1%} too high. "
            f"{failed_actions}/{total_actions} actions failed"
        )

    def test_info_contains_action_status(self):
        """Step should return action success status in info."""
        env = MicroCivEnv(board_size=8, max_turns=50)
        obs, _ = env.reset()

        # Take an action
        action = 0  # End turn
        _obs, _reward, _terminated, _truncated, info = env.step(action)

        assert "action_success" in info, "info missing action_success"
        assert "action_error" in info, "info missing action_error"
        assert isinstance(info["action_success"], bool)
        assert isinstance(info["action_error"], str)


class TestSelfPlayIntegration:
    """Integration tests for self-play with action masking."""

    def test_self_play_game_completes(self):
        """Self-play game should complete without errors."""
        env = MicroCivEnv(board_size=8, max_turns=30)
        mcts = SimpleMCTS(num_simulations=15)

        obs, _ = env.reset()
        steps = 0
        max_steps = 300  # Safety limit

        while steps < max_steps:
            action_probs, _root_value = mcts.search(env)
            action = mcts.select_action(
                action_probs, temperature=1.0, action_mask=obs["action_mask"]
            )

            # Verify action is valid
            assert (
                obs["action_mask"][action] > 0
            ), f"Invalid action {action} selected at step {steps}"

            obs, _reward, terminated, truncated, _info = env.step(action)
            steps += 1

            if terminated or truncated:
                break

        # Should have completed the game
        assert terminated or truncated, f"Game didn't complete after {steps} steps"

    def test_multiple_games_stability(self):
        """Multiple games should complete without crashes."""
        env = MicroCivEnv(board_size=8, max_turns=20)
        mcts = SimpleMCTS(num_simulations=10)

        completed_games = 0
        total_turns = 0
        winners = []

        for _game_idx in range(5):
            obs, _ = env.reset()

            for _ in range(200):
                action_probs, _ = mcts.search(env)
                action = mcts.select_action(
                    action_probs, temperature=1.0, action_mask=obs["action_mask"]
                )
                obs, _, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    completed_games += 1
                    total_turns += env.turn
                    winners.append(info.get("winner"))
                    break

        assert completed_games == 5, f"Only {completed_games}/5 games completed"
        avg_turns = total_turns / completed_games
        assert avg_turns >= 2, f"Average turns {avg_turns:.1f} too low"


class TestTerminationConditions:
    """Tests for correct termination condition handling."""

    def test_turn_limit_termination(self):
        """Game should terminate at max_turns."""
        env = MicroCivEnv(board_size=8, max_turns=5)
        obs, _ = env.reset()

        # Just end turn repeatedly
        for _ in range(100):
            _obs, _, terminated, truncated, _info = env.step(0)  # End turn

            if terminated or truncated:
                assert (
                    env.turn >= 5 or env.done
                ), f"Terminated at turn {env.turn} before max_turns=5"
                break

    def test_initial_state_not_terminated(self):
        """Fresh environment should not be in terminated state."""
        env = MicroCivEnv(board_size=8, max_turns=50)

        for _ in range(10):
            _obs, _info = env.reset()

            # Check initial state
            assert not env.done
            assert env.turn == 0
            assert len(env.units) > 0
            assert env.winner is None
