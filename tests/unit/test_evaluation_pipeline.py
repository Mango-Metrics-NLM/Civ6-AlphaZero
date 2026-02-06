"""Tests for the evaluation module."""

from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.evaluation.baseline_agents import (
    GreedyHeuristicAgent,
    RandomAgent,
)
from alpha_zero.evaluation.evaluation_pipeline import (
    EvaluationPipeline,
    GameResult,
    WinRateResult,
)


class TestRandomAgent:
    """Tests for RandomAgent."""

    def test_initialization(self):
        """Should initialize with optional seed."""
        agent = RandomAgent()
        assert agent is not None

        agent_seeded = RandomAgent(seed=42)
        assert agent_seeded is not None

    def test_select_action_returns_valid_action(self):
        """Should return a valid action from the mask."""
        env = MicroCivEnv(board_size=8, max_turns=50)
        agent = RandomAgent(seed=42)
        obs, _ = env.reset()

        action = agent.select_action(env, obs)

        assert obs["action_mask"][action] > 0

    def test_select_action_deterministic_with_seed(self):
        """Same seed should produce same actions."""
        env = MicroCivEnv(board_size=8, max_turns=50, seed=123)
        obs, _ = env.reset()

        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=42)

        action1 = agent1.select_action(env, obs)
        action2 = agent2.select_action(env, obs)

        assert action1 == action2

    def test_reset_is_no_op(self):
        """Reset should not raise errors."""
        agent = RandomAgent()
        agent.reset()  # Should not raise


class TestGreedyHeuristicAgent:
    """Tests for GreedyHeuristicAgent."""

    def test_initialization(self):
        """Should initialize with default and custom parameters."""
        agent = GreedyHeuristicAgent()
        assert agent.aggression == 0.8
        assert agent.expansion_priority == 0.6

        custom_agent = GreedyHeuristicAgent(aggression=0.5, expansion_priority=0.9)
        assert custom_agent.aggression == 0.5
        assert custom_agent.expansion_priority == 0.9

    def test_select_action_returns_valid_action(self):
        """Should return a valid action from the mask."""
        env = MicroCivEnv(board_size=8, max_turns=50)
        agent = GreedyHeuristicAgent(seed=42)
        obs, _ = env.reset()

        action = agent.select_action(env, obs)

        assert obs["action_mask"][action] > 0

    def test_prefers_attack_over_move(self):
        """Attack actions should be prioritized over movement."""
        agent = GreedyHeuristicAgent(seed=42, aggression=1.0)
        assert agent.priority.attack > agent.priority.move_toward_enemy

    def test_reset_is_no_op(self):
        """Reset should not raise errors."""
        agent = GreedyHeuristicAgent()
        agent.reset()


class TestWinRateResult:
    """Tests for WinRateResult dataclass."""

    def test_win_rate_calculation(self):
        """Should calculate win rate correctly."""
        result = WinRateResult(wins=7, losses=2, draws=1, total_games=10)

        assert result.win_rate == 70.0
        assert result.loss_rate == 20.0
        assert result.draw_rate == 10.0

    def test_win_rate_zero_games(self):
        """Should handle zero games gracefully."""
        result = WinRateResult()

        assert result.win_rate == 0.0
        assert result.loss_rate == 0.0
        assert result.draw_rate == 0.0

    def test_str_representation(self):
        """Should produce readable string representation."""
        result = WinRateResult(wins=5, losses=3, draws=2, total_games=10)
        s = str(result)

        assert "50.0%" in s
        assert "5W" in s
        assert "3L" in s
        assert "2D" in s


class TestGameResult:
    """Tests for GameResult dataclass."""

    def test_creation(self):
        """Should create game result correctly."""
        result = GameResult(
            winner=0,
            num_turns=50,
            player0_cities=3,
            player1_cities=1,
            player0_units=5,
            player1_units=2,
            game_duration_seconds=1.5,
        )

        assert result.winner == 0
        assert result.num_turns == 50
        assert result.player0_cities == 3


class TestEvaluationPipeline:
    """Tests for EvaluationPipeline."""

    def test_initialization(self):
        """Should initialize with default parameters."""
        pipeline = EvaluationPipeline()
        assert pipeline.board_size == 16
        assert pipeline.max_turns == 200

    def test_initialization_with_custom_params(self):
        """Should accept custom parameters."""
        pipeline = EvaluationPipeline(board_size=8, max_turns=50, seed=42)
        assert pipeline.board_size == 8
        assert pipeline.max_turns == 50

    def test_evaluate_vs_random_runs(self):
        """Should complete evaluation against random agent."""
        pipeline = EvaluationPipeline(board_size=8, max_turns=30)
        agent = RandomAgent(seed=123)

        result = pipeline.evaluate_vs_random(agent, n_games=2, log_interval=2)

        assert result.win_rate.total_games == 2
        assert len(result.game_results) == 2

    def test_evaluate_vs_heuristic_runs(self):
        """Should complete evaluation against heuristic agent."""
        pipeline = EvaluationPipeline(board_size=8, max_turns=30)
        agent = GreedyHeuristicAgent(seed=123)

        result = pipeline.evaluate_vs_heuristic(agent, n_games=2, log_interval=2)

        assert result.win_rate.total_games == 2

    def test_heuristic_beats_random(self):
        """Heuristic agent should consistently beat random."""
        pipeline = EvaluationPipeline(board_size=8, max_turns=50, seed=42)
        heuristic = GreedyHeuristicAgent(seed=42)

        result = pipeline.evaluate_vs_random(heuristic, n_games=10, log_interval=10)

        # Heuristic should win more than it loses
        assert result.win_rate.wins >= result.win_rate.losses


class TestBaselineVsBaseline:
    """Integration tests for baseline agent matchups."""

    def test_random_vs_random_roughly_equal(self):
        """Two random agents should have ~50% win rates."""
        pipeline = EvaluationPipeline(board_size=8, max_turns=50, seed=42)
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=123)

        result = pipeline._run_evaluation(
            agent1, agent2, n_games=10, log_interval=10, opponent_name="Random2"
        )

        # With random vs random, should be close to 50%
        assert 20 <= result.win_rate.win_rate <= 80

    def test_game_produces_valid_results(self):
        """A complete game should produce valid statistics."""
        pipeline = EvaluationPipeline(board_size=8, max_turns=50, seed=42)
        agent = RandomAgent(seed=42)
        opponent = RandomAgent(seed=123)

        game_result, _ = pipeline._play_game(agent, opponent, agent_is_player0=True)

        assert game_result.num_turns > 0
        assert game_result.game_duration_seconds > 0
        assert game_result.winner in [0, 1, None]
