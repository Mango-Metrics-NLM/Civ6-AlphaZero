"""Coverage tests for alpha_zero.utils.training_diagnostics.

Targets uncovered paths including:
- TrainingDiagnostics.on_step, on_episode_end, on_iteration_end
- _check_warnings (entropy collapse, value stagnation, no-win streak)
- get_summary, get_learning_curves, reset, create_diagnostics factory
- ValueHeadMonitor: diagnose, check_alerts, reset
- BaselineEvaluator: should_evaluate, evaluate, get_history, get_win_rate_curve
- check_value_head standalone function
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alpha_zero.utils.training_diagnostics import (
    ActionDiversityTracker,
    BaselineEvaluationResult,
    BaselineEvaluator,
    PolicyEntropyTracker,
    TrainingDiagnostics,
    ValueHeadDiagnostics,
    ValueHeadMonitor,
    ValuePredictionTracker,
    create_diagnostics,
)

# ---------------------------------------------------------------------------
# PolicyEntropyTracker
# ---------------------------------------------------------------------------


class TestPolicyEntropyTracker:
    def test_add_uniform_policy(self):
        tracker = PolicyEntropyTracker(window_size=10)
        policy = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = tracker.add(policy)
        assert entropy > 0

    def test_add_with_action_mask(self):
        tracker = PolicyEntropyTracker(window_size=10)
        policy = np.array([0.5, 0.5, 0.0, 0.0])
        mask = np.array([1, 1, 0, 0])
        entropy = tracker.add(policy, action_mask=mask)
        assert entropy > 0

    def test_add_all_zeros(self):
        tracker = PolicyEntropyTracker(window_size=10)
        policy = np.array([0.0, 0.0, 0.0])
        entropy = tracker.add(policy)
        assert entropy == 0.0

    def test_current_property(self):
        tracker = PolicyEntropyTracker(window_size=10)
        assert tracker.current == 0.0
        tracker.add(np.array([0.5, 0.5]))
        assert tracker.current > 0

    def test_moving_average(self):
        tracker = PolicyEntropyTracker(window_size=5)
        assert tracker.moving_average == 0.0
        for _ in range(3):
            tracker.add(np.array([0.5, 0.5]))
        assert tracker.moving_average > 0

    def test_all_values(self):
        tracker = PolicyEntropyTracker(window_size=5)
        for _ in range(3):
            tracker.add(np.array([0.5, 0.5]))
        values = tracker.all_values
        assert len(values) == 3

    def test_reset(self):
        tracker = PolicyEntropyTracker(window_size=5)
        tracker.add(np.array([0.5, 0.5]))
        tracker.reset()
        assert tracker.current == 0.0
        assert len(tracker.all_values) == 0


# ---------------------------------------------------------------------------
# ValuePredictionTracker
# ---------------------------------------------------------------------------


class TestValuePredictionTracker:
    def test_add_returns_error(self):
        tracker = ValuePredictionTracker(window_size=10)
        error = tracker.add(predicted=0.5, actual=0.7)
        assert error == pytest.approx(0.2, abs=1e-6)

    def test_mean_absolute_error_empty(self):
        tracker = ValuePredictionTracker()
        assert tracker.mean_absolute_error == 0.0

    def test_mean_absolute_error(self):
        tracker = ValuePredictionTracker(window_size=10)
        tracker.add(0.5, 0.7)
        tracker.add(0.3, 0.1)
        assert tracker.mean_absolute_error == pytest.approx(0.2, abs=1e-6)

    def test_correlation_too_few_samples(self):
        tracker = ValuePredictionTracker()
        tracker.add(0.5, 0.5)
        assert tracker.correlation == 0.0

    def test_correlation_perfect(self):
        tracker = ValuePredictionTracker(window_size=10)
        for i in range(5):
            tracker.add(float(i), float(i))
        assert tracker.correlation > 0.99

    def test_correlation_constant_predictions(self):
        tracker = ValuePredictionTracker(window_size=10)
        for i in range(5):
            tracker.add(0.5, float(i))
        assert tracker.correlation == 0.0

    def test_prediction_bias_empty(self):
        tracker = ValuePredictionTracker()
        assert tracker.prediction_bias == 0.0

    def test_prediction_bias_overestimate(self):
        tracker = ValuePredictionTracker(window_size=10)
        for _ in range(5):
            tracker.add(1.0, 0.0)
        assert tracker.prediction_bias > 0

    def test_reset(self):
        tracker = ValuePredictionTracker(window_size=10)
        tracker.add(0.5, 0.7)
        tracker.reset()
        assert tracker.mean_absolute_error == 0.0


# ---------------------------------------------------------------------------
# ActionDiversityTracker
# ---------------------------------------------------------------------------


class TestActionDiversityTracker:
    def test_unique_actions(self):
        tracker = ActionDiversityTracker(num_actions=10, window_size=100)
        tracker.add(0)
        tracker.add(1)
        tracker.add(0)
        assert tracker.unique_actions == 2

    def test_action_entropy_empty(self):
        tracker = ActionDiversityTracker(num_actions=10)
        assert tracker.action_entropy == 0.0

    def test_action_entropy_single_action(self):
        tracker = ActionDiversityTracker(num_actions=10)
        for _ in range(5):
            tracker.add(0)
        # Single action has low entropy
        assert tracker.action_entropy == pytest.approx(0.0, abs=0.01)

    def test_diversity_ratio(self):
        tracker = ActionDiversityTracker(num_actions=10, window_size=100)
        tracker.add(0)
        tracker.add(1)
        tracker.add(2)
        assert tracker.diversity_ratio == pytest.approx(0.3)

    def test_most_common_actions(self):
        tracker = ActionDiversityTracker(num_actions=10, window_size=100)
        for _ in range(10):
            tracker.add(0)
        for _ in range(5):
            tracker.add(1)
        common = tracker.most_common_actions
        assert common[0] == (0, 10)
        assert common[1] == (1, 5)

    def test_reset(self):
        tracker = ActionDiversityTracker(num_actions=10)
        tracker.add(0)
        tracker.reset()
        assert tracker.unique_actions == 0


# ---------------------------------------------------------------------------
# TrainingDiagnostics — main class
# ---------------------------------------------------------------------------


class TestTrainingDiagnostics:
    @pytest.fixture
    def diag(self):
        return TrainingDiagnostics(num_actions=50)

    def test_on_step(self, diag):
        policy = np.ones(50) / 50
        diag.on_step(policy=policy, value=0.5, action=3)
        assert diag.action_tracker.unique_actions == 1

    def test_on_step_with_action_mask(self, diag):
        policy = np.zeros(50)
        policy[:5] = 0.2
        mask = np.zeros(50)
        mask[:5] = 1
        diag.on_step(policy=policy, value=0.5, action=2, action_mask=mask)
        assert diag.entropy_tracker.current > 0

    def test_on_episode_end(self, diag):
        policy = np.ones(50) / 50
        diag.on_step(policy=policy, value=0.5, action=3)
        diag.on_step(policy=policy, value=0.6, action=4)
        metrics = diag.on_episode_end(episode_id=0, turns=2, total_reward=1.0, outcome="win")
        assert metrics.episode_id == 0
        assert metrics.turns == 2
        assert metrics.outcome == "win"
        assert metrics.avg_policy_entropy > 0

    def test_on_episode_end_shaped_reward(self, diag):
        policy = np.ones(50) / 50
        diag.on_step(policy=policy, value=0.5, action=3)
        metrics = diag.on_episode_end(
            episode_id=0, turns=1, total_reward=1.0, outcome="win", shaped_reward=0.8
        )
        assert metrics.shaped_reward == 0.8

    def test_on_episode_end_no_steps(self, diag):
        # No on_step calls — empty averages
        metrics = diag.on_episode_end(episode_id=0, turns=0, total_reward=0.0, outcome="draw")
        assert metrics.avg_policy_entropy == 0.0
        assert metrics.avg_value_prediction == 0.0

    def test_on_iteration_end_no_episodes(self, diag):
        metrics = diag.on_iteration_end(
            iteration=1, policy_loss=0.5, value_loss=0.3, total_loss=0.8
        )
        assert metrics.num_episodes == 0
        assert metrics.win_rate == 0.0

    def test_on_iteration_end_with_episodes(self, diag):
        policy = np.ones(50) / 50
        for i in range(5):
            diag.on_step(policy=policy, value=0.5, action=i % 10)
            diag.on_episode_end(
                episode_id=i,
                turns=10,
                total_reward=1.0 if i < 3 else -1.0,
                outcome="win" if i < 3 else "loss",
            )
        metrics = diag.on_iteration_end(
            iteration=1, policy_loss=0.5, value_loss=0.3, total_loss=0.8
        )
        assert metrics.num_episodes == 5
        assert metrics.win_rate == pytest.approx(0.6)
        assert metrics.loss_rate == pytest.approx(0.4)

    def test_check_warnings_entropy_collapse(self, diag):
        """Low entropy triggers warning."""
        policy = np.zeros(50)
        policy[0] = 1.0  # Concentrated policy
        for i in range(5):
            diag.on_step(policy=policy, value=0.5, action=0)
            diag.on_episode_end(episode_id=i, turns=1, total_reward=1.0, outcome="win")
        # entropy will be ~0, which is < default threshold 0.1
        diag.on_iteration_end(iteration=1, policy_loss=0.5, value_loss=0.3, total_loss=0.8)
        assert any("entropy collapse" in w.lower() for w in diag.warnings)

    def test_check_warnings_value_loss_low(self, diag):
        """Very low value loss triggers warning."""
        policy = np.ones(50) / 50
        diag.on_step(policy=policy, value=0.5, action=0)
        diag.on_episode_end(episode_id=0, turns=1, total_reward=1.0, outcome="win")
        diag.on_iteration_end(iteration=1, policy_loss=0.5, value_loss=0.001, total_loss=0.501)
        assert any("value loss very low" in w.lower() for w in diag.warnings)

    def test_check_warnings_no_wins(self, diag):
        """Three iterations with no wins triggers warning."""
        policy = np.ones(50) / 50
        for it in range(3):
            diag.on_step(policy=policy, value=0.5, action=0)
            diag.on_episode_end(episode_id=it, turns=1, total_reward=-1.0, outcome="loss")
            diag.on_iteration_end(iteration=it + 1, policy_loss=0.5, value_loss=0.3, total_loss=0.8)
        assert any("no wins" in w.lower() for w in diag.warnings)

    def test_get_summary(self, diag):
        summary = diag.get_summary()
        assert "entropy" in summary
        assert "value_prediction" in summary
        assert "action_diversity" in summary
        assert "warnings" in summary

    def test_get_summary_with_iteration(self, diag):
        policy = np.ones(50) / 50
        diag.on_step(policy=policy, value=0.5, action=0)
        diag.on_episode_end(episode_id=0, turns=1, total_reward=1.0, outcome="win")
        diag.on_iteration_end(iteration=1, policy_loss=0.5, value_loss=0.3, total_loss=0.8)
        summary = diag.get_summary()
        assert "latest_iteration" in summary
        assert summary["latest_iteration"]["iteration"] == 1

    def test_get_learning_curves_empty(self, diag):
        curves = diag.get_learning_curves()
        assert curves["iterations"] == []

    def test_get_learning_curves_with_data(self, diag):
        policy = np.ones(50) / 50
        for it in range(3):
            diag.on_step(policy=policy, value=0.5, action=0)
            diag.on_episode_end(episode_id=it, turns=1, total_reward=1.0, outcome="win")
            diag.on_iteration_end(
                iteration=it + 1,
                policy_loss=0.5 - it * 0.1,
                value_loss=0.3,
                total_loss=0.8 - it * 0.1,
            )
        curves = diag.get_learning_curves()
        assert len(curves["iterations"]) == 3
        assert len(curves["win_rate"]) == 3

    def test_clear_warnings(self, diag):
        diag._warnings.append("test warning")
        diag.clear_warnings()
        assert len(diag.warnings) == 0

    def test_reset(self, diag):
        policy = np.ones(50) / 50
        diag.on_step(policy=policy, value=0.5, action=0)
        diag.on_episode_end(episode_id=0, turns=1, total_reward=1.0, outcome="win")
        diag.on_iteration_end(iteration=1, policy_loss=0.5, value_loss=0.3, total_loss=0.8)
        diag.reset()
        assert diag.entropy_tracker.current == 0.0
        assert diag.action_tracker.unique_actions == 0


# ---------------------------------------------------------------------------
# create_diagnostics factory
# ---------------------------------------------------------------------------


class TestCreateDiagnostics:
    def test_default(self):
        d = create_diagnostics(num_actions=100)
        assert d.num_actions == 100
        assert d.config.window_size == 100

    def test_custom_window(self):
        d = create_diagnostics(num_actions=50, window_size=200)
        assert d.config.window_size == 200

    def test_custom_thresholds(self):
        d = create_diagnostics(
            num_actions=50,
            entropy_min_threshold=0.5,
            value_loss_threshold=0.1,
        )
        assert d.config.entropy_min_threshold == 0.5


# ---------------------------------------------------------------------------
# ValueHeadDiagnostics / ValueHeadMonitor
# ---------------------------------------------------------------------------


class TestValueHeadDiagnostics:
    def test_str_ok(self):
        d = ValueHeadDiagnostics(is_stuck=False, variance=0.1, mean_prediction=0.0)
        assert "OK" in str(d)

    def test_str_stuck(self):
        d = ValueHeadDiagnostics(is_stuck=True, variance=0.001, mean_prediction=0.0)
        assert "STUCK" in str(d)


class TestValueHeadMonitor:
    def test_diagnose_too_few_samples(self):
        monitor = ValueHeadMonitor()
        for _ in range(5):
            monitor.record(0.5, 1.0)
        diag = monitor.diagnose()
        assert diag.num_samples == 5
        assert diag.is_stuck is False

    def test_diagnose_stuck(self):
        monitor = ValueHeadMonitor(variance_threshold=0.01)
        for _ in range(50):
            monitor.record(0.5, np.random.choice([-1.0, 1.0]))
        diag = monitor.diagnose()
        assert diag.is_stuck is True
        assert diag.variance < 0.01

    def test_diagnose_not_stuck(self):
        monitor = ValueHeadMonitor(variance_threshold=0.01)
        for i in range(50):
            monitor.record(float(i) / 50, float(i % 2))
        diag = monitor.diagnose()
        assert diag.is_stuck is False

    def test_check_alerts_not_enough_data(self):
        monitor = ValueHeadMonitor()
        for _ in range(10):
            monitor.record(0.5, 1.0)
        alerts = monitor.check_alerts()
        assert len(alerts) == 0  # < 50 samples

    def test_check_alerts_stuck(self):
        monitor = ValueHeadMonitor(variance_threshold=0.01)
        for _ in range(100):
            monitor.record(0.5, np.random.choice([-1.0, 1.0]))
        alerts = monitor.check_alerts()
        assert any("STUCK" in a for a in alerts)

    def test_check_alerts_low_correlation(self):
        monitor = ValueHeadMonitor(correlation_threshold=0.5)
        rng = np.random.RandomState(42)
        for _ in range(100):
            monitor.record(rng.uniform(-1, 1), rng.choice([-1.0, 1.0]))
        alerts = monitor.check_alerts()
        assert any("NOT LEARNING" in a for a in alerts)

    def test_check_alerts_overconfident(self):
        monitor = ValueHeadMonitor()
        for _ in range(100):
            monitor.record(0.9 + np.random.uniform(0, 0.1), np.random.choice([-1, 1]))
        alerts = monitor.check_alerts()
        assert any("OVERCONFIDENT" in a for a in alerts)

    def test_check_alerts_pessimistic(self):
        monitor = ValueHeadMonitor()
        for _ in range(100):
            monitor.record(-0.9 - np.random.uniform(0, 0.1), np.random.choice([-1, 1]))
        alerts = monitor.check_alerts()
        assert any("PESSIMISTIC" in a for a in alerts)

    def test_reset(self):
        monitor = ValueHeadMonitor()
        monitor.record(0.5, 1.0)
        monitor.reset()
        diag = monitor.diagnose()
        assert diag.num_samples == 0


# ---------------------------------------------------------------------------
# BaselineEvaluationResult
# ---------------------------------------------------------------------------


class TestBaselineEvaluationResult:
    def test_str_with_heuristic(self):
        r = BaselineEvaluationResult(
            iteration=10,
            win_rate_vs_random=65.0,
            win_rate_vs_heuristic=40.0,
            games_played=30,
            value_correlation=0.85,
        )
        s = str(r)
        assert "vs_random=65.0%" in s
        assert "vs_heuristic=40.0%" in s

    def test_str_without_heuristic(self):
        r = BaselineEvaluationResult(
            iteration=10,
            win_rate_vs_random=65.0,
            win_rate_vs_heuristic=None,
            games_played=20,
            value_correlation=0.85,
        )
        s = str(r)
        assert "vs_random=65.0%" in s
        assert "heuristic" not in s


# ---------------------------------------------------------------------------
# BaselineEvaluator
# ---------------------------------------------------------------------------


class TestBaselineEvaluator:
    def test_should_evaluate(self):
        evaluator = BaselineEvaluator(eval_interval=10)
        assert evaluator.should_evaluate(0) is False
        assert evaluator.should_evaluate(5) is False
        assert evaluator.should_evaluate(10) is True
        assert evaluator.should_evaluate(20) is True

    def test_evaluate_with_mocks(self):
        evaluator = BaselineEvaluator(
            eval_interval=10,
            n_games_random=5,
            n_games_heuristic=3,
            board_size=8,
        )
        mock_mcts = MagicMock()

        mock_win_rate = MagicMock()
        mock_win_rate.win_rate = 50.0
        mock_random_result = MagicMock()
        mock_random_result.win_rate = mock_win_rate
        mock_random_result.value_correlation = 0.7

        mock_heuristic_rate = MagicMock()
        mock_heuristic_rate.win_rate = 30.0
        mock_heuristic_result = MagicMock()
        mock_heuristic_result.win_rate = mock_heuristic_rate

        with patch("alpha_zero.evaluation.evaluation_pipeline.EvaluationPipeline") as MockPipeline:
            pipeline_inst = MagicMock()
            pipeline_inst.evaluate_vs_random.return_value = mock_random_result
            pipeline_inst.evaluate_vs_heuristic.return_value = mock_heuristic_result
            MockPipeline.return_value = pipeline_inst

            result = evaluator.evaluate(mock_mcts, iteration=10)

        assert result.win_rate_vs_random == 50.0
        assert result.win_rate_vs_heuristic == 30.0
        assert result.games_played == 8

    def test_evaluate_no_heuristic(self):
        evaluator = BaselineEvaluator(n_games_random=5)
        mock_mcts = MagicMock()

        mock_win_rate = MagicMock()
        mock_win_rate.win_rate = 50.0
        mock_random_result = MagicMock()
        mock_random_result.win_rate = mock_win_rate
        mock_random_result.value_correlation = 0.7

        with patch("alpha_zero.evaluation.evaluation_pipeline.EvaluationPipeline") as MockPipeline:
            pipeline_inst = MagicMock()
            pipeline_inst.evaluate_vs_random.return_value = mock_random_result
            MockPipeline.return_value = pipeline_inst

            result = evaluator.evaluate(mock_mcts, iteration=10, include_heuristic=False)

        assert result.win_rate_vs_heuristic is None

    def test_evaluate_low_win_rate_alert(self):
        evaluator = BaselineEvaluator(n_games_random=5, n_games_heuristic=0)

        mock_win_rate = MagicMock()
        mock_win_rate.win_rate = 3.0  # Critical
        mock_random_result = MagicMock()
        mock_random_result.win_rate = mock_win_rate
        mock_random_result.value_correlation = 0.0

        with patch("alpha_zero.evaluation.evaluation_pipeline.EvaluationPipeline") as MockPipeline:
            pipeline_inst = MagicMock()
            pipeline_inst.evaluate_vs_random.return_value = mock_random_result
            MockPipeline.return_value = pipeline_inst

            result = evaluator.evaluate(MagicMock(), iteration=1, include_heuristic=False)

        assert any("CRITICAL" in a for a in result.alerts)

    def test_evaluate_regression_alert(self):
        evaluator = BaselineEvaluator(n_games_random=5)
        evaluator._last_win_rate = 80.0

        mock_win_rate = MagicMock()
        mock_win_rate.win_rate = 60.0  # 20% drop
        mock_random_result = MagicMock()
        mock_random_result.win_rate = mock_win_rate
        mock_random_result.value_correlation = 0.5

        with patch("alpha_zero.evaluation.evaluation_pipeline.EvaluationPipeline") as MockPipeline:
            pipeline_inst = MagicMock()
            pipeline_inst.evaluate_vs_random.return_value = mock_random_result
            MockPipeline.return_value = pipeline_inst

            result = evaluator.evaluate(MagicMock(), iteration=10, include_heuristic=False)

        assert any("REGRESSION" in a for a in result.alerts)

    def test_get_history(self):
        evaluator = BaselineEvaluator()
        assert evaluator.get_history() == []

    def test_get_win_rate_curve_empty(self):
        evaluator = BaselineEvaluator()
        curve = evaluator.get_win_rate_curve()
        assert curve["iterations"] == []


# ---------------------------------------------------------------------------
# check_value_head standalone
# ---------------------------------------------------------------------------


class TestCheckValueHead:
    def test_check_value_head_stuck(self):
        mock_network = MagicMock()
        # Return constant value
        import torch

        mock_network.return_value = (torch.zeros(1, 50), torch.tensor([[0.5]]))
        mock_network.eval = MagicMock()

        mock_mcts = MagicMock()
        mock_mcts.network = mock_network
        mock_mcts.device = "cpu"

        from alpha_zero.utils.training_diagnostics import check_value_head

        diag = check_value_head(mock_mcts, n_positions=15, board_size=4)
        assert diag.is_stuck is True
        assert diag.num_samples == 15

    def test_check_value_head_ok(self):
        import torch

        from alpha_zero.utils.training_diagnostics import check_value_head

        # Return varied values
        call_count = {"n": 0}

        def mock_forward(spatial, vector):
            call_count["n"] += 1
            val = (call_count["n"] % 10) / 5.0 - 1.0
            return torch.zeros(1, 50), torch.tensor([[val]])

        mock_network = MagicMock(side_effect=mock_forward)
        mock_network.eval = MagicMock()

        mock_mcts = MagicMock()
        mock_mcts.network = mock_network
        mock_mcts.device = "cpu"

        diag = check_value_head(mock_mcts, n_positions=20, board_size=4)
        assert diag.is_stuck is False
