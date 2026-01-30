"""Unit tests for training diagnostics module."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.utils.training_diagnostics import (
    ActionDiversityTracker,
    DiagnosticsConfig,
    EpisodeMetrics,
    IterationMetrics,
    PolicyEntropyTracker,
    TrainingDiagnostics,
    ValuePredictionTracker,
    create_diagnostics,
)


class TestDiagnosticsConfig:
    """Tests for DiagnosticsConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DiagnosticsConfig()

        assert config.window_size == 100
        assert config.entropy_min_threshold == 0.1
        assert config.entropy_max_threshold == 3.0
        assert config.value_loss_threshold == 0.01
        assert config.track_action_distribution is True
        assert config.track_value_predictions is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = DiagnosticsConfig(
            window_size=50,
            entropy_min_threshold=0.5,
        )

        assert config.window_size == 50
        assert config.entropy_min_threshold == 0.5


class TestEpisodeMetrics:
    """Tests for EpisodeMetrics dataclass."""

    def test_basic_creation(self):
        """Test basic metrics creation."""
        metrics = EpisodeMetrics(
            episode_id=1,
            turns=50,
            total_reward=0.5,
            outcome="win",
            avg_policy_entropy=1.5,
            avg_value_prediction=0.3,
        )

        assert metrics.episode_id == 1
        assert metrics.turns == 50
        assert metrics.total_reward == 0.5
        assert metrics.outcome == "win"
        assert metrics.avg_policy_entropy == 1.5
        assert metrics.avg_value_prediction == 0.3
        assert metrics.action_counts == {}
        assert metrics.shaped_reward is None

    def test_with_action_counts(self):
        """Test metrics with action counts."""
        metrics = EpisodeMetrics(
            episode_id=1,
            turns=10,
            total_reward=0.0,
            outcome="timeout",
            avg_policy_entropy=1.0,
            avg_value_prediction=0.0,
            action_counts={0: 5, 1: 3, 2: 2},
        )

        assert metrics.action_counts[0] == 5
        assert sum(metrics.action_counts.values()) == 10


class TestIterationMetrics:
    """Tests for IterationMetrics dataclass."""

    def test_basic_creation(self):
        """Test basic iteration metrics creation."""
        metrics = IterationMetrics(
            iteration=1,
            num_episodes=10,
            avg_turns=45.5,
            win_rate=0.3,
            loss_rate=0.2,
            draw_rate=0.1,
            timeout_rate=0.4,
            avg_reward=0.1,
            policy_loss=1.5,
            value_loss=0.5,
            total_loss=2.0,
            avg_entropy=1.2,
            action_diversity=0.15,
        )

        assert metrics.iteration == 1
        assert metrics.num_episodes == 10
        assert metrics.win_rate == 0.3
        # Outcome rates should sum to 1
        rates = metrics.win_rate + metrics.loss_rate + metrics.draw_rate + metrics.timeout_rate
        assert rates == pytest.approx(1.0)


class TestPolicyEntropyTracker:
    """Tests for PolicyEntropyTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = PolicyEntropyTracker(window_size=50)
        assert tracker.window_size == 50
        assert tracker.current == 0.0
        assert tracker.moving_average == 0.0

    def test_add_uniform_policy(self):
        """Test entropy of uniform distribution."""
        tracker = PolicyEntropyTracker()

        policy = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = tracker.add(policy)

        # Entropy of uniform over 4 actions = log(4) ≈ 1.386
        expected = -np.sum(policy * np.log(policy))
        assert entropy == pytest.approx(expected)

    def test_add_deterministic_policy(self):
        """Test entropy of deterministic policy."""
        tracker = PolicyEntropyTracker()

        policy = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = tracker.add(policy)

        # Deterministic policy has 0 entropy (with small epsilon tolerance)
        assert entropy == pytest.approx(0.0, abs=1e-9)

    def test_add_with_mask(self):
        """Test entropy calculation with action mask."""
        tracker = PolicyEntropyTracker()

        policy = np.array([0.5, 0.5, 0.0, 0.0])
        mask = np.array([1, 1, 0, 0])

        entropy = tracker.add(policy, action_mask=mask)

        # Entropy of uniform over 2 actions = log(2) ≈ 0.693
        expected = np.log(2)
        assert entropy == pytest.approx(expected)

    def test_moving_average(self):
        """Test moving average calculation."""
        tracker = PolicyEntropyTracker(window_size=3)

        tracker.add(np.array([0.5, 0.5]))  # entropy ≈ 0.693
        tracker.add(np.array([0.5, 0.5]))
        tracker.add(np.array([0.5, 0.5]))

        expected = np.log(2)
        assert tracker.moving_average == pytest.approx(expected)

    def test_window_size_limit(self):
        """Test that window size limits stored values."""
        tracker = PolicyEntropyTracker(window_size=2)

        tracker.add(np.array([0.5, 0.5]))  # ≈ 0.693
        tracker.add(np.array([0.5, 0.5]))
        tracker.add(np.array([1.0, 0.0]))  # 0

        # Only last 2 values in window
        assert len(tracker._entropies) == 2
        # Moving average should include the 0 now
        assert tracker.moving_average < np.log(2)

    def test_all_values(self):
        """Test that all_values returns full history."""
        tracker = PolicyEntropyTracker(window_size=2)

        tracker.add(np.array([0.5, 0.5]))
        tracker.add(np.array([0.5, 0.5]))
        tracker.add(np.array([1.0, 0.0]))

        # all_values should have all 3
        assert len(tracker.all_values) == 3

    def test_reset(self):
        """Test tracker reset."""
        tracker = PolicyEntropyTracker()

        tracker.add(np.array([0.5, 0.5]))
        tracker.add(np.array([0.5, 0.5]))

        tracker.reset()

        assert tracker.current == 0.0
        assert len(tracker._entropies) == 0
        assert len(tracker._all_entropies) == 0


class TestValuePredictionTracker:
    """Tests for ValuePredictionTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ValuePredictionTracker()
        assert tracker.mean_absolute_error == 0.0
        assert tracker.correlation == 0.0

    def test_add_prediction(self):
        """Test adding predictions."""
        tracker = ValuePredictionTracker()

        error = tracker.add(predicted=0.5, actual=0.3)

        assert error == pytest.approx(0.2)
        assert tracker.mean_absolute_error == pytest.approx(0.2)

    def test_mean_absolute_error(self):
        """Test MAE calculation."""
        tracker = ValuePredictionTracker()

        tracker.add(0.5, 0.3)  # error = 0.2
        tracker.add(0.4, 0.5)  # error = 0.1
        tracker.add(0.6, 0.3)  # error = 0.3

        # MAE = (0.2 + 0.1 + 0.3) / 3 = 0.2
        assert tracker.mean_absolute_error == pytest.approx(0.2)

    def test_correlation_perfect(self):
        """Test correlation with perfect predictions."""
        tracker = ValuePredictionTracker()

        tracker.add(0.1, 0.1)
        tracker.add(0.5, 0.5)
        tracker.add(0.9, 0.9)

        assert tracker.correlation == pytest.approx(1.0)

    def test_correlation_negative(self):
        """Test negative correlation."""
        tracker = ValuePredictionTracker()

        tracker.add(0.1, 0.9)
        tracker.add(0.5, 0.5)
        tracker.add(0.9, 0.1)

        assert tracker.correlation == pytest.approx(-1.0)

    def test_correlation_constant_values(self):
        """Test correlation with constant values."""
        tracker = ValuePredictionTracker()

        tracker.add(0.5, 0.5)
        tracker.add(0.5, 0.5)
        tracker.add(0.5, 0.5)

        # Constant values have undefined correlation
        assert tracker.correlation == 0.0

    def test_prediction_bias(self):
        """Test prediction bias calculation."""
        tracker = ValuePredictionTracker()

        tracker.add(0.6, 0.4)  # over by 0.2
        tracker.add(0.7, 0.5)  # over by 0.2
        tracker.add(0.8, 0.6)  # over by 0.2

        assert tracker.prediction_bias == pytest.approx(0.2)

    def test_reset(self):
        """Test tracker reset."""
        tracker = ValuePredictionTracker()

        tracker.add(0.5, 0.3)
        tracker.add(0.4, 0.5)

        tracker.reset()

        assert tracker.mean_absolute_error == 0.0
        assert len(tracker._predictions) == 0


class TestActionDiversityTracker:
    """Tests for ActionDiversityTracker class."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ActionDiversityTracker(num_actions=10)

        assert tracker.num_actions == 10
        assert tracker.unique_actions == 0
        assert tracker.diversity_ratio == 0.0

    def test_add_action(self):
        """Test adding actions."""
        tracker = ActionDiversityTracker(num_actions=10)

        tracker.add(0)
        tracker.add(1)
        tracker.add(0)

        assert tracker.unique_actions == 2
        assert tracker._action_counts[0] == 2
        assert tracker._action_counts[1] == 1

    def test_diversity_ratio(self):
        """Test diversity ratio calculation."""
        tracker = ActionDiversityTracker(num_actions=10)

        tracker.add(0)
        tracker.add(1)
        tracker.add(2)

        assert tracker.diversity_ratio == 0.3  # 3/10

    def test_action_entropy_uniform(self):
        """Test action entropy with uniform usage."""
        tracker = ActionDiversityTracker(num_actions=4)

        for i in range(4):
            for _ in range(10):
                tracker.add(i)

        # Uniform distribution entropy = log(4) ≈ 1.386
        expected = np.log(4)
        assert tracker.action_entropy == pytest.approx(expected, rel=0.01)

    def test_action_entropy_skewed(self):
        """Test action entropy with skewed usage."""
        tracker = ActionDiversityTracker(num_actions=4)

        # Action 0 used 97 times, others 1 each
        for _ in range(97):
            tracker.add(0)
        tracker.add(1)
        tracker.add(2)
        tracker.add(3)

        # Should be low entropy
        assert tracker.action_entropy < 0.5

    def test_most_common_actions(self):
        """Test getting most common actions."""
        tracker = ActionDiversityTracker(num_actions=10)

        tracker.add(0)
        tracker.add(0)
        tracker.add(0)
        tracker.add(1)
        tracker.add(1)
        tracker.add(2)

        most_common = tracker.most_common_actions

        assert most_common[0] == (0, 3)
        assert most_common[1] == (1, 2)
        assert most_common[2] == (2, 1)

    def test_reset(self):
        """Test tracker reset."""
        tracker = ActionDiversityTracker(num_actions=10)

        tracker.add(0)
        tracker.add(1)

        tracker.reset()

        assert tracker.unique_actions == 0
        assert len(tracker._action_counts) == 0


class TestTrainingDiagnostics:
    """Tests for main TrainingDiagnostics class."""

    def test_initialization(self):
        """Test diagnostics initialization."""
        diagnostics = TrainingDiagnostics(num_actions=100)

        assert diagnostics.num_actions == 100
        assert diagnostics.config is not None
        assert len(diagnostics._warnings) == 0

    def test_custom_config(self):
        """Test with custom configuration."""
        config = DiagnosticsConfig(window_size=50)
        diagnostics = TrainingDiagnostics(num_actions=100, config=config)

        assert diagnostics.config.window_size == 50

    def test_on_step(self):
        """Test recording a training step."""
        diagnostics = TrainingDiagnostics(num_actions=4)

        policy = np.array([0.4, 0.3, 0.2, 0.1])
        diagnostics.on_step(policy=policy, value=0.5, action=0)

        assert len(diagnostics._current_episode_entropies) == 1
        assert len(diagnostics._current_episode_values) == 1
        assert len(diagnostics._current_episode_actions) == 1

    def test_on_episode_end(self):
        """Test recording episode completion."""
        diagnostics = TrainingDiagnostics(num_actions=4)

        # Record some steps
        policy = np.array([0.4, 0.3, 0.2, 0.1])
        diagnostics.on_step(policy, 0.5, 0)
        diagnostics.on_step(policy, 0.6, 1)
        diagnostics.on_step(policy, 0.4, 0)

        # End episode
        metrics = diagnostics.on_episode_end(
            episode_id=1,
            turns=3,
            total_reward=0.5,
            outcome="win",
        )

        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.episode_id == 1
        assert metrics.turns == 3
        assert metrics.outcome == "win"
        assert metrics.avg_policy_entropy > 0
        assert metrics.action_counts[0] == 2
        assert metrics.action_counts[1] == 1

        # Episode tracking should be cleared
        assert len(diagnostics._current_episode_entropies) == 0

    def test_on_iteration_end(self):
        """Test recording iteration completion."""
        diagnostics = TrainingDiagnostics(num_actions=4)

        # Record an episode
        policy = np.array([0.4, 0.3, 0.2, 0.1])
        diagnostics.on_step(policy, 0.5, 0)
        diagnostics.on_episode_end(1, 1, 0.5, "win")

        diagnostics.on_step(policy, -0.5, 1)
        diagnostics.on_episode_end(2, 1, -0.5, "loss")

        # End iteration
        metrics = diagnostics.on_iteration_end(
            iteration=1,
            policy_loss=1.5,
            value_loss=0.5,
            total_loss=2.0,
        )

        assert isinstance(metrics, IterationMetrics)
        assert metrics.iteration == 1
        assert metrics.num_episodes == 2
        assert metrics.win_rate == 0.5
        assert metrics.loss_rate == 0.5
        assert metrics.policy_loss == 1.5
        assert metrics.value_loss == 0.5

    def test_warnings_entropy_collapse(self):
        """Test warning generation for entropy collapse."""
        config = DiagnosticsConfig(entropy_min_threshold=0.5)
        diagnostics = TrainingDiagnostics(num_actions=4, config=config)

        # Record episode with low entropy (deterministic policy)
        policy = np.array([1.0, 0.0, 0.0, 0.0])
        diagnostics.on_step(policy, 0.0, 0)
        diagnostics.on_episode_end(1, 1, 0.0, "timeout")

        # End iteration
        diagnostics.on_iteration_end(1, 1.0, 0.1, 1.1)

        assert len(diagnostics.warnings) > 0
        assert "entropy" in diagnostics.warnings[0].lower()

    def test_warnings_value_loss_low(self):
        """Test warning generation for low value loss."""
        config = DiagnosticsConfig(value_loss_threshold=0.1)
        diagnostics = TrainingDiagnostics(num_actions=4, config=config)

        # Record episode
        policy = np.array([0.4, 0.3, 0.2, 0.1])
        diagnostics.on_step(policy, 0.0, 0)
        diagnostics.on_episode_end(1, 1, 0.0, "timeout")

        # End iteration with very low value loss
        diagnostics.on_iteration_end(1, 1.0, 0.001, 1.001)

        assert len(diagnostics.warnings) > 0
        assert "value" in diagnostics.warnings[0].lower()

    def test_clear_warnings(self):
        """Test clearing warnings."""
        diagnostics = TrainingDiagnostics(num_actions=4)
        diagnostics._warnings.append("Test warning")

        diagnostics.clear_warnings()

        assert len(diagnostics.warnings) == 0

    def test_get_summary(self):
        """Test getting diagnostic summary."""
        diagnostics = TrainingDiagnostics(num_actions=4)

        # Record some data
        policy = np.array([0.4, 0.3, 0.2, 0.1])
        diagnostics.on_step(policy, 0.5, 0)
        diagnostics.on_episode_end(1, 1, 0.5, "win")
        diagnostics.on_iteration_end(1, 1.0, 0.5, 1.5)

        summary = diagnostics.get_summary()

        assert "total_iterations" in summary
        assert "entropy" in summary
        assert "value_prediction" in summary
        assert "action_diversity" in summary
        assert "warnings" in summary
        assert "latest_iteration" in summary

    def test_get_learning_curves(self):
        """Test getting learning curve data."""
        diagnostics = TrainingDiagnostics(num_actions=4)

        # Record multiple iterations
        for i in range(3):
            policy = np.array([0.4, 0.3, 0.2, 0.1])
            diagnostics.on_step(policy, 0.0, 0)
            diagnostics.on_episode_end(i, 1, 0.0, "timeout")
            diagnostics.on_iteration_end(i, 1.0 - i * 0.1, 0.5, 1.5 - i * 0.1)

        curves = diagnostics.get_learning_curves()

        assert len(curves["iterations"]) == 3
        assert len(curves["win_rate"]) == 3
        assert len(curves["total_loss"]) == 3
        # Loss should be decreasing
        assert curves["total_loss"][0] > curves["total_loss"][2]

    def test_get_learning_curves_empty(self):
        """Test learning curves with no data."""
        diagnostics = TrainingDiagnostics(num_actions=4)

        curves = diagnostics.get_learning_curves()

        assert curves["iterations"] == []
        assert curves["win_rate"] == []

    def test_reset(self):
        """Test full reset."""
        diagnostics = TrainingDiagnostics(num_actions=4)

        # Add some data
        policy = np.array([0.4, 0.3, 0.2, 0.1])
        diagnostics.on_step(policy, 0.5, 0)
        diagnostics.on_episode_end(1, 1, 0.5, "win")
        diagnostics.on_iteration_end(1, 1.0, 0.5, 1.5)
        diagnostics._warnings.append("Test")

        diagnostics.reset()

        assert len(diagnostics._iteration_metrics) == 0
        assert len(diagnostics._episode_metrics) == 0
        assert len(diagnostics._warnings) == 0
        assert diagnostics.entropy_tracker.current == 0.0


class TestCreateDiagnostics:
    """Tests for create_diagnostics factory function."""

    def test_factory_default(self):
        """Test factory with default arguments."""
        diagnostics = create_diagnostics(num_actions=100)

        assert diagnostics.num_actions == 100
        assert diagnostics.config.window_size == 100

    def test_factory_custom_window(self):
        """Test factory with custom window size."""
        diagnostics = create_diagnostics(num_actions=100, window_size=50)

        assert diagnostics.config.window_size == 50

    def test_factory_custom_config(self):
        """Test factory with custom config options."""
        diagnostics = create_diagnostics(
            num_actions=100,
            entropy_min_threshold=0.5,
            track_value_predictions=False,
        )

        assert diagnostics.config.entropy_min_threshold == 0.5
        assert diagnostics.config.track_value_predictions is False


class TestDiagnosticsIntegration:
    """Integration tests for diagnostics."""

    def test_full_training_simulation(self):
        """Test diagnostics through simulated training."""
        diagnostics = TrainingDiagnostics(num_actions=10)

        # Simulate 3 iterations of training
        for iteration in range(3):
            # Each iteration has 5 episodes
            for episode in range(5):
                # Each episode has 10 steps
                for _step in range(10):
                    policy = np.random.dirichlet([1] * 10)
                    value = np.random.uniform(-1, 1)
                    action = np.random.choice(10, p=policy)

                    diagnostics.on_step(policy, value, action)

                # End episode with random outcome
                outcome = np.random.choice(["win", "loss", "draw", "timeout"])
                reward = {"win": 1.0, "loss": -1.0, "draw": 0.0, "timeout": 0.0}[outcome]

                diagnostics.on_episode_end(
                    episode_id=iteration * 5 + episode,
                    turns=10,
                    total_reward=reward,
                    outcome=outcome,
                )

            # End iteration with decreasing loss
            diagnostics.on_iteration_end(
                iteration=iteration,
                policy_loss=2.0 - iteration * 0.3,
                value_loss=1.0 - iteration * 0.1,
                total_loss=3.0 - iteration * 0.4,
            )

        # Verify tracking
        summary = diagnostics.get_summary()
        assert summary["total_iterations"] == 3

        curves = diagnostics.get_learning_curves()
        assert len(curves["iterations"]) == 3

        # Loss should be decreasing
        assert curves["total_loss"][0] > curves["total_loss"][-1]

    def test_detecting_training_issues(self):
        """Test that diagnostics can detect training issues."""
        config = DiagnosticsConfig(
            entropy_min_threshold=0.5,
            value_loss_threshold=0.01,
        )
        diagnostics = TrainingDiagnostics(num_actions=4, config=config)

        # Simulate training with collapsed policy (always same action)
        for iteration in range(5):
            policy = np.array([0.97, 0.01, 0.01, 0.01])
            for _ in range(10):
                diagnostics.on_step(policy, 0.0, 0)

            diagnostics.on_episode_end(iteration, 10, 0.0, "timeout")
            diagnostics.on_iteration_end(iteration, 2.0, 0.001, 2.001)

        # Should have warnings about entropy and value loss
        warnings = diagnostics.warnings
        assert len(warnings) > 0

        # Check warnings mention the issues
        warning_text = " ".join(warnings).lower()
        assert "entropy" in warning_text or "value" in warning_text
