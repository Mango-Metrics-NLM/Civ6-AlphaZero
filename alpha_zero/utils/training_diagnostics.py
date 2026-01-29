"""Training diagnostics utilities for monitoring learning progress.

This module provides diagnostic tools to track and analyze training progress,
identify issues, and provide actionable insights during AlphaZero training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DiagnosticsConfig:
    """Configuration for training diagnostics."""

    # Moving average window for metrics
    window_size: int = 100
    # Thresholds for warnings
    entropy_min_threshold: float = 0.1  # Warn if entropy drops below
    entropy_max_threshold: float = 3.0  # Expected max entropy
    value_loss_threshold: float = 0.01  # Warn if value loss stays below
    win_rate_improvement_threshold: float = 0.01  # Minimum improvement per iteration
    # Tracking settings
    track_action_distribution: bool = True
    track_value_predictions: bool = True


@dataclass
class EpisodeMetrics:
    """Metrics from a single episode."""

    episode_id: int
    turns: int
    total_reward: float
    outcome: str  # "win", "loss", "draw", "timeout"
    avg_policy_entropy: float
    avg_value_prediction: float
    action_counts: dict[int, int] = field(default_factory=dict)
    shaped_reward: float | None = None


@dataclass
class IterationMetrics:
    """Metrics from a training iteration."""

    iteration: int
    num_episodes: int
    avg_turns: float
    win_rate: float
    loss_rate: float
    draw_rate: float
    timeout_rate: float
    avg_reward: float
    policy_loss: float
    value_loss: float
    total_loss: float
    avg_entropy: float
    action_diversity: float


class PolicyEntropyTracker:
    """Track policy entropy over time."""

    def __init__(self, window_size: int = 100):
        """Initialize entropy tracker.

        Args:
            window_size: Size of moving average window.
        """
        self.window_size = window_size
        self._entropies: deque[float] = deque(maxlen=window_size)
        self._all_entropies: list[float] = []

    def add(self, policy: np.ndarray, action_mask: np.ndarray | None = None) -> float:
        """Add a policy observation and compute entropy.

        Args:
            policy: Policy probability distribution.
            action_mask: Optional mask for valid actions.

        Returns:
            Entropy of the policy.
        """
        if action_mask is not None:
            # Only consider valid actions
            valid_probs = policy[action_mask > 0]
        else:
            valid_probs = policy

        # Filter out zeros for log calculation
        valid_probs = valid_probs[valid_probs > 0]

        if len(valid_probs) == 0:
            entropy = 0.0
        else:
            entropy = -np.sum(valid_probs * np.log(valid_probs + 1e-10))

        self._entropies.append(entropy)
        self._all_entropies.append(entropy)

        return entropy

    @property
    def current(self) -> float:
        """Current entropy (most recent)."""
        if not self._entropies:
            return 0.0
        return self._entropies[-1]

    @property
    def moving_average(self) -> float:
        """Moving average of entropy."""
        if not self._entropies:
            return 0.0
        return sum(self._entropies) / len(self._entropies)

    @property
    def all_values(self) -> list[float]:
        """All recorded entropy values."""
        return self._all_entropies.copy()

    def reset(self) -> None:
        """Reset tracker state."""
        self._entropies.clear()
        self._all_entropies.clear()


class ValuePredictionTracker:
    """Track value prediction accuracy."""

    def __init__(self, window_size: int = 100):
        """Initialize value prediction tracker.

        Args:
            window_size: Size of moving average window.
        """
        self.window_size = window_size
        self._predictions: deque[float] = deque(maxlen=window_size)
        self._actuals: deque[float] = deque(maxlen=window_size)
        self._errors: deque[float] = deque(maxlen=window_size)

    def add(self, predicted: float, actual: float) -> float:
        """Add a prediction-actual pair.

        Args:
            predicted: Predicted value from network.
            actual: Actual outcome.

        Returns:
            Absolute error of prediction.
        """
        error = abs(predicted - actual)

        self._predictions.append(predicted)
        self._actuals.append(actual)
        self._errors.append(error)

        return error

    @property
    def mean_absolute_error(self) -> float:
        """Mean absolute error of predictions."""
        if not self._errors:
            return 0.0
        return sum(self._errors) / len(self._errors)

    @property
    def correlation(self) -> float:
        """Correlation between predictions and actuals."""
        if len(self._predictions) < 2:
            return 0.0

        preds = np.array(list(self._predictions))
        acts = np.array(list(self._actuals))

        # Handle constant arrays
        if np.std(preds) == 0 or np.std(acts) == 0:
            return 0.0

        return float(np.corrcoef(preds, acts)[0, 1])

    @property
    def prediction_bias(self) -> float:
        """Average bias in predictions (positive = overestimating)."""
        if not self._predictions or not self._actuals:
            return 0.0

        preds = np.array(list(self._predictions))
        acts = np.array(list(self._actuals))

        return float(np.mean(preds - acts))

    def reset(self) -> None:
        """Reset tracker state."""
        self._predictions.clear()
        self._actuals.clear()
        self._errors.clear()


class ActionDiversityTracker:
    """Track diversity of actions taken."""

    def __init__(self, num_actions: int, window_size: int = 1000):
        """Initialize action diversity tracker.

        Args:
            num_actions: Total number of possible actions.
            window_size: Size of action history window.
        """
        self.num_actions = num_actions
        self.window_size = window_size
        self._action_history: deque[int] = deque(maxlen=window_size)
        self._action_counts: dict[int, int] = {}
        self._total_actions = 0

    def add(self, action: int) -> None:
        """Add an action observation.

        Args:
            action: Action taken.
        """
        self._action_history.append(action)
        self._action_counts[action] = self._action_counts.get(action, 0) + 1
        self._total_actions += 1

    @property
    def unique_actions(self) -> int:
        """Number of unique actions used."""
        return len(self._action_counts)

    @property
    def action_entropy(self) -> float:
        """Entropy of action distribution (higher = more diverse)."""
        if not self._action_counts:
            return 0.0

        total = sum(self._action_counts.values())
        probs = np.array([count / total for count in self._action_counts.values()])

        return float(-np.sum(probs * np.log(probs + 1e-10)))

    @property
    def diversity_ratio(self) -> float:
        """Ratio of unique actions to total possible actions."""
        return self.unique_actions / self.num_actions

    @property
    def most_common_actions(self) -> list[tuple[int, int]]:
        """Get most common actions with their counts."""
        sorted_actions = sorted(
            self._action_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_actions[:10]

    def reset(self) -> None:
        """Reset tracker state."""
        self._action_history.clear()
        self._action_counts.clear()
        self._total_actions = 0


class TrainingDiagnostics:
    """Main diagnostics class for tracking training progress."""

    def __init__(
        self,
        num_actions: int,
        config: DiagnosticsConfig | None = None,
    ):
        """Initialize training diagnostics.

        Args:
            num_actions: Total number of possible actions.
            config: Diagnostics configuration.
        """
        self.config = config or DiagnosticsConfig()
        self.num_actions = num_actions

        # Initialize trackers
        self.entropy_tracker = PolicyEntropyTracker(self.config.window_size)
        self.value_tracker = ValuePredictionTracker(self.config.window_size)
        self.action_tracker = ActionDiversityTracker(num_actions, self.config.window_size)

        # Episode tracking
        self._current_episode_entropies: list[float] = []
        self._current_episode_values: list[float] = []
        self._current_episode_actions: list[int] = []

        # Iteration tracking
        self._episode_metrics: list[EpisodeMetrics] = []
        self._iteration_metrics: list[IterationMetrics] = []

        # Warnings
        self._warnings: list[str] = []

    def on_step(
        self,
        policy: np.ndarray,
        value: float,
        action: int,
        action_mask: np.ndarray | None = None,
    ) -> None:
        """Record a training step.

        Args:
            policy: Policy probability distribution.
            value: Value prediction.
            action: Action taken.
            action_mask: Optional mask for valid actions.
        """
        # Track entropy
        if self.config.track_action_distribution:
            entropy = self.entropy_tracker.add(policy, action_mask)
            self._current_episode_entropies.append(entropy)

        # Track value
        if self.config.track_value_predictions:
            self._current_episode_values.append(value)

        # Track action
        self.action_tracker.add(action)
        self._current_episode_actions.append(action)

    def on_episode_end(
        self,
        episode_id: int,
        turns: int,
        total_reward: float,
        outcome: str,
        shaped_reward: float | None = None,
    ) -> EpisodeMetrics:
        """Record episode completion.

        Args:
            episode_id: Episode identifier.
            turns: Number of turns in episode.
            total_reward: Total reward received.
            outcome: Episode outcome ("win", "loss", "draw", "timeout").
            shaped_reward: Optional shaped reward total.

        Returns:
            EpisodeMetrics for this episode.
        """
        # Calculate averages
        avg_entropy = (
            np.mean(self._current_episode_entropies)
            if self._current_episode_entropies
            else 0.0
        )
        avg_value = (
            np.mean(self._current_episode_values)
            if self._current_episode_values
            else 0.0
        )

        # Count actions
        action_counts: dict[int, int] = {}
        for action in self._current_episode_actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        # Create metrics
        metrics = EpisodeMetrics(
            episode_id=episode_id,
            turns=turns,
            total_reward=total_reward,
            outcome=outcome,
            avg_policy_entropy=avg_entropy,
            avg_value_prediction=avg_value,
            action_counts=action_counts,
            shaped_reward=shaped_reward,
        )

        self._episode_metrics.append(metrics)

        # Track value prediction accuracy
        if self.config.track_value_predictions and self._current_episode_values:
            avg_pred = np.mean(self._current_episode_values)
            self.value_tracker.add(avg_pred, total_reward)

        # Reset episode tracking
        self._current_episode_entropies.clear()
        self._current_episode_values.clear()
        self._current_episode_actions.clear()

        return metrics

    def on_iteration_end(
        self,
        iteration: int,
        policy_loss: float,
        value_loss: float,
        total_loss: float,
    ) -> IterationMetrics:
        """Record training iteration completion.

        Args:
            iteration: Iteration number.
            policy_loss: Policy loss from training.
            value_loss: Value loss from training.
            total_loss: Total loss from training.

        Returns:
            IterationMetrics for this iteration.
        """
        if not self._episode_metrics:
            return IterationMetrics(
                iteration=iteration,
                num_episodes=0,
                avg_turns=0.0,
                win_rate=0.0,
                loss_rate=0.0,
                draw_rate=0.0,
                timeout_rate=0.0,
                avg_reward=0.0,
                policy_loss=policy_loss,
                value_loss=value_loss,
                total_loss=total_loss,
                avg_entropy=0.0,
                action_diversity=0.0,
            )

        # Calculate metrics from episodes
        num_episodes = len(self._episode_metrics)
        avg_turns = np.mean([m.turns for m in self._episode_metrics])
        avg_reward = np.mean([m.total_reward for m in self._episode_metrics])
        avg_entropy = np.mean([m.avg_policy_entropy for m in self._episode_metrics])

        # Calculate outcome rates
        outcomes = [m.outcome for m in self._episode_metrics]
        win_rate = outcomes.count("win") / num_episodes
        loss_rate = outcomes.count("loss") / num_episodes
        draw_rate = outcomes.count("draw") / num_episodes
        timeout_rate = outcomes.count("timeout") / num_episodes

        metrics = IterationMetrics(
            iteration=iteration,
            num_episodes=num_episodes,
            avg_turns=avg_turns,
            win_rate=win_rate,
            loss_rate=loss_rate,
            draw_rate=draw_rate,
            timeout_rate=timeout_rate,
            avg_reward=avg_reward,
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            avg_entropy=avg_entropy,
            action_diversity=self.action_tracker.diversity_ratio,
        )

        self._iteration_metrics.append(metrics)

        # Check for warnings
        self._check_warnings(metrics)

        # Clear episode metrics for next iteration
        self._episode_metrics.clear()

        return metrics

    def _check_warnings(self, metrics: IterationMetrics) -> None:
        """Check for training issues and add warnings.

        Args:
            metrics: Current iteration metrics.
        """
        # Check entropy collapse
        if metrics.avg_entropy < self.config.entropy_min_threshold:
            self._warnings.append(
                f"Iteration {metrics.iteration}: Policy entropy collapse "
                f"({metrics.avg_entropy:.4f} < {self.config.entropy_min_threshold})",
            )

        # Check value loss stagnation
        if metrics.value_loss < self.config.value_loss_threshold:
            self._warnings.append(
                f"Iteration {metrics.iteration}: Value loss very low "
                f"({metrics.value_loss:.6f}), value head may not be learning",
            )

        # Check for no wins
        if len(self._iteration_metrics) >= 3:
            recent_wins = [m.win_rate for m in self._iteration_metrics[-3:]]
            if all(w == 0 for w in recent_wins):
                self._warnings.append(
                    f"Iteration {metrics.iteration}: No wins in last 3 iterations",
                )

    @property
    def warnings(self) -> list[str]:
        """Get current warnings."""
        return self._warnings.copy()

    def clear_warnings(self) -> None:
        """Clear all warnings."""
        self._warnings.clear()

    def get_summary(self) -> dict[str, Any]:
        """Get diagnostic summary.

        Returns:
            Dictionary with diagnostic summary.
        """
        summary = {
            "total_iterations": len(self._iteration_metrics),
            "entropy": {
                "current": self.entropy_tracker.current,
                "moving_avg": self.entropy_tracker.moving_average,
            },
            "value_prediction": {
                "mae": self.value_tracker.mean_absolute_error,
                "correlation": self.value_tracker.correlation,
                "bias": self.value_tracker.prediction_bias,
            },
            "action_diversity": {
                "unique_actions": self.action_tracker.unique_actions,
                "diversity_ratio": self.action_tracker.diversity_ratio,
                "action_entropy": self.action_tracker.action_entropy,
                "most_common": self.action_tracker.most_common_actions,
            },
            "warnings": self._warnings.copy(),
        }

        if self._iteration_metrics:
            latest = self._iteration_metrics[-1]
            summary["latest_iteration"] = {
                "iteration": latest.iteration,
                "win_rate": latest.win_rate,
                "avg_turns": latest.avg_turns,
                "total_loss": latest.total_loss,
                "policy_loss": latest.policy_loss,
                "value_loss": latest.value_loss,
            }

        return summary

    def get_learning_curves(self) -> dict[str, list[float]]:
        """Get learning curve data.

        Returns:
            Dictionary with lists of metrics over iterations.
        """
        if not self._iteration_metrics:
            return {
                "iterations": [],
                "win_rate": [],
                "total_loss": [],
                "policy_loss": [],
                "value_loss": [],
                "avg_entropy": [],
                "avg_reward": [],
            }

        return {
            "iterations": [m.iteration for m in self._iteration_metrics],
            "win_rate": [m.win_rate for m in self._iteration_metrics],
            "total_loss": [m.total_loss for m in self._iteration_metrics],
            "policy_loss": [m.policy_loss for m in self._iteration_metrics],
            "value_loss": [m.value_loss for m in self._iteration_metrics],
            "avg_entropy": [m.avg_entropy for m in self._iteration_metrics],
            "avg_reward": [m.avg_reward for m in self._iteration_metrics],
        }

    def reset(self) -> None:
        """Reset all diagnostics."""
        self.entropy_tracker.reset()
        self.value_tracker.reset()
        self.action_tracker.reset()
        self._current_episode_entropies.clear()
        self._current_episode_values.clear()
        self._current_episode_actions.clear()
        self._episode_metrics.clear()
        self._iteration_metrics.clear()
        self._warnings.clear()


def create_diagnostics(
    num_actions: int,
    window_size: int = 100,
    **kwargs,
) -> TrainingDiagnostics:
    """Factory function to create training diagnostics.

    Args:
        num_actions: Total number of possible actions.
        window_size: Size of moving average window.
        **kwargs: Additional arguments for DiagnosticsConfig.

    Returns:
        Configured TrainingDiagnostics instance.
    """
    config = DiagnosticsConfig(window_size=window_size, **kwargs)
    return TrainingDiagnostics(num_actions=num_actions, config=config)
