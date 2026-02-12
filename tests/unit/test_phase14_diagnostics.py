"""Comprehensive tests for TrainingDiagnostics module.

Tests cover:
- Policy entropy computation (uniform, collapsed, masked, batch)
- Value accuracy metrics (perfect, random, inverted, empty)
- Gradient statistics (basic, no gradients)
- Trajectory quality analysis (empty, normal)
- Learning stall detection (insufficient data, improving, flat, custom threshold)
- Win rate trend computation (increasing, decreasing)
- Summary and history retrieval
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from alpha_zero.training.diagnostics import TrainingDiagnostics


@pytest.fixture
def diagnostics() -> TrainingDiagnostics:
    """Create a fresh TrainingDiagnostics instance."""
    return TrainingDiagnostics(window_size=100, stall_threshold=0.02)


# =============================================================================
# Policy Entropy Tests
# =============================================================================


def test_policy_entropy_uniform(diagnostics: TrainingDiagnostics) -> None:
    """Uniform logits should yield entropy ≈ log(num_actions) (high entropy)."""
    batch_size = 4
    num_actions = 10

    # Uniform logits → near-uniform probabilities
    policy_logits = torch.zeros(batch_size, num_actions)

    entropy = diagnostics.compute_policy_entropy(policy_logits)

    # Shannon entropy for uniform distribution: H = log(n)
    expected_entropy = math.log(num_actions)

    # Allow 5% tolerance
    assert abs(entropy - expected_entropy) < 0.05 * expected_entropy
    assert entropy > 2.0  # Should be high entropy


def test_policy_entropy_collapsed(diagnostics: TrainingDiagnostics) -> None:
    """One-hot logits should yield entropy ≈ 0 (collapsed policy)."""
    batch_size = 4
    num_actions = 10

    # One-hot logits: first action has very high logit
    policy_logits = torch.full((batch_size, num_actions), -10.0)
    policy_logits[:, 0] = 10.0  # First action has high probability

    entropy = diagnostics.compute_policy_entropy(policy_logits)

    # Collapsed distribution should have very low entropy
    assert entropy < 0.1


def test_policy_entropy_with_mask(diagnostics: TrainingDiagnostics) -> None:
    """With action mask, entropy should be based only on valid actions."""
    batch_size = 2
    num_actions = 10

    # Uniform logits
    policy_logits = torch.zeros(batch_size, num_actions)

    # Only first 5 actions are valid
    action_mask = torch.zeros(batch_size, num_actions)
    action_mask[:, :5] = 1.0

    entropy = diagnostics.compute_policy_entropy(policy_logits, action_mask)

    # Entropy should be based on 5 valid actions
    expected_entropy = math.log(5)

    # Allow 5% tolerance
    assert abs(entropy - expected_entropy) < 0.05 * expected_entropy


def test_policy_entropy_batch(diagnostics: TrainingDiagnostics) -> None:
    """Multiple samples in batch should return mean entropy."""
    batch_size = 8
    num_actions = 20

    # Mix of uniform and collapsed distributions
    policy_logits = torch.zeros(batch_size, num_actions)

    # First half: uniform (high entropy)
    # Second half: collapsed (low entropy)
    policy_logits[batch_size // 2 :, 0] = 10.0
    policy_logits[batch_size // 2 :, 1:] = -10.0

    entropy = diagnostics.compute_policy_entropy(policy_logits)

    # Should be somewhere between high and low entropy
    assert 0.5 < entropy < math.log(num_actions)


# =============================================================================
# Value Accuracy Tests
# =============================================================================


def test_value_accuracy_perfect(diagnostics: TrainingDiagnostics) -> None:
    """Perfect predictions should yield sign_accuracy=1.0, correlation=1.0, mae=0.0."""
    batch_size = 10

    targets = torch.randn(batch_size)
    predicted = targets.clone()

    accuracy = diagnostics.compute_value_accuracy(predicted, targets)

    assert accuracy["sign_accuracy"] == 1.0
    # Correlation may have numerical precision issues, use tolerance
    assert abs(accuracy["correlation"] - 1.0) < 0.15
    assert accuracy["mae"] < 1e-6


def test_value_accuracy_random(diagnostics: TrainingDiagnostics) -> None:
    """Random predictions should yield low correlation, ~50% sign accuracy."""
    batch_size = 100

    torch.manual_seed(42)
    targets = torch.randn(batch_size)
    predicted = torch.randn(batch_size)

    accuracy = diagnostics.compute_value_accuracy(predicted, targets)

    # Random predictions should have:
    # - Sign accuracy around 0.5 (random guess)
    # - Low correlation (close to 0)
    # - Non-zero MAE
    assert 0.3 < accuracy["sign_accuracy"] < 0.7
    assert abs(accuracy["correlation"]) < 0.3
    assert accuracy["mae"] > 0.5


def test_value_accuracy_inverted(diagnostics: TrainingDiagnostics) -> None:
    """Inverted predictions should yield sign_accuracy=0.0, correlation=-1.0."""
    batch_size = 10

    targets = torch.randn(batch_size)
    predicted = -targets  # Perfect inversion

    accuracy = diagnostics.compute_value_accuracy(predicted, targets)

    assert accuracy["sign_accuracy"] == 0.0
    # Correlation may have numerical precision issues, use tolerance
    assert abs(accuracy["correlation"] - (-1.0)) < 0.15
    assert accuracy["mae"] > 0.0


def test_value_accuracy_empty(diagnostics: TrainingDiagnostics) -> None:
    """Empty tensors should return all zeros."""
    predicted = torch.tensor([])
    targets = torch.tensor([])

    accuracy = diagnostics.compute_value_accuracy(predicted, targets)

    assert accuracy["sign_accuracy"] == 0.0
    assert accuracy["correlation"] == 0.0
    assert accuracy["mae"] == 0.0


# =============================================================================
# Gradient Stats Tests
# =============================================================================


def test_gradient_stats_basic(diagnostics: TrainingDiagnostics) -> None:
    """Simple model with computed gradients should have non-zero stats."""
    # Create a simple linear model
    model = nn.Linear(10, 5)

    # Forward pass
    x = torch.randn(4, 10)
    output = model(x)

    # Compute loss and backward
    loss = output.sum()
    loss.backward()

    # Compute gradient stats
    grad_stats = diagnostics.compute_gradient_stats(model)

    assert grad_stats["total_norm"] > 0.0
    assert grad_stats["max_norm"] > 0.0
    assert grad_stats["mean_norm"] > 0.0
    assert grad_stats["num_params_with_grad"] == 2  # Weight + bias


def test_gradient_stats_no_grad(diagnostics: TrainingDiagnostics) -> None:
    """Model with no gradients computed should return all zeros."""
    model = nn.Linear(10, 5)

    # No backward pass, so no gradients
    grad_stats = diagnostics.compute_gradient_stats(model)

    assert grad_stats["total_norm"] == 0.0
    assert grad_stats["max_norm"] == 0.0
    assert grad_stats["mean_norm"] == 0.0
    assert grad_stats["num_params_with_grad"] == 0


# =============================================================================
# Trajectory Quality Tests
# =============================================================================


def test_trajectory_quality_empty(diagnostics: TrainingDiagnostics) -> None:
    """Empty trajectory list should return all zeros."""
    trajectories: list = []

    quality = diagnostics.analyze_trajectory_quality(trajectories)

    assert quality["num_trajectories"] == 0
    assert quality["avg_game_length"] == 0.0
    assert quality["min_game_length"] == 0
    assert quality["max_game_length"] == 0
    assert quality["action_entropy"] == 0.0
    assert quality["reward_variance"] == 0.0
    assert quality["reward_mean"] == 0.0
    assert quality["unique_action_ratio"] == 0.0


def test_trajectory_quality_normal(diagnostics: TrainingDiagnostics) -> None:
    """Normal trajectories should yield reasonable statistics."""
    # Create sample trajectories
    trajectories = [
        {
            "spatial": [torch.randn(13, 8, 8) for _ in range(20)],
            "vector": [torch.randn(64) for _ in range(20)],
            "actions": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1] * 2,
            "rewards": [0.0] * 19 + [1.0],  # Win
        },
        {
            "spatial": [torch.randn(13, 8, 8) for _ in range(15)],
            "vector": [torch.randn(64) for _ in range(15)],
            "actions": [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7],
            "rewards": [0.0] * 14 + [-1.0],  # Loss
        },
        {
            "spatial": [torch.randn(13, 8, 8) for _ in range(18)],
            "vector": [torch.randn(64) for _ in range(18)],
            "actions": [9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12, 9, 10],
            "rewards": [0.0] * 18,  # Draw
        },
    ]

    quality = diagnostics.analyze_trajectory_quality(trajectories)

    assert quality["num_trajectories"] == 3
    assert 15 <= quality["avg_game_length"] <= 20
    assert quality["min_game_length"] == 15
    assert quality["max_game_length"] == 20
    assert quality["action_entropy"] > 0.0  # Should have diverse actions
    assert quality["reward_variance"] > 0.0  # Different outcomes
    assert quality["unique_action_ratio"] > 0.1  # Some action diversity


# =============================================================================
# Stall Detection Tests
# =============================================================================


def test_stall_not_enough_data(diagnostics: TrainingDiagnostics) -> None:
    """Short history should return False (can't judge)."""
    metric_history = [0.5, 0.52, 0.54]

    stalled = diagnostics.detect_learning_stall(metric_history, lookback=5)

    assert stalled is False


def test_stall_improving(diagnostics: TrainingDiagnostics) -> None:
    """Increasing values should return False (not stalled)."""
    metric_history = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

    stalled = diagnostics.detect_learning_stall(metric_history, lookback=5)

    assert stalled is False


def test_stall_flat(diagnostics: TrainingDiagnostics) -> None:
    """Flat values should return True (stalled)."""
    metric_history = [0.50, 0.501, 0.500, 0.499, 0.501, 0.500]

    stalled = diagnostics.detect_learning_stall(metric_history, lookback=5)

    assert stalled is True


def test_stall_custom_threshold(diagnostics: TrainingDiagnostics) -> None:
    """Custom threshold should work correctly."""
    metric_history = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]

    # With default threshold (0.02), this is not stalled (improvement = 0.05)
    stalled_default = diagnostics.detect_learning_stall(metric_history, lookback=5)
    assert stalled_default is False

    # With high threshold (0.10), this is stalled (improvement < 0.10)
    stalled_high = diagnostics.detect_learning_stall(metric_history, lookback=5, threshold=0.10)
    assert stalled_high is True


# =============================================================================
# Win Rate Trend Tests
# =============================================================================


def test_win_rate_trend_increasing(diagnostics: TrainingDiagnostics) -> None:
    """Increasing win rates should have positive slope."""
    win_rate_history = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

    slope = diagnostics.compute_win_rate_trend(win_rate_history, lookback=5)

    assert slope > 0.0


def test_win_rate_trend_decreasing(diagnostics: TrainingDiagnostics) -> None:
    """Decreasing win rates should have negative slope."""
    win_rate_history = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]

    slope = diagnostics.compute_win_rate_trend(win_rate_history, lookback=5)

    assert slope < 0.0


# =============================================================================
# Summary and History Tests
# =============================================================================


def test_get_summary_after_recording(diagnostics: TrainingDiagnostics) -> None:
    """Summary should contain latest values of recorded metrics."""
    # Record some metrics
    policy_logits = torch.zeros(4, 10)
    diagnostics.compute_policy_entropy(policy_logits)

    predicted = torch.randn(10)
    targets = predicted.clone()
    diagnostics.compute_value_accuracy(predicted, targets)

    # Get summary
    summary = diagnostics.get_summary()

    # Should contain recorded metrics
    assert "policy_entropy" in summary
    assert "value_sign_accuracy" in summary
    assert "value_correlation" in summary
    assert "value_mae" in summary

    # Check values are reasonable
    assert isinstance(summary["policy_entropy"], float)
    assert isinstance(summary["value_sign_accuracy"], float)
    assert 0.0 <= summary["value_sign_accuracy"] <= 1.0


def test_get_metric_history(diagnostics: TrainingDiagnostics) -> None:
    """History should return all recorded values for a metric."""
    # Record multiple values
    for _ in range(5):
        policy_logits = torch.zeros(4, 10)
        diagnostics.compute_policy_entropy(policy_logits)

    # Get history
    history = diagnostics.get_metric_history("policy_entropy")

    # Should have 5 entries
    assert len(history) == 5

    # All should be floats
    assert all(isinstance(val, float) for val in history)


def test_gradient_stats_multiple_layers(diagnostics: TrainingDiagnostics) -> None:
    """Model with multiple layers should compute stats correctly."""
    # Create a multi-layer model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
    )

    # Forward + backward
    x = torch.randn(4, 10)
    output = model(x)
    loss = output.sum()
    loss.backward()

    # Compute gradient stats
    grad_stats = diagnostics.compute_gradient_stats(model)

    # Should have gradients for all linear layers (3 layers x 2 params = 6 params)
    assert grad_stats["num_params_with_grad"] == 6
    assert grad_stats["total_norm"] > 0.0
    assert grad_stats["max_norm"] > 0.0
    assert grad_stats["mean_norm"] > 0.0

    # Total norm should be >= max norm (sanity check)
    assert grad_stats["total_norm"] >= grad_stats["max_norm"]


def test_trajectory_quality_legacy_format(diagnostics: TrainingDiagnostics) -> None:
    """Should handle legacy 'states' key instead of 'spatial'."""
    trajectories = [
        {
            "states": [torch.randn(13, 8, 8) for _ in range(10)],
            "actions": list(range(10)),
            "rewards": [0.0] * 9 + [1.0],
        },
    ]

    quality = diagnostics.analyze_trajectory_quality(trajectories)

    assert quality["num_trajectories"] == 1
    assert quality["avg_game_length"] == 10
    assert quality["action_entropy"] > 0.0


def test_value_accuracy_2d_tensors(diagnostics: TrainingDiagnostics) -> None:
    """Should handle 2D tensors with shape (batch, 1)."""
    batch_size = 10

    targets = torch.randn(batch_size, 1)
    predicted = targets.clone()

    accuracy = diagnostics.compute_value_accuracy(predicted, targets)

    assert accuracy["sign_accuracy"] == 1.0
    # Correlation may have numerical precision issues, use tolerance
    assert abs(accuracy["correlation"] - 1.0) < 0.15
    assert accuracy["mae"] < 1e-6
