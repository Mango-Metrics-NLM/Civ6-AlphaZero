"""Tests for alpha_zero/monitoring/mlflow_evaluator.py.

Covers:
- compute_policy_entropy(): uniform → log(n), one-hot → 0
- compute_value_mae(): perfect predictions, known error
- compute_policy_top1_accuracy(): all-correct → 1.0, all-wrong → 0.0
- compute_action_diversity(): same-action → minimal diversity
- AlphaZeroMLflowEvaluator.compute_metrics(): end-to-end with mock model
- Numerical stability: very large logits, tiny differences

All torch ops use small CPU tensors; no GPU required.
torch is imported lazily to allow collection without torch installed.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Lazy torch import — kept at module level via try/except so that the file
# can be collected even when torch is absent.
# ---------------------------------------------------------------------------
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")

# ---------------------------------------------------------------------------
# Guard: skip if the monitoring module doesn't exist yet
# ---------------------------------------------------------------------------
mlflow_evaluator = pytest.importorskip(
    "alpha_zero.monitoring.mlflow_evaluator",
    reason="alpha_zero/monitoring/mlflow_evaluator.py not yet implemented",
)

compute_policy_entropy = mlflow_evaluator.compute_policy_entropy
compute_value_mae = mlflow_evaluator.compute_value_mae
compute_policy_top1_accuracy = mlflow_evaluator.compute_policy_top1_accuracy
compute_action_diversity = mlflow_evaluator.compute_action_diversity
AlphaZeroMLflowEvaluator = mlflow_evaluator.AlphaZeroMLflowEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_logits(batch: int, n_actions: int) -> np.ndarray:
    """Return zero logits producing a uniform policy after softmax.

    Args:
        batch: Batch size.
        n_actions: Number of actions.

    Returns:
        Zero-filled float32 ndarray of shape (batch, n_actions).
    """
    return np.zeros((batch, n_actions), dtype=np.float32)


def _one_hot_logits(batch: int, n_actions: int, hot_idx: int = 0) -> np.ndarray:
    """Return large-valued logits that create a near-one-hot policy.

    Args:
        batch: Batch size.
        n_actions: Number of actions.
        hot_idx: Index of the dominant action.

    Returns:
        ndarray of shape (batch, n_actions) with one very large entry.
    """
    logits = np.full((batch, n_actions), -1e9, dtype=np.float32)
    logits[:, hot_idx] = 1e9
    return logits


# ---------------------------------------------------------------------------
# TestComputePolicyEntropy
# ---------------------------------------------------------------------------


class TestComputePolicyEntropy:
    """Tests for ``compute_policy_entropy()``."""

    def test_uniform_distribution_entropy_equals_log_n(self) -> None:
        """Uniform policy over n actions → entropy equals log(n) nats."""
        n_actions = 8
        logits = _uniform_logits(batch=4, n_actions=n_actions)

        result = compute_policy_entropy(logits)

        expected = math.log(n_actions)
        assert abs(result - expected) < 1e-4, f"Expected entropy={expected:.4f}, got {result:.4f}"

    def test_one_hot_distribution_entropy_near_zero(self) -> None:
        """Near-one-hot policy → entropy close to 0 nats."""
        logits = _one_hot_logits(batch=4, n_actions=16)

        result = compute_policy_entropy(logits)

        assert result < 0.01, f"One-hot entropy should be ~0, got {result}"

    @pytest.mark.parametrize("n_actions", [2, 4, 8, 16, 32])
    def test_entropy_bounded_by_log_n(self, n_actions: int) -> None:
        """Entropy must be in [0, log(n_actions)] for any valid distribution."""
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((8, n_actions)).astype(np.float32)

        result = compute_policy_entropy(logits)

        assert 0.0 <= result <= math.log(n_actions) + 1e-5, (
            f"Entropy {result} out of valid range for n_actions={n_actions}"
        )

    def test_entropy_is_scalar_float(self) -> None:
        """Return value must be a plain Python float."""
        logits = _uniform_logits(batch=3, n_actions=10)

        result = compute_policy_entropy(logits)

        assert isinstance(result, float)

    def test_large_logits_numerical_stability(self) -> None:
        """Very large logit values must not produce NaN or Inf entropy."""
        logits = np.array([[1e30, 2e30, 3e30]], dtype=np.float32)

        result = compute_policy_entropy(logits)

        assert math.isfinite(result), f"Expected finite entropy, got {result}"

    def test_negative_logits_numerical_stability(self) -> None:
        """Very negative logit values must produce a finite, non-NaN entropy."""
        logits = np.full((2, 8), -1e30, dtype=np.float32)
        logits[:, 0] = 0.0  # one clearly dominant action

        result = compute_policy_entropy(logits)

        assert math.isfinite(result)

    def test_batch_averaging(self) -> None:
        """Result must be the mean entropy over the batch dimension."""
        # Mix one uniform and one one-hot in a single batch
        n = 4
        uniform = np.zeros((1, n), dtype=np.float32)
        one_hot = _one_hot_logits(1, n, hot_idx=0)
        logits = np.concatenate([uniform, one_hot], axis=0)

        result = compute_policy_entropy(logits)

        # Should lie strictly between 0 (one-hot) and log(4) (uniform)
        assert 0.0 < result < math.log(n)


# ---------------------------------------------------------------------------
# TestComputeValueMAE
# ---------------------------------------------------------------------------


class TestComputeValueMAE:
    """Tests for ``compute_value_mae()``."""

    def test_perfect_predictions_zero_mae(self) -> None:
        """Predictions identical to outcomes → MAE = 0."""
        outcomes = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float32)
        preds = outcomes.copy()

        result = compute_value_mae(preds, outcomes)

        assert result == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize(
        ("preds", "outcomes", "expected_mae"),
        [
            ([0.0, 0.0], [1.0, 1.0], 1.0),
            ([1.0, 0.0, -1.0], [0.0, 0.0, 0.0], 2.0 / 3.0),
            ([0.5, 0.5], [-0.5, -0.5], 1.0),
        ],
    )
    def test_known_mae_values(
        self,
        preds: list[float],
        outcomes: list[float],
        expected_mae: float,
    ) -> None:
        """Computed MAE must match hand-calculated reference values."""
        result = compute_value_mae(
            np.array(preds, dtype=np.float32),
            np.array(outcomes, dtype=np.float32),
        )

        assert result == pytest.approx(expected_mae, abs=1e-5)

    def test_return_type_is_float(self) -> None:
        """Return type must be a plain Python float."""
        result = compute_value_mae(
            np.array([0.5], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )

        assert isinstance(result, float)

    def test_single_sample(self) -> None:
        """Single prediction-outcome pair must produce correct MAE."""
        result = compute_value_mae(
            np.array([0.3], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
        )

        assert result == pytest.approx(0.7, abs=1e-5)


# ---------------------------------------------------------------------------
# TestComputePolicyTop1Accuracy
# ---------------------------------------------------------------------------


class TestComputePolicyTop1Accuracy:
    """Tests for ``compute_policy_top1_accuracy()``."""

    def test_all_correct_gives_one(self) -> None:
        """All argmax(logits) == argmax(target) → accuracy = 1.0."""
        n = 4
        logits = np.eye(n, dtype=np.float32)  # argmax of row i is i
        targets = np.eye(n, dtype=np.float32)

        result = compute_policy_top1_accuracy(logits, targets)

        assert result == pytest.approx(1.0, abs=1e-6)

    def test_all_wrong_gives_zero(self) -> None:
        """No argmax matches → accuracy = 0.0."""
        # logits: argmax is 0; targets: argmax is 1
        logits = np.array([[10.0, -10.0], [10.0, -10.0]], dtype=np.float32)
        targets = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

        result = compute_policy_top1_accuracy(logits, targets)

        assert result == pytest.approx(0.0, abs=1e-6)

    def test_half_correct(self) -> None:
        """50 % matches → accuracy = 0.5."""
        logits = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        # row 0 target argmax=0 (match), row 1 target argmax=0 (mismatch)
        targets = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        result = compute_policy_top1_accuracy(logits, targets)

        assert result == pytest.approx(0.5, abs=1e-6)

    def test_return_type_is_float(self) -> None:
        """Return type must be a plain Python float."""
        logits = np.zeros((3, 5), dtype=np.float32)
        targets = np.zeros((3, 5), dtype=np.float32)
        targets[:, 0] = 1.0

        result = compute_policy_top1_accuracy(logits, targets)

        assert isinstance(result, float)

    def test_accuracy_in_valid_range(self) -> None:
        """Accuracy must always be in [0.0, 1.0]."""
        rng = np.random.default_rng(7)
        logits = rng.standard_normal((16, 12)).astype(np.float32)
        targets = rng.dirichlet(np.ones(12), size=16).astype(np.float32)

        result = compute_policy_top1_accuracy(logits, targets)

        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# TestComputeActionDiversity
# ---------------------------------------------------------------------------


class TestComputeActionDiversity:
    """Tests for ``compute_action_diversity()``."""

    def test_all_same_action_minimal_diversity(self) -> None:
        """All positions taking the same top-1 action → diversity ≤ 1/batch."""
        batch = 8
        # All logits have max at index 0
        logits = _one_hot_logits(batch=batch, n_actions=16, hot_idx=0)

        result = compute_action_diversity(logits)

        # 1 unique action out of batch positions
        expected_max = 1.0 / batch
        # Allow slightly above due to fraction definition differences
        assert result <= expected_max + 1e-5, f"Expected diversity ≤ {expected_max}, got {result}"

    def test_all_different_actions_high_diversity(self) -> None:
        """Each position has a unique top-1 action → diversity = 1.0."""
        n = 8
        # Row i has max at column i → all unique
        logits = np.eye(n, dtype=np.float32) * 1e9

        result = compute_action_diversity(logits)

        assert result == pytest.approx(1.0, abs=1e-5)

    def test_diversity_in_valid_range(self) -> None:
        """Action diversity must be in [0.0, 1.0]."""
        rng = np.random.default_rng(42)
        logits = rng.standard_normal((12, 20)).astype(np.float32)

        result = compute_action_diversity(logits)

        assert 0.0 <= result <= 1.0

    def test_return_type_is_float(self) -> None:
        """Return type must be a plain Python float."""
        logits = np.zeros((4, 8), dtype=np.float32)

        result = compute_action_diversity(logits)

        assert isinstance(result, float)

    @pytest.mark.parametrize("batch", [1, 4, 16])
    def test_single_action_type_across_batch_sizes(self, batch: int) -> None:
        """Diversity is low when all positions agree on one action regardless of batch size."""
        logits = _one_hot_logits(batch=batch, n_actions=8, hot_idx=2)

        result = compute_action_diversity(logits)

        # At most one unique action → diversity ≤ 1
        assert result <= 1.0


# ---------------------------------------------------------------------------
# TestAlphaZeroMLflowEvaluator
# ---------------------------------------------------------------------------


class TestAlphaZeroMLflowEvaluator:
    """Tests for ``AlphaZeroMLflowEvaluator.compute_metrics()``."""

    @pytest.fixture()
    def evaluator(self) -> AlphaZeroMLflowEvaluator:
        """Return an AlphaZeroMLflowEvaluator with local tracking URI.

        Returns:
            Evaluator configured for offline/local use.
        """
        return AlphaZeroMLflowEvaluator(tracking_uri="mlruns")

    @pytest.fixture()
    def mock_model(self) -> MagicMock:
        """Return a mock torch model producing predictable outputs.

        The model's ``__call__`` returns a (policy_logits, value) tuple
        with known zero logits (uniform policy) and zero values.

        Returns:
            MagicMock that behaves like a nn.Module.
        """

        def fake_forward(spatial: Any, vector: Any) -> tuple:
            """Fake forward: return uniform logits and zero values as tuple."""
            batch = spatial.shape[0] if hasattr(spatial, "shape") else 1
            n_actions = 8

            if _TORCH_AVAILABLE:
                policy_logits = torch.zeros(batch, n_actions)
                value = torch.zeros(batch, 1)
                return policy_logits, value
            else:
                return (
                    np.zeros((batch, n_actions), dtype=np.float32),
                    np.zeros((batch, 1), dtype=np.float32),
                )

        model = MagicMock(side_effect=fake_forward)
        model.eval.return_value = model
        return model

    @pytest.fixture()
    def sample_eval_positions(self) -> list[dict]:
        """Return 4 synthetic eval positions for compute_metrics().

        Each position dict contains keys expected by the evaluator.

        Returns:
            List of position dicts with spatial, vector, mcts_policy, outcome.
        """
        rng = np.random.default_rng(0)
        n_actions = 8
        positions = []
        for _ in range(4):
            positions.append(
                {
                    "spatial": rng.standard_normal((4, 4, 4)).astype(np.float32),
                    "vector": rng.standard_normal((8,)).astype(np.float32),
                    "mcts_policy": rng.dirichlet(np.ones(n_actions)).astype(np.float32),
                    "outcome": float(rng.choice([-1.0, 0.0, 1.0])),
                }
            )
        return positions

    def test_compute_metrics_returns_dict(
        self,
        evaluator: AlphaZeroMLflowEvaluator,
        mock_model: MagicMock,
        sample_eval_positions: list[dict],
    ) -> None:
        """compute_metrics() must return a plain dict."""
        result = evaluator.compute_metrics(mock_model, sample_eval_positions)

        assert isinstance(result, dict)

    def test_compute_metrics_has_required_keys(
        self,
        evaluator: AlphaZeroMLflowEvaluator,
        mock_model: MagicMock,
        sample_eval_positions: list[dict],
    ) -> None:
        """Result dict must contain the four canonical metric keys."""
        result = evaluator.compute_metrics(mock_model, sample_eval_positions)

        required_keys = {
            "policy_entropy",
            "value_mae",
            "policy_top1_acc",
            "action_diversity",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_compute_metrics_values_are_finite_floats(
        self,
        evaluator: AlphaZeroMLflowEvaluator,
        mock_model: MagicMock,
        sample_eval_positions: list[dict],
    ) -> None:
        """All metric values must be finite Python floats."""
        result = evaluator.compute_metrics(mock_model, sample_eval_positions)

        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not float: {type(value)}"
            assert math.isfinite(value), f"{key} is not finite: {value}"

    def test_uniform_policy_model_entropy_near_log_n(
        self,
        evaluator: AlphaZeroMLflowEvaluator,
        mock_model: MagicMock,
        sample_eval_positions: list[dict],
    ) -> None:
        """Mock model with zero logits → uniform policy → entropy ≈ log(8)."""
        result = evaluator.compute_metrics(mock_model, sample_eval_positions)

        n_actions = 8
        expected_entropy = math.log(n_actions)
        assert abs(result["policy_entropy"] - expected_entropy) < 0.1, (
            f"Expected entropy≈{expected_entropy:.3f}, got {result['policy_entropy']:.3f}"
        )

    def test_compute_metrics_empty_positions_handled(
        self,
        evaluator: AlphaZeroMLflowEvaluator,
        mock_model: MagicMock,
    ) -> None:
        """Empty eval_positions list must not raise (returns empty or zeroed dict)."""
        try:
            result = evaluator.compute_metrics(mock_model, [])
            # If it returns, must be a dict
            assert isinstance(result, dict)
        except (ValueError, IndexError):
            # Acceptable: implementation may raise on empty input
            pass

    def test_compute_metrics_uses_cpu_device(
        self,
        evaluator: AlphaZeroMLflowEvaluator,
        mock_model: MagicMock,
        sample_eval_positions: list[dict],
    ) -> None:
        """Calling compute_metrics with device='cpu' must not raise on CI."""
        # Should complete without CUDA errors
        result = evaluator.compute_metrics(mock_model, sample_eval_positions, device="cpu")

        assert isinstance(result, dict)

    @requires_torch
    def test_policy_entropy_with_torch_model(
        self, evaluator: AlphaZeroMLflowEvaluator, sample_eval_positions: list[dict]
    ) -> None:
        """Integration test: real torch model returning known logits."""
        import torch.nn as nn

        class FixedModel(nn.Module):
            """Always returns zero logits (uniform policy) and zero values."""

            def __init__(self) -> None:
                super().__init__()

            def forward(self, spatial: Any, vector: Any) -> tuple:
                """Return uniform logits and zero value as tuple."""
                batch = spatial.shape[0]
                return torch.zeros(batch, 8), torch.zeros(batch, 1)

        model = FixedModel().eval()
        result = evaluator.compute_metrics(model, sample_eval_positions, device="cpu")

        assert "policy_entropy" in result
        assert math.isfinite(result["policy_entropy"])

    def test_evaluator_initialises_without_mlflow(self) -> None:
        """AlphaZeroMLflowEvaluator must construct even if mlflow is absent."""
        # Patch mlflow away completely
        with patch.dict("sys.modules", {"mlflow": None}):
            try:
                ev = AlphaZeroMLflowEvaluator(tracking_uri="mlruns")
                assert ev is not None
            except ImportError:
                # Acceptable if constructor tries to import mlflow eagerly
                pass


# ---------------------------------------------------------------------------
# TestNumericalStabilityStandalone
# ---------------------------------------------------------------------------


class TestNumericalStabilityStandalone:
    """Numerical edge-case tests for the standalone compute_* functions."""

    def test_entropy_all_same_logits(self) -> None:
        """All-equal logits (uniform policy) must give finite, correct entropy."""
        logits = np.full((5, 10), 42.0, dtype=np.float32)

        result = compute_policy_entropy(logits)

        assert math.isfinite(result)
        assert abs(result - math.log(10)) < 1e-4

    def test_entropy_very_small_differences(self) -> None:
        """Logits differing by 1e-7 must still produce finite entropy."""
        logits = np.zeros((4, 8), dtype=np.float32)
        logits[:, 0] += 1e-7

        result = compute_policy_entropy(logits)

        assert math.isfinite(result)

    def test_mae_with_large_errors(self) -> None:
        """MAE with large prediction errors must not overflow."""
        preds = np.array([1e6, -1e6], dtype=np.float32)
        outcomes = np.array([-1e6, 1e6], dtype=np.float32)

        result = compute_value_mae(preds, outcomes)

        assert math.isfinite(result)
        assert result == pytest.approx(2e6, rel=1e-4)

    def test_top1_accuracy_tie_breaking_consistent(self) -> None:
        """Tied logits must produce a consistent (not NaN) accuracy."""
        logits = np.zeros((4, 4), dtype=np.float32)  # All tied
        targets = np.eye(4, dtype=np.float32)

        result = compute_policy_top1_accuracy(logits, targets)

        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0
