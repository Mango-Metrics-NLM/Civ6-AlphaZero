"""
MLflow-integrated model evaluation for AlphaZero.

Provides custom AlphaZero-specific evaluation metrics:
- policy_entropy:   Mean action distribution entropy (higher = more exploratory)
- value_mae:        Mean absolute error of value head vs actual game outcomes
- policy_top1_acc:  Top-1 accuracy of policy vs MCTS-improved target policy
- action_diversity: Fraction of unique top-1 actions across evaluation positions

These metrics are computed directly from model forward passes and can be logged
to MLflow either via mlflow.evaluate() or directly via MLflowLogger.

Example::

    from alpha_zero.monitoring.mlflow_evaluator import AlphaZeroMLflowEvaluator

    evaluator = AlphaZeroMLflowEvaluator(tracking_uri="mlruns")
    metrics = evaluator.compute_metrics(model, eval_positions, device="cuda")
    print(metrics)
    # {'policy_entropy': 3.21, 'value_mae': 0.14, 'policy_top1_acc': 0.43, 'action_diversity': 0.87}
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-function metric computations (no torch dependency at call-site)
# ---------------------------------------------------------------------------


def compute_policy_entropy(policy_logits: Any) -> float:
    """Compute mean Shannon entropy of softmax(policy_logits) over a batch.

    Entropy is computed in nats (natural log base) then averaged over the
    batch dimension.  Higher entropy = more uniform / exploratory policy.

    Args:
        policy_logits: Array of shape ``(batch, n_actions)`` — raw logits
            (pre-softmax).  Accepts numpy arrays or torch tensors.

    Returns:
        Mean entropy in nats, averaged over the batch.
    """
    import numpy as np

    logits = np.asarray(policy_logits, dtype=np.float64)
    # Numerical stability: subtract row-wise max before softmax
    logits -= logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy_per_sample = -np.sum(probs * np.log(probs), axis=-1)
    return float(np.mean(entropy_per_sample))


def compute_value_mae(value_preds: Any, outcomes: Any) -> float:
    """Compute mean absolute error between value predictions and game outcomes.

    Args:
        value_preds: Array of shape ``(batch,)`` or ``(batch, 1)`` — model
            value outputs (should be in [-1, 1]).
        outcomes: Array of shape ``(batch,)`` — actual game outcomes
            (+1 = win, 0 = draw, -1 = loss, or float in [-1, 1]).

    Returns:
        Mean absolute error (lower = better calibration).
    """
    import numpy as np

    preds = np.asarray(value_preds, dtype=np.float64).flatten()
    targets = np.asarray(outcomes, dtype=np.float64).flatten()
    return float(np.mean(np.abs(preds - targets)))


def compute_policy_top1_accuracy(policy_logits: Any, target_policy: Any) -> float:
    """Top-1 accuracy: fraction of positions where argmax(logits) == argmax(target).

    The *target_policy* represents the MCTS-improved policy (a probability
    distribution).  Top-1 accuracy measures whether the model's greedy choice
    matches the MCTS best move.

    Args:
        policy_logits: Array of shape ``(batch, n_actions)`` — raw logits.
        target_policy: Array of shape ``(batch, n_actions)`` — target
            probability distribution (e.g. MCTS visit counts).

    Returns:
        Fraction in [0, 1] where argmax(logits) == argmax(target_policy).
    """
    import numpy as np

    logits = np.asarray(policy_logits, dtype=np.float64)
    targets = np.asarray(target_policy, dtype=np.float64)
    pred_actions = np.argmax(logits, axis=-1)
    target_actions = np.argmax(targets, axis=-1)
    return float(np.mean(pred_actions == target_actions))


def compute_action_diversity(policy_logits: Any) -> float:
    """Fraction of unique top-1 actions across the batch.

    Measures how diverse the model's greedy choices are across a set of
    positions.  Low diversity (always choosing the same action) can indicate
    a degenerate policy.

    Args:
        policy_logits: Array of shape ``(batch, n_actions)`` — raw logits.

    Returns:
        ``n_unique_actions / batch_size`` in (0, 1].
    """
    import numpy as np

    logits = np.asarray(policy_logits, dtype=np.float64)
    top1 = np.argmax(logits, axis=-1)
    n_unique = len(np.unique(top1))
    return float(n_unique) / max(len(top1), 1)


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------


class AlphaZeroMLflowEvaluator:
    """Computes and logs AlphaZero-specific evaluation metrics to MLflow.

    This class wraps the pure metric functions above and integrates them with
    ``MLflowLogger`` so evaluation results appear alongside training metrics
    in the same MLflow run.

    Example::

        evaluator = AlphaZeroMLflowEvaluator(tracking_uri="sqlite:///mlflow.db")
        metrics = evaluator.compute_metrics(model, eval_positions, device="cpu")
        # Attach to an active MLflow run:
        from alpha_zero.monitoring.mlflow_logger import MLflowLogger
        with MLflowLogger(...) as ml:
            ml.log_metrics({f"eval/{k}": v for k, v in metrics.items()})
    """

    def __init__(
        self,
        tracking_uri: str = "mlruns",
    ) -> None:
        """Initialise the evaluator.

        Args:
            tracking_uri: MLflow tracking URI (file-store or DB).
        """
        self.tracking_uri = tracking_uri

    def compute_metrics(
        self,
        model: Any,
        eval_positions: list[dict[str, Any]],
        *,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Run the model on evaluation positions and compute quality metrics.

        Each position in ``eval_positions`` is a dict with keys:
        - ``"spatial"``: numpy array of shape ``(channels, H, W)``
        - ``"vector"``: numpy array of shape ``(vector_dim,)``
        - ``"mcts_policy"``: numpy array of shape ``(n_actions,)`` — target
        - ``"outcome"``: float — actual game outcome

        Args:
            model: A ``Civ6TransformerNetwork`` (or compatible) in eval mode.
            eval_positions: List of position dicts as described above.
            device: PyTorch device string (``"cpu"`` or ``"cuda"``).
            batch_size: Number of positions to process per forward pass.

        Returns:
            Dict with keys ``policy_entropy``, ``value_mae``,
            ``policy_top1_acc``, ``action_diversity``.
        """
        import numpy as np

        try:
            import torch
        except ImportError:
            logger.warning("torch not installed — returning NaN metrics")
            return {
                "policy_entropy": float("nan"),
                "value_mae": float("nan"),
                "policy_top1_acc": float("nan"),
                "action_diversity": float("nan"),
            }

        if not eval_positions:
            logger.warning("No evaluation positions provided")
            return {
                "policy_entropy": 0.0,
                "value_mae": 0.0,
                "policy_top1_acc": 0.0,
                "action_diversity": 0.0,
            }

        dev = torch.device(device)
        model.eval()

        all_policy_logits: list[np.ndarray] = []
        all_values: list[float] = []
        all_outcomes: list[float] = []
        all_mcts_policies: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(eval_positions), batch_size):
                batch = eval_positions[i : i + batch_size]

                spatial = torch.stack(
                    [torch.from_numpy(np.asarray(p["spatial"], dtype=np.float32)) for p in batch]
                ).to(dev)
                vector = torch.stack(
                    [torch.from_numpy(np.asarray(p["vector"], dtype=np.float32)) for p in batch]
                ).to(dev)

                policy_logits, value = model(spatial, vector)

                all_policy_logits.append(policy_logits.cpu().numpy())
                all_values.extend(value.cpu().numpy().flatten().tolist())
                all_outcomes.extend(float(p["outcome"]) for p in batch)
                all_mcts_policies.append(
                    np.stack([np.asarray(p["mcts_policy"], dtype=np.float32) for p in batch])
                )

        logits_arr = np.concatenate(all_policy_logits, axis=0)
        mcts_arr = np.concatenate(all_mcts_policies, axis=0)
        values_arr = np.array(all_values)
        outcomes_arr = np.array(all_outcomes)

        return {
            "policy_entropy": compute_policy_entropy(logits_arr),
            "value_mae": compute_value_mae(values_arr, outcomes_arr),
            "policy_top1_acc": compute_policy_top1_accuracy(logits_arr, mcts_arr),
            "action_diversity": compute_action_diversity(logits_arr),
        }

    def evaluate_and_log(
        self,
        model: Any,
        eval_positions: list[dict[str, Any]],
        ml_logger: Any,  # MLflowLogger
        *,
        device: str = "cpu",
        step: int | None = None,
        metric_prefix: str = "eval",
    ) -> dict[str, float]:
        """Compute metrics and log them to the provided MLflowLogger.

        Args:
            model: Network in eval mode.
            eval_positions: List of evaluation position dicts.
            ml_logger: Active MLflowLogger instance.
            device: PyTorch device.
            step: Optional step number for the metrics.
            metric_prefix: Prefix for metric keys (default: ``"eval"``).

        Returns:
            Computed metrics dict (same as ``compute_metrics()``).
        """
        metrics = self.compute_metrics(model, eval_positions, device=device)
        ml_logger.log_metrics(
            {f"{metric_prefix}/{k}": v for k, v in metrics.items() if not math.isnan(v)},
            step=step,
        )
        return metrics
