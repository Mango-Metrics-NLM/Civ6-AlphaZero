"""Training diagnostics module for learning signal validation.

Computes and tracks training health metrics including policy entropy,
value prediction accuracy, gradient statistics, and trajectory quality.
Designed to be called from any training loop without tight coupling.

All computation methods are pure functions returning plain dicts —
callers decide how to log (W&B, Python logging, etc.).

Usage:
    from alpha_zero.training.diagnostics import TrainingDiagnostics

    diagnostics = TrainingDiagnostics(window_size=100)

    # In training loop:
    entropy = diagnostics.compute_policy_entropy(policy_logits, action_mask)
    accuracy = diagnostics.compute_value_accuracy(predicted_values, target_values)
    grad_stats = diagnostics.compute_gradient_stats(model)

    # After selfplay:
    quality = diagnostics.analyze_trajectory_quality(trajectories)

    # In orchestrator:
    stalled = diagnostics.detect_learning_stall(win_rate_history, lookback=5)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class TrainingDiagnostics:
    """Compute and track training health metrics.

    Uses MetricsTracker internally for rolling-window statistics when available,
    with a lightweight fallback for environments without the full logging stack.
    """

    def __init__(
        self,
        window_size: int = 100,
        stall_threshold: float = 0.02,
    ) -> None:
        """Initialize diagnostics.

        Args:
            window_size: Rolling window size for trend tracking.
            stall_threshold: Minimum improvement to not be considered "stalled".
        """
        self._window_size = window_size
        self._stall_threshold = stall_threshold

        # Try to use MetricsTracker for rolling-window stats
        self._tracker: Any = None
        try:
            from alpha_zero.utils.logging_config import MetricsTracker

            self._tracker = MetricsTracker(window_size=window_size)
        except ImportError:
            logger.debug("MetricsTracker not available, using lightweight tracking")

        # Lightweight fallback storage
        self._history: dict[str, list[float]] = {}

    def _record(self, name: str, value: float) -> None:
        """Record a metric value to the internal tracker."""
        if self._tracker is not None:
            self._tracker.update(name, value)
        else:
            if name not in self._history:
                self._history[name] = []
            self._history[name].append(value)
            # Keep only window_size entries
            if len(self._history[name]) > self._window_size:
                self._history[name] = self._history[name][-self._window_size :]

    def compute_policy_entropy(
        self,
        policy_logits: Tensor,
        action_mask: Tensor | None = None,
    ) -> float:
        """Compute Shannon entropy of policy distribution.

        High entropy (~log(action_space)) means near-uniform (random) policy.
        Low entropy (~0) means collapsed/deterministic policy.
        Healthy training shows gradual decrease from high to moderate.

        Args:
            policy_logits: Raw logits from policy head, shape (batch, action_space).
            action_mask: Optional binary mask, shape (batch, action_space).
                1 = valid action, 0 = invalid.

        Returns:
            Mean entropy across the batch (in nats).
        """
        with torch.no_grad():
            if action_mask is not None:
                # Mask invalid actions with large negative value
                masked_logits = policy_logits.clone()
                masked_logits[action_mask == 0] = -1e9
            else:
                masked_logits = policy_logits

            log_probs = torch.log_softmax(masked_logits, dim=-1)
            probs = torch.softmax(masked_logits, dim=-1)

            # Shannon entropy: H = -sum(p * log(p))
            # Clamp to avoid log(0)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            mean_entropy = float(entropy.mean().item())

        self._record("policy_entropy", mean_entropy)
        logger.debug("Policy entropy: %.4f", mean_entropy)
        return mean_entropy

    def compute_value_accuracy(
        self,
        predicted: Tensor,
        targets: Tensor,
    ) -> dict[str, float]:
        """Compute value head prediction quality metrics.

        Args:
            predicted: Predicted values from value head, shape (batch,) or (batch, 1).
            targets: Target values (game outcomes), shape (batch,) or (batch, 1).

        Returns:
            Dict with keys:
                sign_accuracy: Fraction of correct win/loss sign predictions.
                correlation: Pearson correlation between predicted and actual.
                mae: Mean absolute error.
        """
        with torch.no_grad():
            pred = predicted.detach().flatten().float()
            tgt = targets.detach().flatten().float()

            if pred.numel() == 0:
                return {"sign_accuracy": 0.0, "correlation": 0.0, "mae": 0.0}

            # Sign accuracy: correct win/loss prediction (ignoring draws near 0)
            # Only count samples where target is clearly positive or negative
            nonzero_mask = tgt.abs() > 0.1
            if nonzero_mask.sum() > 0:
                pred_signs = torch.sign(pred[nonzero_mask])
                tgt_signs = torch.sign(tgt[nonzero_mask])
                sign_accuracy = float((pred_signs == tgt_signs).float().mean().item())
            else:
                sign_accuracy = 0.5  # No clear win/loss samples

            # Pearson correlation
            if pred.numel() > 1 and pred.std() > 1e-8 and tgt.std() > 1e-8:
                pred_centered = pred - pred.mean()
                tgt_centered = tgt - tgt.mean()
                correlation = float(
                    (pred_centered * tgt_centered).mean() / (pred.std() * tgt.std())
                )
                # Clamp to [-1, 1] for numerical stability
                correlation = max(-1.0, min(1.0, correlation))
            else:
                correlation = 0.0

            # Mean absolute error
            mae = float((pred - tgt).abs().mean().item())

        result = {
            "sign_accuracy": sign_accuracy,
            "correlation": correlation,
            "mae": mae,
        }

        self._record("value_sign_accuracy", sign_accuracy)
        self._record("value_correlation", correlation)
        self._record("value_mae", mae)

        logger.debug(
            "Value accuracy: sign=%.3f, corr=%.3f, mae=%.3f",
            sign_accuracy,
            correlation,
            mae,
        )
        return result

    def compute_gradient_stats(
        self,
        model: nn.Module,
    ) -> dict[str, float]:
        """Compute gradient statistics across model parameters.

        Should be called AFTER loss.backward() and BEFORE clip_grad_norm_().

        Args:
            model: The neural network model with computed gradients.

        Returns:
            Dict with keys:
                total_norm: L2 norm across all parameters.
                max_norm: Maximum gradient norm among all parameters.
                mean_norm: Mean gradient norm across parameters.
                num_params_with_grad: Number of parameters with gradients.
        """
        total_norm_sq = 0.0
        max_param_norm = 0.0
        norms: list[float] = []

        for param in model.parameters():
            if param.grad is not None:
                param_norm = float(param.grad.data.norm(2).item())
                total_norm_sq += param_norm**2
                max_param_norm = max(max_param_norm, param_norm)
                norms.append(param_norm)

        total_norm = math.sqrt(total_norm_sq)
        mean_norm = float(np.mean(norms)) if norms else 0.0

        result = {
            "total_norm": total_norm,
            "max_norm": max_param_norm,
            "mean_norm": mean_norm,
            "num_params_with_grad": len(norms),
        }

        self._record("grad_total_norm", total_norm)
        self._record("grad_max_norm", max_param_norm)

        logger.debug(
            "Gradient stats: total=%.4f, max=%.4f, mean=%.4f (%d params)",
            total_norm,
            max_param_norm,
            mean_norm,
            len(norms),
        )
        return result

    def analyze_trajectory_quality(
        self,
        trajectories: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Analyze quality and diversity of selfplay trajectories.

        Args:
            trajectories: List of trajectory dicts with keys:
                'spatial', 'vector', 'policies', 'actions', 'rewards'.

        Returns:
            Dict with keys:
                num_trajectories: Total trajectory count.
                avg_game_length: Mean number of steps per trajectory.
                min_game_length: Shortest trajectory.
                max_game_length: Longest trajectory.
                action_entropy: Shannon entropy of action distribution.
                reward_variance: Variance of final rewards.
                reward_mean: Mean of final rewards.
                unique_action_ratio: Fraction of unique actions used.
        """
        if not trajectories:
            return {
                "num_trajectories": 0,
                "avg_game_length": 0.0,
                "min_game_length": 0,
                "max_game_length": 0,
                "action_entropy": 0.0,
                "reward_variance": 0.0,
                "reward_mean": 0.0,
                "unique_action_ratio": 0.0,
            }

        # Game lengths
        lengths = []
        all_actions: list[int] = []
        final_rewards: list[float] = []

        for traj in trajectories:
            # Handle both 'spatial' (new) and 'states' (legacy) keys
            state_key = "spatial" if "spatial" in traj else "states"
            length = len(traj.get(state_key, []))
            lengths.append(length)

            if "actions" in traj:
                all_actions.extend(traj["actions"])

            rewards = traj.get("rewards")
            if rewards:
                # Final reward is the last element (all same in AlphaZero)
                final_rewards.append(float(rewards[-1]))

        avg_length = float(np.mean(lengths)) if lengths else 0.0
        min_length = min(lengths) if lengths else 0
        max_length = max(lengths) if lengths else 0

        # Action entropy: how diverse are the chosen actions?
        if all_actions:
            action_counts = np.bincount(all_actions)
            action_probs = action_counts[action_counts > 0] / len(all_actions)
            action_entropy = float(-np.sum(action_probs * np.log(action_probs + 1e-10)))
        else:
            action_entropy = 0.0

        # Reward statistics
        reward_variance = float(np.var(final_rewards)) if final_rewards else 0.0
        reward_mean = float(np.mean(final_rewards)) if final_rewards else 0.0

        # Unique action ratio
        unique_action_ratio = (
            len(set(all_actions)) / max(len(all_actions), 1) if all_actions else 0.0
        )

        result = {
            "num_trajectories": len(trajectories),
            "avg_game_length": avg_length,
            "min_game_length": min_length,
            "max_game_length": max_length,
            "action_entropy": action_entropy,
            "reward_variance": reward_variance,
            "reward_mean": reward_mean,
            "unique_action_ratio": unique_action_ratio,
        }

        logger.info(
            "Trajectory quality: %d games, avg_len=%.1f, action_entropy=%.2f, "
            "reward_mean=%.2f, reward_var=%.3f",
            len(trajectories),
            avg_length,
            action_entropy,
            reward_mean,
            reward_variance,
        )
        return result

    def detect_learning_stall(
        self,
        metric_history: Sequence[float],
        lookback: int = 5,
        threshold: float | None = None,
    ) -> bool:
        """Detect if learning has stalled.

        Returns True if the metric hasn't improved by more than threshold
        over the last `lookback` entries.

        Args:
            metric_history: Sequence of metric values (e.g., win rates per iteration).
            lookback: Number of recent entries to examine.
            threshold: Minimum improvement required. Defaults to self._stall_threshold.

        Returns:
            True if learning appears stalled.
        """
        if threshold is None:
            threshold = self._stall_threshold

        if len(metric_history) < lookback:
            return False  # Not enough data to judge

        recent = list(metric_history[-lookback:])
        improvement = max(recent) - min(recent)

        stalled = improvement < threshold
        if stalled:
            logger.warning(
                "Learning stall detected: improvement %.4f < threshold %.4f "
                "over last %d entries. Values: %s",
                improvement,
                threshold,
                lookback,
                [f"{v:.3f}" for v in recent],
            )

        return stalled

    def compute_win_rate_trend(
        self,
        win_rate_history: Sequence[float],
        lookback: int = 5,
    ) -> float:
        """Compute linear trend (slope) of win rate history.

        Positive slope = improving, negative = degrading, near-zero = stagnant.

        Args:
            win_rate_history: Sequence of win rates per iteration.
            lookback: Number of recent entries to use for trend.

        Returns:
            Slope of the linear fit (win rate change per iteration).
        """
        if len(win_rate_history) < 2:
            return 0.0

        recent = list(win_rate_history[-lookback:])
        x = np.arange(len(recent), dtype=np.float64)
        y = np.array(recent, dtype=np.float64)

        # Linear regression: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        x_var = ((x - x_mean) ** 2).sum()

        if x_var < 1e-10:
            return 0.0

        slope = float(((x - x_mean) * (y - y_mean)).sum() / x_var)
        return slope

    def get_summary(self) -> dict[str, Any]:
        """Get all current diagnostic metrics as a flat dict.

        Returns:
            Dict suitable for W&B logging (all keys are strings, values are floats).
        """
        if self._tracker is not None:
            all_metrics = self._tracker.get_all_metrics()
            return {
                name: summary["last"]
                for name, summary in all_metrics.items()
                if summary["count"] > 0
            }

        return {name: values[-1] if values else 0.0 for name, values in self._history.items()}

    def get_metric_history(self, name: str) -> list[float]:
        """Get full history for a named metric.

        Args:
            name: Metric name (e.g., 'policy_entropy', 'value_sign_accuracy').

        Returns:
            List of all recorded values.
        """
        if self._tracker is not None:
            return self._tracker.get_history(name)  # type: ignore[no-any-return]
        return list(self._history.get(name, []))


# ---------------------------------------------------------------------------
# Prequential (out-of-sample) diagnostics for the local training loop (LF-A13)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRecord:
    """One step actually played in a self-play game -- never a search rollout.

    ``action_type`` is ``decode_action(action)["type"]`` and ``success`` is the
    environment's ``info["action_success"]`` for that step.
    """

    game_idx: int
    action: int
    action_type: str
    success: bool


class _ExampleLike(Protocol):
    """The fields of ``training_civ6.TrainingExample`` these diagnostics read.

    Structural, because ``training_civ6`` imports this module and not the reverse.
    """

    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    policy: np.ndarray
    value: float


#: Positions per forward pass when scoring an iteration's fresh games.
PREQUENTIAL_BATCH_SIZE = 256


def _network_device(net: nn.Module) -> torch.device:
    first = next(iter(net.parameters()), None)
    return first.device if first is not None else torch.device("cpu")


def _predict(
    net: nn.Module, examples: Sequence[_ExampleLike], batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Scalar values and masked policy probabilities, in eval mode, without gradients.

    The network's training mode is restored afterwards. Eval mode matters for more
    than tidiness: a train-mode forward would update BatchNorm running statistics
    and so change the weights the next training phase starts from.
    """
    device = _network_device(net)
    was_training = net.training
    net.eval()
    values: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                chunk = examples[start : start + batch_size]
                spatial = torch.from_numpy(np.stack([e.spatial for e in chunk])).float()
                vector = torch.from_numpy(np.stack([e.vector for e in chunk])).float()
                mask = torch.from_numpy(np.stack([e.action_mask for e in chunk])).float()
                logits, value = net(spatial.to(device), vector.to(device), mask.to(device))
                masked = logits.float().masked_fill(mask.to(device) <= 0, -1e9)
                probs.append(torch.softmax(masked, dim=-1).cpu().numpy())
                values.append(value.detach().float().reshape(len(chunk), -1)[:, 0].cpu().numpy())
    finally:
        net.train(was_training)
    return np.concatenate(values), np.concatenate(probs)


def _step_counts(step_records: Sequence[StepRecord]) -> dict[str, Any]:
    histogram: dict[str, int] = {}
    for record in step_records:
        histogram[record.action_type] = histogram.get(record.action_type, 0) + 1
    failed = [r for r in step_records if not r.success]
    return {
        "action_type_histogram": dict(sorted(histogram.items())),
        "num_steps": len(step_records),
        "invalid_action_count": len(failed),
        "invalid_move_count": sum(1 for r in failed if r.action_type == "move"),
    }


def prequential_value_metrics(
    net_prev: nn.Module,
    fresh_examples: Sequence[_ExampleLike],
    step_records: Sequence[StepRecord],
    buffer_mean: float,
    *,
    batch_size: int = PREQUENTIAL_BATCH_SIZE,
) -> dict[str, Any]:
    """Score the network that played an iteration's games on those same games.

    Called after self-play and *before* the fresh examples enter the replay
    buffer, so every position is out of sample for ``net_prev``: this is a
    prequential (predict-then-update) measurement, not a training-set fit.

    Args:
        net_prev: The network that generated the games, as MCTS saw it -- a
            module whose ``forward(spatial, vector, mask)`` returns
            ``(policy_logits, value)`` with a scalar (current-player) value.
        fresh_examples: The iteration's new training examples.
        step_records: The steps actually played (see :class:`StepRecord`).
        buffer_mean: Mean value target in the replay buffer before insertion;
            ``0.0`` when the buffer is empty (iteration 0).
        batch_size: Positions per forward pass.

    Returns:
        A JSON-serialisable dict. ``r2_oos = 1 - SSE / SST`` where SST is taken
        around ``buffer_mean`` -- a constant the network could have known
        before seeing these games -- so ``r2_oos > 0`` means the value head
        beats "predict the historical mean". Sign accuracy is over positions
        with ``|z| = 1``, beside the best constant-sign predictor. Undefined
        quantities are ``None``. Recorded, never gated.
    """
    from alpha_zero.monitoring import compute_policy_entropy

    metrics: dict[str, Any] = {
        "n_positions": len(fresh_examples),
        "buffer_mean": float(buffer_mean),
        "value_mse": None,
        "baseline_mse": None,
        "r2_oos": None,
        "n_decisive": 0,
        "sign_accuracy": None,
        "sign_accuracy_baseline": None,
        "target_entropy": None,
        "policy_entropy": None,
        **_step_counts(step_records),
    }
    if not fresh_examples:
        return metrics

    predicted, policy_probs = _predict(net_prev, fresh_examples, batch_size)
    targets = np.array([float(e.value) for e in fresh_examples], dtype=np.float64)
    sse = float(np.sum((targets - predicted) ** 2))
    sst = float(np.sum((targets - buffer_mean) ** 2))
    metrics["value_mse"] = sse / len(targets)
    metrics["baseline_mse"] = sst / len(targets)
    metrics["r2_oos"] = 1.0 - sse / sst if sst > 0.0 else None

    decisive = np.isclose(np.abs(targets), 1.0)
    metrics["n_decisive"] = int(decisive.sum())
    if decisive.any():
        accuracy = TrainingDiagnostics().compute_value_accuracy(
            torch.from_numpy(predicted[decisive]), torch.from_numpy(targets[decisive])
        )
        metrics["sign_accuracy"] = accuracy["sign_accuracy"]
        positive = float(np.mean(targets[decisive] > 0))
        metrics["sign_accuracy_baseline"] = max(positive, 1.0 - positive)

    metrics["target_entropy"] = float(
        np.mean([compute_policy_entropy(e.policy, e.action_mask) for e in fresh_examples])
    )
    metrics["policy_entropy"] = float(
        np.mean(
            [
                compute_policy_entropy(p, e.action_mask)
                for p, e in zip(policy_probs, fresh_examples, strict=True)
            ]
        )
    )
    return metrics
