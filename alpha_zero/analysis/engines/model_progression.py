"""Model progression engine for training run analysis.

Tracks model version evolution, promotion chains, ELO trajectories,
and detects the training-from-scratch bug (where every iteration
trains a fresh model instead of fine-tuning the current best).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from alpha_zero.analysis.data_model import RunData


@dataclass
class ModelProgressionResult:
    """Result of analyzing model version progression."""

    versions: list[int]
    promotion_chain: list[dict[str, Any]]
    promotion_iterations: list[int]
    total_promotions: int
    promotion_rate: float
    elo_trajectory: list[dict[str, Any]]
    version_timeline: list[dict[str, Any]]
    training_from_scratch_detected: bool
    diagnosis: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return dataclasses.asdict(self)


class ModelProgressionAnalyzer:
    """Analyzes model version progression from a training run.

    Detects the training-from-scratch bug: if the majority of iterations
    show high loss (above *fresh_model_loss_threshold*) AND there are zero
    promotions, the training worker was likely creating fresh models each
    iteration instead of fine-tuning the current best.
    """

    def __init__(self, fresh_model_loss_threshold: float = 0.8) -> None:
        self._fresh_loss_threshold = fresh_model_loss_threshold

    def analyze(self, run_data: RunData) -> ModelProgressionResult:
        """Analyze model progression from *run_data*."""
        # Extract unique model versions (preserving order)
        seen: set[int] = set()
        versions: list[int] = []
        for it in run_data.iterations:
            if it.model_version is not None and it.model_version not in seen:
                seen.add(it.model_version)
                versions.append(it.model_version)

        # Build promotion chain
        promotion_chain: list[dict[str, Any]] = []
        promotion_iterations: list[int] = []
        for it in run_data.iterations:
            if it.promoted:
                promotion_chain.append(
                    {"iteration": it.iteration, "model_version": it.model_version}
                )
                promotion_iterations.append(it.iteration)

        total_promotions = len(promotion_chain)
        total_iters = run_data.total_iterations
        promotion_rate = total_promotions / total_iters if total_iters > 0 else 0.0

        # ELO trajectory from models
        elo_trajectory: list[dict[str, Any]] = []
        for m in run_data.models:
            if m.elo is not None:
                elo_trajectory.append({"version": m.version, "elo": m.elo})

        # Version timeline from models
        version_timeline: list[dict[str, Any]] = []
        for m in run_data.models:
            version_timeline.append(
                {"version": m.version, "timestamp": m.timestamp}
            )

        # Detect training-from-scratch bug
        training_from_scratch_detected = self._detect_training_from_scratch(
            run_data, total_promotions
        )

        # Generate diagnosis
        diagnosis = self._generate_diagnosis(
            run_data,
            total_promotions,
            promotion_rate,
            training_from_scratch_detected,
        )

        return ModelProgressionResult(
            versions=versions,
            promotion_chain=promotion_chain,
            promotion_iterations=promotion_iterations,
            total_promotions=total_promotions,
            promotion_rate=promotion_rate,
            elo_trajectory=elo_trajectory,
            version_timeline=version_timeline,
            training_from_scratch_detected=training_from_scratch_detected,
            diagnosis=diagnosis,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_training_from_scratch(
        self, run_data: RunData, total_promotions: int
    ) -> bool:
        """Return True if the training-from-scratch bug is likely present.

        Heuristic: if >80% of iterations have total_loss above the
        fresh-model threshold AND there are 0 promotions, the training
        worker was probably re-initializing weights each iteration.
        """
        if total_promotions > 0:
            return False

        iters_with_loss = [
            it for it in run_data.iterations if it.total_loss is not None
        ]
        if not iters_with_loss:
            return False

        high_loss_count = sum(
            bool(it.total_loss is not None and it.total_loss > self._fresh_loss_threshold)
            for it in iters_with_loss
        )
        fraction = high_loss_count / len(iters_with_loss)
        return fraction > 0.8

    def _generate_diagnosis(
        self,
        run_data: RunData,
        total_promotions: int,
        promotion_rate: float,
        training_from_scratch: bool,
    ) -> str:
        """Return a human-readable diagnosis string."""
        total = run_data.total_iterations
        if total == 0:
            return "No iterations to analyze."

        parts: list[str] = []
        parts.append(
            f"Run had {total} iterations with {total_promotions} promotion(s) "
            f"(rate: {promotion_rate:.1%})."
        )

        if training_from_scratch:
            parts.append(
                "WARNING: Training-from-scratch bug detected. "
                "Loss remains high across iterations with zero promotions, "
                "suggesting the training worker re-initialized weights each iteration."
            )
        elif total_promotions == 0:
            parts.append(
                "No promotions occurred. The challenger never beat the current best model."
            )
        else:
            parts.append(
                f"Promotions occurred at iterations: "
                f"{', '.join(str(it.iteration) for it in run_data.iterations if it.promoted)}."
            )

        return " ".join(parts)
