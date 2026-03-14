"""Unified data model for training run analysis.

Provides dataclass-based representations of training run data that all
analysis engines operate on. This is the foundation layer — all analysis
modules depend on these data structures.

Data flows from W&B and GCS into these normalized structures, which are
then consumed by analysis engines (loss curves, ELO, cost tracking, etc.).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunIdentifier:
    """Identifies a training run across W&B and GCS.

    A run may have data in W&B (metrics, logs), GCS (models, trajectories),
    or both. At least one source identifier should be populated.
    """

    wandb_run_id: str | None = None
    wandb_group: str | None = None
    wandb_project: str = "civ6-alphazero"
    wandb_entity: str | None = None
    gcs_bucket: str | None = None
    gcs_prefix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return dataclasses.asdict(self)


@dataclass
class IterationData:
    """Data for one training iteration.

    All metric fields are optional (None) since not every iteration
    has every metric available (e.g., eval metrics only exist when
    evaluation was run).
    """

    iteration: int

    # Loss metrics
    policy_loss: float | None = None
    value_loss: float | None = None
    total_loss: float | None = None
    learning_rate: float | None = None

    # Evaluation metrics
    eval_win_rate: float | None = None
    eval_draw_rate: float | None = None
    eval_avg_turns: float | None = None
    value_correlation: float | None = None

    # Game / data metrics
    games_completed: int | None = None
    trajectory_count: int | None = None
    buffer_size: int | None = None

    # Model info
    model_version: int | None = None
    promoted: bool = False

    # Timing and cost
    duration_minutes: float | None = None
    selfplay_cost: float | None = None
    training_cost: float | None = None
    eval_cost: float | None = None
    cumulative_cost: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return dataclasses.asdict(self)


@dataclass
class ModelVersionData:
    """Model version metadata from GCS.

    Tracks per-model information including optional ELO ratings
    from tournament play.
    """

    version: int
    iteration: int

    elo: float | None = None
    timestamp: str | None = None
    games_played: int = 0
    model_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return dataclasses.asdict(self)


@dataclass
class RunData:
    """Complete normalized data for a training run.

    Aggregates iteration-level metrics, model version info, run config,
    and summary statistics into a single structure that analysis engines
    can consume.
    """

    identifier: RunIdentifier
    iterations: list[IterationData] = field(default_factory=list)
    models: list[ModelVersionData] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    start_model_version: int | None = None
    end_model_version: int | None = None

    @property
    def total_iterations(self) -> int:
        """Number of iterations in this run."""
        return len(self.iterations)

    @property
    def promotions(self) -> int:
        """Number of iterations where a model was promoted."""
        return sum(1 for it in self.iterations if it.promoted)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary, including computed properties."""
        d = dataclasses.asdict(self)
        d["total_iterations"] = self.total_iterations
        d["promotions"] = self.promotions
        return d
