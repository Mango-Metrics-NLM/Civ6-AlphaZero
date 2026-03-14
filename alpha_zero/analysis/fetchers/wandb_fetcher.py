"""Read-only W&B API fetcher for completed training runs.

Queries W&B via the public API and returns normalized RunData objects.
Column mapping is configurable so the fetcher adapts to different
logging conventions without code changes.
"""
from __future__ import annotations

import logging
import math
from typing import Any

# Optional wandb import (same pattern as alpha_zero/utils/wandb_logger.py)
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]

from alpha_zero.analysis.data_model import (
    IterationData,
    RunData,
    RunIdentifier,
)

logger = logging.getLogger(__name__)

# Configurable column mapping -- maps RunData fields to W&B history column names.
# Values can be str (single column) or list[str] (try each in order).
DEFAULT_COLUMN_MAP: dict[str, str | list[str]] = {
    "policy_loss": "train/policy_loss",
    "value_loss": "train/value_loss",
    "total_loss": ["train/loss", "train/total_loss"],
    "eval_win_rate": "orchestrator/eval_win_rate",
    "eval_draw_rate": "orchestrator/eval_draw_rate",
    "eval_avg_turns": "orchestrator/eval_avg_turns",
    "value_correlation": "orchestrator/value_correlation",
    "cumulative_cost": "orchestrator/cumulative_cost",
    "model_version": "orchestrator/model_version",
    "iteration_duration": "orchestrator/iteration_duration_min",
    "learning_rate": "train/learning_rate",
    "games_completed": "orchestrator/games_completed",
}


class WandbRunFetcher:
    """Fetches data from completed W&B runs via the public API (read-only)."""

    def __init__(
        self,
        project: str = "civ6-alphazero",
        entity: str | None = None,
        column_map: dict[str, str | list[str]] | None = None,
    ) -> None:
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb is not installed. Install with: pip install wandb"
            )
        self._project = project
        self._entity = entity
        self._column_map = column_map or DEFAULT_COLUMN_MAP
        self._api = wandb.Api()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_run(self, run_id: str) -> RunData:
        """Fetch a single run by W&B run ID or run name."""
        path = (
            f"{self._entity}/{self._project}/{run_id}"
            if self._entity
            else f"{self._project}/{run_id}"
        )
        run = self._api.run(path)
        return self._run_to_data(run)

    def fetch_group(self, group_name: str) -> list[RunData]:
        """Fetch all runs in a W&B group."""
        path = (
            f"{self._entity}/{self._project}"
            if self._entity
            else self._project
        )
        runs = self._api.runs(path, filters={"group": group_name})
        return [self._run_to_data(r) for r in runs]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_to_data(self, run: Any) -> RunData:
        """Convert a wandb.Run to RunData."""
        history = run.history(pandas=True)
        iterations = self._extract_iterations(history)
        self._detect_promotions(iterations)

        identifier = RunIdentifier(
            wandb_run_id=run.id,
            wandb_group=run.config.get("group", run.group)
            if isinstance(run.config, dict)
            else getattr(run, "group", None),
            wandb_project=self._project,
            wandb_entity=self._entity,
        )

        versions = [
            it.model_version for it in iterations if it.model_version is not None
        ]

        return RunData(
            identifier=identifier,
            iterations=iterations,
            models=[],  # GCS fetcher fills this in
            config=dict(run.config) if isinstance(run.config, dict) else {},
            summary=dict(run.summary) if isinstance(run.summary, dict) else {},
            start_model_version=min(versions) if versions else None,
            end_model_version=max(versions) if versions else None,
        )

    def _extract_iterations(self, history: Any) -> list[IterationData]:
        """Extract IterationData from W&B history DataFrame."""
        if history is None or len(history) == 0:
            return []

        columns = set(history.columns)
        iterations: list[IterationData] = []
        for idx in range(len(history)):
            row = history.iloc[idx]
            iteration = IterationData(
                iteration=idx,
                policy_loss=self._get_value(row, columns, "policy_loss"),
                value_loss=self._get_value(row, columns, "value_loss"),
                total_loss=self._get_value(row, columns, "total_loss"),
                learning_rate=self._get_value(row, columns, "learning_rate"),
                eval_win_rate=self._get_value(row, columns, "eval_win_rate"),
                eval_draw_rate=self._get_value(row, columns, "eval_draw_rate"),
                eval_avg_turns=self._get_value(row, columns, "eval_avg_turns"),
                value_correlation=self._get_value(row, columns, "value_correlation"),
                cumulative_cost=self._get_value(row, columns, "cumulative_cost"),
                duration_minutes=self._get_value(row, columns, "iteration_duration"),
                games_completed=self._get_int_value(row, columns, "games_completed"),
                model_version=self._get_int_value(row, columns, "model_version"),
            )
            iterations.append(iteration)

        return iterations

    def _get_value(
        self, row: Any, columns: set[str], field: str
    ) -> float | None:
        """Get a float value from a row using the column map.

        Returns None if the column is missing or the value is NaN.
        """
        col_spec = self._column_map.get(field)
        if col_spec is None:
            return None

        candidates = col_spec if isinstance(col_spec, list) else [col_spec]
        for col in candidates:
            if col in columns:
                val = row[col]
                try:
                    fval = float(val)
                    if math.isnan(fval):
                        return None
                    return fval
                except (TypeError, ValueError):
                    return None
        return None

    def _get_int_value(
        self, row: Any, columns: set[str], field: str
    ) -> int | None:
        """Get an int value from a row using the column map."""
        val = self._get_value(row, columns, field)
        return int(val) if val is not None else None

    @staticmethod
    def _detect_promotions(iterations: list[IterationData]) -> None:
        """Detect promotions by model version increases between consecutive iterations."""
        for i in range(1, len(iterations)):
            curr = iterations[i].model_version
            prev = iterations[i - 1].model_version
            if curr is not None and prev is not None and curr > prev:
                iterations[i].promoted = True
