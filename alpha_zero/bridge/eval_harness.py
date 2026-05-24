"""Evaluation harness for the Civ6 Bridge, integrating with MLflow.

W2.3a of the v1.7.0 local-gameplay sprint. This module provides a harness
to track turns, action latencies, and neural network value estimates
during local gameplay via the MLflow logger.
"""

from __future__ import annotations

import logging
from typing import Any

from alpha_zero.monitoring.mlflow_logger import MLflowLogger

logger = logging.getLogger(__name__)

#: Default MLflow experiment name for bridge evaluation runs.
#: Override via ``BridgeEvalHarness(experiment_name="my-experiment")``.
_DEFAULT_EXPERIMENT_NAME: str = "civ6-bridge-eval"


class BridgeEvalHarness:
    """Hooks into the BridgeAgentLoop to log metrics to MLflow.

    Provides start, turn-end, and session-end hooks.  All MLflow operations
    fail silently when MLflow is unavailable or ``enabled=False``.

    Args:
        run_name: Optional MLflow run name.  Defaults to a timestamped name.
        experiment_name: MLflow experiment to log into.  Defaults to
            :data:`_DEFAULT_EXPERIMENT_NAME`.
        enabled: When ``False`` the harness is a no-op (useful in tests or
            when ``--eval-harness`` is off).
    """

    def __init__(
        self,
        run_name: str | None = None,
        experiment_name: str = _DEFAULT_EXPERIMENT_NAME,
        enabled: bool = True,
    ) -> None:
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.enabled = enabled
        self.logger = MLflowLogger(
            run_name=run_name,
            experiment_name=experiment_name,
            enabled=enabled,
        )

    def on_start(self, config_dict: dict[str, Any] | None = None) -> None:
        """Initialize the MLflow run."""
        if not self.enabled:
            return
        self.logger.init(config_dict=config_dict)
        logger.info("BridgeEvalHarness initialized MLflow run: %s", self.run_name)

    def on_turn_end(self, turn: int, diagnostics: dict[str, float]) -> None:
        """Log per-turn metrics like value estimates and policy type."""
        if not self.enabled:
            return
        metrics: dict[str, float] = {"turn": float(turn)}
        metrics.update(diagnostics)
        self.logger.log_metrics(metrics, step=turn)

    def on_stop(self, final_stats: dict[str, Any]) -> None:
        """Log aggregate session statistics and close the run."""
        if not self.enabled:
            return
        metrics: dict[str, float] = {}
        for key, value in final_stats.items():
            if isinstance(value, (int, float, bool)):
                metrics[f"session/{key}"] = float(value)

        if metrics:
            self.logger.log_metrics(metrics)
        self.logger.finish()
        logger.info("BridgeEvalHarness finished.")
