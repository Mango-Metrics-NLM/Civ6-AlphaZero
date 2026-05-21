"""MLflow-based experiment tracking for Civ6-AlphaZero.

Public API::

    from alpha_zero.monitoring import (
        MLflowConfig,
        MLflowLogger,
        TournamentMlflowLogger,
        compute_value_correlation,
        compute_elo_spread,
        compute_policy_entropy,
        MLFLOW_AVAILABLE,
    )

The factory ``create_mlflow_logger`` lives in
``alpha_zero.utils.factories`` to match the project's existing factory
registry pattern.
"""

from __future__ import annotations

from alpha_zero.monitoring.config import MLflowConfig, parse_tag_string
from alpha_zero.monitoring.metrics import (
    compute_elo_spread,
    compute_policy_entropy,
    compute_value_correlation,
)
from alpha_zero.monitoring.mlflow_logger import (
    MLFLOW_AVAILABLE,
    MLflowLogger,
    get_default_tracking_uri,
)
from alpha_zero.monitoring.tournament_dashboards import TournamentMlflowLogger

__all__ = [
    "MLFLOW_AVAILABLE",
    "MLflowConfig",
    "MLflowLogger",
    "TournamentMlflowLogger",
    "compute_elo_spread",
    "compute_policy_entropy",
    "compute_value_correlation",
    "get_default_tracking_uri",
    "parse_tag_string",
]
