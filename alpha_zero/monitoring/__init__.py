"""MLflow-based experiment tracking for Civ6-AlphaZero.

This package combines two independent MLflow implementations (PR #78 +
PR #79) into a single coherent surface area. Both legacy class names are
exported so neither side's downstream code breaks:

- :class:`MLflowConfig` / :class:`MonitoringConfig` - same dataclass under two
  names for backwards compatibility.
- :class:`MLflowLogger` - unified logger supporting both the "config object"
  initialisation (PR #78) and the keyword-argument initialisation (PR #79).
- :class:`AlphaZeroMLflowEvaluator` - custom evaluation-metric helper
  (policy entropy, value MAE, top-1 accuracy, action diversity).
- :class:`TournamentMlflowLogger` - match / round / leaderboard facade.

Stateless metric helpers (``compute_value_correlation``, ``compute_elo_spread``,
``compute_policy_entropy``) live in :mod:`alpha_zero.monitoring.metrics`.

The factory ``create_mlflow_logger`` lives in
:mod:`alpha_zero.utils.factories` to match the project's existing factory
registry pattern.
"""

from __future__ import annotations

from alpha_zero.monitoring.config import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    ENV_PREFIX,
    MLflowConfig,
    MonitoringConfig,
    parse_tag_string,
)
from alpha_zero.monitoring.metrics import (
    compute_elo_spread,
    compute_policy_entropy,
    compute_value_correlation,
)
from alpha_zero.monitoring.mlflow_evaluator import AlphaZeroMLflowEvaluator
from alpha_zero.monitoring.mlflow_logger import (
    MLFLOW_AVAILABLE,
    MlflowClient,
    MLflowLogger,
    get_default_tracking_uri,
    mlflow,
)
from alpha_zero.monitoring.tournament_dashboards import TournamentMlflowLogger

__all__ = [
    "DEFAULT_EXPERIMENT_NAME",
    "DEFAULT_TRACKING_URI",
    "ENV_PREFIX",
    "MLFLOW_AVAILABLE",
    "AlphaZeroMLflowEvaluator",
    "MLflowConfig",
    "MLflowLogger",
    "MlflowClient",
    "MonitoringConfig",
    "TournamentMlflowLogger",
    "compute_elo_spread",
    "compute_policy_entropy",
    "compute_value_correlation",
    "get_default_tracking_uri",
    "mlflow",
    "parse_tag_string",
]
