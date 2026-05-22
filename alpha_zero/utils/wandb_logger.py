"""Deprecated. Use ``alpha_zero.utils.experiment_tracker`` instead.

This module preserves the legacy ``WandbLogger``/``WandbConfig`` symbol
names but routes them to MLflow under the hood. New code should import
from ``alpha_zero.utils.experiment_tracker``.
"""

from __future__ import annotations

import warnings

from alpha_zero.monitoring.config import MonitoringConfig as WandbConfig  # noqa: F401
from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter as WandbLogger  # noqa: F401
from alpha_zero.utils.experiment_tracker import (  # noqa: F401
    get_tracker as get_wandb_logger,
)
from alpha_zero.utils.experiment_tracker import (  # noqa: F401
    init_tracker as init_wandb,
)
from alpha_zero.utils.experiment_tracker import (  # noqa: F401
    log_metrics,
)

#: Legacy compatibility constant — MLflow backend is always available
#: (graceful no-op if mlflow package missing). Kept so existing
#: ``from alpha_zero.utils.wandb_logger import WANDB_AVAILABLE`` lines
#: continue to import successfully until Task B4 migrates them.
WANDB_AVAILABLE: bool = True

warnings.warn(
    "alpha_zero.utils.wandb_logger is deprecated; metrics are now sent to "
    "MLflow via alpha_zero.utils.experiment_tracker. The wandb_logger module "
    "will be removed in v1.6.0.",
    DeprecationWarning,
    stacklevel=2,
)
