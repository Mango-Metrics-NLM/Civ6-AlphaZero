"""Training run analysis toolkit for Civ6 AlphaZero.

Provides composable analysis engines for post-run diagnostics of
Vertex AI training runs, pulling data from W&B and GCS.

Usage:
    python -m alpha_zero.analysis --run-id <wandb_run_id>
    python -m alpha_zero.analysis --group <wandb_group> --output report.json
"""

from __future__ import annotations

from alpha_zero.analysis.data_model import (
    IterationData,
    ModelVersionData,
    RunData,
    RunIdentifier,
)
from alpha_zero.analysis.run_analyzer import main as analyze_run

__all__ = [
    "IterationData",
    "ModelVersionData",
    "RunData",
    "RunIdentifier",
    "analyze_run",
]
