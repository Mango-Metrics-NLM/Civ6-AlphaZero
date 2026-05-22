"""Convergence validation script smoke (production-path).

Replaces the legacy v1.3.0 MicroCivNetwork-based tests with a smoke that
exercises the actual ``scripts/validate_convergence.py`` end-to-end, which
itself uses the production ``train_epoch`` + ``GCSCheckpointStorage`` round
trip.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_validate_convergence_script_smoke(tmp_path):
    """The convergence script runs to completion on a tiny config."""
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": f"file:{tmp_path}/mlruns",
        "MLFLOW_EXPERIMENT_NAME": "ci-validate",
        "MLFLOW_ENABLED": "true",
        "PYTHONUNBUFFERED": "1",
    }
    result = subprocess.run(
        [
            sys.executable, "-u", "scripts/validate_convergence.py",
            "--iterations=2", "--games=2", "--epochs=1", "--eval-games=2",
            "--max-turns=8", "--device=cpu",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=110,
        env=env,
    )
    assert result.returncode == 0, (
        f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    )
    assert (tmp_path / "mlruns").exists()
